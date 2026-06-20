// GPU SPH compute kernels — one entry point per solver pass. Mirrors the CPU
// solver in src/sim.rs. Neighbour search uses a fixed-capacity bucket grid:
// each cell has `grid_cap` slots; particles atomically claim a slot. Overflow
// (more than grid_cap in a cell) is dropped, which never happens at our
// densities.
//
// We benchmarked a cell-sorted (CSR) grid for coalesced neighbour reads, but at
// our particle counts the prefix-sum scan it needs plus the extra passes cost
// more than the coalescing saved — the simple bucket grid is faster here.

struct Params {
    num: u32,
    cols: u32,
    rows: u32,
    grid_cap: u32,

    h: f32,
    mass: f32,
    poly6: f32,
    spiky_grad: f32,

    visc_lap: f32,
    visc: f32,
    rest_dens: f32,
    stiffness: f32,

    gravity_x: f32,
    gravity_y: f32,
    dt: f32,
    max_speed: f32,

    bound_w: f32,
    bound_h: f32,
    bound_shape: u32, // 0 = rect, 1 = circle
    bound_damping: f32,

    particle_radius: f32,
    impulse_x: f32, // one-shot velocity delta per substep (shake)
    impulse_y: f32,
    _pad0: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> force: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> rho: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(6) var<storage, read_write> grid_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> grid_cells: array<u32>;

fn cell_xy(p: vec2<f32>) -> vec2<i32> {
    let cx = i32(clamp(floor(p.x / P.h), 0.0, f32(P.cols) - 1.0));
    let cy = i32(clamp(floor(p.y / P.h), 0.0, f32(P.rows) - 1.0));
    return vec2<i32>(cx, cy);
}

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.cols * P.rows) { return; }
    atomicStore(&grid_count[i], 0u);
}

@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.num) { return; }
    let c = cell_xy(pos[i]);
    let cell = u32(c.y) * P.cols + u32(c.x);
    let slot = atomicAdd(&grid_count[cell], 1u);
    if (slot < P.grid_cap) {
        grid_cells[cell * P.grid_cap + slot] = i;
    }
}

@compute @workgroup_size(64)
fn density_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.num) { return; }
    let pi = pos[i];
    let h2 = P.h * P.h;
    let c = cell_xy(pi);
    var r = 0.0;
    for (var gy = max(c.y - 1, 0); gy <= min(c.y + 1, i32(P.rows) - 1); gy++) {
        for (var gx = max(c.x - 1, 0); gx <= min(c.x + 1, i32(P.cols) - 1); gx++) {
            let cell = u32(gy) * P.cols + u32(gx);
            let count = min(atomicLoad(&grid_count[cell]), P.grid_cap);
            for (var s = 0u; s < count; s++) {
                let j = grid_cells[cell * P.grid_cap + s];
                let d = pos[j] - pi;
                let r2 = dot(d, d);
                if (r2 < h2) {
                    let t = h2 - r2;
                    r += P.mass * P.poly6 * t * t * t;
                }
            }
        }
    }
    rho[i] = r;
    // Clamp pressure >= 0 (no tensile/attractive forces).
    pressure[i] = max(P.stiffness * (r - P.rest_dens), 0.0);
}

@compute @workgroup_size(64)
fn forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.num) { return; }
    let pi = pos[i];
    let vi = vel[i];
    let c = cell_xy(pi);
    var fpress = vec2<f32>(0.0, 0.0);
    var fvisc = vec2<f32>(0.0, 0.0);
    for (var gy = max(c.y - 1, 0); gy <= min(c.y + 1, i32(P.rows) - 1); gy++) {
        for (var gx = max(c.x - 1, 0); gx <= min(c.x + 1, i32(P.cols) - 1); gx++) {
            let cell = u32(gy) * P.cols + u32(gx);
            let count = min(atomicLoad(&grid_count[cell]), P.grid_cap);
            for (var s = 0u; s < count; s++) {
                let j = grid_cells[cell * P.grid_cap + s];
                if (j == i) { continue; }
                let rij = pos[j] - pi;
                let rlen = length(rij);
                if (rlen < P.h && rlen > 0.0) {
                    let dir = rij / rlen;
                    let rho_j = max(rho[j], 1e-6);
                    let hr = P.h - rlen;
                    // Pressure repels (-dir); viscosity diffuses velocity.
                    fpress -= dir * P.mass * (pressure[i] + pressure[j]) / (2.0 * rho_j)
                        * P.spiky_grad * hr * hr;
                    fvisc += P.visc * P.mass * (vel[j] - vi) / rho_j * P.visc_lap * hr;
                }
            }
        }
    }
    force[i] = fpress + fvisc;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.num) { return; }
    let r = max(rho[i], 1e-6);
    let grav = vec2<f32>(P.gravity_x, P.gravity_y);
    var v = vel[i] + P.dt * (force[i] / r + grav);
    v += vec2<f32>(P.impulse_x, P.impulse_y); // one-shot shake (already /substeps)
    let speed = length(v);
    if (speed > P.max_speed) {
        v *= P.max_speed / speed;
    }
    var p = pos[i] + P.dt * v;

    let pad = P.particle_radius;
    if (P.bound_shape == 0u) {
        if (p.x < pad) { p.x = pad; v.x *= -P.bound_damping; }
        else if (p.x > P.bound_w - pad) { p.x = P.bound_w - pad; v.x *= -P.bound_damping; }
        if (p.y < pad) { p.y = pad; v.y *= -P.bound_damping; }
        else if (p.y > P.bound_h - pad) { p.y = P.bound_h - pad; v.y *= -P.bound_damping; }
    } else {
        let center = vec2<f32>(P.bound_w * 0.5, P.bound_h * 0.5);
        let radius = min(P.bound_w, P.bound_h) * 0.5 - pad;
        let dd = p - center;
        let dist = length(dd);
        if (dist > radius && dist > 0.0) {
            let n = dd / dist;
            p = center + n * radius;
            let vn = dot(v, n);
            if (vn > 0.0) { v -= (1.0 + P.bound_damping) * vn * n; }
        }
    }
    vel[i] = v;
    pos[i] = p;
}
