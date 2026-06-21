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

// Position/velocity are double-buffered: the read-only `pos`/`vel` are the
// current state; the fused forces+integrate pass writes the next state into
// `pos_out`/`vel_out`. The host swaps the two sets each step (ping-pong bind
// groups). This is what lets us fuse the force gather and the integrator into a
// single dispatch without a write-after-read race on the shared positions.
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> pos_out: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_out: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> rho: array<f32>;
@group(0) @binding(6) var<storage, read_write> grid_count: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> grid_cells: array<u32>;

// Pressure is a cheap closed form of density (Müller WCSPH, clamped >= 0 so the
// fluid never pulls). We recompute it inline wherever needed instead of storing
// a whole per-particle buffer — the `forces` pass already loads each neighbour's
// rho, so deriving its pressure is a couple of ALU ops vs. an extra random
// gather in the kernel that dominates the step.
fn pressure_of(density: f32) -> f32 {
    return max(P.stiffness * (density - P.rest_dens), 0.0);
}

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
}

// Fused force gather + integration. Computing the SPH force leaves it in a
// register and steps the particle immediately, so there is no separate
// `force` buffer and one fewer dispatch/barrier per step. Reads come from the
// current `pos`/`vel`; the new state is written to `pos_out`/`vel_out` (the
// host ping-pongs the two), which is why the in-place write-after-read race the
// split passes avoided can't happen here.
@compute @workgroup_size(64)
fn forces_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.num) { return; }
    let pi = pos[i];
    let vi = vel[i];
    let rho_i = max(rho[i], 1e-6);
    let press_i = pressure_of(rho[i]);
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
                    let rho_j_raw = rho[j];
                    let rho_j = max(rho_j_raw, 1e-6);
                    let press_j = pressure_of(rho_j_raw);
                    let hr = P.h - rlen;
                    // Pressure repels (-dir); viscosity diffuses velocity.
                    fpress -= dir * P.mass * (press_i + press_j) / (2.0 * rho_j)
                        * P.spiky_grad * hr * hr;
                    fvisc += P.visc * P.mass * (vel[j] - vi) / rho_j * P.visc_lap * hr;
                }
            }
        }
    }

    // --- integrate (was the separate `integrate` pass) ---
    let force = fpress + fvisc;
    let grav = vec2<f32>(P.gravity_x, P.gravity_y);
    var v = vi + P.dt * (force / rho_i + grav);
    v += vec2<f32>(P.impulse_x, P.impulse_y); // one-shot shake (already /substeps)
    let speed = length(v);
    if (speed > P.max_speed) {
        v *= P.max_speed / speed;
    }
    var p = pi + P.dt * v;

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
    vel_out[i] = v;
    pos_out[i] = p;
}
