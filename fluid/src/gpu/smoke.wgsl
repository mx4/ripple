// GPU Eulerian smoke (Jos Stam "Stable Fluids") in compute shaders. Collocated
// grid stored in storage buffers (mirrors the CPU `Smoke` solver). y increases
// upward; one-cell border (kept zero) gives an implicit open/Dirichlet boundary.
//
// The pressure projection's Jacobi relaxation (`jacobi`) is the reusable grid
// primitive — ping-ponged between two bind groups so no per-iteration copy.

struct Params {
    n: u32,
    dt: f32,
    buoyancy: f32,
    dissipation: f32,
    src_amount: f32,
    src_vy: f32,
    src_half: u32,
    src_y: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> nu: array<f32>;
@group(0) @binding(4) var<storage, read_write> nv: array<f32>;
@group(0) @binding(5) var<storage, read_write> dens: array<f32>;
@group(0) @binding(6) var<storage, read_write> ndens: array<f32>;
// For jacobi: read `p`, write `q`. The two ping-pong bind groups swap which
// physical buffer is bound here.
@group(0) @binding(7) var<storage, read_write> p: array<f32>;
@group(0) @binding(8) var<storage, read_write> q: array<f32>;
@group(0) @binding(9) var<storage, read_write> div: array<f32>;

fn ix(i: u32, j: u32) -> u32 {
    return i + (P.n + 2u) * j;
}

// True if this invocation maps to an interior cell; sets i,j via out-params.
fn interior(g: vec3<u32>) -> bool {
    return g.x < P.n && g.y < P.n;
}

@compute @workgroup_size(8, 8)
fn forces(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    v[k] += P.dt * P.buoyancy * dens[k]; // buoyancy: rises (+v)
    // Continuous bottom source band.
    let cx = (P.n + 2u) / 2u;
    if (j >= P.src_y && j < P.src_y + 3u) {
        let dx = i32(i) - i32(cx);
        if (abs(dx) <= i32(P.src_half)) {
            dens[k] += P.dt * P.src_amount;
            v[k] += P.dt * P.src_vy;
        }
    }
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    let h = 1.0 / f32(P.n);
    div[k] = -0.5 * h * (u[ix(i + 1u, j)] - u[ix(i - 1u, j)] + v[ix(i, j + 1u)] - v[ix(i, j - 1u)]);
    p[k] = 0.0;
    q[k] = 0.0;
}

// One Jacobi sweep of the pressure Poisson: q = (div + sum of p-neighbours)/4.
@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    q[k] = (div[k] + p[ix(i - 1u, j)] + p[ix(i + 1u, j)] + p[ix(i, j - 1u)] + p[ix(i, j + 1u)]) * 0.25;
}

@compute @workgroup_size(8, 8)
fn subtract_gradient(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    let h = 1.0 / f32(P.n);
    u[k] -= 0.5 * (p[ix(i + 1u, j)] - p[ix(i - 1u, j)]) / h;
    v[k] -= 0.5 * (p[ix(i, j + 1u)] - p[ix(i, j - 1u)]) / h;
}

fn backtrace(i: u32, j: u32) -> vec2<f32> {
    let dt0 = P.dt * f32(P.n);
    let k = ix(i, j);
    let nf = f32(P.n);
    let x = clamp(f32(i) - dt0 * u[k], 0.5, nf + 0.5);
    let y = clamp(f32(j) - dt0 * v[k], 0.5, nf + 0.5);
    return vec2<f32>(x, y);
}

// Bilinear weights + corner indices for a backtraced point (WGSL can't pass
// storage pointers to a helper, so each field is sampled inline below).
struct Lerp {
    k00: u32,
    k10: u32,
    k01: u32,
    k11: u32,
    s0: f32,
    s1: f32,
    t0: f32,
    t1: f32,
}

fn lerp_at(x: f32, y: f32) -> Lerp {
    let i0 = u32(x);
    let j0 = u32(y);
    let i1 = i0 + 1u;
    let j1 = j0 + 1u;
    var l: Lerp;
    l.k00 = ix(i0, j0);
    l.k10 = ix(i1, j0);
    l.k01 = ix(i0, j1);
    l.k11 = ix(i1, j1);
    l.s1 = x - f32(i0);
    l.s0 = 1.0 - l.s1;
    l.t1 = y - f32(j0);
    l.t0 = 1.0 - l.t1;
    return l;
}

@compute @workgroup_size(8, 8)
fn advect_vel(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    let b = backtrace(i, j);
    let l = lerp_at(b.x, b.y);
    nu[k] = l.s0 * (l.t0 * u[l.k00] + l.t1 * u[l.k01]) + l.s1 * (l.t0 * u[l.k10] + l.t1 * u[l.k11]);
    nv[k] = l.s0 * (l.t0 * v[l.k00] + l.t1 * v[l.k01]) + l.s1 * (l.t0 * v[l.k10] + l.t1 * v[l.k11]);
}

@compute @workgroup_size(8, 8)
fn advect_dens(@builtin(global_invocation_id) g: vec3<u32>) {
    if (!interior(g)) { return; }
    let i = g.x + 1u;
    let j = g.y + 1u;
    let k = ix(i, j);
    let b = backtrace(i, j);
    let l = lerp_at(b.x, b.y);
    let d = l.s0 * (l.t0 * dens[l.k00] + l.t1 * dens[l.k01])
        + l.s1 * (l.t0 * dens[l.k10] + l.t1 * dens[l.k11]);
    ndens[k] = d / (1.0 + P.dt * P.dissipation);
}
