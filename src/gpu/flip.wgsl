// GPU FLIP/PIC water. Particles carry the fluid; each step their velocities are
// splatted onto a staggered MAC grid (P2G), the grid is made divergence-free by
// a Jacobi pressure projection with a free surface (only fluid cells solved, air
// p=0, no flow through solid walls), then read back to particles (G2P, PIC/FLIP
// blend). Mirrors src/flip.rs.
//
// P2G accumulation uses FIXED-POINT integer atomics (WGSL has no atomic<f32>):
// values are scaled by SCALE and atomicAdd'd as i32; the scale cancels when the
// weighted velocity is divided by the weight in `normalize`.

const SCALE: f32 = 65536.0;

struct Params {
    num: u32,
    nx: u32,
    ny: u32,
    pad0: u32,
    h: f32,
    gx: f32,
    gy: f32,
    dt: f32,
    flip: f32,
    impulse_x: f32,
    impulse_y: f32,
    pad3: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> u: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> u_prev: array<f32>;
@group(0) @binding(6) var<storage, read_write> v_prev: array<f32>;
@group(0) @binding(7) var<storage, read_write> au: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> av: array<atomic<i32>>;
@group(0) @binding(9) var<storage, read_write> wu: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> wv: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> s: array<f32>; // 0 solid, 1 else
@group(0) @binding(12) var<storage, read_write> fluid: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> div: array<f32>;
@group(0) @binding(14) var<storage, read_write> p: array<f32>;
@group(0) @binding(15) var<storage, read_write> p2: array<f32>;

fn ui(i: u32, j: u32) -> u32 { return i + (P.nx + 1u) * j; }
fn vi(i: u32, j: u32) -> u32 { return i + P.nx * (j); }
fn ci(i: u32, j: u32) -> u32 { return i + P.nx * j; }

// Bilinear corner indices + weights for a (cols x rows) grid at (x, y).
struct L {
    k00: u32, k10: u32, k01: u32, k11: u32,
    w00: f32, w10: f32, w01: f32, w11: f32,
}

fn lerp(cols: u32, rows: u32, x: f32, y: f32) -> L {
    let xx = clamp(x, 0.0, f32(cols - 1u) - 1e-4);
    let yy = clamp(y, 0.0, f32(rows - 1u) - 1e-4);
    let i0 = u32(xx);
    let j0 = u32(yy);
    let i1 = i0 + 1u;
    let j1 = j0 + 1u;
    let sx = xx - f32(i0);
    let sy = yy - f32(j0);
    var l: L;
    l.k00 = i0 + cols * j0;
    l.k10 = i1 + cols * j0;
    l.k01 = i0 + cols * j1;
    l.k11 = i1 + cols * j1;
    l.w00 = (1.0 - sx) * (1.0 - sy);
    l.w10 = sx * (1.0 - sy);
    l.w01 = (1.0 - sx) * sy;
    l.w11 = sx * sy;
    return l;
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= P.num) { return; }
    var pp = pos[idx];
    var vv = vel[idx];
    vv.x += P.gx * P.dt;
    vv.y += P.gy * P.dt;
    vv.x += P.impulse_x; // one-shot shake (pre-divided by substeps)
    vv.y += P.impulse_y;
    pp += vv * P.dt;
    let eps = 1e-3 * P.h;
    let xmin = P.h + eps;
    let xmax = (f32(P.nx) - 1.0) * P.h - eps;
    let ymin = P.h + eps;
    let ymax = (f32(P.ny) - 1.0) * P.h - eps;
    if (pp.x < xmin) { pp.x = xmin; vv.x = 0.0; } else if (pp.x > xmax) { pp.x = xmax; vv.x = 0.0; }
    if (pp.y < ymin) { pp.y = ymin; vv.y = 0.0; } else if (pp.y > ymax) { pp.y = ymax; vv.y = 0.0; }
    pos[idx] = pp;
    vel[idx] = vv;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    let usz = (P.nx + 1u) * P.ny;
    let vsz = P.nx * (P.ny + 1u);
    let csz = P.nx * P.ny;
    if (idx < usz) { atomicStore(&au[idx], 0); atomicStore(&wu[idx], 0); }
    if (idx < vsz) { atomicStore(&av[idx], 0); atomicStore(&wv[idx], 0); }
    if (idx < csz) { atomicStore(&fluid[idx], 0u); }
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= P.num) { return; }
    let pp = pos[idx];
    let vv = vel[idx];
    let gx = pp.x / P.h;
    let gy = pp.y / P.h;

    let cx = u32(clamp(floor(gx), 0.0, f32(P.nx) - 1.0));
    let cy = u32(clamp(floor(gy), 0.0, f32(P.ny) - 1.0));
    atomicStore(&fluid[ci(cx, cy)], 1u);

    // x-velocity to u-faces (cols nx+1, rows ny) at (gx, gy-0.5)
    let lu = lerp(P.nx + 1u, P.ny, gx, gy - 0.5);
    atomicAdd(&au[lu.k00], i32(lu.w00 * vv.x * SCALE)); atomicAdd(&wu[lu.k00], i32(lu.w00 * SCALE));
    atomicAdd(&au[lu.k10], i32(lu.w10 * vv.x * SCALE)); atomicAdd(&wu[lu.k10], i32(lu.w10 * SCALE));
    atomicAdd(&au[lu.k01], i32(lu.w01 * vv.x * SCALE)); atomicAdd(&wu[lu.k01], i32(lu.w01 * SCALE));
    atomicAdd(&au[lu.k11], i32(lu.w11 * vv.x * SCALE)); atomicAdd(&wu[lu.k11], i32(lu.w11 * SCALE));

    // y-velocity to v-faces (cols nx, rows ny+1) at (gx-0.5, gy)
    let lv = lerp(P.nx, P.ny + 1u, gx - 0.5, gy);
    atomicAdd(&av[lv.k00], i32(lv.w00 * vv.y * SCALE)); atomicAdd(&wv[lv.k00], i32(lv.w00 * SCALE));
    atomicAdd(&av[lv.k10], i32(lv.w10 * vv.y * SCALE)); atomicAdd(&wv[lv.k10], i32(lv.w10 * SCALE));
    atomicAdd(&av[lv.k01], i32(lv.w01 * vv.y * SCALE)); atomicAdd(&wv[lv.k01], i32(lv.w01 * SCALE));
    atomicAdd(&av[lv.k11], i32(lv.w11 * vv.y * SCALE)); atomicAdd(&wv[lv.k11], i32(lv.w11 * SCALE));
}

@compute @workgroup_size(64)
fn normalize(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    let usz = (P.nx + 1u) * P.ny;
    let vsz = P.nx * (P.ny + 1u);
    if (idx < usz) {
        let w = atomicLoad(&wu[idx]);
        let val = select(0.0, f32(atomicLoad(&au[idx])) / f32(w), w != 0);
        u[idx] = val;
        u_prev[idx] = val;
    }
    if (idx < vsz) {
        let w = atomicLoad(&wv[idx]);
        let val = select(0.0, f32(atomicLoad(&av[idx])) / f32(w), w != 0);
        v[idx] = val;
        v_prev[idx] = val;
    }
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.nx || g.y >= P.ny) { return; }
    let i = g.x;
    let j = g.y;
    let c = ci(i, j);
    p[c] = 0.0;
    p2[c] = 0.0;
    if (i == 0u || i == P.nx - 1u || j == 0u || j == P.ny - 1u) { div[c] = 0.0; return; }
    div[c] = u[ui(i + 1u, j)] - u[ui(i, j)] + v[vi(i, j + 1u)] - v[vi(i, j)];
}

// One Jacobi sweep (the reusable grid primitive, free-surface variant).
@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= P.nx || g.y >= P.ny) { return; }
    let i = g.x;
    let j = g.y;
    let c = ci(i, j);
    if (i == 0u || i == P.nx - 1u || j == 0u || j == P.ny - 1u || atomicLoad(&fluid[c]) == 0u) {
        p2[c] = 0.0;
        return;
    }
    let sl = s[ci(i - 1u, j)];
    let sr = s[ci(i + 1u, j)];
    let sd = s[ci(i, j - 1u)];
    let su = s[ci(i, j + 1u)];
    let cnt = sl + sr + sd + su;
    if (cnt > 0.0) {
        p2[c] = (sl * p[ci(i - 1u, j)] + sr * p[ci(i + 1u, j)]
            + sd * p[ci(i, j - 1u)] + su * p[ci(i, j + 1u)] - div[c]) / cnt;
    } else {
        p2[c] = 0.0;
    }
}

@compute @workgroup_size(64)
fn subtract_gradient(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    let usz = (P.nx + 1u) * P.ny;
    let vsz = P.nx * (P.ny + 1u);
    if (idx < usz) {
        let i = idx % (P.nx + 1u);
        let j = idx / (P.nx + 1u);
        if (i == 0u || i == P.nx) {
            u[idx] = 0.0;
        } else {
            let l = ci(i - 1u, j);
            let r = ci(i, j);
            if (s[l] == 0.0 || s[r] == 0.0) {
                u[idx] = 0.0;
            } else {
                u[idx] -= p[r] - p[l];
            }
        }
    }
    if (idx < vsz) {
        let i = idx % P.nx;
        let j = idx / P.nx;
        if (j == 0u || j == P.ny) {
            v[idx] = 0.0;
        } else {
            let d = ci(i, j - 1u);
            let uu = ci(i, j);
            if (s[d] == 0.0 || s[uu] == 0.0) {
                v[idx] = 0.0;
            } else {
                v[idx] -= p[uu] - p[d];
            }
        }
    }
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    if (idx >= P.num) { return; }
    let pp = pos[idx];
    let gx = pp.x / P.h;
    let gy = pp.y / P.h;

    let lu = lerp(P.nx + 1u, P.ny, gx, gy - 0.5);
    let un = lu.w00 * u[lu.k00] + lu.w10 * u[lu.k10] + lu.w01 * u[lu.k01] + lu.w11 * u[lu.k11];
    let uo = lu.w00 * u_prev[lu.k00] + lu.w10 * u_prev[lu.k10] + lu.w01 * u_prev[lu.k01] + lu.w11 * u_prev[lu.k11];

    let lv = lerp(P.nx, P.ny + 1u, gx - 0.5, gy);
    let vn = lv.w00 * v[lv.k00] + lv.w10 * v[lv.k10] + lv.w01 * v[lv.k01] + lv.w11 * v[lv.k11];
    let vo = lv.w00 * v_prev[lv.k00] + lv.w10 * v_prev[lv.k10] + lv.w01 * v_prev[lv.k01] + lv.w11 * v_prev[lv.k11];

    let old = vel[idx];
    let pic = vec2<f32>(un, vn);
    let flip = old + vec2<f32>(un - uo, vn - vo);
    vel[idx] = P.flip * flip + (1.0 - P.flip) * pic;
}
