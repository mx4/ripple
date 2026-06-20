// Marching squares (filled): same density field and 16-case classification as
// the contour, but emit triangles covering the inside (>= threshold) region of
// each cell instead of boundary line segments. Gives a solid liquid with crisp
// straight, sub-cell-interpolated edges — distinct from the metaball's soft
// per-pixel fill. Up to 3 triangles per cell, generated in the vertex shader
// from @builtin(vertex_index).

@group(0) @binding(0) var field_tex: texture_2d<f32>;

const STEP: u32 = 8u;        // grid cell size in texels (must match render.rs MC_STEP)
const THRESHOLD: f32 = 0.8;  // same surface level as the metaball composite
const SKIP: u32 = 255u;

// Per case (0..15): up to 3 triangles = 9 vertex specs. Spec: 0-3 = corners
// c0..c3 (TL, TR, BR, BL), 4-7 = edge points e0..e3 (top,right,bottom,left),
// 255 = skip (degenerate triangle). Saddle cases (5, 10) fill the two opposite
// corners separately.
var<private> TRIS: array<array<u32, 9>, 16> = array<array<u32, 9>, 16>(
    array<u32, 9>(255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u, 255u), // 0
    array<u32, 9>(0u, 4u, 7u, 255u, 255u, 255u, 255u, 255u, 255u),       // 1
    array<u32, 9>(4u, 1u, 5u, 255u, 255u, 255u, 255u, 255u, 255u),       // 2
    array<u32, 9>(0u, 1u, 5u, 0u, 5u, 7u, 255u, 255u, 255u),             // 3
    array<u32, 9>(5u, 2u, 6u, 255u, 255u, 255u, 255u, 255u, 255u),       // 4
    array<u32, 9>(0u, 4u, 7u, 2u, 5u, 6u, 255u, 255u, 255u),             // 5
    array<u32, 9>(4u, 1u, 2u, 4u, 2u, 6u, 255u, 255u, 255u),             // 6
    array<u32, 9>(0u, 1u, 2u, 0u, 2u, 6u, 0u, 6u, 7u),                   // 7
    array<u32, 9>(6u, 3u, 7u, 255u, 255u, 255u, 255u, 255u, 255u),       // 8
    array<u32, 9>(0u, 4u, 6u, 0u, 6u, 3u, 255u, 255u, 255u),             // 9
    array<u32, 9>(1u, 4u, 5u, 3u, 6u, 7u, 255u, 255u, 255u),             // 10
    array<u32, 9>(0u, 1u, 5u, 0u, 5u, 6u, 0u, 6u, 3u),                   // 11
    array<u32, 9>(5u, 2u, 3u, 5u, 3u, 7u, 255u, 255u, 255u),             // 12
    array<u32, 9>(0u, 4u, 5u, 0u, 5u, 2u, 0u, 2u, 3u),                   // 13
    array<u32, 9>(4u, 1u, 2u, 4u, 2u, 3u, 4u, 3u, 7u),                   // 14
    array<u32, 9>(0u, 1u, 2u, 0u, 2u, 3u, 255u, 255u, 255u),             // 15
);

fn cross_t(a: f32, b: f32) -> f32 {
    let d = b - a;
    if (abs(d) < 1e-6) { return 0.5; }
    return clamp((THRESHOLD - a) / d, 0.0, 1.0);
}

// Returns (density, density*speed) at a texel.
fn fld(x: u32, y: u32) -> vec2<f32> {
    let d = textureDimensions(field_tex);
    let xx = min(x, d.x - 1u);
    let yy = min(y, d.y - 1u);
    return textureLoad(field_tex, vec2<i32>(i32(xx), i32(yy)), 0).rg;
}

fn edge_point(
    edge: u32, x0: u32, y0: u32, x1: u32, y1: u32,
    f00: f32, f10: f32, f11: f32, f01: f32,
) -> vec2<f32> {
    let xf0 = f32(x0);
    let yf0 = f32(y0);
    let xf1 = f32(x1);
    let yf1 = f32(y1);
    if (edge == 0u) { return vec2<f32>(mix(xf0, xf1, cross_t(f00, f10)), yf0); }
    if (edge == 1u) { return vec2<f32>(xf1, mix(yf0, yf1, cross_t(f10, f11))); }
    if (edge == 2u) { return vec2<f32>(mix(xf1, xf0, cross_t(f11, f01)), yf1); }
    return vec2<f32>(xf0, mix(yf1, yf0, cross_t(f01, f00)));
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_fill(@builtin(vertex_index) vid: u32) -> VsOut {
    let dims = textureDimensions(field_tex);
    let gc = dims.x / STEP;
    let cell = vid / 9u;
    let k = vid % 9u;
    let cx = cell % gc;
    let cy = cell / gc;
    let x0 = cx * STEP;
    let x1 = x0 + STEP;
    let y0 = cy * STEP;
    let y1 = y0 + STEP;

    let g00 = fld(x0, y0);
    let g10 = fld(x1, y0);
    let g11 = fld(x1, y1);
    let g01 = fld(x0, y1);
    let f00 = g00.x;
    let f10 = g10.x;
    let f11 = g11.x;
    let f01 = g01.x;

    var code = 0u;
    if (f00 >= THRESHOLD) { code |= 1u; }
    if (f10 >= THRESHOLD) { code |= 2u; }
    if (f11 >= THRESHOLD) { code |= 4u; }
    if (f01 >= THRESHOLD) { code |= 8u; }

    let spec = TRIS[code][k];
    var out: VsOut;
    if (spec == SKIP) {
        out.clip = vec4<f32>(100.0, 100.0, 100.0, 1.0); // off-screen -> clipped
        out.color = vec3<f32>(0.0);
        return out;
    }

    var p: vec2<f32>;
    if (spec == 0u) { p = vec2<f32>(f32(x0), f32(y0)); }
    else if (spec == 1u) { p = vec2<f32>(f32(x1), f32(y0)); }
    else if (spec == 2u) { p = vec2<f32>(f32(x1), f32(y1)); }
    else if (spec == 3u) { p = vec2<f32>(f32(x0), f32(y1)); }
    else { p = edge_point(spec - 4u, x0, y0, x1, y1, f00, f10, f11, f01); }

    // Flat per-cell colour from density-weighted average speed (faceted look).
    let sum_r = f00 + f10 + f11 + f01;
    let sum_g = g00.y + g10.y + g11.y + g01.y;
    let avg = sum_g / max(sum_r, 1e-4);
    let t = clamp(avg / 450.0, 0.0, 1.0);

    let ndc = vec2<f32>(p.x / f32(dims.x) * 2.0 - 1.0, 1.0 - p.y / f32(dims.y) * 2.0);
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.color = vec3<f32>(0.25 + 0.7 * t, 0.55 + 0.4 * t, 1.0);
    return out;
}

@fragment
fn fs_fill(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
