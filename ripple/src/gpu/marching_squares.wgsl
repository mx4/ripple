// Marching squares: extract the liquid's contour from the metaball density
// field as line segments. For each grid cell, classify the 4 corners against a
// threshold (16 cases) and emit up to 2 edge-to-edge segments, with linear
// interpolation along each crossed edge. Pure geometry (LineList) — a crisp
// piecewise-linear boundary, distinct from the metaball's soft per-pixel fill.
//
// The whole contour is generated in the vertex shader from @builtin(vertex_index):
// 4 vertices per cell (2 possible segments x 2 endpoints); unused ones are sent
// off-screen so they clip away.

@group(0) @binding(0) var field_tex: texture_2d<f32>;

const STEP: u32 = 8u;        // grid cell size in texels (must match Rust MC_STEP)
const THRESHOLD: f32 = 0.8;  // same surface level as the metaball composite

// Per case (0..15): up to two segments, each two edge ids; 4u = unused.
// Edges: 0 = top (c0-c1), 1 = right (c1-c2), 2 = bottom (c2-c3), 3 = left (c3-c0).
// Corners: c0 top-left, c1 top-right, c2 bottom-right, c3 bottom-left.
var<private> TABLE: array<array<u32, 4>, 16> = array<array<u32, 4>, 16>(
    array<u32, 4>(4u, 4u, 4u, 4u), // 0  ----
    array<u32, 4>(0u, 3u, 4u, 4u), // 1  c0
    array<u32, 4>(0u, 1u, 4u, 4u), // 2  c1
    array<u32, 4>(1u, 3u, 4u, 4u), // 3  c0 c1
    array<u32, 4>(1u, 2u, 4u, 4u), // 4  c2
    array<u32, 4>(0u, 3u, 1u, 2u), // 5  c0 c2 (saddle)
    array<u32, 4>(0u, 2u, 4u, 4u), // 6  c1 c2
    array<u32, 4>(2u, 3u, 4u, 4u), // 7  c0 c1 c2
    array<u32, 4>(2u, 3u, 4u, 4u), // 8  c3
    array<u32, 4>(0u, 2u, 4u, 4u), // 9  c0 c3
    array<u32, 4>(0u, 1u, 2u, 3u), // 10 c1 c3 (saddle)
    array<u32, 4>(1u, 2u, 4u, 4u), // 11 c0 c1 c3
    array<u32, 4>(1u, 3u, 4u, 4u), // 12 c2 c3
    array<u32, 4>(0u, 1u, 4u, 4u), // 13 c0 c2 c3
    array<u32, 4>(0u, 3u, 4u, 4u), // 14 c1 c2 c3
    array<u32, 4>(4u, 4u, 4u, 4u), // 15 all
);

fn cross_t(a: f32, b: f32) -> f32 {
    let d = b - a;
    if (abs(d) < 1e-6) { return 0.5; }
    return clamp((THRESHOLD - a) / d, 0.0, 1.0);
}

fn dens(x: u32, y: u32) -> f32 {
    let d = textureDimensions(field_tex);
    let xx = min(x, d.x - 1u);
    let yy = min(y, d.y - 1u);
    return textureLoad(field_tex, vec2<i32>(i32(xx), i32(yy)), 0).r;
}

// Interpolated crossing point (texel space) on `edge` of the cell.
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
}

@vertex
fn vs_contour(@builtin(vertex_index) vid: u32) -> VsOut {
    let dims = textureDimensions(field_tex);
    let gc = dims.x / STEP;
    let cell = vid / 4u;
    let seg = (vid % 4u) / 2u; // 0 or 1
    let ep = vid % 2u;         // 0 or 1
    let cx = cell % gc;
    let cy = cell / gc;
    let x0 = cx * STEP;
    let x1 = x0 + STEP;
    let y0 = cy * STEP;
    let y1 = y0 + STEP;

    let f00 = dens(x0, y0);
    let f10 = dens(x1, y0);
    let f11 = dens(x1, y1);
    let f01 = dens(x0, y1);

    var code = 0u;
    if (f00 >= THRESHOLD) { code |= 1u; }
    if (f10 >= THRESHOLD) { code |= 2u; }
    if (f11 >= THRESHOLD) { code |= 4u; }
    if (f01 >= THRESHOLD) { code |= 8u; }

    let edge = TABLE[code][seg * 2u + ep];

    var out: VsOut;
    if (edge == 4u) {
        out.clip = vec4<f32>(100.0, 100.0, 100.0, 1.0); // off-screen -> clipped
        return out;
    }
    let p = edge_point(edge, x0, y0, x1, y1, f00, f10, f11, f01);
    let ndc = vec2<f32>(p.x / f32(dims.x) * 2.0 - 1.0, 1.0 - p.y / f32(dims.y) * 2.0);
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    return out;
}

@fragment
fn fs_contour() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.85, 1.0, 1.0); // liquid contour colour
}
