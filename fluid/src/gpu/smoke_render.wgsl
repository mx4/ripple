// Fullscreen render of the GPU smoke density (read straight from the solver's
// storage buffer) into a translucent-looking colour. y is flipped so up is up.

struct RenderParams {
    n: u32,
    display_k: f32,
    _p0: u32,
    _p1: u32,
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> dens: array<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    let c = p[vi];
    var out: VsOut;
    out.clip = vec4<f32>(c, 0.0, 1.0);
    out.uv = vec2<f32>((c.x + 1.0) * 0.5, 1.0 - (c.y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let nf = f32(R.n);
    let i = min(u32(in.uv.x * nf), R.n - 1u) + 1u;
    let jrow = min(u32(in.uv.y * nf), R.n - 1u); // 0 = top
    let j = R.n - jrow; // n = top .. 1 = bottom
    let d = max(dens[i + (R.n + 2u) * j], 0.0);
    let b = 1.0 - exp(-d * R.display_k);
    return vec4<f32>(b * 0.9, b * 0.95, b, 1.0);
}
