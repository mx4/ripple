// Fullscreen blit of a pre-coloured field texture (e.g. the smoke density turned
// into RGBA on the CPU). One big triangle; sample and output.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
    return textureSample(tex, samp, in.uv);
}
