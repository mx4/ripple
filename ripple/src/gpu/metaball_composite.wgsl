// Metaball pass 2: fullscreen. Sample the accumulated field; where it exceeds a
// threshold, draw liquid coloured by average speed (same ramp as the dots) with
// a soft anti-aliased rim. Elsewhere alpha = 0 so the cleared background shows.

@group(0) @binding(0) var field_tex: texture_2d<f32>;
@group(0) @binding(1) var field_samp: sampler;

// Surface threshold and rim softness, in accumulated-falloff units. Tune these
// for a tighter (higher threshold) or blobbier (lower) liquid.
const THRESHOLD: f32 = 0.8;
const EDGE: f32 = 0.4;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_full(@builtin(vertex_index) vi: u32) -> VsOut {
    // One big triangle covering the screen.
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
fn fs_threshold(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(field_tex, field_samp, in.uv);
    let field = s.r;
    let avg_speed = s.g / max(s.r, 1e-4);
    let t = clamp(avg_speed / 450.0, 0.0, 1.0);
    let col = vec3<f32>(0.25 + 0.7 * t, 0.55 + 0.4 * t, 1.0);
    let a = smoothstep(THRESHOLD - EDGE, THRESHOLD + EDGE, field);
    return vec4<f32>(col, a);
}
