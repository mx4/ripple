// Metaball pass 1: accumulate each particle as a soft radial falloff into a
// float field texture with additive blending.
//   R = sum(falloff)            (the metaball/density field)
//   G = sum(falloff * speed)    (so the composite can recover average speed)
// Reuses the same bind group as the dot renderer (uniform + pos + vel).

struct RenderParams {
    domain_w: f32,
    domain_h: f32,
    radius: f32,
    max_speed: f32,
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;

// Influence radius relative to the draw radius. Larger = more overlap = blobbier.
const BLOB_SCALE: f32 = 2.6;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) speed: f32,
}

@vertex
fn vs_blob(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let rad = R.radius * BLOB_SCALE;
    let center = pos[ii];
    let px = center.x + corner.x * rad;
    let py = center.y + corner.y * rad;
    let ndc = vec2<f32>(px / R.domain_w * 2.0 - 1.0, 1.0 - py / R.domain_h * 2.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = corner;
    out.speed = length(vel[ii]);
    return out;
}

@fragment
fn fs_blob(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.local, in.local);
    let f = max(0.0, 1.0 - r2);
    let falloff = f * f; // smooth bump: 1 at centre -> 0 at edge
    return vec4<f32>(falloff, falloff * in.speed, 0.0, falloff);
}
