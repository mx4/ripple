// Instanced particle rendering. One instance per particle, 6 vertices forming a
// quad; the fragment shader carves a soft disc out of it. Positions/velocities
// are read straight from the simulation's storage buffers — no CPU readback.

struct RenderParams {
    domain_w: f32,
    domain_h: f32,
    radius: f32,
    max_speed: f32,
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>, // position within quad, -1..1
    @location(1) speed: f32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let center = pos[ii];
    let px = center.x + corner.x * R.radius;
    let py = center.y + corner.y * R.radius;
    // pixel space (y down) -> NDC (y up)
    let ndc = vec2<f32>(px / R.domain_w * 2.0 - 1.0, 1.0 - py / R.domain_h * 2.0);

    var out: VsOut;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    out.local = corner;
    out.speed = length(vel[ii]);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let d2 = dot(in.local, in.local);
    if (d2 > 1.0) { discard; }
    let t = clamp(in.speed / 450.0, 0.0, 1.0);
    let col = vec3<f32>(0.25 + 0.7 * t, 0.55 + 0.4 * t, 1.0);
    let a = smoothstep(1.0, 0.6, d2); // soft edge
    return vec4<f32>(col, a);
}
