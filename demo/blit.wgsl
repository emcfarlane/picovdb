// Display blit + debug pass visualizer. Normally samples the final
// (denoised / path-traced) texture; the Pass dropdown selects G-buffer,
// motion vectors, depth or the raw per-frame signal so the render can be
// inspected. pass_mode rides in the shared Input uniform.

struct VertexInput { @location(0) position: vec2f }
struct VertexOutput { @builtin(position) pos: vec4f, @location(0) uv: vec2f }

struct Input {
    camera_matrix: mat4x4f,
    fov_scale: f32,
    time_delta: f32,
    pixel_radius: f32,
    debug_iterations: u32,
    frame_index: u32,
    environment: u32,
    max_bounces: u32,
    emission_integral: f32,
    dome_integral: f32,
    exposure: f32,
    light_count: u32,
    white_background: u32,
    rng_frame: u32,
    pass_mode: u32,
    _pad1: u32,
    _pad2: u32,
    prev_view: mat4x4f,
    prev_camera_pos: vec3f,
    wavelength_u: f32,
}
struct Object {
    object_type: u32,
    type_index: u32,
    material_index: u32,
    _pad: u32,
    transform: mat4x4f,
    transform_inverse: mat4x4f,
    motion: mat4x4f,
}

@group(0) @binding(0) var raytracedTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var gbuffer: texture_2d<f32>;
@group(0) @binding(3) var gdepth: texture_2d<f32>;
@group(0) @binding(4) var illum: texture_2d<f32>;
@group(0) @binding(5) var<uniform> input: Input;
@group(0) @binding(6) var<storage> objects: array<Object>;

@vertex
fn vertexMain(v: VertexInput) -> VertexOutput {
    let uv = v.position * 0.5 + 0.5;
    return VertexOutput(vec4f(v.position, 0.0, 1.0), uv);
}

// World position of a G-buffer texel from its exact depth
fn world_pos(uv: vec2f, dims: vec2u, depth: f32) -> vec3f {
    let ndc = uv * 2.0 - 1.0;
    let aspect = f32(dims.x) / f32(dims.y);
    let dir_cam = normalize(vec3f(ndc.x * aspect * input.fov_scale, ndc.y * input.fov_scale, -1.0));
    let origin = (input.camera_matrix * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let dir = normalize((input.camera_matrix * vec4f(dir_cam, 0.0)).xyz);
    return origin + dir * depth;
}

@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
    // Final / Denoised / Iterations: the pipeline already wrote the display
    // texture (denoised, PT, or the iteration heatmap)
    if (input.pass_mode == 0u || input.pass_mode == 1u || input.pass_mode == 6u) {
        return textureSample(raytracedTexture, textureSampler, uv);
    }
    let dims = textureDimensions(gbuffer);
    let px = vec2i(uv * vec2f(dims));
    let g = textureLoad(gbuffer, px, 0);

    // Raw: the per-frame demodulated irradiance the denoiser consumes
    if (input.pass_mode == 2u) {
        let c = textureLoad(illum, px, 0).rgb;
        return vec4f(pow(c / (c + vec3f(1.0)), vec3f(1.0 / 2.2)), 1.0); // Reinhard+gamma
    }
    if (g.w < 0.0) { return vec4f(0.02, 0.02, 0.02, 1.0); } // miss = near-black

    // GBuffer Normals
    if (input.pass_mode == 3u) {
        return vec4f(g.xyz * 0.5 + 0.5, 1.0);
    }
    // Depth (grayscale, exact depth normalized by a nominal scene scale)
    if (input.pass_mode == 5u) {
        let depth = textureLoad(gdepth, px, 0).r;
        let d = clamp(depth / 1200.0, 0.0, 1.0);
        return vec4f(vec3f(1.0 - d), 1.0);
    }
    // Motion Vectors: reproject the current hit through its object's motion
    // + previous view, show screen-space delta (R = +x, G = +y, from 0.5)
    if (input.pass_mode == 4u) {
        let mat_idx = u32(g.w) & 3u;
        let depth = textureLoad(gdepth, px, 0).r;
        let pos = world_pos(uv, dims, depth);
        let obj = objects[min(mat_idx, 1u)];
        let pos_prev = (obj.motion * vec4f(pos, 1.0)).xyz;
        let cam_prev = (input.prev_view * vec4f(pos_prev, 1.0)).xyz;
        if (cam_prev.z >= 0.0) { return vec4f(0.5, 0.5, 0.0, 1.0); }
        let aspect = f32(dims.x) / f32(dims.y);
        let uv_prev = vec2f(
            (cam_prev.x / (-cam_prev.z)) / (aspect * input.fov_scale),
            (cam_prev.y / (-cam_prev.z)) / input.fov_scale,
        ) * 0.5 + 0.5;
        let mv = (uv - uv_prev) * 8.0; // amplify for visibility
        return vec4f(clamp(mv.x * 0.5 + 0.5, 0.0, 1.0), clamp(mv.y * 0.5 + 0.5, 0.0, 1.0), 0.5, 1.0);
    }
    return textureLoad(raytracedTexture, px, 0);
}
