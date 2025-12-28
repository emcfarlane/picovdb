struct VertexInput {
    @location(0) position: vec2f,
}

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

struct FragmentInput {
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var raytracedTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    // Convert from [-1,1] to [0,1] for UV coordinates
    let uv = input.position * 0.5 + 0.5;
    return VertexOutput(
        vec4f(input.position, 0.0, 1.0),
        uv
    );
}

@fragment
fn fragmentMain(
    input: FragmentInput,
) -> @location(0) vec4f {
    let color = textureSample(raytracedTexture, textureSampler, input.uv);
    // GPU hardware handles filtering and edge cases automatically
    return color;
}
