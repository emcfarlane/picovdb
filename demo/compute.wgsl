// Spectral path tracer over PicoVDB volumes.
//
// Light transport (NEE + MIS, Frostbite BRDF, spherical lights) and the
// spectral color model are ported from Christoph Peters' path tracer
// (https://github.com/MomentsInGraphics/path_tracer, branch "spectral",
// src/shaders/pathtrace.frag.glsl and brdfs.glsl, BSD-3 — see spectra.wgsl
// for the full notice). Requires spectra.wgsl and picovdb.wgsl.

const ENVIRONMENT_STUDIO: u32 = 0u;
const ENVIRONMENT_SKY: u32 = 1u;
const ENVIRONMENT_HDRI: u32 = 2u;
const WAVELENGTH_SAMPLE_COUNT: u32 = 4u;
const RAY_OFFSET: f32 = 2e-3;

struct Input {
    camera_matrix: mat4x4f,
    fov_scale: f32, // tan(fov * 0.5)
    time_delta: f32,
    pixel_radius: f32, // Cone spread per unit distance: 1 / (resolution.y * focal_length)
    debug_iterations: u32, // 0 = normal rendering, 1 = debug iteration heatmap
    frame_index: u32, // Accumulated frame count; 0 resets accumulation
    environment: u32, // ENVIRONMENT_*
    max_bounces: u32,
    // Spherical-light emission scale (the illuminant spectrum's integral)
    emission_integral: f32,
    // The studio dome emits the same illuminant, scaled by this
    dome_integral: f32,
    exposure: f32,
    light_count: u32,
    // 1 = primary-ray misses show a clean white backdrop; the environment
    // still lights the scene through secondary rays (photo-backplate split)
    white_background: u32,
    // Monotonic frame counter for RNG seeding (never resets, unlike
    // frame_index, so samples stay fresh while accumulation restarts)
    rng_frame: u32,
    // 0 after the reservoir textures were recreated (resize/render scale)
    reservoir_valid: u32,
    // 1 = ReSTIR DI for direct lighting at the primary hit
    restir: u32,
    _pad: u32,
    // Previous frame's view matrix (world -> camera) for ReSTIR temporal
    // reprojection; combined with per-object motion matrices
    prev_view: mat4x4f,
    // Per-frame random transform for each pairing texture:
    // (offset.x, offset.y, flip.x, flip.y) — decorrelates spatial reuse
    pairing: array<vec4u, 3>,
    // Previous frame's camera position: reconstructs a spatial neighbor's
    // view direction when evaluating their target function (pairwise MIS)
    prev_camera_pos: vec3f,
    _pad2: f32,
}

// --- Object types ---
const OBJECT_TYPE_UNKNOWN: u32 = 0u;
const OBJECT_TYPE_VDB: u32 = 1u;
const OBJECT_TYPE_SDF: u32 = 2u;

struct Object { // 208
    object_type: u32,
    type_index: u32,
    material_index: u32,
    _pad: u32,
    transform: mat4x4f,
    transform_inverse: mat4x4f,
    // Maps a current-frame world position on this object to its world
    // position last frame (identity when the object is still) — the motion
    // vector source for ReSTIR temporal reprojection
    motion: mat4x4f,
}

struct Material { // 32
    // Base color in linear Fourier sRGB
    base_color: vec3f,
    roughness: f32,
    // Colors as linear combinations of base color and pure white
    diffuse_albedo: vec2f,
    fresnel_0: vec2f,
}

// --- Bind group 0: per-frame ---
const MAX_MATERIALS: u32 = 8u;
const MAX_LIGHTS: u32 = 8u;

@group(0) @binding(0) var<uniform> input: Input;
@group(0) @binding(1) var<storage> objects: array<Object>;
@group(0) @binding(2) var<storage, read> skyState: SkyState;
// Uniforms rather than storage: small and fixed-size, and the PicoVDB data
// already uses 6 of the (often 8-10) storage-buffer slots per stage.
@group(0) @binding(3) var<uniform> materials: array<Material, MAX_MATERIALS>;
// Spherical lights: xyz = world-space center, w = radius
@group(0) @binding(4) var<uniform> lights: array<vec4f, MAX_LIGHTS>;
// Illuminant wavelength-sampling LUT (see demo/lib/illuminants.ts): a 1D
// texture stored as resolution x 1; texel = (rgb, phase)
@group(0) @binding(5) var illuminant_spectrum: texture_2d<f32>;
@group(0) @binding(6) var illuminant_sampler: sampler;
// Equirectangular HDRI environment (rgba16float), used by ENVIRONMENT_HDRI
@group(0) @binding(7) var environment_texture: texture_2d<f32>;
// Environment luminance CDFs for importance sampling (r32float, built on
// the CPU — see demo/lib/env.ts): row-normalized conditional CDF over
// columns (W x H) and marginal CDF over rows (H x 1).
@group(0) @binding(8) var env_cdf_conditional: texture_2d<f32>;
@group(0) @binding(9) var env_cdf_marginal: texture_2d<f32>;
// Pairing textures for ReSTIR paired spatial reuse (rg8sint deltas, tiled
// over the screen; see demo/lib/pairing.ts). Different sizes avoid
// tiling-alignment correlation.
@group(0) @binding(10) var pairing_0: texture_2d<i32>;
@group(0) @binding(11) var pairing_1: texture_2d<i32>;
@group(0) @binding(12) var pairing_2: texture_2d<i32>;

// -- Bind group 1: data ---
@group(1) @binding(0) var<storage> picovdb_grids: array<PicoVDBGrid>;
@group(1) @binding(1) var<storage> picovdb_roots: array<PicoVDBRoot>;
@group(1) @binding(2) var<storage> picovdb_uppers: array<PicoVDBUpper>;
@group(1) @binding(3) var<storage> picovdb_lowers: array<PicoVDBLower>;
@group(1) @binding(4) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
@group(1) @binding(5) var<storage> picovdb_buffer: array<u32>;

// --- Bind group 2: pass ---
@group(2) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
// Progressive accumulation: running radiance sum per pixel (rgb) + count (a)
@group(2) @binding(1) var<storage, read_write> accumulation: array<vec4f>;
// ReSTIR DI reservoirs, ping-ponged between frames:
// A = (direction.xyz, W), B = (shading position.xyz, confidence M)
@group(2) @binding(2) var reservoir_prev_a: texture_2d<f32>;
@group(2) @binding(3) var reservoir_prev_b: texture_2d<f32>;
@group(2) @binding(4) var reservoir_out_a: texture_storage_2d<rgba32float, write>;
@group(2) @binding(5) var reservoir_out_b: texture_storage_2d<rgba32float, write>;
// Primary-hit G-buffer (normal.xyz, material index; >= 900 marks a miss),
// ping-ponged with the reservoirs: spatial reuse evaluates each neighbor's
// target function at THEIR surface for pairwise-balance MIS. NOTE: this is
// the 4th storage texture — the WebGPU per-stage limit.
@group(2) @binding(6) var gbuffer_prev: texture_2d<f32>;
@group(2) @binding(7) var gbuffer_out: texture_storage_2d<rgba32float, write>;

const MAX_DIST: f32 = 1e7;
const PI = 3.14159265359;

struct Intersection {
    distance: f32,
    object_index: i32,
    iterations: u32,
    normal: vec3f,
}

fn no_intersection() -> Intersection {
    return Intersection(MAX_DIST, -1, 0, vec3f(0));
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

fn intersect_picovdb(
    ray: Ray,
    grid_index: u32,
    hit_distance: ptr<function, f32>,
    hit_normal: ptr<function, vec3f>,
    hit_iterations: ptr<function, u32>,
) -> bool {
    let tmin = 0.0;
    let tmax = 10000.0;

    let grid = picovdb_grids[grid_index];
    var accessor: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&accessor, grid_index);

    // Inside Check (Works even if camera is in background space)
    let start_val = picovdbSampleTrilinear(&accessor, grid, ray.origin);
    if start_val < 0.0 {
        *hit_distance = tmin;
        *hit_normal = -ray.direction;
        return true;
    }

    return picovdbHDDAZeroCrossing(
        &accessor, grid, ray.origin, tmin, ray.direction, tmax, input.pixel_radius, hit_distance, hit_normal, hit_iterations,
    );
}

// --- SDF primitives (local/index space) ---
fn sdSphere(p: vec3f, r: f32) -> f32 {
    return length(p) - r;
}

fn sdTorus(p: vec3f, major: f32, minor: f32) -> f32 {
    let q = vec2f(length(p.xz) - major, p.y);
    return length(q) - minor;
}

// Brush cursor gizmo: a wide ring (donut) with a small sphere at its centre.
// Local space: ring lies in the XZ plane, axis along +Y (the surface normal).
fn sdBrush(p: vec3f) -> f32 {
    let ring = sdTorus(p, 1.0, 0.12);
    let knob = sdSphere(p, 0.22);
    return min(ring, knob);
}

fn sdBrushNormal(p: vec3f) -> vec3f {
    let e = vec2f(0.0008, 0.0);
    return normalize(vec3f(
        sdBrush(p + e.xyy) - sdBrush(p - e.xyy),
        sdBrush(p + e.yxy) - sdBrush(p - e.yxy),
        sdBrush(p + e.yyx) - sdBrush(p - e.yyx),
    ));
}

fn intersect_sdf(
    ray: Ray,
    index: u32,
    hit_distance: ptr<function, f32>,
    hit_normal: ptr<function, vec3f>,
    iterations: ptr<function, u32>,
) -> bool {
    switch index {
        case 0u: { // ground plane at y=0 in index space
            if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {
                return false;
            }
            let t = -ray.origin.y / ray.direction.y;
            if t < 0.001 {
                return false;
            }
            *hit_distance = t;
            *hit_normal = vec3f(0, 1, 0);
            return true;
        }
        case 1u: { // brush cursor gizmo — sphere-traced torus + centre sphere
            var t = 0.0;
            for (var i = 0u; i < 96u; i++) {
                let p = ray.origin + ray.direction * t;
                let d = sdBrush(p);
                if (d < 0.001) {
                    *hit_distance = t;
                    *hit_normal = sdBrushNormal(p);
                    *iterations = i;
                    return true;
                }
                t += d;
                if (t > 50.0) { break; }
            }
            return false;
        }
        case default: { return false; }
    }
}

fn intersect_scene(world_ray: Ray, iterations: ptr<function, u32>) -> Intersection {
    var min_hit = no_intersection();
    for (var i = 0i; i < i32(arrayLength(&objects)); i++) {
        let obj = objects[i];
        let idx_origin = (obj.transform * vec4f(world_ray.origin, 1.0)).xyz;
        let idx_dir_unnorm = (obj.transform * vec4f(world_ray.direction, 0.0)).xyz;
        let idx_direction = normalize(idx_dir_unnorm);
        let index_ray = Ray(idx_origin, idx_direction);

        var hit = false;
        var hit_distance = MAX_DIST;
        var hit_normal = vec3f(0);
        var hit_iterations = 0u;
        switch obj.object_type {
            case OBJECT_TYPE_VDB: {
                // Skip fog grids during surface intersection — they use volumetric marching instead
                let vdb_grid = picovdb_grids[obj.type_index];
                if (vdb_grid.gridType != GRID_TYPE_FOG_FLOAT) {
                    hit = intersect_picovdb(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);
                }
            }
            case OBJECT_TYPE_SDF: {
                hit = intersect_sdf(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);
            }
            case default: {
                hit = false;
            }
        }
        *iterations += hit_iterations;
        if !hit {
            continue;
        }
        let index_hit_point = index_ray.origin + index_ray.direction * hit_distance;
        let world_hit_point = (obj.transform_inverse * vec4f(index_hit_point, 1.0)).xyz;
        let world_distance = length(world_hit_point - world_ray.origin);
        if world_distance >= min_hit.distance {
            continue;
        }

        min_hit.distance = world_distance;
        min_hit.object_index = i;
        min_hit.normal = (obj.transform_inverse * vec4f(hit_normal, 0.0)).xyz;
    }
    min_hit.normal = normalize(min_hit.normal);
    return min_hit;
}

fn generate_camera_ray(screen_coord: vec2f, screen_size: vec2f) -> Ray {
    // Convert to normalized coordinates [-1, 1k
    let uv = (screen_coord / screen_size) * 2.0 - 1.0;

    // Calculate aspect ratio
    let aspect_ratio = screen_size.x / screen_size.y;

    // Extract camera basis vectors from view matrix
    let right: vec3f = input.camera_matrix[0].xyz;
    let up: vec3f = input.camera_matrix[1].xyz;
    let forward: vec3f = -input.camera_matrix[2].xyz;

    // Extract camera position
    let camera_pos: vec3f = input.camera_matrix[3].xyz;

    // Calculate ray direction
    let ray_direction = normalize(
        forward + uv.x * right * aspect_ratio * input.fov_scale + uv.y * up * input.fov_scale
    );
    return Ray(camera_pos, ray_direction);
}

// ============================================================================
// Random numbers
// ============================================================================

// PCG2D, as described here: https://jcgt.org/published/0009/03/02/
// Returns a uniform, pseudo-random point in [0,1)^2 and updates the seed.
fn get_random_numbers(seed: ptr<function, vec2u>) -> vec2f {
    var s = 1664525u * (*seed) + 1013904223u;
    s.x += 1664525u * s.y;
    s.y += 1664525u * s.x;
    s ^= (s >> vec2u(16u));
    s.x += 1664525u * s.y;
    s.y += 1664525u * s.x;
    s ^= (s >> vec2u(16u));
    *seed = s;
    return vec2f(s) * 2.32830643654e-10;
}

// ============================================================================
// Sampling and the Frostbite BRDF (colors are linear combinations of the
// base color and pure white, expressed as vec2(base_weight, white_weight))
// ============================================================================

struct ShadingData {
    pos: vec3f,
    normal: vec3f,
    out_dir: vec3f, // towards the camera along the path
    lambert_out: f32, // dot(normal, out_dir)
    base_color: vec3f, // linear sRGB or Fourier sRGB depending on color model
    diffuse_albedo: vec2f,
    fresnel_0: vec2f,
    roughness: f32,
}

// Constructs a special orthogonal matrix where the given normalized normal
// vector is the third column: https://www.jcgt.org/published/0006/01/01/
fn get_shading_space(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z > 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let b1 = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let b2 = vec3f(b, s + n.y * n.y * a, -n.y);
    return mat3x3f(b1, b2, n);
}

fn fresnel_schlick_2(f_0: vec2f, f_90: vec2f, lambert: f32) -> vec2f {
    let flip_1 = 1.0 - lambert;
    let flip_2 = flip_1 * flip_1;
    let flip_5 = flip_2 * flip_1 * flip_2;
    return flip_5 * (f_90 - f_0) + f_0;
}

// The Frostbite BRDF (Disney diffuse + GGX/Smith specular):
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
fn frostbite_brdf(s: ShadingData, in_dir: vec3f) -> vec2f {
    let lambert_in = dot(s.normal, in_dir);
    if (min(lambert_in, s.lambert_out) < 0.0) {
        return vec2f(0.0);
    }
    let half_dir = normalize(in_dir + s.out_dir);
    let half_dot_out = dot(half_dir, s.out_dir);
    // Disney diffuse
    let f_90 = (half_dot_out * half_dot_out) * (2.0 * s.roughness) + 0.5;
    let fresnel_diffuse =
        fresnel_schlick_2(vec2f(1.0), vec2f(f_90), s.lambert_out).x *
        fresnel_schlick_2(vec2f(1.0), vec2f(f_90), lambert_in).x;
    var brdf = fresnel_diffuse * s.diffuse_albedo;
    // GGX normal distribution
    let half_dot_normal = dot(half_dir, s.normal);
    let roughness_2 = s.roughness * s.roughness;
    var ggx = (roughness_2 * half_dot_normal - half_dot_normal) * half_dot_normal + 1.0;
    ggx = roughness_2 / (ggx * ggx);
    // Smith masking-shadowing
    let masking = lambert_in * sqrt((s.lambert_out - roughness_2 * s.lambert_out) * s.lambert_out + roughness_2);
    let shadowing = s.lambert_out * sqrt((lambert_in - roughness_2 * lambert_in) * lambert_in + roughness_2);
    let smith = 0.5 / (masking + shadowing);
    // Fresnel-Schlick
    let fresnel = fresnel_schlick_2(s.fresnel_0, vec2f(0.0, 1.0), max(0.0, half_dot_out));
    brdf += ggx * smith * fresnel;
    return brdf * (1.0 / PI);
}

// Samples the distribution of visible normals in the GGX normal distribution
// function (in shading space). Returns the half vector.
// https://doi.org/10.1111/cgf.14867
fn sample_ggx_vndf(out_dir: vec3f, roughness: vec2f, randoms: vec2f) -> vec3f {
    // Warp to the hemisphere configuration
    let out_dir_std = normalize(vec3f(out_dir.xy * roughness, out_dir.z));
    // Sample a spherical cap in (-out_dir_std.z, 1]
    let azimuth = (2.0 * PI) * randoms[0] - PI;
    let z = 1.0 - randoms[1] * (1.0 + out_dir_std.z);
    let sine = sqrt(max(0.0, 1.0 - z * z));
    let cap = vec3f(sine * cos(azimuth), sine * sin(azimuth), z);
    // Compute the half vector in the hemisphere configuration
    let half_dir_std = cap + out_dir_std;
    // Warp back to the ellipsoid configuration
    return normalize(vec3f(half_dir_std.xy * roughness, half_dir_std.z));
}

// Density w.r.t. solid angle sampled by sample_ggx_vndf() for the half vector.
fn get_ggx_vndf_density(lambert_out: f32, half_dot_normal: f32, half_dot_out: f32, roughness: f32) -> f32 {
    if (half_dot_normal < 0.0) {
        return 0.0;
    }
    let roughness_2 = roughness * roughness;
    let flip_roughness_2 = 1.0 - roughness_2;
    let length_M_inv_out_2 = roughness_2 + flip_roughness_2 * lambert_out * lambert_out;
    let D_vis_std = max(0.0, half_dot_out) * (2.0 / PI) / (lambert_out + sqrt(length_M_inv_out_2));
    let length_M_half_2 = 1.0 - flip_roughness_2 * half_dot_normal * half_dot_normal;
    return D_vis_std * roughness_2 / (length_M_half_2 * length_M_half_2);
}

fn sample_ggx_in_dir(out_dir: vec3f, roughness: f32, randoms: vec2f) -> vec3f {
    let half_dir = sample_ggx_vndf(out_dir, vec2f(roughness), randoms);
    return -reflect(out_dir, half_dir);
}

fn get_ggx_in_dir_density(lambert_out: f32, out_dir: vec3f, in_dir: vec3f, normal: vec3f, roughness: f32) -> f32 {
    let half_dir = normalize(in_dir + out_dir);
    let half_dot_out = dot(half_dir, out_dir);
    let half_dot_normal = dot(half_dir, normal);
    let density = get_ggx_vndf_density(lambert_out, half_dot_normal, half_dot_out, roughness);
    return density / (4.0 * half_dot_out);
}

// Uniform w.r.t. projected solid angle in the upper hemisphere (positive z).
fn sample_hemisphere_psa(randoms: vec2f) -> vec3f {
    let azimuth = (2.0 * PI) * randoms[0] - PI;
    let radius = sqrt(randoms[1]);
    let z = sqrt(1.0 - radius * radius);
    return vec3f(radius * cos(azimuth), radius * sin(azimuth), z);
}

fn get_hemisphere_psa_density(sampled_dir_z: f32) -> f32 {
    return (1.0 / PI) * max(0.0, sampled_dir_z);
}

// Probability of using projected-solid-angle sampling vs GGX VNDF sampling
// for the given shading point (defensive: at least 50% specular).
fn get_diffuse_sampling_probability(s: ShadingData) -> f32 {
    return min(0.5, dot(s.base_color, vec3f(0.2126, 0.7152, 0.0722)));
}

fn sample_frostbite_brdf(s: ShadingData, randoms_in: vec2f) -> vec3f {
    var randoms = randoms_in;
    let shading_to_world_space = get_shading_space(s.normal);
    let diffuse_prob = get_diffuse_sampling_probability(s);
    let diffuse = randoms[0] < diffuse_prob;
    var sampled_dir: vec3f;
    if (diffuse) {
        randoms[0] /= diffuse_prob;
        sampled_dir = shading_to_world_space * sample_hemisphere_psa(randoms);
    } else {
        randoms[0] = (randoms[0] - diffuse_prob) / (1.0 - diffuse_prob);
        let local_out_dir = transpose(shading_to_world_space) * s.out_dir;
        let local_in_dir = sample_ggx_in_dir(local_out_dir, s.roughness, randoms);
        sampled_dir = shading_to_world_space * local_in_dir;
    }
    return sampled_dir;
}

fn get_frostbite_brdf_density(s: ShadingData, sampled_dir: vec3f) -> f32 {
    let diffuse_prob = get_diffuse_sampling_probability(s);
    let specular_density = get_ggx_in_dir_density(s.lambert_out, s.out_dir, sampled_dir, s.normal, s.roughness);
    let diffuse_density = get_hemisphere_psa_density(dot(s.normal, sampled_dir));
    return mix(specular_density, diffuse_density, diffuse_prob);
}

// ============================================================================
// Spherical lights (NEE + MIS)
// ============================================================================

// Solid angle of the light divided by 2*pi, or 0 below the horizon.
fn get_spherical_light_importance(center: vec3f, radius: f32, shading_pos: vec3f, normal: vec3f) -> f32 {
    let center_dir = center - shading_pos;
    if (dot(normal, center_dir) < -radius) {
        return 0.0;
    }
    let center_dist_2 = dot(center_dir, center_dir);
    let sin_2 = radius * radius / center_dist_2;
    let z_range = sin_2 / (1.0 + sqrt(max(0.0, 1.0 - sin_2)));
    return z_range;
}

// Samples the solid angle of the given spherical light uniformly. The sampled
// density w.r.t. solid angle is 1 / (2*pi*importance).
fn sample_spherical_light(center: vec3f, importance: f32, shading_pos: vec3f, randoms: vec2f) -> vec3f {
    let azimuth = (2.0 * PI) * randoms[0] - PI;
    let z = 1.0 - importance * randoms[1];
    let r = sqrt(max(0.0, 1.0 - z * z));
    let local_dir = vec3f(r * cos(azimuth), r * sin(azimuth), z);
    let light_to_world_space = get_shading_space(normalize(center - shading_pos));
    return light_to_world_space * local_dir;
}

// Picks a light proportionally to its importance and samples its solid angle.
fn sample_lights(out_total_importance: ptr<function, f32>, shading_pos: vec3f, normal: vec3f, randoms_in: vec2f) -> vec3f {
    var randoms = randoms_in;
    var total_importance = 0.0;
    for (var i = 0u; i < input.light_count; i++) {
        total_importance += get_spherical_light_importance(lights[i].xyz, lights[i].w, shading_pos, normal);
    }
    *out_total_importance = total_importance;
    let target_importance = randoms[0] * total_importance;
    var prefix_importance = 0.0;
    for (var i = 0u; i < input.light_count; i++) {
        let light = lights[i];
        let importance = get_spherical_light_importance(light.xyz, light.w, shading_pos, normal);
        prefix_importance += importance;
        if (prefix_importance > target_importance) {
            // Reuse the random number
            randoms[0] = (target_importance + importance - prefix_importance) / importance;
            return sample_spherical_light(light.xyz, importance, shading_pos, randoms);
        }
    }
    return vec3f(0.0);
}

// Density w.r.t. solid angle sampled by sample_lights(). is_light_dir works
// around numerical issues for directions constructed towards a light.
fn get_lights_density(total_importance: f32, shading_pos: vec3f, sampled_dir: vec3f, is_light_dir: bool) -> f32 {
    if (total_importance <= 0.0) {
        return 0.0;
    }
    var light_count = 0.0;
    for (var i = 0u; i < input.light_count; i++) {
        let light = lights[i];
        let center_dir = light.xyz - shading_pos;
        let center_dist_2 = dot(center_dir, center_dir);
        let center_dot_dir = dot(center_dir, sampled_dir);
        let radius_2 = light.w * light.w;
        let in_sphere = center_dist_2 - radius_2;
        let discriminant = center_dot_dir * center_dot_dir - in_sphere;
        light_count += select(0.0, 1.0, discriminant >= 0.0 && in_sphere >= 0.0 && center_dot_dir >= 0.0);
    }
    if (is_light_dir) {
        light_count = max(1.0, light_count);
    }
    return light_count / (2.0 * PI * total_importance);
}

struct LightHit {
    t: f32,
    index: i32,
}

// Nearest analytic intersection with any spherical light.
fn intersect_lights(ray: Ray) -> LightHit {
    var best = LightHit(MAX_DIST, -1);
    for (var i = 0u; i < input.light_count; i++) {
        let light = lights[i];
        let oc = ray.origin - light.xyz;
        let b = dot(oc, ray.direction);
        let c = dot(oc, oc) - light.w * light.w;
        let discriminant = b * b - c;
        if (discriminant < 0.0) {
            continue;
        }
        let t = -b - sqrt(discriminant);
        if (t > 1e-4 && t < best.t) {
            best = LightHit(t, i32(i));
        }
    }
    return best;
}

// ============================================================================
// ReSTIR DI (temporal-only v1, see docs/restir-plan.md step R1)
//
// Per-pixel reservoirs over *direction* samples at the primary hit, in
// solid-angle measure anchored at the shading point. Candidates come from
// every light-sampling technique we have (lamps, HDRI CDF, sun cone, cosine
// hemisphere, one BRDF sample) under their mixture pdf; temporal reuse
// merges the previous frame's reservoir after a shading-position validation
// (so the shift-mapping Jacobian is ~1 and reuse survives accumulation
// resets). One shadow ray shades the winner. Spatial (paired) reuse and
// unbiased MIS-weighted merging are follow-ups.
// ============================================================================

const RESTIR_LAMP_CANDIDATES: u32 = 4u;
const RESTIR_ENV_CANDIDATES: u32 = 4u;
const RESTIR_M_INIT: f32 = 9.0; // lamps + env + 1 BRDF candidate
// Low cap = fast sample turnover: shading correlated (locked) reservoirs
// into the accumulator freezes noise into the image
const RESTIR_CONFIDENCE_CAP: f32 = 64.0;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Unshadowed emitted radiance (RGB) reaching pos from direction dir:
// analytic lamp hit or the environment. No rays traced.
fn direct_radiance_rgb(pos: vec3f, dir: vec3f) -> vec3f {
    let light_hit = intersect_lights(Ray(pos, dir));
    if (light_hit.index >= 0) {
        return vec3f(input.emission_integral);
    }
    if (input.environment == ENVIRONMENT_STUDIO) {
        return vec3f(input.dome_integral);
    }
    return environment_rgb(dir);
}

// Scalar target function p-hat: luminance of the unshadowed contribution.
fn restir_target(s: ShadingData, lum_albedo: f32, dir: vec3f) -> f32 {
    let lambert = dot(s.normal, dir);
    if (lambert <= 0.0) {
        return 0.0;
    }
    let brdf = frostbite_brdf(s, dir);
    return (lum_albedo * brdf.x + brdf.y) * lambert * luminance(direct_radiance_rgb(s.pos, dir));
}

// True mixture pdf of the initial-candidate pool for a direction.
fn restir_source_pdf(s: ShadingData, total_light_importance: f32, dir: vec3f) -> f32 {
    var m = f32(RESTIR_LAMP_CANDIDATES) * get_lights_density(total_light_importance, s.pos, dir, false);
    if (input.environment == ENVIRONMENT_HDRI) {
        m += f32(RESTIR_ENV_CANDIDATES) * get_environment_density(dir);
    } else if (input.environment == ENVIRONMENT_SKY) {
        m += f32(RESTIR_ENV_CANDIDATES / 2u) * get_sun_density(dir);
        m += f32(RESTIR_ENV_CANDIDATES - RESTIR_ENV_CANDIDATES / 2u) * get_hemisphere_psa_density(dot(s.normal, dir));
    } else {
        m += f32(RESTIR_ENV_CANDIDATES) * get_hemisphere_psa_density(dot(s.normal, dir));
    }
    m += get_frostbite_brdf_density(s, dir);
    return m / RESTIR_M_INIT;
}

struct Reservoir {
    dir: vec3f,
    w_sum: f32,
    M: f32,
}

fn reservoir_update(r: ptr<function, Reservoir>, dir: vec3f, w: f32, rand: f32) {
    (*r).w_sum += w;
    if (w > 0.0 && rand * (*r).w_sum <= w) {
        (*r).dir = dir;
    }
}

fn restir_candidate(r: ptr<function, Reservoir>, s: ShadingData, lum_albedo: f32, total_light_importance: f32, dir: vec3f, rand: f32) {
    if (dot(dir, dir) < 0.5) {
        return; // degenerate sample (e.g. no lamps); still counted in M_INIT
    }
    let q = restir_source_pdf(s, total_light_importance, dir);
    if (q <= 0.0) {
        return;
    }
    reservoir_update(r, dir, restir_target(s, lum_albedo, dir) / q, rand);
}

// Paired-neighbor delta for the current pixel from one pairing texture,
// with the per-frame random flip/offset transform applied. Flips of the
// lookup coordinate require flipping the delta back (keeps A<->B mutual).
fn pairing_delta(delta_raw: vec2i, flip: vec2u) -> vec2i {
    return vec2i(
        select(delta_raw.x, -delta_raw.x, flip.x == 1u),
        select(delta_raw.y, -delta_raw.y, flip.y == 1u),
    );
}

fn pairing_coord(pixel: vec2u, tex_dims: vec2u, transform: vec4u) -> vec2u {
    var p = vec2u(pixel.x % tex_dims.x, pixel.y % tex_dims.y);
    if (transform.z == 1u) {
        p.x = tex_dims.x - 1u - p.x;
    }
    if (transform.w == 1u) {
        p.y = tex_dims.y - 1u - p.y;
    }
    return vec2u((p.x + transform.x) % tex_dims.x, (p.y + transform.y) % tex_dims.y);
}

// Target function evaluated at a spatial neighbor's surface, reconstructed
// from the reservoir position, G-buffer normal/material, and the previous
// camera position (their view direction). Needed for pairwise-balance MIS.
fn restir_target_at(pos: vec3f, normal: vec3f, mat_index: u32, dir: vec3f) -> f32 {
    let lambert = dot(normal, dir);
    if (lambert <= 0.0) {
        return 0.0;
    }
    let mat = materials[mat_index];
    var ns: ShadingData;
    ns.pos = pos;
    ns.normal = normal;
    ns.out_dir = normalize(input.prev_camera_pos - pos);
    ns.lambert_out = max(dot(normal, ns.out_dir), 1e-4);
    ns.base_color = mat.base_color;
    ns.diffuse_albedo = mat.diffuse_albedo;
    ns.fresnel_0 = mat.fresnel_0;
    ns.roughness = mat.roughness;
    let brdf = frostbite_brdf(ns, dir);
    return (luminance(mat.base_color) * brdf.x + brdf.y) * lambert * luminance(direct_radiance_rgb(pos, dir));
}

// One spatial neighbor's previous-frame state for the pairwise-MIS merge
struct SpatialNeighbor {
    dir: vec3f,
    w: f32,   // unbiased contribution weight W (may be 0: dead reservoir)
    m: f32,   // confidence, capped (neighbors are hints)
    pos: vec3f,
    normal: vec3f,
    mat_index: u32,
    valid: bool,
}

fn load_spatial_neighbor(neighbor: vec2i, dims: vec2i) -> SpatialNeighbor {
    var out: SpatialNeighbor;
    out.valid = false;
    if (any(neighbor < vec2i(0)) || any(neighbor >= dims)) {
        return out;
    }
    let a = textureLoad(reservoir_prev_a, neighbor, 0);
    let b = textureLoad(reservoir_prev_b, neighbor, 0);
    let g = textureLoad(gbuffer_prev, neighbor, 0);
    let m = min(b.w, 20.0);
    if (m <= 0.0 || g.w >= 900.0 || !(a.w >= 0.0 && a.w < 1e12)) {
        return out;
    }
    out.dir = a.xyz;
    out.w = a.w;
    out.m = m;
    out.pos = b.xyz;
    out.normal = g.xyz;
    out.mat_index = u32(g.w);
    out.valid = out.mat_index < MAX_MATERIALS;
    return out;
}

// ============================================================================
// Path tracing
// ============================================================================

// Environment radiance in linear sRGB for the sky and HDRI environments.
// Both are RGB-only; the spectral path applies them with the mean throughput
// (approximate — spectral upsampling of environment maps is future work).
fn environment_rgb(direction: vec3f) -> vec3f {
    if (input.environment == ENVIRONMENT_HDRI) {
        // Equirectangular lookup; dome_integral scales the map's intensity
        let d = normalize(direction);
        let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
        let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
        return textureSampleLevel(environment_texture, illuminant_sampler, vec2f(u, v), 0.0).rgb * input.dome_integral;
    }
    return skyRadianceRGB(direction, true);
}

// ============================================================================
// Sun sampling (sky environment). The Hosek-Wilkie solar disk (0.255 deg)
// AND its circumsolar halo (the Mie term spikes ~1000x within a couple of
// degrees of the sun) are far too bright for BRDF sampling alone (fireflies).
// A 2.5-degree cone covering both gets NEE + MIS; the whole sky radiance
// inside the cone is treated as the "sun light".
// ============================================================================

// The disk is ~25000x brighter than the sky and ~100x brighter than the
// halo, so it needs its own sampling tier: sample_sun() picks the disk or
// the halo annulus 50/50, giving a piecewise pdf that keeps every MIS
// weight bounded (a uniform cone pdf leaves 1000x disk jackpots).
const SUN_DISK_COS: f32 = 0.9999900961451244; // cos(0.255 deg)
const SUN_DISK_SOLID_ANGLE: f32 = 6.22277554391107e-05;
const SUN_CONE_COS: f32 = 0.9990482215818578; // cos(2.5 deg)
const SUN_ANNULUS_SOLID_ANGLE: f32 = 5.9179924171e-03;

fn sample_sun(randoms: vec2f) -> vec3f {
    var z: f32;
    if (randoms.y < 0.5) {
        z = 1.0 - (randoms.y * 2.0) * (1.0 - SUN_DISK_COS);
    } else {
        z = SUN_DISK_COS - ((randoms.y - 0.5) * 2.0) * (SUN_DISK_COS - SUN_CONE_COS);
    }
    let r = sqrt(max(0.0, 1.0 - z * z));
    let azimuth = (2.0 * PI) * randoms.x - PI;
    let sun_to_world = get_shading_space(skyState.sunDirection);
    return sun_to_world * vec3f(r * cos(azimuth), r * sin(azimuth), z);
}

// Solid-angle density sample_sun() assigns to a direction (MIS)
fn get_sun_density(direction: vec3f) -> f32 {
    let c = dot(direction, skyState.sunDirection);
    if (c >= SUN_DISK_COS) {
        return 0.5 / SUN_DISK_SOLID_ANGLE + 0.5 / SUN_ANNULUS_SOLID_ANGLE;
    }
    if (c >= SUN_CONE_COS) {
        return 0.5 / SUN_ANNULUS_SOLID_ANGLE;
    }
    return 0.0;
}

// ============================================================================
// Environment importance sampling (HDRI): PBRT-style 2D luminance
// distribution. This is also the candidate sampler ReSTIR DI will reuse.
// ============================================================================

// Solid-angle pdf of the texel (x, y) under the luminance distribution.
fn env_texel_pdf(x: u32, y: u32, dims: vec2u) -> f32 {
    var cond = textureLoad(env_cdf_conditional, vec2u(x, y), 0).r;
    if (x > 0u) {
        cond -= textureLoad(env_cdf_conditional, vec2u(x - 1u, y), 0).r;
    }
    var marg = textureLoad(env_cdf_marginal, vec2u(y, 0u), 0).r;
    if (y > 0u) {
        marg -= textureLoad(env_cdf_marginal, vec2u(y - 1u, 0u), 0).r;
    }
    let sin_theta = sin((f32(y) + 0.5) * PI / f32(dims.y));
    return cond * marg * f32(dims.x) * f32(dims.y) / (2.0 * PI * PI * max(sin_theta, 1e-4));
}

// Solid-angle pdf that sample_environment() would assign to a direction (MIS)
fn get_environment_density(direction: vec3f) -> f32 {
    let dims = textureDimensions(environment_texture);
    let d = normalize(direction);
    let u = atan2(d.z, d.x) * (0.5 / PI) + 0.5;
    let v = acos(clamp(d.y, -1.0, 1.0)) * (1.0 / PI);
    let x = min(u32(u * f32(dims.x)), dims.x - 1u);
    let y = min(u32(v * f32(dims.y)), dims.y - 1u);
    return env_texel_pdf(x, y, dims);
}

// Samples a direction proportional to environment luminance (inverse-CDF,
// binary search over the marginal then the row's conditional CDF).
fn sample_environment(randoms: vec2f, out_pdf: ptr<function, f32>) -> vec3f {
    let dims = textureDimensions(environment_texture);
    var lo = 0u;
    var hi = dims.y - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (textureLoad(env_cdf_marginal, vec2u(mid, 0u), 0).r < randoms.x) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let y = lo;
    lo = 0u;
    hi = dims.x - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (textureLoad(env_cdf_conditional, vec2u(mid, y), 0).r < randoms.y) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let x = lo;
    *out_pdf = env_texel_pdf(x, y, dims);
    let phi = ((f32(x) + 0.5) / f32(dims.x) - 0.5) * (2.0 * PI);
    let theta = (f32(y) + 0.5) / f32(dims.y) * PI;
    let sin_theta = sin(theta);
    return vec3f(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
}

// Monte Carlo estimate of the radiance received along a ray, as linear sRGB.
// Spectral path tracing with NEE and MIS (Peters' method): jittered-
// stratified wavelength samples from the illuminant LUT, one throughput
// weight per wavelength.
fn clear_reservoir(pixel: vec2u) {
    textureStore(reservoir_out_a, pixel, vec4f(0.0));
    textureStore(reservoir_out_b, pixel, vec4f(0.0));
    textureStore(gbuffer_out, pixel, vec4f(0.0, 0.0, 0.0, 999.0));
}

fn path_trace(ray_in: Ray, pixel: vec2u, seed: ptr<function, vec2u>, iterations: ptr<function, u32>) -> vec3f {
    var ray = ray_in;
    let use_restir = input.restir == 1u;
    // MIS weight applied when a BRDF-sampled ray reaches the environment
    // (competes with sample_environment() NEE from the previous vertex)
    var env_weight = 1.0;
    // With ReSTIR, emission seen from the primary hit (k == 2 arrivals) is
    // entirely the reservoir's job — this zeroes it to avoid double counting
    var direct_scale = 1.0;

    var rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>;
    var throughput: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    var nee_throughput: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    let wavelength_rand = get_random_numbers(seed).x;
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        let u = wavelength_rand * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT)) + f32(i) / f32(WAVELENGTH_SAMPLE_COUNT);
        rgb_and_phases[i] = textureSampleLevel(illuminant_spectrum, illuminant_sampler, vec2f(u, 0.5), 0.0);
        throughput[i] = 1.0;
        nee_throughput[i] = 1.0;
    }

    var radiance = vec3f(0.0);
    for (var k = 1u; k <= input.max_bounces; k++) {
        let light_hit = intersect_lights(ray);
        let hit = intersect_scene(ray, iterations);
        // Direct view / BRDF-sampled hit of a light: MIS-weighted emission
        if (light_hit.index >= 0 && light_hit.t < hit.distance) {
            for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                radiance += (nee_throughput[i] * direct_scale * input.emission_integral) * rgb_and_phases[i].rgb;
            }
            if (k == 1u) {
                clear_reservoir(pixel);
            }
            break;
        }
        // Miss: environment. The dome emits the illuminant spectrum; sky and
        // HDRI are RGB-only and approximated with the mean throughput.
        if (hit.object_index < 0) {
            if (k == 1u) {
                clear_reservoir(pixel);
            }
            // Primary-ray miss with the white backdrop: display-only white
            // (scaled so it tonemaps to white); the path ends, so the
            // backdrop does not light the scene.
            if (k == 1u && input.white_background == 1u) {
                radiance = vec3f(6.0 / input.exposure);
                break;
            }
            if (input.environment == ENVIRONMENT_STUDIO) {
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    radiance += (throughput[i] * direct_scale * input.dome_integral) * rgb_and_phases[i].rgb;
                }
            } else if (input.environment == ENVIRONMENT_SKY) {
                // Sky radiance inside the sun cone (disk + circumsolar halo)
                // is MIS-weighted against the previous vertex's sun NEE;
                // the smooth sky outside the cone comes in at full weight.
                var env = skyRadianceRGB(ray.direction, true);
                if (dot(ray.direction, skyState.sunDirection) >= SUN_CONE_COS) {
                    env *= env_weight;
                }
                let mean = (throughput[0] + throughput[1] + throughput[2] + throughput[3]) * 0.25;
                radiance += (mean * direct_scale) * env;
            } else {
                let mean = (throughput[0] + throughput[1] + throughput[2] + throughput[3]) * 0.25;
                radiance += (mean * env_weight * direct_scale) * environment_rgb(ray.direction);
            }
            break;
        }

        // Shading data at the hit point
        let obj = objects[hit.object_index];
        let mat = materials[obj.material_index];
        var s: ShadingData;
        s.pos = ray.origin + ray.direction * hit.distance;
        s.normal = hit.normal;
        // Degenerate gradients (flat trilinear stencils) normalize to NaN;
        // fall back to facing the ray. (dot is NaN -> comparison false.)
        if (!(dot(s.normal, s.normal) > 0.5)) {
            s.normal = -ray.direction;
        }
        // Flip the normal to the ray side (double-sided shading)
        if (dot(s.normal, ray.direction) > 0.0) {
            s.normal = -s.normal;
        }
        s.out_dir = -ray.direction;
        s.lambert_out = dot(s.normal, s.out_dir);
        s.base_color = mat.base_color;
        s.diffuse_albedo = mat.diffuse_albedo;
        s.fresnel_0 = mat.fresnel_0;
        s.roughness = mat.roughness;
        s.pos += s.normal * RAY_OFFSET;

        // Evaluate the reflectance spectrum at the sampled wavelengths
        var reflectance: array<f32, WAVELENGTH_SAMPLE_COUNT>;
        let fourier = fourier_srgb_to_fourier(s.base_color);
        let lagranges = prep_reflectance_real_lagrange_biased_3(fourier);
        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
            reflectance[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, lagranges);
        }

        if (k == input.max_bounces) {
            if (k == 1u) {
                clear_reservoir(pixel);
            }
            break;
        }

        // === ReSTIR DI at the primary hit ===
        if (use_restir && k == 1u) {
            let lum_albedo = luminance(s.base_color);
            var tli = 0.0;
            for (var i = 0u; i < input.light_count; i++) {
                tli += get_spherical_light_importance(lights[i].xyz, lights[i].w, s.pos, s.normal);
            }
            var r: Reservoir;
            r.dir = vec3f(0.0);
            r.w_sum = 0.0;
            r.M = RESTIR_M_INIT;
            // Initial candidates (no rays: target is unshadowed)
            for (var c = 0u; c < RESTIR_LAMP_CANDIDATES; c++) {
                var unused: f32;
                let dir = sample_lights(&unused, s.pos, s.normal, get_random_numbers(seed));
                restir_candidate(&r, s, lum_albedo, tli, dir, get_random_numbers(seed).x);
            }
            let shading_space = get_shading_space(s.normal);
            for (var c = 0u; c < RESTIR_ENV_CANDIDATES; c++) {
                var dir: vec3f;
                if (input.environment == ENVIRONMENT_HDRI) {
                    var env_pdf_unused: f32;
                    dir = sample_environment(get_random_numbers(seed), &env_pdf_unused);
                } else if (input.environment == ENVIRONMENT_SKY && c < RESTIR_ENV_CANDIDATES / 2u) {
                    dir = sample_sun(get_random_numbers(seed));
                } else {
                    dir = shading_space * sample_hemisphere_psa(get_random_numbers(seed));
                }
                restir_candidate(&r, s, lum_albedo, tli, dir, get_random_numbers(seed).x);
            }
            restir_candidate(&r, s, lum_albedo, tli, sample_frostbite_brdf(s, get_random_numbers(seed)), get_random_numbers(seed).x);

            // Temporal reuse with motion-vector reprojection: find where
            // this surface point was last frame (object motion + previous
            // camera), and validate the stored reservoir belonged to it —
            // ReSTIR's own invalidation, independent of PT accumulation.
            if (input.reservoir_valid == 1u) {
                let dims = vec2i(textureDimensions(reservoir_prev_a));
                let pos_prev = (obj.motion * vec4f(s.pos, 1.0)).xyz;
                let cam_prev = (input.prev_view * vec4f(pos_prev, 1.0)).xyz;
                if (cam_prev.z < 0.0) {
                    let aspect = f32(dims.x) / f32(dims.y);
                    let uv = vec2f(
                        (cam_prev.x / (-cam_prev.z)) / (aspect * input.fov_scale),
                        (cam_prev.y / (-cam_prev.z)) / input.fov_scale,
                    ) * 0.5 + 0.5;
                    let prev_pixel = vec2i(uv * vec2f(dims));
                    if (all(prev_pixel >= vec2i(0)) && all(prev_pixel < dims)) {
                        let prev_a = textureLoad(reservoir_prev_a, prev_pixel, 0);
                        let prev_b = textureLoad(reservoir_prev_b, prev_pixel, 0);
                        let prev_m = min(prev_b.w, RESTIR_CONFIDENCE_CAP);
                        let pos_eps = 0.01 * hit.distance + 1e-3;
                        if (prev_m > 0.0 && prev_a.w > 0.0 && prev_a.w < 1e12 &&
                            distance(prev_b.xyz, pos_prev) < pos_eps) {
                            let f_prev = restir_target(s, lum_albedo, prev_a.xyz);
                            reservoir_update(&r, prev_a.xyz, f_prev * prev_a.w * prev_m, get_random_numbers(seed).x);
                            r.M += prev_m;
                        }
                    }
                }
            }

            // Paired spatial reuse — SHADING ONLY (persisting neighbor mass
            // creates mutual A<->B feedback that inflates energy). Merged
            // with pairwise-balance MIS: every sample is weighted by
            // confidence x its own surface's target over the sum across all
            // participating surfaces, so the weights partition unity and
            // the naive-M bias disappears. Target evals are ray-free.
            var neighbors: array<SpatialNeighbor, 3>;
            if (input.reservoir_valid == 1u) {
                let dims_i = vec2i(textureDimensions(reservoir_prev_a));
                {
                    let t = input.pairing[0];
                    let td = textureDimensions(pairing_0);
                    let d = pairing_delta(textureLoad(pairing_0, pairing_coord(pixel, td, t), 0).xy, t.zw);
                    neighbors[0] = load_spatial_neighbor(vec2i(pixel) + d, dims_i);
                }
                {
                    let t = input.pairing[1];
                    let td = textureDimensions(pairing_1);
                    let d = pairing_delta(textureLoad(pairing_1, pairing_coord(pixel, td, t), 0).xy, t.zw);
                    neighbors[1] = load_spatial_neighbor(vec2i(pixel) + d, dims_i);
                }
                {
                    let t = input.pairing[2];
                    let td = textureDimensions(pairing_2);
                    let d = pairing_delta(textureLoad(pairing_2, pairing_coord(pixel, td, t), 0).xy, t.zw);
                    neighbors[2] = load_spatial_neighbor(vec2i(pixel) + d, dims_i);
                }
            }

            var merged: Reservoir;
            merged.dir = vec3f(0.0);
            merged.w_sum = 0.0;
            merged.M = 0.0;
            let c_c = r.M;
            // Canonical (temporal-chain) sample
            let p_c_yc = restir_target(s, lum_albedo, r.dir);
            if (p_c_yc > 0.0 && r.w_sum > 0.0) {
                let w_canonical = r.w_sum / (r.M * p_c_yc);
                var denom = c_c * p_c_yc;
                for (var l = 0u; l < 3u; l++) {
                    if (neighbors[l].valid) {
                        denom += neighbors[l].m * restir_target_at(neighbors[l].pos, neighbors[l].normal, neighbors[l].mat_index, r.dir);
                    }
                }
                if (denom > 0.0) {
                    let mis = c_c * p_c_yc / denom;
                    reservoir_update(&merged, r.dir, mis * p_c_yc * w_canonical, get_random_numbers(seed).x);
                }
            }
            // Neighbor samples
            for (var i = 0u; i < 3u; i++) {
                if (!neighbors[i].valid || neighbors[i].w <= 0.0) {
                    continue;
                }
                let y = neighbors[i].dir;
                let p_c_y = restir_target(s, lum_albedo, y);
                if (p_c_y <= 0.0) {
                    continue;
                }
                var denom = c_c * p_c_y;
                var p_own = 0.0;
                for (var l = 0u; l < 3u; l++) {
                    if (neighbors[l].valid) {
                        let p_l = restir_target_at(neighbors[l].pos, neighbors[l].normal, neighbors[l].mat_index, y);
                        denom += neighbors[l].m * p_l;
                        if (l == i) {
                            p_own = p_l;
                        }
                    }
                }
                if (denom > 0.0 && p_own > 0.0) {
                    let mis = neighbors[i].m * p_own / denom;
                    reservoir_update(&merged, y, mis * p_c_y * neighbors[i].w, get_random_numbers(seed).x);
                }
            }

            // Contribution weight (MIS weights partition unity: no 1/M)
            let f_shade = restir_target(s, lum_albedo, merged.dir);
            var big_w = 0.0;
            if (f_shade > 0.0) {
                big_w = merged.w_sum / f_shade;
            }

            // Shade the winner with one shadow ray
            if (big_w > 0.0) {
                let shade_ray = Ray(s.pos, merged.dir);
                let shade_light = intersect_lights(shade_ray);
                var shade_iterations = 0u;
                let occluder = intersect_scene(shade_ray, &shade_iterations);
                *iterations += shade_iterations;
                let lambert = dot(s.normal, merged.dir);
                let brdf = frostbite_brdf(s, merged.dir);
                if (shade_light.index >= 0) {
                    if (occluder.distance > shade_light.t) {
                        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                            radiance += (throughput[i] * (reflectance[i] * brdf.x + brdf.y) * lambert * big_w * input.emission_integral) * rgb_and_phases[i].rgb;
                        }
                    }
                } else if (occluder.object_index < 0) {
                    if (input.environment == ENVIRONMENT_STUDIO) {
                        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                            radiance += (throughput[i] * (reflectance[i] * brdf.x + brdf.y) * lambert * big_w * input.dome_integral) * rgb_and_phases[i].rgb;
                        }
                    } else {
                        var mean_weight = 0.0;
                        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                            mean_weight += throughput[i] * (reflectance[i] * brdf.x + brdf.y);
                        }
                        mean_weight *= 0.25;
                        radiance += (mean_weight * lambert * big_w) * environment_rgb(merged.dir);
                    }
                }
            }

            // Persist the TEMPORAL-only reservoir. No visibility feedback:
            // zeroing occluded winners preferentially keeps visible samples
            // whose weights were normalized against the UNSHADOWED target —
            // a selection effect that inflates energy wherever visibility
            // varies (measured +19% on the ground). Visibility applied at
            // shading keeps the estimator's energy correct; the low
            // confidence cap + spatial shading smoothing handle the
            // penumbra variance that feedback originally addressed.
            let f_store = restir_target(s, lum_albedo, r.dir);
            var w_store = 0.0;
            if (f_store > 0.0) {
                w_store = r.w_sum / (r.M * f_store);
            }
            textureStore(reservoir_out_a, pixel, vec4f(r.dir, w_store));
            textureStore(reservoir_out_b, pixel, vec4f(s.pos, r.M));
            textureStore(gbuffer_out, pixel, vec4f(s.normal, f32(obj.material_index)));
        }

        // Next event estimation: sample a direction towards a light
        // (vertex 1 direct lighting is handled by ReSTIR when enabled)
        var total_light_importance: f32;
        let light_dir = sample_lights(&total_light_importance, s.pos, s.normal, get_random_numbers(seed));
        let lambert_in_0 = dot(s.normal, light_dir);
        if (lambert_in_0 > 0.0 && !(use_restir && k == 1u)) {
            // Check visibility: the sampled ray must reach a light before any
            // scene surface
            let nee_ray = Ray(s.pos, light_dir);
            let nee_light = intersect_lights(nee_ray);
            if (nee_light.index >= 0) {
                var nee_iterations = 0u;
                let occluder = intersect_scene(nee_ray, &nee_iterations);
                *iterations += nee_iterations;
                if (occluder.distance > nee_light.t) {
                    // MIS with BRDF sampling
                    let light_density_0 = get_lights_density(total_light_importance, s.pos, light_dir, true);
                    let brdf_density_0 = get_frostbite_brdf_density(s, light_dir);
                    let spectrum_scale = lambert_in_0 / (light_density_0 + brdf_density_0);
                    let brdf = frostbite_brdf(s, light_dir);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        radiance += (throughput[i] * (reflectance[i] * brdf.x + brdf.y) * input.emission_integral * spectrum_scale) * rgb_and_phases[i].rgb;
                    }
                }
            }
        }

        // Sun NEE (sky): sample the solar cone, MIS-weighted against BRDF
        // sampling. Samples blocked by lamps or scene geometry contribute
        // nothing, keeping lamp MIS independent.
        if (input.environment == ENVIRONMENT_SKY && !(use_restir && k == 1u)) {
            let sun_dir = sample_sun(get_random_numbers(seed));
            let lambert_sun = dot(s.normal, sun_dir);
            if (lambert_sun > 0.0) {
                let sun_ray = Ray(s.pos, sun_dir);
                let sun_light = intersect_lights(sun_ray);
                var sun_iterations = 0u;
                let sun_occluder = intersect_scene(sun_ray, &sun_iterations);
                *iterations += sun_iterations;
                if (sun_light.index < 0 && sun_occluder.object_index < 0) {
                    let brdf_density_sun = get_frostbite_brdf_density(s, sun_dir);
                    let brdf_sun = frostbite_brdf(s, sun_dir);
                    let sun_scale = lambert_sun / (get_sun_density(sun_dir) + brdf_density_sun);
                    var mean_weight = 0.0;
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        mean_weight += throughput[i] * (reflectance[i] * brdf_sun.x + brdf_sun.y);
                    }
                    mean_weight *= 1.0 / f32(WAVELENGTH_SAMPLE_COUNT);
                    radiance += (mean_weight * sun_scale) * skyRadianceRGB(sun_dir, true);
                }
            }
        }

        // Environment NEE (HDRI): importance-sample a bright env direction,
        // MIS-weighted against BRDF sampling (balance heuristic). Lamp NEE
        // never produces env contributions and env samples blocked by lamps
        // are discarded, so lamp and env MIS stay independent.
        if (input.environment == ENVIRONMENT_HDRI && !(use_restir && k == 1u)) {
            var env_pdf: f32;
            let env_dir = sample_environment(get_random_numbers(seed), &env_pdf);
            let lambert_env = dot(s.normal, env_dir);
            if (lambert_env > 0.0 && env_pdf > 0.0) {
                let env_ray = Ray(s.pos, env_dir);
                let env_light = intersect_lights(env_ray);
                var env_iterations = 0u;
                let env_occluder = intersect_scene(env_ray, &env_iterations);
                *iterations += env_iterations;
                if (env_light.index < 0 && env_occluder.object_index < 0) {
                    let brdf_density_env = get_frostbite_brdf_density(s, env_dir);
                    let brdf_env = frostbite_brdf(s, env_dir);
                    let env_scale = lambert_env / (env_pdf + brdf_density_env);
                    // RGB environment: apply with the mean spectral weight
                    // (same approximation as the miss path)
                    var mean_weight = 0.0;
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        mean_weight += throughput[i] * (reflectance[i] * brdf_env.x + brdf_env.y);
                    }
                    mean_weight *= 1.0 / f32(WAVELENGTH_SAMPLE_COUNT);
                    radiance += (mean_weight * env_scale) * environment_rgb(env_dir);
                }
            }
        }

        // Sample the BRDF for MIS and to continue the path
        ray = Ray(s.pos, sample_frostbite_brdf(s, get_random_numbers(seed)));
        let lambert_in_1 = dot(s.normal, ray.direction);
        if (lambert_in_1 <= 0.0) {
            break;
        }
        let light_density_1 = get_lights_density(total_light_importance, s.pos, ray.direction, false);
        let brdf_density_1 = get_frostbite_brdf_density(s, ray.direction);
        let brdf_lambert_1 = frostbite_brdf(s, ray.direction) * lambert_in_1;
        let mis_factor = 1.0 / (light_density_1 + brdf_density_1);
        let rcp_brdf_density_1 = 1.0 / brdf_density_1;
        env_weight = 1.0;
        if (input.environment == ENVIRONMENT_HDRI) {
            env_weight = brdf_density_1 / (brdf_density_1 + get_environment_density(ray.direction));
        } else if (input.environment == ENVIRONMENT_SKY) {
            env_weight = brdf_density_1 / (brdf_density_1 + get_sun_density(ray.direction));
        }
        // ReSTIR owns everything the primary hit sees directly
        direct_scale = select(1.0, 0.0, use_restir && k == 1u);
        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
            let brdf_lambert_1_spectral = reflectance[i] * brdf_lambert_1.x + brdf_lambert_1.y;
            nee_throughput[i] = throughput[i] * brdf_lambert_1_spectral * mis_factor;
            throughput[i] *= brdf_lambert_1_spectral * rcp_brdf_density_1;
        }

        // Russian roulette after the second bounce: one-sample-MIS weights
        // can stack multiplicatively across bounces (rare 1e3+ samples that
        // read as permanent fireflies); roulette keeps throughput bounded
        // and is also how ReSTIR PT Enhanced treats initial samples.
        if (k >= 2u) {
            let max_throughput = max(max(throughput[0], throughput[1]), max(throughput[2], throughput[3]));
            let survival = clamp(max_throughput, 0.05, 1.0);
            if (get_random_numbers(seed).x >= survival) {
                break;
            }
            let rcp_survival = 1.0 / survival;
            for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                throughput[i] *= rcp_survival;
                nee_throughput[i] *= rcp_survival;
            }
        }
    }
    return radiance * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT));
}

// toneMapping implements ACES
fn toneMapping(color: vec3f) -> vec3f {
    let exposed = color * input.exposure;
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e);
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(output_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    var seed = vec2u(global_id.xy) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    // Sub-pixel jitter for anti-aliasing via accumulation
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5 + jitter, vec2f(dims));

    var iterations = 0u;
    var radiance = path_trace(ray, global_id.xy, &seed, &iterations);
    // Reject non-finite samples — a single NaN/inf poisons the running sum
    // for the rest of the accumulation (permanent bright pixels) — and cap
    // the heavy tail. 100 keeps caustic-path fireflies (lamp -> glossy model
    // -> ground) below saturation within ~100 accumulated samples; direct
    // light is NEE/ReSTIR-sampled and stays well under this, so the clipped
    // energy is glossy-caustic tail only (biased, revisit for Reference).
    if (!(radiance.x + radiance.y + radiance.z < 1e20)) {
        radiance = vec3f(0.0);
    }
    radiance = clamp(radiance, vec3f(0.0), vec3f(100.0));

    // Progressive accumulation: running sum, presented as sum / count.
    // frame_index == 0 restarts (no clear pass needed). The accumulation is
    // always the pure path trace — the ground truth being converged toward.
    let pixel_index = global_id.y * dims.x + global_id.x;
    var sum = vec4f(radiance, 1.0);
    if (input.frame_index > 0u) {
        let prev = accumulation[pixel_index];
        if (input.debug_iterations == 1u) {
            // Diagnostic: track the heaviest single sample per pixel
            sum = vec4f(max(radiance, prev.rgb), prev.a + 1.0);
        } else {
            sum += prev;
        }
    }
    accumulation[pixel_index] = sum;

    var color = sum.rgb / sum.a;
    color = toneMapping(color);
    color = pow(color, vec3f(1.0 / 2.2));  // Gamma correction

    if input.debug_iterations == 1u {
        // Diagnostic: red = per-pixel max sample luminance so far (log
        // scale), green flags > 1e4, blue flags > 1e3
        let l = dot(sum.rgb, vec3f(0.2126, 0.7152, 0.0722));
        color = vec3f(
            clamp(log2(1.0 + l) / 24.0, 0.0, 1.0),
            select(0.0, 1.0, l > 1e4),
            select(0.0, 1.0, l > 1e3),
        );
    }
    textureStore(output_texture, global_id.xy, vec4f(color, 1.0));
}

// ============================================================================
// Sky Model
// ============================================================================
const CHANNEL_R = 0u;
const CHANNEL_G = 1u;
const CHANNEL_B = 2u;
const SOLAR_RADIUS_RADIANS = 0.004450589; // 0.255 degrees

struct SkyState {
    sunDirection: vec3<f32>,
    params: array<f32, 27>,
    skyRadiances: array<f32, 3>,
    solarRadiances: array<f32, 3>,
}

fn radiance(theta: f32, gamma: f32, channel: u32, includeSun: bool) -> f32 {
    let r = skyState.skyRadiances[channel];
    let idx = 9u * channel;
    let p0 = skyState.params[idx + 0u];
    let p1 = skyState.params[idx + 1u];
    let p2 = skyState.params[idx + 2u];
    let p3 = skyState.params[idx + 3u];
    let p4 = skyState.params[idx + 4u];
    let p5 = skyState.params[idx + 5u];
    let p6 = skyState.params[idx + 6u];
    let p7 = skyState.params[idx + 7u];
    let p8 = skyState.params[idx + 8u];

    let cosGamma = cos(gamma);
    let cosGamma2 = cosGamma * cosGamma;
    let cosTheta = abs(cos(theta));

    let expM = exp(p4 * gamma);
    let rayM = cosGamma2;
    let mieMLhs = 1.0 + cosGamma2;
    let mieMRhs = pow(1.0 + p8 * p8 - 2.0 * p8 * cosGamma, 1.5f);
    let mieM = mieMLhs / mieMRhs;
    let zenith = sqrt(cosTheta);
    let radianceLhs = 1.0 + p0 * exp(p1 / (cosTheta + 0.01));
    let radianceRhs = p2 + p3 * expM + p5 * rayM + p6 * mieM + p7 * zenith;
    let radianceDist = radianceLhs * radianceRhs;

    let solarDiskRadius = gamma / SOLAR_RADIUS_RADIANS;
    let solarRadiance = select(0.0, skyState.solarRadiances[channel], includeSun && solarDiskRadius <= 1.0);

    return r * radianceDist + solarRadiance;
}

fn skyRadianceRGB(direction: vec3f, includeSun: bool) -> vec3f {
    let v = normalize(direction);
    let s = skyState.sunDirection;
    let theta = acos(clamp(v.y, -1.0, 1.0));
    let gamma = acos(clamp(dot(v, s), -1.0, 1.0));
    return vec3f(
        radiance(theta, gamma, CHANNEL_R, includeSun),
        radiance(theta, gamma, CHANNEL_G, includeSun),
        radiance(theta, gamma, CHANNEL_B, includeSun)
    );
}
