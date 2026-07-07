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
    // Debug bitmask for reuse-bias bisection (0 in normal operation):
    // bit 0: spatial skips d==2 candidates | bit 1: skips d==3 NEE |
    // bit 2: skips d==3 escape | bit 3: skips d>=4
    restir_debug: u32,
    _pad1: u32,
    _pad2: u32,
    // Previous frame's view matrix (world -> camera): temporal reprojection
    // for the denoiser, combined with per-object motion matrices
    prev_view: mat4x4f,
    // Previous frame's camera position: source view directions when
    // evaluating the reverse shift for temporal MIS
    prev_camera_pos: vec3f,
    // Legacy global hero-wavelength offset. UNUSED: lambdas are per-pixel
    // again (a frame-global set = coherent per-frame color cast, visible as
    // hue swings whenever motion resets accumulation); the wavelength shift
    // rebases reused samples between per-pixel bases instead.
    wavelength_u: f32,
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
// sRGB -> Fourier sRGB LUT (33^3, sRGB-encoded u8) as a 3D texture: lets
// the shader upsample arbitrary RGB radiance (environment light) into a
// reflectance-style spectrum so env chroma multiplies paint chroma
// spectrally instead of washing out through the RGB approximation
@group(0) @binding(10) var fourier_lut_3d: texture_3d<f32>;
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
// Primary-hit G-buffer (normal.xyz, w = material + depth/16384; w<0 = miss)
@group(2) @binding(3) var gbuffer_out: texture_storage_2d<rgba32float, write>;
// Raw per-frame radiance (pre-tonemap) for the denoiser
@group(2) @binding(5) var illum_out: texture_storage_2d<rgba32float, write>;
// Exact primary depth for the denoiser's world-space reconstruction
@group(2) @binding(8) var gdepth_out: texture_storage_2d<r32float, write>;
// Separated primary specular (accumulated), added back after the BRDF remod
@group(2) @binding(2) var spec_out: texture_storage_2d<rgba32float, write>;
// G-buffer + illum accumulation (stride 3 vec4/pixel: [normalSum.xyz,
// hitCount] + [depthSum, material, _, _] + [illumSum.rgb, sampleCount]).
// Accumulated over jittered frames like the radiance sum (reset on
// frame_index 0) so the G-buffer normal/depth AND the denoiser's illum input
// converge to STABLE, anti-aliased values instead of flickering in step with
// the sub-pixel jitter at silhouettes (the denoiser only sees the stable
// average, so its edges stop shimmering).
@group(2) @binding(4) var<storage, read_write> gbuffer_accum: array<vec4f>;

// CMF-project a per-wavelength contribution vector to RGB
fn project_spectral(f: vec4f, rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>) -> vec3f {
    var c = vec3f(0.0);
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        c += f[i] * rgb_and_phases[i].rgb;
    }
    return c * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT));
}

const MAX_DIST: f32 = 1e7;
const PI = 3.14159265359;

struct Intersection {
    distance: f32,
    object_index: i32,
    iterations: u32,
    normal: vec3f,
    // Dense surface-voxel index at the hit (VDB objects; ~0u = none) for
    // the paint and lighting caches
    surface_index: u32,
}

fn no_intersection() -> Intersection {
    return Intersection(MAX_DIST, -1, 0, vec3f(0), 0xffffffffu);
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
    hit_surface: ptr<function, u32>,
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

    var hit_voxel = vec3i(0);
    let hit = picovdbHDDAZeroCrossing(
        &accessor, grid, ray.origin, tmin, ray.direction, tmax, input.pixel_radius, hit_distance, hit_normal, hit_iterations, &hit_voxel,
    );
    if (hit && picovdbIsSurface(&accessor, grid, hit_voxel)) {
        // The accessor's leaf cache is warm from the traversal at the hit
        *hit_surface = picovdbGetSurfaceIndex(&accessor, grid, hit_voxel);
    }
    return hit;
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
        var hit_surface = 0xffffffffu;
        switch obj.object_type {
            case OBJECT_TYPE_VDB: {
                // Skip fog grids during surface intersection — they use volumetric marching instead
                let vdb_grid = picovdb_grids[obj.type_index];
                if (vdb_grid.gridType != GRID_TYPE_FOG_FLOAT) {
                    hit = intersect_picovdb(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations, &hit_surface);
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
        min_hit.surface_index = hit_surface;
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

// Spectral upsampling of RGB radiance via the 33^3 Fourier-sRGB LUT:
// returns the per-wavelength radiance a(phase_i) * maxcomp for the frame's
// 4 sampled wavelengths. The LUT is indexed by sRGB-ENCODED chroma and
// stores sRGB-encoded Fourier sRGB (same convention as material textures).
fn srgb_encode3(v: vec3f) -> vec3f {
    let lo = v * 12.92;
    let hi = 1.055 * pow(max(v, vec3f(1e-6)), vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, v <= vec3f(0.0031308));
}

fn srgb_decode3(v: vec3f) -> vec3f {
    let lo = v / 12.92;
    let hi = pow((v + 0.055) / 1.055, vec3f(2.4));
    return select(hi, lo, v <= vec3f(0.04045));
}

fn spectral_env_weights(rgb: vec3f, rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>) -> vec4f {
    let m = max(max(rgb.r, rgb.g), max(rgb.b, 1e-6));
    let enc = srgb_encode3(clamp(rgb / m, vec3f(0.0), vec3f(1.0)));
    // 33^3 grid: map [0,1] onto texel centers
    let uvw = enc * (32.0 / 33.0) + (0.5 / 33.0);
    let fsrgb = srgb_decode3(textureSampleLevel(fourier_lut_3d, illuminant_sampler, uvw, 0.0).rgb);
    let lagranges = prep_reflectance_real_lagrange_biased_3(fourier_srgb_to_fourier(fsrgb));
    return m * vec4f(
        eval_reflectance_real_lagrange_3(rgb_and_phases[0].w, lagranges),
        eval_reflectance_real_lagrange_3(rgb_and_phases[1].w, lagranges),
        eval_reflectance_real_lagrange_3(rgb_and_phases[2].w, lagranges),
        eval_reflectance_real_lagrange_3(rgb_and_phases[3].w, lagranges),
    );
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
fn clear_gbuffer(pixel: vec2u) {
    // w < 0 marks a primary miss; hits store material + 4 * depth
    textureStore(gbuffer_out, pixel, vec4f(0.0, 0.0, 0.0, -1.0));
}

fn path_trace(ray_in: Ray, pixel: vec2u, seed: ptr<function, vec2u>, iterations: ptr<function, u32>, primary_albedo: ptr<function, vec3f>, primary_normal: ptr<function, vec3f>, primary_depth: ptr<function, f32>, primary_mat: ptr<function, u32>, primary_hit: ptr<function, bool>, spec_out: ptr<function, vec3f>) -> vec3f {
    // Primary (k==1) specular is split off: it is fresnel-white, NOT albedo-
    // colored, so it must NOT ride in radiance/albedo (the demod would divide
    // it by the tiny low-albedo channels and amplify noise). Denoise it and
    // the diffuse irradiance separately; recombine after the BRDF (ReBLUR).
    var spec = vec3f(0.0);
    *primary_albedo = vec3f(1.0); // env/miss default (no demod)
    *primary_normal = vec3f(0.0, 0.0, 1.0);
    *primary_depth = 0.0;
    *primary_mat = 0u;
    *primary_hit = false;
    *spec_out = vec3f(0.0);
    var ray = ray_in;
    // MIS weight applied when a BRDF-sampled ray reaches the environment
    // (competes with sample_environment() NEE from the previous vertex)
    var env_weight = 1.0;

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
                radiance += (nee_throughput[i] * input.emission_integral) * rgb_and_phases[i].rgb;
            }
            break;
        }
        // Miss: environment. The dome emits the illuminant spectrum; sky and
        // HDRI are RGB-only and approximated with the mean throughput.
        if (hit.object_index < 0) {
            // Primary-ray miss with the white backdrop: display-only white
            // (scaled so it tonemaps to white); the path ends, so the
            // backdrop does not light the scene.
            if (k == 1u && input.white_background == 1u) {
                radiance = vec3f(6.0 / input.exposure);
                break;
            }
            if (input.environment == ENVIRONMENT_STUDIO) {
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    radiance += (throughput[i] * input.dome_integral) * rgb_and_phases[i].rgb;
                }
            } else if (input.environment == ENVIRONMENT_SKY) {
                // Sky radiance inside the sun cone (disk + circumsolar halo)
                // is MIS-weighted against the previous vertex's sun NEE;
                // the smooth sky outside the cone comes in at full weight.
                var env = skyRadianceRGB(ray.direction, true);
                if (dot(ray.direction, skyState.sunDirection) >= SUN_CONE_COS) {
                    env *= env_weight;
                }
                let env_w = spectral_env_weights(env, rgb_and_phases);
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    radiance += (throughput[i] * env_w[i]) * rgb_and_phases[i].rgb;
                }
            } else {
                // Spectrally upsampled through the Fourier LUT so env chroma
                // multiplies surface chroma per wavelength
                let env_w = spectral_env_weights(environment_rgb(ray.direction), rgb_and_phases);
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    radiance += (throughput[i] * env_weight * env_w[i]) * rgb_and_phases[i].rgb;
                }
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
            break;
        }

        // Primary-hit G-buffer (normal, material + packed depth) + exact
        // depth, matching initialMain so the denoiser can run on Reference
        // PT too (world_pos reconstruction needs the packed/exact depth)
        if (k == 1u) {
            *primary_albedo = project_spectral(vec4f(reflectance[0], reflectance[1], reflectance[2], reflectance[3]), rgb_and_phases);
            *primary_normal = s.normal;
            *primary_depth = hit.distance;
            *primary_mat = obj.material_index;
            *primary_hit = true;
        }

        // Next event estimation: sample a direction towards a light
        var total_light_importance: f32;
        let light_dir = sample_lights(&total_light_importance, s.pos, s.normal, get_random_numbers(seed));
        let lambert_in_0 = dot(s.normal, light_dir);
        if (lambert_in_0 > 0.0) {
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
                        let base = throughput[i] * input.emission_integral * spectrum_scale * rgb_and_phases[i].rgb;
                        radiance += (reflectance[i] * brdf.x) * base;
                        let sc = brdf.y * base;
                        if (k == 1u) { spec += sc; } else { radiance += sc; }
                    }
                }
            }
        }

        // Sun NEE (sky): sample the solar cone, MIS-weighted against BRDF
        // sampling. Samples blocked by lamps or scene geometry contribute
        // nothing, keeping lamp MIS independent.
        if (input.environment == ENVIRONMENT_SKY) {
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
                    let sun_w = spectral_env_weights(skyRadianceRGB(sun_dir, true), rgb_and_phases);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        let base = throughput[i] * sun_scale * sun_w[i] * rgb_and_phases[i].rgb;
                        radiance += (reflectance[i] * brdf_sun.x) * base;
                        let sc = brdf_sun.y * base;
                        if (k == 1u) { spec += sc; } else { radiance += sc; }
                    }
                }
            }
        }

        // Environment NEE (HDRI): importance-sample a bright env direction,
        // MIS-weighted against BRDF sampling (balance heuristic). Lamp NEE
        // never produces env contributions and env samples blocked by lamps
        // are discarded, so lamp and env MIS stay independent.
        if (input.environment == ENVIRONMENT_HDRI) {
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
                    // Spectrally upsampled RGB environment (Fourier LUT)
                    let nee_env_w = spectral_env_weights(environment_rgb(env_dir), rgb_and_phases);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        let base = throughput[i] * env_scale * nee_env_w[i] * rgb_and_phases[i].rgb;
                        radiance += (reflectance[i] * brdf_env.x) * base;
                        let sc = brdf_env.y * base;
                        if (k == 1u) { spec += sc; } else { radiance += sc; }
                    }
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
    *spec_out = spec * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT));
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

// Sanitize, clamp, accumulate, tonemap, present — shared by the Reference
// megakernel and the ReSTIR shading pass so both feed the same accumulator.
fn write_output(pixel: vec2u, dims: vec2u, radiance_in: vec3f) {
    var radiance = radiance_in;
    // Reject non-finite samples — a single NaN/inf poisons the running sum
    // for the rest of the accumulation (permanent bright pixels) — and cap
    // the heavy tail. 100 keeps caustic-path fireflies below saturation
    // within ~100 accumulated samples; direct light is NEE-sampled and
    // stays well under this, so the clipped energy is the glossy-caustic
    // tail only (biased, revisit for Reference).
    if (!(radiance.x + radiance.y + radiance.z < 1e20)) {
        radiance = vec3f(0.0);
    }
    radiance = clamp(radiance, vec3f(0.0), vec3f(100.0));

    // Progressive accumulation: running sum, presented as sum / count.
    // frame_index == 0 restarts (no clear pass needed).
    let pixel_index = pixel.y * dims.x + pixel.x;
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
    textureStore(output_texture, pixel, vec4f(color, 1.0));
}

// Halton radical-inverse (low-discrepancy) for stable sub-pixel jitter:
// a white-noise per-frame offset makes silhouette coverage a random binary
// each frame -> high-frequency edge flicker; a Halton sequence distributes
// the offsets evenly so the temporal average converges smoothly with far
// less edge variance (the standard TAA jitter).
fn halton(index: u32, base: u32) -> f32 {
    var f = 1.0;
    var r = 0.0;
    var i = index;
    for (var k = 0u; k < 16u; k++) {
        if (i == 0u) { break; }
        f = f / f32(base);
        r = r + f * f32(i % base);
        i = i / base;
    }
    return r;
}

// Reference: the frozen single-kernel path tracer (ground truth)
@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(output_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    var seed = vec2u(global_id.xy) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    // Low-discrepancy Halton(2,3) sub-pixel jitter (cycled over 64 frames)
    let h = (input.rng_frame & 63u) + 1u;
    let jitter = vec2f(halton(h, 2u), halton(h, 3u)) - 0.5;
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5 + jitter, vec2f(dims));

    var iterations = 0u;
    var pa = vec3f(1.0); var pn = vec3f(0.0, 0.0, 1.0); var pd = 0.0; var pm = 0u; var phit = false;
    var pspec = vec3f(0.0);
    let radiance = path_trace(ray, global_id.xy, &seed, &iterations, &pa, &pn, &pd, &pm, &phit, &pspec);

    // Accumulate the G-buffer + demodulated diffuse irradiance + separated
    // specular over jittered frames -> stable, anti-aliased values (reset when
    // frame_index == 0, same as the radiance sum). Stride 4 vec4/pixel.
    let gi = 4u * (global_id.y * dims.x + global_id.x);
    var nsum = vec4f(0.0); var dsum = vec4f(0.0); var isum = vec4f(0.0); var ssum = vec4f(0.0);
    if (input.frame_index > 0u) {
        nsum = gbuffer_accum[gi]; dsum = gbuffer_accum[gi + 1u]; isum = gbuffer_accum[gi + 2u]; ssum = gbuffer_accum[gi + 3u];
    }
    // Accumulate diffuse irradiance + specular ONLY on frames that HIT the
    // surface. A silhouette pixel is sticky-object but the jittered ray
    // misses it ~half the frames; folding those backdrop frames into the
    // object's illum is what made the edge flip object/backdrop. Gating by
    // hit gives each object pixel the stable average of its OWN irradiance
    // (radiance is DIFFUSE here, so the demod by albedo stays clean).
    if (phit) {
        nsum += vec4f(pn, 1.0); dsum.x += pd; dsum.y = f32(pm);
        isum += vec4f(radiance / max(pa, vec3f(0.02)), 1.0);
        ssum += vec4f(pspec, 1.0);
    }
    gbuffer_accum[gi] = nsum;
    gbuffer_accum[gi + 1u] = dsum;
    gbuffer_accum[gi + 2u] = isum;
    gbuffer_accum[gi + 3u] = ssum;
    if (nsum.w > 0.5) {
        let navg = normalize(nsum.xyz);
        let davg = dsum.x / nsum.w;
        let matf = f32(u32(dsum.y));
        textureStore(gbuffer_out, global_id.xy, vec4f(navg, matf + clamp(davg * (1.0 / 16384.0), 0.0, 0.9999)));
        textureStore(gdepth_out, global_id.xy, vec4f(davg, 0.0, 0.0, 0.0));
    } else {
        clear_gbuffer(global_id.xy);
    }
    // Object pixels feed the denoiser their accumulated (stable) diffuse
    // irradiance + specular; miss pixels feed the current backdrop/env
    // radiance (demod_albedo = 1 for a miss, so no remod), which the
    // denoiser's miss branch accumulates in place.
    if (nsum.w > 0.5) {
        textureStore(illum_out, global_id.xy, vec4f(isum.rgb / max(isum.w, 1.0), 1.0));
        textureStore(spec_out, global_id.xy, vec4f(ssum.rgb / max(ssum.w, 1.0), 1.0));
    } else {
        textureStore(illum_out, global_id.xy, vec4f(radiance, 1.0));
        textureStore(spec_out, global_id.xy, vec4f(0.0, 0.0, 0.0, 1.0));
    }
    write_output(global_id.xy, dims, radiance + pspec);
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
