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
// Primary-hit G-buffer (normal.xyz, w = material + 4*depth; w < 0 marks a
// miss), ping-ponged across frames for temporal passes
@group(2) @binding(2) var gbuffer_prev: texture_2d<f32>;
@group(2) @binding(3) var gbuffer_out: texture_storage_2d<rgba32float, write>;
// Raw (pre-accumulation, pre-tonemap) per-frame radiance for the denoiser
@group(2) @binding(5) var illum_out: texture_storage_2d<rgba32float, write>;
// Reuse-pass variant bindings (temporal/spatial only): the CURRENT frame's
// G-buffer + exact primary depth as sampled inputs — reconstructing the
// primary hit replaces ~5 HDDA traversals/pixel of deterministic re-traces
// (the gbuffer w-packing's quantized depth is too coarse: its ~4e-3 step
// exceeds RAY_OFFSET, so the exact distance rides in its own r32float —
// rg32float storage is not supported in WebGPU compatibility mode)
@group(2) @binding(6) var gbuffer_cur: texture_2d<f32>;
@group(2) @binding(7) var gdepth_cur: texture_2d<f32>;
// Written by initial sampling alongside the G-buffer
@group(2) @binding(8) var gdepth_out: texture_storage_2d<r32float, write>;
// Shade-pass upsampling inputs: the LOW-RES (GI) G-buffer + depth, used to
// reconstruct the reservoir cell's primary hit (the reconnection-shift
// source) when a full-res pixel resamples the low-res reservoir grid
@group(2) @binding(9) var gi_gbuffer: texture_2d<f32>;
@group(2) @binding(10) var gi_gdepth: texture_2d<f32>;

// ============================================================================
// ReSTIR PT reservoirs (docs/restir-pt-plan.md, layout after ReSTIR PT
// Enhanced supplemental Alg. 1, adapted to VDB primitives). One storage
// buffer holds both ping-pong halves; the half written this frame is
// selected by rng_frame parity.
// ============================================================================

struct Reservoir {
    // Selected path contribution, PER WAVELENGTH (the frame's 4 hero
    // lambdas — global per frame, so all reservoirs share the set). CMF
    // projection happens only at shading; reuse recomposes per-lambda,
    // eliminating the RGB-round-trip bias class.
    f: vec4f,
    // Incident radiance at the rc vertex along rc_wi, per wavelength,
    // excluding the rc vertex's own BSDF (re-evaluated at reuse)
    rc_radiance: vec4f,
    // Reconnection vertex: object-index-space position, or the env
    // direction when the path reconnects at the environment
    rc_pos: vec3f,
    // Scalar unbiased contribution weight (per-DoF UCWs are a later step)
    w: f32,
    // Confidence weight (sample count analogue, capped in temporal reuse)
    m: f32,
    rc_normal_oct: u32,
    // Suffix direction at the rc vertex (toward x_{k+1} / the light),
    // FULL f32 precision, x component (y/z below). Oct16 quantization here
    // was an energy poison: the env pdf is texel-discontinuous, so a
    // quantized direction re-derives lpdf on the wrong texel while the
    // stored radiance came from the exact direction — a one-sided
    // stored-bright/pdf-dim mismatch that inflated shifts by up to
    // (lpdf+pdf)/pdf at softbox edges (measured +10-35% per spatial merge
    // in shadows, amplified 4-6x by the temporal chain)
    rc_wi_x: f32,
    init_seed: u32,
    // bits 0-3: d | bit 4: rc-is-env | bit 5: NEE | 6-7: rc material |
    // 8-9: rc object
    path_flags: u32,
    // Stratified wavelength offset the sample was BORN with: the lambda
    // set is part of the path sample (a PSS dimension), so reuse carries
    // it and shading projects with it — the wavelength shift is the
    // identity (J = 1), exact for any spectrum
    wavelength_u: f32,
    rc_wi_y: f32,
    rc_wi_z: f32,
}

const PATH_FLAG_RC_ENV = 16u;
const PATH_FLAG_NEE = 32u;
// The d == 3 suffix vertex is on the environment (its radiance is a pure
// function of rc_wi, so the wavelength rebase can recompute it exactly)
const PATH_FLAG_SUFFIX_ENV = 1024u;

@group(2) @binding(4) var<storage, read_write> reservoirs: array<Reservoir>;

// Regions: 0/1 = frame-final ping-pong (parity = rng_frame & 1),
// 2 = this frame's post-temporal scratch that spatial reuse reads from
fn reservoir_index(pixel: vec2u, dims: vec2u, region: u32) -> u32 {
    return region * dims.x * dims.y + pixel.y * dims.x + pixel.x;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// CMF-project a per-wavelength contribution vector to RGB
fn project_spectral(f: vec4f, rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>) -> vec3f {
    var c = vec3f(0.0);
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        c += f[i] * rgb_and_phases[i].rgb;
    }
    return c * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT));
}

// Hero-wavelength LUT entries (rgb + phase per lambda) for a stratified
// offset — the frame's global set or a reservoir sample's stored set
fn wavelengths_at(u0: f32) -> array<vec4f, WAVELENGTH_SAMPLE_COUNT> {
    var rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>;
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        let u = u0 * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT)) + f32(i) / f32(WAVELENGTH_SAMPLE_COUNT);
        rgb_and_phases[i] = textureSampleLevel(illuminant_spectrum, illuminant_sampler, vec2f(u, 0.5), 0.0);
    }
    return rgb_and_phases;
}

fn oct_wrap(v: vec2f) -> vec2f {
    return (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

fn oct_encode(n: vec3f) -> u32 {
    var p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-8);
    if (n.z < 0.0) {
        p = oct_wrap(p);
    }
    let e = vec2u(clamp(p * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)) * 65535.0);
    return e.x | (e.y << 16u);
}

fn oct_decode(bits: u32) -> vec3f {
    let e = vec2f(vec2u(bits & 0xffffu, bits >> 16u)) * (2.0 / 65535.0) - 1.0;
    var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n = vec3f(oct_wrap(n.xy), n.z);
    }
    return normalize(n);
}

const TEMPORAL_CONFIDENCE_CAP = 20.0;
const JACOBIAN_REJECT_THRESHOLD = 0.5;

fn mis_balance(a: f32, b: f32) -> f32 {
    return a / max(a + b, 1e-12);
}

// Light-technique pdf for an env direction (recomputable: env is static)
fn env_light_pdf(dir: vec3f) -> f32 {
    if (input.environment == ENVIRONMENT_HDRI) {
        return get_environment_density(dir);
    }
    if (input.environment == ENVIRONMENT_SKY) {
        return get_sun_density(dir);
    }
    return 0.0; // dome has no light-sampling technique
}

// MIS weight for a BSDF-sampled ray reaching the environment (matches the
// initial/reference gating: sky only competes inside the sun cone)
fn env_escape_mis(dir: vec3f, brdf_pdf: f32) -> f32 {
    if (input.environment == ENVIRONMENT_HDRI) {
        return mis_balance(brdf_pdf, get_environment_density(dir));
    }
    if (input.environment == ENVIRONMENT_SKY && dot(dir, skyState.sunDirection) >= SUN_CONE_COS) {
        return mis_balance(brdf_pdf, get_sun_density(dir));
    }
    return 1.0;
}

fn make_shading_data(pos: vec3f, normal: vec3f, view_dir: vec3f, mat: Material) -> ShadingData {
    var s: ShadingData;
    s.pos = pos;
    s.normal = normal;
    s.out_dir = view_dir;
    s.lambert_out = dot(normal, view_dir);
    s.base_color = mat.base_color;
    s.diffuse_albedo = mat.diffuse_albedo;
    s.fresnel_0 = mat.fresnel_0;
    s.roughness = mat.roughness;
    return s;
}

struct ShiftResult {
    f: vec4f,      // shifted integrand per wavelength (as-if-sampled at dst)
    jacobian: f32, // PSS Jacobian; 0 marks the shift undefined
}

// Raw per-lambda env radiance for a direction in an arbitrary wavelength
// basis (recomputable: the environment is static). Mirrors the initial
// sampler's env_l composition.
fn env_radiance_spectral(wi: vec3f, rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>) -> vec4f {
    if (input.environment == ENVIRONMENT_STUDIO) {
        return vec4f(input.dome_integral);
    }
    if (input.environment == ENVIRONMENT_SKY) {
        return spectral_env_weights(skyRadianceRGB(wi, true), rgb_and_phases);
    }
    return spectral_env_weights(environment_rgb(wi), rgb_and_phases);
}

// Wavelength shift for sampled (non-recomputable) suffix radiance: the 4
// hero lambdas are stratified bins; a change of the global offset u slides
// each lambda within its bin, so linear interpolation toward the adjacent
// bin (pdf ratio 1 — stratified-uniform) maps the stored 4-vector onto the
// destination basis. End bins clamp (tiny extrapolation bias, d >= 4 only).
fn rebase_suffix_radiance(rad: vec4f, u_src: f32, u_dst: f32) -> vec4f {
    let du = u_dst - u_src;
    if (du >= 0.0) {
        return vec4f(
            mix(rad.x, rad.y, du),
            mix(rad.y, rad.z, du),
            mix(rad.z, rad.w, du),
            rad.w);
    }
    return vec4f(
        rad.x,
        mix(rad.y, rad.x, -du),
        mix(rad.z, rad.y, -du),
        mix(rad.w, rad.z, -du));
}

// Reconnection shift of a reservoir path onto a destination primary hit
// (GRIS §7.4: reconnect at x2; direction copy for env suffixes — formulas
// follow ReSTIR_PT Shift.slang::computeShiftedIntegrandReconnection).
// Combined-lobe BSDF (lobe splitting arrives with the hybrid shift in P3).
// use_prev evaluates in the previous frame's domain (the reverse shift for
// temporal MIS): object-space rc data maps through the object's motion.
// Visibility always uses the current scene (accepted approximation).
// The RGB-copy spectral scheme: local BSDF weights are evaluated at the
// frame's wavelengths and CMF-projected, then multiply the stored suffix
// radiance componentwise.
fn shift_reconnect(
    r: Reservoir,
    dst_s: ShadingData,
    dst_reflectance: array<f32, WAVELENGTH_SAMPLE_COUNT>,
    rgb_and_phases: array<vec4f, WAVELENGTH_SAMPLE_COUNT>,
    dst_u: f32,
    src_s: ShadingData,
    use_prev: bool,
    min_cos: f32,
    iterations: ptr<function, u32>,
) -> ShiftResult {
    // RATIO composition: F_shift = F_src * (dst factors)/(src factors) for
    // exactly the factors the shift changes (BSDFs, cosines, pdfs, MIS).
    // Everything else (suffix radiance, RR compensation, wavelength set)
    // cancels by construction — recomposing F from parts measured a
    // systematic +1.2%/merge energy bias from estimator mismatch.
    var out: ShiftResult;
    out.f = vec4f(0.0);
    out.jacobian = 0.0;
    let d = r.path_flags & 15u;
    if (d < 2u || !(r.w > 0.0)) {
        return out;
    }
    let is_env_rc = (r.path_flags & PATH_FLAG_RC_ENV) != 0u;
    let is_nee = (r.path_flags & PATH_FLAG_NEE) != 0u;

    if (is_env_rc) {
        // d = 2: x2 IS the environment — direction copy (world direction)
        let wi = vec3f(r.rc_wi_x, r.rc_wi_y, r.rc_wi_z);
        let lambert_dst = dot(dst_s.normal, wi);
        let lambert_src = dot(src_s.normal, wi);
        if (lambert_dst <= 0.0 || lambert_src <= 0.0) {
            return out;
        }
        var vis_iter = 0u;
        let occ = intersect_scene(Ray(dst_s.pos, wi), &vis_iter);
        *iterations += vis_iter;
        if (occ.object_index >= 0 || intersect_lights(Ray(dst_s.pos, wi)).index >= 0) {
            return out;
        }
        let b_dst = frostbite_brdf(dst_s, wi);
        // Wavelength shift: the env vertex's radiance is a pure function of
        // the direction, so re-evaluate it EXACTLY in the destination basis
        // (the stored per-lambda vector belongs to the sample's birth basis)
        let rc_rad = env_radiance_spectral(wi, rgb_and_phases);
        var f_l = vec4f(0.0);
        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
            f_l[i] = (dst_reflectance[i] * b_dst.x + b_dst.y) * rc_rad[i];
        }
        f_l *= lambert_dst;
        let pdf1_dst = get_frostbite_brdf_density(dst_s, wi);
        let pdf1_src = get_frostbite_brdf_density(src_s, wi);
        if (pdf1_src <= 0.0 || pdf1_dst <= 0.0) {
            return out;
        }
        let lpdf = env_light_pdf(wi);
        if (is_nee) {
            if (lpdf <= 0.0) {
                return out;
            }
            out.f = (f_l / lpdf) * mis_balance(lpdf, pdf1_dst);
            out.jacobian = 1.0;
        } else {
            out.f = (f_l / pdf1_dst) * env_escape_mis(wi, pdf1_dst);
            out.jacobian = pdf1_dst / pdf1_src;
        }
        let s_env = dot(out.f, vec4f(1.0));
        if (!(out.jacobian > 0.0) || !(s_env < 1e20) || !(s_env >= 0.0)) {
            out.f = vec4f(0.0);
            out.jacobian = 0.0;
        }
        return out;
    }

    // Surface reconnection vertex, stored in its object's index space
    let rc_obj = objects[(r.path_flags >> 8u) & 3u];
    var rc_pos = (rc_obj.transform_inverse * vec4f(r.rc_pos, 1.0)).xyz;
    var rc_n = normalize((rc_obj.transform_inverse * vec4f(oct_decode(r.rc_normal_oct), 0.0)).xyz);
    var rc_wi = vec3f(r.rc_wi_x, r.rc_wi_y, r.rc_wi_z);
    if (d >= 4u) {
        rc_wi = normalize((rc_obj.transform_inverse * vec4f(rc_wi, 0.0)).xyz);
    }
    if (use_prev) {
        rc_pos = (rc_obj.motion * vec4f(rc_pos, 1.0)).xyz;
        rc_n = normalize((rc_obj.motion * vec4f(rc_n, 0.0)).xyz);
        if (d >= 4u) {
            rc_wi = normalize((rc_obj.motion * vec4f(rc_wi, 0.0)).xyz);
        }
    }

    let dst_diff = rc_pos - dst_s.pos;
    let dst_d2 = dot(dst_diff, dst_diff);
    let dir = dst_diff * inverseSqrt(max(dst_d2, 1e-12));
    let src_diff = rc_pos - src_s.pos;
    let src_d2 = dot(src_diff, src_diff);
    let src_dir = src_diff * inverseSqrt(max(src_d2, 1e-12));
    let lambert_dst = dot(dst_s.normal, dir);
    let lambert_src = dot(src_s.normal, src_dir);
    // Grazing guard (symmetric => invertible => unbiased): near-parallel
    // connections make the reconstruction ratio's denominator hinge on a
    // tiny cosine whose relative error explodes — a one-sided (Jensen)
    // energy inflation measured at silhouettes and the far ground. The
    // footprint criteria (P3) subsume this with a principled bound.
    if (lambert_dst <= min_cos || lambert_src <= min_cos || dst_d2 < 1e-6 || src_d2 < 1e-6) {
        return out;
    }

    // Geometric Jacobian (GRIS Eq. 52): cosines at the rc vertex
    let cos_dst = abs(dot(rc_n, dir));
    let cos_src = abs(dot(rc_n, src_dir));
    if (cos_src <= min_cos || cos_dst <= min_cos) {
        return out;
    }
    var jacobian = (cos_dst / dst_d2) * (src_d2 / cos_src);

    let pdf1_dst = get_frostbite_brdf_density(dst_s, dir);
    let pdf1_src = get_frostbite_brdf_density(src_s, src_dir);
    if (pdf1_dst <= 0.0 || pdf1_src <= 0.0) {
        return out;
    }
    jacobian *= pdf1_dst / pdf1_src;

    // rc shading data for both incomings
    let rc_mat = materials[(r.path_flags >> 6u) & 3u];
    // Back-facing reconnection is a shift FAILURE, not a normal flip: the
    // initial sampler only ever produces front-facing x2 connections, so a
    // flipped-normal evaluation manufactures energy the true integrand
    // doesn't have (measured as +1.2%/merge concentrated at grazing
    // geometry). Rejecting symmetrically keeps the shift invertible.
    let rc_s = make_shading_data(rc_pos, rc_n, -dir, rc_mat);
    if (rc_s.lambert_out <= 1e-4) {
        return out;
    }
    let rc_src_s = make_shading_data(rc_pos, rc_n, -src_dir, rc_mat);
    if (rc_src_s.lambert_out <= 1e-4) {
        return out;
    }
    let lambert2_dst = dot(rc_s.normal, rc_wi);
    let lambert2_src = dot(rc_src_s.normal, rc_wi);
    if (lambert2_dst <= 0.0 || lambert2_src <= 0.0) {
        return out;
    }
    var rc_refl: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    let rc_lagranges = prep_reflectance_real_lagrange_biased_3(fourier_srgb_to_fourier(rc_mat.base_color));
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        rc_refl[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, rc_lagranges);
    }
    let b1_dst = frostbite_brdf(dst_s, dir);
    let b2_dst = frostbite_brdf(rc_s, rc_wi);
    // Wavelength shift of the suffix radiance into the destination basis:
    // exact recompute when the suffix vertex is on the environment,
    // stratified-bin interpolation for sampled deep suffixes
    var rc_rad = r.rc_radiance;
    if (d == 3u && (r.path_flags & PATH_FLAG_SUFFIX_ENV) != 0u) {
        rc_rad = env_radiance_spectral(rc_wi, rgb_and_phases);
    } else {
        rc_rad = rebase_suffix_radiance(rc_rad, r.wavelength_u, dst_u);
    }
    var f_l = vec4f(0.0);
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        let w1d = dst_reflectance[i] * b1_dst.x + b1_dst.y;
        let w2d = rc_refl[i] * b2_dst.x + b2_dst.y;
        f_l[i] = w1d * w2d * rc_rad[i];
    }
    f_l *= lambert_dst * lambert2_dst / pdf1_dst;

    let pdf2_dst = get_frostbite_brdf_density(rc_s, rc_wi);
    let pdf2_src = get_frostbite_brdf_density(rc_src_s, rc_wi);
    if (is_nee && d == 3u) {
        // Suffix = NEE at rc: light pdf is pixel-independent
        let lpdf = env_light_pdf(rc_wi);
        if (lpdf <= 0.0) {
            return out;
        }
        f_l *= mis_balance(lpdf, pdf2_dst) / lpdf;
    } else {
        if (pdf2_dst <= 0.0 || pdf2_src <= 0.0) {
            return out;
        }
        f_l /= pdf2_dst;
        if (!is_nee && d == 3u) {
            f_l *= env_escape_mis(rc_wi, pdf2_dst);
        }
        jacobian *= pdf2_dst / pdf2_src;
    }

    // Visibility: destination primary hit <-> rc vertex (current scene)
    var vis_iter = 0u;
    let occ = intersect_scene(Ray(dst_s.pos, dir), &vis_iter);
    *iterations += vis_iter;
    let dist = sqrt(dst_d2);
    if (occ.distance < dist - max(0.01 * dist, 1e-3)) {
        return out;
    }
    let l_hit = intersect_lights(Ray(dst_s.pos, dir));
    if (l_hit.index >= 0 && l_hit.t < dist) {
        return out;
    }

    // Jacobian rejection (ReSTIR_PT rejectShiftBasedOnJacobian): extreme
    // reconnection Jacobians at voxel silhouettes carry a measurable energy
    // bias; the rejection is symmetric in J <-> 1/J so both directions of
    // the shift declare the same pairs undefined (unbiased)
    if (max(jacobian, 1.0 / max(jacobian, 1e-9)) > 1.0 + JACOBIAN_REJECT_THRESHOLD) {
        return out;
    }
    out.f = f_l;
    out.jacobian = jacobian;
    let fs = dot(out.f, vec4f(1.0));
    if (!(out.jacobian > 0.0 && out.jacobian < 1e12) || !(fs < 1e20) || !(fs >= 0.0)) {
        out.f = vec4f(0.0);
        out.jacobian = 0.0;
    }
    return out;
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

fn path_trace(ray_in: Ray, pixel: vec2u, seed: ptr<function, vec2u>, iterations: ptr<function, u32>) -> vec3f {
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
            if (k == 1u) {
                clear_gbuffer(pixel);
            }
            break;
        }
        // Miss: environment. The dome emits the illuminant spectrum; sky and
        // HDRI are RGB-only and approximated with the mean throughput.
        if (hit.object_index < 0) {
            if (k == 1u) {
                clear_gbuffer(pixel);
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
            if (k == 1u) {
                clear_gbuffer(pixel);
            }
            break;
        }

        // Primary-hit G-buffer (normal + material) for denoise passes
        if (k == 1u) {
            textureStore(gbuffer_out, pixel, vec4f(s.normal, f32(obj.material_index)));
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
                        radiance += (throughput[i] * (reflectance[i] * brdf.x + brdf.y) * input.emission_integral * spectrum_scale) * rgb_and_phases[i].rgb;
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
                        let thr = throughput[i] * (reflectance[i] * brdf_sun.x + brdf_sun.y);
                        radiance += (thr * sun_scale) * sun_w[i] * rgb_and_phases[i].rgb;
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
                        let thr = throughput[i] * (reflectance[i] * brdf_env.x + brdf_env.y);
                        radiance += (thr * env_scale) * nee_env_w[i] * rgb_and_phases[i].rgb;
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
    return radiance * (1.0 / f32(WAVELENGTH_SAMPLE_COUNT));
}

// ============================================================================
// ReSTIR PT initial sampling (plan P0): the same path tree as path_trace,
// but every contribution becomes a RIS candidate instead of being summed.
// One representative path per pixel survives, with unbiased contribution
// weight W = sum(luminance(F_i)) / luminance(F_Y); shading with F*W has the
// same expected value as the full sum (candidates cover disjoint strata of
// the path space, so MIS weights between candidates are 1).
//
// Reconnection-vertex bookkeeping targets the reconnection shift at x2
// (GRIS §7.4 baseline; footprint criteria arrive in P3): the stored suffix
// is the incident radiance at x2 (excluding x2's BSDF) plus its direction,
// or the environment direction itself for paths whose x2 is the env vertex.
// ============================================================================
fn initial_sample(ray_in: Ray, pixel: vec2u, dims: vec2u, seed: ptr<function, vec2u>, iterations: ptr<function, u32>) {
    var ray = ray_in;
    var env_weight = 1.0;

    var throughput: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    var nee_throughput: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    // PER-PIXEL hero-wavelength offset (like the Reference): a frame-global
    // set makes the whole frame share one 4-lambda projection, a coherent
    // per-frame color cast that accumulation hides while static but shows
    // at FULL strength during motion (sharp hue swings on rotation). The
    // wavelength shift rebases reused samples between per-pixel bases, so
    // cross-pixel reuse no longer needs a shared basis.
    let wavelength_u = get_random_numbers(seed).x;
    let rgb_and_phases = wavelengths_at(wavelength_u);
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        throughput[i] = 1.0;
        nee_throughput[i] = 1.0;
    }
    let spectral_norm = 1.0 / f32(WAVELENGTH_SAMPLE_COUNT);

    // Streaming RIS state (contributions tracked per wavelength)
    var w_sum = 0.0;
    var sel_f = vec4f(0.0);
    var sel_flags = 0u;
    var sel_rc_pos = vec3f(0.0);
    var sel_rc_normal = vec3f(0.0, 0.0, 1.0);
    var sel_rc_wi = vec3f(0.0);
    var sel_rc_radiance = vec4f(0.0);

    // Reconnection-vertex (x2) tracking. Positions/normals are stored in
    // the owning object's INDEX space so rigid motion never stales them;
    // suffix directions are world for d == 3 (they point at the env) and
    // index space for deeper suffixes.
    var x2_pos = vec3f(0.0);        // index space
    var x2_normal = vec3f(0.0, 0.0, 1.0); // index space
    var x2_meta = 0u;               // material/object bits for path_flags
    var have_x2 = false;
    var x2_brdf_dir = vec3f(0.0);       // world
    var x2_brdf_dir_idx = vec3f(0.0);   // index space
    var x2_brdf_pdf = 0.0;
    var thr2: array<f32, WAVELENGTH_SAMPLE_COUNT>; // throughput after x2's BSDF (suffix divider)
    var have_thr2 = false;
    var sel_init_seed = input.rng_frame;

    for (var k = 1u; k <= input.max_bounces; k++) {
        let light_hit = intersect_lights(ray);
        let hit = intersect_scene(ray, iterations);
        // BSDF-sampled hit of a lamp: MIS-weighted emission
        if (light_hit.index >= 0 && light_hit.t < hit.distance) {
            var c_l = vec4f(0.0);
            var inc_l = vec4f(0.0);
            for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                c_l[i] = nee_throughput[i] * input.emission_integral;
                if (have_thr2) {
                    inc_l[i] = c_l[i] / max(thr2[i], 1e-12);
                }
            }
            let pl = luminance(project_spectral(c_l, rgb_and_phases));
            if (pl > 0.0) {
                w_sum += pl;
                if (get_random_numbers(seed).x * w_sum < pl) {
                    sel_f = c_l;
                    sel_flags = k | x2_meta; // d = k (x_k on the lamp)
                    sel_rc_pos = x2_pos;
                    sel_rc_normal = x2_normal;
                    sel_rc_wi = x2_brdf_dir_idx;
                    sel_rc_radiance = inc_l;
                }
            }
            if (k == 1u) {
                clear_gbuffer(pixel);
            }
            break;
        }
        // Miss: environment (or the display-only white backdrop at k == 1)
        if (hit.object_index < 0) {
            if (k == 1u) {
                clear_gbuffer(pixel);
            }
            if (k == 1u && input.white_background == 1u) {
                // Flat spectrum scaled so E[projection] equals the
                // reference's display constant (6/exposure * 1/4)
                let c_l = vec4f(6.0 / input.exposure * 0.25);
                w_sum = luminance(project_spectral(c_l, rgb_and_phases));
                sel_f = c_l;
                sel_flags = 1u;
                break;
            }
            var c_l = vec4f(0.0);
            var inc_l = vec4f(0.0);
            // Raw (MIS-free) per-lambda env radiance for the reservoir
            var env_l = vec4f(0.0);
            if (input.environment == ENVIRONMENT_STUDIO) {
                env_l = vec4f(input.dome_integral);
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    c_l[i] = throughput[i] * input.dome_integral;
                }
            } else if (input.environment == ENVIRONMENT_SKY) {
                env_l = spectral_env_weights(skyRadianceRGB(ray.direction, true), rgb_and_phases);
                var mw = 1.0;
                if (dot(ray.direction, skyState.sunDirection) >= SUN_CONE_COS) {
                    mw = env_weight;
                }
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    c_l[i] = throughput[i] * mw * env_l[i];
                }
            } else {
                env_l = spectral_env_weights(environment_rgb(ray.direction), rgb_and_phases);
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    c_l[i] = throughput[i] * env_weight * env_l[i];
                }
            }
            if (have_thr2) {
                for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                    inc_l[i] = c_l[i] / max(thr2[i], 1e-12);
                }
            }
            let pl = luminance(project_spectral(c_l, rgb_and_phases));
            if (pl > 0.0) {
                w_sum += pl;
                if (get_random_numbers(seed).x * w_sum < pl) {
                    sel_f = c_l;
                    sel_flags = k; // d = k (x_k on the environment)
                    if (k == 2u) {
                        // x2 IS the env vertex: reconnect by direction copy
                        sel_flags |= PATH_FLAG_RC_ENV;
                        sel_rc_pos = ray.direction;
                        sel_rc_normal = -ray.direction;
                        sel_rc_wi = ray.direction;
                        sel_rc_radiance = env_l; // raw; MIS re-derived at shift
                    } else if (k == 3u) {
                        // Suffix escaped right after rc: raw env + WORLD dir
                        // so the shift re-derives its MIS
                        sel_flags |= x2_meta | PATH_FLAG_SUFFIX_ENV;
                        sel_rc_pos = x2_pos;
                        sel_rc_normal = x2_normal;
                        sel_rc_wi = ray.direction;
                        sel_rc_radiance = env_l;
                    } else {
                        sel_flags |= x2_meta;
                        sel_rc_pos = x2_pos;
                        sel_rc_normal = x2_normal;
                        sel_rc_wi = x2_brdf_dir_idx;
                        sel_rc_radiance = inc_l;
                    }
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
        if (!(dot(s.normal, s.normal) > 0.5)) {
            s.normal = -ray.direction;
        }
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

        if (k == 2u) {
            x2_pos = (obj.transform * vec4f(s.pos, 1.0)).xyz;
            x2_normal = normalize((obj.transform * vec4f(s.normal, 0.0)).xyz);
            x2_meta = ((obj.material_index & 3u) << 6u) | ((u32(hit.object_index) & 3u) << 8u);
            have_x2 = true;
        }

        var reflectance: array<f32, WAVELENGTH_SAMPLE_COUNT>;
        let fourier = fourier_srgb_to_fourier(s.base_color);
        let lagranges = prep_reflectance_real_lagrange_biased_3(fourier);
        for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
            reflectance[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, lagranges);
        }

        if (k == input.max_bounces) {
            if (k == 1u) {
                clear_gbuffer(pixel);
            }
            break;
        }

        if (k == 1u) {
            // w = material + depth/16384 in [mat, mat+1): u32(w) decodes
            // the material EXACTLY for any depth (the old material+4*depth
            // packing was undecodable for fractional depths — validation
            // passed on depth-contour bands only, the banding root cause)
            textureStore(gbuffer_out, pixel,
                vec4f(s.normal, f32(obj.material_index) + clamp(hit.distance * (1.0 / 16384.0), 0.0, 0.9999)));
            textureStore(gdepth_out, pixel, vec4f(hit.distance, 0.0, 0.0, 0.0));
        }

        // Lamp NEE
        var total_light_importance: f32;
        let light_dir = sample_lights(&total_light_importance, s.pos, s.normal, get_random_numbers(seed));
        let lambert_in_0 = dot(s.normal, light_dir);
        if (lambert_in_0 > 0.0) {
            let nee_ray = Ray(s.pos, light_dir);
            let nee_light = intersect_lights(nee_ray);
            if (nee_light.index >= 0) {
                var nee_iterations = 0u;
                let occluder = intersect_scene(nee_ray, &nee_iterations);
                *iterations += nee_iterations;
                if (occluder.distance > nee_light.t) {
                    let light_density_0 = get_lights_density(total_light_importance, s.pos, light_dir, true);
                    let brdf_density_0 = get_frostbite_brdf_density(s, light_dir);
                    let spectrum_scale = lambert_in_0 / (light_density_0 + brdf_density_0);
                    let brdf = frostbite_brdf(s, light_dir);
                    var c_l = vec4f(0.0);
                    var inc_l = vec4f(0.0);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        c_l[i] = throughput[i] * (reflectance[i] * brdf.x + brdf.y) * input.emission_integral * spectrum_scale;
                        if (have_thr2) {
                            inc_l[i] = c_l[i] / max(thr2[i], 1e-12);
                        }
                    }
                    let pl = luminance(project_spectral(c_l, rgb_and_phases));
                    if (pl > 0.0) {
                        w_sum += pl;
                        if (get_random_numbers(seed).x * w_sum < pl) {
                            sel_f = c_l;
                            sel_flags = (k + 1u) | PATH_FLAG_NEE | x2_meta; // d = k+1
                            sel_rc_pos = x2_pos;
                            sel_rc_normal = x2_normal;
                            if (k == 2u) {
                                sel_rc_wi = light_dir;
                                sel_rc_radiance = vec4f(input.emission_integral);
                            } else {
                                sel_rc_wi = x2_brdf_dir_idx;
                                sel_rc_radiance = inc_l;
                            }
                        }
                    }
                }
            }
        }

        // Sun NEE (sky)
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
                    var c_l = vec4f(0.0);
                    var inc_l = vec4f(0.0);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        c_l[i] = throughput[i] * (reflectance[i] * brdf_sun.x + brdf_sun.y) * sun_scale * sun_w[i];
                        if (have_thr2) {
                            inc_l[i] = c_l[i] / max(thr2[i], 1e-12);
                        }
                    }
                    let pl = luminance(project_spectral(c_l, rgb_and_phases));
                    if (pl > 0.0) {
                        w_sum += pl;
                        if (get_random_numbers(seed).x * w_sum < pl) {
                            sel_f = c_l;
                            sel_flags = (k + 1u) | PATH_FLAG_NEE;
                            if (k == 1u) {
                                // d = 2: the sun IS x2 — env reconnection
                                sel_flags |= PATH_FLAG_RC_ENV;
                                sel_rc_pos = sun_dir;
                                sel_rc_normal = -sun_dir;
                                sel_rc_wi = sun_dir;
                                sel_rc_radiance = sun_w;
                            } else if (k == 2u) {
                                sel_flags |= x2_meta | PATH_FLAG_SUFFIX_ENV;
                                sel_rc_pos = x2_pos;
                                sel_rc_normal = x2_normal;
                                sel_rc_wi = sun_dir;
                                sel_rc_radiance = sun_w;
                            } else {
                                sel_flags |= x2_meta;
                                sel_rc_pos = x2_pos;
                                sel_rc_normal = x2_normal;
                                sel_rc_wi = x2_brdf_dir_idx;
                                sel_rc_radiance = inc_l;
                            }
                        }
                    }
                }
            }
        }

        // Environment NEE (HDRI)
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
                    let nee_env_w = spectral_env_weights(environment_rgb(env_dir), rgb_and_phases);
                    var c_l = vec4f(0.0);
                    var inc_l = vec4f(0.0);
                    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                        c_l[i] = throughput[i] * (reflectance[i] * brdf_env.x + brdf_env.y) * env_scale * nee_env_w[i];
                        if (have_thr2) {
                            inc_l[i] = c_l[i] / max(thr2[i], 1e-12);
                        }
                    }
                    let pl = luminance(project_spectral(c_l, rgb_and_phases));
                    if (pl > 0.0) {
                        w_sum += pl;
                        if (get_random_numbers(seed).x * w_sum < pl) {
                            sel_f = c_l;
                            sel_flags = (k + 1u) | PATH_FLAG_NEE;
                            if (k == 1u) {
                                sel_flags |= PATH_FLAG_RC_ENV;
                                sel_rc_pos = env_dir;
                                sel_rc_normal = -env_dir;
                                sel_rc_wi = env_dir;
                                sel_rc_radiance = nee_env_w;
                            } else if (k == 2u) {
                                sel_flags |= x2_meta | PATH_FLAG_SUFFIX_ENV;
                                sel_rc_pos = x2_pos;
                                sel_rc_normal = x2_normal;
                                sel_rc_wi = env_dir;
                                sel_rc_radiance = nee_env_w;
                            } else {
                                sel_flags |= x2_meta;
                                sel_rc_pos = x2_pos;
                                sel_rc_normal = x2_normal;
                                sel_rc_wi = x2_brdf_dir_idx;
                                sel_rc_radiance = inc_l;
                            }
                        }
                    }
                }
            }
        }

        // Sample the BRDF to continue the path
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

        if (k == 2u) {
            x2_brdf_dir = ray.direction;
            x2_brdf_dir_idx = normalize((obj.transform * vec4f(ray.direction, 0.0)).xyz);
            x2_brdf_pdf = brdf_density_1;
            // Suffix divider captured BEFORE the k=2 RR division: the RR
            // compensation belongs to the suffix estimator (rc_radiance =
            // c_l / thr2 must keep 1/survival so E[rc_radiance] is the true
            // incident radiance). Capturing after RR made the identity
            // shift return f * survival — a systematic energy loss on every
            // reuse of d >= 4 paths.
            for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                thr2[i] = throughput[i];
            }
            have_thr2 = true;
        }

        // Russian roulette (kept in the sampling PDF; decoupled from replay
        // in P3 per supplemental §6)
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

    // Write the reservoir (empty when w_sum == 0). Temporal reuse runs in
    // its own pass (temporalMain) to keep this kernel's register footprint
    // sane — fusing the shifts here stalled the GPU into watchdog territory.

    var r: Reservoir;
    r.f = sel_f;
    r.w = 0.0;
    let pl_sel = luminance(project_spectral(sel_f, rgb_and_phases));
    if (pl_sel > 0.0) {
        r.w = w_sum / pl_sel;
    }
    r.rc_pos = sel_rc_pos;
    r.m = 1.0;
    r.rc_normal_oct = oct_encode(sel_rc_normal);
    r.rc_wi_x = sel_rc_wi.x;
    r.rc_wi_y = sel_rc_wi.y;
    r.rc_wi_z = sel_rc_wi.z;
    r.init_seed = sel_init_seed;
    r.path_flags = sel_flags;
    r.rc_radiance = sel_rc_radiance;
    r.wavelength_u = wavelength_u;
    reservoirs[reservoir_index(pixel, dims, 2u)] = r;
    // Mirror into the frame-final region: shade + next frame's temporal
    // read it directly while the spatial pass is disabled (see P2 note)
    reservoirs[reservoir_index(pixel, dims, input.rng_frame & 1u)] = r;
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

// Reference: the frozen single-kernel path tracer (ground truth)
@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(output_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    var seed = vec2u(global_id.xy) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    // Sub-pixel jitter for anti-aliasing via accumulation
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5 + jitter, vec2f(dims));

    var iterations = 0u;
    let radiance = path_trace(ray, global_id.xy, &seed, &iterations);
    write_output(global_id.xy, dims, radiance);
}

// ReSTIR PT pass 0: full-res primary visibility. Traces ONLY the primary
// ray and writes the full-res G-buffer + exact depth. The reservoir passes
// run at a lower GI resolution; shade reconstructs THIS primary and shifts
// the low-res reservoir onto it (sharp silhouettes + full-res shadows).
@compute @workgroup_size(8, 8)
fn primaryMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(gbuffer_out);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }
    let pixel = global_id.xy;
    var seed = vec2u(pixel) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(pixel) + 0.5 + jitter, vec2f(dims));
    var it = 0u;
    let hit = intersect_scene(ray, &it);
    let lh = intersect_lights(ray);
    if (hit.object_index < 0 || (lh.index >= 0 && lh.t < hit.distance)) {
        textureStore(gbuffer_out, pixel, vec4f(0.0, 0.0, 0.0, -1.0));
        textureStore(gdepth_out, pixel, vec4f(0.0));
        return;
    }
    let obj = objects[hit.object_index];
    var n = hit.normal;
    if (!(dot(n, n) > 0.5)) { n = -ray.direction; }
    if (dot(n, ray.direction) > 0.0) { n = -n; }
    textureStore(gbuffer_out, pixel,
        vec4f(n, f32(obj.material_index) + clamp(hit.distance * (1.0 / 16384.0), 0.0, 0.9999)));
    textureStore(gdepth_out, pixel, vec4f(hit.distance, 0.0, 0.0, 0.0));
}

// ReSTIR PT pass 1: initial sampling (1spp path tree -> RIS -> reservoir)
@compute @workgroup_size(8, 8)
fn initialMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(gbuffer_out);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    var seed = vec2u(global_id.xy) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5 + jitter, vec2f(dims));

    var iterations = 0u;
    initial_sample(ray, global_id.xy, dims, &seed, &iterations);
}

// ReSTIR PT pass 2: temporal reuse (plan P1). Re-derives the primary hit
// (deterministic: same seed stream as initialMain), reprojects it through
// the object motion + previous view, validates against the previous
// G-buffer, then GRIS-merges the canonical reservoir with the temporal one
// using reconnection shifts both ways and generalized Talbot MIS with
// confidence weights (GRIS Eq. 36), cap 20. Runs as its own pass to keep
// register pressure bounded.
@compute @workgroup_size(8, 8)
fn temporalMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(gbuffer_cur);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }
    let pixel = global_id.xy;

    // Same RNG derivation as initialMain: jitter, then the pixel's
    // per-pixel wavelength offset (must match initialMain's draw exactly)
    var seed = vec2u(pixel) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(pixel) + 0.5 + jitter, vec2f(dims));
    let wavelength_u = get_random_numbers(&seed).x;
    let rgb_and_phases = wavelengths_at(wavelength_u);
    // Decorrelate the merge decision from initialMain's stream
    seed ^= vec2u(0x9e3779b9u, 0x85ebca6bu);

    var iterations = 0u;
    // Reconstruct this frame's primary hit from the G-buffer (initialMain
    // already traced it: same jitter stream, exact depth in gdepth_cur)
    let g = textureLoad(gbuffer_cur, vec2i(pixel), 0);
    if (g.w < 0.0) {
        return; // miss pixels keep the canonical (d = 1) sample
    }
    let mat_idx = u32(g.w) & 3u;
    let depth = textureLoad(gdepth_cur, vec2i(pixel), 0).r;
    // Materials map 1:1 to objects (0 = model, 1 = ground)
    let obj = objects[min(mat_idx, 1u)];
    let mat = materials[mat_idx];
    var s: ShadingData;
    s.pos = ray.origin + ray.direction * depth;
    s.normal = g.xyz; // stored post-faceforward
    s.out_dir = -ray.direction;
    s.lambert_out = dot(s.normal, s.out_dir);
    s.base_color = mat.base_color;
    s.diffuse_albedo = mat.diffuse_albedo;
    s.fresnel_0 = mat.fresnel_0;
    s.roughness = mat.roughness;
    s.pos += s.normal * RAY_OFFSET;

    var reflectance: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    let lagranges = prep_reflectance_real_lagrange_biased_3(fourier_srgb_to_fourier(s.base_color));
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        reflectance[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, lagranges);
    }

    // Reproject through object motion + previous camera
    let pos_prev = (obj.motion * vec4f(s.pos, 1.0)).xyz;
    let cam_prev = (input.prev_view * vec4f(pos_prev, 1.0)).xyz;
    if (cam_prev.z >= 0.0) {
        return;
    }
    let aspect = f32(dims.x) / f32(dims.y);
    let uv = vec2f(
        (cam_prev.x / (-cam_prev.z)) / (aspect * input.fov_scale),
        (cam_prev.y / (-cam_prev.z)) / input.fov_scale,
    ) * 0.5 + 0.5;
    // Validated 2x2-footprint reprojection: a single truncating vec2i
    // beats against the pixel grid under camera zoom (moire rings of
    // alternating reuse/skip that persist in the reservoirs and leak into
    // the image). Pick the footprint tap with the best geometric match.
    let base_f = uv * vec2f(dims) - 0.5;
    let base = vec2i(floor(base_f));
    let n_prev = normalize((obj.motion * vec4f(s.normal, 0.0)).xyz);
    var best_tap = vec2i(-1);
    var best_score = 0.9; // minimum acceptable normal agreement
    for (var ty = 0; ty < 2; ty++) {
        for (var tx = 0; tx < 2; tx++) {
            let pp = base + vec2i(tx, ty);
            if (any(pp < vec2i(0)) || any(pp >= vec2i(dims))) {
                continue;
            }
            let gp = textureLoad(gbuffer_prev, pp, 0);
            if (gp.w < 0.0 || (u32(gp.w) & 3u) != mat_idx) {
                continue;
            }
            let score = dot(gp.xyz, n_prev);
            if (score > best_score) {
                best_score = score;
                best_tap = pp;
            }
        }
    }
    if (best_tap.x < 0) {
        return;
    }
    let r_t = reservoirs[reservoir_index(vec2u(best_tap), dims, 1u - (input.rng_frame & 1u))];
    // Debug bit 6: hard-shorten the temporal chain (cap 3) to test whether
    // the shadow inflation compounds with chain length
    var cap = TEMPORAL_CONFIDENCE_CAP;
    if ((input.restir_debug & 64u) != 0u) {
        cap = 3.0;
    }
    let m_t = min(r_t.m, cap);
    if (!(m_t > 0.0) || !(r_t.w > 0.0) || (r_t.path_flags & 15u) < 2u) {
        return;
    }

    let cur_index = reservoir_index(pixel, dims, 2u);
    var r_c = reservoirs[cur_index];
    let p_c = luminance(project_spectral(r_c.f, rgb_and_phases));

    // Forward shift: temporal sample into this pixel's domain, INCLUDING
    // the wavelength shift onto the current frame's basis (exact for env
    // suffixes, stratified-bin interpolation for sampled deep suffixes).
    // All candidates then live in ONE basis, which makes Talbot MIS's
    // density ratios well-defined again (per-sample bases had disjoint
    // wavelength supports — the old +3-5% Talbot failure).
    let prev_sd = make_shading_data(pos_prev, n_prev,
        normalize(input.prev_camera_pos - pos_prev), mat);
    let fwd = shift_reconnect(r_t, s, reflectance, rgb_and_phases,
        wavelength_u, prev_sd, false, 0.05, &iterations);
    let p_fwd = luminance(project_spectral(fwd.f, rgb_and_phases));

    // Canonical Talbot weight needs the reverse shift: the canonical
    // sample evaluated in the temporal technique's domain (prev geometry,
    // the temporal sample's wavelength basis)
    let phases_t = wavelengths_at(r_t.wavelength_u);
    var refl_t: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        refl_t[i] = eval_reflectance_real_lagrange_3(phases_t[i].w, lagranges);
    }
    var m_can = 1.0;
    if (p_c > 0.0 && r_c.w > 0.0) {
        let rev = shift_reconnect(r_c, prev_sd, refl_t, phases_t,
            r_t.wavelength_u, s, true, 0.05, &iterations);
        let p_rev = luminance(project_spectral(rev.f, phases_t));
        if (p_rev > 0.0 && rev.jacobian > 0.0) {
            // GRIS Eq. 36: canonical density vs the temporal technique's
            // confidence-weighted density at the canonical sample
            m_can = p_c / (p_c + m_t * p_rev * rev.jacobian);
        }
    }
    let w_c = m_can * p_c * r_c.w;

    if (!(p_fwd > 0.0) || !(fwd.jacobian > 0.0)) {
        // Temporal shift undefined: the canonical keeps its Talbot weight
        // (the temporal technique could still have produced it) and ages
        r_c.w = 0.0;
        if (p_c > 0.0) {
            r_c.w = w_c / p_c;
        }
        r_c.m = 1.0 + m_t;
        if ((input.restir_debug & 32u) == 0u) {
            reservoirs[cur_index] = r_c;
        }
        reservoirs[reservoir_index(pixel, dims, input.rng_frame & 1u)] = r_c;
        return;
    }

    // Temporal candidate's Talbot weight: its own confidence-weighted
    // source density (source p-hat over the forward Jacobian) vs the
    // canonical's density at the shifted sample
    let p_src = luminance(project_spectral(r_t.f, phases_t));
    let dens_t = m_t * p_src / max(fwd.jacobian, 1e-9);
    let m_temp = dens_t / max(p_fwd + dens_t, 1e-12);
    let w_t = m_temp * p_fwd * r_t.w * fwd.jacobian;
    let w_total = w_c + w_t;

    var out = r_c;
    if (get_random_numbers(&seed).x * w_total < w_t) {
        out = r_t;
        out.f = fwd.f; // rebased to the current basis by the shift
        // Keep the stored suffix radiance consistent with wavelength_u
        // (env suffixes get recomputed from the direction at reuse anyway)
        out.rc_radiance = rebase_suffix_radiance(r_t.rc_radiance,
            r_t.wavelength_u, wavelength_u);
    }
    out.wavelength_u = wavelength_u;
    out.w = 0.0;
    let pl = luminance(project_spectral(out.f, rgb_and_phases));
    if (pl > 0.0 && w_total > 0.0) {
        out.w = w_total / pl;
    }
    out.m = 1.0 + m_t;
    // Debug bit 5: leave region 2 holding initialMain's FRESH reservoirs so
    // spatialMain merges un-aged neighbors (bisects temporal-aged W in the
    // spatial injection); the temporal result still reaches shading via the
    // final region, which spatialMain uses as its canonical under this bit
    if ((input.restir_debug & 32u) == 0u) {
        reservoirs[cur_index] = out;
    }
    reservoirs[reservoir_index(pixel, dims, input.rng_frame & 1u)] = out;
}

// ReSTIR PT pass 3: spatial reuse (plan P2). Reads every pixel's
// post-temporal reservoir from the scratch region and merges 3 Gaussian
// neighbors (sigma = 16, the paper's R = 30 disk equivalent) into the
// frame-final region via chained 2-candidate GRIS merges with
// confidence-weighted Talbot MIS. Each neighbor's primary hit is re-traced
// deterministically (same jitter stream), which doubles as validation.
@compute @workgroup_size(8, 8)
fn spatialMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(gbuffer_cur);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }
    let pixel = global_id.xy;

    // Canonical = post-temporal reservoir. Normally mirrored into region 2;
    // under debug bit 5 it lives only in the final region (region 2 then
    // holds initialMain's fresh reservoirs for the neighbor reads)
    var canon_region = 2u;
    if ((input.restir_debug & 32u) != 0u) {
        canon_region = input.rng_frame & 1u;
    }
    let r_canon = reservoirs[reservoir_index(pixel, dims, canon_region)];
    var out = r_canon;
    let out_index = reservoir_index(pixel, dims, input.rng_frame & 1u);

    // Same primary derivation as initialMain
    var seed = vec2u(pixel) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(pixel) + 0.5 + jitter, vec2f(dims));
    let wavelength_u = get_random_numbers(&seed).x; // matches initialMain's draw
    let rgb_and_phases = wavelengths_at(wavelength_u);
    seed ^= vec2u(0xc2b2ae35u, 0x27d4eb2fu); // decorrelate from other passes

    var iterations = 0u;
    // Reconstruct the primary hit from the G-buffer (see temporalMain)
    let g = textureLoad(gbuffer_cur, vec2i(pixel), 0);
    if (g.w < 0.0) {
        reservoirs[out_index] = out;
        return;
    }
    let mat_idx = u32(g.w) & 3u;
    let mat = materials[mat_idx];
    var s: ShadingData;
    s.pos = ray.origin + ray.direction * textureLoad(gdepth_cur, vec2i(pixel), 0).r;
    s.normal = g.xyz;
    s.out_dir = -ray.direction;
    s.lambert_out = dot(s.normal, s.out_dir);
    s.base_color = mat.base_color;
    s.diffuse_albedo = mat.diffuse_albedo;
    s.fresnel_0 = mat.fresnel_0;
    s.roughness = mat.roughness;
    s.pos += s.normal * RAY_OFFSET;
    var reflectance: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    let lagranges = prep_reflectance_real_lagrange_biased_3(fourier_srgb_to_fourier(s.base_color));
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        reflectance[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, lagranges);
    }

    // Defensive pairwise MIS (GRIS Eq. 38, the paper's spatial choice):
    // per-candidate density ratios via one forward + one reverse shift per
    // neighbor, streamed into a fresh reservoir; the canonical merges last
    // with its accumulated defensive weight, and the final UCW divides by
    // (validNeighbors + 1). All reservoirs share the frame's wavelength
    // basis after the temporal pass's wavelength shift.
    let p_c = luminance(project_spectral(r_canon.f, rgb_and_phases));
    let m_c = r_canon.m;
    var canonical_weight = 1.0;
    var valid_neighbors = 0.0;
    var w_sum = 0.0;
    var m_sum = m_c;
    var sel = r_canon;
    var have_sel = false;

    // Debug bit 8: single SELF-merge (np = pixel, exact identity shift —
    // any deviation from parity is a mechanical merge bug); bit 9: single
    // fixed-offset neighbor (+8, 0)
    var neighbor_count = 3u;
    if ((input.restir_debug & (256u | 512u)) != 0u) {
        neighbor_count = 1u;
    }
    let k_norm = f32(neighbor_count);
    for (var n = 0u; n < neighbor_count; n++) {
        // Gaussian neighbor offset (Box-Muller, sigma = 16)
        let u = get_random_numbers(&seed);
        let rad = 16.0 * sqrt(max(-2.0 * log(max(u.x, 1e-6)), 0.0));
        let ang = 2.0 * PI * u.y;
        var np = vec2i(pixel) + vec2i(rad * vec2f(cos(ang), sin(ang)));
        if ((input.restir_debug & 256u) != 0u) {
            np = vec2i(pixel);
        } else if ((input.restir_debug & 512u) != 0u) {
            np = vec2i(pixel) + vec2i(8, 0);
        }
        if (any(np < vec2i(0)) || any(np >= vec2i(dims)) ||
            (all(np == vec2i(pixel)) && (input.restir_debug & 256u) == 0u)) {
            continue;
        }
        let r_n = reservoirs[reservoir_index(vec2u(np), dims, 2u)];
        let m_n = min(r_n.m, TEMPORAL_CONFIDENCE_CAP + 1.0);
        if (!(m_n > 0.0)) {
            continue;
        }
        // Bias-bisection gates (restir_debug == 0 in normal operation)
        if (input.restir_debug != 0u) {
            let d_n = r_n.path_flags & 15u;
            let nee_n = (r_n.path_flags & PATH_FLAG_NEE) != 0u;
            if (((input.restir_debug & 1u) != 0u && d_n == 2u) ||
                ((input.restir_debug & 2u) != 0u && d_n == 3u && nee_n) ||
                ((input.restir_debug & 4u) != 0u && d_n == 3u && !nee_n) ||
                ((input.restir_debug & 8u) != 0u && d_n >= 4u)) {
                continue;
            }
        }
        // Reconstruct the neighbor's primary hit from the G-buffer
        // (their jitter stream; no trace needed)
        let g_n = textureLoad(gbuffer_cur, np, 0);
        if (g_n.w < 0.0 || (u32(g_n.w) & 3u) != mat_idx) {
            continue;
        }
        var nseed = vec2u(vec2u(np)) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
        let njitter = get_random_numbers(&nseed) - 0.5;
        let nray = generate_camera_ray(vec2f(vec2u(np)) + 0.5 + njitter, vec2f(dims));
        var ns: ShadingData;
        ns.pos = nray.origin + nray.direction * textureLoad(gdepth_cur, np, 0).r;
        ns.normal = g_n.xyz;
        ns.out_dir = -nray.direction;
        ns.lambert_out = dot(ns.normal, ns.out_dir);
        ns.base_color = s.base_color;
        ns.diffuse_albedo = s.diffuse_albedo;
        ns.fresnel_0 = s.fresnel_0;
        ns.roughness = s.roughness;
        ns.pos += ns.normal * RAY_OFFSET;
        if (dot(ns.normal, s.normal) <= 0.9) {
            continue;
        }
        valid_neighbors += 1.0;
        m_sum += m_n;

        // Defensive canonical share for this pair: the canonical sample's
        // density in the NEIGHBOR's domain (reverse shift, evaluated in the
        // neighbor's own per-pixel wavelength basis) vs its own
        canonical_weight += 1.0;
        let phases_n = wavelengths_at(r_n.wavelength_u);
        if (p_c > 0.0 && r_canon.w > 0.0) {
            var refl_nb: array<f32, WAVELENGTH_SAMPLE_COUNT>;
            for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
                refl_nb[i] = eval_reflectance_real_lagrange_3(phases_n[i].w, lagranges);
            }
            let rev = shift_reconnect(r_canon, ns, refl_nb, phases_n,
                r_n.wavelength_u, s, false, 0.05, &iterations);
            let p_rev = luminance(project_spectral(rev.f, phases_n)) * rev.jacobian;
            if (p_rev > 0.0) {
                canonical_weight -= p_rev * m_n / (p_rev * m_n + m_c * p_c / k_norm);
            }
        }

        if (!(r_n.w > 0.0) || (r_n.path_flags & 15u) < 2u) {
            continue; // neighbor carries no usable sample (confidence counted)
        }
        // Forward shift: neighbor path into THIS pixel's per-pixel basis
        // (the wavelength shift rebases the suffix radiance)
        let fwd = shift_reconnect(r_n, s, reflectance, rgb_and_phases,
            wavelength_u, ns, false, 0.05, &iterations);
        let p_fwd = luminance(project_spectral(fwd.f, rgb_and_phases));
        if (!(p_fwd > 0.0) || !(fwd.jacobian > 0.0)) {
            continue;
        }
        // Pairwise MIS weight (GRIS Eq. 38): the neighbor technique's
        // source density is its p-hat in ITS OWN basis
        let p_src_n = luminance(project_spectral(r_n.f, phases_n));
        let dens_n = p_src_n / max(fwd.jacobian, 1e-9) * m_n;
        let nw = dens_n / max(dens_n + p_fwd * m_c / k_norm, 1e-12);
        let w_n = nw * p_fwd * r_n.w * fwd.jacobian;
        w_sum += w_n;
        if (get_random_numbers(&seed).x * w_sum < w_n) {
            sel = r_n;
            sel.f = fwd.f; // rebased to this pixel's basis by the shift
            sel.rc_radiance = rebase_suffix_radiance(r_n.rc_radiance,
                r_n.wavelength_u, wavelength_u);
            sel.wavelength_u = wavelength_u;
            have_sel = true;
        }
    }

    // Canonical sample merges last with its accumulated defensive weight
    let w_c = canonical_weight * p_c * r_canon.w;
    w_sum += w_c;
    if (w_c > 0.0 && get_random_numbers(&seed).x * w_sum < w_c) {
        sel = r_canon;
        have_sel = true;
    } else if (!have_sel) {
        sel = r_canon;
    }

    out = sel;
    out.m = m_sum;
    out.w = 0.0;
    let pl = luminance(project_spectral(out.f, rgb_and_phases));
    if (pl > 0.0 && w_sum > 0.0) {
        // Pairwise weights are not pre-divided by (k + 1); normalize here
        out.w = w_sum / pl / (valid_neighbors + 1.0);
    }
    reservoirs[out_index] = out;
}

// ReSTIR PT final pass: FULL-RES shade + GI-grid upsample. Reconstructs
// this pixel's full-res primary (from the full-res G-buffer written by
// primaryMain), finds the covering low-res reservoir cell, reconstructs
// THAT cell's primary as the shift source, and reconnection-shifts the
// reservoir onto the full-res primary. The shift re-casts visibility from
// the full-res hit, so shadows stay sharp at 1:1 even though the light
// samples live on the low-res grid. Single-cell resample: the shifted
// integrand F * W * J is an unbiased estimate of this pixel's radiance
// (GRIS, |R| = 1, no canonical); a geometrically-invalid nearest cell
// falls through to its 2x2 neighbours, a shift failure yields 0 (denoiser
// + accumulation fill the residual).
@compute @workgroup_size(8, 8)
fn shadeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(gbuffer_cur); // full-res dst G-buffer
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }
    let pixel = global_id.xy;

    // Reconstruct the full-res primary ray (same jitter as primaryMain),
    // then this pixel's own per-pixel wavelength basis
    var seed = vec2u(pixel) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
    let jitter = get_random_numbers(&seed) - 0.5;
    let ray = generate_camera_ray(vec2f(pixel) + 0.5 + jitter, vec2f(dims));
    let wavelength_u = get_random_numbers(&seed).x;
    let rgb_and_phases = wavelengths_at(wavelength_u);

    let g = textureLoad(gbuffer_cur, vec2i(pixel), 0);
    if (g.w < 0.0) {
        // Primary miss: full-res, view-dependent backdrop/env (sharp)
        var c_l: vec4f;
        if (input.white_background == 1u) {
            c_l = vec4f(6.0 / input.exposure * 0.25);
        } else {
            c_l = env_radiance_spectral(ray.direction, rgb_and_phases);
        }
        let bg = project_spectral(c_l, rgb_and_phases);
        textureStore(illum_out, pixel, vec4f(bg, 1.0));
        write_output(pixel, dims, bg);
        return;
    }

    // Full-res primary hit (shift destination)
    let mat_idx = u32(g.w) & 3u;
    let mat = materials[mat_idx];
    let depth = textureLoad(gdepth_cur, vec2i(pixel), 0).r;
    var dst_s: ShadingData;
    dst_s.pos = ray.origin + ray.direction * depth;
    dst_s.normal = g.xyz;
    dst_s.out_dir = -ray.direction;
    dst_s.lambert_out = dot(dst_s.normal, dst_s.out_dir);
    dst_s.base_color = mat.base_color;
    dst_s.diffuse_albedo = mat.diffuse_albedo;
    dst_s.fresnel_0 = mat.fresnel_0;
    dst_s.roughness = mat.roughness;
    dst_s.pos += dst_s.normal * RAY_OFFSET;
    var dst_refl: array<f32, WAVELENGTH_SAMPLE_COUNT>;
    let lagr = prep_reflectance_real_lagrange_biased_3(fourier_srgb_to_fourier(dst_s.base_color));
    for (var i = 0u; i < WAVELENGTH_SAMPLE_COUNT; i++) {
        dst_refl[i] = eval_reflectance_real_lagrange_3(rgb_and_phases[i].w, lagr);
    }

    // Covering low-res cell (nearest-first over the 2x2 footprint). Cell
    // selection depends only on GEOMETRY (material match + occupancy), never
    // on the shift outcome, so shift failure -> 0 stays value-independent.
    let gi_dims = textureDimensions(gi_gbuffer);
    let base = vec2i(vec2f(pixel) * (vec2f(gi_dims) / vec2f(dims)));
    var radiance = vec3f(0.0);
    var it = 0u;
    for (var t = 0; t < 4; t++) {
        let cell = clamp(base + vec2i(t & 1, (t >> 1) & 1), vec2i(0), vec2i(gi_dims) - 1);
        let g_gi = textureLoad(gi_gbuffer, cell, 0);
        if (g_gi.w < 0.0 || (u32(g_gi.w) & 3u) != mat_idx) { continue; }
        let r = reservoirs[reservoir_index(vec2u(cell), gi_dims, input.rng_frame & 1u)];
        if (!(r.w > 0.0) || (r.path_flags & 15u) < 2u) { continue; }
        // Reconstruct the cell's primary (its own jitter stream) as the src
        var gseed = vec2u(vec2u(cell)) ^ vec2u(input.rng_frame << 16u, (input.rng_frame + 237u) << 16u);
        let gjit = get_random_numbers(&gseed) - 0.5;
        let gray = generate_camera_ray(vec2f(vec2u(cell)) + 0.5 + gjit, vec2f(gi_dims));
        var src_s: ShadingData;
        src_s.pos = gray.origin + gray.direction * textureLoad(gi_gdepth, cell, 0).r;
        src_s.normal = g_gi.xyz;
        src_s.out_dir = -gray.direction;
        src_s.lambert_out = dot(src_s.normal, src_s.out_dir);
        src_s.base_color = mat.base_color;
        src_s.diffuse_albedo = mat.diffuse_albedo;
        src_s.fresnel_0 = mat.fresnel_0;
        src_s.roughness = mat.roughness;
        src_s.pos += src_s.normal * RAY_OFFSET;
        let fwd = shift_reconnect(r, dst_s, dst_refl, rgb_and_phases, wavelength_u, src_s, false, 0.005, &it);
        if (fwd.jacobian > 0.0) {
            radiance = project_spectral(fwd.f, rgb_and_phases) * r.w * fwd.jacobian;
        }
        break; // first geometrically-valid cell decides (value-independent)
    }
    if (!(radiance.x + radiance.y + radiance.z < 1e20)) { radiance = vec3f(0.0); }
    radiance = clamp(radiance, vec3f(0.0), vec3f(100.0));
    textureStore(illum_out, pixel, vec4f(radiance, 1.0));
    write_output(pixel, dims, radiance);
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
