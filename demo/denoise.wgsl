// SVGF denoiser (Schied 2017; docs/restir-pt-plan.md P5) for the ReSTIR
// output. Standalone module: temporal accumulation with motion-vector
// reprojection + luminance moments -> variance, then a 5-level edge-aware
// a-trous wavelet with variance-guided luminance weights. The first
// wavelet iteration's output feeds back as next frame's color history.
//
// v1 simplifications (documented in the plan): no albedo demodulation (all
// materials are flat colors; material-id + normal edge stopping preserves
// their boundaries), single-tap reprojection (no 2x2 bilinear), variance
// floor instead of the 7x7 spatial fallback under 4 frames of history.

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
    _pad0: u32,
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

struct DenoiseParams {
    step: u32,          // a-trous hole size = 1 << step
    write_history: u32, // 1 on the first wavelet iteration
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> input: Input;
@group(0) @binding(1) var<storage> objects: array<Object>;
@group(0) @binding(2) var<uniform> params: DenoiseParams;
// Per-frame raw radiance from the ReSTIR shade pass
@group(0) @binding(3) var illum: texture_2d<f32>;
// Current G-buffer (normal.xyz, material + 4*depth; w < 0 = miss)
@group(0) @binding(4) var gbuf: texture_2d<f32>;
@group(0) @binding(5) var gbuf_prev: texture_2d<f32>;
// Color history (rgb, history length) and moments (mu1, mu2) ping-pong
@group(0) @binding(6) var hist_prev: texture_2d<f32>;
@group(0) @binding(7) var moments_prev: texture_2d<f32>;
@group(0) @binding(8) var hist_out: texture_storage_2d<rgba32float, write>;
@group(0) @binding(9) var moments_out: texture_storage_2d<rgba32float, write>;
// A-trous ping-pong (rgb color, w = variance)
@group(0) @binding(10) var atrous_src: texture_2d<f32>;
@group(0) @binding(11) var atrous_dst: texture_storage_2d<rgba32float, write>;
// Final 8-bit output (resolve pass)
@group(0) @binding(12) var resolve_out: texture_storage_2d<rgba8unorm, write>;
// Per-material linear-RGB diffuse albedo (authored colours). SVGF albedo
// demodulation: filter irradiance = radiance / albedo (smooth, denoisable)
// and remodulate at resolve, so the a-trous blurs the LIGHTING without
// smearing the sharp material colour (the co-design 1-spp signal).
@group(0) @binding(14) var<uniform> material_albedo: array<vec4f, 8>;

// Demodulation albedo for a G-buffer texel (1 for env/miss = no demod)
fn demod_albedo(g_w: f32) -> vec3f {
    if (g_w < 0.0) { return vec3f(1.0); }
    return max(material_albedo[u32(g_w) & 3u].rgb, vec3f(0.04));
}

fn dn_luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// Reconstruct the primary-hit world position from the G-buffer depth
fn world_pos(pixel: vec2u, dims: vec2u, depth: f32) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / vec2f(dims) * 2.0 - 1.0;
    let aspect = f32(dims.x) / f32(dims.y);
    let dir_cam = normalize(vec3f(uv.x * aspect * input.fov_scale, uv.y * input.fov_scale, -1.0));
    let origin = (input.camera_matrix * vec4f(0.0, 0.0, 0.0, 1.0)).xyz;
    let dir = normalize((input.camera_matrix * vec4f(dir_cam, 0.0)).xyz);
    return origin + dir * depth;
}

// True when the 3x3 neighborhood spans more than one material (or touches
// a miss) — the AA-jittered silhouette band
fn material_boundary(pixel: vec2u, dims: vec2u, mat_c: u32) -> bool {
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = vec2i(pixel) + vec2i(dx, dy);
            if (any(q < vec2i(0)) || any(q >= vec2i(dims))) {
                continue;
            }
            let gq = textureLoad(gbuf, q, 0);
            if (gq.w < 0.0 || (u32(gq.w) & 3u) != mat_c) {
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Pass 1: temporal accumulation + moments (SVGF §4.1-4.2)
// ============================================================================
@compute @workgroup_size(8, 8)
fn temporalAccumMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(illum);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let pixel = gid.xy;

    let g = textureLoad(gbuf, pixel, 0);
    // illum is already albedo-demodulated irradiance (shade-side, cast-
    // cancelling), so the denoiser filters it directly; resolve remodulates
    let c_in = textureLoad(illum, pixel, 0).rgb;
    let l_in = dn_luminance(c_in);

    var color = c_in;
    var mu = vec2f(l_in, l_in * l_in);
    var hist_len = 1.0;

    if (g.w < 0.0) {
        // Primary miss (backdrop/env): accumulate in place — the value is
        // a smooth env/backdrop color, and without accumulation the whole
        // background shows each frame's raw wavelength-set cast (hue
        // flicker). Validated against the previous frame also missing.
        let gp = textureLoad(gbuf_prev, vec2i(pixel), 0);
        if (gp.w < 0.0) {
            let h = textureLoad(hist_prev, vec2i(pixel), 0);
            let m = textureLoad(moments_prev, vec2i(pixel), 0);
            hist_len = min(h.a + 1.0, 64.0);
            let alpha = max(0.1, 1.0 / hist_len);
            color = mix(h.rgb, c_in, alpha);
            mu = mix(m.xy, mu, alpha);
        }
    } else if (material_boundary(pixel, dims, u32(g.w) & 3u)) {
        // Material-boundary (silhouette) pixels flip material under the AA
        // jitter every frame; ANY material-gated history dies immediately
        // and the pixel never settles (the edge jitter). Their converged
        // value is the jitter-weighted MIX of both materials, so always
        // accumulate the identity tap — one consistent estimator. The 0.15
        // alpha floor keeps motion ghosting on the ~1px rim short-lived.
        let h = textureLoad(hist_prev, vec2i(pixel), 0);
        let m = textureLoad(moments_prev, vec2i(pixel), 0);
        hist_len = min(h.a + 1.0, 64.0);
        let alpha = max(0.15, 1.0 / hist_len);
        color = mix(h.rgb, c_in, alpha);
        mu = mix(m.xy, mu, alpha);
    } else {
        let mat_idx = u32(g.w) & 3u;
        let depth = (g.w - f32(mat_idx)) * 16384.0;
        let pos = world_pos(pixel, dims, depth);
        // Our two objects map 1:1 to materials (0 = model, 1 = ground)
        let obj = objects[min(mat_idx, 1u)];
        let pos_prev = (obj.motion * vec4f(pos, 1.0)).xyz;
        let cam_prev = (input.prev_view * vec4f(pos_prev, 1.0)).xyz;
        if (cam_prev.z < 0.0) {
            let aspect = f32(dims.x) / f32(dims.y);
            let uv = vec2f(
                (cam_prev.x / (-cam_prev.z)) / (aspect * input.fov_scale),
                (cam_prev.y / (-cam_prev.z)) / input.fov_scale,
            ) * 0.5 + 0.5;
            // 2x2 bilinear reprojection with PER-TAP validation (SVGF
            // §4.1): invalid taps redistribute their weight — voxel-normal
            // noise no longer kills whole pixels' history
            let base_f = uv * vec2f(dims) - 0.5;
            let base = vec2i(floor(base_f));
            let frac = base_f - floor(base_f);
            let n_prev = normalize((obj.motion * vec4f(g.xyz, 0.0)).xyz);
            var h_sum = vec4f(0.0);
            var m_sum = vec2f(0.0);
            var w_sum = 0.0;
            for (var ty = 0; ty < 2; ty++) {
                for (var tx = 0; tx < 2; tx++) {
                    let pp = base + vec2i(tx, ty);
                    if (any(pp < vec2i(0)) || any(pp >= vec2i(dims))) {
                        continue;
                    }
                    let gp = textureLoad(gbuf_prev, pp, 0);
                    if (gp.w < 0.0 || (u32(gp.w) & 3u) != mat_idx || dot(gp.xyz, n_prev) <= 0.75) {
                        continue;
                    }
                    let bw = mix(1.0 - frac.x, frac.x, f32(tx)) * mix(1.0 - frac.y, frac.y, f32(ty));
                    h_sum += bw * textureLoad(hist_prev, pp, 0);
                    m_sum += bw * textureLoad(moments_prev, pp, 0).xy;
                    w_sum += bw;
                }
            }
            if (w_sum > 0.02) {
                let h = h_sum / w_sum;
                let m = m_sum / w_sum;
                hist_len = min(h.a + 1.0, 64.0);
                // Converge like an average early, EMA later. Floor lowered
                // 0.1 -> 0.05 now that the illum is shade-side albedo-
                // demodulated (cast-free): the residual per-frame variance
                // is smaller, so longer temporal averaging is cheap and it
                // replaces the removed accumulation-blend crutch's stability
                // (~20-frame EMA). Motion/disocclusion still resets history.
                let alpha = max(0.05, 1.0 / hist_len);
                color = mix(h.rgb, c_in, alpha);
                mu = mix(m, mu, alpha);
            }
        }
    }

    // Fresh/disoccluded pixels (boundary pixels are handled above): fall
    // back to a material-gated 3x3 mean of the raw input instead of
    // passing single-sample noise through (the dark edge fringe)
    if (hist_len < 2.0 && g.w >= 0.0) {
        let mat_c = u32(g.w) & 3u;
        var csum = vec3f(0.0);
        var ccnt = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let q = vec2i(pixel) + vec2i(dx, dy);
                if (any(q < vec2i(0)) || any(q >= vec2i(dims))) {
                    continue;
                }
                let gq = textureLoad(gbuf, q, 0);
                if (gq.w < 0.0 || (u32(gq.w) & 3u) != mat_c) {
                    continue;
                }
                csum += textureLoad(illum, vec2u(q), 0).rgb;
                ccnt += 1.0;
            }
        }
        if (ccnt > 1.0) {
            color = csum / ccnt;
            let l = dn_luminance(color);
            mu = vec2f(l, l * l);
        }
    }

    // Temporal variance; while history is young, estimate it SPATIALLY
    // from the raw input's luminance moments over a gated 5x5 (SVGF §4.2 —
    // the flat floor used before disabled the luminance edge-stop and let
    // the a-trous kernel ring across the shadow)
    var variance = max(mu.y - mu.x * mu.x, 0.0);
    if (g.w >= 0.0 && hist_len < 4.0) {
        let mat_c = u32(g.w) & 3u;
        var s1 = 0.0;
        var s2 = 0.0;
        var cnt = 0.0;
        for (var dy = -2; dy <= 2; dy++) {
            for (var dx = -2; dx <= 2; dx++) {
                let q = vec2i(pixel) + vec2i(dx, dy);
                if (any(q < vec2i(0)) || any(q >= vec2i(dims))) {
                    continue;
                }
                let gq = textureLoad(gbuf, q, 0);
                if (gq.w < 0.0 || (u32(gq.w) & 3u) != mat_c || dot(gq.xyz, g.xyz) <= 0.75) {
                    continue;
                }
                let lq = dn_luminance(textureLoad(illum, vec2u(q), 0).rgb);
                s1 += lq;
                s2 += lq * lq;
                cnt += 1.0;
            }
        }
        if (cnt > 1.0) {
            variance = max(s2 / cnt - (s1 / cnt) * (s1 / cnt), 0.0);
        }
    }
    textureStore(hist_out, pixel, vec4f(color, hist_len));
    textureStore(moments_out, pixel, vec4f(mu, variance, hist_len));
    // Seed the a-trous chain: (color, variance)
    textureStore(atrous_dst, pixel, vec4f(color, variance));
}

// ============================================================================
// Pass 2 (x5): edge-avoiding a-trous wavelet (SVGF §4.3-4.4)
// ============================================================================
const KERNEL = array<f32, 3>(3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
const SIGMA_Z = 1.0;
const SIGMA_N = 128.0;
const SIGMA_L = 4.0;

@compute @workgroup_size(8, 8)
fn atrousMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(atrous_src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let pixel = vec2i(gid.xy);

    let center = textureLoad(atrous_src, pixel, 0);
    let g = textureLoad(gbuf, pixel, 0);
    if (g.w < 0.0) {
        // Miss pixels (backdrop) pass through unfiltered
        textureStore(atrous_dst, gid.xy, center);
        if (params.write_history == 1u) {
            // moments slot holds THIS frame's moments; w = history length
            textureStore(hist_out, gid.xy, vec4f(center.rgb, textureLoad(moments_prev, pixel, 0).w));
        }
        return;
    }
    let mat_c = u32(g.w) & 3u;
    let depth_c = (g.w - f32(mat_c)) * 16384.0;
    let n_c = g.xyz;
    let pos_c = world_pos(gid.xy, dims, depth_c);
    let l_c = dn_luminance(center.rgb);

    // 3x3 Gaussian-prefiltered variance drives the luminance weight only
    var var_pre = 0.0;
    {
        var wsum = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let q = clamp(pixel + vec2i(dx, dy), vec2i(0), vec2i(dims) - 1);
                let wg = select(select(1.0, 2.0, (dx == 0) != (dy == 0)), 4.0, dx == 0 && dy == 0);
                var_pre += wg * textureLoad(atrous_src, q, 0).a;
                wsum += wg;
            }
        }
        var_pre /= wsum;
    }
    let sigma_l_den = SIGMA_L * sqrt(max(var_pre, 1e-10));

    let step = i32(1u << params.step);
    var sum_c = vec3f(0.0);
    var sum_v = 0.0;
    var sum_w = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = pixel + vec2i(dx, dy) * step;
            if (any(q < vec2i(0)) || any(q >= vec2i(dims))) {
                continue;
            }
            let gq = textureLoad(gbuf, q, 0);
            if (gq.w < 0.0 || (u32(gq.w) & 3u) != mat_c) {
                continue;
            }
            let sq = textureLoad(atrous_src, q, 0);
            var kern = KERNEL;
            let h = kern[abs(dx)] * kern[abs(dy)];
            // Edge-stopping: plane distance (depth), normal cosine power,
            // variance-normalized luminance
            let depth_q = (gq.w - f32(u32(gq.w) & 3u)) * 16384.0;
            let pos_q = world_pos(vec2u(q), dims, depth_q);
            let plane_d = abs(dot(n_c, pos_q - pos_c));
            let wz = exp(-plane_d / (SIGMA_Z * (0.01 * depth_c) + 1e-4));
            let wn = pow(max(dot(n_c, gq.xyz), 0.0), SIGMA_N);
            let wl = exp(-abs(dn_luminance(sq.rgb) - l_c) / (sigma_l_den + 1e-8));
            let w = h * wz * wn * wl;
            sum_c += w * sq.rgb;
            sum_v += w * w * sq.a;
            sum_w += w;
        }
    }
    var out_c = center.rgb;
    var out_v = center.a;
    if (sum_w > 1e-8) {
        out_c = sum_c / sum_w;
        out_v = sum_v / (sum_w * sum_w);
    }
    textureStore(atrous_dst, gid.xy, vec4f(out_c, out_v));
    // First iteration's output becomes next frame's color history (SVGF)
    if (params.write_history == 1u) {
        textureStore(hist_out, gid.xy, vec4f(out_c, textureLoad(moments_prev, pixel, 0).w));
    }
}

// ============================================================================
// Pass 2 (alt): WALR — Weighted A-Trous Linear Regression (src-walr, AMD 2022)
// Instead of the a-trous weighted AVERAGE, fit a per-pixel weighted linear
// model of the (demodulated) irradiance against position-offset features
// [1, dx, dy, dz] over the 5x5 stride window (moving least squares), solved
// by 4x4 Cholesky. The intercept beta0 IS the denoised centre value. A
// planar fit preserves the smooth gradients of indirect shadows/highlights
// that averaging flattens into piecewise-constant patches, so large filters
// stay usable on a 1-spp signal. Regularisation degrades gracefully to the
// weighted average when the neighbourhood is degenerate (flat/too few taps).
// ============================================================================
@compute @workgroup_size(8, 8)
fn walrMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(atrous_src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    let pixel = vec2i(gid.xy);

    let center = textureLoad(atrous_src, pixel, 0);
    let g = textureLoad(gbuf, pixel, 0);
    if (g.w < 0.0) {
        textureStore(atrous_dst, gid.xy, center);
        if (params.write_history == 1u) {
            textureStore(hist_out, gid.xy, vec4f(center.rgb, textureLoad(moments_prev, pixel, 0).w));
        }
        return;
    }
    let mat_c = u32(g.w) & 3u;
    let depth_c = (g.w - f32(mat_c)) * 16384.0;
    let n_c = g.xyz;
    let pos_c = world_pos(gid.xy, dims, depth_c);
    let step = i32(1u << params.step);
    // Scale position deltas to ~unit magnitude over the footprint so the
    // 4x4 normal-equation matrix stays well conditioned
    let fscale = 1.0 / (0.01 * depth_c * f32(step) + 1e-4);
    let l_c = dn_luminance(center.rgb);
    // 3x3-prefiltered variance drives the luminance edge-stop. This is
    // ESSENTIAL, not optional: on a flat plane the geometric weights
    // (plane-distance/normal) are identical for lit and shadowed pixels, so
    // only the luminance weight stops the regression from fitting ACROSS a
    // hard shadow edge and smearing it (the removed-shadow + ghosting bug).
    var var_pre = 0.0;
    {
        var vwsum = 0.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let q = clamp(pixel + vec2i(dx, dy), vec2i(0), vec2i(dims) - 1);
                let wg = select(select(1.0, 2.0, (dx == 0) != (dy == 0)), 4.0, dx == 0 && dy == 0);
                var_pre += wg * textureLoad(atrous_src, q, 0).a;
                vwsum += wg;
            }
        }
        var_pre /= vwsum;
    }
    let sigma_l_den = SIGMA_L * sqrt(max(var_pre, 1e-10));

    // Weighted normal-equation moments: M (4x4 symmetric upper triangle) and
    // b = X^T W y (one vec3 per feature row)
    var m00 = 0.0; var m01 = 0.0; var m02 = 0.0; var m03 = 0.0;
    var m11 = 0.0; var m12 = 0.0; var m13 = 0.0;
    var m22 = 0.0; var m23 = 0.0; var m33 = 0.0;
    var b0 = vec3f(0.0); var b1 = vec3f(0.0); var b2 = vec3f(0.0); var b3 = vec3f(0.0);
    var wsum = 0.0;
    var sum_v = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let q = pixel + vec2i(dx, dy) * step;
            if (any(q < vec2i(0)) || any(q >= vec2i(dims))) { continue; }
            let gq = textureLoad(gbuf, q, 0);
            if (gq.w < 0.0 || (u32(gq.w) & 3u) != mat_c) { continue; }
            let depth_q = (gq.w - f32(u32(gq.w) & 3u)) * 16384.0;
            let pos_q = world_pos(vec2u(q), dims, depth_q);
            let sq = textureLoad(atrous_src, q, 0);
            let y = sq.rgb;
            // Edge-stopping: plane distance + normal + variance-guided
            // LUMINANCE (the last preserves hard shadow discontinuities)
            let plane_d = abs(dot(n_c, pos_q - pos_c));
            let wz = exp(-plane_d / (SIGMA_Z * (0.01 * depth_c) + 1e-4));
            let wn = pow(max(dot(n_c, gq.xyz), 0.0), SIGMA_N);
            let wl = exp(-abs(dn_luminance(y) - l_c) / (sigma_l_den + 1e-8));
            var kern = KERNEL;
            let h = kern[abs(dx)] * kern[abs(dy)];
            let w = h * wz * wn * wl;
            if (w <= 0.0) { continue; }
            let d = (pos_q - pos_c) * fscale;
            sum_v += w * w * sq.a;
            // accumulate w * x x^T (x = [1, d.x, d.y, d.z]) and w * x y
            m00 += w;          m01 += w * d.x;      m02 += w * d.y;      m03 += w * d.z;
            m11 += w * d.x*d.x; m12 += w * d.x*d.y;  m13 += w * d.x*d.z;
            m22 += w * d.y*d.y; m23 += w * d.y*d.z;
            m33 += w * d.z*d.z;
            b0 += w * y; b1 += w * (d.x * y); b2 += w * (d.y * y); b3 += w * (d.z * y);
            wsum += w;
        }
    }
    var out_c = center.rgb;
    if (m00 > 1e-8) {
        // Ridge regularisation on the gradient terms: pulls beta_{1..3} -> 0
        // (i.e. toward the plain weighted average beta0 = sum(w*y)/sum(w))
        // when the fit is under-determined. 0.3 keeps a mild bias for
        // stability while still fitting real gradients.
        let lambda = 0.3 * m00;
        m11 += lambda; m22 += lambda; m33 += lambda;
        // Cholesky M = L L^T (lower), then forward/back substitution.
        let l00 = sqrt(max(m00, 1e-8));
        let l10 = m01 / l00; let l20 = m02 / l00; let l30 = m03 / l00;
        let l11 = sqrt(max(m11 - l10*l10, 1e-8));
        let l21 = (m12 - l20*l10) / l11; let l31 = (m13 - l30*l10) / l11;
        let l22 = sqrt(max(m22 - l20*l20 - l21*l21, 1e-8));
        let l32 = (m23 - l30*l20 - l31*l21) / l22;
        let l33 = sqrt(max(m33 - l30*l30 - l31*l31 - l32*l32, 1e-8));
        let z0 = b0 / l00;
        let z1 = (b1 - l10*z0) / l11;
        let z2 = (b2 - l20*z0 - l21*z1) / l22;
        let z3 = (b3 - l30*z0 - l31*z1 - l32*z2) / l33;
        let beta3 = z3 / l33;
        let beta2 = (z2 - l32*beta3) / l22;
        let beta1 = (z1 - l21*beta2 - l31*beta3) / l11;
        let beta0 = (z0 - l10*beta1 - l20*beta2 - l30*beta3) / l00;
        // beta0 is the fit at the centre (x = [1,0,0,0]); clamp to the
        // neighbourhood mean +/- guard to reject extrapolation overshoot
        let mean = b0 / wsum;
        out_c = clamp(beta0, mean - abs(mean) - vec3f(0.05), mean + abs(mean) + vec3f(0.05));
        out_c = max(out_c, vec3f(0.0));
    }
    let out_v = select(center.a, sum_v / (wsum * wsum), wsum > 1e-8);
    textureStore(atrous_dst, gid.xy, vec4f(out_c, out_v));
    if (params.write_history == 1u) {
        textureStore(hist_out, gid.xy, vec4f(out_c, textureLoad(moments_prev, pixel, 0).w));
    }
}

// ============================================================================
// Pass 3: resolve — tonemap the filtered radiance to the display target
// ============================================================================
fn dn_tonemap(color: vec3f) -> vec3f {
    let exposed = color * input.exposure;
    return (exposed * (2.51 * exposed + 0.03)) / (exposed * (2.43 * exposed + 0.59) + 0.14);
}

@compute @workgroup_size(8, 8)
fn resolveMain(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(atrous_src);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    var color = textureLoad(atrous_src, vec2i(gid.xy), 0).rgb;
    // Remodulate the filtered irradiance by the cast-free material albedo.
    // No accumulation-blend crutch: with a cast-free irradiance input the
    // temporal EMA converges to the true mean and the variance-guided
    // a-trous removes the residual, so the image settles on its own with
    // low lag (SVGF as designed).
    color = color * demod_albedo(textureLoad(gbuf, vec2i(gid.xy), 0).w);
    color = dn_tonemap(color);
    color = pow(color, vec3f(1.0 / 2.2));
    textureStore(resolve_out, gid.xy, vec4f(color, 1.0));
}
