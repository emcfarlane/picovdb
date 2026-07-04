// Moment-based reflectance spectra for spectral rendering.
//
// WGSL port of spectra.glsl from Christoph Peters' path tracer
// (https://github.com/MomentsInGraphics/path_tracer, branch "spectral").
// Method: "Using Moments to Represent Bounded Signals for Spectral Rendering"
// https://doi.org/10.1145/3306346.3322964
//
// A material color is stored as Fourier sRGB (3 channels, sRGB-encoded u8).
// After the usual sRGB decode, fourier_srgb_to_fourier() turns it into three
// bounded trigonometric moments, prep_reflectance_real_lagrange_biased_3()
// converts those to Lagrange multipliers once per hit, and
// eval_reflectance_real_lagrange_3() evaluates the reflectance in [0, 1] at a
// phase (a warped wavelength) per sampled wavelength.
//
// Copyright (c) 2019, 2025, Christoph Peters
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and / or other materials provided with the distribution.
//     * Neither the name of the Karlsruhe Institute of Technology nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Complex numbers are represented as vec2f (x = real, y = imaginary).

// Applies complex conjugation to the given complex number (i.e. it flips the
// sign of the imaginary part).
fn cconj(z: vec2f) -> vec2f {
    return vec2f(z.x, -z.y);
}

// This function implements complex multiplication.
fn cmul(lhs: vec2f, rhs: vec2f) -> vec2f {
    return vec2f(fma(lhs.x, rhs.x, -lhs.y * rhs.y), fma(lhs.x, rhs.y, lhs.y * rhs.x));
}

// This function implements complex multiplication followed by addition.
fn cfma(a: vec2f, b: vec2f, c: vec2f) -> vec2f {
    return vec2f(fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y)));
}

// This function computes the squared magnitude of the given complex number.
fn cabs_sq(z: vec2f) -> f32 {
    return dot(z, z);
}

// Implements Eq. 6 and 7 for three real trigonometric moments:
// https://doi.org/10.1145/3306346.3322964
fn trig_to_exp_moments_real_3(trig_moments: vec3f) -> array<vec2f, 3> {
    var out_exp_moments: array<vec2f, 3>;
    let moment_0_phase = fma(3.14159265, trig_moments[0], -1.57079633);
    out_exp_moments[0] = vec2f(cos(moment_0_phase), sin(moment_0_phase));
    out_exp_moments[0] = 0.0795774715 * out_exp_moments[0];
    out_exp_moments[1] = (trig_moments[1] * 6.28318531) * vec2f(-out_exp_moments[0].y, out_exp_moments[0].x);
    out_exp_moments[2] = fma(vec2f(trig_moments[2] * 6.28318531), vec2f(-out_exp_moments[0].y, out_exp_moments[0].x), (trig_moments[1] * 3.14159265) * vec2f(-out_exp_moments[1].y, out_exp_moments[1].x));
    out_exp_moments[0] = 2.0 * out_exp_moments[0];
    return out_exp_moments;
}

// Implements Levinson's algorithm with biasing for complex 3x3 Toeplitz
// matrices (Alg. 2): https://doi.org/10.2312/mam.20191304
// first_column may get modified slightly by the biasing procedure.
fn levinson_3_biased(first_column: ptr<function, array<vec2f, 3>>) -> array<vec2f, 3> {
    var out_solution: array<vec2f, 3>;
    var one_minus_bias = 0.9999;
    var corrected_factor = 1.0 / (1.0 - 0.9999 * 0.9999);
    out_solution[0] = vec2f(1.0 / (*first_column)[0].x, 0.0);
    var scaled_center: vec2f;
    var dot_product: vec2f;
    var dot_sq: f32;
    var flipped_1: vec2f;
    var flipped_2: vec2f;
    var factor: f32;
    scaled_center = vec2f(0.0, 0.0);
    dot_product = fma(out_solution[0].xx, (*first_column)[1], scaled_center);
    dot_sq = cabs_sq(dot_product);
    factor = 1.0 / (1.0 - dot_sq);
    if (factor < 0.0) {
        dot_product = (one_minus_bias * inverseSqrt(dot_sq)) * dot_product;
        (*first_column)[1] = (dot_product - scaled_center) * (1.0 / out_solution[0].x);
        factor = corrected_factor;
        one_minus_bias = 0.0;
        corrected_factor = 1.0;
    }
    flipped_1 = vec2f(out_solution[0].x, 0.0);
    out_solution[0] = vec2f(factor * out_solution[0].x, 0.0);
    out_solution[1] = factor * (-flipped_1.x * dot_product);
    scaled_center = cmul(out_solution[1], (*first_column)[1]);
    dot_product = fma(out_solution[0].xx, (*first_column)[2], scaled_center);
    dot_sq = cabs_sq(dot_product);
    factor = 1.0 / (1.0 - dot_sq);
    if (factor < 0.0) {
        dot_product = (one_minus_bias * inverseSqrt(dot_sq)) * dot_product;
        (*first_column)[2] = (dot_product - scaled_center) * (1.0 / out_solution[0].x);
        factor = corrected_factor;
    }
    flipped_1 = cconj(out_solution[1]);
    flipped_2 = vec2f(out_solution[0].x, 0.0);
    out_solution[0] = vec2f(factor * out_solution[0].x, 0.0);
    out_solution[1] = factor * cfma(-flipped_1, dot_product, out_solution[1]);
    out_solution[2] = factor * (-flipped_2.x * dot_product);
    return out_solution;
}

// Evaluates the autocorrelation of a complex signal with 3 entries and
// outputs results for index shifts of 0, 1 or 2.
fn real_autocorrelation_3(signal: array<vec2f, 3>) -> array<vec2f, 3> {
    var out_autocorrelation: array<vec2f, 3>;
    out_autocorrelation[0] = cfma(signal[0], cconj(signal[0]), cfma(signal[1], cconj(signal[1]), cmul(signal[2], cconj(signal[2]))));
    out_autocorrelation[1] = cfma(signal[0], cconj(signal[1]), cmul(signal[1], cconj(signal[2])));
    out_autocorrelation[2] = cmul(signal[0], cconj(signal[2]));
    return out_autocorrelation;
}

// Evaluates the first sum in Eq. 10: https://doi.org/10.1145/3306346.3322964
fn imag_correlation_3(lhs: array<vec2f, 3>, rhs: array<vec2f, 3>) -> vec3f {
    return vec3f(
        fma(lhs[0].x, rhs[0].y, fma(lhs[0].y, rhs[0].x, fma(lhs[1].x, rhs[1].y, fma(lhs[1].y, rhs[1].x, fma(lhs[2].x, rhs[2].y, lhs[2].y * rhs[2].x))))),
        fma(lhs[1].x, rhs[0].y, fma(lhs[1].y, rhs[0].x, fma(lhs[2].x, rhs[1].y, lhs[2].y * rhs[1].x))),
        fma(lhs[2].x, rhs[0].y, lhs[2].y * rhs[0].x)
    );
}

// Evaluates a Fourier series that is known to take real values, given cosine
// and sine of the evaluation point and real Fourier coefficients 0, 1 and 2.
fn eval_fourier_series_real_3(point: vec2f, fouriers: vec3f) -> f32 {
    let cos_1 = point.x;
    let cos_2 = fma(point.x, point.x, -point.y * point.y);
    return 2.0 * fma(fouriers[1], cos_1, fma(fouriers[2], cos_2, 0.5 * fouriers[0]));
}

// Applies the linear transform that turns linear Fourier sRGB into Fourier
// coefficients that can be fed to prep_reflectance_real_lagrange_biased_3().
fn fourier_srgb_to_fourier(fourier_srgb: vec3f) -> vec3f {
    return vec3f(
        dot(vec3f(0.2276800310, 0.4748793271, 0.2993498525), fourier_srgb),
        dot(vec3f(0.2035160895, 0.0770505049, -0.2808208130), fourier_srgb),
        dot(vec3f(0.1563903497, -0.3230828819, 0.1668540863), fourier_srgb)
    );
}

// Prepares evaluation of a reflectance spectrum at specific wavelengths.
// trig_moments are three real, bounded trigonometric moments, i.e. Fourier
// coefficients of a reflectance spectrum (from fourier_srgb_to_fourier()).
// Returns three Lagrange multipliers that should be fed into
// eval_reflectance_real_lagrange_3() to evaluate the spectrum.
// Implements the algorithm described at the end of Sec. 3.6:
// https://doi.org/10.1145/3306346.3322964
fn prep_reflectance_real_lagrange_biased_3(trig_moments: vec3f) -> vec3f {
    var moments = trig_moments;
    moments[0] = clamp(moments[0], 0.0001, 0.9999);
    var exp_moments = trig_to_exp_moments_real_3(moments);
    var eval_poly = levinson_3_biased(&exp_moments);
    eval_poly[0] *= 6.28318531;
    eval_poly[1] *= 6.28318531;
    eval_poly[2] *= 6.28318531;
    let autocorrelation = real_autocorrelation_3(eval_poly);
    exp_moments[0] *= 0.5;
    let normalization_factor = 1.0 / (3.14159265 * eval_poly[0].x);
    return normalization_factor * imag_correlation_3(autocorrelation, exp_moments);
}

// Evaluates a reflectance spectrum at the given phase (which is a warped
// version of the wavelength) given Lagrange multipliers from
// prep_reflectance_real_lagrange_biased_3().
fn eval_reflectance_real_lagrange_3(phase: f32, lagranges: vec3f) -> f32 {
    let conj_circle_point = vec2f(cos(-phase), sin(-phase));
    let lagrange_series = eval_fourier_series_real_3(conj_circle_point, lagranges);
    return fma(atan(lagrange_series), 0.318309886, 0.5);
}
