// Illuminant spectra and the wavelength-sampling LUT for spectral rendering.
//
// Port of tools/illuminant_spectra.py from Peters' path tracer (spectral
// branch). Each illuminant becomes a small 1D texture whose texel i answers
// "given uniform random number u = (i + 0.5) / resolution, which wavelength
// do I sample, and what is its importance-weighted CMF response in linear
// sRGB?" — texels are (r, g, b, phase). CMFs, importance sampling and the
// XYZ -> Rec.709 projection are all baked in; the shader never sees a CMF.
//
// The illuminant flux itself cancels in the estimator (we importance-sample
// proportionally to flux x CMF weight), so a path's radiance estimate is
//   mean_i( throughput_i * texel_i.rgb ) * integral
// where `integral` is the illuminant's total flux from buildIlluminantLut().

import { wavelengthToPhase } from "./spectra.ts";

// CIE 1931 2-deg color matching functions, 360..830 nm in 5 nm steps
// (x,y,z triples). Treated as piecewise constant over their 5 nm bins,
// matching the reference tooling.
export const CMF_MIN_NM = 360;
export const CMF_STEP_NM = 5;
export const CMF = [
  0.000130, 0.000004, 0.000606, 0.000232, 0.000007, 0.001086, 0.000415, 0.000012, 0.001946,
  0.000742, 0.000022, 0.003486, 0.001368, 0.000039, 0.006450, 0.002236, 0.000064, 0.010550,
  0.004243, 0.000120, 0.020050, 0.007650, 0.000217, 0.036210, 0.014310, 0.000396, 0.067850,
  0.023190, 0.000640, 0.110200, 0.043510, 0.001210, 0.207400, 0.077630, 0.002180, 0.371300,
  0.134380, 0.004000, 0.645600, 0.214770, 0.007300, 1.039050, 0.283900, 0.011600, 1.385600,
  0.328500, 0.016840, 1.622960, 0.348280, 0.023000, 1.747060, 0.348060, 0.029800, 1.782600,
  0.336200, 0.038000, 1.772110, 0.318700, 0.048000, 1.744100, 0.290800, 0.060000, 1.669200,
  0.251100, 0.073900, 1.528100, 0.195360, 0.090980, 1.287640, 0.142100, 0.112600, 1.041900,
  0.095640, 0.139020, 0.812950, 0.057950, 0.169300, 0.616200, 0.032010, 0.208020, 0.465180,
  0.014700, 0.258600, 0.353300, 0.004900, 0.323000, 0.272000, 0.002400, 0.407300, 0.212300,
  0.009300, 0.503000, 0.158200, 0.029100, 0.608200, 0.111700, 0.063270, 0.710000, 0.078250,
  0.109600, 0.793200, 0.057250, 0.165500, 0.862000, 0.042160, 0.225750, 0.914850, 0.029840,
  0.290400, 0.954000, 0.020300, 0.359700, 0.980300, 0.013400, 0.433450, 0.994950, 0.008750,
  0.512050, 1.000000, 0.005750, 0.594500, 0.995000, 0.003900, 0.678400, 0.978600, 0.002750,
  0.762100, 0.952000, 0.002100, 0.842500, 0.915400, 0.001800, 0.916300, 0.870000, 0.001650,
  0.978600, 0.816300, 0.001400, 1.026300, 0.757000, 0.001100, 1.056700, 0.694900, 0.001000,
  1.062200, 0.631000, 0.000800, 1.045600, 0.566800, 0.000600, 1.002600, 0.503000, 0.000340,
  0.938400, 0.441200, 0.000240, 0.854450, 0.381000, 0.000190, 0.751400, 0.321000, 0.000100,
  0.642400, 0.265000, 0.000050, 0.541900, 0.217000, 0.000030, 0.447900, 0.175000, 0.000020,
  0.360800, 0.138200, 0.000010, 0.283500, 0.107000, 0.000000, 0.218700, 0.081600, 0.000000,
  0.164900, 0.061000, 0.000000, 0.121200, 0.044580, 0.000000, 0.087400, 0.032000, 0.000000,
  0.063600, 0.023200, 0.000000, 0.046770, 0.017000, 0.000000, 0.032900, 0.011920, 0.000000,
  0.022700, 0.008210, 0.000000, 0.015840, 0.005723, 0.000000, 0.011359, 0.004102, 0.000000,
  0.008111, 0.002929, 0.000000, 0.005790, 0.002091, 0.000000, 0.004109, 0.001484, 0.000000,
  0.002899, 0.001047, 0.000000, 0.002049, 0.000740, 0.000000, 0.001440, 0.000520, 0.000000,
  0.001000, 0.000361, 0.000000, 0.000690, 0.000249, 0.000000, 0.000476, 0.000172, 0.000000,
  0.000332, 0.000120, 0.000000, 0.000235, 0.000085, 0.000000, 0.000166, 0.000060, 0.000000,
  0.000117, 0.000042, 0.000000, 0.000083, 0.000030, 0.000000, 0.000059, 0.000021, 0.000000,
  0.000042, 0.000015, 0.000000, 0.000029, 0.000011, 0.000000, 0.000021, 0.000007, 0.000000,
  0.000015, 0.000005, 0.000000, 0.000010, 0.000004, 0.000000, 0.000007, 0.000003, 0.000000,
  0.000005, 0.000002, 0.000000, 0.000004, 0.000001, 0.000000, 0.000003, 0.000001, 0.000000,
  0.000002, 0.000001, 0.000000, 0.000001, 0.000000, 0.000000,
];
export const CMF_COUNT = CMF.length / 3; // 95

export const XYZ_TO_REC709 = [
  [+3.2406255, -1.5372080, -0.4986286],
  [-0.9689307, +1.8757561, +0.0415175],
  [+0.0557101, -0.2040211, +1.0569959],
];

// Linear-sRGB response of the CMFs at a wavelength (piecewise constant).
function cmfRgb(lambda: number): [number, number, number] {
  const bin = Math.min(CMF_COUNT - 1, Math.max(0, Math.round((lambda - CMF_MIN_NM) / CMF_STEP_NM)));
  const x = CMF[bin * 3], y = CMF[bin * 3 + 1], z = CMF[bin * 3 + 2];
  return [
    XYZ_TO_REC709[0][0] * x + XYZ_TO_REC709[0][1] * y + XYZ_TO_REC709[0][2] * z,
    XYZ_TO_REC709[1][0] * x + XYZ_TO_REC709[1][1] * y + XYZ_TO_REC709[1][2] * z,
    XYZ_TO_REC709[2][0] * x + XYZ_TO_REC709[2][1] * y + XYZ_TO_REC709[2][2] * z,
  ];
}

/** A relative spectral power distribution over the visible range (nm). */
export type Spd = (lambda: number) => number;

// --- Illuminants (license-clean: analytic or synthetic) -------------------
// Note: measured lamp spectra (e.g. the LSPDD database used by the reference
// implementation) are CC BY-NC-ND licensed, so none are bundled here; real
// measurements and CIE D65 can be loaded from user-supplied data later.

/** Illuminant E: equal energy. The Fourier-sRGB upsampling basis is built on
 * E, so this is the neutral "matches the RGB renderer" studio light. */
export const illuminantE: Spd = () => 1.0;

/** CIE standard illuminant A (incandescent): Planckian radiator at 2856 K
 * with c2 = 1.4388e-2 m*K. Relative SPD; scale is irrelevant. */
export const illuminantA: Spd = (lambda) => {
  const lm = lambda * 1e-9;
  return (1e-50 / (lm * lm * lm * lm * lm)) / (Math.exp(1.4388e-2 / (lm * 2856.0)) - 1.0);
};

/** Low-pressure sodium lamp: the 589.0/589.6 nm doublet, modeled as two
 * narrow Gaussians. Nearly monochromatic — overrides surface color. */
export const illuminantSodium: Spd = (lambda) => {
  const g = (c: number, s: number) => Math.exp(-0.5 * ((lambda - c) / s) ** 2);
  return g(589.0, 1.0) + g(589.6, 1.0);
};

/** Synthetic warm-white LED: 450 nm InGaN pump plus a broad phosphor hump.
 * Representative shape, not a measurement. */
export const illuminantLedWarm: Spd = (lambda) => {
  const g = (c: number, s: number) => Math.exp(-0.5 * ((lambda - c) / s) ** 2);
  return 0.25 * g(450, 11) + g(600, 55);
};

export const ILLUMINANTS: { name: string; spd: Spd }[] = [
  { name: "E (neutral)", spd: illuminantE },
  { name: "Incandescent (A)", spd: illuminantA },
  { name: "LED warm (synthetic)", spd: illuminantLedWarm },
  { name: "Sodium", spd: illuminantSodium },
];

export interface IlluminantLut {
  /** (r, g, b, phase) per texel; length resolution * 4. */
  rgbAndPhase: Float32Array;
  /** Aggregate linear-sRGB color of the illuminant, i.e. what a perfect
   * white surface reflects (= mean rgb over the texels). White-balanced so
   * illuminant E gives (1, 1, 1). */
  totalRgb: [number, number, number];
}

/** Builds the wavelength-sampling LUT for an illuminant: the inverse CDF of
 * flux(l) * |cmfRgb(l)|_1, with importance-weighted RGB and warped phase per
 * texel. Port of prepare_illuminant_spectrum(), with two normalizations on
 * top of the reference:
 * - flux is normalized away entirely (the shader's emission scale is a plain
 *   user intensity, identical across illuminants), and
 * - rgb is white-balanced against equal-energy white (illuminant E), so a
 *   white surface under E renders neutral on an sRGB display and the
 *   spectral render matches the RGB render exactly under E. Other
 *   illuminants keep their tint relative to E (sodium stays yellow). */
export function buildIlluminantLut(spd: Spd, resolution = 1024): IlluminantLut {
  const lut = buildIlluminantLutRaw(spd, resolution);
  const white = eWhite();
  const totalRgb: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < resolution; ++i) {
    for (let c = 0; c < 3; ++c) {
      lut.rgbAndPhase[i * 4 + c] /= white[c];
      totalRgb[c] += lut.rgbAndPhase[i * 4 + c];
    }
  }
  for (let c = 0; c < 3; ++c) totalRgb[c] /= resolution;
  return { rgbAndPhase: lut.rgbAndPhase, totalRgb };
}

// Mean rgb of the raw (unbalanced) illuminant-E LUT, the white-balance
// reference. Lazily computed once.
let eWhiteCache: [number, number, number] | null = null;
function eWhite(): [number, number, number] {
  if (!eWhiteCache) {
    eWhiteCache = buildIlluminantLutRaw(illuminantE, 1024).totalRgb;
  }
  return eWhiteCache;
}

function buildIlluminantLutRaw(spd: Spd, resolution: number): IlluminantLut {
  const lambdaMin = CMF_MIN_NM - 0.4995 * CMF_STEP_NM;
  const lambdaMax = CMF_MIN_NM + (CMF_COUNT - 1 + 0.4995) * CMF_STEP_NM;
  // Densely sample the combined importance flux(l) * |rgb(l)|_1 and its CDF.
  const denseCount = 100001;
  const denseStep = (lambdaMax - lambdaMin) / (denseCount - 1);
  const cdf = new Float64Array(denseCount);
  let cumulative = 0.0;
  for (let i = 0; i < denseCount; ++i) {
    const lambda = lambdaMin + i * denseStep;
    const rgb = cmfRgb(lambda);
    const importance = spd(lambda) * (Math.abs(rgb[0]) + Math.abs(rgb[1]) + Math.abs(rgb[2]));
    cumulative += importance;
    cdf[i] = cumulative;
  }
  // Normalization of the |rgb|_1 importance density (flux excluded — it
  // cancels against the illuminant in the integrand, as in the reference).
  let rgbImportanceIntegral = 0.0;
  for (let i = 0; i < denseCount; ++i) {
    const rgb = cmfRgb(lambdaMin + i * denseStep);
    rgbImportanceIntegral += Math.abs(rgb[0]) + Math.abs(rgb[1]) + Math.abs(rgb[2]);
  }
  rgbImportanceIntegral *= denseStep;

  const rgbAndPhase = new Float32Array(resolution * 4);
  const totalRgb: [number, number, number] = [0, 0, 0];
  let cdfIndex = 0;
  for (let i = 0; i < resolution; ++i) {
    const target = ((i + 0.5) / resolution) * cumulative;
    while (cdfIndex < denseCount - 1 && cdf[cdfIndex] < target) ++cdfIndex;
    // Linear interpolation within the CDF segment for the sampled wavelength
    const c1 = cdf[cdfIndex];
    const c0 = cdfIndex > 0 ? cdf[cdfIndex - 1] : 0.0;
    const t = c1 > c0 ? (target - c0) / (c1 - c0) : 0.5;
    const lambda = lambdaMin + (cdfIndex - 1 + t) * denseStep;
    const rgb = cmfRgb(lambda);
    const density = (Math.abs(rgb[0]) + Math.abs(rgb[1]) + Math.abs(rgb[2])) / rgbImportanceIntegral;
    for (let c = 0; c < 3; ++c) {
      const v = rgb[c] / density;
      rgbAndPhase[i * 4 + c] = v;
      totalRgb[c] += v;
    }
    rgbAndPhase[i * 4 + 3] = wavelengthToPhase(lambda);
  }
  for (let c = 0; c < 3; ++c) totalRgb[c] /= resolution;
  return { rgbAndPhase, totalRgb };
}

/** IEEE 754 binary16 bit pattern for a number (round-to-nearest-even),
 * for uploading LUTs as rgba16float. */
export function toFloat16Bits(value: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const em = x & 0x7fffffff;
  if (em >= 0x47800000) {
    // Overflow to inf (or propagate NaN)
    return sign | 0x7c00 | (em > 0x7f800000 ? 0x200 : 0);
  }
  if (em < 0x38800000) {
    // Subnormal or zero: shift mantissa (implicit leading 1) into units of
    // 2^-24, the f16 subnormal quantum
    const shift = 126 - (em >>> 23);
    if (shift > 24) return sign;
    const mant = (em & 0x7fffff) | 0x800000;
    const half = mant >> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1))) return sign | (half + 1);
    return sign | half;
  }
  // Normal: rebias exponent, round mantissa to 10 bits
  let h = ((em >>> 13) - 0x1c000) & 0x7fff;
  const rem = em & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h += 1;
  return sign | h;
}

/** Packs a Float32Array into binary16 for texture upload. */
export function packFloat16(values: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; ++i) out[i] = toFloat16Bits(values[i]);
  return out;
}
