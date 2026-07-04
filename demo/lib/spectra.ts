// CPU reference of demo/spectra.wgsl — moment-based reflectance spectra.
// Mirrors the WGSL line by line (complex numbers as [re, im] pairs) so the
// two implementations can be verified against shared golden vectors
// (see spectra.test.ts). Method by Christoph Peters, BSD-3; see spectra.wgsl
// for the full notice and paper references.

type Vec2 = [number, number];
export type Vec3 = [number, number, number];

function cconj(z: Vec2): Vec2 {
  return [z[0], -z[1]];
}

function cmul(lhs: Vec2, rhs: Vec2): Vec2 {
  return [lhs[0] * rhs[0] - lhs[1] * rhs[1], lhs[0] * rhs[1] + lhs[1] * rhs[0]];
}

function cfma(a: Vec2, b: Vec2, c: Vec2): Vec2 {
  return [a[0] * b[0] - a[1] * b[1] + c[0], a[0] * b[1] + a[1] * b[0] + c[1]];
}

function cabsSq(z: Vec2): number {
  return z[0] * z[0] + z[1] * z[1];
}

function trigToExpMomentsReal3(trigMoments: Vec3): [Vec2, Vec2, Vec2] {
  const moment0Phase = 3.14159265 * trigMoments[0] - 1.57079633;
  let e0: Vec2 = [Math.cos(moment0Phase), Math.sin(moment0Phase)];
  e0 = [0.0795774715 * e0[0], 0.0795774715 * e0[1]];
  const s1 = trigMoments[1] * 6.28318531;
  const e1: Vec2 = [s1 * -e0[1], s1 * e0[0]];
  const s2 = trigMoments[2] * 6.28318531;
  const s3 = trigMoments[1] * 3.14159265;
  const e2: Vec2 = [s2 * -e0[1] + s3 * -e1[1], s2 * e0[0] + s3 * e1[0]];
  return [[2.0 * e0[0], 2.0 * e0[1]], e1, e2];
}

// Levinson's algorithm with biasing; may modify firstColumn (biasing).
function levinson3Biased(firstColumn: [Vec2, Vec2, Vec2]): [Vec2, Vec2, Vec2] {
  let oneMinusBias = 0.9999;
  let correctedFactor = 1.0 / (1.0 - 0.9999 * 0.9999);
  const sol: [Vec2, Vec2, Vec2] = [[1.0 / firstColumn[0][0], 0.0], [0, 0], [0, 0]];
  let scaledCenter: Vec2 = [0.0, 0.0];
  let dotProduct: Vec2 = [
    sol[0][0] * firstColumn[1][0] + scaledCenter[0],
    sol[0][0] * firstColumn[1][1] + scaledCenter[1],
  ];
  let dotSq = cabsSq(dotProduct);
  let factor = 1.0 / (1.0 - dotSq);
  if (factor < 0.0) {
    const s = oneMinusBias / Math.sqrt(dotSq);
    dotProduct = [s * dotProduct[0], s * dotProduct[1]];
    const inv = 1.0 / sol[0][0];
    firstColumn[1] = [(dotProduct[0] - scaledCenter[0]) * inv, (dotProduct[1] - scaledCenter[1]) * inv];
    factor = correctedFactor;
    oneMinusBias = 0.0;
    correctedFactor = 1.0;
  }
  let flipped1: Vec2 = [sol[0][0], 0.0];
  sol[0] = [factor * sol[0][0], 0.0];
  sol[1] = [factor * -flipped1[0] * dotProduct[0], factor * -flipped1[0] * dotProduct[1]];
  scaledCenter = cmul(sol[1], firstColumn[1]);
  dotProduct = [
    sol[0][0] * firstColumn[2][0] + scaledCenter[0],
    sol[0][0] * firstColumn[2][1] + scaledCenter[1],
  ];
  dotSq = cabsSq(dotProduct);
  factor = 1.0 / (1.0 - dotSq);
  if (factor < 0.0) {
    const s = oneMinusBias / Math.sqrt(dotSq);
    dotProduct = [s * dotProduct[0], s * dotProduct[1]];
    const inv = 1.0 / sol[0][0];
    firstColumn[2] = [(dotProduct[0] - scaledCenter[0]) * inv, (dotProduct[1] - scaledCenter[1]) * inv];
    factor = correctedFactor;
  }
  flipped1 = cconj(sol[1]);
  const flipped2: Vec2 = [sol[0][0], 0.0];
  sol[0] = [factor * sol[0][0], 0.0];
  const t = cfma([-flipped1[0], -flipped1[1]], dotProduct, sol[1]);
  sol[1] = [factor * t[0], factor * t[1]];
  sol[2] = [factor * -flipped2[0] * dotProduct[0], factor * -flipped2[0] * dotProduct[1]];
  return sol;
}

function realAutocorrelation3(signal: [Vec2, Vec2, Vec2]): [Vec2, Vec2, Vec2] {
  return [
    cfma(signal[0], cconj(signal[0]), cfma(signal[1], cconj(signal[1]), cmul(signal[2], cconj(signal[2])))),
    cfma(signal[0], cconj(signal[1]), cmul(signal[1], cconj(signal[2]))),
    cmul(signal[0], cconj(signal[2])),
  ];
}

function imagCorrelation3(lhs: [Vec2, Vec2, Vec2], rhs: [Vec2, Vec2, Vec2]): Vec3 {
  return [
    lhs[0][0] * rhs[0][1] + lhs[0][1] * rhs[0][0] + lhs[1][0] * rhs[1][1] + lhs[1][1] * rhs[1][0] + lhs[2][0] * rhs[2][1] + lhs[2][1] * rhs[2][0],
    lhs[1][0] * rhs[0][1] + lhs[1][1] * rhs[0][0] + lhs[2][0] * rhs[1][1] + lhs[2][1] * rhs[1][0],
    lhs[2][0] * rhs[0][1] + lhs[2][1] * rhs[0][0],
  ];
}

function evalFourierSeriesReal3(point: Vec2, fouriers: Vec3): number {
  const cos1 = point[0];
  const cos2 = point[0] * point[0] - point[1] * point[1];
  return 2.0 * (fouriers[1] * cos1 + fouriers[2] * cos2 + 0.5 * fouriers[0]);
}

/** Turns linear Fourier sRGB (i.e. after the usual sRGB EOTF decode) into
 * three bounded trigonometric moments. */
export function fourierSrgbToFourier(fourierSrgb: Vec3): Vec3 {
  return [
    0.2276800310 * fourierSrgb[0] + 0.4748793271 * fourierSrgb[1] + 0.2993498525 * fourierSrgb[2],
    0.2035160895 * fourierSrgb[0] + 0.0770505049 * fourierSrgb[1] - 0.2808208130 * fourierSrgb[2],
    0.1563903497 * fourierSrgb[0] - 0.3230828819 * fourierSrgb[1] + 0.1668540863 * fourierSrgb[2],
  ];
}

/** Converts trigonometric moments to Lagrange multipliers for
 * evalReflectanceRealLagrange3(). Done once per material/hit. */
export function prepReflectanceRealLagrangeBiased3(trigMoments: Vec3): Vec3 {
  const moments: Vec3 = [
    Math.min(Math.max(trigMoments[0], 0.0001), 0.9999),
    trigMoments[1],
    trigMoments[2],
  ];
  const expMoments = trigToExpMomentsReal3(moments);
  const evalPoly = levinson3Biased(expMoments);
  for (let i = 0; i < 3; ++i) {
    evalPoly[i] = [evalPoly[i][0] * 6.28318531, evalPoly[i][1] * 6.28318531];
  }
  const autocorrelation = realAutocorrelation3(evalPoly);
  expMoments[0] = [expMoments[0][0] * 0.5, expMoments[0][1] * 0.5];
  const normalizationFactor = 1.0 / (3.14159265 * evalPoly[0][0]);
  const lag = imagCorrelation3(autocorrelation, expMoments);
  return [normalizationFactor * lag[0], normalizationFactor * lag[1], normalizationFactor * lag[2]];
}

/** Evaluates a reflectance spectrum in [0, 1] at the given phase (a warped
 * wavelength, see wavelengthToPhase()). */
export function evalReflectanceRealLagrange3(phase: number, lagranges: Vec3): number {
  const conjCirclePoint: Vec2 = [Math.cos(-phase), Math.sin(-phase)];
  const lagrangeSeries = evalFourierSeriesReal3(conjCirclePoint, lagranges);
  return Math.atan(lagrangeSeries) * 0.318309886 + 0.5;
}

/** The sRGB EOTF (decode); Fourier sRGB textures/colors are sRGB-encoded. */
export function srgbToLinear(v: number): number {
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
}

// The XYZ warp mapping wavelengths (360..830 nm, 5 nm steps) to phases in
// [-pi, 0], from tools/illuminant_spectra.py in the reference implementation.
const WARP_PHASES = [
  -3.141592654, -3.141592654, -3.141592654, -3.141592654, -3.141591857,
  -3.141590597, -3.141590237, -3.141432053, -3.140119041, -3.137863071,
  -3.133438967, -3.123406739, -3.106095749, -3.073470612, -3.024748900,
  -2.963566246, -2.894461907, -2.819659701, -2.741784136, -2.660533432,
  -2.576526605, -2.490368187, -2.407962868, -2.334138406, -2.269339880,
  -2.213127747, -2.162806279, -2.114787412, -2.065873394, -2.012511127,
  -1.952877310, -1.886377224, -1.813129945, -1.735366957, -1.655108108,
  -1.573400329, -1.490781436, -1.407519056, -1.323814008, -1.239721795,
  -1.155352390, -1.071041833, -0.986956525, -0.903007113, -0.819061538,
  -0.735505101, -0.653346027, -0.573896987, -0.498725202, -0.428534515,
  -0.363884284, -0.304967687, -0.251925536, -0.205301867, -0.165356255,
  -0.131442191, -0.102998719, -0.079687644, -0.061092401, -0.046554594,
  -0.035419229, -0.027113640, -0.021085743, -0.016716885, -0.013468661,
  -0.011125245, -0.009497032, -0.008356318, -0.007571826, -0.006902676,
  -0.006366945, -0.005918355, -0.005533442, -0.005193920, -0.004886397,
  -0.004601975, -0.004334090, -0.004077698, -0.003829183, -0.003585923,
  -0.003346286, -0.003109231, -0.002873996, -0.002640047, -0.002406990,
  -0.002174598, -0.001942639, -0.001711031, -0.001479624, -0.001248405,
  -0.001017282, -0.000786134, -0.000557770, -0.000332262, 0.000000000,
];

/** Converts a wavelength in nm to a phase in [-pi, 0] using the XYZ warp. */
export function wavelengthToPhase(wavelength: number): number {
  const x = (wavelength - 360.0) / 5.0;
  const i = Math.floor(x);
  if (i < 0) return WARP_PHASES[0];
  if (i >= 94) return WARP_PHASES[94];
  const t = x - i;
  return WARP_PHASES[i] * (1 - t) + WARP_PHASES[i + 1] * t;
}
