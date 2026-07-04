// Verifies the moment-based spectra implementation against golden vectors
// extracted from the reference implementation (Peters' path_tracer, branch
// "spectral"): sRGB colors were pushed through the original 256^3
// srgb_to_fourier_srgb.dat LUT, and Lagrange multipliers / reflectance
// probes were computed with an independent (complex-number) implementation
// of the same algorithm.
//
// Run: node --test demo/lib/
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fourierSrgbToFourier,
  prepReflectanceRealLagrangeBiased3,
  evalReflectanceRealLagrange3,
  srgbToLinear,
  wavelengthToPhase,
  type Vec3,
} from "./spectra.ts";

// Each golden: input sRGB u8, its Fourier-sRGB u8 triple (LUT output, as a
// texture would store it), the expected Lagrange multipliers, and expected
// reflectance at phases [-3.0, -2.4, -1.8, -1.2, -0.6, -0.1].
const PROBE_PHASES = [-3.0, -2.4, -1.8, -1.2, -0.6, -0.1];
const GOLDENS: {
  srgb: Vec3;
  fourierSrgbU8: Vec3;
  lagranges: Vec3;
  probes: number[];
}[] = [
  { srgb: [255, 255, 255], fourierSrgbU8: [254, 254, 254],
    lagranges: [45.60245789561006, -1.7175543869457164, 1.1008755684536538],
    probes: [0.9937737392033292, 0.9934145095155169, 0.9928334387424023, 0.9925527540228, 0.992694758895464, 0.9928227577611486] },
  { srgb: [128, 128, 128], fourierSrgbU8: [128, 128, 128],
    lagranges: [-1.2380279391768252, -0.0004365143699040086, 0.0002773838231022795],
    probes: [0.21644827851578052, 0.2163596272934657, 0.2162349989804005, 0.21618144964005576, 0.21620731299033413, 0.21623175500249525] },
  { srgb: [10, 10, 10], fourierSrgbU8: [10, 10, 10],
    lagranges: [-104.66740670453373, -0.026550759200012698, 0.016871519976640926],
    probes: [0.0030435341089858903, 0.0030422873952377083, 0.003040535002671929, 0.0030397821394991253, 0.0030401457371176943, 0.0030404893707108305] },
  { srgb: [200, 50, 40], fourierSrgbU8: [197, 53, 51],
    lagranges: [-5.6141782087051615, 2.40452363830347, 0.8436863178433577],
    probes: [0.03620088422917578, 0.03517407241457071, 0.03853461602095942, 0.061445671584671735, 0.2447308159009564, 0.7194908721535094] },
  { srgb: [50, 180, 60], fourierSrgbU8: [54, 174, 78],
    lagranges: [-2.2364457600653433, 0.42713882713130685, -1.2667232810031313],
    probes: [0.05709965542717532, 0.09968559572676383, 0.4499175710107078, 0.4813226514356634, 0.12338021153439233, 0.0805023277727222] },
  { srgb: [40, 60, 190], fourierSrgbU8: [50, 74, 183],
    lagranges: [-5.20877714063048, -3.1899001247682213, -0.43647060209238026],
    probes: [0.583646820634933, 0.33252671122717026, 0.10317114781148623, 0.04596501236176698, 0.029414876668424572, 0.025589556972093586] },
  { srgb: [220, 210, 50], fourierSrgbU8: [218, 203, 72],
    lagranges: [-0.6394993806020336, 1.0884095962279405, -0.6118815057877479],
    probes: [0.07855329997455301, 0.1279774243295121, 0.48833678386709534, 0.7580167152303042, 0.6973010988436514, 0.6006203083664958] },
  { srgb: [210, 160, 130], fourierSrgbU8: [208, 158, 132],
    lagranges: [-0.4895138349222907, 0.35072853475608684, 0.07263995158319168],
    probes: [0.2430793033315649, 0.2509494705344496, 0.28930801849557425, 0.39497510277531817, 0.5449207174816483, 0.607400200520356] },
  { srgb: [255, 0, 0], fourierSrgbU8: [251, 7, 35],
    lagranges: [-15.247843113083643, 7.06403742548029, 10.538288134444674],
    probes: [0.035233271016987255, 0.0133543875065667, 0.008518410146765465, 0.012393729389228059, 0.9229429134456079, 0.9836623486280602] },
  { srgb: [0, 255, 0], fourierSrgbU8: [46, 246, 82],
    lagranges: [-1.3980846323266498, 2.009798872727904, -5.467256082896184],
    probes: [0.020022734901307848, 0.059154858023239276, 0.9577756547088458, 0.9610027477192833, 0.14490648873944373, 0.03902755895842153] },
  { srgb: [0, 0, 255], fourierSrgbU8: [21, 77, 244],
    lagranges: [-14.528871892527574, -12.061742935202966, -1.4441866957517822],
    probes: [0.9519910451786084, 0.8978029639181038, 0.04890241295355047, 0.015045777979769637, 0.008967774393057959, 0.007694090847064183] },
  { srgb: [64, 0, 128], fourierSrgbU8: [68, 3, 121],
    lagranges: [-9.990460326497924, -2.0545744687429877, 2.9086727548566618],
    probes: [0.39659398290657866, 0.04895019718534699, 0.0222642138204574, 0.02015865815687834, 0.028160463935552216, 0.037816003413021215] },
];

function lagrangesFor(fourierSrgbU8: Vec3): Vec3 {
  const linear = fourierSrgbU8.map((v) => srgbToLinear(v / 255)) as Vec3;
  return prepReflectanceRealLagrangeBiased3(fourierSrgbToFourier(linear));
}

function assertClose(actual: number, expected: number, tol: number, msg: string) {
  assert.ok(
    Math.abs(actual - expected) <= tol,
    `${msg}: ${actual} !~ ${expected} (tol ${tol})`,
  );
}

test("Lagrange multipliers match the reference implementation", () => {
  for (const g of GOLDENS) {
    const lag = lagrangesFor(g.fourierSrgbU8);
    for (let i = 0; i < 3; ++i) {
      assertClose(lag[i], g.lagranges[i], 1e-9 * (1 + Math.abs(g.lagranges[i])),
        `lagranges[${i}] for sRGB ${g.srgb}`);
    }
  }
});

test("reflectance probes match the reference implementation", () => {
  for (const g of GOLDENS) {
    const lag = lagrangesFor(g.fourierSrgbU8);
    g.probes.forEach((expected, i) => {
      const a = evalReflectanceRealLagrange3(PROBE_PHASES[i], lag);
      assertClose(a, expected, 1e-9, `a(${PROBE_PHASES[i]}) for sRGB ${g.srgb}`);
      assert.ok(a >= 0 && a <= 1, `reflectance in [0,1] for sRGB ${g.srgb}`);
    });
  }
});

import { CMF, XYZ_TO_REC709 } from "./illuminants.ts";

// Integrates a reflectance spectrum (or a(λ)=1 when lagranges is null)
// against the CMFs under illuminant E and projects to linear sRGB.
// 1 nm steps, CMFs treated as piecewise constant over their 5 nm bins,
// matching the reference tooling.
function integrateToRgb(lagranges: Vec3 | null): Vec3 {
  const rgb: Vec3 = [0, 0, 0];
  for (let lam = 357.5; lam < 832.5; lam += 1.0) {
    const bin = Math.min(94, Math.max(0, Math.round((lam - 360) / 5)));
    const a = lagranges
      ? evalReflectanceRealLagrange3(wavelengthToPhase(lam), lagranges)
      : 1.0;
    for (let r = 0; r < 3; ++r) {
      for (let c = 0; c < 3; ++c) {
        rgb[r] += a * XYZ_TO_REC709[r][c] * CMF[bin * 3 + c];
      }
    }
  }
  return rgb;
}

test("round trip: Fourier sRGB spectrum integrates back to the input color", () => {
  // White-balanced against a perfect reflector under illuminant E (RGB
  // rendering implicitly assumes the illuminant is the display whitepoint).
  const white = integrateToRgb(null);
  for (const g of GOLDENS) {
    const out = integrateToRgb(lagrangesFor(g.fourierSrgbU8));
    for (let i = 0; i < 3; ++i) {
      // Residual error comes from the u8 LUT quantization and the 3-moment
      // representation (worst measured: 0.0074 for white).
      assertClose(out[i] / white[i], srgbToLinear(g.srgb[i] / 255), 0.01,
        `round trip channel ${i} for sRGB ${g.srgb}`);
    }
  }
});
