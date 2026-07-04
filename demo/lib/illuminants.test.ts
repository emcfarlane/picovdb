// Verifies the illuminant wavelength-sampling LUTs. The key test mirrors the
// shader's estimator exactly: sweeping all texels of the illuminant-E LUT and
// averaging a(phase) * texel.rgb must reproduce the input sRGB color of a
// Fourier-sRGB material (white-balanced), because the upsampling basis is
// built on illuminant E.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildIlluminantLut,
  illuminantE,
  illuminantSodium,
  packFloat16,
} from "./illuminants.ts";
import {
  fourierSrgbToFourier,
  prepReflectanceRealLagrangeBiased3,
  evalReflectanceRealLagrange3,
  srgbToLinear,
  type Vec3,
} from "./spectra.ts";

// Subset of the golden vectors from spectra.test.ts (sRGB -> Fourier sRGB u8
// via the reference 256^3 LUT).
const MATERIALS: { srgb: Vec3; fourierSrgbU8: Vec3 }[] = [
  { srgb: [255, 255, 255], fourierSrgbU8: [254, 254, 254] },
  { srgb: [128, 128, 128], fourierSrgbU8: [128, 128, 128] },
  { srgb: [200, 50, 40], fourierSrgbU8: [197, 53, 51] },
  { srgb: [50, 180, 60], fourierSrgbU8: [54, 174, 78] },
  { srgb: [40, 60, 190], fourierSrgbU8: [50, 74, 183] },
];

function lagrangesFor(fourierSrgbU8: Vec3): Vec3 {
  const linear = fourierSrgbU8.map((v) => srgbToLinear(v / 255)) as Vec3;
  return prepReflectanceRealLagrangeBiased3(fourierSrgbToFourier(linear));
}

// The shader's estimator: mean over texels of a(phase) * rgb. lagranges=null
// means a perfect white surface (a = 1).
function estimate(lut: ReturnType<typeof buildIlluminantLut>, lagranges: Vec3 | null): Vec3 {
  const out: Vec3 = [0, 0, 0];
  const n = lut.rgbAndPhase.length / 4;
  for (let i = 0; i < n; ++i) {
    const a = lagranges
      ? evalReflectanceRealLagrange3(lut.rgbAndPhase[i * 4 + 3], lagranges)
      : 1.0;
    for (let c = 0; c < 3; ++c) out[c] += a * lut.rgbAndPhase[i * 4 + c];
  }
  for (let c = 0; c < 3; ++c) out[c] /= n;
  return out;
}

test("illuminant E LUT reproduces material colors (white-balanced)", () => {
  const lut = buildIlluminantLut(illuminantE);
  const white = estimate(lut, null);
  for (const m of MATERIALS) {
    const out = estimate(lut, lagrangesFor(m.fourierSrgbU8));
    for (let c = 0; c < 3; ++c) {
      const ref = srgbToLinear(m.srgb[c] / 255);
      assert.ok(Math.abs(out[c] / white[c] - ref) < 0.015,
        `channel ${c} of sRGB ${m.srgb}: ${out[c] / white[c]} !~ ${ref}`);
    }
  }
});

test("illuminant E is white-balanced to neutral", () => {
  const lut = buildIlluminantLut(illuminantE);
  const white = estimate(lut, null);
  for (let c = 0; c < 3; ++c) {
    // A perfect white surface under E must render as sRGB white
    assert.ok(Math.abs(white[c] - 1.0) < 1e-6, `white estimate channel ${c}: ${white[c]}`);
    assert.ok(Math.abs(lut.totalRgb[c] - white[c]) < 1e-9, `totalRgb[${c}]`);
  }
});

test("sodium light collapses surface hue", () => {
  const lut = buildIlluminantLut(illuminantSodium);
  // Two very different surfaces must reflect (near-)parallel RGB directions
  // under (near-)monochromatic light — only the magnitude may differ.
  const dirs = [MATERIALS[2], MATERIALS[4]].map((m) => {
    const rgb = estimate(lut, lagrangesFor(m.fourierSrgbU8));
    const len = Math.hypot(...rgb);
    return rgb.map((v) => v / len);
  });
  const cosine = dirs[0][0] * dirs[1][0] + dirs[0][1] * dirs[1][1] + dirs[0][2] * dirs[1][2];
  assert.ok(cosine > 0.9999, `hue directions diverge: cos = ${cosine}`);
});

test("float16 packing round-trips within half precision", () => {
  const values = new Float32Array([0, 1, -1, 0.5, 2.25, -3.14159265, 65504, 1e-8, 0.000061, 1000.5]);
  const packed = packFloat16(values);
  for (let i = 0; i < values.length; ++i) {
    // Decode binary16
    const bits = packed[i];
    const sign = bits & 0x8000 ? -1 : 1;
    const exp = (bits >> 10) & 0x1f;
    const mant = bits & 0x3ff;
    const decoded =
      exp === 0 ? sign * mant * 2 ** -24 :
      exp === 31 ? sign * Infinity :
      sign * (1 + mant / 1024) * 2 ** (exp - 15);
    const scale = Math.max(Math.abs(values[i]), 6.1e-5);
    assert.ok(Math.abs(decoded - values[i]) <= scale * 2 ** -10,
      `f16 round trip of ${values[i]} gave ${decoded}`);
  }
});
