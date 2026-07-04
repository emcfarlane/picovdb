// Verifies the downsampled Fourier LUT against the golden vectors from the
// full 256^3 table, and end-to-end: an arbitrary color converted through the
// LUT must survive the spectral round trip (upsample -> integrate under
// illuminant E) back to itself.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseFourierLut, srgbToFourierSrgb } from "./fourier_lut.ts";
import {
  fourierSrgbToFourier,
  prepReflectanceRealLagrangeBiased3,
  evalReflectanceRealLagrange3,
  srgbToLinear,
  type Vec3,
} from "./spectra.ts";
import { buildIlluminantLut, illuminantE } from "./illuminants.ts";

const lutPath = fileURLToPath(new URL("../srgb_to_fourier_srgb.lut", import.meta.url));
const lut = parseFourierLut(readFileSync(lutPath).buffer as ArrayBuffer);

// (input sRGB u8, Fourier sRGB u8 from the full 256^3 LUT) — the goldens of
// spectra.test.ts.
const GOLDENS: { srgb: Vec3; fourierSrgbU8: Vec3 }[] = [
  { srgb: [255, 255, 255], fourierSrgbU8: [254, 254, 254] },
  { srgb: [128, 128, 128], fourierSrgbU8: [128, 128, 128] },
  { srgb: [10, 10, 10], fourierSrgbU8: [10, 10, 10] },
  { srgb: [200, 50, 40], fourierSrgbU8: [197, 53, 51] },
  { srgb: [50, 180, 60], fourierSrgbU8: [54, 174, 78] },
  { srgb: [40, 60, 190], fourierSrgbU8: [50, 74, 183] },
  { srgb: [220, 210, 50], fourierSrgbU8: [218, 203, 72] },
  { srgb: [210, 160, 130], fourierSrgbU8: [208, 158, 132] },
  { srgb: [255, 0, 0], fourierSrgbU8: [251, 7, 35] },
  { srgb: [0, 255, 0], fourierSrgbU8: [46, 246, 82] },
  { srgb: [0, 0, 255], fourierSrgbU8: [21, 77, 244] },
  { srgb: [64, 0, 128], fourierSrgbU8: [68, 3, 121] },
];

test("downsampled LUT matches the full table within trilerp error", () => {
  for (const g of GOLDENS) {
    const linear = g.srgb.map((v) => srgbToLinear(v / 255)) as Vec3;
    const fourier = srgbToFourierSrgb(lut, linear);
    for (let c = 0; c < 3; ++c) {
      // Compare in the encoded u8 domain the goldens are expressed in
      const expected = g.fourierSrgbU8[c];
      const gotEncoded = 255 * (fourier[c] <= 0.0031308
        ? fourier[c] * 12.92
        : 1.055 * Math.pow(fourier[c], 1 / 2.4) - 0.055);
      assert.ok(Math.abs(gotEncoded - expected) <= 6,
        `channel ${c} of sRGB ${g.srgb}: ${gotEncoded} !~ ${expected}`);
    }
  }
});

test("arbitrary colors round-trip through LUT + spectrum under E", () => {
  const illum = buildIlluminantLut(illuminantE);
  const texels = illum.rgbAndPhase.length / 4;
  const colors: Vec3[] = [
    [0.7, 0.3, 0.15], [0.05, 0.4, 0.6], [0.9, 0.85, 0.7], [0.33, 0.33, 0.33],
    [0.6, 0.05, 0.4], [0.1, 0.55, 0.25],
  ];
  for (const linear of colors) {
    const lagranges = prepReflectanceRealLagrangeBiased3(
      fourierSrgbToFourier(srgbToFourierSrgb(lut, linear)));
    const out: Vec3 = [0, 0, 0];
    for (let i = 0; i < texels; ++i) {
      const a = evalReflectanceRealLagrange3(illum.rgbAndPhase[i * 4 + 3], lagranges);
      for (let c = 0; c < 3; ++c) out[c] += a * illum.rgbAndPhase[i * 4 + c];
    }
    for (let c = 0; c < 3; ++c) {
      out[c] /= texels;
      assert.ok(Math.abs(out[c] - linear[c]) < 0.03,
        `round trip channel ${c} of ${linear}: ${out[c]} !~ ${linear[c]}`);
    }
  }
});
