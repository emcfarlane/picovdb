// Verifies the environment-map luminance CDFs: the implied solid-angle pdf
// must integrate to 1 over the sphere, and importance sampling must estimate
// the map's average radiance correctly (the same estimator the shader uses).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseEnv, buildEnvCdfs, envTexelPdf, fromFloat16Bits } from "./env.ts";

const envPath = fileURLToPath(new URL("../studio_small_03.env", import.meta.url));
const env = parseEnv(readFileSync(envPath).buffer as ArrayBuffer);
const cdfs = buildEnvCdfs(env);

test("environment asset parses", () => {
  assert.equal(env.width, 1024);
  assert.equal(env.height, 512);
  assert.ok(env.avgRgb[0] > 1 && env.avgRgb[0] < 3);
});

test("env sampling pdf integrates to 1 over the sphere", () => {
  const { width: w, height: h } = env;
  let integral = 0;
  for (let y = 0; y < h; ++y) {
    const sinTheta = Math.sin(((y + 0.5) / h) * Math.PI);
    const texelSolidAngle = (2 * Math.PI * Math.PI * sinTheta) / (w * h);
    for (let x = 0; x < w; ++x) {
      integral += envTexelPdf(cdfs, w, h, x, y) * texelSolidAngle;
    }
  }
  assert.ok(Math.abs(integral - 1) < 0.01, `pdf integral = ${integral}`);
});

test("importance sampling estimates the map's mean radiance", () => {
  const { width: w, height: h, rgba16 } = env;
  // Reference: integral of radiance over the sphere / 4pi
  let ref = 0;
  for (let y = 0; y < h; ++y) {
    const sinTheta = Math.sin(((y + 0.5) / h) * Math.PI);
    const texelSolidAngle = (2 * Math.PI * Math.PI * sinTheta) / (w * h);
    for (let x = 0; x < w; ++x) {
      const i = (y * w + x) * 4;
      const lum = (fromFloat16Bits(rgba16[i]) + fromFloat16Bits(rgba16[i + 1]) + fromFloat16Bits(rgba16[i + 2])) / 3;
      ref += lum * texelSolidAngle;
    }
  }
  ref /= 4 * Math.PI;
  // Monte Carlo with the CDFs (inverse-CDF sampling, deterministic stratified u)
  const N = 20000;
  let estimate = 0;
  for (let s = 0; s < N; ++s) {
    const u1 = (s + 0.5) / N;
    const u2 = ((s * 0.754877666) % 1);
    // marginal row
    let y = cdfs.marginal.findIndex(v => v >= u1);
    if (y < 0) y = h - 1;
    // conditional column
    let x = 0, lo = 0, hi = w - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdfs.conditional[y * w + mid] < u2) lo = mid + 1; else hi = mid;
    }
    x = lo;
    const pdf = envTexelPdf(cdfs, w, h, x, y);
    if (pdf <= 0) continue;
    const i = (y * w + x) * 4;
    const lum = (fromFloat16Bits(rgba16[i]) + fromFloat16Bits(rgba16[i + 1]) + fromFloat16Bits(rgba16[i + 2])) / 3;
    estimate += lum / pdf;
  }
  estimate /= N * 4 * Math.PI;
  assert.ok(Math.abs(estimate - ref) / ref < 0.02,
    `estimate ${estimate} vs reference ${ref}`);
});
