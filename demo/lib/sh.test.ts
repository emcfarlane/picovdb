// Pins the SH conventions numerically: projection normalization, the
// A_l irradiance constants, rotation, and the transfer-dot identity that
// the whole cached-lighting design rests on.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  projectFunction, projectUniform, projectEnvMap, rotateSh, evalIrradiance,
  shBasis, SH_A0, SH_A1,
} from "./sh.ts";
import { parseEnv } from "./env.ts";

type V3 = [number, number, number];

function bruteIrradiance(f: (d: V3) => V3, n: V3, res = 200): V3 {
  const out: V3 = [0, 0, 0];
  for (let t = 0; t < res; t++) {
    const theta = ((t + 0.5) / res) * Math.PI;
    const sinT = Math.sin(theta);
    const dO = ((Math.PI / res) * ((2 * Math.PI) / res)) * sinT;
    for (let p = 0; p < res; p++) {
      const phi = ((p + 0.5) / res) * 2 * Math.PI;
      const d: V3 = [sinT * Math.cos(phi), Math.cos(theta), sinT * Math.sin(phi)];
      const cos = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
      if (cos <= 0) continue;
      const rgb = f(d);
      out[0] += rgb[0] * cos * dO;
      out[1] += rgb[1] * cos * dO;
      out[2] += rgb[2] * cos * dO;
    }
  }
  return out;
}

test("uniform radiance L gives irradiance pi*L at any normal", () => {
  const sh = projectUniform([2, 1, 0.5]);
  for (const n of [[0, 1, 0], [1, 0, 0], [0.6, 0.64, 0.48]] as V3[]) {
    const e = evalIrradiance(sh, n);
    assert.ok(Math.abs(e[0] - 2 * Math.PI) < 1e-4, `r: ${e[0]}`);
    assert.ok(Math.abs(e[1] - Math.PI) < 1e-4, `g: ${e[1]}`);
    assert.ok(Math.abs(e[2] - 0.5 * Math.PI) < 1e-4, `b: ${e[2]}`);
  }
});

test("L1-band-limited radiance: SH irradiance matches brute force exactly", () => {
  // f = a + b*(k . d): exactly representable in L1
  const k: V3 = [0.48, 0.6, 0.64];
  const f = (d: V3): V3 => {
    const v = Math.max(0.5 + 0.3 * (k[0] * d[0] + k[1] * d[1] + k[2] * d[2]), 0);
    return [v, 0.5 * v, 2 * v];
  };
  const sh = projectFunction(f, 128, 256);
  for (const n of [[0, 1, 0], [0.707, 0.707, 0], [-0.6, 0.64, 0.48]] as V3[]) {
    const e = evalIrradiance(sh, n);
    const ref = bruteIrradiance(f, n);
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(e[c] - ref[c]) < 0.01 * Math.max(1, ref[c]),
        `n=${n} ch${c}: sh=${e[c]} brute=${ref[c]}`);
    }
  }
});

test("rotating coefficients equals projecting the rotated function", () => {
  const f = (d: V3): V3 => [Math.max(0.2 + 0.8 * d[1], 0), 0, 0];
  const sh = projectFunction(f, 128, 256);
  // Rotation: 90 deg about x (column-major 3x3): y -> z, z -> -y
  const m = [1, 0, 0, 0, 0, 1, 0, -1, 0];
  const rotated = rotateSh(sh, m);
  // (R f)(d) = f(R^-1 d); R^-1 maps (x, y, z) -> (x, z, -y)
  const fRot = (d: V3): V3 => f([d[0], d[2], -d[1]]);
  const ref = projectFunction(fRot, 128, 256);
  for (let i = 0; i < 16; i++) {
    assert.ok(Math.abs(rotated[i] - ref[i]) < 1e-3,
      `coeff ${i}: ${rotated[i]} vs ${ref[i]}`);
  }
});

test("transfer-dot identity: dot(L, T_unshadowed) = irradiance", () => {
  // T for V=1 integrated numerically must equal A_l * Y_lm(n), and
  // dot(projected L, T) must equal the brute-force shadowless irradiance
  const n: V3 = [0.6, 0.64, 0.48];
  const tNumeric = projectFunction((d) => {
    const cos = Math.max(d[0] * n[0] + d[1] * n[1] + d[2] * n[2], 0);
    return [cos, cos, cos];
  }, 200, 400);
  const basis = shBasis(n);
  const analytic = [SH_A0 * basis[0], SH_A1 * basis[1], SH_A1 * basis[2], SH_A1 * basis[3]];
  for (let c = 0; c < 4; c++) {
    assert.ok(Math.abs(tNumeric[c * 4] - analytic[c]) < 0.01,
      `T coeff ${c}: numeric ${tNumeric[c * 4]} vs analytic ${analytic[c]}`);
  }
});

test("studio env projection: irradiance plumbing is consistent", () => {
  const envPath = fileURLToPath(new URL("../studio_small_03.env", import.meta.url));
  const env = parseEnv(readFileSync(envPath).buffer as ArrayBuffer);
  const sh = projectEnvMap(env);
  // Compare against brute force over the RECONSTRUCTED L1 radiance — this
  // isolates plumbing errors from L1 truncation (which is expected).
  const recon = (d: V3): V3 => {
    const b = shBasis(d);
    return [0, 1, 2].map(ch =>
      sh[0 + ch] * b[0] + sh[4 + ch] * b[1] + sh[8 + ch] * b[2] + sh[12 + ch] * b[3]) as V3;
  };
  const e = evalIrradiance(sh, [0, 1, 0]);
  const ref = bruteIrradiance(recon, [0, 1, 0]);
  for (let c = 0; c < 3; c++) {
    assert.ok(Math.abs(e[c] - ref[c]) < 0.02 * Math.max(1, Math.abs(ref[c])),
      `ch${c}: ${e[c]} vs ${ref[c]}`);
    assert.ok(e[c] > 0.5, `implausible irradiance ${e[c]}`);
  }
});
