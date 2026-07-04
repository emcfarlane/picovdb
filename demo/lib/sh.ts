// L1 spherical harmonics (4 coefficients: Y00, Y1-1, Y10, Y11) for the
// cached lighting path (docs/lighting-v2.md). Conventions:
// - Radiance L projected plainly: L_lm = integral L(w) Y_lm(w) dw.
// - Baked transfer T_lm = integral V(w) cos+(n.w) Y_lm(w) dw per voxel.
// - Shadowed irradiance E = dot(L_lm, T_lm) — no extra constants.
// - Unshadowed transfer has the closed form T_lm(n) = A_l Y_lm(n) with
//   A_0 = pi, A_1 = 2pi/3 (Ramamoorthi), used for the ground plane.
// Coefficient storage: 4 x vec4 (coefficient-major, rgb + pad) so buffers
// upload to the GPU directly.

import type { EnvMap } from './env.ts';
import { fromFloat16Bits } from './env.ts';

export const SH_Y00 = 0.28209479177; // 1/(2 sqrt(pi))
export const SH_Y1 = 0.48860251190;  // sqrt(3)/(2 sqrt(pi))
export const SH_A0 = Math.PI;
export const SH_A1 = (2 * Math.PI) / 3;

/** Basis values (Y00, Y1-1, Y10, Y11) for a unit direction. */
export function shBasis(dir: [number, number, number]): [number, number, number, number] {
  return [SH_Y00, SH_Y1 * dir[1], SH_Y1 * dir[2], SH_Y1 * dir[0]];
}

/** 16 floats: 4 coefficients x (r, g, b, pad). */
export type ShRgb = Float32Array;

export function shZero(): ShRgb {
  return new Float32Array(16);
}

/** Projects a spherical radiance function onto L1 SH by uniform-grid
 * quadrature over the equirect parameterization. */
export function projectFunction(
  radiance: (dir: [number, number, number]) => [number, number, number],
  resTheta = 64,
  resPhi = 128,
): ShRgb {
  const sh = shZero();
  for (let t = 0; t < resTheta; t++) {
    const theta = ((t + 0.5) / resTheta) * Math.PI;
    const sinTheta = Math.sin(theta);
    const dOmega = ((Math.PI / resTheta) * ((2 * Math.PI) / resPhi)) * sinTheta;
    for (let p = 0; p < resPhi; p++) {
      const phi = ((p + 0.5) / resPhi) * 2 * Math.PI - Math.PI;
      const dir: [number, number, number] = [
        sinTheta * Math.cos(phi), Math.cos(theta), sinTheta * Math.sin(phi)];
      const rgb = radiance(dir);
      const basis = shBasis(dir);
      for (let c = 0; c < 4; c++) {
        const w = basis[c] * dOmega;
        sh[c * 4] += rgb[0] * w;
        sh[c * 4 + 1] += rgb[1] * w;
        sh[c * 4 + 2] += rgb[2] * w;
      }
    }
  }
  return sh;
}

/** Projects an equirect environment map (matches the shader's mapping:
 * u = atan2(z,x)/2pi + 0.5, v = acos(y)/pi). */
export function projectEnvMap(env: EnvMap, intensity = 1): ShRgb {
  const { width: w, height: h, rgba16 } = env;
  const sh = shZero();
  for (let y = 0; y < h; y++) {
    const theta = ((y + 0.5) / h) * Math.PI;
    const sinTheta = Math.sin(theta);
    const dOmega = ((Math.PI / h) * ((2 * Math.PI) / w)) * sinTheta;
    for (let x = 0; x < w; x++) {
      const phi = ((x + 0.5) / w - 0.5) * 2 * Math.PI;
      const dir: [number, number, number] = [
        sinTheta * Math.cos(phi), Math.cos(theta), sinTheta * Math.sin(phi)];
      const i = (y * w + x) * 4;
      const basis = shBasis(dir);
      for (let c = 0; c < 4; c++) {
        const wgt = basis[c] * dOmega * intensity;
        sh[c * 4] += fromFloat16Bits(rgba16[i]) * wgt;
        sh[c * 4 + 1] += fromFloat16Bits(rgba16[i + 1]) * wgt;
        sh[c * 4 + 2] += fromFloat16Bits(rgba16[i + 2]) * wgt;
      }
    }
  }
  return sh;
}

/** Uniform dome of the given radiance: only the DC coefficient. */
export function projectUniform(rgb: [number, number, number]): ShRgb {
  const sh = shZero();
  const w = SH_Y00 * 4 * Math.PI;
  sh[0] = rgb[0] * w;
  sh[1] = rgb[1] * w;
  sh[2] = rgb[2] * w;
  return sh;
}

/** Rotates the L1 band by a 3x3 rotation matrix (column-major, applied to
 * the (x, y, z) coefficient triplet; Y00 is invariant). */
export function rotateSh(sh: ShRgb, m: ArrayLike<number>): ShRgb {
  const out = new Float32Array(sh);
  for (let ch = 0; ch < 3; ch++) {
    // Coefficient triplet as a vector: x = Y11, y = Y1-1, z = Y10
    const x = sh[3 * 4 + ch], y = sh[1 * 4 + ch], z = sh[2 * 4 + ch];
    const rx = m[0] * x + m[3] * y + m[6] * z;
    const ry = m[1] * x + m[4] * y + m[7] * z;
    const rz = m[2] * x + m[5] * y + m[8] * z;
    out[3 * 4 + ch] = rx;
    out[1 * 4 + ch] = ry;
    out[2 * 4 + ch] = rz;
  }
  return out;
}

/** Unshadowed irradiance at normal n: E = sum A_l L_lm Y_lm(n). */
export function evalIrradiance(sh: ShRgb, n: [number, number, number]): [number, number, number] {
  const basis = shBasis(n);
  const weights = [SH_A0 * basis[0], SH_A1 * basis[1], SH_A1 * basis[2], SH_A1 * basis[3]];
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 4; c++) {
    out[0] += sh[c * 4] * weights[c];
    out[1] += sh[c * 4 + 1] * weights[c];
    out[2] += sh[c * 4 + 2] * weights[c];
  }
  return out;
}
