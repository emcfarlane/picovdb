// L2 spherical harmonics (9 coefficients) for the cached lighting path
// (docs/lighting-v2.md). Conventions:
// - Radiance L projected plainly: L_lm = integral L(w) Y_lm(w) dw.
// - Baked transfer T_lm = integral V(w) cos+(n.w) Y_lm(w) dw per voxel.
// - Shadowed irradiance E = dot(L_lm, T_lm) — no extra constants.
// - Unshadowed transfer has the closed form T_lm(n) = A_l Y_lm(n) with
//   A_0 = pi, A_1 = 2pi/3, A_2 = pi/4 (Ramamoorthi).
// Coefficient order: [Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22],
// stored coefficient-major as 9 x vec4 (rgb + pad) for direct GPU upload.
//
// Band-2 rotation is built numerically per call: Y2(R u) = W Y2(u) for any
// u within the band, so W = B A^-1 from 5 sample directions (A^-1 is
// precomputed). No hand-derived Wigner matrices.

import type { EnvMap } from './env.ts';
import { fromFloat16Bits } from './env.ts';

export const SH_COEFFS = 9;
export const SH_Y00 = 0.28209479177; // 1/(2 sqrt(pi))
export const SH_Y1 = 0.48860251190;  // sqrt(3)/(2 sqrt(pi))
export const SH_Y2_2 = 1.09254843059; // sqrt(15)/(2 sqrt(pi)), for xy, yz, xz
export const SH_Y20 = 0.31539156525;  // sqrt(5)/(4 sqrt(pi)), for 3z^2-1
export const SH_Y22 = 0.54627421529;  // sqrt(15)/(4 sqrt(pi)), for x^2-y^2
export const SH_A0 = Math.PI;
export const SH_A1 = (2 * Math.PI) / 3;
export const SH_A2 = Math.PI / 4;
export const SH_A = [SH_A0, SH_A1, SH_A1, SH_A1, SH_A2, SH_A2, SH_A2, SH_A2, SH_A2];

/** Basis values for a unit direction, coefficient order as above. */
export function shBasis(dir: [number, number, number]): number[] {
  const [x, y, z] = dir;
  return [
    SH_Y00,
    SH_Y1 * y, SH_Y1 * z, SH_Y1 * x,
    SH_Y2_2 * x * y, SH_Y2_2 * y * z, SH_Y20 * (3 * z * z - 1),
    SH_Y2_2 * x * z, SH_Y22 * (x * x - y * y),
  ];
}

/** 36 floats: 9 coefficients x (r, g, b, pad). */
export type ShRgb = Float32Array;

export function shZero(): ShRgb {
  return new Float32Array(SH_COEFFS * 4);
}

/** Projects a spherical radiance function onto L2 SH by uniform-grid
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
      for (let c = 0; c < SH_COEFFS; c++) {
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
      for (let c = 0; c < SH_COEFFS; c++) {
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

// --- Band-2 rotation machinery ---
// Fixed sample directions whose L2 projections span the band.
const L2_DIRS: [number, number, number][] = [
  [1, 0, 0],
  [0, 0, 1],
  [Math.SQRT1_2, Math.SQRT1_2, 0],
  [0, Math.SQRT1_2, Math.SQRT1_2],
  [Math.SQRT1_2, 0, Math.SQRT1_2],
];

function l2Basis(d: [number, number, number]): number[] {
  return shBasis(d).slice(4);
}

/** Invert a 5x5 matrix (column-major, m[col*5+row]) by Gauss-Jordan. */
function invert5(m: Float64Array): Float64Array {
  const a = Float64Array.from(m);
  const inv = new Float64Array(25);
  for (let i = 0; i < 5; i++) inv[i * 5 + i] = 1;
  for (let col = 0; col < 5; col++) {
    // Partial pivot
    let pivot = col;
    for (let r = col + 1; r < 5; r++) {
      if (Math.abs(a[col * 5 + r]) > Math.abs(a[col * 5 + pivot])) pivot = r;
    }
    if (Math.abs(a[col * 5 + pivot]) < 1e-12) throw new Error('L2 basis matrix singular');
    if (pivot !== col) {
      for (let c = 0; c < 5; c++) {
        [a[c * 5 + col], a[c * 5 + pivot]] = [a[c * 5 + pivot], a[c * 5 + col]];
        [inv[c * 5 + col], inv[c * 5 + pivot]] = [inv[c * 5 + pivot], inv[c * 5 + col]];
      }
    }
    const d = 1 / a[col * 5 + col];
    for (let c = 0; c < 5; c++) { a[c * 5 + col] *= d; inv[c * 5 + col] *= d; }
    for (let r = 0; r < 5; r++) {
      if (r === col) continue;
      const f = a[col * 5 + r];
      if (f === 0) continue;
      for (let c = 0; c < 5; c++) {
        a[c * 5 + r] -= f * a[c * 5 + col];
        inv[c * 5 + r] -= f * inv[c * 5 + col];
      }
    }
  }
  return inv;
}

// A^-1 where A's columns are the L2 basis at the sample directions
const L2_A_INV = (() => {
  const A = new Float64Array(25);
  L2_DIRS.forEach((d, i) => l2Basis(d).forEach((v, r) => { A[i * 5 + r] = v; }));
  return invert5(A);
})();

/** Rotates all bands by a 3x3 rotation matrix (column-major, same
 * convention as before: rotating the function by M transforms the L1
 * coefficient triplet by M; Y00 is invariant; L2 rotates by the induced
 * 5x5, built numerically from Y2(M u) = W Y2(u). */
export function rotateSh(sh: ShRgb, m: ArrayLike<number>): ShRgb {
  const out = new Float32Array(sh);
  const rot = (d: [number, number, number]): [number, number, number] => [
    m[0] * d[0] + m[3] * d[1] + m[6] * d[2],
    m[1] * d[0] + m[4] * d[1] + m[7] * d[2],
    m[2] * d[0] + m[5] * d[1] + m[8] * d[2],
  ];
  for (let ch = 0; ch < 3; ch++) {
    // L1: coefficient triplet as a vector (x = Y11, y = Y1-1, z = Y10)
    const x = sh[3 * 4 + ch], y = sh[1 * 4 + ch], z = sh[2 * 4 + ch];
    out[3 * 4 + ch] = m[0] * x + m[3] * y + m[6] * z;
    out[1 * 4 + ch] = m[1] * x + m[4] * y + m[7] * z;
    out[2 * 4 + ch] = m[2] * x + m[5] * y + m[8] * z;
  }
  // W = B * A^-1, B columns = Y2 at rotated sample directions
  const B = new Float64Array(25);
  L2_DIRS.forEach((d, i) => l2Basis(rot(d)).forEach((v, r) => { B[i * 5 + r] = v; }));
  const W = new Float64Array(25);
  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      let s = 0;
      for (let k = 0; k < 5; k++) s += B[k * 5 + r] * L2_A_INV[c * 5 + k];
      W[c * 5 + r] = s;
    }
  }
  for (let ch = 0; ch < 3; ch++) {
    const c2 = [0, 1, 2, 3, 4].map(i => sh[(4 + i) * 4 + ch]);
    for (let r = 0; r < 5; r++) {
      let s = 0;
      for (let c = 0; c < 5; c++) s += W[c * 5 + r] * c2[c];
      out[(4 + r) * 4 + ch] = s;
    }
  }
  return out;
}

/** Unshadowed irradiance at normal n: E = sum A_l L_lm Y_lm(n). */
export function evalIrradiance(sh: ShRgb, n: [number, number, number]): [number, number, number] {
  const basis = shBasis(n);
  const out: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < SH_COEFFS; c++) {
    const w = SH_A[c] * basis[c];
    out[0] += sh[c * 4] * w;
    out[1] += sh[c * 4 + 1] * w;
    out[2] += sh[c * 4 + 2] * w;
  }
  return out;
}
