// Equirectangular HDRI environment: asset parsing and the luminance CDFs
// used for importance sampling (PBRT-style 2D distribution: a marginal CDF
// over rows and a conditional CDF over columns per row, both including the
// sin(theta) solid-angle weight).
//
// Asset format: "PVDBENV1" + u32 w,h + f32 avg rgb + rgba16f equirect payload.

export interface EnvMap {
  width: number;
  height: number;
  avgRgb: [number, number, number];
  /** rgba16f texels, ready for texture upload. */
  rgba16: Uint16Array<ArrayBuffer>;
}

export function parseEnv(buffer: ArrayBuffer): EnvMap {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== 'PVDBENV1') {
    throw new Error(`Bad environment asset magic: ${magic}`);
  }
  const header = new DataView(buffer);
  const width = header.getUint32(8, true);
  const height = header.getUint32(12, true);
  const avgRgb: [number, number, number] = [
    header.getFloat32(16, true), header.getFloat32(20, true), header.getFloat32(24, true),
  ];
  const rgba16 = new Uint16Array(buffer.slice(28)) as Uint16Array<ArrayBuffer>;
  return { width, height, avgRgb, rgba16 };
}

/** Decodes an IEEE binary16 bit pattern (inverse of toFloat16Bits). */
export function fromFloat16Bits(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

export interface EnvCdfs {
  /** Row-normalized column CDF per row; width * height floats. */
  conditional: Float32Array<ArrayBuffer>;
  /** CDF over rows (of sin-weighted row luminance sums); height floats. */
  marginal: Float32Array<ArrayBuffer>;
}

/** Builds the sampling CDFs from an environment map's luminance. */
export function buildEnvCdfs(env: EnvMap): EnvCdfs {
  const { width: w, height: h, rgba16 } = env;
  const conditional = new Float32Array(w * h);
  const marginal = new Float32Array(h);
  let total = 0;
  for (let y = 0; y < h; ++y) {
    const sinTheta = Math.sin(((y + 0.5) / h) * Math.PI);
    let rowSum = 0;
    for (let x = 0; x < w; ++x) {
      const i = (y * w + x) * 4;
      const lum =
        0.2126 * fromFloat16Bits(rgba16[i]) +
        0.7152 * fromFloat16Bits(rgba16[i + 1]) +
        0.0722 * fromFloat16Bits(rgba16[i + 2]);
      rowSum += Math.max(0, lum) * sinTheta;
      conditional[y * w + x] = rowSum;
    }
    if (rowSum > 0) {
      for (let x = 0; x < w; ++x) conditional[y * w + x] /= rowSum;
    }
    total += rowSum;
    marginal[y] = total;
  }
  for (let y = 0; y < h; ++y) marginal[y] /= total;
  return { conditional, marginal };
}

/** CPU mirror of the shader's env_texel_pdf (solid-angle pdf of a texel);
 * used by tests to verify the distribution integrates to one. */
export function envTexelPdf(cdfs: EnvCdfs, w: number, h: number, x: number, y: number): number {
  const cond = cdfs.conditional[y * w + x] - (x > 0 ? cdfs.conditional[y * w + x - 1] : 0);
  const marg = cdfs.marginal[y] - (y > 0 ? cdfs.marginal[y - 1] : 0);
  const sinTheta = Math.sin(((y + 0.5) / h) * Math.PI);
  return (cond * marg * w * h) / (2 * Math.PI * Math.PI * Math.max(sinTheta, 1e-4));
}
