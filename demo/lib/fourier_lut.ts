// sRGB -> Fourier sRGB lookup for arbitrary colors (paint picker, materials).
//
// demo/srgb_to_fourier_srgb.lut is a 33^3 downsample of the 256^3 table from
// Peters' path tracer (data/srgb_to_fourier_srgb.dat, BSD-3), stored as
// sRGB-encoded u8 triples; trilinear interpolation reconstructs the full
// table with a mean error of ~0.6/255 (worst ~5/255 at gamut corners).
import { srgbToLinear, type Vec3 } from "./spectra.ts";

const MAGIC = "FSRGBLUT";

export interface FourierLut {
  n: number;
  data: Uint8Array; // n^3 * 3, index ((b*n + g)*n + r) * 3
}

export function parseFourierLut(buffer: ArrayBuffer): FourierLut {
  const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
  if (magic !== MAGIC) {
    throw new Error(`Bad Fourier LUT magic: ${magic}`);
  }
  const n = new DataView(buffer).getUint32(8, true);
  const data = new Uint8Array(buffer, 12, n * n * n * 3);
  return { n, data };
}

function srgbEncode(v: number): number {
  return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}

/** Converts a linear-sRGB color to linear Fourier sRGB, ready for
 * fourierSrgbToFourier() / the Material buffer. */
export function srgbToFourierSrgb(lut: FourierLut, linearRgb: Vec3): Vec3 {
  const encoded = linearRgb.map((v) =>
    Math.min(255, Math.max(0, srgbEncode(v) * 255))) as Vec3;
  const out: Vec3 = [0, 0, 0];
  const n = lut.n;
  const locate = (x: number): [number, number] => {
    const fx = (x * (n - 1)) / 255.0;
    const i0 = Math.min(n - 2, Math.floor(fx));
    return [i0, fx - i0];
  };
  const [ri, tr] = locate(encoded[0]);
  const [gi, tg] = locate(encoded[1]);
  const [bi, tb] = locate(encoded[2]);
  for (let dr = 0; dr < 2; ++dr) {
    for (let dg = 0; dg < 2; ++dg) {
      for (let db = 0; db < 2; ++db) {
        const w = (dr ? tr : 1 - tr) * (dg ? tg : 1 - tg) * (db ? tb : 1 - tb);
        const base = (((bi + db) * n + (gi + dg)) * n + (ri + dr)) * 3;
        for (let c = 0; c < 3; ++c) out[c] += w * lut.data[base + c];
      }
    }
  }
  return out.map((v) => srgbToLinear(v / 255)) as Vec3;
}
