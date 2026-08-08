import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { refSign, refRasterize, parseBinarySTL } from './reference.ts';
import { Binner } from './mesh_to_grid.ts';
import { Signer } from './sign.ts';

const gpu = await hasWebGPU();

let stl: Uint8Array | null = null;
try {
  stl = Deno.readFileSync(new URL('../../data/bases/base_32mm.stl', import.meta.url));
} catch {
  // Sample data not present (e.g. CI); the STL test is skipped.
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function checkSigns(
  binner: Binner,
  signer: Signer,
  points: Float32Array<ArrayBuffer>,
  triangles: Uint32Array<ArrayBuffer>,
  voxelSize: number,
  halfWidth: number,
  label: string
): Promise<{ inside: number; total: number }> {
  const bin = await binner.bin(points, triangles, { voxelSize, halfWidth });
  const sign = await signer.sign(bin);
  const got = await readBackU32(binner.device, sign.inside, bin.leafCount * 16);
  const leafKeys = await readBackU32(binner.device, bin.leafKeys, bin.leafCount);
  const expected = refSign(points, triangles, voxelSize, leafKeys, bin.leafMin);
  // The GPU computes crossings in f32, the CPU (and this reference) in f64;
  // parity may flip only for voxels whose column crossing lies within f32
  // noise of their center plane — those are on the surface, so their
  // narrow-band distance must be ~0.
  const ref = refRasterize(points, triangles, voxelSize, halfWidth, bin.leafMin);
  const ON_SURFACE = 1e-2; // voxel units
  let inside = 0;
  let flips = 0;
  let firstBad = '';
  for (let w = 0; w < got.length; w++) {
    inside += popcount(expected[w]);
    let diff = (got[w] ^ expected[w]) >>> 0;
    while (diff !== 0) {
      const bit = 31 - Math.clz32(diff);
      diff = (diff & ~(1 << bit)) >>> 0;
      flips++;
      const n = ((w % 16) * 32) + bit;
      const d2 = ref.values[(w >> 4) * 512 + n];
      if (Math.sqrt(d2) > ON_SURFACE && !firstBad) {
        firstBad = `leaf ${w >> 4} voxel ${n}: sign flip at distance ${Math.sqrt(d2)}`;
      }
    }
  }
  if (firstBad || flips > got.length * 32 * 0.001) {
    throw new Error(`${label}: ${flips} sign flips; ${firstBad || 'all on-surface but too many'}`);
  }
  return { inside, total: got.length * 32 };
}

function popcount(v: number): number {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

Deno.test({ name: 'parity signing matches f64 reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const signer = new Signer(device);

  // A closed cube (12 triangles, watertight) with interior voxels.
  // deno-fmt-ignore
  const cubePts = new Float32Array([
    -6, -6, -6,  6, -6, -6,  6, 6, -6,  -6, 6, -6, // z = -6 face corners
    -6, -6, 6,   6, -6, 6,   6, 6, 6,   -6, 6, 6,  // z = +6 face corners
  ]);
  const quads = [
    [0, 1, 2, 3], // bottom (z-)
    [4, 6, 5, 7], // top (z+) — winding irrelevant for parity
    [0, 4, 1, 5],
    [1, 5, 2, 6],
    [2, 6, 3, 7],
    [3, 7, 0, 4],
  ];
  const cubeTris: number[] = [];
  for (const [a, b, c, d] of quads) cubeTris.push(a, b, c, b, c, d);
  const cube = await checkSigns(binner, signer, cubePts, new Uint32Array(cubeTris), 1, 3, 'cube');
  if (cube.inside === 0) throw new Error('cube has no inside voxels');

  // Random soup: not watertight, parity is arbitrary but must be
  // deterministic and match the reference.
  const rand = mulberry32(5);
  const triCount = 100;
  const points = new Float32Array(triCount * 9);
  for (let i = 0; i < points.length; i++) points[i] = (rand() - 0.5) * 40;
  await checkSigns(binner, signer, points, new Uint32Array([...Array(triCount * 3).keys()]), 0.5, 3, 'soup');
});

Deno.test({ name: 'STL parity signing matches f64 reference', ignore: !gpu || !stl }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const signer = new Signer(device);
  const { points, triangles } = parseBinarySTL(stl!);
  const { inside, total } = await checkSigns(binner, signer, points, triangles, 0.25, 3, 'stl');
  if (inside < 1000) throw new Error(`implausibly few inside voxels: ${inside}/${total}`);
});
