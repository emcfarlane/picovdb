import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { refRasterize, parseBinarySTL } from './reference.ts';
import { Binner } from './mesh_to_grid.ts';
import { Rasterizer } from './rasterize.ts';

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

async function checkDistances(
  binner: Binner,
  rasterizer: Rasterizer,
  points: Float32Array<ArrayBuffer>,
  triangles: Uint32Array<ArrayBuffer>,
  voxelSize: number,
  halfWidth: number,
  label: string
): Promise<{ band: number; total: number }> {
  const bin = await binner.bin(points, triangles, { voxelSize, halfWidth });
  const leafValues = rasterizer.rasterize(bin, { halfWidth });
  const got = await readBackU32(binner.device, leafValues, bin.leafCount * 512);
  const ref = refRasterize(points, triangles, voxelSize, halfWidth, bin.leafMin);
  const expected = new Uint32Array(ref.values.buffer);

  if (got.length !== expected.length) throw new Error(`${label}: slab size mismatch`);
  // Band membership must match exactly. Values may diverge slightly: GPU
  // compilers fuse multiply-adds, which can flip closest-feature branches in
  // the distance function; the branches are continuous so the distance error
  // stays tiny (measured < 1e-5 voxels), but it is far beyond ulp gating.
  const INF = 0x7f800000;
  const MAX_ABS_D = 1e-3; // voxel units, on sqrt(d^2)
  const gotF = new Float32Array(got.buffer);
  let band = 0;
  let maxAbs = 0;
  let membershipFlips = 0;
  let firstBad = '';
  for (let i = 0; i < got.length; i++) {
    const e = expected[i];
    if (e !== INF) band++;
    if ((e === INF) !== (got[i] === INF)) {
      membershipFlips++;
      if (!firstBad) firstBad = `[${i}] membership: got 0x${got[i].toString(16)}, expected 0x${e.toString(16)}`;
      continue;
    }
    if (e === INF) continue;
    const d = Math.abs(Math.sqrt(gotF[i]) - Math.sqrt(ref.values[i]));
    if (d > maxAbs) maxAbs = d;
    if (d > MAX_ABS_D && !firstBad) {
      firstBad = `[${i}] |Δd|=${d}: got ${Math.sqrt(gotF[i])}, expected ${Math.sqrt(ref.values[i])}`;
    }
  }
  if (membershipFlips > 0 || maxAbs > MAX_ABS_D) {
    throw new Error(`${label}: ${membershipFlips} membership flips, max |Δd| ${maxAbs}; first ${firstBad}`);
  }
  return { band, total: got.length };
}

Deno.test({ name: 'distance rasterization matches reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const rasterizer = new Rasterizer(device);
  const rand = mulberry32(4);

  const triCount = 100;
  const points = new Float32Array(triCount * 9);
  for (let i = 0; i < points.length; i++) points[i] = (rand() - 0.5) * 40;
  const triangles = new Uint32Array([...Array(triCount * 3).keys()]);
  const soup = await checkDistances(binner, rasterizer, points, triangles, 0.5, 3, 'soup');
  if (soup.band === 0) throw new Error('no band voxels in soup');

  // Degenerate (collinear) triangle exercises the segment fallback.
  const line = new Float32Array([0, 0, 0, 4, 0, 0, 8, 0, 0]);
  await checkDistances(binner, rasterizer, line, new Uint32Array([0, 1, 2]), 1, 2, 'collinear');
});

Deno.test({ name: 'STL distance rasterization matches reference', ignore: !gpu || !stl }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const rasterizer = new Rasterizer(device);
  const { points, triangles } = parseBinarySTL(stl!);
  const { band, total } = await checkDistances(binner, rasterizer, points, triangles, 0.25, 3, 'stl');
  if (band < 1000) throw new Error(`implausibly few band voxels: ${band}/${total}`);
});
