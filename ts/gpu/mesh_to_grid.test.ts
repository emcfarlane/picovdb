import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { refBin, parseBinarySTL } from './reference.ts';
import { Binner, type BinResult } from './mesh_to_grid.ts';

const gpu = await hasWebGPU();

let stl: Uint8Array | null = null;
try {
  stl = Deno.readFileSync(new URL('../../data/bases/base_32mm.stl', import.meta.url));
} catch {
  // Sample data not present, so the STL test skips.
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

async function checkAgainstRef(
  binner: Binner,
  points: Float32Array<ArrayBuffer>,
  triangles: Uint32Array<ArrayBuffer>,
  voxelSize: number,
  halfWidth: number,
  label: string
): Promise<BinResult> {
  const result = await binner.bin(points, triangles, { voxelSize, halfWidth });
  const ref = refBin(points, triangles, voxelSize, halfWidth, result.leafMin);
  if (result.pairCount !== ref.pairKeys.length) {
    throw new Error(`${label}: pair count ${result.pairCount} != ref ${ref.pairKeys.length}`);
  }
  assertU32ArrayEqual(await readBackU32(binner.device, result.pairKeys, result.pairCount), ref.pairKeys, `${label} pair keys`);
  assertU32ArrayEqual(await readBackU32(binner.device, result.pairTris, result.pairCount), ref.pairTris, `${label} pair tris`);
  if (result.leafCount !== ref.leafKeys.length) {
    throw new Error(`${label}: leaf count ${result.leafCount} != ref ${ref.leafKeys.length}`);
  }
  assertU32ArrayEqual(await readBackU32(binner.device, result.leafKeys, result.leafCount), ref.leafKeys, `${label} leaf keys`);
  return result;
}

Deno.test({ name: 'triangle binning matches reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const rand = mulberry32(3);

  // Random triangle soup with negative coords included.
  const triCount = 300;
  const points = new Float32Array(triCount * 9);
  for (let i = 0; i < points.length; i++) points[i] = (rand() - 0.5) * 60;
  const triangles = new Uint32Array([...Array(triCount * 3).keys()]);
  await checkAgainstRef(binner, points, triangles, 0.25, 3, 'soup');

  // Thin sliver whose dilated bounds span no integer coordinate on some axes.
  const sliver = new Float32Array([0.2, 0.21, 0.2, 0.4, 0.22, 0.2, 0.3, 0.23, 0.21]);
  await checkAgainstRef(binner, sliver, new Uint32Array([0, 1, 2]), 1, 0.1, 'sliver');

  // Degenerate point triangle on a leaf boundary.
  const point = new Float32Array([8, -8, 16, 8, -8, 16, 8, -8, 16]);
  await checkAgainstRef(binner, point, new Uint32Array([0, 1, 2]), 1, 3, 'point');
});

Deno.test({ name: 'STL binning matches reference', ignore: !gpu || !stl }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const { points, triangles } = parseBinarySTL(stl!);
  const result = await checkAgainstRef(binner, points, triangles, 0.25, 3, 'stl');
  if (result.leafCount < 100) throw new Error(`implausibly few leaves: ${result.leafCount}`);
});
