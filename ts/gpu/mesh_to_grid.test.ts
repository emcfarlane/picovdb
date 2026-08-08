import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { Binner, type BinResult } from './mesh_to_grid.ts';

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

// Reference implementation of the binning contract (mirrors rasterizeTriangle
// in src/mesh_to_grid.zig): dilated bbox [ceil(min-hw), floor(max+hw)],
// leaves [lo>>3, hi>>3] per axis, f32 arithmetic throughout.
function refBin(
  points: Float32Array,
  triangles: Uint32Array,
  voxelSize: number,
  halfWidth: number,
  leafMin: [number, number, number]
): { pairKeys: Uint32Array; pairTris: Uint32Array; leafKeys: Uint32Array } {
  const inv = Math.fround(1 / voxelSize);
  const pts = new Float32Array(points.length);
  for (let i = 0; i < points.length; i++) pts[i] = Math.fround(points[i] * inv);

  const pairs: Array<[number, number]> = [];
  for (let t = 0; t < triangles.length / 3; t++) {
    const lo = [0, 0, 0];
    const hi = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      const a = pts[triangles[t * 3] * 3 + axis];
      const b = pts[triangles[t * 3 + 1] * 3 + axis];
      const c = pts[triangles[t * 3 + 2] * 3 + axis];
      lo[axis] = Math.ceil(Math.fround(Math.min(a, b, c) - halfWidth)) >> 3;
      hi[axis] = Math.floor(Math.fround(Math.max(a, b, c) + halfWidth)) >> 3;
    }
    for (let x = lo[0]; x <= hi[0]; x++) {
      for (let y = lo[1]; y <= hi[1]; y++) {
        for (let z = lo[2]; z <= hi[2]; z++) {
          const key = (((x - leafMin[0]) << 20) | ((y - leafMin[1]) << 10) | (z - leafMin[2])) >>> 0;
          pairs.push([key, t]);
        }
      }
    }
  }
  pairs.sort((a, b) => a[0] - b[0] || 0); // Array.sort is stable; tri order preserved per key
  const pairKeys = new Uint32Array(pairs.map((p) => p[0]));
  const pairTris = new Uint32Array(pairs.map((p) => p[1]));
  const leafKeys = new Uint32Array([...new Set(pairKeys)]);
  return { pairKeys, pairTris, leafKeys };
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

  // Random triangle soup, negative coords included.
  const triCount = 300;
  const points = new Float32Array(triCount * 9);
  for (let i = 0; i < points.length; i++) points[i] = (rand() - 0.5) * 60;
  const triangles = new Uint32Array([...Array(triCount * 3).keys()]);
  await checkAgainstRef(binner, points, triangles, 0.25, 3, 'soup');

  // Thin sliver whose dilated bbox spans no integer coordinate on some axes.
  const sliver = new Float32Array([0.2, 0.21, 0.2, 0.4, 0.22, 0.2, 0.3, 0.23, 0.21]);
  await checkAgainstRef(binner, sliver, new Uint32Array([0, 1, 2]), 1, 0.1, 'sliver');

  // Degenerate point-triangle on a leaf boundary.
  const point = new Float32Array([8, -8, 16, 8, -8, 16, 8, -8, 16]);
  await checkAgainstRef(binner, point, new Uint32Array([0, 1, 2]), 1, 3, 'point');
});

function parseBinarySTL(bytes: Uint8Array): { points: Float32Array<ArrayBuffer>; triangles: Uint32Array<ArrayBuffer> } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triCount = view.getUint32(80, true);
  const points = new Float32Array(triCount * 9);
  const triangles = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const base = 84 + t * 50 + 12; // skip normal
    for (let i = 0; i < 9; i++) points[t * 9 + i] = view.getFloat32(base + i * 4, true);
    for (let v = 0; v < 3; v++) triangles[t * 3 + v] = t * 3 + v;
  }
  return { points, triangles };
}

Deno.test({ name: 'STL binning matches reference', ignore: !gpu || !stl }, async () => {
  const device = await requestDevice();
  const binner = new Binner(device);
  const { points, triangles } = parseBinarySTL(stl!);
  const result = await checkAgainstRef(binner, points, triangles, 0.25, 3, 'stl');
  if (result.leafCount < 100) throw new Error(`implausibly few leaves: ${result.leafCount}`);
});
