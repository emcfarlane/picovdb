import { hasWebGPU, requestDevice, createU32Buffer, readBackU32 } from './device.ts';
import { mulberry32, assertU32ArrayEqual } from './test_util.ts';
import { Dilator } from './dilate.ts';

const gpu = await hasWebGPU();

const pack = (x: number, y: number, z: number): number => ((x << 20) | (y << 10) | z) >>> 0;

// Brute force reference. Expands every active voxel to its face neighbors
// in global voxel space and rebuilds leaves, keeping originals even when
// empty.
function refDilate(keys: Uint32Array, masks: Uint32Array): { keys: Uint32Array; masks: Uint32Array } {
  const active = new Set<string>();
  keys.forEach((key, li) => {
    const lx = (key >>> 20) & 0x3ff;
    const ly = (key >>> 10) & 0x3ff;
    const lz = key & 0x3ff;
    for (let n = 0; n < 512; n++) {
      if ((masks[li * 16 + (n >> 5)] >>> (n & 31)) & 1) {
        active.add(`${lx * 8 + (n >> 6)},${ly * 8 + ((n >> 3) & 7)},${lz * 8 + (n & 7)}`);
      }
    }
  });
  const dilated = new Set(active);
  for (const v of active) {
    const [x, y, z] = v.split(',').map(Number);
    for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (nx >= 0 && ny >= 0 && nz >= 0 && nx < 8192 && ny < 8192 && nz < 8192) {
        dilated.add(`${nx},${ny},${nz}`);
      }
    }
  }
  const leafSet = new Set<number>(keys);
  for (const v of dilated) {
    const [x, y, z] = v.split(',').map(Number);
    leafSet.add(pack(x >> 3, y >> 3, z >> 3));
  }
  const outKeys = new Uint32Array([...leafSet].sort((a, b) => a - b));
  const slot = new Map<number, number>();
  outKeys.forEach((k, i) => slot.set(k, i));
  const outMasks = new Uint32Array(outKeys.length * 16);
  for (const v of dilated) {
    const [x, y, z] = v.split(',').map(Number);
    const li = slot.get(pack(x >> 3, y >> 3, z >> 3))!;
    const n = ((x & 7) << 6) | ((y & 7) << 3) | (z & 7);
    outMasks[li * 16 + (n >> 5)] |= 1 << (n & 31);
  }
  return { keys: outKeys, masks: outMasks };
}

async function checkDilate(dilator: Dilator, keys: Uint32Array<ArrayBuffer>, masks: Uint32Array<ArrayBuffer>, label: string) {
  const keyBuf = createU32Buffer(dilator.device, keys);
  const maskBuf = createU32Buffer(dilator.device, masks);
  const result = await dilator.dilate(keyBuf, maskBuf, keys.length);
  const ref = refDilate(keys, masks);
  if (result.leafCount !== ref.keys.length) {
    throw new Error(`${label}: leaf count ${result.leafCount} != ref ${ref.keys.length}`);
  }
  assertU32ArrayEqual(await readBackU32(dilator.device, result.leafKeys, result.leafCount), ref.keys, `${label} keys`);
  assertU32ArrayEqual(await readBackU32(dilator.device, result.masks, result.leafCount * 16), ref.masks, `${label} masks`);
}

Deno.test({ name: 'dilate matches brute-force reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const dilator = new Dilator(device);
  const rand = mulberry32(6);

  // A single center voxel dilates within its leaf.
  const single = new Uint32Array(16);
  single[(((4 << 6) | (4 << 3) | 4) >> 5)] |= 1 << (((4 << 6) | (4 << 3) | 4) & 31);
  await checkDilate(dilator, new Uint32Array([pack(10, 10, 10)]), single, 'center');

  // A full leaf spills into all six neighbors.
  await checkDilate(dilator, new Uint32Array([pack(10, 10, 10)]), new Uint32Array(16).fill(0xffffffff), 'full');

  // A corner voxel spills across three faces. Face dilation must not
  // spawn diagonal leaves.
  const corner = new Uint32Array(16);
  corner[0] |= 1; // voxel zero
  await checkDilate(dilator, new Uint32Array([pack(10, 10, 10)]), corner, 'corner');

  // Random cluster of adjacent leaves with random masks.
  const keySet = new Set<number>();
  while (keySet.size < 30) {
    keySet.add(pack(20 + Math.floor(rand() * 4), 20 + Math.floor(rand() * 4), 20 + Math.floor(rand() * 4)));
  }
  const keys = new Uint32Array([...keySet].sort((a, b) => a - b));
  const masks = new Uint32Array(keys.length * 16);
  for (let i = 0; i < masks.length; i++) {
    if (rand() < 0.3) masks[i] = Math.floor(rand() * 4294967296);
  }
  await checkDilate(dilator, keys, masks, 'cluster');
});
