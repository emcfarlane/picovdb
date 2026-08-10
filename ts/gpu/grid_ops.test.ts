import { hasWebGPU, requestDevice, createU32Buffer, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { refCsgMerge } from './reference.ts';
import { Pruner } from './prune.ts';
import { Merger } from './merge.ts';

const gpu = await hasWebGPU();

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

const pack = (x: number, y: number, z: number): number => ((x << 20) | (y << 10) | z) >>> 0;

function randomGrid(rand: () => number, count: number, withValues: boolean) {
  const keySet = new Set<number>();
  while (keySet.size < count) {
    keySet.add(pack(10 + Math.floor(rand() * 5), 10 + Math.floor(rand() * 5), 10 + Math.floor(rand() * 5)));
  }
  const keys = new Uint32Array([...keySet].sort((a, b) => a - b));
  const masks = new Uint32Array(keys.length * 16);
  for (let i = 0; i < masks.length; i++) {
    if (rand() < 0.4) masks[i] = Math.floor(rand() * 4294967296);
  }
  const values = withValues ? new Float32Array(keys.length * 512) : undefined;
  if (values) for (let i = 0; i < values.length; i++) values[i] = Math.fround((rand() - 0.5) * 6);
  return { keys, masks, values };
}

Deno.test({ name: 'prune matches reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const pruner = new Pruner(device);
  const rand = mulberry32(7);
  const { keys, masks } = randomGrid(rand, 40, false);
  const retain = new Uint32Array(keys.length * 16);
  for (let i = 0; i < retain.length; i++) {
    if (rand() < 0.5) retain[i] = Math.floor(rand() * 4294967296);
  }

  const result = await pruner.prune(
    createU32Buffer(device, keys),
    createU32Buffer(device, masks),
    createU32Buffer(device, retain),
    keys.length
  );

  // The reference ANDs and drops empty leaves.
  const refKeys: number[] = [];
  const refMasks: number[] = [];
  for (let i = 0; i < keys.length; i++) {
    const anded = [];
    let any = 0;
    for (let w = 0; w < 16; w++) {
      const m = (masks[i * 16 + w] & retain[i * 16 + w]) >>> 0;
      anded.push(m);
      any |= m;
    }
    if (any) {
      refKeys.push(keys[i]);
      refMasks.push(...anded);
    }
  }
  if (result.leafCount !== refKeys.length) throw new Error(`count ${result.leafCount} != ${refKeys.length}`);
  assertU32ArrayEqual(await readBackU32(device, result.leafKeys, result.leafCount), new Uint32Array(refKeys), 'prune keys');
  assertU32ArrayEqual(await readBackU32(device, result.masks, result.leafCount * 16), new Uint32Array(refMasks), 'prune masks');
});

Deno.test({ name: 'topology merge matches reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const merger = new Merger(device);
  const rand = mulberry32(8);
  const a = randomGrid(rand, 30, false);
  const b = randomGrid(rand, 30, false); // overlapping coordinate range

  const result = await merger.merge(
    { leafKeys: createU32Buffer(device, a.keys), masks: createU32Buffer(device, a.masks), leafCount: a.keys.length },
    { leafKeys: createU32Buffer(device, b.keys), masks: createU32Buffer(device, b.masks), leafCount: b.keys.length }
  );

  // The reference is the sorted key union with OR masks.
  const union = new Uint32Array([...new Set([...a.keys, ...b.keys])].sort((x, y) => x - y));
  if (result.leafCount !== union.length) throw new Error(`count ${result.leafCount} != ${union.length}`);
  assertU32ArrayEqual(await readBackU32(device, result.leafKeys, result.leafCount), union, 'merge keys');

  const aIdx = new Map<number, number>();
  a.keys.forEach((k, i) => aIdx.set(k, i));
  const bIdx = new Map<number, number>();
  b.keys.forEach((k, i) => bIdx.set(k, i));
  const refMasks = new Uint32Array(union.length * 16);
  union.forEach((key, i) => {
    const ai = aIdx.get(key);
    const bi = bIdx.get(key);
    for (let w = 0; w < 16; w++) {
      refMasks[i * 16 + w] = ((ai !== undefined ? a.masks[ai * 16 + w] : 0) | (bi !== undefined ? b.masks[bi * 16 + w] : 0)) >>> 0;
    }
  });
  assertU32ArrayEqual(await readBackU32(device, result.masks, result.leafCount * 16), refMasks, 'merge masks');
});

Deno.test({ name: 'CSG merge matches reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const merger = new Merger(device);
  const rand = mulberry32(9);
  const halfWidth = 3;
  const a = randomGrid(rand, 30, true);
  const b = randomGrid(rand, 30, true); // overlapping coordinate range

  const result = await merger.merge(
    {
      leafKeys: createU32Buffer(device, a.keys),
      masks: createU32Buffer(device, a.masks),
      values: createU32Buffer(device, new Uint32Array(a.values!.buffer)),
      leafCount: a.keys.length,
    },
    {
      leafKeys: createU32Buffer(device, b.keys),
      masks: createU32Buffer(device, b.masks),
      values: createU32Buffer(device, new Uint32Array(b.values!.buffer)),
      leafCount: b.keys.length,
    },
    { halfWidth }
  );

  const ref = refCsgMerge(a.keys, a.values!, b.keys, b.values!, halfWidth);
  if (result.leafCount !== ref.keys.length) throw new Error(`count ${result.leafCount} != ${ref.keys.length}`);
  assertU32ArrayEqual(await readBackU32(device, result.leafKeys, result.leafCount), ref.keys, 'csg keys');
  assertU32ArrayEqual(await readBackU32(device, result.masks, result.leafCount * 16), ref.masks, 'csg masks');
  assertU32ArrayEqual(
    await readBackU32(device, result.values!, result.leafCount * 512),
    new Uint32Array(ref.values.buffer),
    'csg values'
  );
});
