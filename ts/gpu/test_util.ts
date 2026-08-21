// Shared helpers for the GPU tests.

import assert from 'node:assert/strict';
import { readBackU32 } from './device.ts';
import { LOWER_U32, UPPER_U32, type EmitResult } from './emit.ts';
import { LEAF_U32, emptyOpGrid, type OpGrid } from './opgrid.ts';
import { createU32Buffer } from './device.ts';
import type { PicoVDBFile } from '../picovdb.ts';

/** Fast equality for large typed arrays. assert.deepEqual takes minutes at this size. */
export function assertU32ArrayEqual(got: Uint32Array, expected: Uint32Array, label: string): void {
  assert.equal(got.length, expected.length, `${label}: length`);
  for (let i = 0; i < got.length; i++) {
    if (got[i] !== expected[i]) {
      assert.fail(`${label}: first mismatch at [${i}]: got ${got[i]}, expected ${expected[i]}`);
    }
  }
}

/** Deterministic PRNG so failures reproduce. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Compares an emitted tree against a CPU converted file. Node buffers must
 * match byte for byte and values must match in sign and within tolerance.
 * Returns the largest value difference.
 */
export async function compareTreeToCpu(device: GPUDevice, tree: EmitResult, cpu: PicoVDBFile): Promise<number> {
  const h = cpu.header;
  if (tree.leafCount !== h.leafCount || tree.lowerCount !== h.lowerCount || tree.upperCount !== h.upperCount) {
    throw new Error(
      `counts: gpu ${tree.leafCount}/${tree.lowerCount}/${tree.upperCount} != cpu ${h.leafCount}/${h.lowerCount}/${h.upperCount}`
    );
  }
  const grid = cpu.getGrid(0);
  if (tree.dataElemCount !== grid.dataElemCount) {
    throw new Error(`dataElemCount: gpu ${tree.dataElemCount} != cpu ${grid.dataElemCount}`);
  }
  for (let a = 0; a < 3; a++) {
    if (tree.indexBoundsMin[a] !== grid.indexBoundsMin[a] || tree.indexBoundsMax[a] !== grid.indexBoundsMax[a]) {
      throw new Error(`index bounds mismatch on axis ${a}`);
    }
  }
  const cpuU32 = (bytes: Uint8Array) => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  assertU32ArrayEqual(
    await readBackU32(device, tree.roots, tree.upperCount * 2),
    cpuU32(cpu.rootsBuffer).slice(0, tree.upperCount * 2),
    'roots'
  );
  assertU32ArrayEqual(await readBackU32(device, tree.uppers, tree.upperCount * UPPER_U32), cpuU32(cpu.uppersBuffer), 'uppers');
  assertU32ArrayEqual(await readBackU32(device, tree.lowers, tree.lowerCount * LOWER_U32), cpuU32(cpu.lowersBuffer), 'lowers');
  assertU32ArrayEqual(await readBackU32(device, tree.leaves, tree.leafCount * LEAF_U32), cpuU32(cpu.leavesBuffer), 'leaves');

  const gpuData = new Float32Array((await readBackU32(device, tree.data, tree.dataElemCount)).buffer);
  const cpuData = new Float32Array(cpu.dataBuffer.buffer, cpu.dataBuffer.byteOffset, tree.dataElemCount);
  let maxAbs = 0;
  for (let i = 0; i < tree.dataElemCount; i++) {
    if ((gpuData[i] < 0) !== (cpuData[i] < 0)) {
      throw new Error(`value sign mismatch at ${i}: ${gpuData[i]} vs ${cpuData[i]}`);
    }
    maxAbs = Math.max(maxAbs, Math.abs(gpuData[i] - cpuData[i]));
  }
  if (maxAbs > 1e-3) throw new Error(`value divergence ${maxAbs}`);
  return maxAbs;
}

export function emptyGrid(device: GPUDevice): OpGrid {
  return emptyOpGrid(device, [0, 0, 0], [1023, 1023, 1023]);
}

export interface GridView {
  keys: Uint32Array;
  leaves: Uint32Array;
  data: Float32Array;
  /** Value of voxel n of leaf i, mirroring the WGSL reader. */
  value(i: number, n: number): number;
  band(i: number, n: number): boolean;
}

/** Reads an op layer grid back with a voxel accessor. */
export async function readGrid(device: GPUDevice, grid: OpGrid, halfWidth: number): Promise<GridView> {
  const keys = await readBackU32(device, grid.leafKeys, grid.leafCount);
  const leaves = await readBackU32(device, grid.leaves, grid.leafCount * LEAF_U32);
  const data = new Float32Array((await readBackU32(device, grid.data, 2 + grid.activeVoxels)).buffer);
  const popcount = (v: number) => {
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  };
  const band = (i: number, n: number) => ((leaves[i * LEAF_U32 + 5 + (n >> 5) * 3] >>> (n & 31)) & 1) === 1;
  const value = (i: number, n: number) => {
    const e = i * LEAF_U32 + 4 + (n >> 5) * 3;
    const bit = n & 31;
    if (band(i, n)) {
      const below = bit === 0 ? 0 : (leaves[e + 1] & ((1 << bit) - 1)) >>> 0;
      return data[leaves[i * LEAF_U32 + 2] + (leaves[e + 2] & 0xffff) + popcount(below)];
    }
    return ((leaves[e] >>> bit) & 1) === 1 ? -halfWidth : halfWidth;
  };
  return { keys, leaves, data, value, band };
}

/** Dense view of a grid: a 512 value slab and 16 band mask words per leaf. */
export async function unpackGrid(device: GPUDevice, grid: OpGrid, halfWidth: number): Promise<{ keys: Uint32Array; masks: Uint32Array; values: Float32Array }> {
  const view = await readGrid(device, grid, halfWidth);
  const values = new Float32Array(grid.leafCount * 512);
  const masks = new Uint32Array(grid.leafCount * 16);
  for (let i = 0; i < grid.leafCount; i++) {
    for (let n = 0; n < 512; n++) {
      values[i * 512 + n] = view.value(i, n);
      if (view.band(i, n)) masks[i * 16 + (n >> 5)] |= 1 << (n & 31);
    }
  }
  return { keys: view.keys, masks, values };
}

/** Uploads a dense grid of sorted keys and 512 value slabs as an op layer grid, dropping leaves without band voxels. */
export function packGrid(device: GPUDevice, keys: Uint32Array, values: Float32Array, halfWidth: number): OpGrid {
  const outKeys: number[] = [];
  const leaves: number[] = [];
  const data: number[] = [halfWidth, -halfWidth];
  for (let i = 0; i < keys.length; i++) {
    const record = [0, 0, data.length, 0];
    const slab: number[] = [];
    for (let w = 0; w < 16; w++) {
      let band = 0;
      let inside = 0;
      for (let b = 0; b < 32; b++) {
        const v = values[i * 512 + w * 32 + b];
        if (Math.abs(v) < halfWidth) {
          band |= 1 << b;
          slab.push(v);
        }
        if (v < 0) inside |= 1 << b;
      }
      record.push((inside & ~band) >>> 0, band >>> 0, 0);
    }
    if (slab.length === 0) continue;
    // Recompute the per word prefixes from the band words.
    let prefix = 0;
    for (let w = 0; w < 16; w++) {
      record[4 + w * 3 + 2] = prefix;
      let m = record[4 + w * 3 + 1];
      let c = 0;
      while (m) { m &= m - 1; c++; }
      prefix += c;
    }
    outKeys.push(keys[i]);
    leaves.push(...record);
    data.push(...slab);
  }
  const f32 = new Float32Array(data);
  return {
    leafKeys: createU32Buffer(device, new Uint32Array(outKeys)),
    leaves: createU32Buffer(device, new Uint32Array(leaves)),
    data: createU32Buffer(device, new Uint32Array(f32.buffer)),
    leafCount: outKeys.length,
    activeVoxels: data.length - 2,
    leafMin: [0, 0, 0],
    leafMax: [1023, 1023, 1023],
  };
}

// Every stored voxel must match the expected SDF within tolerance, and
// its band bit must match away from the band edge. With exactBelow,
// values only need to match where |expected| is below it. Elsewhere the
// sign must agree.
export async function checkAnalytic(
  device: GPUDevice,
  grid: OpGrid,
  expected: (p: [number, number, number]) => number,
  halfWidth: number,
  label: string,
  exactBelow = Infinity,
  tolerance = 1e-3
): Promise<{ band: number }> {
  const view = await readGrid(device, grid, halfWidth);
  const TOL = tolerance;
  let band = 0;
  for (let i = 0; i < grid.leafCount; i++) {
    const ox = (((view.keys[i] >>> 20) & 0x3ff) + grid.leafMin[0]) * 8;
    const oy = (((view.keys[i] >>> 10) & 0x3ff) + grid.leafMin[1]) * 8;
    const oz = ((view.keys[i] & 0x3ff) + grid.leafMin[2]) * 8;
    for (let n = 0; n < 512; n++) {
      const p: [number, number, number] = [ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7)];
      const e = Math.max(-halfWidth, Math.min(halfWidth, expected(p)));
      const v = view.value(i, n);
      const exact = Math.abs(e) < exactBelow;
      if (exact ? Math.abs(v - e) > TOL : (v < 0) !== (e < 0) && Math.abs(e) > TOL) {
        throw new Error(`${label}: value at ${p}: got ${v}, expected ${e}`);
      }
      const bit = view.band(i, n) ? 1 : 0;
      if (bit) band++;
      if (exact && Math.abs(Math.abs(e) - halfWidth) > 2 * TOL && bit !== (Math.abs(e) < halfWidth ? 1 : 0)) {
        throw new Error(`${label}: band bit at ${p}: got ${bit}, |v|=${Math.abs(v)}`);
      }
    }
  }
  return { band };
}
