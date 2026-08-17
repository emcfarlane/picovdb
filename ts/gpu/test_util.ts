// Shared helpers for the GPU tests.

import { readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { LEAF_U32, LOWER_U32, UPPER_U32, type EmitResult } from './emit.ts';
import type { PicoVDBFile } from '../picovdb.ts';

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
