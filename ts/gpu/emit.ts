// Host side of wgsl/emit.wgsl: builds the picovdb tree of an op layer
// grid, and turns the mesh converter's distance slabs into an op layer
// grid.

import emitWgsl from 'picovdb/wgsl/emit.wgsl' with { type: 'text' };
import { PICOVDB_LOWER_SIZE, PICOVDB_UPPER_SIZE } from '../picovdb.ts';
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals, readBackU32 } from './device.ts';
import { GridWriter, LEAF_U32, preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';
import type { BinResult } from './mesh_to_grid.ts';
import type { SignResult } from './sign.ts';

export { LEAF_U32 };
export type { OpGrid };

const WG_SIZE = 256;
export const LOWER_U32 = PICOVDB_LOWER_SIZE / 4;
export const UPPER_U32 = PICOVDB_UPPER_SIZE / 4;

export interface EmitOptions {
  /** Narrow band half width in voxels. Must match the earlier stages. */
  halfWidth: number;
}

export interface EmitResult {
  roots: GPUBuffer; // two u32 key words per upper, unpadded
  uppers: GPUBuffer;
  lowers: GPUBuffer;
  leaves: GPUBuffer;
  data: GPUBuffer; // f32 values, two implicit background slots then per voxel values
  leafCount: number;
  lowerCount: number;
  upperCount: number;
  /** Total value slots including the two implicit entries. */
  dataElemCount: number;
  activeVoxels: number;
  surfaceVoxels: number;
  indexBoundsMin: [number, number, number];
  indexBoundsMax: [number, number, number];
}

export class Emitter {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, scanner: Scanner = new Scanner(device), sorter: Sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    const code = preludeWgsl + readerWgsl('cand', 'params.cand_count') + emitWgsl;
    // The shader hardcodes the node strides. Fail construction on drift
    // from the picovdb sizes.
    for (const [name, value] of [['LEAF_U32', LEAF_U32], ['LOWER_U32', LOWER_U32], ['UPPER_U32', UPPER_U32]] as const) {
      if (!code.includes(`${name}: u32 = ${value}u`)) {
        throw new Error(`wgsl/emit.wgsl ${name} does not match the picovdb node size`);
      }
    }
    const module = device.createShaderModule({ code });
    for (const entryPoint of [
      'classify_mark', 'classify_apply', 'opgrid_compact', 'leaf_stats', 'hier_lo', 'hier_hi', 'reorder_final',
      'leaf_value_counts', 'mark_lower', 'compact_lower', 'mark_upper', 'compact_upper',
      'surface', 'write_leaves', 'write_data', 'write_lowers', 'write_uppers', 'write_roots',
    ]) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  /** The tree of a converted mesh. */
  async emit(bin: BinResult, dist2: GPUBuffer, sign: SignResult, opts: EmitOptions): Promise<EmitResult> {
    const grid = await this.classifyOnly(bin, dist2, sign, opts);
    const tree = await this.reEmit(grid, opts);
    grid.leafKeys.destroy();
    grid.leaves.destroy();
    grid.data.destroy();
    return tree;
  }

  /** An op layer grid from the converter's squared distance slabs and inside masks. */
  classifyOnly(bin: BinResult, dist2: GPUBuffer, sign: SignResult, opts: EmitOptions): Promise<OpGrid> {
    const params = this.params(bin.leafCount, bin.leafMin, opts.halfWidth);
    const run = this.runner(params);
    const writer = new GridWriter(this.device, bin.leafCount);
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'classify_mark', bin.leafCount + 1, { 4: dist2, 11: sign.inside, ...writer.markBindings });
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
    return writer.finish(this.scanner, this.pipelines['opgrid_compact'], bin.leafKeys, opts.halfWidth, bin, (pass, out, count) => {
      run(pass, 'classify_apply', count, { 1: bin.leafKeys, 4: dist2, 11: sign.inside, ...out });
    });
  }

  /** The tree of an op layer grid. */
  async reEmit(grid: OpGrid, opts: EmitOptions): Promise<EmitResult> {
    const device = this.device;
    const cand = grid.leafCount;
    if (cand === 0) throw new Error('no active voxels');
    const params = this.params(cand, grid.leafMin, opts.halfWidth);
    const run = this.runner(params);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const reader = { 1: grid.leafKeys, 2: grid.leaves, 3: grid.data };
    const bandCounts = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    const bounds = device.createBuffer({ size: 24, usage: storage });
    device.queue.writeBuffer(bounds, 0, new Int32Array([0x7fffffff, 0x7fffffff, 0x7fffffff, -0x80000000, -0x80000000, -0x80000000]));
    const flags = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    const hier = device.createBuffer({ size: cand * 4, usage: storage });
    const idx = device.createBuffer({ size: cand * 4, usage: storage });
    const finalKeys = device.createBuffer({ size: cand * 4, usage: storage });
    const finalCand = device.createBuffer({ size: cand * 4, usage: storage });
    const valueCounts = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    {
      // Sort the leaves into the CPU's order, then derive value offsets
      // and lowers.
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'leaf_stats', cand + 1, { 1: grid.leafKeys, 2: grid.leaves, 5: bandCounts, 6: bounds });
      run(pass, 'hier_lo', cand, { 1: grid.leafKeys, 26: hier, 27: idx });
      this.sorter.plan(hier, idx, cand).encode(pass);
      run(pass, 'hier_hi', cand, { 1: grid.leafKeys, 26: hier, 27: idx });
      this.sorter.plan(hier, idx, cand).encode(pass);
      run(pass, 'reorder_final', cand, { 1: grid.leafKeys, 8: finalKeys, 9: finalCand, 27: idx });
      run(pass, 'leaf_value_counts', cand + 1, { 5: bandCounts, 9: finalCand, 10: valueCounts });
      this.scanner.plan(valueCounts, cand + 1).encode(pass);
      run(pass, 'mark_lower', cand + 1, { 7: flags, 8: finalKeys });
      this.scanner.plan(flags, cand + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [dataValues, lowerCount] = await readBackTotals(device, [
      { buffer: valueCounts, index: cand },
      { buffer: flags, index: cand },
    ]);

    device.queue.writeBuffer(params, 4, new Uint32Array([lowerCount]));
    const lowerKeys = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const lowerFirst = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const flatLower = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const flatVals = device.createBuffer({ size: lowerCount * 4, usage: GPUBufferUsage.STORAGE });
    {
      const encoder = device.createCommandEncoder();
      const passA = encoder.beginComputePass();
      run(passA, 'compact_lower', cand, { 7: flags, 8: finalKeys, 17: lowerKeys, 18: lowerFirst });
      passA.end();
      encoder.copyBufferToBuffer(lowerKeys, 0, flatLower, 0, lowerCount * 4);
      const passB = encoder.beginComputePass();
      this.sorter.plan(flatLower, flatVals, lowerCount).encode(passB);
      run(passB, 'mark_upper', lowerCount + 1, { 7: flags, 17: lowerKeys });
      this.scanner.plan(flags, lowerCount + 1).encode(passB);
      passB.end();
      device.queue.submit([encoder.finish()]);
    }
    const [upperCount] = await readBackTotals(device, [{ buffer: flags, index: lowerCount }]);

    // Surface masks and all node and value outputs.
    device.queue.writeBuffer(params, 8, new Uint32Array([upperCount]));
    const upperKeys = device.createBuffer({ size: upperCount * 4, usage: storage });
    const upperFirst = device.createBuffer({ size: upperCount * 4, usage: storage });
    const surfMasks = device.createBuffer({ size: cand * 16 * 4, usage: storage });
    const surfCounts = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    const leaves = device.createBuffer({ size: cand * LEAF_U32 * 4, usage: storage });
    const dataBytes = Math.ceil(((2 + dataValues) * 4) / 16) * 16;
    const data = device.createBuffer({ size: dataBytes, usage: storage });
    const lowers = device.createBuffer({ size: lowerCount * LOWER_U32 * 4, usage: storage });
    const uppers = device.createBuffer({ size: upperCount * UPPER_U32 * 4, usage: storage });
    const roots = device.createBuffer({ size: upperCount * 2 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'compact_upper', lowerCount, { 7: flags, 17: lowerKeys, 20: upperKeys, 21: upperFirst });
      run(pass, 'surface', cand + 1, { ...reader, 9: finalCand, 13: surfMasks, 14: surfCounts });
      this.scanner.plan(surfCounts, cand + 1).encode(pass);
      run(pass, 'write_leaves', cand, { 2: grid.leaves, 9: finalCand, 10: valueCounts, 13: surfMasks, 14: surfCounts, 15: leaves });
      run(pass, 'write_data', cand, { 2: grid.leaves, 3: grid.data, 5: bandCounts, 9: finalCand, 10: valueCounts, 16: data });
      run(pass, 'write_lowers', lowerCount, { ...reader, 10: valueCounts, 17: lowerKeys, 18: lowerFirst, 19: lowers });
      run(pass, 'write_uppers', upperCount, { ...reader, 19: lowers, 20: upperKeys, 21: upperFirst, 22: uppers, 28: flatLower });
      run(pass, 'write_roots', upperCount, { 20: upperKeys, 23: roots });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [surfaceVoxels] = await readBackTotals(device, [{ buffer: surfCounts, index: cand }]);
    const boundsOut = new Int32Array((await readBackU32(device, bounds, 6)).buffer);
    for (const b of [bandCounts, bounds, flags, hier, idx, finalKeys, finalCand, valueCounts, lowerKeys, lowerFirst, flatLower, flatVals, upperKeys, upperFirst, surfMasks, surfCounts, params]) {
      b.destroy();
    }

    return {
      roots,
      uppers,
      lowers,
      leaves,
      data,
      leafCount: cand,
      lowerCount,
      upperCount,
      dataElemCount: 2 + dataValues,
      activeVoxels: dataValues,
      surfaceVoxels,
      indexBoundsMin: [boundsOut[0], boundsOut[1], boundsOut[2]],
      indexBoundsMax: [boundsOut[3], boundsOut[4], boundsOut[5]],
    };
  }

  private params(cand: number, leafMin: [number, number, number], halfWidth: number): GPUBuffer {
    const device = this.device;
    const lowerMin = leafMin.map((v) => v >> 4);
    const upperMin = lowerMin.map((v) => v >> 5);
    const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([cand]));
    device.queue.writeBuffer(params, 12, new Float32Array([halfWidth]));
    device.queue.writeBuffer(params, 16, new Int32Array(leafMin));
    device.queue.writeBuffer(params, 32, new Int32Array(lowerMin));
    device.queue.writeBuffer(params, 48, new Int32Array(upperMin));
    return params;
  }

  private runner(params: GPUBuffer) {
    return (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>): void => {
      if (threads === 0) return;
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(
        0,
        this.device.createBindGroup({
          layout: this.pipelines[name].getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: params } },
            ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
          ],
        })
      );
      dispatch2D(pass, Math.ceil(threads / WG_SIZE));
    };
  }
}
