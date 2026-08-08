// Host side of wgsl/emit.wgsl. Stage 4 of the GPU mesh-to-grid pipeline:
// turn the distance slabs and inside masks into the picovdb node buffers
// (roots/uppers/lowers/leaves/data), GPU-resident and in the same layout
// the renderer uploads. reEmit() runs the same emission from a grid-op
// result (keys + band masks + signed values), closing the edit loop.
//
//   const emitter = new Emitter(device);
//   const tree = await emitter.emit(bin, leafValues, sign, { halfWidth });
//   const grid = await emitter.classifyOnly(bin, leafValues, sign, { halfWidth });
//   const tree2 = await emitter.reEmit(editedGrid, { halfWidth });

import emitWgsl from 'picovdb/wgsl/emit.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackU32 } from './device.ts';
import type { BinResult } from './mesh_to_grid.ts';
import type { SignResult } from './sign.ts';

const WG_SIZE = 256;
const LEAF_U32 = 52;
const LOWER_U32 = 388;
const UPPER_U32 = 3076;

export interface EmitOptions {
  /** Narrow band half-width in voxels; must match the earlier stages. */
  halfWidth: number;
}

/** A grid at the op layer: sorted leaf keys, band masks, signed value slabs. */
export interface OpGrid {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  values: GPUBuffer;
  leafCount: number;
  leafMin: [number, number, number];
}

export interface EmitResult {
  roots: GPUBuffer; // upperCount x 2 u32 keys (unpadded)
  uppers: GPUBuffer;
  lowers: GPUBuffer;
  leaves: GPUBuffer;
  data: GPUBuffer; // f32 values; [0]=+hw [1]=-hw, then per-voxel, zero-padded to 16B
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

interface Prep {
  params: GPUBuffer;
  candKeys: GPUBuffer;
  values: GPUBuffer;
  bandMasks: GPUBuffer;
  bandCounts: GPUBuffer;
  bounds: GPUBuffer;
  flags: GPUBuffer;
  cand: number;
}

export class Emitter {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice) {
    this.device = device;
    this.scanner = new Scanner(device);
    this.sorter = new Sorter(device);
    const module = device.createShaderModule({ code: emitWgsl });
    for (const entryPoint of [
      'classify', 'classify_re', 'mark_band', 'compact_band', 'hier_lo', 'hier_hi', 'reorder_final',
      'leaf_value_counts', 'mark_lower', 'compact_lower', 'mark_upper', 'compact_upper',
      'surface', 'write_leaves', 'write_data', 'write_lowers', 'write_uppers', 'write_roots',
    ]) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  /** Full emission from the converter stages. */
  async emit(bin: BinResult, leafValues: GPUBuffer, sign: SignResult, opts: EmitOptions): Promise<EmitResult> {
    const prep = this.prepare(bin.leafKeys, leafValues, bin.leafCount, bin.leafMin, opts, null);
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      this.run(pass, 'classify', prep.cand + 1, {
        1: prep.candKeys, 2: prep.values, 3: sign.inside, 4: prep.bandMasks, 5: prep.bandCounts, 6: prep.bounds,
      }, prep.params);
      this.markBand(pass, prep);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
    return this.core(prep);
  }

  /**
   * Run only the classification pass, turning converter outputs into an
   * op-layer grid: leafValues become signed in place, masks mark the band.
   */
  async classifyOnly(bin: BinResult, leafValues: GPUBuffer, sign: SignResult, opts: EmitOptions): Promise<OpGrid> {
    const prep = this.prepare(bin.leafKeys, leafValues, bin.leafCount, bin.leafMin, opts, null);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.run(pass, 'classify', prep.cand + 1, {
      1: prep.candKeys, 2: prep.values, 3: sign.inside, 4: prep.bandMasks, 5: prep.bandCounts, 6: prep.bounds,
    }, prep.params);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return { leafKeys: bin.leafKeys, masks: prep.bandMasks, values: leafValues, leafCount: bin.leafCount, leafMin: bin.leafMin };
  }

  /** Emission from an op-layer grid (e.g. after dilate/prune/merge). */
  async reEmit(grid: OpGrid, opts: EmitOptions): Promise<EmitResult> {
    const prep = this.prepare(grid.leafKeys, grid.values, grid.leafCount, grid.leafMin, opts, grid.masks);
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      this.run(pass, 'classify_re', prep.cand + 1, {
        1: prep.candKeys, 4: prep.bandMasks, 5: prep.bandCounts, 6: prep.bounds,
      }, prep.params);
      this.markBand(pass, prep);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }
    return this.core(prep);
  }

  private prepare(
    candKeys: GPUBuffer,
    values: GPUBuffer,
    cand: number,
    leafMin: [number, number, number],
    opts: EmitOptions,
    masks: GPUBuffer | null
  ): Prep {
    const device = this.device;
    const lowerMin = leafMin.map((v) => v >> 4);
    const upperMin = lowerMin.map((v) => v >> 5);
    const params = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([cand]));
    device.queue.writeBuffer(params, 16, new Float32Array([opts.halfWidth]));
    device.queue.writeBuffer(params, 32, new Int32Array(leafMin));
    device.queue.writeBuffer(params, 48, new Int32Array(lowerMin));
    device.queue.writeBuffer(params, 64, new Int32Array(upperMin));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const bandMasks = masks ?? device.createBuffer({ size: cand * 16 * 4, usage: storage });
    const bandCounts = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    const bounds = device.createBuffer({ size: 24, usage: storage });
    device.queue.writeBuffer(bounds, 0, new Int32Array([0x7fffffff, 0x7fffffff, 0x7fffffff, -0x80000000, -0x80000000, -0x80000000]));
    const flags = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    return { params, candKeys, values, bandMasks, bandCounts, bounds, flags, cand };
  }

  private markBand(pass: GPUComputePassEncoder, prep: Prep): void {
    this.run(pass, 'mark_band', prep.cand + 1, { 5: prep.bandCounts, 7: prep.flags }, prep.params);
    this.scanner.plan(prep.flags, prep.cand + 1).encode(pass);
  }

  private async core(prep: Prep): Promise<EmitResult> {
    const device = this.device;
    const { params, candKeys, values, bandMasks, bandCounts, bounds, flags, cand } = prep;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const finalCount = (await readBackU32(device, flags, cand + 1))[cand];
    if (finalCount === 0) throw new Error('no active voxels');

    // Compact band leaves, re-sort into the CPU's depth-first emission order
    // via the two-word hierarchical key, then value offsets and lowers.
    device.queue.writeBuffer(params, 4, new Uint32Array([finalCount]));
    const tmpKeys = device.createBuffer({ size: finalCount * 4, usage: storage });
    const tmpCand = device.createBuffer({ size: finalCount * 4, usage: storage });
    const hier = device.createBuffer({ size: finalCount * 4, usage: storage });
    const idx = device.createBuffer({ size: finalCount * 4, usage: storage });
    const finalKeys = device.createBuffer({ size: finalCount * 4, usage: storage });
    const finalCand = device.createBuffer({ size: finalCount * 4, usage: storage });
    const valueCounts = device.createBuffer({ size: (finalCount + 1) * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      this.run(pass, 'compact_band', cand, { 1: candKeys, 5: bandCounts, 7: flags, 24: tmpKeys, 25: tmpCand }, params);
      this.run(pass, 'hier_lo', finalCount, { 24: tmpKeys, 26: hier, 27: idx }, params);
      this.sorter.plan(hier, idx, finalCount).encode(pass);
      this.run(pass, 'hier_hi', finalCount, { 24: tmpKeys, 26: hier, 27: idx }, params);
      this.sorter.plan(hier, idx, finalCount).encode(pass);
      this.run(pass, 'reorder_final', finalCount, { 8: finalKeys, 9: finalCand, 24: tmpKeys, 25: tmpCand, 27: idx }, params);
      this.run(pass, 'leaf_value_counts', finalCount + 1, { 5: bandCounts, 9: finalCand, 10: valueCounts }, params);
      this.scanner.plan(valueCounts, finalCount + 1).encode(pass);
      this.run(pass, 'mark_lower', finalCount + 1, { 7: flags, 8: finalKeys }, params);
      this.scanner.plan(flags, finalCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const dataValues = (await readBackU32(device, valueCounts, finalCount + 1))[finalCount];
    const lowerCount = (await readBackU32(device, flags, finalCount + 1))[finalCount];

    device.queue.writeBuffer(params, 8, new Uint32Array([lowerCount]));
    const lowerKeys = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const lowerFirst = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const flatLower = device.createBuffer({ size: lowerCount * 4, usage: storage });
    const flatVals = device.createBuffer({ size: lowerCount * 4, usage: GPUBufferUsage.STORAGE });
    {
      const encoder = device.createCommandEncoder();
      const passA = encoder.beginComputePass();
      this.run(passA, 'compact_lower', finalCount, { 7: flags, 8: finalKeys, 17: lowerKeys, 18: lowerFirst }, params);
      passA.end();
      encoder.copyBufferToBuffer(lowerKeys, 0, flatLower, 0, lowerCount * 4);
      const passB = encoder.beginComputePass();
      this.sorter.plan(flatLower, flatVals, lowerCount).encode(passB);
      this.run(passB, 'mark_upper', lowerCount + 1, { 7: flags, 17: lowerKeys }, params);
      this.scanner.plan(flags, lowerCount + 1).encode(passB);
      passB.end();
      device.queue.submit([encoder.finish()]);
    }
    const upperCount = (await readBackU32(device, flags, lowerCount + 1))[lowerCount];

    // Surface masks + all node/value outputs.
    device.queue.writeBuffer(params, 12, new Uint32Array([upperCount]));
    const upperKeys = device.createBuffer({ size: upperCount * 4, usage: storage });
    const upperFirst = device.createBuffer({ size: upperCount * 4, usage: storage });
    const surfMasks = device.createBuffer({ size: finalCount * 16 * 4, usage: storage });
    const surfCounts = device.createBuffer({ size: (finalCount + 1) * 4, usage: storage });
    const leaves = device.createBuffer({ size: finalCount * LEAF_U32 * 4, usage: storage });
    const dataBytes = Math.ceil(((2 + dataValues) * 4) / 16) * 16;
    const data = device.createBuffer({ size: dataBytes, usage: storage });
    const lowers = device.createBuffer({ size: lowerCount * LOWER_U32 * 4, usage: storage });
    const uppers = device.createBuffer({ size: upperCount * UPPER_U32 * 4, usage: storage });
    const roots = device.createBuffer({ size: upperCount * 2 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      this.run(pass, 'compact_upper', lowerCount, { 7: flags, 17: lowerKeys, 20: upperKeys, 21: upperFirst }, params);
      this.run(pass, 'surface', finalCount + 1, {
        1: candKeys, 2: values, 4: bandMasks, 9: finalCand, 13: surfMasks, 14: surfCounts, 24: tmpKeys, 25: tmpCand,
      }, params);
      this.scanner.plan(surfCounts, finalCount + 1).encode(pass);
      this.run(pass, 'write_leaves', finalCount, {
        2: values, 4: bandMasks, 9: finalCand, 10: valueCounts, 13: surfMasks, 14: surfCounts, 15: leaves,
      }, params);
      this.run(pass, 'write_data', finalCount, { 2: values, 4: bandMasks, 9: finalCand, 10: valueCounts, 16: data }, params);
      this.run(pass, 'write_lowers', lowerCount, {
        2: values, 10: valueCounts, 17: lowerKeys, 18: lowerFirst, 19: lowers, 24: tmpKeys, 25: tmpCand,
      }, params);
      this.run(pass, 'write_uppers', upperCount, {
        2: values, 19: lowers, 20: upperKeys, 21: upperFirst, 22: uppers, 24: tmpKeys, 25: tmpCand, 28: flatLower,
      }, params);
      this.run(pass, 'write_roots', upperCount, { 20: upperKeys, 23: roots }, params);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const surfaceVoxels = (await readBackU32(device, surfCounts, finalCount + 1))[finalCount];
    const boundsOut = new Int32Array((await readBackU32(device, bounds, 6)).buffer);

    return {
      roots,
      uppers,
      lowers,
      leaves,
      data,
      leafCount: finalCount,
      lowerCount,
      upperCount,
      dataElemCount: 2 + dataValues,
      activeVoxels: dataValues,
      surfaceVoxels,
      indexBoundsMin: [boundsOut[0], boundsOut[1], boundsOut[2]],
      indexBoundsMax: [boundsOut[3], boundsOut[4], boundsOut[5]],
    };
  }

  private run(
    pass: GPUComputePassEncoder,
    name: string,
    threads: number,
    buffers: Record<number, GPUBuffer>,
    params: GPUBuffer
  ): void {
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
  }
}
