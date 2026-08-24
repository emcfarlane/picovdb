// Host side of wgsl/merge.wgsl. merge ORs two mask grids, for topology.
// mergeCsg is an SDF boolean of two op layer grids.

import mergeWgsl from 'picovdb/wgsl/merge.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals } from './device.ts';
import { GridWriter, preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';

const WG_SIZE = 256;

export interface MergeInput {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  leafCount: number;
}

export type CsgOp = 'union' | 'intersect' | 'subtract';
const CSG_OPS: Record<CsgOp, number> = { union: 0, intersect: 1, subtract: 2 };

export interface CsgOptions {
  halfWidth: number;
  op?: CsgOp;
}

export interface MergeResult {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  leafCount: number;
}

export class Merger {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, scanner: Scanner = new Scanner(device), sorter: Sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    const code = preludeWgsl + readerWgsl('a', 'params.a_count') + readerWgsl('b', 'params.b_count') + mergeWgsl;
    const module = device.createShaderModule({ code });
    for (const entryPoint of ['mark_unique', 'compact_unique', 'merge_masks', 'csg_mark', 'csg_apply', 'opgrid_compact']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  /** Topology merge: the union of the leaf tables with ORed masks. */
  async merge(a: MergeInput, b: MergeInput): Promise<MergeResult> {
    const device = this.device;
    const { params, run, outKeys, outCount } = await this.unionKeys(a, b);
    const outMasks = device.createBuffer({ size: outCount * 16 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    run(pass, 'merge_masks', outCount, { 3: outKeys, 4: a.leafKeys, 5: a.masks, 6: b.leafKeys, 7: b.masks, 8: outMasks });
    pass.end();
    device.queue.submit([encoder.finish()]);
    params.destroy();
    return { leafKeys: outKeys, masks: outMasks, leafCount: outCount };
  }

  /** SDF boolean of two op layer grids that share a key origin. */
  async mergeCsg(a: OpGrid, b: OpGrid, opts: CsgOptions): Promise<OpGrid> {
    const device = this.device;
    if (a.leafMin.some((v, i) => v !== b.leafMin[i])) throw new Error('grids have different key origins');
    const { params, run, outKeys, outCount } = await this.unionKeys(a, b);
    device.queue.writeBuffer(params, 16, new Float32Array([opts.halfWidth]));
    device.queue.writeBuffer(params, 20, new Uint32Array([CSG_OPS[opts.op ?? 'union']]));
    const inputs = { 4: a.leafKeys, 10: a.leaves, 11: a.data, 6: b.leafKeys, 12: b.leaves, 13: b.data };
    const writer = new GridWriter(device, outCount);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'csg_mark', outCount + 1, { ...inputs, 3: outKeys, ...writer.markBindings });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const leafMax = a.leafMax.map((v, i) => Math.max(v, b.leafMax[i])) as [number, number, number];
    const out = await writer.finish(this.scanner, this.pipelines['opgrid_compact'], outKeys, opts.halfWidth, { leafMin: a.leafMin, leafMax }, (pass, bindings, count) => {
      run(pass, 'csg_apply', count, { ...inputs, ...bindings });
    });
    outKeys.destroy();
    params.destroy();
    return out;
  }

  /** The sorted union of two leaf tables. */
  private async unionKeys(a: { leafKeys: GPUBuffer; leafCount: number }, b: { leafKeys: GPUBuffer; leafCount: number }) {
    const device = this.device;
    const concatCount = a.leafCount + b.leafCount;
    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([a.leafCount, b.leafCount, concatCount]));
    const run = (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>) => {
      if (threads === 0) return;
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: this.pipelines[name].getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: params } },
            ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
          ],
        })
      );
      dispatch2D(pass, Math.ceil(threads / WG_SIZE));
    };

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const concatKeys = device.createBuffer({ size: concatCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: concatCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (concatCount + 1) * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(a.leafKeys, 0, concatKeys, 0, a.leafCount * 4);
      encoder.copyBufferToBuffer(b.leafKeys, 0, concatKeys, a.leafCount * 4, b.leafCount * 4);
      const pass = encoder.beginComputePass();
      this.sorter.plan(concatKeys, sortVals, concatCount).encode(pass);
      run(pass, 'mark_unique', concatCount + 1, { 1: concatKeys, 2: flags });
      this.scanner.plan(flags, concatCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [outCount] = await readBackTotals(device, [{ buffer: flags, index: concatCount }]);
    device.queue.writeBuffer(params, 12, new Uint32Array([outCount]));
    const outKeys = device.createBuffer({ size: Math.max(outCount, 1) * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'compact_unique', concatCount, { 1: concatKeys, 2: flags, 3: outKeys });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    concatKeys.destroy();
    sortVals.destroy();
    flags.destroy();
    return { params, run, outKeys, outCount };
  }
}
