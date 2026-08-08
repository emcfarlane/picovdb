// Host side of wgsl/merge.wgsl: union two grids' leaf tables, OR their
// masks, and (when value slabs are provided) min-combine values — the SDF
// union.
//
//   const merger = new Merger(device);
//   const out = await merger.merge(a, b);

import mergeWgsl from 'picovdb/wgsl/merge.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackU32 } from './device.ts';

const WG_SIZE = 256;

export interface MergeInput {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  leafCount: number;
  /** Optional leafCount x 512 f32 value slabs. */
  values?: GPUBuffer;
}

export interface MergeResult {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  leafCount: number;
  values?: GPUBuffer;
}

export class Merger {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice) {
    this.device = device;
    this.scanner = new Scanner(device);
    this.sorter = new Sorter(device);
    const module = device.createShaderModule({ code: mergeWgsl });
    for (const entryPoint of ['mark_unique', 'compact_unique', 'merge_masks', 'merge_values']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  async merge(a: MergeInput, b: MergeInput): Promise<MergeResult> {
    const device = this.device;
    const concatCount = a.leafCount + b.leafCount;
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([a.leafCount, b.leafCount, concatCount]));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const concatKeys = device.createBuffer({ size: concatCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: concatCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (concatCount + 1) * 4, usage: storage });

    const group = (name: string, buffers: Record<number, GPUBuffer>) =>
      device.createBindGroup({
        layout: this.pipelines[name].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
        ],
      });
    const run = (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>) => {
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, group(name, buffers));
      dispatch2D(pass, Math.ceil(threads / WG_SIZE));
    };

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
    const outCount = (await readBackU32(device, flags, concatCount + 1))[concatCount];

    device.queue.writeBuffer(params, 12, new Uint32Array([outCount]));
    const outKeys = device.createBuffer({ size: outCount * 4, usage: storage });
    const outMasks = device.createBuffer({ size: outCount * 16 * 4, usage: storage });
    const outValues = a.values && b.values
      ? device.createBuffer({ size: outCount * 512 * 4, usage: storage })
      : undefined;
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'compact_unique', concatCount, { 1: concatKeys, 2: flags, 3: outKeys });
      run(pass, 'merge_masks', outCount, {
        3: outKeys, 4: a.leafKeys, 5: a.masks, 6: b.leafKeys, 7: b.masks, 8: outMasks,
      });
      if (outValues) {
        run(pass, 'merge_values', outCount, {
          3: outKeys, 4: a.leafKeys, 5: a.values!, 6: b.leafKeys, 7: b.values!, 8: outValues,
        });
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return { leafKeys: outKeys, masks: outMasks, leafCount: outCount, values: outValues };
  }
}
