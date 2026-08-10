// Host side of wgsl/prune.wgsl. ANDs leaf masks with a retain set and
// drops leaves left empty.

import pruneWgsl from 'picovdb/wgsl/prune.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { dispatch2D, readBackU32 } from './device.ts';

const WG_SIZE = 256;

export interface PruneResult {
  leafKeys: GPUBuffer;
  masks: GPUBuffer;
  leafCount: number;
}

export class Pruner {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly mark: GPUComputePipeline;
  readonly compact: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    this.scanner = new Scanner(device);
    const module = device.createShaderModule({ code: pruneWgsl });
    this.mark = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'mark' } });
    this.compact = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'compact' } });
  }

  async prune(leafKeys: GPUBuffer, masks: GPUBuffer, retain: GPUBuffer, leafCount: number): Promise<PruneResult> {
    const device = this.device;
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([leafCount]));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const flags = device.createBuffer({ size: (leafCount + 1) * 4, usage: storage });

    const group = (pipeline: GPUComputePipeline, buffers: Record<number, GPUBuffer>) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
        ],
      });

    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.mark);
      pass.setBindGroup(0, group(this.mark, { 2: masks, 3: retain, 4: flags }));
      dispatch2D(pass, Math.ceil((leafCount + 1) / WG_SIZE));
      this.scanner.plan(flags, leafCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const outCount = (await readBackU32(device, flags, leafCount + 1))[leafCount];

    const outKeys = device.createBuffer({ size: Math.max(outCount, 1) * 4, usage: storage });
    const outMasks = device.createBuffer({ size: Math.max(outCount, 1) * 16 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.compact);
      pass.setBindGroup(0, group(this.compact, { 1: leafKeys, 2: masks, 3: retain, 4: flags, 5: outKeys, 6: outMasks }));
      dispatch2D(pass, Math.ceil(leafCount / WG_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return { leafKeys: outKeys, masks: outMasks, leafCount: outCount };
  }
}
