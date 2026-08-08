// Host side of wgsl/radix_sort.wgsl: stable sort of u32 keys with u32
// payloads. 8 passes of 4-bit digits, ping-ponging through scratch buffers;
// the result lands back in the caller's buffers.
//
//   const sorter = new Sorter(device);
//   const plan = sorter.plan(keys, vals, n);
//   plan.encode(pass);

import sortWgsl from 'picovdb/wgsl/radix_sort.wgsl' with { type: 'text' };
import { Scanner, ScanPlan } from './scan.ts';

const TILE = 1024;
const RADIX = 16;
const PASSES = 8;

export class Sorter {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly layout: GPUBindGroupLayout;
  readonly histogram: GPUComputePipeline;
  readonly scatter: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    this.scanner = new Scanner(device);
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const module = device.createShaderModule({ code: sortWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.histogram = device.createComputePipeline({ layout, compute: { module, entryPoint: 'histogram' } });
    this.scatter = device.createComputePipeline({ layout, compute: { module, entryPoint: 'scatter' } });
  }

  /** Plan a sort of the first `n` (key, value) pairs; sorts in place. */
  plan(keys: GPUBuffer, vals: GPUBuffer, n: number): SortPlan {
    return new SortPlan(this, keys, vals, n);
  }
}

export class SortPlan {
  private readonly sorter: Sorter;
  private readonly numTiles: number;
  private readonly bindGroups: GPUBindGroup[] = []; // one per pass
  private readonly histScan: ScanPlan;

  constructor(sorter: Sorter, keys: GPUBuffer, vals: GPUBuffer, n: number) {
    this.sorter = sorter;
    const { device } = sorter;
    this.numTiles = Math.max(1, Math.ceil(n / TILE));
    if (this.numTiles > 65535) throw new Error(`sort of ${n} elements exceeds one dispatch dimension`);

    const scratch = { size: Math.max(n, 1) * 4, usage: GPUBufferUsage.STORAGE };
    const keysB = device.createBuffer(scratch);
    const valsB = device.createBuffer(scratch);
    const hist = device.createBuffer({
      size: RADIX * this.numTiles * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.histScan = sorter.scanner.plan(hist, RADIX * this.numTiles);

    for (let pass = 0; pass < PASSES; pass++) {
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([n, pass * 4, this.numTiles]));
      const forward = pass % 2 === 0;
      this.bindGroups.push(
        device.createBindGroup({
          layout: sorter.layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: forward ? keys : keysB } },
            { binding: 2, resource: { buffer: forward ? vals : valsB } },
            { binding: 3, resource: { buffer: forward ? keysB : keys } },
            { binding: 4, resource: { buffer: forward ? valsB : vals } },
            { binding: 5, resource: { buffer: hist } },
          ],
        })
      );
    }
  }

  encode(pass: GPUComputePassEncoder): void {
    for (const bindGroup of this.bindGroups) {
      pass.setPipeline(this.sorter.histogram);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(this.numTiles);
      this.histScan.encode(pass);
      pass.setPipeline(this.sorter.scatter);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(this.numTiles);
    }
  }
}
