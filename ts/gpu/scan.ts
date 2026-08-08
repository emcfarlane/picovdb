// Host side of wgsl/scan.wgsl: device-wide exclusive prefix scan (u32),
// in place over a storage buffer.
//
//   const scanner = new Scanner(device);
//   const plan = scanner.plan(buffer, n); // reusable for a fixed buffer/n
//   plan.encode(pass);                    // inside a compute pass

import scanWgsl from 'picovdb/wgsl/scan.wgsl' with { type: 'text' };

const TILE = 1024;

interface ScanLevel {
  count: number; // workgroups = tiles at this level
  partials: GPUBuffer;
  bindGroup: GPUBindGroup;
}

export class Scanner {
  readonly device: GPUDevice;
  readonly layout: GPUBindGroupLayout;
  readonly scanTile: GPUComputePipeline;
  readonly addOffsets: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const module = device.createShaderModule({ code: scanWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.scanTile = device.createComputePipeline({ layout, compute: { module, entryPoint: 'scan_tile' } });
    this.addOffsets = device.createComputePipeline({ layout, compute: { module, entryPoint: 'add_offsets' } });
  }

  /** Plan an exclusive scan of the first `n` u32 elements of `buffer`. */
  plan(buffer: GPUBuffer, n: number): ScanPlan {
    return new ScanPlan(this, buffer, n);
  }
}

export class ScanPlan {
  private readonly scanner: Scanner;
  private readonly levels: ScanLevel[] = [];

  constructor(scanner: Scanner, buffer: GPUBuffer, n: number) {
    this.scanner = scanner;
    const { device } = scanner;
    let data = buffer;
    let size = n;
    for (;;) {
      const count = Math.max(1, Math.ceil(size / TILE));
      if (count > 65535) throw new Error(`scan of ${n} elements exceeds one dispatch dimension`);
      const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([size]));
      const partials = device.createBuffer({
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.levels.push({
        count,
        partials,
        bindGroup: device.createBindGroup({
          layout: scanner.layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: data } },
            { binding: 2, resource: { buffer: partials } },
          ],
        }),
      });
      if (count === 1) break;
      data = partials;
      size = count;
    }
  }

  encode(pass: GPUComputePassEncoder): void {
    pass.setPipeline(this.scanner.scanTile);
    for (const level of this.levels) {
      pass.setBindGroup(0, level.bindGroup);
      pass.dispatchWorkgroups(level.count);
    }
    pass.setPipeline(this.scanner.addOffsets);
    for (let i = this.levels.length - 2; i >= 0; i--) {
      pass.setBindGroup(0, this.levels[i].bindGroup);
      pass.dispatchWorkgroups(this.levels[i].count);
    }
  }
}
