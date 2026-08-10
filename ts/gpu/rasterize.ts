// Host side of wgsl/rasterize.wgsl. Fills each binned leaf's slab with
// the minimum squared distance to any incident triangle in voxel units,
// with infinity where no triangle is within the band.

import rasterWgsl from 'picovdb/wgsl/rasterize.wgsl' with { type: 'text' };
import { dispatch2D } from './device.ts';
import type { BinResult } from './mesh_to_grid.ts';

export interface RasterizeOptions {
  /** Narrow band half width in voxels. Must match the binning pass. */
  halfWidth: number;
}

export class Rasterizer {
  readonly device: GPUDevice;
  readonly layout: GPUBindGroupLayout;
  readonly rasterizePipeline: GPUComputePipeline;

  constructor(device: GPUDevice) {
    this.device = device;
    const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.layout = device.createBindGroupLayout({
      entries: [
        entry(0, 'uniform'),
        entry(1, 'read-only-storage'),
        entry(2, 'read-only-storage'),
        entry(3, 'read-only-storage'),
        entry(4, 'read-only-storage'),
        entry(5, 'read-only-storage'),
        entry(6, 'storage'),
      ],
    });
    const module = device.createShaderModule({ code: rasterWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.rasterizePipeline = device.createComputePipeline({ layout, compute: { module, entryPoint: 'rasterize' } });
  }

  /** Returns one slab of 512 squared distances per leaf as f32 bits. */
  rasterize(bin: BinResult, opts: RasterizeOptions): GPUBuffer {
    const device = this.device;
    const valueBytes = bin.leafCount * 512 * 4;
    if (valueBytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error(`${bin.leafCount} leaf slabs (${valueBytes} bytes) exceed the storage binding limit`);
    }

    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([bin.pairCount, bin.leafCount]));
    device.queue.writeBuffer(params, 8, new Float32Array([opts.halfWidth]));
    device.queue.writeBuffer(params, 16, new Int32Array(bin.leafMin));

    const leafValues = device.createBuffer({ size: valueBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bindGroup = device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: bin.pointsIndex } },
        { binding: 2, resource: { buffer: bin.triangles } },
        { binding: 3, resource: { buffer: bin.pairKeys } },
        { binding: 4, resource: { buffer: bin.pairTris } },
        { binding: 5, resource: { buffer: bin.leafKeys } },
        { binding: 6, resource: { buffer: leafValues } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bindGroup);
    pass.setPipeline(this.rasterizePipeline);
    dispatch2D(pass, bin.leafCount);
    pass.end();
    device.queue.submit([encoder.finish()]);
    return leafValues;
  }
}
