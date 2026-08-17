// Host side of wgsl/dilate.wgsl. Dilates an active voxel mask set by one
// voxel across face neighbors, growing the leaf table where masks spill
// across leaf boundaries.

import dilateWgsl from 'picovdb/wgsl/dilate.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals } from './device.ts';

const WG_SIZE = 256;

export interface DilateResult {
  /** Sorted unique leaf keys including spawned face neighbors. */
  leafKeys: GPUBuffer;
  /** Dilated masks with 16 words per leaf. */
  masks: GPUBuffer;
  leafCount: number;
}

export class Dilator {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  readonly layout: GPUBindGroupLayout;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, scanner = new Scanner(device), sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
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
        entry(3, 'storage'),
        entry(4, 'storage'),
        entry(5, 'storage'),
        entry(6, 'storage'),
        entry(7, 'storage'),
        entry(8, 'storage'),
      ],
    });
    const module = device.createShaderModule({ code: dilateWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    for (const entryPoint of ['count_spawn', 'emit_spawn', 'mark_unique', 'compact_unique', 'dilate_masks']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout, compute: { module, entryPoint } });
    }
  }

  async dilate(leafKeys: GPUBuffer, masks: GPUBuffer, leafCount: number): Promise<DilateResult> {
    const device = this.device;
    if (leafCount === 0) {
      return { leafKeys, masks, leafCount: 0 };
    }
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([leafCount]));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const counts = device.createBuffer({ size: (leafCount + 1) * 4, usage: storage });
    const clipped = device.createBuffer({ size: 4, usage: storage });
    const placeholder = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });

    const bindGroup = (spawnKeys: GPUBuffer, flags: GPUBuffer, newKeys: GPUBuffer, newMasks: GPUBuffer) =>
      device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: leafKeys } },
          { binding: 2, resource: { buffer: masks } },
          { binding: 3, resource: { buffer: counts } },
          { binding: 4, resource: { buffer: spawnKeys } },
          { binding: 5, resource: { buffer: flags } },
          { binding: 6, resource: { buffer: newKeys } },
          { binding: 7, resource: { buffer: newMasks } },
          { binding: 8, resource: { buffer: clipped } },
        ],
      });

    // Count spawned leaves, scan, and read the total.
    const countScan = this.scanner.plan(counts, leafCount + 1);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindGroup(placeholder, placeholder, placeholder, placeholder));
      pass.setPipeline(this.pipelines['count_spawn']);
      dispatch2D(pass, Math.ceil((leafCount + 1) / WG_SIZE));
      countScan.encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [spawnCount, clippedCount] = await readBackTotals(device, [
      { buffer: counts, index: leafCount },
      { buffer: clipped, index: 0 },
    ]);
    if (clippedCount > 0) {
      throw new Error(`dilation spills past the leaf key space boundary at ${clippedCount} faces`);
    }

    // Emit, sort, and dedupe.
    device.queue.writeBuffer(params, 4, new Uint32Array([spawnCount]));
    const spawnKeys = device.createBuffer({ size: spawnCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: spawnCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (spawnCount + 1) * 4, usage: storage });
    const newKeys = device.createBuffer({ size: spawnCount * 4, usage: storage });
    const flagScan = this.scanner.plan(flags, spawnCount + 1);
    const group = bindGroup(spawnKeys, flags, newKeys, placeholder);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, group);
      pass.setPipeline(this.pipelines['emit_spawn']);
      dispatch2D(pass, Math.ceil(leafCount / WG_SIZE));
      this.sorter.plan(spawnKeys, sortVals, spawnCount).encode(pass);
      pass.setBindGroup(0, group);
      pass.setPipeline(this.pipelines['mark_unique']);
      dispatch2D(pass, Math.ceil((spawnCount + 1) / WG_SIZE));
      flagScan.encode(pass);
      pass.setBindGroup(0, group);
      pass.setPipeline(this.pipelines['compact_unique']);
      dispatch2D(pass, Math.ceil(spawnCount / WG_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [newCount] = await readBackTotals(device, [{ buffer: flags, index: spawnCount }]);

    // Build the dilated masks.
    device.queue.writeBuffer(params, 8, new Uint32Array([newCount]));
    const newMasks = device.createBuffer({ size: newCount * 16 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindGroup(spawnKeys, flags, newKeys, newMasks));
      pass.setPipeline(this.pipelines['dilate_masks']);
      dispatch2D(pass, Math.ceil(newCount / WG_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return { leafKeys: newKeys, masks: newMasks, leafCount: newCount };
  }
}
