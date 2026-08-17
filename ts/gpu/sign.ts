// Host side of wgsl/sign.wgsl. Computes inside parity for every voxel of
// the binned leaves as one mask per leaf in slab bit order.

import signWgsl from 'picovdb/wgsl/sign.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals } from './device.ts';
import type { BinResult } from './mesh_to_grid.ts';

const WG_SIZE = 256;

export interface SignResult {
  /** One 512 bit mask per leaf. A set bit marks an inside voxel. */
  inside: GPUBuffer;
  crossingCount: number;
}

export class Signer {
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
        entry(6, 'read-only-storage'),
        entry(7, 'storage'),
      ],
    });
    const module = device.createShaderModule({ code: signWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    for (const entryPoint of ['count_crossings', 'emit_crossings', 'sign_leaves']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout, compute: { module, entryPoint } });
    }
  }

  async sign(bin: BinResult): Promise<SignResult> {
    const device = this.device;
    const triangleCount = bin.triangles.size / 12; // 3 u32 indices per triangle
    // Column grid covering all candidate leaves. Parity per column does
    // not depend on the grid bounds.
    const minX = bin.leafMin[0] * 8;
    const minY = bin.leafMin[1] * 8;
    const nx = (bin.leafMax[0] - bin.leafMin[0] + 1) * 8;
    const ny = (bin.leafMax[1] - bin.leafMin[1] + 1) * 8;

    const params = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([triangleCount, 0]));
    device.queue.writeBuffer(params, 8, new Int32Array([minX, minY]));
    device.queue.writeBuffer(params, 16, new Uint32Array([nx, ny, bin.leafCount, 0]));
    device.queue.writeBuffer(params, 32, new Int32Array(bin.leafMin));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const counts = device.createBuffer({ size: (triangleCount + 1) * 4, usage: storage });
    const placeholder = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    const inside = device.createBuffer({ size: bin.leafCount * 16 * 4, usage: storage });

    const bindGroup = (crossCols: GPUBuffer, crossZ: GPUBuffer) =>
      device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: bin.pointsIndex } },
          { binding: 2, resource: { buffer: bin.triangles } },
          { binding: 3, resource: { buffer: counts } },
          { binding: 4, resource: { buffer: crossCols } },
          { binding: 5, resource: { buffer: crossZ } },
          { binding: 6, resource: { buffer: bin.leafKeys } },
          { binding: 7, resource: { buffer: inside } },
        ],
      });

    // Count crossings per triangle, scan, and read the total.
    const countScan = this.scanner.plan(counts, triangleCount + 1);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, bindGroup(placeholder, placeholder));
      pass.setPipeline(this.pipelines['count_crossings']);
      dispatch2D(pass, Math.ceil((triangleCount + 1) / WG_SIZE));
      countScan.encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [crossingCount] = await readBackTotals(device, [{ buffer: counts, index: triangleCount }]);

    // Emit, sort by column then height, and walk each leaf column.
    device.queue.writeBuffer(params, 4, new Uint32Array([crossingCount]));
    const crossCols = device.createBuffer({ size: Math.max(crossingCount, 1) * 4, usage: storage });
    const crossZ = device.createBuffer({ size: Math.max(crossingCount, 1) * 4, usage: storage });
    const group = bindGroup(crossCols, crossZ);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, group);
      pass.setPipeline(this.pipelines['emit_crossings']);
      dispatch2D(pass, Math.ceil(triangleCount / WG_SIZE));
      if (crossingCount > 0) {
        // Two stable sorts give column then height order.
        this.sorter.plan(crossZ, crossCols, crossingCount).encode(pass);
        this.sorter.plan(crossCols, crossZ, crossingCount).encode(pass);
      }
      pass.setBindGroup(0, group);
      pass.setPipeline(this.pipelines['sign_leaves']);
      dispatch2D(pass, bin.leafCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return { inside, crossingCount };
  }
}
