// Host side of wgsl/extract.wgsl: a level set of an op layer grid as
// triangles, for redistancing and export.

import extractWgsl from 'picovdb/wgsl/extract.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { createU32Buffer, dispatch2D, readBackTotals } from './device.ts';
import { MC_TRI_COUNT, MC_TRI_TABLE } from './mc_table.ts';
import { preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';

export interface Mesh {
  /** xyz f32 triples, three unshared vertices per triangle, in the grid's relative voxel coordinates. */
  points: GPUBuffer;
  /** Vertex index triples 0..3n. */
  triangles: GPUBuffer;
  triangleCount: number;
}

export class Extractor {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};
  private readonly table: GPUBuffer;

  constructor(device: GPUDevice, scanner = new Scanner(device)) {
    this.device = device;
    this.scanner = scanner;
    const module = device.createShaderModule({ code: preludeWgsl + readerWgsl('old', 'params.leaf_count') + extractWgsl });
    for (const entryPoint of ['count', 'emit']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
    const table = new Int32Array(MC_TRI_TABLE.length + MC_TRI_COUNT.length);
    table.set(MC_TRI_TABLE);
    table.set(MC_TRI_COUNT, MC_TRI_TABLE.length);
    this.table = createU32Buffer(device, new Uint32Array(table.buffer));
  }

  /** The level set at iso as triangles. iso must lie inside the band. */
  async extract(grid: OpGrid, halfWidth: number, iso = 0): Promise<Mesh> {
    const device = this.device;
    if (grid.leafCount === 0) throw new Error('empty grid');
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([grid.leafCount]));
    device.queue.writeBuffer(params, 4, new Float32Array([halfWidth, iso]));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const counts = device.createBuffer({ size: (grid.leafCount + 1) * 4, usage: storage });

    // Auto layouts only hold the bindings an entry point uses.
    const run = (pass: GPUComputePassEncoder, name: string, extra: GPUBindGroupEntry[]) => {
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: this.pipelines[name].getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: grid.leafKeys } },
            { binding: 2, resource: { buffer: grid.leaves } },
            { binding: 7, resource: { buffer: grid.data } },
            { binding: 3, resource: { buffer: this.table } },
            { binding: 4, resource: { buffer: counts } },
            ...extra,
          ],
        })
      );
      dispatch2D(pass, grid.leafCount);
    };
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'count', []);
      this.scanner.plan(counts, grid.leafCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [triangleCount] = await readBackTotals(device, [{ buffer: counts, index: grid.leafCount }]);
    if (triangleCount === 0) throw new Error('no surface');
    const points = device.createBuffer({ size: triangleCount * 9 * 4, usage: storage });
    const triangles = device.createBuffer({ size: triangleCount * 3 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'emit', [
        { binding: 5, resource: { buffer: points } },
        { binding: 6, resource: { buffer: triangles } },
      ]);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    counts.destroy();
    return { points, triangles, triangleCount };
  }
}
