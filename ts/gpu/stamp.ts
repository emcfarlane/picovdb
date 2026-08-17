// Host side of wgsl/stamp.wgsl. Applies sphere brush stamps that add or
// carve material, the sculpting primitive of the grid only modeller.
// Works from an empty grid.

import stampWgsl from 'picovdb/wgsl/stamp.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { checkBindingSize, dispatch2D, readBackTotals } from './device.ts';
import type { OpGrid } from './emit.ts';

const WG_SIZE = 256;

export interface StampOptions {
  /** Brush center in absolute index space voxels. */
  center: [number, number, number];
  /** Brush radius in voxels. */
  radius: number;
  mode: 'add' | 'carve';
  halfWidth: number;
}

export class Stamper {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, scanner = new Scanner(device), sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    const module = device.createShaderModule({ code: stampWgsl });
    for (const entryPoint of ['generate_candidates', 'mark_unique', 'compact_unique', 'apply']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  async stamp(grid: OpGrid, opts: StampOptions): Promise<OpGrid> {
    const device = this.device;
    const hw = opts.halfWidth;
    // Candidate box of leaves the dilated stamp can touch. A stamp that
    // reaches the packed key range fails so nothing truncates silently.
    const rel = opts.center.map((c, a) => c - grid.leafMin[a] * 8);
    const lo = rel.map((c) => Math.floor((c - opts.radius - hw) / 8));
    const hi = rel.map((c) => Math.floor((c + opts.radius + hw) / 8));
    if (lo.some((v) => v < 0) || hi.some((v) => v > 1023)) {
      throw new Error('stamp reaches the leaf key space boundary');
    }
    const dims = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    const boxVol = dims[0] * dims[1] * dims[2];
    const concatCount = grid.leafCount + boxVol;

    const params = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([grid.leafCount, concatCount, 0, opts.mode === 'carve' ? 1 : 0]));
    device.queue.writeBuffer(params, 16, new Float32Array([rel[0], rel[1], rel[2], opts.radius, hw]));
    device.queue.writeBuffer(params, 48, new Int32Array(lo));
    device.queue.writeBuffer(params, 64, new Int32Array(dims));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const concatKeys = device.createBuffer({ size: concatCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: concatCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (concatCount + 1) * 4, usage: storage });

    const run = (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>) => {
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

    {
      const encoder = device.createCommandEncoder();
      if (grid.leafCount > 0) {
        encoder.copyBufferToBuffer(grid.leafKeys, 0, concatKeys, 0, grid.leafCount * 4);
      }
      const pass = encoder.beginComputePass();
      run(pass, 'generate_candidates', boxVol, { 3: concatKeys });
      this.sorter.plan(concatKeys, sortVals, concatCount).encode(pass);
      run(pass, 'mark_unique', concatCount + 1, { 3: concatKeys, 4: flags });
      this.scanner.plan(flags, concatCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [newCount] = await readBackTotals(device, [{ buffer: flags, index: concatCount }]);

    device.queue.writeBuffer(params, 8, new Uint32Array([newCount]));
    checkBindingSize(device, newCount * 512 * 4, 'stamped value slabs');
    const newKeys = device.createBuffer({ size: newCount * 4, usage: storage });
    const outValues = device.createBuffer({ size: newCount * 512 * 4, usage: storage });
    const outMasks = device.createBuffer({ size: newCount * 16 * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'compact_unique', concatCount, { 3: concatKeys, 4: flags, 5: newKeys });
      run(pass, 'apply', newCount, {
        1: grid.leafKeys, 2: grid.values, 5: newKeys, 6: outValues, 7: outMasks, 8: grid.masks,
      });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    return { leafKeys: newKeys, masks: outMasks, values: outValues, leafCount: newCount, leafMin: grid.leafMin, leafMax: grid.leafMax };
  }
}
