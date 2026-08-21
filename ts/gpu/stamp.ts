// Host side of wgsl/stamp.wgsl: stamps analytic shapes into op layer
// grids, adding or carving material.

import stampWgsl from 'picovdb/wgsl/stamp.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals } from './device.ts';
import { GridWriter, preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';

const WG_SIZE = 256;

export type Vec3 = [number, number, number];

/** Analytic brush shapes. All coordinates are absolute index space voxels. */
export type Shape =
  | { kind: 'sphere'; center: Vec3; radius: number }
  /** Half extents per axis, edges rounded by radius. */
  | { kind: 'box'; center: Vec3; half: Vec3; radius?: number }
  | { kind: 'capsule'; a: Vec3; b: Vec3; radius: number }
  | { kind: 'cylinder'; a: Vec3; b: Vec3; radius: number };

const SHAPES = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;

export interface StampOptions {
  shape: Shape;
  mode: 'add' | 'carve';
  halfWidth: number;
}

/** Voxel bounds of a shape's zero level set, before the band. */
export function shapeBounds(shape: Shape): { min: Vec3; max: Vec3 } {
  const min: Vec3 = [0, 0, 0];
  const max: Vec3 = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    if (shape.kind === 'sphere') {
      min[a] = shape.center[a] - shape.radius;
      max[a] = shape.center[a] + shape.radius;
    } else if (shape.kind === 'box') {
      min[a] = shape.center[a] - shape.half[a] - (shape.radius ?? 0);
      max[a] = shape.center[a] + shape.half[a] + (shape.radius ?? 0);
    } else {
      min[a] = Math.min(shape.a[a], shape.b[a]) - shape.radius;
      max[a] = Math.max(shape.a[a], shape.b[a]) + shape.radius;
    }
  }
  return { min, max };
}

export class Stamper {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, scanner: Scanner = new Scanner(device), sorter: Sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    const module = device.createShaderModule({ code: preludeWgsl + readerWgsl('old', 'params.old_count') + stampWgsl });
    for (const entryPoint of ['generate_candidates', 'mark_unique', 'compact_unique', 'mark', 'apply', 'opgrid_compact']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  async stamp(grid: OpGrid, opts: StampOptions): Promise<OpGrid> {
    const device = this.device;
    const hw = opts.halfWidth;
    const shape = opts.shape;
    // The box of leaves the shape's band can touch. A carve only changes
    // existing material, so its box clips to the grid's leaves. An add
    // that reaches the key range throws, so nothing truncates silently.
    const bounds = shapeBounds(shape);
    let lo = bounds.min.map((c, a) => Math.floor((c - hw) / 8) - grid.leafMin[a]);
    let hi = bounds.max.map((c, a) => Math.floor((c + hw) / 8) - grid.leafMin[a]);
    if (opts.mode === 'carve') {
      lo = lo.map((v) => Math.max(v, 0));
      hi = hi.map((v, a) => Math.min(v, grid.leafMax[a] - grid.leafMin[a]));
    } else if (lo.some((v) => v < 0) || hi.some((v) => v > 1023)) {
      throw new Error('stamp reaches the leaf key space boundary');
    }
    const rel = (p: Vec3) => p.map((c, a) => c - grid.leafMin[a] * 8);
    const p0 = shape.kind === 'sphere' || shape.kind === 'box' ? rel(shape.center) : rel(shape.a);
    const p1 = shape.kind === 'box' ? shape.half : shape.kind === 'sphere' ? [0, 0, 0] : rel(shape.b);
    const radius = shape.kind === 'box' ? (shape.radius ?? 0) : shape.radius;
    const dims = lo.map((v, a) => Math.max(hi[a] - v + 1, 0));
    const boxVol = dims[0] * dims[1] * dims[2];
    const concatCount = grid.leafCount + boxVol;

    const params = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([grid.leafCount, concatCount, 0, opts.mode === 'carve' ? 1 : 0]));
    device.queue.writeBuffer(params, 16, new Float32Array([p0[0], p0[1], p0[2], radius, p1[0], p1[1], p1[2], hw]));
    device.queue.writeBuffer(params, 48, new Int32Array(lo));
    device.queue.writeBuffer(params, 60, new Uint32Array([SHAPES[shape.kind]]));
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
    const newKeys = device.createBuffer({ size: newCount * 4, usage: storage });
    const old = { 1: grid.leafKeys, 2: grid.leaves, 6: grid.data };
    const writer = new GridWriter(device, newCount);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'compact_unique', concatCount, { 3: concatKeys, 4: flags, 5: newKeys });
      run(pass, 'mark', newCount + 1, { ...old, 5: newKeys, ...writer.markBindings });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    concatKeys.destroy();
    sortVals.destroy();
    flags.destroy();
    const leafMax = grid.leafMax.map((v, a) => (opts.mode === 'add' ? Math.max(v, hi[a] + grid.leafMin[a]) : v)) as Vec3;
    const out = await writer.finish(this.scanner, this.pipelines['opgrid_compact'], newKeys, hw, { leafMin: grid.leafMin, leafMax }, (pass, bindings, count) => {
      run(pass, 'apply', count, { ...old, ...bindings });
    });
    newKeys.destroy();
    params.destroy();
    return out;
  }
}
