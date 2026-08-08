// Host side of wgsl/mesh_to_grid.wgsl. Stage 1 of the GPU mesh-to-grid
// pipeline: bin triangles into the 8^3 leaf blocks their half-width-dilated
// bounds touch, producing the sorted (leaf key, triangle) pair list and the
// deduplicated leaf table, all left on the GPU.
//
//   const binner = new Binner(device);
//   const result = await binner.bin(points, triangles, { voxelSize, halfWidth });

import binWgsl from 'picovdb/wgsl/mesh_to_grid.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { readBackU32 } from './device.ts';

const WG_SIZE = 256;

export interface BinOptions {
  /** World units per voxel. */
  voxelSize: number;
  /** Narrow band half-width in voxels. */
  halfWidth: number;
}

export interface BinResult {
  /** (leaf key, triangle index) pairs sorted by key. */
  pairKeys: GPUBuffer;
  pairTris: GPUBuffer;
  pairCount: number;
  /** Deduplicated leaf keys, sorted. */
  leafKeys: GPUBuffer;
  leafCount: number;
  /** Leaf-space bias: leaf coordinate = unpacked key + leafMin; voxel origin = coordinate * 8. */
  leafMin: [number, number, number];
}

export class Binner {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  readonly layout: GPUBindGroupLayout;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice) {
    this.device = device;
    this.scanner = new Scanner(device);
    this.sorter = new Sorter(device);
    const storage = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type },
    });
    this.layout = device.createBindGroupLayout({
      entries: [
        storage(0, 'uniform'),
        storage(1, 'read-only-storage'),
        storage(2, 'storage'),
        storage(3, 'read-only-storage'),
        storage(4, 'storage'),
        storage(5, 'storage'),
        storage(6, 'storage'),
        storage(7, 'storage'),
        storage(8, 'storage'),
      ],
    });
    const module = device.createShaderModule({ code: binWgsl });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    for (const entryPoint of ['transform_points', 'count_pairs', 'emit_pairs', 'mark_unique', 'compact_unique']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout, compute: { module, entryPoint } });
    }
  }

  async bin(points: Float32Array<ArrayBuffer>, triangles: Uint32Array<ArrayBuffer>, opts: BinOptions): Promise<BinResult> {
    const device = this.device;
    const pointCount = points.length / 3;
    const triangleCount = triangles.length / 3;
    if (triangleCount === 0) throw new Error('empty mesh');
    const invVoxelSize = Math.fround(1 / opts.voxelSize);
    const leafMin = leafBoundsMin(points, invVoxelSize, opts.halfWidth);

    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([pointCount, triangleCount]));
    device.queue.writeBuffer(params, 8, new Float32Array([invVoxelSize, opts.halfWidth]));
    device.queue.writeBuffer(params, 16, new Int32Array(leafMin));

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const pointsWorld = device.createBuffer({ size: points.byteLength, usage: storage });
    device.queue.writeBuffer(pointsWorld, 0, points);
    const pointsIndex = device.createBuffer({ size: points.byteLength, usage: GPUBufferUsage.STORAGE });
    const trianglesBuf = device.createBuffer({ size: triangles.byteLength, usage: storage });
    device.queue.writeBuffer(trianglesBuf, 0, triangles);
    const counts = device.createBuffer({ size: (triangleCount + 1) * 4, usage: storage });
    const placeholder = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });

    const bindGroup = (pairKeys: GPUBuffer, pairTris: GPUBuffer, flags: GPUBuffer, uniqueKeys: GPUBuffer) =>
      device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: pointsWorld } },
          { binding: 2, resource: { buffer: pointsIndex } },
          { binding: 3, resource: { buffer: trianglesBuf } },
          { binding: 4, resource: { buffer: counts } },
          { binding: 5, resource: { buffer: pairKeys } },
          { binding: 6, resource: { buffer: pairTris } },
          { binding: 7, resource: { buffer: flags } },
          { binding: 8, resource: { buffer: uniqueKeys } },
        ],
      });

    // Phase 1: transform + count + scan, then read the total pair count.
    const countGroup = bindGroup(placeholder, placeholder, placeholder, placeholder);
    const countScan = this.scanner.plan(counts, triangleCount + 1);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, countGroup);
      this.dispatch(pass, 'transform_points', pointCount * 3);
      this.dispatch(pass, 'count_pairs', triangleCount + 1);
      countScan.encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const pairCount = (await readBackU32(device, counts, triangleCount + 1))[triangleCount];
    if (pairCount === 0) throw new Error('no leaves touched (degenerate mesh?)');

    // Phase 2: emit, sort by key, mark/compact unique leaves.
    device.queue.writeBuffer(params, 28, new Uint32Array([pairCount]));
    const pairKeys = device.createBuffer({ size: pairCount * 4, usage: storage });
    const pairTris = device.createBuffer({ size: pairCount * 4, usage: storage });
    const flags = device.createBuffer({ size: (pairCount + 1) * 4, usage: storage });
    const leafKeys = device.createBuffer({ size: pairCount * 4, usage: storage });
    const pairGroup = bindGroup(pairKeys, pairTris, flags, leafKeys);
    const sortPlan = this.sorter.plan(pairKeys, pairTris, pairCount);
    const flagScan = this.scanner.plan(flags, pairCount + 1);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setBindGroup(0, pairGroup);
      this.dispatch(pass, 'emit_pairs', triangleCount);
      sortPlan.encode(pass);
      pass.setBindGroup(0, pairGroup);
      this.dispatch(pass, 'mark_unique', pairCount + 1);
      flagScan.encode(pass);
      pass.setBindGroup(0, pairGroup);
      this.dispatch(pass, 'compact_unique', pairCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const leafCount = (await readBackU32(device, flags, pairCount + 1))[pairCount];

    return { pairKeys, pairTris, pairCount, leafKeys, leafCount, leafMin };
  }

  private dispatch(pass: GPUComputePassEncoder, entryPoint: string, threads: number): void {
    const groups = Math.ceil(threads / WG_SIZE);
    if (groups > 65535) throw new Error(`dispatch of ${threads} threads exceeds one dimension`);
    pass.setPipeline(this.pipelines[entryPoint]);
    pass.dispatchWorkgroups(groups);
  }
}

/** Minimum leaf coordinate over all dilated vertex bounds, f32-exact to the GPU math. */
function leafBoundsMin(points: Float32Array, invVoxelSize: number, halfWidth: number): [number, number, number] {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < points.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const p = Math.fround(points[i + axis] * invVoxelSize);
      if (p < min[axis]) min[axis] = p;
      if (p > max[axis]) max[axis] = p;
    }
  }
  const lo: number[] = [];
  for (let axis = 0; axis < 3; axis++) {
    const loLeaf = Math.ceil(Math.fround(min[axis] - halfWidth)) >> 3;
    const hiLeaf = Math.floor(Math.fround(max[axis] + halfWidth)) >> 3;
    if (hiLeaf - loLeaf >= 1024) {
      throw new Error(`grid exceeds 1024 leaves on axis ${axis}: ${loLeaf}..${hiLeaf}`);
    }
    lo.push(loLeaf);
  }
  return lo as [number, number, number];
}
