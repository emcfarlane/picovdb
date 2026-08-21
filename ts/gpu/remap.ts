// Host side of wgsl/remap.wgsl: SDF offset, integer translation, and key
// rebasing of op layer grids.

import remapWgsl from 'picovdb/wgsl/remap.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { Extractor } from './extract.ts';
import { Binner } from './mesh_to_grid.ts';
import { Rasterizer } from './rasterize.ts';
import { dispatch2D, readBackTotals } from './device.ts';
import { GridWriter, preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';

const WG_SIZE = 256;

export class Remapper {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};
  private extractor?: Extractor;
  private binner?: Binner;
  private rasterizer?: Rasterizer;

  constructor(device: GPUDevice, scanner: Scanner = new Scanner(device), sorter: Sorter = new Sorter(device, scanner)) {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    const module = device.createShaderModule({ code: preludeWgsl + readerWgsl('old', 'params.old_count') + remapWgsl });
    for (const entryPoint of ['rebase', 'generate_candidates', 'mark_unique', 'compact_unique', 'mark', 'apply', 'opgrid_compact']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  /**
   * Rewrites the keys relative to a new origin. The result shares the
   * input's leaves and data; only leafKeys is new. Every leaf must stay
   * inside 0..1023 of the new origin.
   */
  rebase(grid: OpGrid, leafMin: [number, number, number]): OpGrid {
    const device = this.device;
    const delta = grid.leafMin.map((v, a) => v - leafMin[a]);
    if (delta.every((d) => d === 0)) return grid;
    const params = this.params({ oldCount: grid.leafCount, delta });
    const newKeys = device.createBuffer({
      size: Math.max(grid.leafCount, 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    this.run(pass, 'rebase', grid.leafCount, params, { 1: grid.leafKeys, 5: newKeys });
    pass.end();
    device.queue.submit([encoder.finish()]);
    return { ...grid, leafKeys: newKeys, leafMin };
  }

  /**
   * Offsets the level set by amount voxels. Positive grows the solid.
   * Each step extracts the old zero level set and redistances the whole
   * new band from its triangles. So the result has no seam and does not
   * depend on stored values away from the surface. Steps are at most
   * halfWidth - 1, to bound the reach. The leaf table must fit the current
   * key origin with leafMargin(amount, halfWidth) leaves of margin; rebase
   * first if it does not.
   */
  async offset(grid: OpGrid, amount: number, halfWidth: number): Promise<OpGrid> {
    const maxStep = halfWidth - 1;
    if (!(maxStep > 0)) throw new Error('offset needs a half width above one voxel');
    let out = grid;
    for (let remaining = amount; remaining !== 0;) {
      const step = Math.sign(remaining) * Math.min(Math.abs(remaining), maxStep);
      const next = await this.offsetStep(out, step, halfWidth);
      if (out !== grid) {
        out.leafKeys.destroy();
        out.leaves.destroy();
        out.data.destroy();
      }
      out = next;
      remaining -= step;
    }
    return out;
  }

  private async offsetStep(grid: OpGrid, amount: number, halfWidth: number): Promise<OpGrid> {
    this.extractor ??= new Extractor(this.device, this.scanner);
    this.binner ??= new Binner(this.device, this.scanner, this.sorter);
    this.rasterizer ??= new Rasterizer(this.device);
    const reach = halfWidth + Math.abs(amount);
    // Distances to the old surface, in the grid's own key space.
    const mesh = await this.extractor.extract(grid, halfWidth);
    const bin = await this.binner.binBuffers(mesh.points, mesh.triangleCount * 3, mesh.triangles, mesh.triangleCount, {
      voxelSize: 1,
      halfWidth: reach,
      bounds: { leafMin: [0, 0, 0], leafMax: [1023, 1023, 1023] },
    });
    const dist = this.rasterizer.rasterize(bin, { halfWidth: reach });
    mesh.points.destroy();
    mesh.triangles.destroy();
    bin.pointsIndex.destroy();
    bin.pairKeys.destroy();
    bin.pairTris.destroy();
    const out = await this.finish(grid, halfWidth, { mode: 0, amount }, bin.leafKeys, bin.leafCount, { 9: dist });
    bin.leafKeys.destroy();
    dist.destroy();
    return out;
  }

  /** Leaves an offset's band can reach beyond the input's leaves. */
  static leafMargin(amount: number, halfWidth: number): number {
    return Math.ceil((halfWidth + Math.abs(amount) + 1) / 8);
  }

  /** Translates by whole voxels. Multiples of a leaf only move the key origin. */
  translate(grid: OpGrid, shift: [number, number, number], halfWidth: number): Promise<OpGrid> {
    if (shift.some((s) => !Number.isInteger(s))) throw new Error('translate takes whole voxels');
    if (shift.every((s) => s % 8 === 0)) {
      const leafMin = grid.leafMin.map((v, a) => v + shift[a] / 8) as [number, number, number];
      const leafMax = grid.leafMax.map((v, a) => v + shift[a] / 8) as [number, number, number];
      return Promise.resolve({ ...grid, leafMin, leafMax });
    }
    const nbLo = shift.map((s) => Math.floor(s / 8));
    const nbHi = shift.map((s) => Math.floor((s + 7) / 8));
    return this.remap(grid, halfWidth, { mode: 1, shift, nbLo, nbDims: nbHi.map((h, a) => h - nbLo[a] + 1) });
  }

  /** Builds the candidates from the old leaves and their neighbors, then finishes. */
  private async remap(
    grid: OpGrid,
    halfWidth: number,
    opts: { mode: number; shift: number[]; nbLo: number[]; nbDims: number[] }
  ): Promise<OpGrid> {
    const device = this.device;
    const vol = opts.nbDims[0] * opts.nbDims[1] * opts.nbDims[2];
    const concatCount = grid.leafCount * (1 + vol);
    const params = this.params({ oldCount: grid.leafCount, concatCount, halfWidth, ...opts });

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const concatKeys = device.createBuffer({ size: concatCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: concatCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (concatCount + 1) * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(grid.leafKeys, 0, concatKeys, 0, grid.leafCount * 4);
      const pass = encoder.beginComputePass();
      this.run(pass, 'generate_candidates', grid.leafCount * vol, params, { 1: grid.leafKeys, 3: concatKeys });
      this.sorter.plan(concatKeys, sortVals, concatCount).encode(pass);
      this.run(pass, 'mark_unique', concatCount + 1, params, { 3: concatKeys, 4: flags });
      this.scanner.plan(flags, concatCount + 1).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [candCount] = await readBackTotals(device, [{ buffer: flags, index: concatCount }]);
    device.queue.writeBuffer(params, 8, new Uint32Array([candCount]));
    const candKeys = device.createBuffer({ size: candCount * 4, usage: storage });
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      this.run(pass, 'compact_unique', concatCount, params, { 3: concatKeys, 4: flags, 5: candKeys });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    concatKeys.destroy();
    sortVals.destroy();
    flags.destroy();
    // The distance binding exists in every entry point's auto layout.
    const placeholder = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
    const out = await this.finish(grid, halfWidth, opts, candKeys, candCount, { 9: placeholder });
    candKeys.destroy();
    placeholder.destroy();
    const leafMax = grid.leafMax.map((v, a) => Math.max(v, v + opts.nbLo[a] + opts.nbDims[a] - 1)) as [number, number, number];
    return { ...out, leafMax };
  }

  /** Runs the writer over the candidates. extra holds the mode's own bindings. */
  private async finish(
    grid: OpGrid,
    halfWidth: number,
    opts: { mode: number; amount?: number; shift?: number[] },
    candKeys: GPUBuffer,
    candCount: number,
    extra: Record<number, GPUBuffer>
  ): Promise<OpGrid> {
    const device = this.device;
    const params = this.params({ oldCount: grid.leafCount, halfWidth, distCount: candCount, ...opts });
    device.queue.writeBuffer(params, 8, new Uint32Array([candCount]));
    const run = (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>) => this.run(pass, name, threads, params, buffers);
    const inputs = { 1: grid.leafKeys, 2: grid.leaves, 5: candKeys, 6: grid.data, ...extra };
    const writer = new GridWriter(device, candCount);
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      run(pass, 'mark', candCount + 1, { ...inputs, ...writer.markBindings });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const out = await writer.finish(this.scanner, this.pipelines['opgrid_compact'], candKeys, halfWidth, grid, (pass, bindings, count) => {
      run(pass, 'apply', count, { ...inputs, ...bindings });
    });
    params.destroy();
    return out;
  }

  private params(p: {
    oldCount: number;
    concatCount?: number;
    mode?: number;
    shift?: number[];
    amount?: number;
    nbLo?: number[];
    halfWidth?: number;
    nbDims?: number[];
    distCount?: number;
    delta?: number[];
  }): GPUBuffer {
    const buffer = this.device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const q = this.device.queue;
    q.writeBuffer(buffer, 0, new Uint32Array([p.oldCount, p.concatCount ?? 0, 0, p.mode ?? 0]));
    q.writeBuffer(buffer, 16, new Int32Array(p.shift ?? [0, 0, 0]));
    q.writeBuffer(buffer, 28, new Float32Array([p.amount ?? 0]));
    q.writeBuffer(buffer, 32, new Int32Array(p.nbLo ?? [0, 0, 0]));
    q.writeBuffer(buffer, 44, new Float32Array([p.halfWidth ?? 0]));
    q.writeBuffer(buffer, 48, new Int32Array(p.nbDims ?? [1, 1, 1]));
    q.writeBuffer(buffer, 60, new Uint32Array([p.distCount ?? 0]));
    q.writeBuffer(buffer, 64, new Int32Array(p.delta ?? [0, 0, 0]));
    return buffer;
  }

  private run(pass: GPUComputePassEncoder, name: string, threads: number, params: GPUBuffer, buffers: Record<number, GPUBuffer>): void {
    if (threads === 0) return;
    pass.setPipeline(this.pipelines[name]);
    pass.setBindGroup(
      0,
      this.device.createBindGroup({
        layout: this.pipelines[name].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
          ...Object.entries(buffers).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
        ],
      })
    );
    dispatch2D(pass, Math.ceil(threads / WG_SIZE));
  }
}
