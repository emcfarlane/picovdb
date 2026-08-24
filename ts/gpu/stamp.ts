// Host side of wgsl/stamp.wgsl: stamps shapes into op layer grids,
// adding or carving material. A shape is a WGSL distance function, from
// the built-in library or the caller's.

import stampWgsl from 'picovdb/wgsl/stamp.wgsl' with { type: 'text' };
import { Scanner } from './scan.ts';
import { Sorter } from './radix_sort.ts';
import { dispatch2D, readBackTotals } from './device.ts';
import { GridWriter, preludeWgsl, readerWgsl, type OpGrid } from './opgrid.ts';

const WG_SIZE = 256;

export type Vec3 = [number, number, number];

/**
 * A shape: a WGSL signed distance function by name. The function has the
 * form `fn name(p: vec3f) -> f32`, takes absolute voxel coordinates, and
 * returns the signed distance in voxels. Its arguments arrive in the uniform
 * `args`, an `array<vec4f, 8>`.
 */
export interface Shape {
  fn: string;
  /** Up to 32 numbers, four per slot: args[0..3] is args[0] in WGSL, and so on. */
  args?: number[];
  /** Voxel bounds of the zero level set. Needed to add; a carve clips to the grid. */
  bounds?: { min: Vec3; max: Vec3 };
}

/** The built-in shapes. Their helpers below fill args to match. */
export const shapesWgsl = /* wgsl */ `
fn picovdb_sphere(p: vec3<f32>) -> f32 {
    return length(p - args[0].xyz) - args[0].w;
}

fn picovdb_box(p: vec3<f32>) -> f32 {
    let q = abs(p - args[0].xyz) - args[1].xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - args[1].w;
}

fn picovdb_capsule(p: vec3<f32>) -> f32 {
    let ba = args[1].xyz - args[0].xyz;
    let pa = p - args[0].xyz;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - (ba * h)) - args[0].w;
}

fn picovdb_cylinder(p: vec3<f32>) -> f32 {
    let ba = args[1].xyz - args[0].xyz;
    let pa = p - args[0].xyz;
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);
    let x = length((pa * baba) - (ba * paba)) - (args[0].w * baba);
    let y = abs(paba - (baba * 0.5)) - (baba * 0.5);
    let x2 = x * x;
    let y2 = y * y * baba;
    var d: f32;
    if (max(x, y) < 0.0) {
        d = -min(x2, y2);
    } else {
        d = select(0.0, x2, x > 0.0) + select(0.0, y2, y > 0.0);
    }
    return sign(d) * sqrt(abs(d)) / baba;
}
`;

function aabb(points: Vec3[], radius: number): { min: Vec3; max: Vec3 } {
  const min = [Infinity, Infinity, Infinity] as Vec3;
  const max = [-Infinity, -Infinity, -Infinity] as Vec3;
  for (const p of points) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], p[a] - radius);
      max[a] = Math.max(max[a], p[a] + radius);
    }
  }
  return { min, max };
}

export function sphere(center: Vec3, radius: number): Shape {
  return { fn: 'picovdb_sphere', args: [...center, radius], bounds: aabb([center], radius) };
}

/** Half extents per axis, edges rounded by radius. */
export function box(center: Vec3, half: Vec3, radius = 0): Shape {
  const corner = center.map((c, a) => c + half[a]) as Vec3;
  const opposite = center.map((c, a) => c - half[a]) as Vec3;
  return { fn: 'picovdb_box', args: [...center, 0, ...half, radius], bounds: aabb([corner, opposite], radius) };
}

export function capsule(a: Vec3, b: Vec3, radius: number): Shape {
  return { fn: 'picovdb_capsule', args: [...a, radius, ...b, 0], bounds: aabb([a, b], radius) };
}

export function cylinder(a: Vec3, b: Vec3, radius: number): Shape {
  return { fn: 'picovdb_cylinder', args: [...a, radius, ...b, 0], bounds: aabb([a, b], radius) };
}

export interface StampOptions {
  shape: Shape;
  mode: 'add' | 'carve';
  halfWidth: number;
}

export class Stamper {
  readonly device: GPUDevice;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  /** The shape library: the built-ins plus the caller's functions. */
  readonly shapes: string;
  private readonly pipelines = new Map<string, Record<string, GPUComputePipeline>>();

  constructor(device: GPUDevice, scanner: Scanner = new Scanner(device), sorter: Sorter = new Sorter(device, scanner), shapes = '') {
    this.device = device;
    this.scanner = scanner;
    this.sorter = sorter;
    this.shapes = shapesWgsl + shapes;
  }

  /** The pipelines for one shape function, compiled on first use. */
  private async pipelinesFor(fn: string): Promise<Record<string, GPUComputePipeline>> {
    const cached = this.pipelines.get(fn);
    if (cached) return cached;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(fn)) throw new Error(`shape function name ${JSON.stringify(fn)} is not an identifier`);
    const code = preludeWgsl + readerWgsl('old', 'params.old_count') + this.shapes + `fn sdf(p: vec3<f32>) -> f32 { return ${fn}(p); }\n` + stampWgsl;
    const module = this.device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length > 0) {
      throw new Error(`shapes do not compile for ${fn}:\n` + errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n'));
    }
    const pipelines: Record<string, GPUComputePipeline> = {};
    for (const entryPoint of ['generate_candidates', 'mark_unique', 'compact_unique', 'mark', 'apply', 'opgrid_compact']) {
      pipelines[entryPoint] = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
    this.pipelines.set(fn, pipelines);
    return pipelines;
  }

  async stamp(grid: OpGrid, opts: StampOptions): Promise<OpGrid> {
    const device = this.device;
    const hw = opts.halfWidth;
    const shape = opts.shape;
    const pipelines = await this.pipelinesFor(shape.fn);
    // The box of leaves the shape's band can touch. A carve only changes
    // existing material, so its box clips to the grid's leaves. An add
    // that reaches the key range throws, so nothing truncates silently.
    let lo = [0, 0, 0];
    let hi = grid.leafMax.map((v, a) => v - grid.leafMin[a]);
    if (shape.bounds) {
      lo = shape.bounds.min.map((c, a) => Math.floor((c - hw) / 8) - grid.leafMin[a]);
      hi = shape.bounds.max.map((c, a) => Math.floor((c + hw) / 8) - grid.leafMin[a]);
    }
    if (opts.mode === 'carve') {
      lo = lo.map((v) => Math.max(v, 0));
      hi = hi.map((v, a) => Math.min(v, grid.leafMax[a] - grid.leafMin[a]));
    } else if (!shape.bounds) {
      throw new Error(`shape ${shape.fn} needs bounds to add material`);
    } else if (lo.some((v) => v < 0) || hi.some((v) => v > 1023)) {
      throw new Error('stamp reaches the leaf key space boundary');
    }
    const dims = lo.map((v, a) => Math.max(hi[a] - v + 1, 0));
    const boxVol = dims[0] * dims[1] * dims[2];
    const concatCount = grid.leafCount + boxVol;

    const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([grid.leafCount, concatCount, 0, opts.mode === 'carve' ? 1 : 0]));
    device.queue.writeBuffer(params, 16, new Float32Array([grid.leafMin[0] * 8, grid.leafMin[1] * 8, grid.leafMin[2] * 8, hw]));
    device.queue.writeBuffer(params, 32, new Int32Array(lo));
    device.queue.writeBuffer(params, 48, new Int32Array(dims));
    const u = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    if ((shape.args?.length ?? 0) > 32) throw new Error(`shape ${shape.fn} has more than 32 arguments`);
    const packed = new Float32Array(32);
    packed.set(shape.args ?? []);
    device.queue.writeBuffer(u, 0, packed);

    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const concatKeys = device.createBuffer({ size: concatCount * 4, usage: storage });
    const sortVals = device.createBuffer({ size: concatCount * 4, usage: GPUBufferUsage.STORAGE });
    const flags = device.createBuffer({ size: (concatCount + 1) * 4, usage: storage });

    const run = (pass: GPUComputePassEncoder, name: string, threads: number, buffers: Record<number, GPUBuffer>) => {
      pass.setPipeline(pipelines[name]);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: pipelines[name].getBindGroupLayout(0),
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
      run(pass, 'generate_candidates', boxVol, { 3: concatKeys, 7: u });
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
      run(pass, 'mark', newCount + 1, { ...old, 5: newKeys, 7: u, ...writer.markBindings });
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    concatKeys.destroy();
    sortVals.destroy();
    flags.destroy();
    const leafMax = grid.leafMax.map((v, a) => (opts.mode === 'add' ? Math.max(v, hi[a] + grid.leafMin[a]) : v)) as Vec3;
    const out = await writer.finish(this.scanner, pipelines['opgrid_compact'], newKeys, hw, { leafMin: grid.leafMin, leafMax }, (pass, bindings, count) => {
      run(pass, 'apply', count, { ...old, 7: u, ...bindings });
    });
    newKeys.destroy();
    params.destroy();
    u.destroy();
    return out;
  }
}
