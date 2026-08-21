// A csg.js style modelling API over the GPU grid ops.
//
// A Space is a device plus a narrow band half width. A Solid is an
// immutable GPU resident SDF grid that you own. Destroy it, or declare it
// with `using`. Operations on a Solid return an Op: a recipe that runs
// when awaited and resolves to a new Solid. A chain of ops frees its own
// intermediates, so only the solids you keep need destroying. All
// coordinates are voxels.
//
//   import { Space } from '@emcfarlane/picovdb/model';
//
//   const space = new Space(device, { halfWidth: 3 });
//
//   // Primitives and booleans. Shapes can stand in for solids as operands,
//   // which stamps them directly and skips the intermediate solid.
//   using ball = await space.sphere([0, 0, 0], 20);
//   using bolt = await ball
//     .union(space.cylinder([0, -30, 0], [0, 30, 0], 6))
//     .subtract({ kind: 'box', center: [0, 0, 0], half: [30, 4, 4] });
//
//   // Edit a file: grow the bunny by two voxels, hollow it, and move it.
//   using bunny = space.fromPvdb(bytes);
//   using shell = await bunny.offset(2).subtract(bunny).translate([0, 0, -10]);
//
//   // Outputs: file bytes, or the picovdb tree left on the GPU for a renderer.
//   await Deno.writeFile('shell.pvdb', new Uint8Array(await shell.toPvdb()));
//   const tree = await shell.toTree(); // roots, uppers, lowers, leaves, data
//
// Solids carry their own key origin, so position is unbounded. The extent
// of one solid, or of the two operands of a boolean, is limited to 1024
// leaves (8192 voxels) per axis.

import { Binner } from './gpu/mesh_to_grid.ts';
import { Rasterizer } from './gpu/rasterize.ts';
import { Signer } from './gpu/sign.ts';
import { Emitter, type EmitResult, LOWER_U32, UPPER_U32 } from './gpu/emit.ts';
import { LEAF_U32, emptyOpGrid, type OpGrid } from './gpu/opgrid.ts';
import { Merger, type CsgOp } from './gpu/merge.ts';
import { Stamper, shapeBounds, type Shape, type Vec3 } from './gpu/stamp.ts';
import { Remapper } from './gpu/remap.ts';
import { Loader } from './gpu/load.ts';
import { Scanner } from './gpu/scan.ts';
import { Sorter } from './gpu/radix_sort.ts';
import {
  GRID_TYPE_SDF_FLOAT,
  PICOVDB_FILE_HEADER_SIZE,
  PICOVDB_GRID_SIZE,
  PICOVDB_MAGIC,
  PICOVDB_ROOT_SIZE,
  PicoVDBFile,
} from './picovdb.ts';

export type { Shape, Vec3, OpGrid };

/** The picovdb node buffers of a solid, GPU resident in the file layout. */
export type PicoVDBTree = EmitResult;

/** Inclusive leaf coordinate bounds, or null for an empty solid. */
export type Bounds = { min: Vec3; max: Vec3 } | null;

/** A boolean operand: a solid, a pending op, or a shape stamped directly. */
export type Operand = Solid | Op | Shape;

export interface SpaceOptions {
  /** Narrow band half width in voxels. */
  halfWidth?: number;
}

const KEY_RANGE = 1024;

function unionBounds(a: Bounds, b: Bounds): Bounds {
  if (!a) return b;
  if (!b) return a;
  return {
    min: a.min.map((v, i) => Math.min(v, b.min[i])) as Vec3,
    max: a.max.map((v, i) => Math.max(v, b.max[i])) as Vec3,
  };
}

function growBounds(b: Bounds, lo: Vec3, hi: Vec3): Bounds {
  if (!b) return b;
  return { min: b.min.map((v, i) => v + lo[i]) as Vec3, max: b.max.map((v, i) => v + hi[i]) as Vec3 };
}

/** Leaf bounds a shape's band can touch. */
function shapeLeafBounds(shape: Shape, halfWidth: number): Bounds {
  const { min, max } = shapeBounds(shape);
  return {
    min: min.map((v) => Math.floor((v - halfWidth) / 8)) as Vec3,
    max: max.map((v) => Math.floor((v + halfWidth) / 8)) as Vec3,
  };
}

/** The key origin that fits bounds, or throws when the extent is too large. */
function originFor(bounds: Bounds): Vec3 {
  if (!bounds) return [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const extent = bounds.max[a] - bounds.min[a] + 1;
    if (extent > KEY_RANGE) {
      throw new Error(`solid extent of ${extent} leaves on axis ${a} exceeds the ${KEY_RANGE} leaf key range`);
    }
  }
  return [...bounds.min] as Vec3;
}

export class Space {
  readonly device: GPUDevice;
  readonly halfWidth: number;
  readonly scanner: Scanner;
  readonly sorter: Sorter;
  readonly emitter: Emitter;
  readonly merger: Merger;
  readonly stamper: Stamper;
  readonly remapper: Remapper;
  readonly loader: Loader;
  private binner?: Binner;
  private rasterizer?: Rasterizer;
  private signer?: Signer;

  constructor(device: GPUDevice, opts: SpaceOptions = {}) {
    this.device = device;
    this.halfWidth = opts.halfWidth ?? 3;
    this.scanner = new Scanner(device);
    this.sorter = new Sorter(device, this.scanner);
    this.emitter = new Emitter(device, this.scanner, this.sorter);
    this.merger = new Merger(device, this.scanner, this.sorter);
    this.stamper = new Stamper(device, this.scanner, this.sorter);
    this.remapper = new Remapper(device, this.scanner, this.sorter);
    this.loader = new Loader(device);
  }

  /** A solid with no leaves. */
  empty(): Solid {
    return new Solid(this, emptyOpGrid(this.device), null);
  }

  /** An analytic primitive. */
  shape(shape: Shape): Op {
    return new Op(this, async () => {
      const empty = this.empty();
      try {
        return await stamp(empty, shape, 'add');
      } finally {
        empty.destroy();
      }
    });
  }

  sphere(center: Vec3, radius: number): Op {
    return this.shape({ kind: 'sphere', center, radius });
  }

  /** Half extents per axis, edges rounded by radius. */
  box(center: Vec3, half: Vec3, radius = 0): Op {
    return this.shape({ kind: 'box', center, half, radius });
  }

  capsule(a: Vec3, b: Vec3, radius: number): Op {
    return this.shape({ kind: 'capsule', a, b, radius });
  }

  cylinder(a: Vec3, b: Vec3, radius: number): Op {
    return this.shape({ kind: 'cylinder', a, b, radius });
  }

  /** Grid 0 of an f32 or u8 SDF picovdb file. Values rescale to this half width. */
  fromPvdb(file: PicoVDBFile | ArrayBuffer): Solid {
    const grid = this.loader.load(file instanceof PicoVDBFile ? file : new PicoVDBFile(file), { halfWidth: this.halfWidth });
    return new Solid(this, grid, { min: grid.leafMin, max: grid.leafMax });
  }

  /** A closed triangle mesh in world units, voxelized at voxelSize world units per voxel. */
  fromMesh(points: Float32Array<ArrayBuffer>, triangles: Uint32Array<ArrayBuffer>, voxelSize: number): Op {
    return new Op(this, async () => {
      this.binner ??= new Binner(this.device, this.scanner, this.sorter);
      this.rasterizer ??= new Rasterizer(this.device);
      this.signer ??= new Signer(this.device, this.scanner, this.sorter);
      const halfWidth = this.halfWidth;
      const bin = await this.binner.bin(points, triangles, { voxelSize, halfWidth });
      const dist2 = this.rasterizer.rasterize(bin, { halfWidth });
      const sign = await this.signer.sign(bin);
      const grid = await this.emitter.classifyOnly(bin, dist2, sign, { halfWidth });
      for (const b of [bin.pointsIndex, bin.triangles, bin.pairKeys, bin.pairTris, bin.leafKeys, dist2, sign.inside]) b.destroy();
      return new Solid(this, grid, { min: grid.leafMin, max: grid.leafMax });
    });
  }
}

export class Solid {
  readonly space: Space;
  readonly grid: OpGrid;
  readonly bounds: Bounds;

  constructor(space: Space, grid: OpGrid, bounds: Bounds) {
    this.space = space;
    this.grid = grid;
    this.bounds = bounds;
  }

  get leafCount(): number {
    return this.grid.leafCount;
  }

  get activeVoxels(): number {
    return this.grid.activeVoxels;
  }

  union(other: Operand): Op {
    return new Op(this.space, () => combine(this, other, 'union'));
  }

  subtract(other: Operand): Op {
    return new Op(this.space, () => combine(this, other, 'subtract'));
  }

  intersect(other: Solid | Op): Op {
    return new Op(this.space, () => combine(this, other, 'intersect'));
  }

  /**
   * Offsets the surface by amount voxels. Positive grows. The new band is
   * redistanced from the old surface, exact to marching cubes precision.
   * Amounts beyond half width - 1 run in several steps.
   */
  offset(amount: number): Op {
    return new Op(this.space, () => offset(this, amount));
  }

  /** Translates by whole voxels. */
  translate(shift: Vec3): Op {
    return new Op(this.space, () => translate(this, shift));
  }

  /** The picovdb tree on the GPU, for a renderer. The caller destroys its buffers. */
  toTree(): Promise<PicoVDBTree> {
    return this.space.emitter.reEmit(this.grid, { halfWidth: this.space.halfWidth });
  }

  /** The solid as a single grid .pvdb file. */
  async toPvdb(): Promise<ArrayBuffer> {
    const tree = await this.toTree();
    try {
      return await writePvdb(this.space.device, tree);
    } finally {
      for (const b of [tree.roots, tree.uppers, tree.lowers, tree.leaves, tree.data]) b.destroy();
    }
  }

  /** Releases the GPU buffers. The solid is unusable afterwards. */
  destroy(): void {
    this.grid.leafKeys.destroy();
    this.grid.leaves.destroy();
    this.grid.data.destroy();
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}

/**
 * A pending operation. Awaiting it runs the chain and resolves to a new
 * Solid that the caller owns. Intermediates and Op operands are destroyed
 * along the way. Each await runs the recipe again.
 */
export class Op implements PromiseLike<Solid> {
  readonly space: Space;
  private readonly run: () => Promise<Solid>;

  constructor(space: Space, run: () => Promise<Solid>) {
    this.space = space;
    this.run = run;
  }

  union(other: Operand): Op {
    return this.then_((s) => combine(s, other, 'union'));
  }

  subtract(other: Operand): Op {
    return this.then_((s) => combine(s, other, 'subtract'));
  }

  intersect(other: Solid | Op): Op {
    return this.then_((s) => combine(s, other, 'intersect'));
  }

  offset(amount: number): Op {
    return this.then_((s) => offset(s, amount));
  }

  translate(shift: Vec3): Op {
    return this.then_((s) => translate(s, shift));
  }

  then<R1 = Solid, R2 = never>(
    onfulfilled?: ((value: Solid) => R1 | PromiseLike<R1>) | null,
    onrejected?: ((reason: unknown) => R2 | PromiseLike<R2>) | null,
  ): Promise<R1 | R2> {
    return this.run().then(onfulfilled, onrejected);
  }

  /** The next step, which owns and frees this step's result. */
  private then_(step: (s: Solid) => Promise<Solid>): Op {
    return new Op(this.space, async () => {
      const s = await this.run();
      try {
        return await step(s);
      } finally {
        s.destroy();
      }
    });
  }
}

// The immediate operations. Each returns a new Solid and leaves its inputs as they are.

async function combine(self: Solid, other: Operand, op: CsgOp): Promise<Solid> {
  if (other instanceof Op) {
    const s = await other;
    try {
      return await combine(self, s, op);
    } finally {
      s.destroy();
    }
  }
  if (!(other instanceof Solid)) {
    if (op === 'intersect') throw new Error('intersect takes a solid');
    return stamp(self, other, op === 'union' ? 'add' : 'carve');
  }
  const space = self.space;
  if (other.space !== space) throw new Error('solids belong to different spaces');
  if (self.grid.leafCount === 0 || other.grid.leafCount === 0) {
    // An empty solid is all outside: union and subtract keep this,
    // intersect empties it. Subtracting from empty stays empty.
    const keep = op === 'intersect' ? null : self.grid.leafCount === 0 && op === 'union' ? other : self;
    return keep ? copy(keep) : space.empty();
  }
  const bounds = unionBounds(self.bounds, other.bounds);
  const a = rebased(self, bounds);
  const b = rebased(other, bounds);
  const out = await space.merger.mergeCsg(a.grid, b.grid, { halfWidth: space.halfWidth, op });
  a.release();
  b.release();
  return new Solid(space, { ...out, leafMax: bounds!.max }, bounds);
}

async function stamp(self: Solid, shape: Shape, mode: 'add' | 'carve'): Promise<Solid> {
  const space = self.space;
  // Carving cannot extend the solid, so a carve keeps its bounds.
  const bounds = mode === 'carve' ? self.bounds : unionBounds(self.bounds, shapeLeafBounds(shape, space.halfWidth));
  const { grid, release } = rebased(self, bounds);
  const out = await space.stamper.stamp(grid, { shape, mode, halfWidth: space.halfWidth });
  release();
  return new Solid(space, out, bounds);
}

async function offset(self: Solid, amount: number): Promise<Solid> {
  const space = self.space;
  if (self.grid.leafCount === 0) return space.empty();
  const m = Remapper.leafMargin(amount, space.halfWidth);
  const bounds = growBounds(self.bounds, [-m, -m, -m], [m, m, m]);
  const { grid, release } = rebased(self, bounds);
  const out = await space.remapper.offset(grid, amount, space.halfWidth);
  release();
  return new Solid(space, out, bounds);
}

async function translate(self: Solid, shift: Vec3): Promise<Solid> {
  const space = self.space;
  if (self.grid.leafCount === 0) return space.empty();
  if (shift.some((s) => !Number.isInteger(s))) throw new Error('translate takes whole voxels');
  // Leaf multiples move the key origin; the remainder remaps voxels.
  const q = shift.map((s) => Math.floor(s / 8)) as Vec3;
  const r = shift.map((s, a) => s - 8 * q[a]) as Vec3;
  const shifted: OpGrid = { ...self.grid, leafMin: self.grid.leafMin.map((v, a) => v + q[a]) as Vec3, leafMax: self.grid.leafMax.map((v, a) => v + q[a]) as Vec3 };
  let bounds = growBounds(self.bounds, q, q);
  if (r.every((v) => v === 0)) return new Solid(space, copyGrid(space.device, shifted), bounds);
  bounds = growBounds(bounds, [0, 0, 0], r.map((v) => (v > 0 ? 1 : 0)) as Vec3);
  const grid = space.remapper.rebase(shifted, originFor(bounds));
  const out = await space.remapper.translate(grid, r, space.halfWidth);
  if (grid.leafKeys !== self.grid.leafKeys) grid.leafKeys.destroy();
  return new Solid(space, out, bounds);
}

/** The grid keyed from an origin that fits bounds, carrying the solid's exact leaf bounds. release frees the temporary keys. */
function rebased(self: Solid, bounds: Bounds): { grid: OpGrid; release: () => void } {
  const origin = originFor(bounds);
  const grid = { ...self.space.remapper.rebase(self.grid, origin), leafMax: bounds ? bounds.max : origin };
  return { grid, release: () => { if (grid.leafKeys !== self.grid.leafKeys) grid.leafKeys.destroy(); } };
}

function copy(self: Solid): Solid {
  return new Solid(self.space, copyGrid(self.space.device, self.grid), self.bounds);
}

function copyGrid(device: GPUDevice, grid: OpGrid): OpGrid {
  const dup = (src: GPUBuffer, size: number) => {
    const dst = device.createBuffer({ size: Math.max(size, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    if (size > 0) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(src, 0, dst, 0, size);
      device.queue.submit([encoder.finish()]);
    }
    return dst;
  };
  const n = grid.leafCount;
  return { ...grid, leafKeys: dup(grid.leafKeys, n * 4), leaves: dup(grid.leaves, n * LEAF_U32 * 4), data: dup(grid.data, (2 + grid.activeVoxels) * 4) };
}

/** Reads a tree back into .pvdb bytes: header, one grid record, then the node buffers. */
async function writePvdb(device: GPUDevice, tree: PicoVDBTree): Promise<ArrayBuffer> {
  const rootsPadded = Math.ceil(tree.upperCount / 2) * 2;
  const dataCount = Math.ceil((tree.dataElemCount * 4) / 16);
  const sections: [GPUBuffer, number, number][] = [
    [tree.roots, tree.upperCount * PICOVDB_ROOT_SIZE, rootsPadded * PICOVDB_ROOT_SIZE],
    [tree.uppers, tree.upperCount * UPPER_U32 * 4, tree.upperCount * UPPER_U32 * 4],
    [tree.lowers, tree.lowerCount * LOWER_U32 * 4, tree.lowerCount * LOWER_U32 * 4],
    [tree.leaves, tree.leafCount * LEAF_U32 * 4, tree.leafCount * LEAF_U32 * 4],
    [tree.data, tree.dataElemCount * 4, dataCount * 16],
  ];
  const headSize = PICOVDB_FILE_HEADER_SIZE + PICOVDB_GRID_SIZE;
  const head = new ArrayBuffer(headSize);
  new Uint32Array(head, 0, 8).set([PICOVDB_MAGIC[0], PICOVDB_MAGIC[1], 0, 1, tree.upperCount, tree.lowerCount, tree.leafCount, dataCount]);
  new Uint32Array(head, PICOVDB_FILE_HEADER_SIZE, 8).set([0, 0, 0, 0, 0, tree.dataElemCount, GRID_TYPE_SDF_FLOAT, 0]);
  new Int32Array(head, PICOVDB_FILE_HEADER_SIZE + 32, 3).set(tree.indexBoundsMin);
  new Int32Array(head, PICOVDB_FILE_HEADER_SIZE + 48, 3).set(tree.indexBoundsMax);

  const size = headSize + sections.reduce((n, [, , padded]) => n + padded, 0);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(staging, 0, head);
  const encoder = device.createCommandEncoder();
  let offset = headSize;
  for (const [src, bytes, padded] of sections) {
    if (bytes > 0) encoder.copyBufferToBuffer(src, 0, staging, offset, bytes);
    offset += padded;
  }
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.destroy();
  return out;
}
