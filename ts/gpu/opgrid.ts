// The op layer grid: what the GPU ops read and write between edits. It is
// the picovdb leaf level, so files load and emit without conversion.
//
//   leafKeys  sorted leaf keys
//   leaves    leaf records in the file layout: inside and band masks, per
//             word value prefixes, and the absolute value index
//   data      +hw, -hw, then the band values only
//
// Surface bits and surface bases stay zero until emission. Every leaf
// holds at least one band voxel.
//
// Kernels read voxels through the shared reader (readerWgsl) and write
// grids through the shared writer. A mark pass folds each candidate leaf's
// values into a record and a band count. The host scans the counts.
// opgrid_compact drops band-empty leaves and assigns value bases. An apply
// pass writes the band values into their slots.

import { PICOVDB_LEAF_SIZE } from '../picovdb.ts';
import type { Scanner } from './scan.ts';
import { checkBindingSize, dispatch2D, readBackTotals } from './device.ts';

export const LEAF_U32 = PICOVDB_LEAF_SIZE / 4;

export interface OpGrid {
  /** Sorted leaf keys, 10 bits per axis relative to leafMin. */
  leafKeys: GPUBuffer;
  /** Leaf records in the picovdb leaf layout, in key order. */
  leaves: GPUBuffer;
  /** f32 bits: +half width, -half width, then the band values by leaf and bit order. */
  data: GPUBuffer;
  leafCount: number;
  activeVoxels: number;
  leafMin: [number, number, number];
  /** Maximum leaf coordinate inclusive. */
  leafMax: [number, number, number];
}

export function emptyOpGrid(device: GPUDevice, leafMin: [number, number, number] = [0, 0, 0], leafMax: [number, number, number] = [0, 0, 0]): OpGrid {
  const placeholder = () => device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  return { leafKeys: placeholder(), leaves: placeholder(), data: placeholder(), leafCount: 0, activeVoxels: 0, leafMin, leafMax };
}

/** Prelude of every kernel: indexing, key packing, the leaf layout, and the writer. Kernels must declare params.half_width. */
export const preludeWgsl = /* wgsl */ `
const DISPATCH_STRIDE: u32 = 65535u;
const NOT_FOUND: u32 = 0xffffffffu;
const LEAF_U32: u32 = ${LEAF_U32}u;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

fn unpack(key: u32) -> vec3<i32> {
    return vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu));
}

fn pack(c: vec3<i32>) -> u32 {
    return (u32(c.x) << 20u) | (u32(c.y) << 10u) | u32(c.z);
}

fn voxelOffset(ijk: vec3<i32>) -> u32 {
    return (u32(ijk.x & 7) << 6u) | (u32(ijk.y & 7) << 3u) | u32(ijk.z & 7);
}

fn voxelLocal(n: u32) -> vec3<i32> {
    return vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
}

// Writer bindings. w_counts and w_kept have one extra entry for the scan
// totals.
@group(0) @binding(40) var<storage, read> w_cand_keys: array<u32>;
@group(0) @binding(41) var<storage, read_write> w_cand_leaves: array<u32>;
@group(0) @binding(42) var<storage, read_write> w_counts: array<u32>; // band counts, then their exclusive scan
@group(0) @binding(43) var<storage, read_write> w_kept: array<u32>; // kept flags, then their exclusive scan
@group(0) @binding(44) var<storage, read_write> w_keys: array<u32>;
@group(0) @binding(45) var<storage, read_write> w_leaves: array<u32>;
@group(0) @binding(46) var<storage, read_write> w_data: array<u32>;

// Writes the extra entries. Returns true when i is past the candidates.
fn markSentinel(i: u32, cand: u32) -> bool {
    if (i == cand) {
        w_counts[i] = 0u;
        w_kept[i] = 0u;
    }
    return i >= cand;
}

// Folds the values of candidate leaf i into its record, one word per 32
// voxels. Call in voxel order.
struct LeafAcc {
    band: u32,
    inside: u32,
    local: u32,
    count: u32,
}

fn accPush(acc: ptr<function, LeafAcc>, i: u32, n: u32, v: f32) {
    let bit = 1u << (n & 31u);
    if (abs(v) < params.half_width) {
        (*acc).band = (*acc).band | bit;
    }
    if (v < 0.0) {
        (*acc).inside = (*acc).inside | bit;
    }
    if ((n & 31u) == 31u) {
        let e = (i * LEAF_U32) + 4u + ((n >> 5u) * 3u);
        w_cand_leaves[e] = (*acc).inside & ~(*acc).band;
        w_cand_leaves[e + 1u] = (*acc).band;
        w_cand_leaves[e + 2u] = (*acc).local;
        let c = countOneBits((*acc).band);
        (*acc).local = (*acc).local + c;
        (*acc).count = (*acc).count + c;
        (*acc).band = 0u;
        (*acc).inside = 0u;
    }
}

fn accFinish(acc: ptr<function, LeafAcc>, i: u32) {
    for (var k = 0u; k < 4u; k = k + 1u) {
        w_cand_leaves[(i * LEAF_U32) + k] = 0u;
    }
    w_counts[i] = (*acc).count;
    w_kept[i] = select(0u, 1u, (*acc).count > 0u);
}

// Band mask word w of output leaf j.
fn bandWord(j: u32, w: u32) -> u32 {
    return w_leaves[(j * LEAF_U32) + 5u + (w * 3u)];
}

// Copies kept candidates to the output with their value bases.
@compute @workgroup_size(256)
fn opgrid_compact(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    let cand = arrayLength(&w_counts) - 1u;
    if (i >= cand || w_kept[i] == w_kept[i + 1u]) {
        return;
    }
    let j = w_kept[i];
    w_keys[j] = w_cand_keys[i];
    for (var k = 0u; k < LEAF_U32; k = k + 1u) {
        w_leaves[(j * LEAF_U32) + k] = w_cand_leaves[(i * LEAF_U32) + k];
    }
    w_leaves[(j * LEAF_U32) + 2u] = 2u + w_counts[i];
}
`;

/**
 * Reader of a grid bound as {p}_keys, {p}_leaves, {p}_data with count
 * leaves. {p}_valueAt gives the value at any relative voxel: the stored
 * band value, or the implicit background of the inside bit. Leafless
 * space takes the sign of the nearest leaf in its column, and a column
 * with no leaves is outside. This matches the CPU converter.
 */
export function readerWgsl(p: string, count: string): string {
  return /* wgsl */ `
fn ${p}_find(key: u32) -> u32 {
    var lo = 0u;
    var hi = ${count};
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (${p}_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < ${count} && ${p}_keys[lo] == key) {
        return lo;
    }
    return NOT_FOUND;
}

fn ${p}_leafValue(i: u32, n: u32) -> f32 {
    let e = (i * LEAF_U32) + 4u + ((n >> 5u) * 3u);
    let value = ${p}_leaves[e + 1u];
    let bit = 1u << (n & 31u);
    if ((value & bit) != 0u) {
        let d = ${p}_leaves[(i * LEAF_U32) + 2u] + (${p}_leaves[e + 2u] & 0xffffu) + countOneBits(value & (bit - 1u));
        return bitcast<f32>(${p}_data[d]);
    }
    return select(params.half_width, -params.half_width, (${p}_leaves[e] & bit) != 0u);
}

fn ${p}_valueAt(ijk: vec3<i32>) -> f32 {
    let leaf = ijk >> vec3<u32>(3u);
    if (leaf.x < 0 || leaf.x > 1023 || leaf.y < 0 || leaf.y > 1023) {
        return params.half_width;
    }
    let n = voxelOffset(ijk);
    if (leaf.z >= 0 && leaf.z <= 1023) {
        let i = ${p}_find(pack(leaf));
        if (i != NOT_FOUND) {
            return ${p}_leafValue(i, n);
        }
    }
    let col = (u32(leaf.x) << 20u) | (u32(leaf.y) << 10u);
    var lo = 0u;
    var hi = ${count};
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (${p}_keys[mid] < col) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= ${count} || (${p}_keys[lo] & 0xfffffc00u) != col) {
        return params.half_width;
    }
    var best = lo;
    var best_d = abs(i32(${p}_keys[lo] & 0x3ffu) - leaf.z);
    var i = lo + 1u;
    while (i < ${count} && (${p}_keys[i] & 0xfffffc00u) == col) {
        let d = abs(i32(${p}_keys[i] & 0x3ffu) - leaf.z);
        if (d >= best_d) {
            break;
        }
        best = i;
        best_d = d;
        i = i + 1u;
    }
    let zloc = select(0u, 7u, i32(${p}_keys[best] & 0x3ffu) < leaf.z);
    return select(params.half_width, -params.half_width, ${p}_leafValue(best, (n & 0x1f8u) | zloc) < 0.0);
}
`;
}

/** Host side of the writer. Construct one per output grid, run the kernel's mark pass with markBindings, then call finish. */
export class GridWriter {
  readonly device: GPUDevice;
  readonly cand: number;
  readonly candLeaves: GPUBuffer;
  readonly counts: GPUBuffer;
  readonly kept: GPUBuffer;

  constructor(device: GPUDevice, cand: number) {
    this.device = device;
    this.cand = cand;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.candLeaves = device.createBuffer({ size: Math.max(cand, 1) * LEAF_U32 * 4, usage: storage });
    this.counts = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
    this.kept = device.createBuffer({ size: (cand + 1) * 4, usage: storage });
  }

  get markBindings(): Record<number, GPUBuffer> {
    return { 41: this.candLeaves, 42: this.counts, 43: this.kept };
  }

  /**
   * Scans the mark results, allocates the grid, runs the kernel's
   * opgrid_compact pipeline, then runs apply with the output bindings.
   * candKeys must be the candidate keys the mark pass indexed.
   */
  async finish(
    scanner: Scanner,
    compact: GPUComputePipeline,
    candKeys: GPUBuffer,
    halfWidth: number,
    bounds: { leafMin: [number, number, number]; leafMax: [number, number, number] },
    apply: (pass: GPUComputePassEncoder, bindings: Record<number, GPUBuffer>, leafCount: number) => void
  ): Promise<OpGrid> {
    const device = this.device;
    const n = this.cand + 1;
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      scanner.plan(this.kept, n).encode(pass);
      scanner.plan(this.counts, n).encode(pass);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    const [leafCount, activeVoxels] = await readBackTotals(device, [
      { buffer: this.kept, index: this.cand },
      { buffer: this.counts, index: this.cand },
    ]);
    const release = () => {
      this.candLeaves.destroy();
      this.counts.destroy();
      this.kept.destroy();
    };
    if (leafCount === 0) {
      release();
      return emptyOpGrid(device, bounds.leafMin, bounds.leafMax);
    }
    checkBindingSize(device, leafCount * LEAF_U32 * 4, 'op grid leaves');
    checkBindingSize(device, (2 + activeVoxels) * 4, 'op grid values');
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const leafKeys = device.createBuffer({ size: leafCount * 4, usage: storage });
    const leaves = device.createBuffer({ size: leafCount * LEAF_U32 * 4, usage: storage });
    const data = device.createBuffer({ size: (2 + activeVoxels) * 4, usage: storage });
    device.queue.writeBuffer(data, 0, new Float32Array([halfWidth, -halfWidth]));
    {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(compact);
      pass.setBindGroup(
        0,
        device.createBindGroup({
          layout: compact.getBindGroupLayout(0),
          entries: Object.entries({ 40: candKeys, ...this.markBindings, 44: leafKeys, 45: leaves }).map(([binding, buffer]) => ({
            binding: Number(binding),
            resource: { buffer },
          })),
        })
      );
      dispatch2D(pass, Math.ceil(this.cand / 256));
      apply(pass, { 44: leafKeys, 45: leaves, 46: data }, leafCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    release();
    return { leafKeys, leaves, data, leafCount, activeVoxels, leafMin: bounds.leafMin, leafMax: bounds.leafMax };
  }
}
