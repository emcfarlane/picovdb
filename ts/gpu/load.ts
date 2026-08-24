// Host side of wgsl/load.wgsl: a picovdb file as an op layer grid. The
// CPU walks the tree for the leaf origins; the GPU reorders the leaves
// and rescales the values.

import loadWgsl from 'picovdb/wgsl/load.wgsl' with { type: 'text' };
import { GRID_TYPE_SDF_FLOAT, GRID_TYPE_SDF_UINT8, PICOVDB_LOWER_SIZE, PICOVDB_UPPER_SIZE, type PicoVDBFile } from '../picovdb.ts';
import { checkBindingSize, createU32Buffer, dispatch2D } from './device.ts';
import { LEAF_U32, type OpGrid } from './opgrid.ts';

const WG_SIZE = 256;

export interface LoadOptions {
  /** Narrow band half width in voxels of the op layer. */
  halfWidth: number;
  /** Grid index in the file. */
  grid?: number;
}

/**
 * Leaf origins in voxels for every leaf of one grid, in leaf index order.
 * Walks roots, uppers, and lowers using the child counts packed per mask
 * word.
 */
export function leafOrigins(file: PicoVDBFile, gridIndex = 0): Int32Array {
  const u32 = (bytes: Uint8Array) => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const roots = u32(file.rootsBuffer);
  const uppers = u32(file.uppersBuffer);
  const lowers = u32(file.lowersBuffer);
  const range = file.getGridRange(gridIndex);
  const lowerOrigins = new Int32Array(range.lowerCount * 3);
  const origins = new Int32Array(range.leafCount * 3);
  // Child indices in a node are relative to the grid's first child node.
  const walk = (
    node: Uint32Array,
    base: number,
    words: number,
    bits: number,
    origin: [number, number, number],
    childSize: number,
    out: Int32Array
  ) => {
    const first = node[base];
    for (let w = 0; w < words; w++) {
      const children = node[base + 4 + w * 3] & node[base + 5 + w * 3];
      if (children === 0) continue;
      let index = first + (node[base + 6 + w * 3] >>> 16);
      for (let b = 0; b < 32; b++) {
        if (((children >>> b) & 1) === 0) continue;
        const n = w * 32 + b;
        const mask = (1 << bits) - 1;
        out[index * 3] = origin[0] + ((n >>> (2 * bits)) & mask) * childSize;
        out[index * 3 + 1] = origin[1] + ((n >>> bits) & mask) * childSize;
        out[index * 3 + 2] = origin[2] + (n & mask) * childSize;
        index++;
      }
    }
  };
  for (let u = 0; u < range.upperCount; u++) {
    // Mirrors picovdb coordToKey.
    const w0 = roots[(range.upperStart + u) * 2];
    const w1 = roots[(range.upperStart + u) * 2 + 1];
    const iu = w1 >>> 10;
    const ju = (w0 >>> 21) | ((w1 & 0x3ff) << 11);
    const ku = w0 & 0x1fffff;
    const origin: [number, number, number] = [(iu << 12) | 0, (ju << 12) | 0, (ku << 12) | 0];
    walk(uppers, (range.upperStart + u) * (PICOVDB_UPPER_SIZE / 4), 1024, 5, origin, 128, lowerOrigins);
  }
  for (let l = 0; l < range.lowerCount; l++) {
    const origin: [number, number, number] = [lowerOrigins[l * 3], lowerOrigins[l * 3 + 1], lowerOrigins[l * 3 + 2]];
    walk(lowers, (range.lowerStart + l) * (PICOVDB_LOWER_SIZE / 4), 128, 4, origin, 8, origins);
  }
  return origins;
}

export class Loader {
  readonly device: GPUDevice;
  private readonly pipelines: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: loadWgsl });
    for (const entryPoint of ['gather_leaves', 'convert_data']) {
      this.pipelines[entryPoint] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    }
  }

  /**
   * Loads one grid of an f32 or u8 SDF file. Values scale so the file's
   * background equals the half width. That also brings world unit files
   * into voxel units. u8 values map to [-3, 3], as the renderer reads
   * them.
   */
  load(file: PicoVDBFile, opts: LoadOptions): OpGrid {
    const device = this.device;
    const gridIndex = opts.grid ?? 0;
    const grid = file.getGrid(gridIndex);
    if (grid.gridType !== GRID_TYPE_SDF_FLOAT && grid.gridType !== GRID_TYPE_SDF_UINT8) {
      throw new Error(`grid type ${grid.gridType} is not an SDF`);
    }
    const range = file.getGridRange(gridIndex);
    const leafCount = range.leafCount;
    if (leafCount === 0) throw new Error('empty tree');
    const u8 = grid.gridType === GRID_TYPE_SDF_UINT8;
    // The grid's records and values. dataStart counts 16 byte units.
    const leafBytes = file.leavesBuffer.subarray(range.leafStart * LEAF_U32 * 4, (range.leafStart + leafCount) * LEAF_U32 * 4);
    const dataBytes = file.dataBuffer.subarray(range.dataStart * 16, range.dataStart * 16 + Math.ceil((grid.dataElemCount * (u8 ? 1 : 4)) / 4) * 4);
    const background = u8 ? (dataBytes[0] / 127.5 - 1) * 3 : new Float32Array(dataBytes.buffer, dataBytes.byteOffset, 1)[0];
    if (!(background > 0)) throw new Error(`background ${background} is not positive`);

    const origins = leafOrigins(file, gridIndex);
    const leafMin: [number, number, number] = [Infinity, Infinity, Infinity];
    const leafMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < leafCount; i++) {
      for (let a = 0; a < 3; a++) {
        const c = origins[i * 3 + a] >> 3;
        if (c < leafMin[a]) leafMin[a] = c;
        if (c > leafMax[a]) leafMax[a] = c;
      }
    }
    for (let a = 0; a < 3; a++) {
      if (leafMax[a] - leafMin[a] >= 1024) throw new Error(`tree exceeds 1024 leaves on axis ${a}`);
    }
    const keys = new Uint32Array(leafCount);
    for (let i = 0; i < leafCount; i++) {
      keys[i] =
        (((origins[i * 3] >> 3) - leafMin[0]) << 20) |
        (((origins[i * 3 + 1] >> 3) - leafMin[1]) << 10) |
        ((origins[i * 3 + 2] >> 3) - leafMin[2]);
    }
    const order = Array.from(keys.keys()).sort((a, b) => keys[a] - keys[b]);
    const sortedKeys = new Uint32Array(leafCount);
    for (let j = 0; j < leafCount; j++) sortedKeys[j] = keys[order[j]];

    const dataCount = grid.dataElemCount;
    checkBindingSize(device, dataCount * 4, 'loaded values');
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(params, 0, new Uint32Array([leafCount, dataCount]));
    device.queue.writeBuffer(params, 8, new Float32Array([opts.halfWidth / background]));
    device.queue.writeBuffer(params, 12, new Uint32Array([grid.gridType]));
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const leafKeys = createU32Buffer(device, sortedKeys);
    const orderBuffer = createU32Buffer(device, Uint32Array.from(order));
    const fileLeaves = createU32Buffer(device, new Uint32Array(leafBytes.buffer, leafBytes.byteOffset, leafCount * LEAF_U32));
    const fileData = createU32Buffer(device, new Uint32Array(dataBytes.buffer, dataBytes.byteOffset, dataBytes.byteLength / 4));
    const leaves = device.createBuffer({ size: leafCount * LEAF_U32 * 4, usage: storage });
    const data = device.createBuffer({ size: dataCount * 4, usage: storage });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    const run = (name: string, threads: number, buffers: Record<number, GPUBuffer>) => {
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
    run('gather_leaves', leafCount, { 1: orderBuffer, 2: fileLeaves, 4: leaves });
    run('convert_data', dataCount, { 3: fileData, 5: data });
    pass.end();
    device.queue.submit([encoder.finish()]);
    orderBuffer.destroy();
    fileLeaves.destroy();
    fileData.destroy();
    params.destroy();
    return { leafKeys, leaves, data, leafCount, activeVoxels: dataCount - 2, leafMin, leafMax };
  }
}
