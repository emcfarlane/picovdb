// Full-pipeline test: convert two meshes on the GPU in a shared key space,
// merge them (SDF union), re-emit the tree — and compare against the CPU
// converter run on the combined mesh. The two copies are far enough apart
// that their narrow bands and leaf sets are disjoint, so the union is exact
// and the CPU oracle applies.

import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { parseBinarySTL } from './reference.ts';
import { Binner, leafBounds } from './mesh_to_grid.ts';
import { Rasterizer } from './rasterize.ts';
import { Signer } from './sign.ts';
import { Emitter } from './emit.ts';
import { Merger } from './merge.ts';
import { initSTL, importSTL } from '../stl.ts';

const gpu = await hasWebGPU();

let stl: Uint8Array<ArrayBuffer> | null = null;
let wasm: Uint8Array<ArrayBuffer> | null = null;
try {
  stl = Deno.readFileSync(new URL('../../data/bases/base_32mm.stl', import.meta.url));
  wasm = Deno.readFileSync(new URL('../../zig-out/wasm/picovdb.wasm', import.meta.url));
} catch {
  // skip
}

function writeBinarySTL(points: Float32Array, triangles: Uint32Array): Uint8Array<ArrayBuffer> {
  const triCount = triangles.length / 3;
  const bytes = new Uint8Array(84 + triCount * 50);
  const view = new DataView(bytes.buffer);
  view.setUint32(80, triCount, true);
  for (let t = 0; t < triCount; t++) {
    const base = 84 + t * 50 + 12; // normal left zero
    for (let v = 0; v < 3; v++) {
      for (let c = 0; c < 3; c++) {
        view.setFloat32(base + (v * 3 + c) * 4, points[triangles[t * 3 + v] * 3 + c], true);
      }
    }
  }
  return bytes;
}

Deno.test({ name: 'full pipeline: convert x2, merge, re-emit matches CPU', ignore: !gpu || !stl || !wasm }, async () => {
  const device = await requestDevice();
  const meshA = parseBinarySTL(stl!);
  const voxelSize = 0.25;
  const halfWidth = 3;

  // Mesh B: the same base translated +50 world units in x (200 voxels —
  // far beyond the band, so leaf sets stay disjoint).
  const pointsB = new Float32Array(meshA.points);
  for (let i = 0; i < pointsB.length; i += 3) pointsB[i] = Math.fround(pointsB[i] + 50);

  // CPU oracle: convert the combined mesh.
  const combinedPoints = new Float32Array(meshA.points.length * 2);
  combinedPoints.set(meshA.points, 0);
  combinedPoints.set(pointsB, meshA.points.length);
  const combinedTris = new Uint32Array(meshA.triangles.length * 2);
  combinedTris.set(meshA.triangles, 0);
  for (let i = 0; i < meshA.triangles.length; i++) {
    combinedTris[meshA.triangles.length + i] = meshA.triangles[i] + meshA.points.length / 3;
  }
  initSTL({ wasmBinary: wasm! });
  const cpu = (await importSTL(writeBinarySTL(combinedPoints, combinedTris), { voxelsPerUnit: 4 })).file;

  // GPU: convert both meshes in the combined key space, merge, re-emit.
  const bounds = leafBounds(combinedPoints, Math.fround(1 / voxelSize), halfWidth);
  const opts = { voxelSize, halfWidth, bounds };
  const binner = new Binner(device);
  const rasterizer = new Rasterizer(device);
  const signer = new Signer(device);
  const emitter = new Emitter(device);

  const convert = async (points: Float32Array<ArrayBuffer>, triangles: Uint32Array<ArrayBuffer>) => {
    const bin = await binner.bin(points, triangles, opts);
    const values = rasterizer.rasterize(bin, opts);
    const sign = await signer.sign(bin);
    return emitter.classifyOnly(bin, values, sign, opts);
  };
  const gridA = await convert(meshA.points, meshA.triangles);
  const gridB = await convert(pointsB, meshA.triangles);

  const merged = await new Merger(device).merge(gridA, gridB);
  if (!merged.values) throw new Error('merge dropped values');
  const tree = await emitter.reEmit(
    { leafKeys: merged.leafKeys, masks: merged.masks, values: merged.values, leafCount: merged.leafCount, leafMin: bounds.leafMin },
    { halfWidth }
  );

  // Compare against the CPU tree.
  const h = cpu.header;
  if (tree.leafCount !== h.leafCount || tree.lowerCount !== h.lowerCount || tree.upperCount !== h.upperCount) {
    throw new Error(
      `counts: gpu ${tree.leafCount}/${tree.lowerCount}/${tree.upperCount} != cpu ${h.leafCount}/${h.lowerCount}/${h.upperCount}`
    );
  }
  const grid = cpu.getGrid(0);
  if (tree.dataElemCount !== grid.dataElemCount) {
    throw new Error(`dataElemCount: gpu ${tree.dataElemCount} != cpu ${grid.dataElemCount}`);
  }
  for (let a = 0; a < 3; a++) {
    if (tree.indexBoundsMin[a] !== grid.indexBoundsMin[a] || tree.indexBoundsMax[a] !== grid.indexBoundsMax[a]) {
      throw new Error(`index bounds mismatch on axis ${a}`);
    }
  }
  const cpuU32 = (bytes: Uint8Array) => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  assertU32ArrayEqual(
    await readBackU32(device, tree.roots, tree.upperCount * 2),
    cpuU32(cpu.rootsBuffer).slice(0, tree.upperCount * 2),
    'roots'
  );
  assertU32ArrayEqual(await readBackU32(device, tree.uppers, tree.upperCount * 3076), cpuU32(cpu.uppersBuffer), 'uppers');
  assertU32ArrayEqual(await readBackU32(device, tree.lowers, tree.lowerCount * 388), cpuU32(cpu.lowersBuffer), 'lowers');
  assertU32ArrayEqual(await readBackU32(device, tree.leaves, tree.leafCount * 52), cpuU32(cpu.leavesBuffer), 'leaves');

  const gpuData = new Float32Array((await readBackU32(device, tree.data, tree.dataElemCount)).buffer);
  const cpuData = new Float32Array(cpu.dataBuffer.buffer, cpu.dataBuffer.byteOffset, tree.dataElemCount);
  let maxAbs = 0;
  for (let i = 0; i < tree.dataElemCount; i++) {
    if ((gpuData[i] < 0) !== (cpuData[i] < 0)) throw new Error(`value sign mismatch at ${i}`);
    maxAbs = Math.max(maxAbs, Math.abs(gpuData[i] - cpuData[i]));
  }
  if (maxAbs > 1e-3) throw new Error(`value divergence ${maxAbs}`);
  console.log(
    `  pipeline: ${tree.leafCount} leaves / ${tree.lowerCount} lowers / ${tree.upperCount} uppers, ` +
    `${tree.activeVoxels} active, max |Δv| ${maxAbs.toExponential(2)}`
  );
});
