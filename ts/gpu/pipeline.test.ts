// Converts two meshes on the GPU in a shared key space, merges them, and
// emits the tree, comparing against the CPU converter run on the combined
// mesh. The copies sit far enough apart that bands and leaf sets stay
// disjoint, so the CPU oracle applies.

import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { compareTreeToCpu } from './test_util.ts';
import { parseBinarySTL, refCsgMerge } from './reference.ts';
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

  // Mesh B is the same base translated 50 world units in x, far beyond
  // the band, so leaf sets stay disjoint.
  const pointsB = new Float32Array(meshA.points);
  for (let i = 0; i < pointsB.length; i += 3) pointsB[i] = Math.fround(pointsB[i] + 50);

  // The CPU oracle converts the combined mesh.
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

  // The GPU converts both meshes in the combined key space, merges, and emits.
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

  const merged = await new Merger(device).merge(gridA, gridB, { halfWidth });
  if (!merged.values) throw new Error('merge dropped values');
  const tree = await emitter.reEmit(
    { leafKeys: merged.leafKeys, masks: merged.masks, values: merged.values, leafCount: merged.leafCount, leafMin: bounds.leafMin, leafMax: bounds.leafMax },
    { halfWidth }
  );

  // Compare against the CPU tree.
  const maxAbs = await compareTreeToCpu(device, tree, cpu);
  console.log(
    `  pipeline: ${tree.leafCount} leaves / ${tree.lowerCount} lowers / ${tree.upperCount} uppers, ` +
    `${tree.activeVoxels} active, max |Δv| ${maxAbs.toExponential(2)}`
  );
});

Deno.test({ name: 'overlapping solids: CSG merge deactivates swallowed band', ignore: !gpu || !stl }, async () => {
  const device = await requestDevice();
  const meshA = parseBinarySTL(stl!);
  const voxelSize = 0.25;
  const halfWidth = 3;

  // Mesh B overlaps A, shifted 2.5 world units in x.
  const pointsB = new Float32Array(meshA.points);
  for (let i = 0; i < pointsB.length; i += 3) pointsB[i] = Math.fround(pointsB[i] + 2.5);
  const combinedPoints = new Float32Array(meshA.points.length * 2);
  combinedPoints.set(meshA.points, 0);
  combinedPoints.set(pointsB, meshA.points.length);

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
  const merged = await new Merger(device).merge(gridA, gridB, { halfWidth });

  // The GPU CSG merge must match the JS reference built from the two
  // source grids.
  const aKeys = await readBackU32(device, gridA.leafKeys, gridA.leafCount);
  const aVals = new Float32Array((await readBackU32(device, gridA.values, gridA.leafCount * 512)).buffer);
  const bKeys = await readBackU32(device, gridB.leafKeys, gridB.leafCount);
  const bVals = new Float32Array((await readBackU32(device, gridB.values, gridB.leafCount * 512)).buffer);
  const ref = refCsgMerge(aKeys, aVals, bKeys, bVals, halfWidth);
  if (merged.leafCount !== ref.keys.length) throw new Error(`count ${merged.leafCount} != ${ref.keys.length}`);
  assertU32ArrayEqual(await readBackU32(device, merged.leafKeys, merged.leafCount), ref.keys, 'overlap keys');
  assertU32ArrayEqual(await readBackU32(device, merged.masks, merged.leafCount * 16), ref.masks, 'overlap masks');
  assertU32ArrayEqual(
    await readBackU32(device, merged.values!, merged.leafCount * 512),
    new Uint32Array(ref.values.buffer),
    'overlap values'
  );

  // The swallowed band must deactivate. The union band is smaller than the
  // two bands combined and at least one solid's worth.
  const popcount = (v: number) => {
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  };
  const sumBits = (words: Uint32Array) => words.reduce((s, w) => s + popcount(w), 0);
  const activeA = sumBits(await readBackU32(device, gridA.masks, gridA.leafCount * 16));
  const activeB = sumBits(await readBackU32(device, gridB.masks, gridB.leafCount * 16));
  const activeUnion = sumBits(ref.masks);
  if (activeUnion >= activeA + activeB) throw new Error(`no deactivation: ${activeUnion} >= ${activeA} + ${activeB}`);
  if (activeUnion < activeA) throw new Error(`union band implausibly small: ${activeUnion} < ${activeA}`);

  // The edited grid emits into a tree.
  const tree = await emitter.reEmit(
    { leafKeys: merged.leafKeys, masks: merged.masks, values: merged.values!, leafCount: merged.leafCount, leafMin: bounds.leafMin, leafMax: bounds.leafMax },
    { halfWidth }
  );
  if (tree.activeVoxels !== activeUnion) throw new Error(`tree active ${tree.activeVoxels} != ${activeUnion}`);
  console.log(
    `  overlap: A=${activeA} B=${activeB} union=${activeUnion} active; ` +
    `tree ${tree.leafCount} leaves / ${tree.lowerCount} lowers / ${tree.upperCount} uppers, ${tree.surfaceVoxels} surface`
  );
});
