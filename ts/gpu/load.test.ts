import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { compareTreeToCpu } from './test_util.ts';
import { Emitter } from './emit.ts';
import { Loader, leafOrigins } from './load.ts';
import { initSTL, importSTL } from '../stl.ts';
import { PicoVDBFile } from '../picovdb.ts';

const gpu = await hasWebGPU();

let stl: Uint8Array<ArrayBuffer> | null = null;
let wasm: Uint8Array<ArrayBuffer> | null = null;
let bunny: Uint8Array<ArrayBuffer> | null = null;
let bunnyU8: Uint8Array<ArrayBuffer> | null = null;
try {
  stl = Deno.readFileSync(new URL('../../data/bases/base_32mm.stl', import.meta.url));
  wasm = Deno.readFileSync(new URL('../../zig-out/wasm/picovdb.wasm', import.meta.url));
} catch {
  // skip
}
try {
  bunny = Deno.readFileSync(new URL('../../data/bunny.pvdb', import.meta.url));
} catch {
  // skip
}
try {
  const gz = Deno.readFileSync(new URL('../../data/bunny.u8.pvdb.gz', import.meta.url));
  const raw = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  bunnyU8 = new Uint8Array(raw);
} catch {
  // skip
}

Deno.test({ name: 'load then emit reproduces the CPU tree', ignore: !gpu || !stl || !wasm }, async () => {
  const device = await requestDevice();
  initSTL({ wasmBinary: wasm! });
  const cpu = (await importSTL(stl!, { voxelsPerUnit: 4 })).file;

  const grid = new Loader(device).load(cpu, { halfWidth: 3 });
  const tree = await new Emitter(device).reEmit(grid, { halfWidth: 3 });
  const maxAbs = await compareTreeToCpu(device, tree, cpu);
  console.log(`  load round trip: ${tree.leafCount} leaves, ${tree.activeVoxels} active, max |Δv| ${maxAbs.toExponential(2)}`);
});

Deno.test({ name: 'leaf origins recover the leaf order of a converted file', ignore: !bunny }, () => {
  const file = new PicoVDBFile(bunny!.buffer);
  const origins = leafOrigins(file);
  const grid = file.getGrid(0);
  // Every origin sits inside the grid's index bounds and on a leaf boundary.
  for (let i = 0; i < file.header.leafCount; i++) {
    for (let a = 0; a < 3; a++) {
      const o = origins[i * 3 + a];
      if (o & 7) throw new Error(`leaf ${i} origin ${o} not leaf aligned`);
      if (o + 7 < grid.indexBoundsMin[a] || o > grid.indexBoundsMax[a]) throw new Error(`leaf ${i} origin ${o} outside bounds on axis ${a}`);
    }
  }
});

Deno.test({ name: 'bunny loads and re-emits with the same topology', ignore: !gpu || !bunny }, async () => {
  const device = await requestDevice();
  const file = new PicoVDBFile(bunny!.buffer);
  const grid = new Loader(device).load(file, { halfWidth: 3 });
  const tree = await new Emitter(device).reEmit(grid, { halfWidth: 3 });
  const h = file.header;
  if (tree.leafCount !== h.leafCount || tree.lowerCount !== h.lowerCount || tree.upperCount !== h.upperCount) {
    throw new Error(`counts: gpu ${tree.leafCount}/${tree.lowerCount}/${tree.upperCount} != file ${h.leafCount}/${h.lowerCount}/${h.upperCount}`);
  }
  if (tree.dataElemCount !== file.getGrid(0).dataElemCount) throw new Error('active voxel count differs');
  // Values scaled from the file's background to the half width.
  const data = new Float32Array((await readBackU32(device, tree.data, tree.dataElemCount)).buffer);
  const src = new Float32Array(file.dataBuffer.buffer, file.dataBuffer.byteOffset, tree.dataElemCount);
  const scale = 3 / src[0];
  let maxAbs = 0;
  for (let i = 2; i < tree.dataElemCount; i++) maxAbs = Math.max(maxAbs, Math.abs(data[i] - src[i] * scale));
  if (maxAbs > 1e-4) throw new Error(`value divergence ${maxAbs}`);
  console.log(`  bunny: ${tree.leafCount} leaves, ${tree.activeVoxels} active, ${tree.surfaceVoxels} surface, max |Δv| ${maxAbs.toExponential(2)}`);
});

Deno.test({ name: 'u8 bunny loads to the f32 bunny within quantization', ignore: !gpu || !bunny || !bunnyU8 }, async () => {
  const device = await requestDevice();
  const loader = new Loader(device);
  const f32 = loader.load(new PicoVDBFile(bunny!.buffer), { halfWidth: 3 });
  const u8 = loader.load(new PicoVDBFile(bunnyU8!.buffer), { halfWidth: 3 });
  if (u8.leafCount !== f32.leafCount) throw new Error(`leaf counts ${u8.leafCount} != ${f32.leafCount}`);
  const keysA = await readBackU32(device, f32.leafKeys, f32.leafCount);
  const keysB = await readBackU32(device, u8.leafKeys, u8.leafCount);
  for (let i = 0; i < keysA.length; i++) if (keysA[i] !== keysB[i]) throw new Error(`leaf key ${i} differs`);
  if (u8.activeVoxels !== f32.activeVoxels) throw new Error(`active counts ${u8.activeVoxels} != ${f32.activeVoxels}`);
  const a = new Float32Array((await readBackU32(device, f32.data, 2 + f32.activeVoxels)).buffer);
  const b = new Float32Array((await readBackU32(device, u8.data, 2 + u8.activeVoxels)).buffer);
  let maxAbs = 0;
  for (let i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  // One u8 step is 6 / 255 voxels; the f32 file is 3 / 0.15 = 20x rescaled.
  if (maxAbs > 6 / 255) throw new Error(`u8 values off by ${maxAbs}`);
  console.log(`  u8 bunny: ${u8.leafCount} leaves, max |Δv| ${maxAbs.toExponential(2)}`);
});
