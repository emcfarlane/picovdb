import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { parseBinarySTL } from './reference.ts';
import { Binner } from './mesh_to_grid.ts';
import { Rasterizer } from './rasterize.ts';
import { Signer } from './sign.ts';
import { Emitter } from './emit.ts';
import { initSTL, importSTL } from '../stl.ts';

const gpu = await hasWebGPU();

// The CPU oracle is the wasm converter run on the same mesh. The test
// skips when either file is missing.
let stl: Uint8Array<ArrayBuffer> | null = null;
let wasm: Uint8Array<ArrayBuffer> | null = null;
try {
  stl = Deno.readFileSync(new URL('../../data/bases/base_32mm.stl', import.meta.url));
  wasm = Deno.readFileSync(new URL('../../zig-out/wasm/picovdb.wasm', import.meta.url));
} catch {
  // skip
}

Deno.test({ name: 'GPU tree emission matches CPU converter', ignore: !gpu || !stl || !wasm }, async () => {
  const device = await requestDevice();
  const { points, triangles } = parseBinarySTL(stl!);

  // CPU reference conversion.
  initSTL({ wasmBinary: wasm! });
  const cpu = (await importSTL(stl!, { voxelsPerUnit: 4 })).file;

  // GPU pipeline.
  const binner = new Binner(device);
  const opts = { voxelSize: 0.25, halfWidth: 3 };
  const bin = await binner.bin(points, triangles, opts);
  const leafValues = new Rasterizer(device).rasterize(bin, opts);
  const sign = await new Signer(device).sign(bin);
  const tree = await new Emitter(device).emit(bin, leafValues, sign, opts);

  // Structure counts.
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

  // Node buffers must match byte for byte.
  const cpuU32 = (bytes: Uint8Array) => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  assertU32ArrayEqual(
    await readBackU32(device, tree.roots, tree.upperCount * 2),
    cpuU32(cpu.rootsBuffer).slice(0, tree.upperCount * 2),
    'roots'
  );
  assertU32ArrayEqual(await readBackU32(device, tree.uppers, tree.upperCount * 3076), cpuU32(cpu.uppersBuffer), 'uppers');
  assertU32ArrayEqual(await readBackU32(device, tree.lowers, tree.lowerCount * 388), cpuU32(cpu.lowersBuffer), 'lowers');
  assertU32ArrayEqual(await readBackU32(device, tree.leaves, tree.leafCount * 52), cpuU32(cpu.leavesBuffer), 'leaves');

  // Values match within tolerance and signs match exactly.
  const gpuData = new Float32Array((await readBackU32(device, tree.data, tree.dataElemCount)).buffer);
  const cpuData = new Float32Array(cpu.dataBuffer.buffer, cpu.dataBuffer.byteOffset, tree.dataElemCount);
  let maxAbs = 0;
  for (let i = 0; i < tree.dataElemCount; i++) {
    if ((gpuData[i] < 0) !== (cpuData[i] < 0)) throw new Error(`value sign mismatch at ${i}: ${gpuData[i]} vs ${cpuData[i]}`);
    const d = Math.abs(gpuData[i] - cpuData[i]);
    if (d > maxAbs) maxAbs = d;
  }
  if (maxAbs > 1e-3) throw new Error(`value divergence ${maxAbs} voxels`);
  console.log(
    `  tree: ${tree.leafCount} leaves / ${tree.lowerCount} lowers / ${tree.upperCount} uppers, ` +
    `${tree.activeVoxels} active, ${tree.surfaceVoxels} surface, max |Δv| ${maxAbs.toExponential(2)}`
  );
});
