import { hasWebGPU, requestDevice } from './device.ts';
import { compareTreeToCpu } from './test_util.ts';
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

  initSTL({ wasmBinary: wasm! });
  const cpu = (await importSTL(stl!, { voxelsPerUnit: 4 })).file;

  const binner = new Binner(device);
  const opts = { voxelSize: 0.25, halfWidth: 3 };
  const bin = await binner.bin(points, triangles, opts);
  const leafValues = new Rasterizer(device).rasterize(bin, opts);
  const sign = await new Signer(device).sign(bin);
  const tree = await new Emitter(device).emit(bin, leafValues, sign, opts);

  const maxAbs = await compareTreeToCpu(device, tree, cpu);
  console.log(
    `  tree: ${tree.leafCount} leaves / ${tree.lowerCount} lowers / ${tree.upperCount} uppers, ` +
    `${tree.activeVoxels} active, ${tree.surfaceVoxels} surface, max |Δv| ${maxAbs.toExponential(2)}`
  );
});
