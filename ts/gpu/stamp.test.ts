import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { Stamper } from './stamp.ts';
import { Emitter, type OpGrid } from './emit.ts';

const gpu = await hasWebGPU();

function emptyGrid(device: GPUDevice): OpGrid {
  const placeholder = () => device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  return { leafKeys: placeholder(), masks: placeholder(), values: placeholder(), leafCount: 0, leafMin: [0, 0, 0] };
}

// Every stored voxel of the stamped grid must match the expected SDF
// within tolerance and its band bit must match away from the boundary
// shadow.
async function checkAnalytic(
  device: GPUDevice,
  grid: OpGrid,
  expected: (p: [number, number, number]) => number,
  halfWidth: number,
  label: string
): Promise<{ band: number }> {
  const keys = await readBackU32(device, grid.leafKeys, grid.leafCount);
  const values = new Float32Array((await readBackU32(device, grid.values, grid.leafCount * 512)).buffer);
  const masks = await readBackU32(device, grid.masks, grid.leafCount * 16);
  const TOL = 1e-3;
  let band = 0;
  for (let i = 0; i < grid.leafCount; i++) {
    const ox = (((keys[i] >>> 20) & 0x3ff) + grid.leafMin[0]) * 8;
    const oy = (((keys[i] >>> 10) & 0x3ff) + grid.leafMin[1]) * 8;
    const oz = ((keys[i] & 0x3ff) + grid.leafMin[2]) * 8;
    for (let n = 0; n < 512; n++) {
      const p: [number, number, number] = [ox + (n >> 6), oy + ((n >> 3) & 7), oz + (n & 7)];
      const e = Math.max(-halfWidth, Math.min(halfWidth, expected(p)));
      const v = values[i * 512 + n];
      if (Math.abs(v - e) > TOL) {
        throw new Error(`${label}: value at ${p}: got ${v}, expected ${e}`);
      }
      const bit = (masks[i * 16 + (n >> 5)] >>> (n & 31)) & 1;
      if (bit) band++;
      if (Math.abs(Math.abs(e) - halfWidth) > 2 * TOL && bit !== (Math.abs(e) < halfWidth ? 1 : 0)) {
        throw new Error(`${label}: band bit at ${p}: got ${bit}, |v|=${Math.abs(v)}`);
      }
    }
  }
  return { band };
}

Deno.test({ name: 'brush stamps sculpt from an empty grid', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const stamper = new Stamper(device);
  const emitter = new Emitter(device);
  const halfWidth = 3;
  const center: [number, number, number] = [100.3, 97.2, 88.9];

  const sphere = (r: number) => (p: [number, number, number]) =>
    Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]) - r;

  // Add a sphere to empty space.
  const added = await stamper.stamp(emptyGrid(device), { center, radius: 20, mode: 'add', halfWidth });
  const a = await checkAnalytic(device, added, sphere(20), halfWidth, 'add');
  if (a.band === 0) throw new Error('no band voxels after add');

  // Carve a concentric hole so a shell remains.
  const carved = await stamper.stamp(added, { center, radius: 12, mode: 'carve', halfWidth });
  const shell = (p: [number, number, number]) => Math.max(sphere(20)(p), -sphere(12)(p));
  const c = await checkAnalytic(device, carved, shell, halfWidth, 'carve');
  if (c.band <= a.band) throw new Error(`carving should grow the band: ${c.band} <= ${a.band}`);

  // The sculpted grid emits into a tree.
  const tree = await emitter.reEmit(carved, { halfWidth });
  if (tree.surfaceVoxels === 0) throw new Error('no surface voxels in sculpted tree');
  const spanOk = tree.indexBoundsMax.every((v, axis) => v - tree.indexBoundsMin[axis] > 40);
  if (!spanOk) throw new Error(`implausible bounds: ${tree.indexBoundsMin} .. ${tree.indexBoundsMax}`);
  console.log(
    `  sculpt: add band=${a.band}, shell band=${c.band}; tree ${tree.leafCount} leaves, ` +
    `${tree.activeVoxels} active, ${tree.surfaceVoxels} surface`
  );
});
