import { hasWebGPU, requestDevice } from './gpu/device.ts';
import { checkAnalytic, compareTreeToCpu } from './gpu/test_util.ts';
import { Space, box as boxShape, sphere as sphereShape } from './model.ts';
import { PicoVDBFile } from './picovdb.ts';

const gpu = await hasWebGPU();

let bunny: Uint8Array<ArrayBuffer> | null = null;
try {
  bunny = Deno.readFileSync(new URL('../data/bunny.pvdb', import.meta.url));
} catch {
  // skip
}

type P = [number, number, number];
const sphere = (c: P, r: number) => (p: P) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - r;
const box = (c: P, h: P) => (p: P) => {
  const q = p.map((x, i) => Math.abs(x - c[i]) - h[i]);
  return Math.hypot(...q.map((x) => Math.max(x, 0))) + Math.min(Math.max(q[0], q[1], q[2]), 0);
};

Deno.test({ name: 'booleans between distant solids rebase and match the SDFs', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const space = new Space(device);
  const hw = space.halfWidth;

  // Two solids in unrelated key regions, far from any shared origin.
  const cs: P = [100.3, 97.2, 88.9];
  const cb: P = [3000, -2000.5, 500];
  using a = await space.solid(sphereShape(cs, 20));
  using b = await space.solid(boxShape(cb, [12, 7, 9]));
  using u = await a.union(b);
  await checkAnalytic(device, u.grid, (p) => Math.min(sphere(cs, 20)(p), box(cb, [12, 7, 9])(p)), hw, 'union');
  if (!u.bounds || u.bounds.min[1] !== Math.floor((cb[1] - 7 - hw) / 8)) throw new Error(`union bounds ${JSON.stringify(u.bounds)}`);

  // Overlapping booleans, with solid, op, and shape operands.
  const cc: P = [cs[0] + 15, cs[1], cs[2]];
  using c = await space.solid(sphereShape(cc, 20));
  using sub = await a.subtract(c);
  await checkAnalytic(device, sub.grid, (p) => Math.max(sphere(cs, 20)(p), -sphere(cc, 20)(p)), hw, 'subtract');
  using inter = await a.intersect(space.solid(sphereShape(cc, 20)));
  await checkAnalytic(device, inter.grid, (p) => Math.max(sphere(cs, 20)(p), sphere(cc, 20)(p)), hw, 'intersect');
  using shell = await a.subtract(sphereShape(cs, 12));
  await checkAnalytic(device, shell.grid, (p) => Math.max(sphere(cs, 20)(p), -sphere(cs, 12)(p)), hw, 'shape subtract');
  // A carve clips to the solid, so a half space far beyond the key range works.
  const hs: P = [cs[0] + 4000, 0, 0];
  using half = await a.subtract(boxShape(hs, [4000, 4000, 4000]));
  await checkAnalytic(device, half.grid, (p) => Math.max(sphere(cs, 20)(p), -box(hs, [4000, 4000, 4000])(p)), hw, 'half space carve');
  if (!(half.leafCount < a.leafCount)) throw new Error(`half space carve should drop leaves: ${half.leafCount} vs ${a.leafCount}`);

  // A chain frees its intermediates and resolves to the same result.
  using chained = await space.solid(sphereShape(cs, 20)).subtract(c).union(sphereShape(cs, 12)).intersect(a);
  await checkAnalytic(device, chained.grid, (p) => Math.max(Math.min(Math.max(sphere(cs, 20)(p), -sphere(cc, 20)(p)), sphere(cs, 12)(p)), sphere(cs, 20)(p)), hw, 'chain');

  // Empty solids.
  using e = space.empty();
  using eu = await e.union(a);
  await checkAnalytic(device, eu.grid, sphere(cs, 20), hw, 'empty union');
  using ei = await a.intersect(e);
  using es = await e.subtract(a);
  if (ei.leafCount !== 0 || es.leafCount !== 0) throw new Error('empty intersect or subtract should be empty');

  // toPvdb round trips the tree byte for byte.
  const tree = await u.toTree();
  const file = new PicoVDBFile(await u.toPvdb());
  await compareTreeToCpu(device, tree, file);
  using back = space.fromPvdb(file);
  await checkAnalytic(device, back.grid, (p) => Math.min(sphere(cs, 20)(p), box(cb, [12, 7, 9])(p)), hw, 'fromPvdb(toPvdb)');
  console.log(`  union: ${u.leafCount} leaves -> tree ${tree.leafCount} leaves / ${tree.upperCount} uppers, ${tree.surfaceVoxels} surface, ${file.getSize()} bytes`);
});

Deno.test({ name: 'bunny grows, hollows, moves, and emits', ignore: !gpu || !bunny }, async () => {
  const device = await requestDevice();
  const space = new Space(device);
  using bunny_ = space.fromPvdb(new PicoVDBFile(bunny!.buffer));
  const t0 = performance.now();
  using moved = await bunny_.offset(1.5).subtract(bunny_).translate([1000, 5, -3]);
  const tree = await moved.toTree();
  const ms = performance.now() - t0;
  const base = await bunny_.toTree();
  // The shell has an inner and an outer surface.
  if (tree.surfaceVoxels < 1.5 * base.surfaceVoxels) throw new Error(`hollow shell should double the surface: ${tree.surfaceVoxels} vs ${base.surfaceVoxels}`);
  if (tree.indexBoundsMin[0] < base.indexBoundsMin[0] + 990) throw new Error(`translate did not move the bounds: ${tree.indexBoundsMin}`);
  console.log(`  bunny: ${base.leafCount} leaves -> shell ${tree.leafCount} leaves, ${tree.surfaceVoxels} surface, ${ms.toFixed(0)} ms for offset + subtract + translate + emit`);
});
