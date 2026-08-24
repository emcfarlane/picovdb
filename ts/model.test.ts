import { hasWebGPU, requestDevice } from './gpu/device.ts';
import { checkAnalytic, compareTreeToCpu } from './gpu/test_util.ts';
import { Space, box as boxShape, sphere as sphereShape } from './model.ts';
import { PICOVDB_LEAF_SIZE, PICOVDB_LOWER_SIZE, PICOVDB_MAGIC, PICOVDB_UPPER_SIZE, PicoVDBFile } from './picovdb.ts';

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

Deno.test({ name: 'custom shape functions stamp by name and report compile errors', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const space = new Space(device, {
    shapes: /* wgsl */ `
      // A torus in the xz plane: args[0].xyz center, args[1].x ring radius, args[1].y tube radius.
      fn torus(p: vec3<f32>) -> f32 {
        let d = p - args[0].xyz;
        let q = vec2<f32>(length(d.xz) - args[1].x, d.y);
        return length(q) - args[1].y;
      }
    `,
  });
  const hw = space.halfWidth;
  const c: P = [100.3, 97.2, 88.9];
  const torus = (p: P) => Math.hypot(Math.hypot(p[0] - c[0], p[2] - c[2]) - 18, p[1] - c[1]) - 6;
  const shape = { fn: 'torus', args: [...c, 0, 18, 6], bounds: { min: [c[0] - 24, c[1] - 6, c[2] - 24] as P, max: [c[0] + 24, c[1] + 6, c[2] + 24] as P } };
  using ring = await space.solid(shape);
  await checkAnalytic(device, ring.grid, torus, hw, 'torus');
  // As an operand the same function carves.
  using ball = await space.solid(sphereShape(c, 20));
  using notched = await ball.subtract(shape);
  await checkAnalytic(device, notched.grid, (p) => Math.max(sphere(c, 20)(p), -torus(p)), hw, 'torus carve');
  // Adding needs bounds; carving does not.
  let message = '';
  try { await space.solid({ fn: 'torus', args: shape.args }); } catch (e) { message = (e as Error).message; }
  if (!message.includes('bounds')) throw new Error(`expected a bounds error, got: ${message}`);
  using whole = await ball.subtract({ fn: 'torus', args: shape.args });
  await checkAnalytic(device, whole.grid, (p) => Math.max(sphere(c, 20)(p), -torus(p)), hw, 'unbounded torus carve');
  // An unknown function, and a library that does not compile, report the WGSL error.
  message = '';
  try { await ball.subtract({ fn: 'missing' }); } catch (e) { message = (e as Error).message; }
  if (!message.includes('missing')) throw new Error(`expected an unknown function error, got: ${message}`);
  const bad = new Space(device, { shapes: 'fn broken(p: vec3<f32>) -> f32 { return p; }' });
  message = '';
  try { await bad.empty().subtract({ fn: 'broken' }); } catch (e) { message = (e as Error).message; }
  if (!message.includes('broken') || !message.includes('line')) throw new Error(`expected a compile error, got: ${message}`);
});

/** A two grid file from two single grid files, as a multi-grid writer would lay it out. */
function stitch(a: PicoVDBFile, b: PicoVDBFile): ArrayBuffer {
  const ha = a.header;
  const hb = b.header;
  const upperCount = ha.upperCount + hb.upperCount;
  const rootsPadded = Math.ceil(upperCount / 2) * 2;
  const sections = [
    [a.rootsBuffer.subarray(0, ha.upperCount * 8), b.rootsBuffer.subarray(0, hb.upperCount * 8), new Uint8Array((rootsPadded - upperCount) * 8)],
    [a.uppersBuffer, b.uppersBuffer],
    [a.lowersBuffer, b.lowersBuffer],
    [a.leavesBuffer, b.leavesBuffer],
    [a.dataBuffer, b.dataBuffer],
  ];
  const bodyBytes = sections.flat().reduce((n, s) => n + s.byteLength, 0);
  const out = new Uint8Array(32 + 2 * 64 + bodyBytes);
  new Uint32Array(out.buffer, 0, 8).set([PICOVDB_MAGIC[0], PICOVDB_MAGIC[1], 0, 2, upperCount, ha.lowerCount + hb.lowerCount, ha.leafCount + hb.leafCount, ha.dataCount + hb.dataCount]);
  out.set(a.gridsBuffer, 32);
  out.set(b.gridsBuffer, 96);
  new Uint32Array(out.buffer, 96 + 4, 4).set([ha.upperCount, ha.lowerCount, ha.leafCount, ha.dataCount]);
  let offset = 160;
  for (const part of sections.flat()) {
    out.set(part, offset);
    offset += part.byteLength;
  }
  if (a.uppersBuffer.byteLength !== ha.upperCount * PICOVDB_UPPER_SIZE || a.lowersBuffer.byteLength !== ha.lowerCount * PICOVDB_LOWER_SIZE || a.leavesBuffer.byteLength !== ha.leafCount * PICOVDB_LEAF_SIZE) throw new Error('unexpected node buffer sizes');
  return out.buffer;
}

Deno.test({ name: 'fromPvdb loads a chosen grid of a multi-grid file', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const space = new Space(device);
  const hw = space.halfWidth;
  const cs: P = [100.3, 97.2, 88.9];
  const cb: P = [-200, 50, 30];
  using a = await space.solid(sphereShape(cs, 20));
  using b = await space.solid(boxShape(cb, [12, 7, 9]));
  const file = new PicoVDBFile(stitch(new PicoVDBFile(await a.toPvdb()), new PicoVDBFile(await b.toPvdb())));
  if (file.header.gridCount !== 2) throw new Error(`stitched file has ${file.header.gridCount} grids`);
  using g0 = space.fromPvdb(file, 0);
  await checkAnalytic(device, g0.grid, sphere(cs, 20), hw, 'grid 0');
  using g1 = space.fromPvdb(file, 1);
  await checkAnalytic(device, g1.grid, box(cb, [12, 7, 9]), hw, 'grid 1');
  if (g0.leafCount !== a.leafCount || g1.leafCount !== b.leafCount) throw new Error(`leaf counts ${g0.leafCount}/${g1.leafCount} vs ${a.leafCount}/${b.leafCount}`);
});
