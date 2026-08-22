import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { checkAnalytic, emptyGrid, assertU32ArrayEqual } from './test_util.ts';
import { Stamper, sphere as sphereShape } from './stamp.ts';
import { Remapper } from './remap.ts';

const gpu = await hasWebGPU();

Deno.test({ name: 'offset, translate, and rebase match the analytic sphere', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const stamper = new Stamper(device);
  const remapper = new Remapper(device);
  const halfWidth = 3;
  const center: [number, number, number] = [100.3, 97.2, 88.9];
  const sphere = (c: number[], r: number) => (p: [number, number, number]) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]) - r;
  const base = await stamper.stamp(emptyGrid(device), { shape: sphereShape(center, 20), mode: 'add', halfWidth });

  // Offsets redistance from the offset surface, exact to marching cubes
  // precision over the whole band: about a hundredth of a voxel per step
  // at this radius, with no seam. Offsets past halfWidth - 1 run in
  // steps, so their bias adds up.
  const grown = await remapper.offset(base, 2, halfWidth);
  const g = await checkAnalytic(device, grown, sphere(center, 22), halfWidth, 'grow', Infinity, 0.03);
  const shrunk = await remapper.offset(base, -2, halfWidth);
  const s = await checkAnalytic(device, shrunk, sphere(center, 18), halfWidth, 'shrink', Infinity, 0.03);
  if (!(g.band > s.band)) throw new Error(`grown band ${g.band} should exceed shrunk band ${s.band}`);
  const far = await remapper.offset(base, 5, halfWidth);
  await checkAnalytic(device, far, sphere(center, 25), halfWidth, 'grow past the band', Infinity, 0.05);

  // Translation is exact.
  const shift: [number, number, number] = [3, -5, 8];
  const moved = await remapper.translate(base, shift, halfWidth);
  await checkAnalytic(device, moved, sphere(center.map((c, a) => c + shift[a]), 20), halfWidth, 'translate');
  const movedLeaf = await remapper.translate(base, [8, -16, 0], halfWidth);
  if (movedLeaf.leafKeys !== base.leafKeys || movedLeaf.leafMin[1] !== base.leafMin[1] - 2) throw new Error('leaf multiple translate should only move the origin');
  await checkAnalytic(device, movedLeaf, sphere([center[0] + 8, center[1] - 16, center[2]], 20), halfWidth, 'translate leaves');

  // Rebase moves the key origin and nothing else.
  const rebased = remapper.rebase(base, [-3, 1, -7]);
  await checkAnalytic(device, rebased, sphere(center, 20), halfWidth, 'rebase');
  const keys = await readBackU32(device, rebased.leafKeys, rebased.leafCount);
  const expect = (await readBackU32(device, base.leafKeys, base.leafCount)).map((k) => (k + (3 << 20) + (-1 << 10) + 7) >>> 0);
  assertU32ArrayEqual(keys, expect, 'rebased keys');
  console.log(`  offset: base ${base.leafCount} leaves, grown ${grown.leafCount} (${g.band} band), shrunk ${shrunk.leafCount} (${s.band} band), moved ${moved.leafCount}`);
});
