import { hasWebGPU, requestDevice } from './device.ts';
import { checkAnalytic, emptyGrid } from './test_util.ts';
import { Stamper, box as boxShape, capsule as capsuleShape, cylinder as cylinderShape, sphere as sphereShape } from './stamp.ts';
import { Emitter } from './emit.ts';

const gpu = await hasWebGPU();

Deno.test({ name: 'brush stamps sculpt from an empty grid', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const stamper = new Stamper(device);
  const emitter = new Emitter(device);
  const halfWidth = 3;
  const center: [number, number, number] = [100.3, 97.2, 88.9];

  const sphere = (r: number) => (p: [number, number, number]) =>
    Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]) - r;

  // Add a sphere to empty space.
  const added = await stamper.stamp(emptyGrid(device), { shape: sphereShape(center, 20), mode: 'add', halfWidth });
  const a = await checkAnalytic(device, added, sphere(20), halfWidth, 'add');
  if (a.band === 0) throw new Error('no band voxels after add');

  // Carve a concentric hole so a shell remains.
  const carved = await stamper.stamp(added, { shape: sphereShape(center, 12), mode: 'carve', halfWidth });
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

Deno.test({ name: 'box, capsule, and cylinder stamps match their SDFs', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const stamper = new Stamper(device);
  const halfWidth = 3;
  const v = (a: number[], b: number[], s = 1) => a.map((x, i) => (x - b[i]) * s);
  const len = (a: number[]) => Math.hypot(a[0], a[1], a[2]);
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

  const center: [number, number, number] = [60.5, 70.25, 80];
  const half: [number, number, number] = [15, 9, 6];
  const box = (p: number[]) => {
    const q = v(p, center).map((x, i) => Math.abs(x) - half[i]);
    return len(q.map((x) => Math.max(x, 0))) + Math.min(Math.max(q[0], q[1], q[2]), 0) - 1.5;
  };
  const boxed = await stamper.stamp(emptyGrid(device), { shape: boxShape(center, half, 1.5), mode: 'add', halfWidth });
  await checkAnalytic(device, boxed, box, halfWidth, 'box');

  const a: [number, number, number] = [40, 40, 40];
  const b: [number, number, number] = [90.7, 63.1, 55];
  const capsule = (p: number[]) => {
    const pa = v(p, a);
    const ba = v(b, a);
    const h = Math.max(0, Math.min(1, dot(pa, ba) / dot(ba, ba)));
    return len(pa.map((x, i) => x - ba[i] * h)) - 7;
  };
  const capsuled = await stamper.stamp(emptyGrid(device), { shape: capsuleShape(a, b, 7), mode: 'add', halfWidth });
  await checkAnalytic(device, capsuled, capsule, halfWidth, 'capsule');

  const cylinder = (p: number[]) => {
    const pa = v(p, a);
    const ba = v(b, a);
    const baba = dot(ba, ba);
    const paba = dot(pa, ba);
    const x = len(pa.map((c, i) => c * baba - ba[i] * paba)) - 7 * baba;
    const y = Math.abs(paba - baba * 0.5) - baba * 0.5;
    const x2 = x * x;
    const y2 = y * y * baba;
    const d = Math.max(x, y) < 0 ? -Math.min(x2, y2) : (x > 0 ? x2 : 0) + (y > 0 ? y2 : 0);
    return Math.sign(d) * Math.sqrt(Math.abs(d)) / baba;
  };
  const cylindered = await stamper.stamp(emptyGrid(device), { shape: cylinderShape(a, b, 7), mode: 'add', halfWidth });
  await checkAnalytic(device, cylindered, cylinder, halfWidth, 'cylinder');
});
