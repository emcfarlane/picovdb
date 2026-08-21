import { hasWebGPU, requestDevice, readBackU32 } from './device.ts';
import { emptyGrid } from './test_util.ts';
import { Stamper } from './stamp.ts';
import { Extractor } from './extract.ts';

const gpu = await hasWebGPU();

Deno.test({ name: 'extracted sphere mesh sits on the level set with the right area', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const halfWidth = 3;
  const center = [100.3, 97.2, 88.9];
  const r = 20;
  const grid = await new Stamper(device).stamp(emptyGrid(device), { shape: { kind: 'sphere', center: [center[0], center[1], center[2]], radius: r }, mode: 'add', halfWidth });
  const mesh = await new Extractor(device).extract(grid, halfWidth);
  const pts = new Float32Array((await readBackU32(device, mesh.points, mesh.triangleCount * 9)).buffer);
  let maxDev = 0;
  let area = 0;
  for (let t = 0; t < mesh.triangleCount; t++) {
    const p = [0, 1, 2].map((v) => [pts[t * 9 + v * 3], pts[t * 9 + v * 3 + 1], pts[t * 9 + v * 3 + 2]]);
    for (const q of p) maxDev = Math.max(maxDev, Math.abs(Math.hypot(q[0] - center[0], q[1] - center[1], q[2] - center[2]) - r));
    const u = p[1].map((x, i) => x - p[0][i]);
    const w = p[2].map((x, i) => x - p[0][i]);
    const c = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    area += 0.5 * Math.hypot(c[0], c[1], c[2]);
  }
  const expected = 4 * Math.PI * r * r;
  console.log(`  extract: ${mesh.triangleCount} triangles, max deviation ${maxDev.toFixed(4)} voxels, area ${area.toFixed(0)} vs ${expected.toFixed(0)}`);
  if (maxDev > 0.05) throw new Error(`vertices off the sphere by ${maxDev}`);
  if (Math.abs(area - expected) / expected > 0.02) throw new Error(`area ${area} vs ${expected}`);
});
