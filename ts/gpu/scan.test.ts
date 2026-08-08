import { hasWebGPU, requestDevice, createU32Buffer, readBackU32 } from './device.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { Scanner } from './scan.ts';

const gpu = await hasWebGPU();

// Deterministic PRNG so failures reproduce.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function exclusiveScanRef(input: Uint32Array): Uint32Array {
  const out = new Uint32Array(input.length);
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    out[i] = sum;
    sum = (sum + input[i]) >>> 0;
  }
  return out;
}

Deno.test({ name: 'exclusive scan matches JS reference', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const scanner = new Scanner(device);
  const rand = mulberry32(1);

  for (const n of [1, 7, 256, 1024, 1025, 4096, 65536, 1 << 20]) {
    const input = new Uint32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.floor(rand() * 1000);
    const expected = exclusiveScanRef(input);

    const buffer = createU32Buffer(device, input);
    const plan = scanner.plan(buffer, n);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    plan.encode(pass);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const got = await readBackU32(device, buffer, n);
    assertU32ArrayEqual(got, expected, `scan n=${n}`);
    buffer.destroy();
  }
});
