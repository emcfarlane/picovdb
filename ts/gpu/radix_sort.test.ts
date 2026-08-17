import { hasWebGPU, requestDevice, createU32Buffer, readBackU32 } from './device.ts';
import { mulberry32 } from './test_util.ts';
import { assertU32ArrayEqual } from './compare.ts';
import { Sorter } from './radix_sort.ts';

const gpu = await hasWebGPU();

async function sortOnGpu(sorter: Sorter, keys: Uint32Array<ArrayBuffer>): Promise<{ keys: Uint32Array; vals: Uint32Array }> {
  const device = sorter.device;
  const n = keys.length;
  const vals = new Uint32Array(n);
  for (let i = 0; i < n; i++) vals[i] = i;

  const keyBuf = createU32Buffer(device, keys);
  const valBuf = createU32Buffer(device, vals);
  const plan = sorter.plan(keyBuf, valBuf, n);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  plan.encode(pass);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const out = { keys: await readBackU32(device, keyBuf, n), vals: await readBackU32(device, valBuf, n) };
  keyBuf.destroy();
  valBuf.destroy();
  return out;
}

Deno.test({ name: 'radix sort matches stable JS sort', ignore: !gpu }, async () => {
  const device = await requestDevice();
  const sorter = new Sorter(device);
  const rand = mulberry32(2);

  const cases: Array<{ n: number; keyBits: number }> = [
    { n: 1, keyBits: 32 },
    { n: 100, keyBits: 32 },
    { n: 1024, keyBits: 32 },
    { n: 1030, keyBits: 32 },
    { n: 65536, keyBits: 32 },
    { n: 1 << 20, keyBits: 32 },
    // Few distinct keys exercise stability with long equal runs.
    { n: 100000, keyBits: 4 },
  ];
  for (const { n, keyBits } of cases) {
    const keys = new Uint32Array(n);
    const mask = keyBits === 32 ? 0xffffffff : (1 << keyBits) - 1;
    for (let i = 0; i < n; i++) keys[i] = (Math.floor(rand() * 4294967296) & mask) >>> 0;

    // Array.prototype.sort is stable, so sorting indices by key is the
    // reference for both outputs.
    const order = [...Array(n).keys()].sort((a, b) => keys[a] - keys[b]);
    const expectedKeys = new Uint32Array(order.map((i) => keys[i]));
    const expectedVals = new Uint32Array(order);

    const got = await sortOnGpu(sorter, keys);
    assertU32ArrayEqual(got.keys, expectedKeys, `sorted keys n=${n} bits=${keyBits}`);
    assertU32ArrayEqual(got.vals, expectedVals, `payload order n=${n} bits=${keyBits}`);
  }
});
