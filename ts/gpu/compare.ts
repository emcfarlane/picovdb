import assert from 'node:assert/strict';

/** Fast equality for large typed arrays; assert.deepEqual is O(minutes) at 1M elements. */
export function assertU32ArrayEqual(got: Uint32Array, expected: Uint32Array, label: string): void {
  assert.equal(got.length, expected.length, `${label}: length`);
  for (let i = 0; i < got.length; i++) {
    if (got[i] !== expected[i]) {
      assert.fail(`${label}: first mismatch at [${i}]: got ${got[i]}, expected ${expected[i]}`);
    }
  }
}
