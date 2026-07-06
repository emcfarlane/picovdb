// Verifies the TS tree walk (leaf origins) and the derived surface-voxel
// count against structural invariants of real model files.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { PicoVDBFile } from "../../picovdb.ts";

function load(name: string): PicoVDBFile {
  const path = fileURLToPath(new URL(`../../data/${name}`, import.meta.url));
  const raw = gunzipSync(readFileSync(path));
  return new PicoVDBFile(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer);
}

for (const name of ["sphere.pvdb.gz", "bunny.pvdb.gz"]) {
  test(`${name}: leaf origins are unique, aligned, in bounds`, () => {
    const file = load(name);
    const origins = file.getLeafOrigins(); // throws if any leaf unreachable
    const grid = file.getGrid(0);
    const seen = new Set<string>();
    for (let i = 0; i < file.header.leafCount; i++) {
      const x = origins[i * 4], y = origins[i * 4 + 1], z = origins[i * 4 + 2];
      assert.ok((x & 7) === 0 && (y & 7) === 0 && (z & 7) === 0, `leaf ${i} misaligned`);
      assert.ok(
        x >= grid.indexBoundsMin[0] - 8 && x <= grid.indexBoundsMax[0] + 8 &&
        y >= grid.indexBoundsMin[1] - 8 && y <= grid.indexBoundsMax[1] + 8 &&
        z >= grid.indexBoundsMin[2] - 8 && z <= grid.indexBoundsMax[2] + 8,
        `leaf ${i} origin (${x},${y},${z}) outside grid bounds`);
      const k = `${x},${y},${z}`;
      assert.ok(!seen.has(k), `duplicate origin ${k}`);
      seen.add(k);
    }
  });

  test(`${name}: surface count matches the leaf prefix sums`, () => {
    const file = load(name);
    const surfaceCount = file.getSurfaceVoxelCount();
    assert.ok(surfaceCount > 0);
    // The converter assigns baseInsideIndex as a running prefix sum over
    // leaves; the final leaf's base + its own surface bits = the total.
    let maxEnd = 0;
    for (let i = 0; i < file.header.leafCount; i++) {
      const leaf = file.getLeaf(i);
      let bits = 0;
      for (const e of leaf.elements) {
        let v = e.stateMask & e.valueMask;
        v = v - ((v >>> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
        bits += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
      }
      maxEnd = Math.max(maxEnd, leaf.baseInsideIndex + bits);
    }
    assert.equal(surfaceCount, maxEnd);
  });
}
