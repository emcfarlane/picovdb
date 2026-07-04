import { test } from "node:test";
import assert from "node:assert/strict";
import { buildPairingTexture } from "./pairing.ts";

// Deterministic RNG for reproducible tests
function mulberry32(seed: number) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SIZE = 254;
const SIGMA = 16.0;
const tex = buildPairingTexture(SIZE, SIGMA, mulberry32(7));

test("pairing is self-inverting on the torus", () => {
  for (let y = 0; y < SIZE; y += 3) {
    for (let x = 0; x < SIZE; x += 3) {
      const i = (y * SIZE + x) * 2;
      const dx = tex[i], dy = tex[i + 1];
      const px = ((x + dx) % SIZE + SIZE) % SIZE;
      const py = ((y + dy) % SIZE + SIZE) % SIZE;
      const j = (py * SIZE + px) * 2;
      // Partner points back (modulo the tile-break wrapping of both deltas)
      const bx = ((tex[j] + dx) % SIZE + SIZE) % SIZE;
      const by = ((tex[j + 1] + dy) % SIZE + SIZE) % SIZE;
      assert.ok(bx === 0 && by === 0,
        `pixel (${x},${y}) delta (${dx},${dy}) partner delta (${tex[j]},${tex[j + 1]})`);
    }
  }
});

test("deltas are zero-mean with std near sigma", () => {
  let sx = 0, sy = 0, sxx = 0, syy = 0;
  const n = SIZE * SIZE;
  for (let i = 0; i < n; i++) {
    sx += tex[i * 2]; sy += tex[i * 2 + 1];
    sxx += tex[i * 2] ** 2; syy += tex[i * 2 + 1] ** 2;
  }
  const meanX = sx / n, meanY = sy / n;
  const stdX = Math.sqrt(sxx / n - meanX ** 2);
  const stdY = Math.sqrt(syy / n - meanY ** 2);
  assert.ok(Math.abs(meanX) < 0.5 && Math.abs(meanY) < 0.5, `mean (${meanX}, ${meanY})`);
  // The 2D delta std should be near sigma; allow generous tolerance since
  // the repeat-count formula is itself a fit
  const std2d = Math.sqrt(stdX ** 2 + stdY ** 2);
  assert.ok(std2d > SIGMA * 0.6 && std2d < SIGMA * 1.5, `2d std ${std2d} vs sigma ${SIGMA}`);
});

test("no pixel links to itself", () => {
  for (let i = 0; i < SIZE * SIZE; i++) {
    assert.ok(tex[i * 2] !== 0 || tex[i * 2 + 1] !== 0, `pixel ${i} links to itself`);
  }
});
