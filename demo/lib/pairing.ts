// Pairing textures for ReSTIR paired spatial reuse (ReSTIR PT Enhanced §3,
// Lin/Kettunen/Wyman 2026). Each texel stores a signed coordinate delta to
// its paired neighbor; pairing is self-inverting (A links to B, B to A) and
// the deltas follow ~N(0, sigma) by construction: consecutive link indices
// are shuffled n_sigma times in 2x2 blocks (every other pass offset
// diagonally by one, wrapping), then each index pairs with index^1.
//
// The texture tiles over the screen; per-frame random flips/offsets (see
// compute.wgsl) decorrelate reuse across frames.

/** Builds a size x size pairing texture of int8 (dx, dy) deltas. */
export function buildPairingTexture(size: number, sigma: number, rand: () => number = Math.random): Int8Array<ArrayBuffer> {
  const n = size * size;
  const grid = new Int32Array(n);
  for (let i = 0; i < n; i++) grid[i] = i;
  // Repeat count from Eq. 3 (function-fit correction for small sigma)
  const iters = Math.round(
    (sigma * sigma) / 2 + 1.46 / sigma + 1.76 / (sigma * sigma) + 0.656 / (sigma ** 3) + 0.5);
  const cells = new Int32Array(4);
  for (let it = 0; it < iters; it++) {
    const off = it & 1;
    for (let by = 0; by < size; by += 2) {
      for (let bx = 0; bx < size; bx += 2) {
        const x0 = (bx + off) % size, x1 = (bx + off + 1) % size;
        const y0 = (by + off) % size, y1 = (by + off + 1) % size;
        cells[0] = y0 * size + x0; cells[1] = y0 * size + x1;
        cells[2] = y1 * size + x0; cells[3] = y1 * size + x1;
        // Fisher-Yates over the 4 cells
        for (let k = 3; k > 0; k--) {
          const j = (rand() * (k + 1)) | 0;
          const a = grid[cells[k]];
          grid[cells[k]] = grid[cells[j]];
          grid[cells[j]] = a;
        }
      }
    }
  }
  // Locate each link index, pair i with i^1, store deltas
  const posX = new Int32Array(n);
  const posY = new Int32Array(n);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = grid[y * size + x];
      posX[i] = x;
      posY[i] = y;
    }
  }
  const out = new Int8Array(n * 2);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const partner = grid[y * size + x] ^ 1;
      let dx = posX[partner] - x;
      let dy = posY[partner] - y;
      // Break links longer than half the texture (tileability)
      if (dx > half) dx -= size;
      if (dx < -half) dx += size;
      if (dy > half) dy -= size;
      if (dy < -half) dy += size;
      out[(y * size + x) * 2] = dx;
      out[(y * size + x) * 2 + 1] = dy;
    }
  }
  return out;
}
