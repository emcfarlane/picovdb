// f32 exact reference implementations of the GPU stages, used by the
// tests. Every operation rounds through Math.fround, which matches the
// correctly rounded f32 result for f32 operands evaluated in f64, so
// these mirror the WGSL bit for bit.

const f = Math.fround;

type V3 = [number, number, number];

function sub(a: V3, b: V3): V3 {
  return [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
}

function add(a: V3, b: V3): V3 {
  return [f(a[0] + b[0]), f(a[1] + b[1]), f(a[2] + b[2])];
}

function scale(a: V3, s: number): V3 {
  return [f(a[0] * s), f(a[1] * s), f(a[2] * s)];
}

function dot3(a: V3, b: V3): number {
  return f(f(f(a[0] * b[0]) + f(a[1] * b[1])) + f(a[2] * b[2]));
}

function dsq(a: V3): number {
  return dot3(a, a);
}

function distSqPointSegment(p: V3, a: V3, b: V3): number {
  const ab = sub(b, a);
  const denom = dsq(ab);
  if (denom <= 0) return dsq(sub(p, a));
  const t = Math.min(Math.max(f(dot3(sub(p, a), ab) / denom), 0), 1);
  return dsq(sub(p, add(a, scale(ab, t))));
}

/** Mirrors distSqPointTriangle in src/mesh_to_grid.zig and wgsl/rasterize.wgsl. */
export function distSqPointTriangle(p: V3, a: V3, b: V3, c: V3): number {
  const ab = sub(b, a);
  const ac = sub(c, a);
  const ap = sub(p, a);
  const d1 = dot3(ab, ap);
  const d2 = dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) return dsq(ap); // vertex a

  const bp = sub(p, b);
  const d3 = dot3(ab, bp);
  const d4 = dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) return dsq(bp); // vertex b

  const vc = f(f(d1 * d4) - f(d3 * d2));
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const denom = f(d1 - d3);
    if (denom > 0) return dsq(sub(ap, scale(ab, f(d1 / denom)))); // edge ab
  }

  const cp = sub(p, c);
  const d5 = dot3(ab, cp);
  const d6 = dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) return dsq(cp); // vertex c

  const vb = f(f(d5 * d2) - f(d1 * d6));
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const denom = f(d2 - d6);
    if (denom > 0) return dsq(sub(ap, scale(ac, f(d2 / denom)))); // edge ac
  }

  const va = f(f(d3 * d6) - f(d5 * d4));
  if (va <= 0 && f(d4 - d3) >= 0 && f(d5 - d6) >= 0) {
    const denom = f(f(d4 - d3) + f(d5 - d6));
    if (denom > 0) return dsq(sub(bp, scale(sub(c, b), f(f(d4 - d3) / denom)))); // edge bc
  }

  const denom = f(f(va + vb) + vc);
  if (denom <= 0) {
    // Degenerate triangles fall back to edge distances.
    return Math.min(distSqPointSegment(p, a, b), distSqPointSegment(p, a, c), distSqPointSegment(p, b, c));
  }
  const inv = f(1 / denom);
  return dsq(sub(ap, add(scale(ab, f(vb * inv)), scale(ac, f(vc * inv))))); // face
}

export interface RefBin {
  pairKeys: Uint32Array;
  pairTris: Uint32Array;
  leafKeys: Uint32Array;
}

/**
 * Reference of the binning contract in rasterizeTriangle from
 * src/mesh_to_grid.zig, with f32 arithmetic throughout.
 */
export function refBin(
  points: Float32Array,
  triangles: Uint32Array,
  voxelSize: number,
  halfWidth: number,
  leafMin: [number, number, number]
): RefBin {
  const pts = transform(points, voxelSize);
  const pairs: Array<[number, number]> = [];
  for (let t = 0; t < triangles.length / 3; t++) {
    const { lo, hi } = leafRange(pts, triangles, t, halfWidth);
    for (let x = lo[0]; x <= hi[0]; x++) {
      for (let y = lo[1]; y <= hi[1]; y++) {
        for (let z = lo[2]; z <= hi[2]; z++) {
          const key = (((x - leafMin[0]) << 20) | ((y - leafMin[1]) << 10) | (z - leafMin[2])) >>> 0;
          pairs.push([key, t]);
        }
      }
    }
  }
  pairs.sort((a, b) => a[0] - b[0] || 0); // stable sort preserves triangle order per key
  const pairKeys = new Uint32Array(pairs.map((p) => p[0]));
  const pairTris = new Uint32Array(pairs.map((p) => p[1]));
  const leafKeys = new Uint32Array([...new Set(pairKeys)]);
  return { pairKeys, pairTris, leafKeys };
}

/**
 * Reference of the distance stage. Returns one slab of minimum squared
 * distances per leaf, with infinity where untouched.
 */
export function refRasterize(
  points: Float32Array,
  triangles: Uint32Array,
  voxelSize: number,
  halfWidth: number,
  leafMin: [number, number, number]
): { bin: RefBin; values: Float32Array } {
  const bin = refBin(points, triangles, voxelSize, halfWidth, leafMin);
  const pts = transform(points, voxelSize);
  const slot = new Map<number, number>();
  bin.leafKeys.forEach((key, i) => slot.set(key, i));
  const values = new Float32Array(bin.leafKeys.length * 512).fill(Infinity);
  const hw2 = f(halfWidth * halfWidth);

  for (let i = 0; i < bin.pairKeys.length; i++) {
    const t = bin.pairTris[i];
    const a = vert(pts, triangles[t * 3]);
    const b = vert(pts, triangles[t * 3 + 1]);
    const c = vert(pts, triangles[t * 3 + 2]);
    const { loV, hiV } = voxelRange(pts, triangles, t, halfWidth);
    const key = bin.pairKeys[i];
    const origin = [
      (((key >>> 20) & 0x3ff) + leafMin[0]) << 3,
      (((key >>> 10) & 0x3ff) + leafMin[1]) << 3,
      ((key & 0x3ff) + leafMin[2]) << 3,
    ];
    const base = slot.get(key)! * 512;
    for (let x = Math.max(loV[0], origin[0]); x <= Math.min(hiV[0], origin[0] + 7); x++) {
      for (let y = Math.max(loV[1], origin[1]); y <= Math.min(hiV[1], origin[1] + 7); y++) {
        for (let z = Math.max(loV[2], origin[2]); z <= Math.min(hiV[2], origin[2] + 7); z++) {
          const d2 = distSqPointTriangle([x, y, z], a, b, c);
          if (d2 <= hw2) {
            const n = base + ((x & 7) << 6) + ((y & 7) << 3) + (z & 7);
            if (d2 < values[n]) values[n] = d2;
          }
        }
      }
    }
  }
  return { bin, values };
}

/**
 * Reference of the sign stage, mirroring ColumnGrid from
 * src/mesh_to_grid.zig in f64, the same precision as the CPU. Returns one
 * inside mask per leaf in voxel bit order.
 */
export function refSign(
  points: Float32Array,
  triangles: Uint32Array,
  voxelSize: number,
  leafKeys: Uint32Array,
  leafMin: [number, number, number]
): Uint32Array {
  const pts = transform(points, voxelSize);
  const edge = (px: number, py: number, qx: number, qy: number, sx: number, sy: number): number =>
    (qx - px) * (sy - py) - (qy - py) * (sx - px);
  const accept = (w: number, ex: number, ey: number): boolean => {
    if (w > 0) return true;
    if (w < 0) return false;
    return ey < 0 || (ey === 0 && ex > 0);
  };

  const cols = new Map<string, number[]>();
  for (let t = 0; t < triangles.length / 3; t++) {
    const [ax, ay, az] = vert(pts, triangles[t * 3]);
    const [bx, by, bz] = vert(pts, triangles[t * 3 + 1]);
    const [cx, cy, cz] = vert(pts, triangles[t * 3 + 2]);
    const signedArea = edge(ax, ay, bx, by, cx, cy);
    if (signedArea === 0) continue; // vertical triangle
    const flip = signedArea < 0 ? -1 : 1;
    const area = flip * signedArea;
    const x0 = Math.ceil(Math.min(ax, bx, cx));
    const x1 = Math.floor(Math.max(ax, bx, cx));
    const y0 = Math.ceil(Math.min(ay, by, cy));
    const y1 = Math.floor(Math.max(ay, by, cy));
    for (let x = x0; x <= x1; x++) {
      for (let y = y0; y <= y1; y++) {
        const w0 = flip * edge(bx, by, cx, cy, x, y);
        const w1 = flip * edge(cx, cy, ax, ay, x, y);
        const w2 = flip * edge(ax, ay, bx, by, x, y);
        const inside =
          accept(w0, flip * (cx - bx), flip * (cy - by)) &&
          accept(w1, flip * (ax - cx), flip * (ay - cy)) &&
          accept(w2, flip * (bx - ax), flip * (by - ay));
        if (!inside) continue;
        const z = (w0 * az + w1 * bz + w2 * cz) / area;
        const key = `${x},${y}`;
        let list = cols.get(key);
        if (!list) cols.set(key, (list = []));
        list.push(z);
      }
    }
  }
  for (const list of cols.values()) list.sort((a, b) => a - b);

  const masks = new Uint32Array(leafKeys.length * 16);
  leafKeys.forEach((key, li) => {
    const ox = (((key >>> 20) & 0x3ff) + leafMin[0]) << 3;
    const oy = (((key >>> 10) & 0x3ff) + leafMin[1]) << 3;
    const oz = ((key & 0x3ff) + leafMin[2]) << 3;
    for (let c = 0; c < 64; c++) {
      const list = cols.get(`${ox + (c >> 3)},${oy + (c & 7)}`) ?? [];
      let ptr = 0;
      let count = 0;
      let bits = 0;
      for (let z = 0; z < 8; z++) {
        while (ptr < list.length && list[ptr] < oz + z) {
          ptr++;
          count++;
        }
        bits |= (count & 1) << z;
      }
      const n0 = c * 8;
      masks[li * 16 + (n0 >> 5)] |= (bits << (n0 & 31)) >>> 0;
    }
  });
  return masks;
}

function transform(points: Float32Array, voxelSize: number): Float32Array {
  const inv = f(1 / voxelSize);
  const pts = new Float32Array(points.length);
  for (let i = 0; i < points.length; i++) pts[i] = f(points[i] * inv);
  return pts;
}

function vert(pts: Float32Array, i: number): V3 {
  return [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]];
}

function voxelRange(pts: Float32Array, triangles: Uint32Array, t: number, halfWidth: number) {
  const loV = [0, 0, 0];
  const hiV = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    const a = pts[triangles[t * 3] * 3 + axis];
    const b = pts[triangles[t * 3 + 1] * 3 + axis];
    const c = pts[triangles[t * 3 + 2] * 3 + axis];
    loV[axis] = Math.ceil(f(Math.min(a, b, c) - halfWidth));
    hiV[axis] = Math.floor(f(Math.max(a, b, c) + halfWidth));
  }
  return { loV, hiV };
}

function leafRange(pts: Float32Array, triangles: Uint32Array, t: number, halfWidth: number) {
  const { loV, hiV } = voxelRange(pts, triangles, t, halfWidth);
  return { lo: loV.map((v) => v >> 3), hi: hiV.map((v) => v >> 3) };
}

/**
 * Reference of merge_csg in wgsl/merge.wgsl. Unions the leaf tables and
 * takes per voxel minima, where a grid without the leaf contributes its
 * implicit background. The band holds voxels with |v| below halfWidth.
 */
export function refCsgMerge(
  aKeys: Uint32Array,
  aValues: Float32Array,
  bKeys: Uint32Array,
  bValues: Float32Array,
  halfWidth: number
): { keys: Uint32Array; masks: Uint32Array; values: Float32Array } {
  const implicit = (keys: Uint32Array, values: Float32Array, leaf: [number, number, number], n: number): number => {
    const colBase = ((leaf[0] << 20) | (leaf[1] << 10)) >>> 0;
    let lo = 0;
    let hi = keys.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (keys[mid] < colBase) lo = mid + 1;
      else hi = mid;
    }
    if (lo >= keys.length || (keys[lo] & 0xfffffc00) >>> 0 !== colBase) return halfWidth;
    let best = lo;
    let bestD = Math.abs((keys[lo] & 0x3ff) - leaf[2]);
    for (let i = lo + 1; i < keys.length && ((keys[i] & 0xfffffc00) >>> 0) === colBase; i++) {
      const d = Math.abs((keys[i] & 0x3ff) - leaf[2]);
      if (d >= bestD) break;
      best = i;
      bestD = d;
    }
    const zloc = (keys[best] & 0x3ff) < leaf[2] ? 7 : 0;
    const facing = (n & 0x1f8) | zloc;
    return values[best * 512 + facing] < 0 ? -halfWidth : halfWidth;
  };

  const keys = new Uint32Array([...new Set([...aKeys, ...bKeys])].sort((x, y) => x - y));
  const aIdx = new Map<number, number>();
  aKeys.forEach((k, i) => aIdx.set(k, i));
  const bIdx = new Map<number, number>();
  bKeys.forEach((k, i) => bIdx.set(k, i));
  const masks = new Uint32Array(keys.length * 16);
  const values = new Float32Array(keys.length * 512);
  keys.forEach((key, i) => {
    const leaf: [number, number, number] = [(key >>> 20) & 0x3ff, (key >>> 10) & 0x3ff, key & 0x3ff];
    const ai = aIdx.get(key);
    const bi = bIdx.get(key);
    for (let n = 0; n < 512; n++) {
      const va = ai !== undefined ? aValues[ai * 512 + n] : implicit(aKeys, aValues, leaf, n);
      const vb = bi !== undefined ? bValues[bi * 512 + n] : implicit(bKeys, bValues, leaf, n);
      const v = Math.min(va, vb);
      values[i * 512 + n] = v;
      if (Math.abs(v) < halfWidth) masks[i * 16 + (n >> 5)] |= 1 << (n & 31);
    }
  });
  return { keys, masks, values };
}

/** Binary STL parser producing a triangle soup for test inputs. */
export function parseBinarySTL(bytes: Uint8Array): { points: Float32Array<ArrayBuffer>; triangles: Uint32Array<ArrayBuffer> } {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const triCount = view.getUint32(80, true);
  const points = new Float32Array(triCount * 9);
  const triangles = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const base = 84 + t * 50 + 12; // skip normal
    for (let i = 0; i < 9; i++) points[t * 9 + i] = view.getFloat32(base + i * 4, true);
    for (let v = 0; v < 3; v++) triangles[t * 3 + v] = t * 3 + v;
  }
  return { points, triangles };
}
