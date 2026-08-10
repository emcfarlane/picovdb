// Computes inside or outside parity per voxel from vertical ray
// crossings, mirroring ColumnGrid in src/mesh_to_grid.zig.
//
// Each triangle records a surface crossing height for every lattice column
// its XY projection covers. The tie break matches the CPU, so an edge
// shared by two triangles counts exactly once and vertical triangles
// contribute nothing. The host scans the counts, sorts the crossings by
// column then height with two stable radix passes, and sign_leaves walks
// each candidate leaf column counting crossings below each voxel center.
// An odd count means inside.
//
// The CPU computes crossings in f64. WGSL has no f64, so this runs in f32.
// All triangles evaluate identical expressions, which keeps the parity
// consistent. Only voxels whose column crossing sits within f32 noise of
// their center plane can differ from the CPU, and those lie on the
// surface.

struct SignParams {
    triangle_count: u32,
    crossing_count: u32,
    min_x: i32,
    min_y: i32,
    nx: u32,
    ny: u32,
    leaf_count: u32,
    pad: u32,
    leaf_min: vec3<i32>,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SignParams;
@group(0) @binding(1) var<storage, read> points_index: array<f32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> cross_cols: array<u32>;
@group(0) @binding(5) var<storage, read_write> cross_z: array<u32>;
@group(0) @binding(6) var<storage, read> leaf_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> inside: array<atomic<u32>>;

const DISPATCH_STRIDE: u32 = 65535u;

fn loadPoint(i: u32) -> vec3<f32> {
    return vec3<f32>(points_index[i * 3u], points_index[(i * 3u) + 1u], points_index[(i * 3u) + 2u]);
}

fn edgeFn(px: f32, py: f32, qx: f32, qy: f32, sx: f32, sy: f32) -> f32 {
    return ((qx - px) * (sy - py)) - ((qy - py) * (sx - px));
}

fn accept(w: f32, ex: f32, ey: f32) -> bool {
    if (w > 0.0) {
        return true;
    }
    if (w < 0.0) {
        return false;
    }
    return ey < 0.0 || (ey == 0.0 && ex > 0.0);
}

// Monotone f32 to u32 transform so crossing order survives the u32 sort.
fn sortableFromF32(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b >> 31u) == 1u) {
        return ~b;
    }
    return b | 0x80000000u;
}

// Shared by the count and emit passes, which must agree.
fn binTriangle(t: u32, emit: bool, offset: u32) -> u32 {
    let a = loadPoint(triangles[t * 3u]);
    let b = loadPoint(triangles[(t * 3u) + 1u]);
    let c = loadPoint(triangles[(t * 3u) + 2u]);

    let signed_area = edgeFn(a.x, a.y, b.x, b.y, c.x, c.y);
    if (signed_area == 0.0) {
        return 0u; // vertical triangle
    }
    var flip = 1.0;
    if (signed_area < 0.0) {
        flip = -1.0;
    }
    let area = flip * signed_area;

    let x0 = i32(ceil(min(a.x, min(b.x, c.x))));
    let x1 = i32(floor(max(a.x, max(b.x, c.x))));
    let y0 = i32(ceil(min(a.y, min(b.y, c.y))));
    let y1 = i32(floor(max(a.y, max(b.y, c.y))));

    var w = offset;
    var n = 0u;
    for (var x = x0; x <= x1; x = x + 1) {
        for (var y = y0; y <= y1; y = y + 1) {
            let px = f32(x);
            let py = f32(y);
            let w0 = flip * edgeFn(b.x, b.y, c.x, c.y, px, py);
            let w1 = flip * edgeFn(c.x, c.y, a.x, a.y, px, py);
            let w2 = flip * edgeFn(a.x, a.y, b.x, b.y, px, py);
            let ins = accept(w0, flip * (c.x - b.x), flip * (c.y - b.y))
                && accept(w1, flip * (a.x - c.x), flip * (a.y - c.y))
                && accept(w2, flip * (b.x - a.x), flip * (b.y - a.y));
            if (!ins) {
                continue;
            }
            if (x < params.min_x || y < params.min_y) {
                continue;
            }
            let ix = u32(x - params.min_x);
            let iy = u32(y - params.min_y);
            if (ix >= params.nx || iy >= params.ny) {
                continue;
            }
            if (emit) {
                let z = (((w0 * a.z) + (w1 * b.z)) + (w2 * c.z)) / area;
                cross_cols[w] = (ix * params.ny) + iy;
                cross_z[w] = sortableFromF32(z);
                w = w + 1u;
            }
            n = n + 1u;
        }
    }
    return n;
}

// counts has one extra entry so the scanned total lands in the last slot.
@compute @workgroup_size(256)
fn count_crossings(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t > params.triangle_count) {
        return;
    }
    if (t == params.triangle_count) {
        counts[t] = 0u;
        return;
    }
    counts[t] = binTriangle(t, false, 0u);
}

// counts now holds the scanned write offsets.
@compute @workgroup_size(256)
fn emit_crossings(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t < params.triangle_count) {
        let unused = binTriangle(t, true, counts[t]);
    }
}

// One workgroup per candidate leaf and one thread per column. Crossings
// arrive sorted by column then height. Each thread ORs the parity bits of
// its column's 8 voxels into the zero initialized mask.
@compute @workgroup_size(64)
fn sign_leaves(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let leaf_i = (wid.y * DISPATCH_STRIDE) + wid.x;
    if (leaf_i >= params.leaf_count) {
        return;
    }
    let key = leaf_keys[leaf_i];
    let leaf = vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu)) + params.leaf_min;
    let origin = leaf << vec3<u32>(3u);

    let x = origin.x + i32(lid.x >> 3u);
    let y = origin.y + i32(lid.x & 7u);
    let col = (u32(x - params.min_x) * params.ny) + u32(y - params.min_y);

    // Lower bound of this column's crossing run.
    var lo = 0u;
    var hi = params.crossing_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (cross_cols[mid] < col) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    var i = lo;
    var count = 0u;
    var bits = 0u;
    for (var z = 0u; z < 8u; z = z + 1u) {
        let voxel_z = sortableFromF32(f32(origin.z + i32(z)));
        while (i < params.crossing_count && cross_cols[i] == col && cross_z[i] < voxel_z) {
            i = i + 1u;
            count = count + 1u;
        }
        bits = bits | ((count & 1u) << z);
    }
    // This column's 8 voxels form one byte of the leaf mask.
    let n0 = lid.x << 3u;
    atomicOr(&inside[(leaf_i * 16u) + (n0 >> 5u)], bits << (n0 & 31u));
}
