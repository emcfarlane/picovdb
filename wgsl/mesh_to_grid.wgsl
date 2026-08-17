// Bins triangles into the 8^3 leaf blocks their dilated bounding boxes
// touch.
//
// The bounds match rasterizeTriangle in src/mesh_to_grid.zig. A triangle
// covers the voxels inside its index space bounds dilated by the half
// width and touches the leaves containing them. count_pairs counts leaves
// per triangle, the host scans the counts into write offsets, emit_pairs
// writes (leaf key, triangle) pairs, radix sort orders them by key, and
// mark_unique plus compact_unique build the deduplicated leaf table.
//
// A leaf key packs the leaf coordinate relative to leaf_min with 10 bits
// per axis, so grids may span up to 1024 leaves per axis. The host
// validates the range.

struct BinParams {
    point_count: u32,
    triangle_count: u32,
    inv_voxel_size: f32,
    half_width: f32,
    leaf_min: vec3<i32>,
    pair_count: u32,
}

@group(0) @binding(0) var<uniform> params: BinParams;
@group(0) @binding(1) var<storage, read> points_world: array<f32>;
@group(0) @binding(2) var<storage, read_write> points_index: array<f32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> pair_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> pair_tris: array<u32>;
@group(0) @binding(7) var<storage, read_write> flags: array<u32>;
@group(0) @binding(8) var<storage, read_write> unique_keys: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

@compute @workgroup_size(256)
fn transform_points(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i < (params.point_count * 3u)) {
        points_index[i] = points_world[i] * params.inv_voxel_size;
    }
}

fn loadPoint(i: u32) -> vec3<f32> {
    return vec3<f32>(points_index[i * 3u], points_index[(i * 3u) + 1u], points_index[(i * 3u) + 2u]);
}

struct LeafRange {
    lo: vec3<i32>,
    hi: vec3<i32>,
}

fn leafRange(t: u32) -> LeafRange {
    let a = loadPoint(triangles[t * 3u]);
    let b = loadPoint(triangles[(t * 3u) + 1u]);
    let c = loadPoint(triangles[(t * 3u) + 2u]);
    let hw = vec3<f32>(params.half_width);
    let lo = vec3<i32>(ceil(min(a, min(b, c)) - hw));
    let hi = vec3<i32>(floor(max(a, max(b, c)) + hw));
    return LeafRange(lo >> vec3<u32>(3u), hi >> vec3<u32>(3u));
}

// counts has one extra entry so the scanned total lands in the last slot.
@compute @workgroup_size(256)
fn count_pairs(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = globalIndex(wid, lid);
    if (t > params.triangle_count) {
        return;
    }
    if (t == params.triangle_count) {
        counts[t] = 0u;
        return;
    }
    let r = leafRange(t);
    let span = max((r.hi - r.lo) + vec3<i32>(1), vec3<i32>(0));
    counts[t] = u32(span.x * span.y * span.z);
}

// counts now holds the scanned write offsets.
@compute @workgroup_size(256)
fn emit_pairs(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = globalIndex(wid, lid);
    if (t >= params.triangle_count) {
        return;
    }
    let r = leafRange(t);
    var w = counts[t];
    for (var x = r.lo.x; x <= r.hi.x; x = x + 1) {
        for (var y = r.lo.y; y <= r.hi.y; y = y + 1) {
            for (var z = r.lo.z; z <= r.hi.z; z = z + 1) {
                let rel = vec3<u32>(vec3<i32>(x, y, z) - params.leaf_min);
                pair_keys[w] = (rel.x << 20u) | (rel.y << 10u) | rel.z;
                pair_tris[w] = t;
                w = w + 1u;
            }
        }
    }
}

// flags has one extra entry so the scanned total lands in the last slot.
@compute @workgroup_size(256)
fn mark_unique(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.pair_count) {
        return;
    }
    if (i == params.pair_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || pair_keys[i] != pair_keys[i - 1u]);
}

// flags now holds the scanned unique positions.
@compute @workgroup_size(256)
fn compact_unique(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.pair_count) {
        return;
    }
    if (i == 0u || pair_keys[i] != pair_keys[i - 1u]) {
        unique_keys[flags[i]] = pair_keys[i];
    }
}
