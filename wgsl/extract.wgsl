// Extracts a level set of an op layer grid as triangles, with marching
// cubes. Used to redistance after an offset, and to export meshes.
//
// The grid binds as old_keys, old_leaves, old_data. params.iso is the
// level and must lie inside the band. Run count, scan counts on the host,
// then run emit. The output is three points per triangle in the grid's
// relative voxel coordinates, plus triangle indices.

struct ExtractParams {
    leaf_count: u32,
    half_width: f32,
    iso: f32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ExtractParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_leaves: array<u32>;
@group(0) @binding(3) var<storage, read> tri_table: array<i32>; // 256 x 16 edge triples, then 256 counts
@group(0) @binding(4) var<storage, read_write> counts: array<u32>; // per leaf, one extra for the scan total
@group(0) @binding(5) var<storage, read_write> out_points: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_tris: array<u32>;
@group(0) @binding(7) var<storage, read> old_data: array<u32>;

const CELLS: u32 = 729u; // 9^3 cell bases in the block

// One workgroup per leaf. The block caches the leaf's voxels and a one
// voxel margin around them.
var<workgroup> block: array<f32, 1000>; // voxel origin - 1 first
var<workgroup> wg_count: atomic<u32>;

fn hasLeaf(leaf: vec3<i32>) -> bool {
    if (any(leaf < vec3<i32>(0)) || any(leaf > vec3<i32>(1023))) {
        return false;
    }
    return old_find(pack(leaf)) != NOT_FOUND;
}

// Fills the block for leaf l and returns the leaf's voxel origin.
fn loadBlock(l: u32, lid: u32) -> vec3<i32> {
    let leaf = unpack(old_keys[l]);
    let origin = leaf << vec3<u32>(3u);
    for (var b = lid; b < 1000u; b = b + 256u) {
        let c = vec3<i32>(i32(b / 100u), i32((b / 10u) % 10u), i32(b % 10u));
        if (all(c >= vec3<i32>(1)) && all(c <= vec3<i32>(8))) {
            block[b] = old_leafValue(l, voxelOffset(c - vec3<i32>(1))) - params.iso;
        } else {
            block[b] = old_valueAt(origin - vec3<i32>(1) + c) - params.iso;
        }
    }
    return origin;
}

const CORNER = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0), vec3<i32>(1, 0, 0), vec3<i32>(1, 1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, 1), vec3<i32>(1, 0, 1), vec3<i32>(1, 1, 1), vec3<i32>(0, 1, 1),
);
const EDGE_A = array<u32, 12>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u);
const EDGE_B = array<u32, 12>(1u, 2u, 3u, 0u, 5u, 6u, 7u, 4u, 4u, 5u, 6u, 7u);

fn blockAt(c: vec3<i32>) -> f32 {
    return block[(u32(c.x) * 100u) + (u32(c.y) * 10u) + u32(c.z)];
}

// Cube index of the cell with block base b, or -1 when another leaf owns
// the cell. A cell belongs to the leaf that holds its base voxel, so each
// cell is emitted once. When that leaf is missing, this leaf takes it.
fn cellIndex(leaf: vec3<i32>, b: vec3<i32>) -> i32 {
    if (any(b == vec3<i32>(0))) {
        let owner = leaf - vec3<i32>(select(0, 1, b.x == 0), select(0, 1, b.y == 0), select(0, 1, b.z == 0));
        if (hasLeaf(owner)) {
            return -1;
        }
    }
    var index = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (blockAt(b + CORNER[i]) < 0.0) {
            index = index | (1u << i);
        }
    }
    return i32(index);
}

@compute @workgroup_size(256)
fn count(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let l = (wid.y * DISPATCH_STRIDE) + wid.x;
    if (l >= params.leaf_count) {
        return;
    }
    if (lid.x == 0u) {
        atomicStore(&wg_count, 0u);
    }
    let origin = loadBlock(l, lid.x);
    workgroupBarrier();
    let leaf = origin >> vec3<u32>(3u);
    var n = 0u;
    for (var c = lid.x; c < CELLS; c = c + 256u) {
        let b = vec3<i32>(i32(c / 81u), i32((c / 9u) % 9u), i32(c % 9u));
        let index = cellIndex(leaf, b);
        if (index > 0 && index < 255) {
            n = n + u32(tri_table[4096 + index]);
        }
    }
    atomicAdd(&wg_count, n);
    workgroupBarrier();
    if (lid.x == 0u) {
        counts[l] = atomicLoad(&wg_count);
    }
}

fn writeVertex(t: u32, v: u32, p: vec3<f32>) {
    let o = ((t * 3u) + v) * 3u;
    out_points[o] = p.x;
    out_points[o + 1u] = p.y;
    out_points[o + 2u] = p.z;
    out_tris[(t * 3u) + v] = (t * 3u) + v;
}

// counts holds the scanned triangle offsets.
@compute @workgroup_size(256)
fn emit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let l = (wid.y * DISPATCH_STRIDE) + wid.x;
    if (l >= params.leaf_count) {
        return;
    }
    if (lid.x == 0u) {
        atomicStore(&wg_count, 0u);
    }
    let origin = loadBlock(l, lid.x);
    workgroupBarrier();
    let leaf = origin >> vec3<u32>(3u);
    let base = counts[l];
    for (var c = lid.x; c < CELLS; c = c + 256u) {
        let b = vec3<i32>(i32(c / 81u), i32((c / 9u) % 9u), i32(c % 9u));
        let index = cellIndex(leaf, b);
        if (index <= 0 || index >= 255) {
            continue;
        }
        let ntri = u32(tri_table[4096 + index]);
        var t = base + atomicAdd(&wg_count, ntri);
        let cell = origin - vec3<i32>(1) + b; // base voxel, relative coords
        for (var k = 0u; k < ntri; k = k + 1u) {
            for (var v = 0u; v < 3u; v = v + 1u) {
                let e = u32(tri_table[(u32(index) * 16u) + (k * 3u) + v]);
                let ca = CORNER[EDGE_A[e]];
                let cb = CORNER[EDGE_B[e]];
                let va = blockAt(b + ca);
                let vb = blockAt(b + cb);
                let s = clamp(va / (va - vb), 0.0, 1.0);
                writeVertex(t, v, vec3<f32>(cell + ca) + (vec3<f32>(cb - ca) * s));
            }
            t = t + 1u;
        }
    }
}
