// Stamps an analytic shape into a grid: a sphere, rounded box, capsule,
// or capped cylinder. Add takes the min with the shape's distance. Carve
// takes the max with its negation. Stamping an empty grid makes the shape.
//
// The old grid binds as old_keys, old_leaves, old_data. The candidates
// are the old leaves plus the leaves the shape's shell crosses, deduped
// here. Voxels read the old leaf where one exists and the implicit
// background elsewhere, so a carve through a leafless interior forms a
// correct band.

struct StampParams {
    old_count: u32,
    concat_count: u32,
    new_count: u32,
    mode: u32, // 0 adds material and 1 carves
    p0: vec3<f32>, // relative voxel space: center, or segment start
    radius: f32,
    p1: vec3<f32>, // box half extents, or segment end
    half_width: f32,
    box_lo: vec3<i32>, // candidate box, relative leaf coords
    shape: u32, // 0 sphere, 1 box, 2 capsule, 3 cylinder
    box_dims: vec3<i32>,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: StampParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_leaves: array<u32>;
@group(0) @binding(3) var<storage, read_write> concat_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> new_keys: array<u32>;
@group(0) @binding(6) var<storage, read> old_data: array<u32>;

// Candidate leaves go after the old keys in concat_keys. Only leaves the
// shell crosses can gain band voxels. A leaf fully inside or outside the
// shape keeps its old band, and old leaves are candidates already. The
// rest repeat the box corner key, which the dedupe drops.
@compute @workgroup_size(256)
fn generate_candidates(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    let vol = u32(params.box_dims.x * params.box_dims.y * params.box_dims.z);
    if (i >= vol) {
        return;
    }
    let dy = u32(params.box_dims.y);
    let dz = u32(params.box_dims.z);
    let c = params.box_lo + vec3<i32>(i32(i / (dy * dz)), i32((i / dz) % dy), i32(i % dz));
    let center = vec3<f32>(c << vec3<u32>(3u)) + vec3<f32>(3.5);
    var key = pack(params.box_lo);
    if (abs(brushDistance(center)) < params.half_width + 6.1) { // 3.5 * sqrt(3) to any voxel of the leaf
        key = pack(c);
    }
    concat_keys[params.old_count + i] = key;
}

// flags has one extra entry for the scan total.
@compute @workgroup_size(256)
fn mark_unique(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.concat_count) {
        return;
    }
    if (i == params.concat_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || concat_keys[i] != concat_keys[i - 1u]);
}

// flags now holds the scanned unique positions.
@compute @workgroup_size(256)
fn compact_unique(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.concat_count) {
        return;
    }
    if (i == 0u || concat_keys[i] != concat_keys[i - 1u]) {
        new_keys[flags[i]] = concat_keys[i];
    }
}

fn brushDistance(p: vec3<f32>) -> f32 {
    if (params.shape == 0u) {
        return length(p - params.p0) - params.radius;
    }
    if (params.shape == 1u) {
        // Box with half extents p1, edges rounded by radius.
        let q = abs(p - params.p0) - params.p1;
        return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - params.radius;
    }
    let ba = params.p1 - params.p0;
    let pa = p - params.p0;
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);
    if (params.shape == 2u) {
        let h = clamp(paba / baba, 0.0, 1.0);
        return length(pa - (ba * h)) - params.radius;
    }
    // Capped cylinder between p0 and p1.
    let x = length((pa * baba) - (ba * paba)) - (params.radius * baba);
    let y = abs(paba - (baba * 0.5)) - (baba * 0.5);
    let x2 = x * x;
    let y2 = y * y * baba;
    var d: f32;
    if (max(x, y) < 0.0) {
        d = -min(x2, y2);
    } else {
        d = select(0.0, x2, x > 0.0) + select(0.0, y2, y > 0.0);
    }
    return sign(d) * sqrt(abs(d)) / baba;
}

// New value of voxel n of leaf `leaf`. old is the leaf's index in the old
// grid, or NOT_FOUND.
fn newValue(leaf: vec3<i32>, old: u32, n: u32) -> f32 {
    let p = (leaf << vec3<u32>(3u)) + voxelLocal(n);
    var v: f32;
    if (old != NOT_FOUND) {
        v = old_leafValue(old, n);
    } else {
        v = old_valueAt(p);
    }
    // Leaves outside the candidate box cannot change.
    let box_hi = params.box_lo + params.box_dims - vec3<i32>(1);
    if (any(leaf < params.box_lo) || any(leaf > box_hi)) {
        return v;
    }
    let brush = brushDistance(vec3<f32>(p));
    if (params.mode == 0u) {
        v = min(v, brush);
    } else {
        v = max(v, -brush);
    }
    return clamp(v, -params.half_width, params.half_width);
}

@compute @workgroup_size(256)
fn mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (markSentinel(i, params.new_count)) {
        return;
    }
    let key = new_keys[i];
    let leaf = unpack(key);
    let old = old_find(key);
    var acc: LeafAcc;
    for (var n = 0u; n < 512u; n = n + 1u) {
        accPush(&acc, i, n, newValue(leaf, old, n));
    }
    accFinish(&acc, i);
}

@compute @workgroup_size(256)
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= arrayLength(&w_keys)) {
        return;
    }
    let key = w_keys[j];
    let leaf = unpack(key);
    let old = old_find(key);
    let base = w_leaves[(j * LEAF_U32) + 2u];
    var k = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = bandWord(j, w);
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 0u) {
                continue;
            }
            w_data[base + k] = bitcast<u32>(newValue(leaf, old, (w * 32u) + b));
            k = k + 1u;
        }
    }
}
