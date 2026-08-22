// Stamps a shape into a grid. The shape is a WGSL signed distance
// function, sdf(p), in absolute voxels; the host generates it from the
// shape library and the shape's name. Add takes the min with the shape's
// distance. Carve takes the max with its negation. Stamping an empty grid
// makes the shape.
//
// The old grid binds as old_keys, old_leaves, old_data. args holds the
// shape's arguments. The candidates are the old leaves plus the leaves
// the shape's shell crosses, deduped here. Voxels read the old leaf where
// one exists and the implicit background elsewhere, so a carve through a
// leafless interior forms a correct band.

struct StampParams {
    old_count: u32,
    concat_count: u32,
    new_count: u32,
    mode: u32, // 0 adds material and 1 carves
    origin: vec3<f32>, // voxel origin of the key space, absolute voxels
    half_width: f32,
    box_lo: vec3<i32>, // candidate box, relative leaf coords
    pad0: u32,
    box_dims: vec3<i32>,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: StampParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_leaves: array<u32>;
@group(0) @binding(3) var<storage, read_write> concat_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> new_keys: array<u32>;
@group(0) @binding(6) var<storage, read> old_data: array<u32>;
@group(0) @binding(7) var<uniform> args: array<vec4<f32>, 8>; // the shape's arguments

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
    if (abs(shapeDistance(center)) < params.half_width + 6.1) { // 3.5 * sqrt(3) to any voxel of the leaf
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

// Shape distance at a relative voxel position.
fn shapeDistance(p: vec3<f32>) -> f32 {
    return sdf(p + params.origin);
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
    let d = shapeDistance(vec3<f32>(p));
    if (params.mode == 0u) {
        v = min(v, d);
    } else {
        v = max(v, -d);
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
