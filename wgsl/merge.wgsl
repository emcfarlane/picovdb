// Merges two grids over the union of their leaf tables.
//
// merge_masks ORs 16 word masks, for topology. csg_mark and csg_apply
// combine two op layer grids, bound as a_* and b_*: per voxel min for a
// union, max for an intersection, max(a, -b) for a subtraction. A grid
// without the leaf contributes its implicit background, so band voxels
// inside the other solid deactivate.

struct MergeParams {
    a_count: u32,
    b_count: u32,
    concat_count: u32,
    out_count: u32,
    half_width: f32,
    op: u32, // 0 union, 1 intersect, 2 subtract
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: MergeParams;
@group(0) @binding(1) var<storage, read> concat_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(4) var<storage, read> a_keys: array<u32>;
@group(0) @binding(5) var<storage, read> a_masks: array<u32>;
@group(0) @binding(6) var<storage, read> b_keys: array<u32>;
@group(0) @binding(7) var<storage, read> b_masks: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_masks: array<u32>;
@group(0) @binding(10) var<storage, read> a_leaves: array<u32>;
@group(0) @binding(11) var<storage, read> a_data: array<u32>;
@group(0) @binding(12) var<storage, read> b_leaves: array<u32>;
@group(0) @binding(13) var<storage, read> b_data: array<u32>;

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

@compute @workgroup_size(256)
fn compact_unique(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.concat_count) {
        return;
    }
    if (i == 0u || concat_keys[i] != concat_keys[i - 1u]) {
        out_keys[flags[i]] = concat_keys[i];
    }
}

// Topology merge: ORs the 16 mask words per leaf.
@compute @workgroup_size(256)
fn merge_masks(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.out_count) {
        return;
    }
    let a = a_find(out_keys[i]);
    let b = b_find(out_keys[i]);
    for (var w = 0u; w < 16u; w = w + 1u) {
        var m = 0u;
        if (a != NOT_FOUND) {
            m = m | a_masks[(a * 16u) + w];
        }
        if (b != NOT_FOUND) {
            m = m | b_masks[(b * 16u) + w];
        }
        out_masks[(i * 16u) + w] = m;
    }
}

// Boolean of voxel n of leaf `leaf`. a and b are the leaf's indices in
// the two grids, or NOT_FOUND.
fn csgValue(leaf: vec3<i32>, a: u32, b: u32, n: u32) -> f32 {
    let p = (leaf << vec3<u32>(3u)) + voxelLocal(n);
    var va: f32;
    if (a != NOT_FOUND) {
        va = a_leafValue(a, n);
    } else {
        va = a_valueAt(p);
    }
    var vb: f32;
    if (b != NOT_FOUND) {
        vb = b_leafValue(b, n);
    } else {
        vb = b_valueAt(p);
    }
    if (params.op == 0u) {
        return min(va, vb);
    }
    if (params.op == 1u) {
        return max(va, vb);
    }
    return max(va, -vb);
}

@compute @workgroup_size(256)
fn csg_mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (markSentinel(i, params.out_count)) {
        return;
    }
    let key = out_keys[i];
    let leaf = unpack(key);
    let a = a_find(key);
    let b = b_find(key);
    var acc: LeafAcc;
    for (var n = 0u; n < 512u; n = n + 1u) {
        accPush(&acc, i, n, csgValue(leaf, a, b, n));
    }
    accFinish(&acc, i);
}

@compute @workgroup_size(256)
fn csg_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= arrayLength(&w_keys)) {
        return;
    }
    let key = w_keys[j];
    let leaf = unpack(key);
    let a = a_find(key);
    let b = b_find(key);
    let base = w_leaves[(j * LEAF_U32) + 2u];
    var k = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = bandWord(j, w);
        for (var b_ = 0u; b_ < 32u; b_ = b_ + 1u) {
            if (((band >> b_) & 1u) == 0u) {
                continue;
            }
            w_data[base + k] = bitcast<u32>(csgValue(leaf, a, b, (w * 32u) + b_));
            k = k + 1u;
        }
    }
}
