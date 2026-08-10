// Merges two grids. The output leaf table is the union of both sorted
// leaf tables. merge_masks ORs masks for a topology merge. merge_csg is
// the SDF union for value carrying grids and handles overlapping solids.
// Each voxel takes the min of the two values, where a grid without the
// leaf contributes its implicit background, and the band becomes the
// voxels with |v| below the half width, so band voxels swallowed by the
// other solid's interior deactivate.

struct MergeParams {
    a_count: u32,
    b_count: u32,
    concat_count: u32,
    out_count: u32,
    half_width: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: MergeParams;
@group(0) @binding(1) var<storage, read> concat_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(4) var<storage, read> a_keys: array<u32>;
@group(0) @binding(5) var<storage, read> a_data: array<u32>;
@group(0) @binding(6) var<storage, read> b_keys: array<u32>;
@group(0) @binding(7) var<storage, read> b_data: array<u32>;
@group(0) @binding(8) var<storage, read_write> out_data: array<u32>;
@group(0) @binding(9) var<storage, read_write> out_csg_masks: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;
const NOT_FOUND: u32 = 0xffffffffu;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

// flags has one extra entry so the scanned total lands in the last slot.
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

fn findA(key: u32) -> u32 {
    var lo = 0u;
    var hi = params.a_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (a_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < params.a_count && a_keys[lo] == key) {
        return lo;
    }
    return NOT_FOUND;
}

fn findB(key: u32) -> u32 {
    var lo = 0u;
    var hi = params.b_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < params.b_count && b_keys[lo] == key) {
        return lo;
    }
    return NOT_FOUND;
}

// The data bindings hold 16 word masks. The output is the union.
@compute @workgroup_size(256)
fn merge_masks(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.out_count) {
        return;
    }
    let a = findA(out_keys[i]);
    let b = findB(out_keys[i]);
    for (var w = 0u; w < 16u; w = w + 1u) {
        var m = 0u;
        if (a != NOT_FOUND) {
            m = m | a_data[(a * 16u) + w];
        }
        if (b != NOT_FOUND) {
            m = m | b_data[(b * 16u) + w];
        }
        out_masks_write(i, w, m);
    }
}

fn out_masks_write(i: u32, w: u32, m: u32) {
    out_data[(i * 16u) + w] = m;
}

// Implicit background of grid A at a voxel. Takes the facing voxel sign
// of the nearest leaf in the column and reads as outside when the column
// is empty.
fn implicitA(leaf: vec3<i32>, n: u32) -> f32 {
    let col_base = (u32(leaf.x) << 20u) | (u32(leaf.y) << 10u);
    var lo = 0u;
    var hi = params.a_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (a_keys[mid] < col_base) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= params.a_count || (a_keys[lo] & 0xfffffc00u) != col_base) {
        return params.half_width;
    }
    var best = lo;
    var best_d = abs(i32(a_keys[lo] & 0x3ffu) - leaf.z);
    var i = lo + 1u;
    while (i < params.a_count && (a_keys[i] & 0xfffffc00u) == col_base) {
        let d = abs(i32(a_keys[i] & 0x3ffu) - leaf.z);
        if (d >= best_d) {
            break;
        }
        best = i;
        best_d = d;
        i = i + 1u;
    }
    let zloc = select(0u, 7u, i32(a_keys[best] & 0x3ffu) < leaf.z);
    let facing = (n & 0x1f8u) | zloc; // keeps local x and y, swaps z
    return select(params.half_width, -params.half_width, bitcast<f32>(a_data[(best * 512u) + facing]) < 0.0);
}

fn implicitB(leaf: vec3<i32>, n: u32) -> f32 {
    let col_base = (u32(leaf.x) << 20u) | (u32(leaf.y) << 10u);
    var lo = 0u;
    var hi = params.b_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b_keys[mid] < col_base) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= params.b_count || (b_keys[lo] & 0xfffffc00u) != col_base) {
        return params.half_width;
    }
    var best = lo;
    var best_d = abs(i32(b_keys[lo] & 0x3ffu) - leaf.z);
    var i = lo + 1u;
    while (i < params.b_count && (b_keys[i] & 0xfffffc00u) == col_base) {
        let d = abs(i32(b_keys[i] & 0x3ffu) - leaf.z);
        if (d >= best_d) {
            break;
        }
        best = i;
        best_d = d;
        i = i + 1u;
    }
    let zloc = select(0u, 7u, i32(b_keys[best] & 0x3ffu) < leaf.z);
    let facing = (n & 0x1f8u) | zloc;
    return select(params.half_width, -params.half_width, bitcast<f32>(b_data[(best * 512u) + facing]) < 0.0);
}

// SDF union over 512 value f32 slabs. out_csg_masks receives the band.
@compute @workgroup_size(256)
fn merge_csg(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.out_count) {
        return;
    }
    let key = out_keys[i];
    let leaf = vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu));
    let a = findA(key);
    let b = findB(key);
    var mask = 0u;
    for (var n = 0u; n < 512u; n = n + 1u) {
        var va: f32;
        if (a != NOT_FOUND) {
            va = bitcast<f32>(a_data[(a * 512u) + n]);
        } else {
            va = implicitA(leaf, n);
        }
        var vb: f32;
        if (b != NOT_FOUND) {
            vb = bitcast<f32>(b_data[(b * 512u) + n]);
        } else {
            vb = implicitB(leaf, n);
        }
        let v = min(va, vb);
        out_data[(i * 512u) + n] = bitcast<u32>(v);
        if (abs(v) < params.half_width) {
            mask = mask | (1u << (n & 31u));
        }
        if ((n & 31u) == 31u) {
            out_csg_masks[(i * 16u) + (n >> 5u)] = mask;
            mask = 0u;
        }
    }
}
