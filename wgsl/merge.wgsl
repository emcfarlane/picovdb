// Grid op: merge two grids. The output leaf table is the union of both
// sorted leaf tables (host concatenates, sorts, and scans between the mark
// and compact passes); masks OR together and, when value slabs are given,
// values combine with min — the SDF union.

struct MergeParams {
    a_count: u32,
    b_count: u32,
    concat_count: u32,
    out_count: u32,
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

const DISPATCH_STRIDE: u32 = 65535u;
const NOT_FOUND: u32 = 0xffffffffu;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

// flags has concat_count + 1 entries; the trailing 0 makes the exclusive
// scan's last element the unique-leaf count.
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

// a_data/b_data/out_data are 16-word masks: union.
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

// a_data/b_data/out_data are 512-value f32 slabs: min (SDF union); a leaf
// present in only one grid copies through.
@compute @workgroup_size(256)
fn merge_values(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.out_count) {
        return;
    }
    let a = findA(out_keys[i]);
    let b = findB(out_keys[i]);
    for (var n = 0u; n < 512u; n = n + 1u) {
        var v: f32;
        if (a != NOT_FOUND && b != NOT_FOUND) {
            v = min(bitcast<f32>(a_data[(a * 512u) + n]), bitcast<f32>(b_data[(b * 512u) + n]));
        } else if (a != NOT_FOUND) {
            v = bitcast<f32>(a_data[(a * 512u) + n]);
        } else {
            v = bitcast<f32>(b_data[(b * 512u) + n]);
        }
        out_data[(i * 512u) + n] = bitcast<u32>(v);
    }
}
