// Remaps a grid into a new grid by a per voxel rule. Two modes: an SDF
// offset, and a translation by whole voxels. Only candidate leaves that
// end up with band voxels are kept.
//
// The old grid binds as old_keys, old_leaves, old_data. The host supplies
// the candidate leaves. For a translation they are the old leaves and
// their neighbors, deduped here. For an offset they are the leaves of the
// distance table.
//
// An offset uses only the old zero level set. The host extracts it
// (extract.wgsl) and rasterizes exact distances to its triangles out to
// half_width + |amount| (mesh_to_grid.wgsl, rasterize.wgsl). The new value
// is that distance, signed by the old grid, minus the amount. Stored
// values away from the surface are not used: files and earlier ops may
// carry them with less accuracy. The error is the marching cubes chord.
// The host splits large offsets into steps to bound the reach.

struct RemapParams {
    old_count: u32,
    concat_count: u32,
    new_count: u32,
    mode: u32, // 0 offset, 1 translate
    shift: vec3<i32>, // translate, voxels
    amount: f32, // offset, voxels
    nb_lo: vec3<i32>, // neighborhood box, leaf units
    half_width: f32,
    nb_dims: vec3<i32>,
    dist_count: u32, // offset, leaves in the distance table
    delta: vec3<i32>, // rebase, leaf units
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: RemapParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_leaves: array<u32>;
@group(0) @binding(3) var<storage, read_write> concat_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> new_keys: array<u32>;
@group(0) @binding(6) var<storage, read> old_data: array<u32>;
// Offset only. new_keys is the sorted leaf table of the distances,
// dist_count long. dist_values holds their squared distances as f32 bits,
// INF_BITS beyond the reach.
@group(0) @binding(9) var<storage, read> dist_values: array<u32>;

const INF_BITS: u32 = 0x7f800000u;

// Shifts every key by the same delta, so the order holds.
@compute @workgroup_size(256)
fn rebase(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.old_count) {
        return;
    }
    new_keys[i] = pack(unpack(old_keys[i]) + params.delta);
}

// One candidate per old leaf and neighborhood cell, after the old keys.
// Cells outside the key range repeat the leaf's own key.
@compute @workgroup_size(256)
fn generate_candidates(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    let vol = u32(params.nb_dims.x * params.nb_dims.y * params.nb_dims.z);
    if (i >= params.old_count * vol) {
        return;
    }
    let leaf = unpack(old_keys[i / vol]);
    let j = i % vol;
    let dy = u32(params.nb_dims.y);
    let dz = u32(params.nb_dims.z);
    let c = leaf + params.nb_lo + vec3<i32>(i32(j / (dy * dz)), i32((j / dz) % dy), i32(j % dz));
    var key = old_keys[i / vol];
    if (all(c >= vec3<i32>(0)) && all(c <= vec3<i32>(1023))) {
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

// Squared distance to the old surface at a relative voxel coordinate, as
// f32 bits. INF_BITS beyond the reach.
fn distSqBitsAt(ijk: vec3<i32>) -> u32 {
    let leaf = ijk >> vec3<u32>(3u);
    if (any(leaf < vec3<i32>(0)) || any(leaf > vec3<i32>(1023))) {
        return INF_BITS;
    }
    let key = pack(leaf);
    var lo = 0u;
    var hi = params.dist_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (new_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < params.dist_count && new_keys[lo] == key) {
        return dist_values[(lo * 512u) + voxelOffset(ijk)];
    }
    return INF_BITS;
}

// New value at a relative voxel coordinate.
fn newValueAt(p: vec3<i32>) -> f32 {
    let hw = params.half_width;
    var v: f32;
    if (params.mode == 0u) {
        let s = select(1.0, -1.0, old_valueAt(p) < 0.0);
        let d2 = distSqBitsAt(p);
        if (d2 == INF_BITS) {
            v = s * hw;
        } else {
            v = (s * sqrt(bitcast<f32>(d2))) - params.amount;
        }
    } else {
        v = old_valueAt(p - params.shift);
    }
    return clamp(v, -hw, hw);
}

// The writer passes. See the writer in opgrid.ts.
@compute @workgroup_size(256)
fn mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (markSentinel(i, params.new_count)) {
        return;
    }
    let origin = unpack(new_keys[i]) << vec3<u32>(3u);
    var acc: LeafAcc;
    for (var n = 0u; n < 512u; n = n + 1u) {
        accPush(&acc, i, n, newValueAt(origin + voxelLocal(n)));
    }
    accFinish(&acc, i);
}

@compute @workgroup_size(256)
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= arrayLength(&w_keys)) {
        return;
    }
    let origin = unpack(w_keys[j]) << vec3<u32>(3u);
    let base = w_leaves[(j * LEAF_U32) + 2u];
    var k = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = bandWord(j, w);
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 0u) {
                continue;
            }
            w_data[base + k] = bitcast<u32>(newValueAt(origin + voxelLocal((w * 32u) + b)));
            k = k + 1u;
        }
    }
}
