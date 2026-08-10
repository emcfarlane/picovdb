// Brush stamp. Composes an analytic sphere SDF into a grid. A clamped min
// adds material and a clamped max against the negated brush carves.
//
// generate_candidates emits every leaf in the stamp's dilated bounding
// box. The host concatenates them with the existing leaf table, sorts,
// and dedupes. apply writes each output leaf's values from the existing
// slab where the leaf existed and from the grid's implicit background
// elsewhere, so carving through a solid's leafless interior forms a
// correct new band. The band mask holds the voxels with |v| below the
// half width. Stamping an empty grid builds a sphere from nothing.

struct StampParams {
    old_count: u32,
    concat_count: u32,
    new_count: u32,
    mode: u32, // 0 adds material and 1 carves
    center: vec3<f32>, // relative voxel space
    radius: f32,
    half_width: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    box_lo: vec3<i32>, // candidate box, relative leaf coords
    pad3: u32,
    box_dims: vec3<i32>,
    pad4: u32,
}

@group(0) @binding(0) var<uniform> params: StampParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_values: array<u32>;
@group(0) @binding(3) var<storage, read_write> concat_keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> new_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_masks: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;
const NOT_FOUND: u32 = 0xffffffffu;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

// Candidate leaves are appended after the old keys in concat_keys.
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
    concat_keys[params.old_count + i] = (u32(c.x) << 20u) | (u32(c.y) << 10u) | u32(c.z);
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

fn findOld(key: u32) -> u32 {
    var lo = 0u;
    var hi = params.old_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (old_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < params.old_count && old_keys[lo] == key) {
        return lo;
    }
    return NOT_FOUND;
}

// Implicit background of the old grid at a voxel, as in merge.wgsl.
fn implicitOld(leaf: vec3<i32>, n: u32) -> f32 {
    let col_base = (u32(leaf.x) << 20u) | (u32(leaf.y) << 10u);
    var lo = 0u;
    var hi = params.old_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (old_keys[mid] < col_base) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= params.old_count || (old_keys[lo] & 0xfffffc00u) != col_base) {
        return params.half_width;
    }
    var best = lo;
    var best_d = abs(i32(old_keys[lo] & 0x3ffu) - leaf.z);
    var i = lo + 1u;
    while (i < params.old_count && (old_keys[i] & 0xfffffc00u) == col_base) {
        let d = abs(i32(old_keys[i] & 0x3ffu) - leaf.z);
        if (d >= best_d) {
            break;
        }
        best = i;
        best_d = d;
        i = i + 1u;
    }
    let zloc = select(0u, 7u, i32(old_keys[best] & 0x3ffu) < leaf.z);
    let facing = (n & 0x1f8u) | zloc;
    return select(params.half_width, -params.half_width, bitcast<f32>(old_values[(best * 512u) + facing]) < 0.0);
}

@compute @workgroup_size(256)
fn apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.new_count) {
        return;
    }
    let key = new_keys[i];
    let leaf = vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu));
    let origin = leaf << vec3<u32>(3u);
    let old = findOld(key);
    let hw = params.half_width;
    var mask = 0u;
    for (var n = 0u; n < 512u; n = n + 1u) {
        var v: f32;
        if (old != NOT_FOUND) {
            v = bitcast<f32>(old_values[(old * 512u) + n]);
        } else {
            v = implicitOld(leaf, n);
        }
        let local = vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
        let p = vec3<f32>(origin + local);
        let brush = length(p - params.center) - params.radius;
        if (params.mode == 0u) {
            v = min(v, brush);
        } else {
            v = max(v, -brush);
        }
        v = clamp(v, -hw, hw);
        out_values[(i * 512u) + n] = bitcast<u32>(v);
        if (abs(v) < hw) {
            mask = mask | (1u << (n & 31u));
        }
        if ((n & 31u) == 31u) {
            out_masks[(i * 16u) + (n >> 5u)] = mask;
            mask = 0u;
        }
    }
}
