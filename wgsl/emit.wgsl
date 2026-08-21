// Builds the picovdb tree of an op layer grid: the leaf, lower, upper,
// and root node buffers and the value array, in the file layout. Leaves
// come out in the CPU converter's order, so the output matches it byte
// for byte.
//
// The grid binds as cand_keys, cand_leaves, cand_data. classify_mark and
// classify_apply are the mesh converter's entry: they build an op layer
// grid from per leaf squared distance slabs and inside masks.
//
// Each entry point uses a few of the bindings. Pipelines use auto layouts,
// which keeps storage buffer counts under the WebGPU limit.

struct EmitParams {
    cand_count: u32,
    lower_count: u32,
    upper_count: u32,
    half_width: f32,
    leaf_min: vec3<i32>,
    pad0: u32,
    lower_min: vec3<i32>,
    pad1: u32,
    upper_min: vec3<i32>,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: EmitParams;
@group(0) @binding(1) var<storage, read> cand_keys: array<u32>;
@group(0) @binding(2) var<storage, read> cand_leaves: array<u32>;
@group(0) @binding(3) var<storage, read> cand_data: array<u32>;
@group(0) @binding(4) var<storage, read> dist2: array<u32>; // classify: squared distances as f32 bits, 512 per leaf
@group(0) @binding(5) var<storage, read_write> band_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> bounds: array<atomic<i32>>; // min xyz, max xyz
@group(0) @binding(7) var<storage, read_write> flags: array<u32>;
@group(0) @binding(8) var<storage, read_write> final_keys: array<u32>;
@group(0) @binding(9) var<storage, read_write> final_cand: array<u32>;
@group(0) @binding(10) var<storage, read_write> value_counts: array<u32>;
@group(0) @binding(11) var<storage, read> inside_masks: array<u32>; // classify: 16 words per leaf
@group(0) @binding(13) var<storage, read_write> surf_masks: array<u32>;
@group(0) @binding(14) var<storage, read_write> surf_counts: array<u32>;
@group(0) @binding(15) var<storage, read_write> leaves_out: array<u32>;
@group(0) @binding(16) var<storage, read_write> data_out: array<u32>; // f32 bits
@group(0) @binding(17) var<storage, read_write> lower_keys: array<u32>;
@group(0) @binding(18) var<storage, read_write> lower_first: array<u32>;
@group(0) @binding(19) var<storage, read_write> lowers_out: array<u32>;
@group(0) @binding(20) var<storage, read_write> upper_keys: array<u32>;
@group(0) @binding(21) var<storage, read_write> upper_first: array<u32>;
@group(0) @binding(22) var<storage, read_write> uppers_out: array<u32>;
@group(0) @binding(23) var<storage, read_write> roots_out: array<u32>;
@group(0) @binding(26) var<storage, read_write> hier: array<u32>;
@group(0) @binding(27) var<storage, read_write> idx: array<u32>;
@group(0) @binding(28) var<storage, read_write> flat_lower: array<u32>;

const LOWER_U32: u32 = 388u; // 1552 bytes
const UPPER_U32: u32 = 3076u; // 12304 bytes

// Signed value of voxel n of binned leaf c: its distance inside the band,
// the background elsewhere.
fn classifyValue(c: u32, n: u32) -> f32 {
    let hw = params.half_width;
    let inside = ((inside_masks[(c * 16u) + (n >> 5u)] >> (n & 31u)) & 1u) == 1u;
    let d2 = bitcast<f32>(dist2[(c * 512u) + n]);
    var v = select(hw, -hw, inside);
    if (d2 <= hw * hw) {
        v = select(sqrt(d2), -sqrt(d2), inside);
    }
    return v;
}

@compute @workgroup_size(256)
fn classify_mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (markSentinel(i, params.cand_count)) {
        return;
    }
    var acc: LeafAcc;
    for (var n = 0u; n < 512u; n = n + 1u) {
        accPush(&acc, i, n, classifyValue(i, n));
    }
    accFinish(&acc, i);
}

@compute @workgroup_size(256)
fn classify_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= arrayLength(&w_keys)) {
        return;
    }
    let c = cand_find(w_keys[j]);
    let base = w_leaves[(j * LEAF_U32) + 2u];
    var k = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = bandWord(j, w);
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 0u) {
                continue;
            }
            w_data[base + k] = bitcast<u32>(classifyValue(c, (w * 32u) + b));
            k = k + 1u;
        }
    }
}

// Band count and voxel bounds per leaf. band_counts has an extra entry
// for the scan total.
@compute @workgroup_size(256)
fn leaf_stats(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.cand_count) {
        return;
    }
    if (i == params.cand_count) {
        band_counts[i] = 0u;
        return;
    }
    let origin = (unpack(cand_keys[i]) + params.leaf_min) << vec3<u32>(3u);
    var count = 0u;
    var leaf_min = vec3<i32>(2147483647);
    var leaf_max = vec3<i32>(-2147483648);
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = cand_leaves[(i * LEAF_U32) + 5u + (w * 3u)];
        count = count + countOneBits(band);
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 1u) {
                let ijk = origin + voxelLocal((w * 32u) + b);
                leaf_min = min(leaf_min, ijk);
                leaf_max = max(leaf_max, ijk);
            }
        }
    }
    band_counts[i] = count;
    if (count > 0u) {
        for (var a = 0u; a < 3u; a = a + 1u) {
            atomicMin(&bounds[a], leaf_min[a]);
            atomicMax(&bounds[a + 3u], leaf_max[a]);
        }
    }
}

// The CPU emits leaves depth first: uppers in coordinate order, then
// child slots per level. Flat key order interleaves parents. So the leaves
// sort by a hierarchical key: upper coordinate, lower slot, leaf slot. It
// spans two u32 words, sorted in two stable passes.
fn hierLoOf(key: u32) -> u32 {
    let a = unpack(key) + params.leaf_min;
    let low = (a >> vec3<u32>(4u)) & vec3<i32>(31);
    let lf = a & vec3<i32>(15);
    return (u32(low.x) << 22u) | (u32(low.y) << 17u) | (u32(low.z) << 12u)
        | (u32(lf.x) << 8u) | (u32(lf.y) << 4u) | u32(lf.z);
}

fn hierHiOf(key: u32) -> u32 {
    let a = unpack(key) + params.leaf_min;
    let up = (a >> vec3<u32>(9u)) - params.upper_min;
    return (u32(up.x) << 20u) | (u32(up.y) << 10u) | u32(up.z);
}

@compute @workgroup_size(256)
fn hier_lo(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j < params.cand_count) {
        hier[j] = hierLoOf(cand_keys[j]);
        idx[j] = j;
    }
}

@compute @workgroup_size(256)
fn hier_hi(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j < params.cand_count) {
        hier[j] = hierHiOf(cand_keys[idx[j]]);
    }
}

@compute @workgroup_size(256)
fn reorder_final(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j < params.cand_count) {
        final_keys[j] = cand_keys[idx[j]];
        final_cand[j] = idx[j];
    }
}

@compute @workgroup_size(256)
fn leaf_value_counts(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j > params.cand_count) {
        return;
    }
    if (j == params.cand_count) {
        value_counts[j] = 0u;
        return;
    }
    value_counts[j] = band_counts[final_cand[j]];
}

fn lowerKeyOf(leaf_key: u32) -> u32 {
    let abs = unpack(leaf_key) + params.leaf_min;
    return pack((abs >> vec3<u32>(4u)) - params.lower_min);
}

fn upperKeyOf(lower_key: u32) -> u32 {
    let abs = unpack(lower_key) + params.lower_min;
    return pack((abs >> vec3<u32>(5u)) - params.upper_min);
}

@compute @workgroup_size(256)
fn mark_lower(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.cand_count) {
        return;
    }
    if (i == params.cand_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || lowerKeyOf(final_keys[i]) != lowerKeyOf(final_keys[i - 1u]));
}

@compute @workgroup_size(256)
fn compact_lower(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.cand_count) {
        return;
    }
    if (i == 0u || lowerKeyOf(final_keys[i]) != lowerKeyOf(final_keys[i - 1u])) {
        lower_keys[flags[i]] = lowerKeyOf(final_keys[i]);
        lower_first[flags[i]] = i;
    }
}

@compute @workgroup_size(256)
fn mark_upper(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.lower_count) {
        return;
    }
    if (i == params.lower_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || upperKeyOf(lower_keys[i]) != upperKeyOf(lower_keys[i - 1u]));
}

@compute @workgroup_size(256)
fn compact_upper(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.lower_count) {
        return;
    }
    if (i == 0u || upperKeyOf(lower_keys[i]) != upperKeyOf(lower_keys[i - 1u])) {
        upper_keys[flags[i]] = upperKeyOf(lower_keys[i]);
        upper_first[flags[i]] = i;
    }
}

// Sign of the empty space at a voxel, from the grid's implicit background.
fn leafInside(ijk: vec3<i32>) -> bool {
    return cand_valueAt(ijk - (params.leaf_min << vec3<u32>(3u))) < 0.0;
}

const NEIGHBORS = array<vec3<i32>, 7>(
    vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1),
    vec3<i32>(1, 1, 0), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1),
    vec3<i32>(1, 1, 1),
);

// A voxel is surface when its sign differs from any positive neighbor.
// Mirrors Builder.emitLeaf.
@compute @workgroup_size(256)
fn surface(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j > params.cand_count) {
        return;
    }
    if (j == params.cand_count) {
        surf_counts[j] = 0u;
        return;
    }
    let c = final_cand[j];
    let origin = unpack(cand_keys[c]) << vec3<u32>(3u);
    var total = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = cand_leaves[(c * LEAF_U32) + 5u + (w * 3u)];
        var surf = 0u;
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 0u) {
                continue;
            }
            let n = (w * 32u) + b;
            let v = cand_leafValue(c, n);
            let local = voxelLocal(n);
            for (var k = 0u; k < 7u; k = k + 1u) {
                let nl = local + NEIGHBORS[k];
                var nv: f32;
                if (all(nl < vec3<i32>(8))) {
                    nv = cand_leafValue(c, voxelOffset(nl));
                } else {
                    nv = cand_valueAt(origin + nl);
                }
                let strict = (v < 0.0) != (nv < 0.0);
                let nonstrict = (v <= 0.0) != (nv <= 0.0);
                if (strict || nonstrict) {
                    surf = surf | (1u << b);
                    break;
                }
            }
        }
        surf_masks[(j * 16u) + w] = surf;
        total = total + countOneBits(surf);
    }
    surf_counts[j] = total;
}

// Writes leaf nodes: the op layer records plus surface bits and final
// value bases. surf_counts and value_counts hold exclusive scans.
@compute @workgroup_size(256)
fn write_leaves(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= params.cand_count) {
        return;
    }
    let c = final_cand[j];
    let base = j * LEAF_U32;
    leaves_out[base] = surf_counts[j]; // running surface count
    leaves_out[base + 1u] = 0u;
    leaves_out[base + 2u] = 2u + value_counts[j];
    leaves_out[base + 3u] = 0u;
    var local_state = 0u;
    var local_value = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let e = (c * LEAF_U32) + 4u + (w * 3u);
        let band = cand_leaves[e + 1u];
        let surf = surf_masks[(j * 16u) + w];
        let state = (cand_leaves[e] & ~band) | (surf & band);
        leaves_out[base + 4u + (w * 3u)] = state;
        leaves_out[base + 5u + (w * 3u)] = band;
        leaves_out[base + 6u + (w * 3u)] = (local_state << 16u) | local_value;
        local_value = local_value + countOneBits(band);
        local_state = local_state + countOneBits(band & state);
    }
}

// Copies each leaf's values to their final slot. The first two slots hold
// the implicit background values.
@compute @workgroup_size(256)
fn write_data(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= params.cand_count) {
        return;
    }
    if (j == 0u) {
        data_out[0] = bitcast<u32>(params.half_width);
        data_out[1] = bitcast<u32>(-params.half_width);
    }
    let c = final_cand[j];
    let src = cand_leaves[(c * LEAF_U32) + 2u];
    let dst = 2u + value_counts[j];
    let count = band_counts[c];
    for (var k = 0u; k < count; k = k + 1u) {
        data_out[dst + k] = cand_data[src + k];
    }
}

fn hasFinalLeaf(key: u32) -> bool {
    return cand_find(key) != NOT_FOUND;
}

// flat_lower is the flat sorted copy of the lower keys.
fn hasLower(key: u32) -> bool {
    var lo = 0u;
    var hi = params.lower_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (flat_lower[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < params.lower_count && flat_lower[lo] == key;
}

// Writes lower nodes in the CPU's child slot order.
@compute @workgroup_size(256)
fn write_lowers(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let l = globalIndex(wid, lid);
    if (l >= params.lower_count) {
        return;
    }
    let lower_abs = unpack(lower_keys[l]) + params.lower_min;
    let base = l * LOWER_U32;
    lowers_out[base] = lower_first[l];
    lowers_out[base + 1u] = 0u;
    lowers_out[base + 2u] = 2u + value_counts[lower_first[l]];
    lowers_out[base + 3u] = 0u;
    var local_state = 0u;
    for (var w = 0u; w < 128u; w = w + 1u) {
        var state = 0u;
        var value = 0u;
        for (var b = 0u; b < 32u; b = b + 1u) {
            let n = (w * 32u) + b;
            let local = vec3<i32>(i32((n >> 8u) & 15u), i32((n >> 4u) & 15u), i32(n & 15u));
            let leaf_abs = (lower_abs << vec3<u32>(4u)) + local;
            if (hasFinalLeaf(pack(leaf_abs - params.leaf_min))) {
                state = state | (1u << b);
                value = value | (1u << b);
            } else if (leafInside((leaf_abs << vec3<u32>(3u)) + vec3<i32>(4))) {
                state = state | (1u << b);
            }
        }
        lowers_out[base + 4u + (w * 3u)] = state;
        lowers_out[base + 5u + (w * 3u)] = value;
        lowers_out[base + 6u + (w * 3u)] = local_state << 16u;
        local_state = local_state + countOneBits(value & state);
    }
}

// Writes upper nodes in the CPU's child slot order.
@compute @workgroup_size(256)
fn write_uppers(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = globalIndex(wid, lid);
    if (u >= params.upper_count) {
        return;
    }
    let upper_abs = unpack(upper_keys[u]) + params.upper_min;
    let base = u * UPPER_U32;
    uppers_out[base] = upper_first[u];
    uppers_out[base + 1u] = 0u;
    uppers_out[base + 2u] = lowers_out[(upper_first[u] * LOWER_U32) + 2u];
    uppers_out[base + 3u] = 0u;
    var local_state = 0u;
    for (var w = 0u; w < 1024u; w = w + 1u) {
        var state = 0u;
        var value = 0u;
        for (var b = 0u; b < 32u; b = b + 1u) {
            let n = (w * 32u) + b;
            let local = vec3<i32>(i32((n >> 10u) & 31u), i32((n >> 5u) & 31u), i32(n & 31u));
            let lower_abs = (upper_abs << vec3<u32>(5u)) + local;
            if (hasLower(pack(lower_abs - params.lower_min))) {
                state = state | (1u << b);
                value = value | (1u << b);
            } else if (leafInside((lower_abs << vec3<u32>(7u)) + vec3<i32>(64))) {
                state = state | (1u << b);
            }
        }
        uppers_out[base + 4u + (w * 3u)] = state;
        uppers_out[base + 5u + (w * 3u)] = value;
        uppers_out[base + 6u + (w * 3u)] = local_state << 16u;
        local_state = local_state + countOneBits(value & state);
    }
}

// Root keys, mirroring picovdb coordToKey.
@compute @workgroup_size(256)
fn write_roots(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let u = globalIndex(wid, lid);
    if (u >= params.upper_count) {
        return;
    }
    let origin = (unpack(upper_keys[u]) + params.upper_min) << vec3<u32>(12u);
    let iu = bitcast<u32>(origin.x) >> 12u;
    let ju = bitcast<u32>(origin.y) >> 12u;
    let ku = bitcast<u32>(origin.z) >> 12u;
    roots_out[u * 2u] = ku | (ju << 21u);
    roots_out[(u * 2u) + 1u] = (iu << 10u) | (ju >> 11u);
}
