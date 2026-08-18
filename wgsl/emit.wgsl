// Emits the picovdb tree Combines the distance slabs and inside masks into
// signed narrow band values, drops empty leaves, derives the lower and upper
// and root tables by key truncation over the sorted leaf list, and writes the
// node buffers in the layout the renderer uploads. Node order matches the CPU
// because every level is emitted in sorted key order.
//
// Each entry point uses a subset of the bindings. Pipelines use auto layouts
// so per kernel storage buffer counts stay under the WebGPU limit.

struct EmitParams {
    cand_count: u32,
    final_count: u32,
    lower_count: u32,
    upper_count: u32,
    half_width: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    leaf_min: vec3<i32>,
    pad3: u32,
    lower_min: vec3<i32>,
    pad4: u32,
    upper_min: vec3<i32>,
    pad5: u32,
}

@group(0) @binding(0) var<uniform> params: EmitParams;
@group(0) @binding(1) var<storage, read> cand_keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>; // f32 bits, squared distances in and signed distances out
@group(0) @binding(3) var<storage, read> inside_masks: array<u32>;
@group(0) @binding(4) var<storage, read_write> band_masks: array<u32>;
@group(0) @binding(5) var<storage, read_write> band_counts: array<u32>;
@group(0) @binding(6) var<storage, read_write> bounds: array<atomic<i32>>; // min xyz, max xyz
@group(0) @binding(7) var<storage, read_write> flags: array<u32>;
@group(0) @binding(8) var<storage, read_write> final_keys: array<u32>;
@group(0) @binding(9) var<storage, read_write> final_cand: array<u32>;
@group(0) @binding(10) var<storage, read_write> value_counts: array<u32>;
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
@group(0) @binding(24) var<storage, read_write> tmp_keys: array<u32>;
@group(0) @binding(25) var<storage, read_write> tmp_cand: array<u32>;
@group(0) @binding(26) var<storage, read_write> hier: array<u32>;
@group(0) @binding(27) var<storage, read_write> idx: array<u32>;
@group(0) @binding(28) var<storage, read_write> flat_lower: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;
const LEAF_U32: u32 = 52u;   // 208 bytes
const LOWER_U32: u32 = 388u; // 1552 bytes
const UPPER_U32: u32 = 3076u; // 12304 bytes

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

fn unpack(key: u32) -> vec3<i32> {
    return vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu));
}

fn pack(c: vec3<i32>) -> u32 {
    return (u32(c.x) << 20u) | (u32(c.y) << 10u) | u32(c.z);
}

// Entry for emitting an edited grid. The inputs already hold signed
// values and band masks, so this pass only counts band voxels and
// collects bounds.
@compute @workgroup_size(256)
fn classify_re(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
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
    for (var n = 0u; n < 512u; n = n + 1u) {
        if (((band_masks[(i * 16u) + (n >> 5u)] >> (n & 31u)) & 1u) == 1u) {
            count = count + 1u;
            let ijk = origin + vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
            leaf_min = min(leaf_min, ijk);
            leaf_max = max(leaf_max, ijk);
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

// Writes the signed value slab and band mask per candidate leaf and
// collects the global active bounds.
@compute @workgroup_size(256)
fn classify(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.cand_count) {
        return;
    }
    if (i == params.cand_count) {
        band_counts[i] = 0u; // extra entry so the scanned total lands in the last slot
        return;
    }
    let origin = (unpack(cand_keys[i]) + params.leaf_min) << vec3<u32>(3u);
    let hw = params.half_width;
    let hw2 = hw * hw;
    var count = 0u;
    var leaf_min = vec3<i32>(2147483647);
    var leaf_max = vec3<i32>(-2147483648);
    for (var n = 0u; n < 512u; n = n + 1u) {
        let inside = ((inside_masks[(i * 16u) + (n >> 5u)] >> (n & 31u)) & 1u) == 1u;
        let d2 = bitcast<f32>(values[(i * 512u) + n]);
        var v = select(hw, -hw, inside);
        if (d2 <= hw2) {
            let d = sqrt(d2);
            v = select(d, -d, inside);
            band_masks[(i * 16u) + (n >> 5u)] = band_masks[(i * 16u) + (n >> 5u)] | (1u << (n & 31u));
            count = count + 1u;
            let ijk = origin + vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
            leaf_min = min(leaf_min, ijk);
            leaf_max = max(leaf_max, ijk);
        }
        values[(i * 512u) + n] = bitcast<u32>(v);
    }
    band_counts[i] = count;
    if (count > 0u) {
        for (var a = 0u; a < 3u; a = a + 1u) {
            atomicMin(&bounds[a], leaf_min[a]);
            atomicMax(&bounds[a + 3u], leaf_max[a]);
        }
    }
}

// Flags candidate leaves that have band voxels.
@compute @workgroup_size(256)
fn mark_band(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.cand_count) {
        return;
    }
    if (i == params.cand_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, band_counts[i] > 0u);
}

@compute @workgroup_size(256)
fn compact_band(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.cand_count || band_counts[i] == 0u) {
        return;
    }
    tmp_keys[flags[i]] = cand_keys[i];
    tmp_cand[flags[i]] = i;
}

// The CPU emits leaves depth first, uppers in coordinate order and child
// slots per level. Flat leaf key order interleaves parents, so the final
// leaves sort by a hierarchical key of upper coordinate then lower local
// slot then leaf local slot, split across two u32 words and sorted in two
// stable passes.
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
    if (j < params.final_count) {
        hier[j] = hierLoOf(tmp_keys[j]);
        idx[j] = j;
    }
}

@compute @workgroup_size(256)
fn hier_hi(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j < params.final_count) {
        hier[j] = hierHiOf(tmp_keys[idx[j]]);
    }
}

@compute @workgroup_size(256)
fn reorder_final(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j < params.final_count) {
        final_keys[j] = tmp_keys[idx[j]];
        final_cand[j] = tmp_cand[idx[j]];
    }
}

@compute @workgroup_size(256)
fn leaf_value_counts(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j > params.final_count) {
        return;
    }
    if (j == params.final_count) {
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
    if (i > params.final_count) {
        return;
    }
    if (i == params.final_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || lowerKeyOf(final_keys[i]) != lowerKeyOf(final_keys[i - 1u]));
}

@compute @workgroup_size(256)
fn compact_lower(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.final_count) {
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

// Sign of the empty space at a voxel, derived from the leaves. Every
// surface crossing lives inside a band leaf, so sign is constant across
// leafless space. The nearest final leaf in the voxel's column shares its
// facing voxel sign and a column with no leaves is outside. This matches
// the CPU's column parity and also works for edited grids.
fn leafInside(ijk: vec3<i32>) -> bool {
    let rel = (ijk >> vec3<u32>(3u)) - params.leaf_min;
    if (rel.x < 0 || rel.x > 1023 || rel.y < 0 || rel.y > 1023) {
        return false;
    }
    let col_base = (u32(rel.x) << 20u) | (u32(rel.y) << 10u);
    var lo = 0u;
    var hi = params.final_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (tmp_keys[mid] < col_base) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= params.final_count || (tmp_keys[lo] & 0xfffffc00u) != col_base) {
        return false; // no leaves in this column
    }
    // The run sorts by z. Walk while the distance decreases.
    var best = lo;
    var best_d = abs(i32(tmp_keys[lo] & 0x3ffu) - rel.z);
    var i = lo + 1u;
    while (i < params.final_count && (tmp_keys[i] & 0xfffffc00u) == col_base) {
        let d = abs(i32(tmp_keys[i] & 0x3ffu) - rel.z);
        if (d >= best_d) {
            break;
        }
        best = i;
        best_d = d;
        i = i + 1u;
    }
    let zloc = select(0u, 7u, i32(tmp_keys[best] & 0x3ffu) < rel.z);
    let n = (u32(ijk.x & 7) << 6u) | (u32(ijk.y & 7) << 3u) | zloc;
    return bitcast<f32>(values[(tmp_cand[best] * 512u) + n]) < 0.0;
}

fn findCand(key: u32) -> u32 {
    var lo = 0u;
    var hi = params.cand_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (cand_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo < params.cand_count && cand_keys[lo] == key) {
        return lo;
    }
    return 0xffffffffu;
}

// Signed value at any voxel. Candidate leaves supply their stored slab
// value and everything else takes the implicit background, mirroring
// Builder.valueAt.
fn valueAt(ijk: vec3<i32>) -> f32 {
    let leaf = ijk >> vec3<u32>(3u);
    let rel = leaf - params.leaf_min;
    if (all(rel >= vec3<i32>(0)) && all(rel <= vec3<i32>(1023))) {
        let c = findCand(pack(rel));
        if (c != 0xffffffffu) {
            let n = (u32(ijk.x & 7) << 6u) | (u32(ijk.y & 7) << 3u) | u32(ijk.z & 7);
            return bitcast<f32>(values[(c * 512u) + n]);
        }
    }
    return select(params.half_width, -params.half_width, leafInside(ijk));
}

const NEIGHBORS = array<vec3<i32>, 7>(
    vec3<i32>(1, 0, 0), vec3<i32>(0, 1, 0), vec3<i32>(0, 0, 1),
    vec3<i32>(1, 1, 0), vec3<i32>(1, 0, 1), vec3<i32>(0, 1, 1),
    vec3<i32>(1, 1, 1),
);

// Surface bit per active voxel. A voxel is surface when the sign changes
// toward any positive neighbor, mirroring Builder.emitLeaf.
@compute @workgroup_size(256)
fn surface(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j > params.final_count) {
        return;
    }
    if (j == params.final_count) {
        surf_counts[j] = 0u;
        return;
    }
    let c = final_cand[j];
    let origin = (unpack(cand_keys[c]) + params.leaf_min) << vec3<u32>(3u);
    var total = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        let band = band_masks[(c * 16u) + w];
        var surf = 0u;
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (((band >> b) & 1u) == 0u) {
                continue;
            }
            let n = (w * 32u) + b;
            let v = bitcast<f32>(values[(c * 512u) + n]);
            let local = vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
            for (var k = 0u; k < 7u; k = k + 1u) {
                let nl = local + NEIGHBORS[k];
                var nv: f32;
                if (all(nl < vec3<i32>(8))) {
                    let nn = (u32(nl.x) << 6u) | (u32(nl.y) << 3u) | u32(nl.z);
                    nv = bitcast<f32>(values[(c * 512u) + nn]);
                } else {
                    nv = valueAt(origin + nl);
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

// Writes leaf nodes. surf_counts and value_counts hold their exclusive
// scans by now.
@compute @workgroup_size(256)
fn write_leaves(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= params.final_count) {
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
        let band = band_masks[(c * 16u) + w];
        let surf = surf_masks[(j * 16u) + w];
        // Inactive voxels read as inside when their fill is negative, which
        // equals the parity bit because classify signs the fills.
        var inside = 0u;
        for (var b = 0u; b < 32u; b = b + 1u) {
            if (bitcast<f32>(values[(c * 512u) + (w * 32u) + b]) < 0.0) {
                inside = inside | (1u << b);
            }
        }
        let state = (inside & ~band) | (surf & band);
        leaves_out[base + 4u + (w * 3u)] = state;
        leaves_out[base + 5u + (w * 3u)] = band;
        leaves_out[base + 6u + (w * 3u)] = (local_state << 16u) | local_value;
        local_value = local_value + countOneBits(band);
        local_state = local_state + countOneBits(band & state);
    }
}

// Writes active values in leaf and bit order. The first two slots hold
// the implicit background values.
@compute @workgroup_size(256)
fn write_data(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= params.final_count) {
        return;
    }
    if (j == 0u) {
        data_out[0] = bitcast<u32>(params.half_width);
        data_out[1] = bitcast<u32>(-params.half_width);
    }
    let c = final_cand[j];
    var w = 2u + value_counts[j];
    for (var n = 0u; n < 512u; n = n + 1u) {
        if (((band_masks[(c * 16u) + (n >> 5u)] >> (n & 31u)) & 1u) == 1u) {
            data_out[w] = values[(c * 512u) + n];
            w = w + 1u;
        }
    }
}

// tmp_keys stays flat sorted after compaction and serves existence tests.
fn hasFinalLeaf(key: u32) -> bool {
    var lo = 0u;
    var hi = params.final_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (tmp_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo < params.final_count && tmp_keys[lo] == key;
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
