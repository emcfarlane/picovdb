// Dilates an active voxel mask set by one voxel across face neighbors,
// the word shift approach of NanoVDB DilateGrid on u32 words.
//
// Input is a sorted unique leaf key table with one 512 bit mask per leaf.
// Each u32 word covers one x and four y values, one byte per column of 8
// z bits. count_spawn and emit_spawn emit each leaf plus any face neighbor
// its boundary planes spill into. The host sorts and dedupes the keys and
// dilate_masks builds each output leaf's mask from its own shifted words
// plus the six neighbors' boundary planes.
//
// Neighbors falling outside the packed key range are dropped, so callers
// keep a margin of one leaf.

struct DilateParams {
    old_count: u32,
    spawn_count: u32,
    new_count: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: DilateParams;
@group(0) @binding(1) var<storage, read> old_keys: array<u32>;
@group(0) @binding(2) var<storage, read> old_masks: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read_write> spawn_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> new_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> new_masks: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;

fn unpack(key: u32) -> vec3<i32> {
    return vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu));
}

fn pack(c: vec3<i32>) -> u32 {
    return (u32(c.x) << 20u) | (u32(c.y) << 10u) | u32(c.z);
}

fn inRange(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c <= vec3<i32>(1023));
}

// True when the leaf's boundary plane facing direction d has any active
// voxel. Directions order as negative then positive x, then y, then z.
fn spills(leaf: u32, d: u32) -> bool {
    let base = leaf * 16u;
    var acc = 0u;
    switch (d) {
        case 0u: { acc = old_masks[base] | old_masks[base + 1u]; }
        case 1u: { acc = old_masks[base + 14u] | old_masks[base + 15u]; }
        case 2u: {
            for (var k = 0u; k < 8u; k = k + 1u) {
                acc = acc | (old_masks[base + (2u * k)] & 0x000000ffu);
            }
        }
        case 3u: {
            for (var k = 0u; k < 8u; k = k + 1u) {
                acc = acc | (old_masks[base + (2u * k) + 1u] & 0xff000000u);
            }
        }
        case 4u: {
            for (var w = 0u; w < 16u; w = w + 1u) {
                acc = acc | (old_masks[base + w] & 0x01010101u);
            }
        }
        default: {
            for (var w = 0u; w < 16u; w = w + 1u) {
                acc = acc | (old_masks[base + w] & 0x80808080u);
            }
        }
    }
    return acc != 0u;
}

const DIRS = array<vec3<i32>, 6>(
    vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
    vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
    vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1),
);

// Shared by the count and emit passes, which must agree.
fn spawn(i: u32, emit: bool, offset: u32) -> u32 {
    let c = unpack(old_keys[i]);
    var w = offset;
    if (emit) {
        spawn_keys[w] = old_keys[i];
    }
    w = w + 1u;
    var n = 1u;
    for (var d = 0u; d < 6u; d = d + 1u) {
        let nc = c + DIRS[d];
        if (!inRange(nc) || !spills(i, d)) {
            continue;
        }
        if (emit) {
            spawn_keys[w] = pack(nc);
        }
        w = w + 1u;
        n = n + 1u;
    }
    return n;
}

// counts has one extra entry so the scanned total lands in the last slot.
@compute @workgroup_size(256)
fn count_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i > params.old_count) {
        return;
    }
    if (i == params.old_count) {
        counts[i] = 0u;
        return;
    }
    counts[i] = spawn(i, false, 0u);
}

// counts now holds the scanned write offsets.
@compute @workgroup_size(256)
fn emit_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.old_count) {
        let unused = spawn(i, true, counts[i]);
    }
}

// flags has one extra entry so the scanned total lands in the last slot.
@compute @workgroup_size(256)
fn mark_unique(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i > params.spawn_count) {
        return;
    }
    if (i == params.spawn_count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, i == 0u || spawn_keys[i] != spawn_keys[i - 1u]);
}

// flags now holds the scanned unique positions.
@compute @workgroup_size(256)
fn compact_unique(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.spawn_count) {
        return;
    }
    if (i == 0u || spawn_keys[i] != spawn_keys[i - 1u]) {
        new_keys[flags[i]] = spawn_keys[i];
    }
}

// Loads the leaf's 16 words, or zeros when the leaf is absent.
fn loadMask(key: u32, out: ptr<function, array<u32, 16>>) {
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
    let found = lo < params.old_count && old_keys[lo] == key;
    for (var w = 0u; w < 16u; w = w + 1u) {
        (*out)[w] = select(0u, old_masks[(lo * 16u) + w], found);
    }
}

@compute @workgroup_size(256)
fn dilate_masks(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
    if (i >= params.new_count) {
        return;
    }
    let c = unpack(new_keys[i]);
    var s: array<u32, 16>;
    loadMask(new_keys[i], &s);

    var out: array<u32, 16>;
    // Within the leaf, identity plus one voxel shifts in y and z.
    for (var k = 0u; k < 8u; k = k + 1u) {
        let lo = s[2u * k];
        let hi = s[(2u * k) + 1u];
        out[2u * k] = lo
            | (lo << 8u) | ((lo >> 8u) | ((hi & 0xffu) << 24u))
            | ((lo << 1u) & 0xfefefefeu) | ((lo >> 1u) & 0x7f7f7f7fu);
        out[(2u * k) + 1u] = hi
            | ((hi << 8u) | (lo >> 24u)) | (hi >> 8u)
            | ((hi << 1u) & 0xfefefefeu) | ((hi >> 1u) & 0x7f7f7f7fu);
    }
    // x shifts move whole word pairs.
    for (var w = 0u; w < 16u; w = w + 1u) {
        if (w >= 2u) {
            out[w] = out[w] | s[w - 2u];
        }
        if (w < 14u) {
            out[w] = out[w] | s[w + 2u];
        }
    }

    // Boundary planes from the six face neighbors.
    var nb: array<u32, 16>;
    if (c.x > 0) {
        loadMask(pack(c + vec3<i32>(-1, 0, 0)), &nb);
        out[0] = out[0] | nb[14];
        out[1] = out[1] | nb[15];
    }
    if (c.x < 1023) {
        loadMask(pack(c + vec3<i32>(1, 0, 0)), &nb);
        out[14] = out[14] | nb[0];
        out[15] = out[15] | nb[1];
    }
    if (c.y > 0) {
        loadMask(pack(c + vec3<i32>(0, -1, 0)), &nb);
        for (var k = 0u; k < 8u; k = k + 1u) {
            out[2u * k] = out[2u * k] | (nb[(2u * k) + 1u] >> 24u);
        }
    }
    if (c.y < 1023) {
        loadMask(pack(c + vec3<i32>(0, 1, 0)), &nb);
        for (var k = 0u; k < 8u; k = k + 1u) {
            out[(2u * k) + 1u] = out[(2u * k) + 1u] | ((nb[2u * k] & 0xffu) << 24u);
        }
    }
    if (c.z > 0) {
        loadMask(pack(c + vec3<i32>(0, 0, -1)), &nb);
        for (var w = 0u; w < 16u; w = w + 1u) {
            out[w] = out[w] | ((nb[w] & 0x80808080u) >> 7u);
        }
    }
    if (c.z < 1023) {
        loadMask(pack(c + vec3<i32>(0, 0, 1)), &nb);
        for (var w = 0u; w < 16u; w = w + 1u) {
            out[w] = out[w] | ((nb[w] & 0x01010101u) << 7u);
        }
    }

    for (var w = 0u; w < 16u; w = w + 1u) {
        new_masks[(i * 16u) + w] = out[w];
    }
}
