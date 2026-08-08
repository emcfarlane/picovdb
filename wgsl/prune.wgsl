// Grid op: prune. AND each leaf's mask with a retain mask and drop leaves
// left empty, compacting the leaf table (the topology half of NanoVDB's
// PruneGrid). The host scans flags between mark and compact.

struct PruneParams {
    count: u32,
}

@group(0) @binding(0) var<uniform> params: PruneParams;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> masks: array<u32>;
@group(0) @binding(3) var<storage, read> retain: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_masks: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

fn survives(i: u32) -> bool {
    var any = 0u;
    for (var w = 0u; w < 16u; w = w + 1u) {
        any = any | (masks[(i * 16u) + w] & retain[(i * 16u) + w]);
    }
    return any != 0u;
}

// flags has count + 1 entries; the trailing 0 makes the exclusive scan's
// last element the surviving-leaf count.
@compute @workgroup_size(256)
fn mark(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i > params.count) {
        return;
    }
    if (i == params.count) {
        flags[i] = 0u;
        return;
    }
    flags[i] = select(0u, 1u, survives(i));
}

// flags now holds the scanned output positions.
@compute @workgroup_size(256)
fn compact(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = globalIndex(wid, lid);
    if (i >= params.count || !survives(i)) {
        return;
    }
    let o = flags[i];
    out_keys[o] = keys[i];
    for (var w = 0u; w < 16u; w = w + 1u) {
        out_masks[(o * 16u) + w] = masks[(i * 16u) + w] & retain[(i * 16u) + w];
    }
}
