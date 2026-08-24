// Exclusive prefix scan of u32 values with wrapping addition.
//
// scan_tile writes each 1024 element tile's exclusive scan in place and the
// tile total to partials. The host scans partials recursively with the same
// kernels and add_offsets folds the scanned totals back in. The dispatch
// plan lives in ts/gpu/scan.ts.

const WG_SIZE: u32 = 256u;
const ITEMS: u32 = 4u;
const TILE: u32 = 1024u;

struct ScanParams {
    n: u32,
}

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<u32>;

var<workgroup> thread_sums: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn scan_tile(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let base = (wid.x * TILE) + (tid * ITEMS);

    var v: array<u32, ITEMS>;
    var sum = 0u;
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        var x = 0u;
        if ((base + i) < params.n) {
            x = data[base + i];
        }
        v[i] = x;
        sum = sum + x;
    }

    thread_sums[tid] = sum;
    workgroupBarrier();
    // Inclusive scan over the 256 thread sums.
    for (var offset = 1u; offset < WG_SIZE; offset = offset << 1u) {
        var s = thread_sums[tid];
        if (tid >= offset) {
            s = s + thread_sums[tid - offset];
        }
        workgroupBarrier();
        thread_sums[tid] = s;
        workgroupBarrier();
    }

    var running = thread_sums[tid] - sum; // exclusive offset of this thread
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        if ((base + i) < params.n) {
            data[base + i] = running;
        }
        running = running + v[i];
    }
    if (tid == (WG_SIZE - 1u)) {
        partials[wid.x] = thread_sums[tid];
    }
}

@compute @workgroup_size(256)
fn add_offsets(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let offset = partials[wid.x];
    let base = (wid.x * TILE) + (lid.x * ITEMS);
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        if ((base + i) < params.n) {
            data[base + i] = data[base + i] + offset;
        }
    }
}
