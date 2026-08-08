// Stable LSD radix sort of u32 keys with u32 payloads: 4-bit digits, 8
// passes, ping-ponging between two key/payload buffer pairs.
//
// Per pass: histogram counts each tile's digit occurrences into hist laid out
// digit-major (hist[digit * num_tiles + tile]), the host runs the scan.wgsl
// exclusive scan over hist (turning counts into global scatter bases), and
// scatter re-reads the tile, ranks items stably in workgroup memory, and
// writes them to keys_out/vals_out. See ts/gpu/radix_sort.ts.

const WG_SIZE: u32 = 256u;
const ITEMS: u32 = 4u;
const TILE: u32 = 1024u;
const RADIX: u32 = 16u;
const NO_ITEM: u32 = 0xffffffffu;

struct SortParams {
    n: u32,
    shift: u32,
    num_tiles: u32,
}

@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> keys_in: array<u32>;
@group(0) @binding(2) var<storage, read> vals_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> vals_out: array<u32>;
@group(0) @binding(5) var<storage, read_write> hist: array<u32>;

var<workgroup> counts: array<atomic<u32>, RADIX>;

@compute @workgroup_size(256)
fn histogram(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    if (lid.x < RADIX) {
        atomicStore(&counts[lid.x], 0u);
    }
    workgroupBarrier();
    let base = (wid.x * TILE) + (lid.x * ITEMS);
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        if ((base + i) < params.n) {
            let d = (keys_in[base + i] >> params.shift) & (RADIX - 1u);
            atomicAdd(&counts[d], 1u);
        }
    }
    workgroupBarrier();
    if (lid.x < RADIX) {
        hist[(lid.x * params.num_tiles) + wid.x] = atomicLoad(&counts[lid.x]);
    }
}

var<workgroup> scan_buf: array<u32, WG_SIZE>;

@compute @workgroup_size(256)
fn scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let base = (wid.x * TILE) + (tid * ITEMS);

    // Items are assigned to threads in tile order (thread t owns elements
    // [t*ITEMS, t*ITEMS+ITEMS)) so thread order x item order is the stable
    // within-tile order.
    var k: array<u32, ITEMS>;
    var v: array<u32, ITEMS>;
    var d: array<u32, ITEMS>;
    var cnt: array<u32, RADIX>;
    for (var b = 0u; b < RADIX; b = b + 1u) {
        cnt[b] = 0u;
    }
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        d[i] = NO_ITEM;
        if ((base + i) < params.n) {
            k[i] = keys_in[base + i];
            v[i] = vals_in[base + i];
            let digit = (k[i] >> params.shift) & (RADIX - 1u);
            d[i] = digit;
            cnt[digit] = cnt[digit] + 1u;
        }
    }

    // Per digit, exclusive scan of per-thread counts across the workgroup:
    // thread_base[b] = items with digit b in lower-numbered threads.
    var thread_base: array<u32, RADIX>;
    for (var b = 0u; b < RADIX; b = b + 1u) {
        scan_buf[tid] = cnt[b];
        workgroupBarrier();
        for (var offset = 1u; offset < WG_SIZE; offset = offset << 1u) {
            var s = scan_buf[tid];
            if (tid >= offset) {
                s = s + scan_buf[tid - offset];
            }
            workgroupBarrier();
            scan_buf[tid] = s;
            workgroupBarrier();
        }
        thread_base[b] = scan_buf[tid] - cnt[b];
        workgroupBarrier();
    }

    // hist now holds globally scanned bases; cnt is reused as the count of
    // this thread's items already placed per digit.
    for (var b = 0u; b < RADIX; b = b + 1u) {
        cnt[b] = 0u;
    }
    for (var i = 0u; i < ITEMS; i = i + 1u) {
        if (d[i] != NO_ITEM) {
            let digit = d[i];
            let dst = hist[(digit * params.num_tiles) + wid.x] + thread_base[digit] + cnt[digit];
            keys_out[dst] = k[i];
            vals_out[dst] = v[i];
            cnt[digit] = cnt[digit] + 1u;
        }
    }
}
