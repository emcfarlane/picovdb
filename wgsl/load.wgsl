// Loads a picovdb tree into the op layer. The host orders the leaves by
// key. gather_leaves copies the leaf records in that order. convert_data
// rescales the values to the op layer's half width, and turns u8 values
// into f32. Records keep their value indices, so the value array stays in
// file order.

struct LoadParams {
    leaf_count: u32,
    data_count: u32, // value slots including the two implicit entries
    scale: f32, // multiplies stored values, maps the file's background to the half width
    grid_type: u32, // 1 f32, 2 u8 (unorm bytes mapped to [-3, 3] as the renderer reads them)
}

@group(0) @binding(0) var<uniform> params: LoadParams;
@group(0) @binding(1) var<storage, read> order: array<u32>; // output slot -> tree leaf index
@group(0) @binding(2) var<storage, read> leaves: array<u32>; // tree leaf nodes
@group(0) @binding(3) var<storage, read> data: array<u32>; // f32 bits, or packed u8
@group(0) @binding(4) var<storage, read_write> out_leaves: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_data: array<u32>;

const DISPATCH_STRIDE: u32 = 65535u;
const LEAF_U32: u32 = 52u; // 208 bytes

fn globalIndex(wid: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (((wid.y * DISPATCH_STRIDE) + wid.x) * 256u) + lid.x;
}

@compute @workgroup_size(256)
fn gather_leaves(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = globalIndex(wid, lid);
    if (j >= params.leaf_count) {
        return;
    }
    let src = order[j] * LEAF_U32;
    let dst = j * LEAF_U32;
    for (var k = 0u; k < LEAF_U32; k = k + 1u) {
        out_leaves[dst + k] = leaves[src + k];
    }
}

@compute @workgroup_size(256)
fn convert_data(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d = globalIndex(wid, lid);
    if (d >= params.data_count) {
        return;
    }
    var v: f32;
    if (params.grid_type == 2u) {
        v = fma(unpack4x8unorm(data[d >> 2u])[d & 3u], 6.0, -3.0);
    } else {
        v = bitcast<f32>(data[d]);
    }
    out_data[d] = bitcast<u32>(v * params.scale);
}
