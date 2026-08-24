// Computes the narrow band squared distances for binned leaves.
//
// One workgroup per leaf walks its contiguous run of the key sorted pair
// list, each thread covers two of the 512 voxels, and minima accumulate in
// a workgroup memory slab written back with plain stores. The workgroup
// owns its leaf, so there are no global atomics and no slot searches.
//
// Distances are nonnegative, so the u32 bit pattern of an f32 orders the
// same as the float value and atomicMin on the bitcast is an exact float
// min.
//
// The distance function mirrors distSqPointTriangle in src/mesh_to_grid.zig
// with an explicit summation order. A voxel updates only when it lies in
// the triangle's dilated bounds and its squared distance is within the
// squared half width, the same gate the CPU uses.

struct RasterParams {
    pair_count: u32,
    leaf_count: u32,
    half_width: f32,
    pad: u32,
    leaf_min: vec3<i32>,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: RasterParams;
@group(0) @binding(1) var<storage, read> points_index: array<f32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> pair_keys: array<u32>;
@group(0) @binding(4) var<storage, read> pair_tris: array<u32>;
@group(0) @binding(5) var<storage, read> leaf_keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> leaf_values: array<u32>;

const INF_BITS: u32 = 0x7f800000u;
const DISPATCH_STRIDE: u32 = 65535u;

fn dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return ((a.x * b.x) + (a.y * b.y)) + (a.z * b.z);
}

fn dsq(a: vec3<f32>) -> f32 {
    return dot3(a, a);
}

fn distSqPointSegment(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
    let ab = b - a;
    let denom = dsq(ab);
    if (denom <= 0.0) {
        return dsq(p - a);
    }
    let t = clamp(dot3(p - a, ab) / denom, 0.0, 1.0);
    return dsq(p - (a + (ab * t)));
}

fn distSqPointTriangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot3(ab, ap);
    let d2 = dot3(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return dsq(ap); // vertex a
    }

    let bp = p - b;
    let d3 = dot3(ab, bp);
    let d4 = dot3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return dsq(bp); // vertex b
    }

    let vc = (d1 * d4) - (d3 * d2);
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let denom = d1 - d3;
        if (denom > 0.0) {
            return dsq(ap - (ab * (d1 / denom))); // edge ab
        }
    }

    let cp = p - c;
    let d5 = dot3(ab, cp);
    let d6 = dot3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return dsq(cp); // vertex c
    }

    let vb = (d5 * d2) - (d1 * d6);
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let denom = d2 - d6;
        if (denom > 0.0) {
            return dsq(ap - (ac * (d2 / denom))); // edge ac
        }
    }

    let va = (d3 * d6) - (d5 * d4);
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let denom = (d4 - d3) + (d5 - d6);
        if (denom > 0.0) {
            return dsq(bp - ((c - b) * ((d4 - d3) / denom))); // edge bc
        }
    }

    let denom = (va + vb) + vc;
    if (denom <= 0.0) {
        // Degenerate triangles fall back to edge distances.
        let e0 = distSqPointSegment(p, a, b);
        let e1 = distSqPointSegment(p, a, c);
        let e2 = distSqPointSegment(p, b, c);
        return min(e0, min(e1, e2));
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    return dsq(ap - ((ab * v) + (ac * w))); // face
}

fn loadPoint(i: u32) -> vec3<f32> {
    return vec3<f32>(points_index[i * 3u], points_index[(i * 3u) + 1u], points_index[(i * 3u) + 2u]);
}

var<workgroup> slab: array<atomic<u32>, 512>;

@compute @workgroup_size(256)
fn rasterize(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * DISPATCH_STRIDE) + wid.x;
    if (i >= params.leaf_count) {
        return;
    }
    let key = leaf_keys[i];
    let leaf = vec3<i32>(i32(key >> 20u), i32((key >> 10u) & 0x3ffu), i32(key & 0x3ffu)) + params.leaf_min;
    let origin = leaf << vec3<u32>(3u);

    atomicStore(&slab[lid.x], INF_BITS);
    atomicStore(&slab[lid.x + 256u], INF_BITS);
    workgroupBarrier();

    // Find this leaf's run in the key sorted pair list.
    var lo = 0u;
    var hi = params.pair_count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (pair_keys[mid] < key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    let hw = params.half_width;
    let hw2 = hw * hw;
    for (var p = lo; p < params.pair_count && pair_keys[p] == key; p = p + 1u) {
        let t = pair_tris[p];
        let a = loadPoint(triangles[t * 3u]);
        let b = loadPoint(triangles[(t * 3u) + 1u]);
        let c = loadPoint(triangles[(t * 3u) + 2u]);
        let lo_v = vec3<i32>(ceil(min(a, min(b, c)) - vec3<f32>(hw)));
        let hi_v = vec3<i32>(floor(max(a, max(b, c)) + vec3<f32>(hw)));

        for (var n = lid.x; n < 512u; n = n + 256u) {
            // Voxel order matches picovdb leafCoordToOffset.
            let local = vec3<i32>(i32(n >> 6u), i32((n >> 3u) & 7u), i32(n & 7u));
            let ijk = origin + local;
            if (any(ijk < lo_v) || any(ijk > hi_v)) {
                continue;
            }
            let d2 = distSqPointTriangle(vec3<f32>(ijk), a, b, c);
            if (d2 <= hw2) {
                atomicMin(&slab[n], bitcast<u32>(d2));
            }
        }
    }

    workgroupBarrier();
    leaf_values[(i * 512u) + lid.x] = atomicLoad(&slab[lid.x]);
    leaf_values[(i * 512u) + lid.x + 256u] = atomicLoad(&slab[lid.x + 256u]);
}
