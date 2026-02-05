
//@group(0) @binding(0) var<storage> picovdb_grids: array<PicoVDBGrid>;
//@group(0) @binding(1) var<storage> picovdb_roots: array<PicoVDBRoot>;
//@group(0) @binding(2) var<storage> picovdb_uppers: array<PicoVDBUpper>;
//@group(0) @binding(3) var<storage> picovdb_lowers: array<PicoVDBLower>;
//@group(0) @binding(4) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
//@group(0) @binding(5) var<storage> picovdb_buffer: array<u32>;

struct PicoVDBFileHeader {
  magic: vec2u,    // 'PicoVDB0' little endian (8 bytes)
  version: u32,    // Format version (4 bytes)
  gridCount: u32,  // Number of grids (4 bytes)
  upperCount: u32, // Total upper nodes (4 bytes)
  lowerCount: u32, // Total lower nodes (4 bytes)
  leafCount: u32,  // Total leaf nodes (4 bytes)
  dataCount: u32,  // Total data buffer size in 16-byte units (4 bytes)
}

struct PicoVDBGrid {
  gridIndex: u32,     // This grid's index (4 bytes)
  upperStart: u32,    // Index into uppers array (= root index) (4 bytes)
  lowerStart: u32,    // Index into lowers array (4 bytes)
  leafStart: u32,     // Index into leaves array (4 bytes)
  dataStart: u32,     // 16-byte index into data buffer (4 bytes)
  dataElemCount: u32, // Number of data elements for this grid (4 bytes)
  gridType: u32,      // GRID_TYPE_SDF_FLOAT=1, GRID_TYPE_SDF_UINT8=2 (4 bytes)
  _pad1: u32,
  indexBoundsMin: vec3i, // Index min (12 bytes)
  _pad2: u32,
  indexBoundsMax: vec3i, // Index min (12 bytes)
  _pad3: u32,
}

const GRID_TYPE_SDF_FLOAT = 1;
const GRID_TYPE_SDF_UINT8 = 2;

// https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html

// Root key for spatial lookup - maps coordinate to upper node index.
// Roots are 1:1 with uppers (root[i] -> upper[i]).
// Count derived from upperCount. Padded to 16-byte alignment.
struct PicoVDBRoot {
  key: vec2u,  // 64-bit coordinate key (8 bytes)
}

struct PicoVDBNodeMask {
  inside: u32,      // Bitmask of outside/inside (+/-) (4 bytes)
  value: u32,       // Bitmask of value, inside && value set this is a child (4 bytes)
  valueOffset: u32, // Prefix sum offset of value (4 bytes)
  childOffset: u32, // Prefix sum offset of child (4 bytes)
}

struct PicoVDBLeafMask{
  inside: u32,      // Bitmask of outside/inside (+/-) (4 bytes)
  value: u32,       // Bifmask of value, inside && value always is 0 (4 bytes)
  valueOffset: u32, // Prefix sum offset of value (4 bytes)
}

struct PicoVDBUpper {
  mask: array<PicoVDBNodeMask,1024>,
}

struct PicoVDBLower {
  mask: array<PicoVDBNodeMask,128>,
}

struct PicoVDBLeaf {
  mask: array<PicoVDBLeafMask,16>,
}

struct PicoVDBLevelCount {
    level: u32,  // Level of value found
    count: u32,  // Count offset (0 means no active values/background)
}

struct PicoVDBReadAccessor {
  key: vec3i,
  grid: u32,
  upper: u32,
  lower: u32,
  leaf: u32,
  _pad: u32,
}

const PICOVDB_INVALID_INDEX: u32 = 0xFFFFFFFFu;

fn picovdbReadAccessorInit(acc: ptr<function, PicoVDBReadAccessor>, grid: u32) {
    (*acc).key = vec3i(0x7FFFFFFF);
    (*acc).grid = grid;
    (*acc).upper = PICOVDB_INVALID_INDEX;
    (*acc).lower = PICOVDB_INVALID_INDEX;
    (*acc).leaf = PICOVDB_INVALID_INDEX;
    (*acc)._pad = 0u;
}

fn picovdbReadAccessorIsCachedLeaf(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).leaf;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0x7i) == 0; // Leaf is 8x8x8 (bits 0-2)
    (*acc).leaf = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorIsCachedLower(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).lower;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0x7Fi) == 0; // Lower is 128x128x128 (bits 0-6)
    (*acc).lower = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorIsCachedUpper(acc: ptr<function, PicoVDBReadAccessor>, dirty: i32) -> bool {
    let addr = (*acc).upper;
    let is_cached = (addr != PICOVDB_INVALID_INDEX) && (dirty & ~0xFFFi) == 0; // Upper is 4096x4096x4096 (bits 0-11)
    (*acc).upper = select(PICOVDB_INVALID_INDEX, addr, is_cached);
    return is_cached;
}

fn picovdbReadAccessorComputeDirty(acc: ptr<function, PicoVDBReadAccessor>, ijk: vec3i) -> i32 {
    return (ijk.x ^ (*acc).key.x) | (ijk.y ^ (*acc).key.y) | (ijk.z ^ (*acc).key.z);
}

fn picovdbCoordToKey(ijk: vec3i) -> vec2u {
    // Use the non-native 64-bit path since WGSL doesn't have native 64-bit
    let iu = u32(ijk.x) >> 12u;
    let ju = u32(ijk.y) >> 12u;
    let ku = u32(ijk.z) >> 12u;
    let key_x = ku | (ju << 21u);
    let key_y = (iu << 10u) | (ju >> 11u);
    return vec2u(key_x, key_y);
}

fn picovdbUpperCoordToOffset(ijk: vec3i) -> u32 {
    return (((u32(ijk.x) & 0xFFFu) >> 7u) << 10u) |
           (((u32(ijk.y) & 0xFFFu) >> 7u) << 5u)  |
            ((u32(ijk.z) & 0xFFFu) >> 7u);
}

fn picovdbLowerCoordToOffset(ijk: vec3i) -> u32 {
    return (((u32(ijk.x) & 0x7Fu) >> 3u) << 8u) |
           (((u32(ijk.y) & 0x7Fu) >> 3u) << 4u) |
            ((u32(ijk.z) & 0x7Fu) >> 3u);
}

fn picovdbLeafCoordToOffset(ijk: vec3i) -> u32 {
    return ((u32(ijk.x) & 0x7u) << 6u) |
           ((u32(ijk.y) & 0x7u) << 3u) |
            (u32(ijk.z) & 0x7u);
}


// Find root/upper index for coordinate within grid bounds.
// Roots are 1:1 with uppers, so the returned index works for both.
fn picovdbReadAccessorFindUpperIndex(
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> i32 {
    let coordKey = picovdbCoordToKey(ijk);
    let startIndex = grid.upperStart;
    let endIndex = select(
      picovdb_grids[grid.gridIndex + 1u].upperStart, // false: use next grid's start
      arrayLength(&picovdb_roots),                   // true: use total roots count
      arrayLength(&picovdb_grids) - 1u == grid.gridIndex,
    );
    for (var i = startIndex; i < endIndex; i++) {
        let root = picovdb_roots[i];
        if (coordKey.x == root.key.x && coordKey.y == root.key.y) {
            return i32(i);
        }
    }
    return -1; // Not found
}

fn picovdbReadAccessorLeafGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbLeafCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_leaves[grid.leafStart + (*acc).leaf].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;

    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(u32(is_inside), mask.valueOffset + countOneBits(preceding_bits), is_value);
    (*acc).key = ijk;
    return PicoVDBLevelCount(0u, count);
}

fn picovdbReadAccessorLowerGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbLowerCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_lowers[grid.lowerStart + (*acc).lower].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;
    if (!is_value) {
        return PicoVDBLevelCount(1u, u32(is_inside)); // fast path
    }
    if (is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).leaf = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = mask.valueOffset + countOneBits(preceding_bits);
    return PicoVDBLevelCount(1u, count);
}

fn picovdbReadAccessorUpperGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let n = picovdbUpperCoordToOffset(ijk);
    let word_index = n >> 5u; // Fast divide by 32
    let bit_index = n & 31u; // Fast modulo 32
    let mask = picovdb_uppers[grid.upperStart + (*acc).upper].mask[word_index];

    let bit_at_pos = 1u << bit_index;
    let is_value = (mask.value & bit_at_pos) != 0u;
    let is_inside = (mask.inside & bit_at_pos) != 0u;

    if (!is_value) {
        return PicoVDBLevelCount(2u, u32(is_inside)); // fast path
    }
    if (is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).lower = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = mask.valueOffset + countOneBits(preceding_bits);
    return PicoVDBLevelCount(2u, count);
}

// Get level and count from root and update cache
fn picovdbReadAccessorRootGetLevelCountAndCache(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let rootIndex = picovdbReadAccessorFindUpperIndex(ijk, grid);
    if (rootIndex == -1) {
        // No matching root tile, return background
        return PicoVDBLevelCount(3u, 0u);
    }
    (*acc).upper = u32(rootIndex);
    (*acc).key = ijk;
    return picovdbReadAccessorUpperGetLevelCountAndCache(acc, ijk, grid);
}

fn picovdbReadAccessorGetLevelCount(
    acc: ptr<function, PicoVDBReadAccessor>,
    ijk: vec3i,
    grid: PicoVDBGrid,
) -> PicoVDBLevelCount {
    let dirty = picovdbReadAccessorComputeDirty(acc, ijk);
    if (picovdbReadAccessorIsCachedLeaf(acc, dirty)) {
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    } else if (picovdbReadAccessorIsCachedLower(acc, dirty)) {
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    } else if (picovdbReadAccessorIsCachedUpper(acc, dirty)) {
        return picovdbReadAccessorUpperGetLevelCountAndCache(acc, ijk, grid);
    } else {
        return picovdbReadAccessorRootGetLevelCountAndCache(acc, ijk, grid);
    }
}

// --- HDDA (Hierarchical Digital Differential Analyzer) ---
const PICOVDB_HDDA_FLOAT_MAX: f32 = 1e38;

struct PicoVDBHDDA {
    voxel: vec3i,
    dim: i32,
    step: vec3i,
    tmin: f32,
    delta: vec3f,
    tmax: f32,
    next: vec3f,
}

fn picovdbHDDAPosToVoxel(pos: vec3f, dim: i32) -> vec3i {
    let mask = ~(dim - 1);
    return vec3i(floor(pos)) & vec3i(mask);
}

fn picovdbHDDAInit(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    direction_inv: vec3f,
    dim: i32
) {
    let pos = origin + direction * tmin;
    let vox = picovdbHDDAPosToVoxel(pos, dim);

    (*hdda).dim = dim;
    (*hdda).tmin = tmin;
    (*hdda).tmax = tmax;
    (*hdda).voxel = vox;
    (*hdda).step = vec3i(sign(direction));
    (*hdda).delta = abs(f32(dim) * direction_inv); // Pre-multiply delta by dim

    let boundary = select(vec3f(vox), vec3f(vox + vec3i(dim)), direction > vec3f(0.0));

    // Safety: handle cases where direction is 0 to avoid NaNs
    (*hdda).next = select(
        tmin + (boundary - pos) * direction_inv,
        vec3f(PICOVDB_HDDA_FLOAT_MAX),
        direction == vec3f(0.0)
    );
}

// Update HDDA to switch hierarchical level
fn picovdbHDDAUpdate(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    dim: i32,
    direction: vec3f,
    direction_inv: vec3f,
) {
    (*hdda).dim = dim;
    (*hdda).delta = abs(f32(dim) * direction_inv);

    // Re-calculate position at the exact current tmin plus a safety nudge
    let eps = max(1e-4f, (*hdda).tmin * 1e-7f);
    let pos = origin + direction * ((*hdda).tmin + eps);

    // Crucial: Re-mask the voxel to the new dimension
    (*hdda).voxel = picovdbHDDAPosToVoxel(pos, dim);

    // Re-calculate next boundary for the new grid scale
    let boundary = select(vec3f((*hdda).voxel), vec3f((*hdda).voxel + vec3i(dim)), direction > vec3f(0.0));
    let t_to_boundary = (boundary - (origin + direction * (*hdda).tmin)) * direction_inv;

    // Ensure the next step is always forward
    (*hdda).next = (*hdda).tmin + max(t_to_boundary, vec3f(eps));
}

fn picovdbHDDAStep(hdda: ptr<function, PicoVDBHDDA>) -> bool {
    // Determine which axis has the nearest boundary
    let next = (*hdda).next;
    if (next.x < next.y && next.x < next.z) { // X is smallest
        (*hdda).tmin = (*hdda).next.x;
        (*hdda).next.x += (*hdda).delta.x;
        (*hdda).voxel.x += (*hdda).dim * (*hdda).step.x;
    } else if (next.y < next.z) { // Y is smallest
        (*hdda).tmin = (*hdda).next.y;
        (*hdda).next.y += (*hdda).delta.y;
        (*hdda).voxel.y += (*hdda).dim * (*hdda).step.y;
    } else { // Z is smallest
        (*hdda).tmin = (*hdda).next.z;
        (*hdda).next.z += (*hdda).delta.z;
        (*hdda).voxel.z += (*hdda).dim * (*hdda).step.z;
    }
    return (*hdda).tmin <= (*hdda).tmax;
}

// Clip ray to bounding box
fn picovdbHDDARayClip(
    bbox_min: vec3f,
    bbox_max: vec3f,
    origin: vec3f,
    tmin: ptr<function, f32>,
    dir_inv: vec3f,
    tmax: ptr<function, f32>
) -> bool {
    let t0 = (bbox_min - origin) * dir_inv;
    let t1 = (bbox_max - origin) * dir_inv;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tnear = max(tmin3.x, max(tmin3.y, tmin3.z));
    let tfar = min(tmax3.x, min(tmax3.y, tmax3.z));
    let hit = tnear <= tfar;
    *tmin = max(*tmin, tnear);
    *tmax = min(*tmax, tfar);
    return hit;
}

// Dimension based on level (for HDDA stepping)
// Level 0 (Leaf) -> 1
// Level 1 (Lower) -> 8 (2^3)
// Level 2 (Upper) -> 128 (2^7)
// Level 3 (Root) -> 4096 (2^12)
const picovdbDimForLevel = array(1, 8, 128, 4096);

// Check if voxel is active (count > 1 means has value)
fn picovdbIsActive(level_count: PicoVDBLevelCount) -> bool {
    return level_count.count > 1u;
}
// Get float value from data buffer using grid offset and value index
fn picovdbGetValue(grid: PicoVDBGrid, count: u32) -> f32 {
    // dataStart is in 16-byte units, multiply by 4 to get u32 index (16 bytes = 4 u32s)
    let u32Index = grid.dataStart * 4u + count;
    return bitcast<f32>(picovdb_buffer[u32Index]);
}

// Zero-crossing detection for level set raymarching
fn picovdbHDDAZeroCrossing(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    pixel_radius: f32,
    out_distance: ptr<function, f32>,
    out_normal: ptr<function, vec3f>,
    out_iterations: ptr<function, u32>,
) -> bool {
    let direction_inv = 1 / direction;
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
        *out_iterations = 0u;
        return false;
    }

    // Get initial hierarchy level
    let start_pos = origin + direction * tmin_mut;
    let res0 = picovdbReadAccessorGetLevelCount(acc, vec3i(floor(start_pos)), grid);
    let v0 = picovdbGetValue(grid, res0.count);

    var hdda: PicoVDBHDDA;
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, direction_inv, picovdbDimForLevel[res0.level]);

    var step_count = 0u;
    for (var i = 0; i < 512; i++) { // Fixed loop limit for GPU safety
        step_count += 1u;
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        // If hierarchy changed, update HDDA and re-read
        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
            continue; // Re-evaluate with the new aligned voxel
        }

        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if ((val <= 0.0) != (v0 <= 0.0)) {
                let cone_radius = hdda.tmin * pixel_radius;
                if (cone_radius < 0.5) {
                    // Voxel projects larger than a pixel — use analytical cubic solver
                    // for smooth, sub-voxel accurate intersection.
                    let stencil = picovdbSampleStencil(acc, grid, hdda.voxel);
                    let o_local = origin + direction * hdda.tmin - vec3f(hdda.voxel);
                    let t_exit = min(min(hdda.next.x, hdda.next.y), hdda.next.z) - hdda.tmin;
                    let hit = picovdbVoxelIntersect(o_local, direction, t_exit, stencil);
                    if (hit.hit) {
                        *out_distance = hdda.tmin + hit.t;
                        *out_normal = hit.normal;
                        *out_iterations = step_count;
                        return true;
                    }
                } else {
                    let stencil = picovdbSampleStencil(acc, grid, hdda.voxel);
                    let p_local = fract(origin + direction * hdda.tmin);
                    *out_distance = hdda.tmin;
                    *out_normal = picovdbTrilinearGradient(p_local, stencil);
                    *out_iterations = step_count;
                    return true;
                }
            }
        }
        // Step to next boundary
        if (!picovdbHDDAStep(&hdda)) {
            break;
        }
    }
    *out_iterations = step_count;
    return false;
}

struct PicoVDBStencil {
    v000: f32, v001: f32, v010: f32, v011: f32,
    v100: f32, v101: f32, v110: f32, v111: f32,
}

// Sample 2x2x2 stencil of voxel values around a point
fn picovdbSampleStencil(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    ijk: vec3i
) -> PicoVDBStencil {
    var s: PicoVDBStencil;
    // Plane Z=0
    s.v000 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 0), grid).count);
    s.v100 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 0), grid).count);
    s.v010 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 0), grid).count);
    s.v110 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 0), grid).count);
    // Plane Z=1
    s.v001 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 1), grid).count);
    s.v101 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 1), grid).count);
    s.v011 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 1), grid).count);
    s.v111 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 1), grid).count);
    return s;
}

// Compute trilinear gradient from 2x2x2 stencil
fn picovdbTrilinearGradient(uvw: vec3f, s: PicoVDBStencil) -> vec3f {
    // Interpolate values along Z for the four XY columns
    let v00z = mix(s.v000, s.v001, uvw.z);
    let v01z = mix(s.v010, s.v011, uvw.z);
    let v10z = mix(s.v100, s.v101, uvw.z);
    let v11z = mix(s.v110, s.v111, uvw.z);

    // Interpolate values along Y for the two X slabs
    let v0yz = mix(v00z, v01z, uvw.y);
    let v1yz = mix(v10z, v11z, uvw.y);

    // X Gradient: Difference between the two YZ-interpolated slabs
    let grad_x = v1yz - v0yz;

    // Y Gradient: Interpolate the differences along X
    let grad_y = mix(v01z - v00z, v11z - v10z, uvw.x);

    // Z Gradient: Interpolate the differences along X and Y
    let dZ00 = s.v001 - s.v000;
    let dZ01 = s.v011 - s.v010;
    let dZ10 = s.v101 - s.v100;
    let dZ11 = s.v111 - s.v110;
    let grad_z = mix(mix(dZ00, dZ01, uvw.y), mix(dZ10, dZ11, uvw.y), uvw.x);

    return vec3f(grad_x, grad_y, grad_z);
}

// Trilinear interpolation of a value at position uvw within a voxel stencil
fn picovdbTrilinearInterpolation(uvw: vec3f, s: PicoVDBStencil) -> f32 {
    // Interpolate along Z
    let v00 = mix(s.v000, s.v001, uvw.z);
    let v01 = mix(s.v010, s.v011, uvw.z);
    let v10 = mix(s.v100, s.v101, uvw.z);
    let v11 = mix(s.v110, s.v111, uvw.z);
    // Interpolate along Y
    let v0 = mix(v00, v01, uvw.y);
    let v1 = mix(v10, v11, uvw.y);
    // Interpolate along X
    return mix(v0, v1, uvw.x);
}

fn picovdbSampleTrilinear(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    pos: vec3f
) -> f32 {
    let ijk = vec3i(floor(pos));
    let uvw = fract(pos);
    let s = picovdbSampleStencil(acc, grid, ijk);
    return picovdbTrilinearInterpolation(uvw, s);
}

fn picovdbHDDAIsOccluded(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32
) -> bool {
    let direction_inv = 1 / direction;
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction_inv, &tmax_mut)) {
        return false;
    }

    var hdda: PicoVDBHDDA;
    // Shadow rays can start at a coarser level (e.g., dim 8) to skip empty space faster
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, direction_inv, 8);

    for (var i = 0; i < 256; i++) { // Shadow rays usually need fewer steps
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, target_dim, direction, direction_inv);
            continue;
        }

        // For SDFs, active leaf usually means "near or inside the surface"
        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if (val <= 0.0) { return true; } // Inside or on the surface
        }

        if (!picovdbHDDAStep(&hdda)) { break; }
    }
    return false;
}

// ============================================================================
// Analytical Ray–Voxel Intersection for Trilinearly Interpolated SDF Grids
//
// Based on: Hansson-Söderlund, Evans, Akenine-Möller,
//   "Ray Tracing of Signed Distance Function Grids", JCGT 2022
//   https://jcgt.org/published/0011/03/06/
//
// Given the 2x2x2 SDF stencil at a voxel's corners, trilinear interpolation
// defines a cubic implicit surface f(x,y,z) = 0. Substituting the ray
// parametrically yields a cubic polynomial in t: c3*t^3 + c2*t^2 + c1*t + c0 = 0.
// We use Marmitt's interval splitting (via the derivative roots) to isolate
// monotonic sub-intervals, then Newton-Raphson to refine the root.
// ============================================================================

struct PicoVDBVoxelHit {
    hit:    bool,
    t:      f32,     // parametric distance in voxel-local space
    uvw:    vec3f,   // hit position in voxel-local [0,1]^3
    normal: vec3f,   // analytic surface normal
}

// Newton-Raphson refinement within a monotonic interval.
// 3 fixed iterations with regula falsi seed — no convergence branch.
fn picovdbSolveNewton(
    c: vec4f,
    t_start: f32, t_end: f32,
    g_start: f32, g_end: f32,
    o: vec3f, d: vec3f,
    stencil: PicoVDBStencil,
) -> PicoVDBVoxelHit {
    // Regula falsi initial guess
    var t = (g_end * t_start - g_start * t_end) / (g_end - g_start);

    // 3 NR iterations — quadratic convergence from a good initial guess
    // means ~12 digits of precision, well beyond f32's ~7.
    for (var i = 0; i < 3; i++) {
        let gt  = ((c.w * t + c.z) * t + c.y) * t + c.x;
        let gdt = (3.0 * c.w * t + 2.0 * c.z) * t + c.y;
        // Guard: if derivative is near zero (tangential graze), stop.
        // Without this, t can fly to infinity and corrupt the result.
        if (abs(gdt) < 1e-10) { break; }
        t -= gt / gdt;
    }

    t = clamp(t, t_start, t_end);
    let uvw = o + t * d;
    return PicoVDBVoxelHit(
        true,
        t,
        uvw,
        picovdbTrilinearGradient(uvw, stencil),  // unnormalized
    );
}

fn picovdbVoxelIntersect(
    o:       vec3f,
    d:       vec3f,
    t_far:   f32,
    stencil: PicoVDBStencil,
) -> PicoVDBVoxelHit {
    var result: PicoVDBVoxelHit;
    result.hit = false;

    // --- k-coefficients (Equation 3) ---
    let k0 = stencil.v000;
    let k1 = stencil.v100 - stencil.v000;
    let k2 = stencil.v010 - stencil.v000;
    let a  = stencil.v101 - stencil.v001;
    let k3 = stencil.v110 - stencil.v010 - k1;
    let k4 = k0 - stencil.v001;
    let k5 = k1 - a;
    let k6 = k2 - (stencil.v011 - stencil.v001);
    let k7 = k3 - (stencil.v111 - stencil.v011 - a);

    // --- m-intermediates (Equation 7) ---
    let m0 = o.x * o.y;
    let m1 = d.x * d.y;
    let m2 = o.x * d.y + o.y * d.x;
    let m3 = k5 * o.z - k1;
    let m4 = k6 * o.z - k2;
    let m5 = k7 * o.z - k3;

    // --- Cubic coefficients c3*t^3 + c2*t^2 + c1*t + c0 = 0 (Equation 6) ---
    // Packed as vec4f(c0, c1, c2, c3).
    // c.x == trilinear value at ray origin (t=0), proven algebraically.
    let c = vec4f(
        (k4 * o.z - k0) + o.x * m3 + o.y * m4 + m0 * m5,
        d.x * m3 + d.y * m4 + m2 * m5 + d.z * (k4 + k5 * o.x + k6 * o.y + k7 * m0),
        m1 * m5 + d.z * (k5 * d.x + k6 * d.y + k7 * m2),
        k7 * m1 * d.z,
    );

    // --- Solid voxel test (Section 2) ---
    // NOTE: c.x = -f(o) due to Equation 2's sign convention:
    //   f = z*(k4+...) - (k0+...), so c0 = -k0 - k1*ox - ... + oz*(k4+...)
    // which equals -f(ox,oy,oz). Therefore c.x > 0 means f(o) < 0 (inside).
    if (c.x > 0.0) {
        return PicoVDBVoxelHit(
            true, 0.0, o,
            picovdbTrilinearGradient(o, stencil),
        );
    }

    // --- Derivative roots for Marmitt interval splitting ---
    // g'(t) = 3*c3*t^2 + 2*c2*t + c1. Roots split [0, t_far] into
    // monotonic sub-intervals. Solved inline, no function call overhead.
    let qA = 3.0 * c.w;
    let qB = 2.0 * c.z;
    let qC = c.y;

    // Default: roots outside range (effectively ignored in interval checks)
    var r0 = -1.0;
    var r1 = -1.0;

    if (abs(qA) > 1e-8) {
        let disc = qB * qB - 4.0 * qA * qC;
        if (disc >= 0.0) {
            let inv2A = 0.5 / qA;
            let sqrtDisc = sqrt(disc);
            r0 = (-qB - sqrtDisc) * inv2A;
            r1 = (-qB + sqrtDisc) * inv2A;
        }
    } else if (abs(qB) > 1e-8) {
        r0 = -qC / qB;
    }

    // --- Unrolled interval checking ---
    // Up to 3 intervals: [0, r0], [r0, r1], [last_boundary, t_far]
    // Walk front-to-back, return at first sign change.
    var t_start = 0.0;
    var g_start = c.x;  // Already computed, reuse

    // Interval 1: [0, r0]
    if (r0 > 0.0 && r0 < t_far) {
        let g_r0 = ((c.w * r0 + c.z) * r0 + c.y) * r0 + c.x;
        if (g_start * g_r0 <= 0.0) {
            return picovdbSolveNewton(c, t_start, r0, g_start, g_r0, o, d, stencil);
        }
        t_start = r0;
        g_start = g_r0;
    }

    // Interval 2: [r0, r1]
    if (r1 > t_start && r1 < t_far) {
        let g_r1 = ((c.w * r1 + c.z) * r1 + c.y) * r1 + c.x;
        if (g_start * g_r1 <= 0.0) {
            return picovdbSolveNewton(c, t_start, r1, g_start, g_r1, o, d, stencil);
        }
        t_start = r1;
        g_start = g_r1;
    }

    // Interval 3: [last_boundary, t_far]
    let g_far = ((c.w * t_far + c.z) * t_far + c.y) * t_far + c.x;
    if ((g_start <= 0.0) != (g_far <= 0.0)) {
        return picovdbSolveNewton(c, t_start, t_far, g_start, g_far, o, d, stencil);
    }

    return result;
}
