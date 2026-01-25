
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
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
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

    if (is_value && is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).leaf = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLeafGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
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

    if (is_value && is_inside) {
        let preceding_bits = extractBits(mask.value & mask.inside, 0u, bit_index);
        (*acc).lower = mask.childOffset + countOneBits(preceding_bits);
        (*acc).key = ijk;
        return picovdbReadAccessorLowerGetLevelCountAndCache(acc, ijk, grid);
    }
    let preceding_bits = extractBits(mask.value & ~mask.inside, 0u, bit_index);
    let count = select(
        u32(is_inside),
        mask.valueOffset + countOneBits(preceding_bits),
        is_value,
    );
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
    dim: i32,
    tmin: f32,
    tmax: f32,
    voxel: vec3i,
    step: vec3i,
    inv_dir: vec3f,
    delta: vec3f,
    next: vec3f,
}

fn picovdbHDDAPosToIjk(pos: vec3f) -> vec3i {
    return vec3i(floor(pos));
}

fn picovdbHDDAPosToVoxel(pos: vec3f, dim: i32) -> vec3i {
    let mask = ~(dim - 1);
    return vec3i(floor(pos)) & vec3i(mask);
}

fn picovdbHDDARayStart(origin: vec3f, tmin: f32, direction: vec3f) -> vec3f {
    return origin + direction * tmin;
}

fn picovdbHDDAInit(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    dim: i32
) {
    let dir_inv = 1.0 / direction;
    let pos = origin + direction * tmin;
    let vox = picovdbHDDAPosToVoxel(pos, dim);

    (*hdda).dim = dim;
    (*hdda).tmin = tmin;
    (*hdda).tmax = tmax;
    (*hdda).voxel = vox;
    (*hdda).inv_dir = dir_inv;
    (*hdda).step = vec3i(sign(direction));
    (*hdda).delta = abs(f32(dim) * dir_inv); // Pre-multiply delta by dim

    let boundary = select(vec3f(vox), vec3f(vox + vec3i(dim)), direction > vec3f(0.0));

    // Safety: handle cases where direction is 0 to avoid NaNs
    (*hdda).next = select(
        tmin + (boundary - pos) * dir_inv,
        vec3f(PICOVDB_HDDA_FLOAT_MAX),
        direction == vec3f(0.0)
    );
}

// Update HDDA to switch hierarchical level
fn picovdbHDDAUpdate(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    direction: vec3f,
    dim: i32
) {
    (*hdda).dim = dim;
    (*hdda).delta = abs(f32(dim) * (*hdda).inv_dir);

    // Re-calculate position at the exact current tmin plus a safety nudge
    let eps = max(1e-4f, (*hdda).tmin * 1e-7f);
    let pos = origin + direction * ((*hdda).tmin + eps);

    // Crucial: Re-mask the voxel to the new dimension
    (*hdda).voxel = picovdbHDDAPosToVoxel(pos, dim);

    // Re-calculate next boundary for the new grid scale
    let boundary = select(vec3f((*hdda).voxel), vec3f((*hdda).voxel + vec3i(dim)), direction > vec3f(0.0));
    let t_to_boundary = (boundary - (origin + direction * (*hdda).tmin)) * (*hdda).inv_dir;

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
    direction: vec3f,
    tmax: ptr<function, f32>
) -> bool {
    let dir_inv = 1.0 / direction;
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
    thit: ptr<function, f32>,
    v: ptr<function, f32>
) -> bool {
    var tmin_mut = tmin;
    var tmax_mut = tmax;
    
    // Clip to bounding box
    if (!picovdbHDDARayClip(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax + vec3i(1)), origin, &tmin_mut, direction, &tmax_mut)) {
        return false;
    }

    // Get initial hierarchy level
    let start_pos = origin + direction * tmin_mut;
    let res0 = picovdbReadAccessorGetLevelCount(acc, vec3i(floor(start_pos)), grid);
    let v0 = picovdbGetValue(grid, res0.count);
    
    var hdda: PicoVDBHDDA;
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, picovdbDimForLevel[res0.level]);

    for (var i = 0; i < 512; i++) { // Fixed loop limit for GPU safety
        let result = picovdbReadAccessorGetLevelCount(acc, hdda.voxel, grid);
        let target_dim = picovdbDimForLevel[result.level];

        // If hierarchy changed, update HDDA and re-read
        if (hdda.dim != target_dim) {
            picovdbHDDAUpdate(&hdda, origin, direction, target_dim);
            continue; // Re-evaluate with the new aligned voxel
        }

        // Leaf level logic (Surface detection)
        if (hdda.dim == 1 && picovdbIsActive(result)) {
            let val = picovdbGetValue(grid, result.count);
            if (val * v0 < 0.0) {
                *thit = hdda.tmin;
                *v = val;
                return true;
            }
            // Optional: v0 = val; // Update previous value for continuous crossing
        }

        // Step to next boundary
        if (!picovdbHDDAStep(&hdda)) { break; }
    }
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
    s.v000 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 0), grid).count);
    s.v001 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 0, 1), grid).count);
    s.v010 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 0), grid).count);
    s.v011 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(0, 1, 1), grid).count);
    s.v100 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 0), grid).count);
    s.v101 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 0, 1), grid).count);
    s.v110 = picovdbGetValue(grid, picovdbReadAccessorGetLevelCount(acc, ijk + vec3i(1, 1, 0), grid).count);
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
