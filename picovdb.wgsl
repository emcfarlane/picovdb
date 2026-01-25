
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
        return PicoVDBLevelCount(4u, 0u);
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

// Initialize HDDA for hierarchical grid traversal
fn picovdbHDDAInit(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    tmin: f32,
    direction: vec3f,
    tmax: f32,
    dim: i32
) {
    (*hdda).dim = dim;
    (*hdda).tmin = tmin;
    (*hdda).tmax = tmax;

    let pos = picovdbHDDARayStart(origin, tmin, direction);
    let dir_inv = 1.0 / direction;

    (*hdda).voxel = picovdbHDDAPosToVoxel(pos, dim);

    // X axis
    if (direction.x == 0.0) {
        (*hdda).next.x = PICOVDB_HDDA_FLOAT_MAX;
        (*hdda).step.x = 0;
        (*hdda).delta.x = 0.0;
    } else if (dir_inv.x > 0.0) {
        (*hdda).step.x = 1;
        (*hdda).next.x = (*hdda).tmin + (f32((*hdda).voxel.x) + f32(dim) - pos.x) * dir_inv.x;
        (*hdda).delta.x = dir_inv.x;
    } else {
        (*hdda).step.x = -1;
        (*hdda).next.x = (*hdda).tmin + (f32((*hdda).voxel.x) - pos.x) * dir_inv.x;
        (*hdda).delta.x = -dir_inv.x;
    }

    // Y axis
    if (direction.y == 0.0) {
        (*hdda).next.y = PICOVDB_HDDA_FLOAT_MAX;
        (*hdda).step.y = 0;
        (*hdda).delta.y = 0.0;
    } else if (dir_inv.y > 0.0) {
        (*hdda).step.y = 1;
        (*hdda).next.y = (*hdda).tmin + (f32((*hdda).voxel.y) + f32(dim) - pos.y) * dir_inv.y;
        (*hdda).delta.y = dir_inv.y;
    } else {
        (*hdda).step.y = -1;
        (*hdda).next.y = (*hdda).tmin + (f32((*hdda).voxel.y) - pos.y) * dir_inv.y;
        (*hdda).delta.y = -dir_inv.y;
    }

    // Z axis
    if (direction.z == 0.0) {
        (*hdda).next.z = PICOVDB_HDDA_FLOAT_MAX;
        (*hdda).step.z = 0;
        (*hdda).delta.z = 0.0;
    } else if (dir_inv.z > 0.0) {
        (*hdda).step.z = 1;
        (*hdda).next.z = (*hdda).tmin + (f32((*hdda).voxel.z) + f32(dim) - pos.z) * dir_inv.z;
        (*hdda).delta.z = dir_inv.z;
    } else {
        (*hdda).step.z = -1;
        (*hdda).next.z = (*hdda).tmin + (f32((*hdda).voxel.z) - pos.z) * dir_inv.z;
        (*hdda).delta.z = -dir_inv.z;
    }
}

// Update HDDA to switch hierarchical level
fn picovdbHDDAUpdate(
    hdda: ptr<function, PicoVDBHDDA>,
    origin: vec3f,
    direction: vec3f,
    dim: i32
) -> bool {
    if ((*hdda).dim == dim) {
        return false;
    }
    (*hdda).dim = dim;

    let pos = picovdbHDDARayStart(origin, (*hdda).tmin, direction);
    let dir_inv = 1.0 / direction;

    (*hdda).voxel = picovdbHDDAPosToVoxel(pos, dim);

    if ((*hdda).step.x != 0) {
        (*hdda).next.x = (*hdda).tmin + (f32((*hdda).voxel.x) - pos.x) * dir_inv.x;
        if ((*hdda).step.x > 0) {
            (*hdda).next.x += f32(dim) * dir_inv.x;
        }
    }
    if ((*hdda).step.y != 0) {
        (*hdda).next.y = (*hdda).tmin + (f32((*hdda).voxel.y) - pos.y) * dir_inv.y;
        if ((*hdda).step.y > 0) {
            (*hdda).next.y += f32(dim) * dir_inv.y;
        }
    }
    if ((*hdda).step.z != 0) {
        (*hdda).next.z = (*hdda).tmin + (f32((*hdda).voxel.z) - pos.z) * dir_inv.z;
        if ((*hdda).step.z > 0) {
            (*hdda).next.z += f32(dim) * dir_inv.z;
        }
    }

    return true;
}

// Step to next voxel boundary
fn picovdbHDDAStep(hdda: ptr<function, PicoVDBHDDA>) -> bool {
    var ret: bool;
    if ((*hdda).next.x < (*hdda).next.y && (*hdda).next.x < (*hdda).next.z) {
        // Enforce forward stepping
        if ((*hdda).next.x <= (*hdda).tmin) {
            (*hdda).next.x += (*hdda).tmin - 0.999999 * (*hdda).next.x + 1.0e-6;
        }
        (*hdda).tmin = (*hdda).next.x;
        (*hdda).next.x += f32((*hdda).dim) * (*hdda).delta.x;
        (*hdda).voxel.x += (*hdda).dim * (*hdda).step.x;
        ret = (*hdda).tmin <= (*hdda).tmax;
    } else if ((*hdda).next.y < (*hdda).next.z) {
        // Enforce forward stepping
        if ((*hdda).next.y <= (*hdda).tmin) {
            (*hdda).next.y += (*hdda).tmin - 0.999999 * (*hdda).next.y + 1.0e-6;
        }
        (*hdda).tmin = (*hdda).next.y;
        (*hdda).next.y += f32((*hdda).dim) * (*hdda).delta.y;
        (*hdda).voxel.y += (*hdda).dim * (*hdda).step.y;
        ret = (*hdda).tmin <= (*hdda).tmax;
    } else {
        // Enforce forward stepping
        if ((*hdda).next.z <= (*hdda).tmin) {
            (*hdda).next.z += (*hdda).tmin - 0.999999 * (*hdda).next.z + 1.0e-6;
        }
        (*hdda).tmin = (*hdda).next.z;
        (*hdda).next.z += f32((*hdda).dim) * (*hdda).delta.z;
        (*hdda).voxel.z += (*hdda).dim * (*hdda).step.z;
        ret = (*hdda).tmin <= (*hdda).tmax;
    }
    return ret;
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

// Get dimension based on level (for HDDA stepping)
fn picovdbGetDimForLevel(level: u32) -> i32 {
    switch (level) {
        case 0u: { return 1; }      // Leaf (8^3)
        case 1u: { return 8; }      // Lower (16^3, step by 8)
        case 2u: { return 128; }    // Upper (32^3, step by 128)
        default: { return 4096; }   // Root/background
    }
}

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
    let bbox_minf = vec3f(grid.indexBoundsMin);
    let bbox_maxf = vec3f(grid.indexBoundsMax + vec3i(1));

    var tmin_mut = tmin;
    var tmax_mut = tmax;
    let hit = picovdbHDDARayClip(bbox_minf, bbox_maxf, origin, &tmin_mut, direction, &tmax_mut);
    
    if (!hit || tmax_mut > 1.0e20) {
        return false;
    }

    let pos = picovdbHDDARayStart(origin, tmin_mut, direction);
    var ijk = picovdbHDDAPosToIjk(pos);

    // Get initial value
    let result0 = picovdbReadAccessorGetLevelCount(acc, ijk, grid);
    let v0 = picovdbGetValue(grid, result0.count);
    let dim = picovdbGetDimForLevel(result0.level);
    
    var hdda: PicoVDBHDDA;
    picovdbHDDAInit(&hdda, origin, tmin_mut, direction, tmax_mut, dim);

    var outer_loop_count = 0;
    while (picovdbHDDAStep(&hdda) && outer_loop_count < 1000) {
        outer_loop_count++;
        
        let pos_start = picovdbHDDARayStart(origin, hdda.tmin + 1.0001, direction);
        ijk = picovdbHDDAPosToIjk(pos_start);
        
        let result = picovdbReadAccessorGetLevelCount(acc, ijk, grid);
        let new_dim = picovdbGetDimForLevel(result.level);
        let updated = picovdbHDDAUpdate(&hdda, origin, direction, new_dim);
        
        // Skip if not at leaf level or not active
        if (hdda.dim > 1 || !picovdbIsActive(result)) {
            continue;
        }
        
        // Inner loop: step through active leaf voxels
        var inner_loop_count = 0;
        while (inner_loop_count < 100) {
            inner_loop_count++;
            
            if (!picovdbHDDAStep(&hdda)) {
                break;
            }
            
            ijk = hdda.voxel;
            let voxel_result = picovdbReadAccessorGetLevelCount(acc, ijk, grid);
            
            if (!picovdbIsActive(voxel_result)) {
                break;
            }
            
            *v = picovdbGetValue(grid, voxel_result.count);
            
            // Check for zero crossing
            if ((*v) * v0 < 0.0) {
                *thit = hdda.tmin;
                return true;
            }
        }
    }
    return false;
}

// Sample 2x2x2 stencil of voxel values around a point
fn picovdbSampleStencil(
    acc: ptr<function, PicoVDBReadAccessor>,
    grid: PicoVDBGrid,
    ijk: vec3i
) -> array<array<array<f32, 2>, 2>, 2> {
    var v: array<array<array<f32, 2>, 2>, 2>;
    for (var x = 0; x < 2; x++) {
        for (var y = 0; y < 2; y++) {
            for (var z = 0; z < 2; z++) {
                let offset = vec3i(x, y, z);
                let result = picovdbReadAccessorGetLevelCount(
                    acc,
                    ijk + offset,
                    grid
                );
                v[x][y][z] = picovdbGetValue(grid, result.count);
            }
        }
    }
    return v;
}

// Compute trilinear gradient from 2x2x2 stencil
fn picovdbTrilinearGradient(
    uvw: vec3f,
    v: array<array<array<f32, 2>, 2>, 2>
) -> vec3f {
    // Compute differences along Z axis for all 4 XY corners
    var D: array<f32, 4>;
    D[0] = v[0][0][1] - v[0][0][0];
    D[1] = v[0][1][1] - v[0][1][0];
    D[2] = v[1][0][1] - v[1][0][0];
    D[3] = v[1][1][1] - v[1][1][0];

    // Z component: interpolate the Z differences
    let grad_z = mix(mix(D[0], D[1], uvw.y), mix(D[2], D[3], uvw.y), uvw.x);

    // Interpolate along Z to get 4 values at the correct Z position
    let w = uvw.z;
    D[0] = v[0][0][0] + D[0] * w;
    D[1] = v[0][1][0] + D[1] * w;
    D[2] = v[1][0][0] + D[2] * w;
    D[3] = v[1][1][0] + D[3] * w;

    // X component: difference between interpolated X edges
    let grad_x = mix(D[2], D[3], uvw.y) - mix(D[0], D[1], uvw.y);

    // Y component: difference between interpolated Y edges
    let grad_y = mix(D[1] - D[0], D[3] - D[2], uvw.x);

    return vec3f(grad_x, grad_y, grad_z);
}

// Trilinear interpolation of a value at position uvw within a voxel stencil
fn picovdbTrilinearInterpolation(
    uvw: vec3f,
    v: array<array<array<f32, 2>, 2>, 2>
) -> f32 {
    // Interpolate along Z
    let v00 = mix(v[0][0][0], v[0][0][1], uvw.z);
    let v01 = mix(v[0][1][0], v[0][1][1], uvw.z);
    let v10 = mix(v[1][0][0], v[1][0][1], uvw.z);
    let v11 = mix(v[1][1][0], v[1][1][1], uvw.z);
    
    // Interpolate along Y
    let v0 = mix(v00, v01, uvw.y);
    let v1 = mix(v10, v11, uvw.y);
    
    // Interpolate along X
    return mix(v0, v1, uvw.x);
}
