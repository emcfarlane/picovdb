const std = @import("std");
const testing = std.testing;

// PicoVDB format structures - matching WGSL and TypeScript definitions
// All fields are 32-bit for WebGPU compatibility with proper alignment

// "PicoVDB0" in hex little endian
const PICOVDB_MAGIC: [2]u32 = .{ 0x6f636950, 0x30424456 };

// Result structure for getValue operations
pub const AddressAndLevel = struct {
    address: u32, // Offset into data buffer
    level: u32, // Tree level (0=leaf, 1=lower, 2=upper, 3=root tile, 4=background)
};

// File header (32 bytes)
pub const PicoVDBFileHeader = extern struct {
    magic: [2]u32, // 'PicoVDB0' little endian (8 bytes)
    version: u32, // Format version (4 bytes)
    grid_count: u32, // Number of grids (4 bytes)
    upper_count: u32, // Total upper nodes (= root count) (4 bytes)
    lower_count: u32, // Total lower nodes (4 bytes)
    leaf_count: u32, // Total leaf nodes (4 bytes)
    data_count: u32, // Total data buffer size in 16-byte units (4 bytes)
};

// Grid constants
pub const GRID_TYPE_SDF_FLOAT = 1;
pub const GRID_TYPE_SDF_UINT8 = 2;

// Grid header (64 bytes)
pub const PicoVDBGrid = extern struct {
    grid_index: u32, // This grid's index (4 bytes)
    upper_start: u32, // Index into uppers array (= root index) (4 bytes)
    lower_start: u32, // Index into lowers array (4 bytes)
    leaf_start: u32, // Index into leaves array (4 bytes)
    data_start: u32, // 16-byte index into data buffer (4 bytes)
    data_elem_count: u32, // Number of data elements for this grid (4 bytes)
    grid_type: u32, // GRID_TYPE_SDF_FLOAT=1, GRID_TYPE_SDF_UINT8=2 (4 bytes)
    _pad1: u32,
    index_bounds_min: [3]i32, // Index spaac bounding box min (12 bytes)
    _pad2: u32,
    index_bounds_max: [3]i32, // Index space bounding box max (12 bytes)
    _pad3: u32, // Padding to 64 bytes (4 bytes)
};

// Root tile (8 bytes)
// Always stored as an even count for alignment to 16 bytes.
pub const PicoVDBRoot = extern struct {
    key: [2]u32, // 64-bit coordinate key (8 bytes)
};

// Node mask structure (16 bytes)
// Encoding: inside=0,value=0 -> outside implicit (background at index 0)
//           inside=0,value=1 -> stored value (explicit tile)
//           inside=1,value=0 -> inside implicit (inside value at index 1)
//           inside=1,value=1 -> child node reference
pub const PicoVDBNodeMask = extern struct {
    inside: u32, // Inside mask bits (4 bytes)
    value: u32, // Value mask bits (4 bytes)
    value_offset: u32, // Prefix sum offset of values (4 bytes)
    child_offset: u32, // Prefix sum offset of children (4 bytes)

    pub inline fn isValue(self: PicoVDBNodeMask, index: u5) bool {
        return (self.value >> index) & 1 != 0;
    }
    pub inline fn isInside(self: PicoVDBNodeMask, index: u5) bool {
        return (self.inside >> index) & 1 != 0;
    }
    pub inline fn valueIndex(self: PicoVDBNodeMask, index: u5) u32 {
        const value_mask = self.value & ~self.inside;
        return rankQuery(value_mask, self.value_offset, index);
    }
    pub inline fn childIndex(self: PicoVDBNodeMask, index: u5) u32 {
        const child_mask = self.value & self.inside;
        return rankQuery(child_mask, self.child_offset, index);
    }
};

// Leaf mask structure (12 bytes)
// Encoding: inside=0,value=0 -> outside implicit (background at index 0)
//           inside=0,value=1 -> stored value (explicit tile)
//           inside=1,value=0 -> inside implicit (inside value at index 1)
// Note: inside=1,value=1 is not valid for leaves (no children)
pub const PicoVDBLeafMask = extern struct {
    inside: u32, // Inside mask bits (4 bytes)
    value: u32, // Value mask bits (4 bytes)
    value_offset: u32, // Prefix sum offset of values (4 bytes)

    pub inline fn isValue(self: PicoVDBLeafMask, index: u5) bool {
        return (self.value >> index) & 1 != 0;
    }
    pub inline fn isInside(self: PicoVDBLeafMask, index: u5) bool {
        return (self.inside >> index) & 1 != 0;
    }
    pub inline fn valueIndex(self: PicoVDBLeafMask, index: u5) u32 {
        return rankQuery(self.value, self.value_offset, index);
    }
};

// Upper internal node (16384 bytes = 1024 * 16)
pub const PicoVDBUpper = extern struct {
    mask: [1024]PicoVDBNodeMask,
};

// Lower internal node (2048 bytes = 128 * 16)
pub const PicoVDBLower = extern struct {
    mask: [128]PicoVDBNodeMask,
};

// Leaf node (192 bytes = 16 * 12)
pub const PicoVDBLeaf = extern struct {
    mask: [16]PicoVDBLeafMask,
};

// Read accessor (32 bytes)
pub const PicoVDBReadAccessor = extern struct {
    key: [3]i32, // Current coordinate key (12 bytes)
    grid: u32, // Grid index (4 bytes)
    upper: u32, // Upper node index (4 bytes)
    lower: u32, // Lower node index (4 bytes)
    leaf: u32, // Leaf node index (4 bytes)
    _pad: u32, // Padding to 32 bytes (4 bytes)

    const LevelCount = struct {
        // Level of value found.
        level: u32,
        // Count offset. 0 = background, 1 = inside implicit.
        count: u32,
    };

    pub fn init(grid: u32) PicoVDBReadAccessor {
        return PicoVDBReadAccessor{
            .key = [3]i32{ 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF },
            .grid = grid,
            .upper = std.math.maxInt(u32),
            .lower = std.math.maxInt(u32),
            .leaf = std.math.maxInt(u32),
            ._pad = 0,
        };
    }

    // Check if leaf level is cached (dirty bits for 8^3 = 512 voxels)
    fn isCachedLeaf(self: *PicoVDBReadAccessor, dirty: i32) bool {
        const is_cached = (self.leaf != std.math.maxInt(u32)) and ((dirty & ~(@as(i32, (1 << 3) - 1))) == 0);
        self.leaf = if (is_cached) self.leaf else std.math.maxInt(u32);
        return is_cached;
    }

    // Check if lower level is cached (dirty bits for 16^3 = 4096 voxels)
    fn isCachedLower(self: *PicoVDBReadAccessor, dirty: i32) bool {
        const is_cached = (self.lower != std.math.maxInt(u32)) and ((dirty & ~(@as(i32, (1 << 7) - 1))) == 0);
        self.lower = if (is_cached) self.lower else std.math.maxInt(u32);
        return is_cached;
    }

    // Check if upper level is cached (dirty bits for 32^3 = 32768 voxels)
    fn isCachedUpper(self: *PicoVDBReadAccessor, dirty: i32) bool {
        const is_cached = (self.upper != std.math.maxInt(u32)) and ((dirty & ~(@as(i32, (1 << 12) - 1))) == 0);
        self.upper = if (is_cached) self.upper else std.math.maxInt(u32);
        return is_cached;
    }

    // Compute dirty bits by XOR of coordinates
    fn computeDirty(self: *const PicoVDBReadAccessor, ijk: [3]i32) i32 {
        return ((ijk[0] ^ self.key[0]) | (ijk[1] ^ self.key[1]) | (ijk[2] ^ self.key[2]));
    }

    // Find upper/root index for coordinate within current grid bounds.
    // Roots are 1:1 with uppers, so the returned index works for both.
    fn findUpperIndex(ijk: [3]i32, grid: *const PicoVDBGrid, picovdb_file: *const PicoVDBFile) ?u32 {
        const coord_key = coordToKey(ijk);

        // Determine bounds for this grid's root tiles
        const start_index = grid.upper_start;
        const end_index = if (grid.grid_index + 1 < picovdb_file.grids.len)
            picovdb_file.grids[grid.grid_index + 1].upper_start
        else
            @as(u32, @intCast(picovdb_file.roots.len));

        // Search within grid's root tile range
        var i = start_index;
        while (i < end_index) : (i += 1) {
            const root = &picovdb_file.roots[i];
            if (coord_key[0] == root.key[0] and coord_key[1] == root.key[1]) {
                return i;
            }
        }
        return null; // No matching root tile found
    }

    // Get level and count from leaf node and update cache - 1 branch (no children)
    fn leafGetLevelCountAndCache(
        self: *PicoVDBReadAccessor,
        ijk: [3]i32,
        grid: *const PicoVDBGrid,
        picovdb_file: *const PicoVDBFile,
    ) LevelCount {
        const leaf = &picovdb_file.leaves[grid.leaf_start + self.leaf];
        const n = leafCoordToOffset(ijk);
        const word_index = n / 32;
        const bit_index: u5 = @intCast(n % 32);
        const mask = &leaf.mask[word_index];

        const is_value = mask.isValue(bit_index);
        const is_inside = mask.isInside(bit_index);

        if (is_value) {
            self.key = ijk;
            const value_index = mask.valueIndex(bit_index);
            return LevelCount{ .level = 0, .count = value_index };
        }
        return LevelCount{ .level = 0, .count = @intFromBool(is_inside) };
    }

    // Get value and level from lower node and update cache - 2 branches (child, value)
    fn lowerGetLevelCountAndCache(
        self: *PicoVDBReadAccessor,
        ijk: [3]i32,
        grid: *const PicoVDBGrid,
        picovdb_file: *const PicoVDBFile,
    ) LevelCount {
        const lower = &picovdb_file.lowers[grid.lower_start + self.lower];
        const n = lowerCoordToOffset(ijk);
        const word_index = n / 32;
        const bit_index: u5 = @intCast(n % 32);
        const mask = &lower.mask[word_index];

        const is_value = mask.isValue(bit_index);
        const is_inside = mask.isInside(bit_index);

        if (is_value and is_inside) {
            self.leaf = mask.childIndex(bit_index);
            self.key = ijk;
            return self.leafGetLevelCountAndCache(ijk, grid, picovdb_file);
        }
        if (is_value) {
            const value_index = mask.valueIndex(bit_index);
            return LevelCount{ .level = 1, .count = value_index };
        }
        return LevelCount{ .level = 1, .count = @intFromBool(is_inside) };
    }

    // Get level and count from upper node and update cache - 2 branches (child, value)
    fn upperGetLevelCountAndCache(
        self: *PicoVDBReadAccessor,
        ijk: [3]i32,
        grid: *const PicoVDBGrid,
        picovdb_file: *const PicoVDBFile,
    ) LevelCount {
        const upper = &picovdb_file.uppers[grid.upper_start + self.upper];
        const n = upperCoordToOffset(ijk);
        const word_index = n / 32;
        const bit_index: u5 = @intCast(n % 32);
        const mask = &upper.mask[word_index];

        const is_value = mask.isValue(bit_index);
        const is_inside = mask.isInside(bit_index);

        if (is_value and is_inside) {
            self.lower = mask.childIndex(bit_index);
            self.key = ijk;
            return self.lowerGetLevelCountAndCache(ijk, grid, picovdb_file);
        }
        if (is_value) {
            const value_index = mask.valueIndex(bit_index);
            return LevelCount{ .level = 2, .count = value_index };
        }
        return LevelCount{ .level = 2, .count = @intFromBool(is_inside) };
    }

    fn rootGetLevelCountAndCache(
        self: *PicoVDBReadAccessor,
        ijk: [3]i32,
        grid: *const PicoVDBGrid,
        picovdb_file: *const PicoVDBFile,
    ) LevelCount {
        const upper_index = findUpperIndex(ijk, grid, picovdb_file);
        if (upper_index == null) {
            // No matching root tile, return background value
            return LevelCount{ .level = 4, .count = 0 };
        }
        // Roots are 1:1 with uppers, so upper_index works for both
        self.upper = upper_index.?;
        self.key = ijk;
        return self.upperGetLevelCountAndCache(ijk, grid, picovdb_file);
    }

    pub fn getLevelCount(self: *PicoVDBReadAccessor, ijk: [3]i32, grid: *const PicoVDBGrid, picovdb_file: *const PicoVDBFile) LevelCount {
        const dirty = self.computeDirty(ijk);
        if (self.isCachedLeaf(dirty)) {
            return self.leafGetLevelCountAndCache(ijk, grid, picovdb_file);
        } else if (self.isCachedLower(dirty)) {
            return self.lowerGetLevelCountAndCache(ijk, grid, picovdb_file);
        } else if (self.isCachedUpper(dirty)) {
            return self.upperGetLevelCountAndCache(ijk, grid, picovdb_file);
        } else {
            return self.rootGetLevelCountAndCache(ijk, grid, picovdb_file);
        }
    }
};

pub fn getGridFloat(
    picovdb_file: *const PicoVDBFile,
    grid: *const PicoVDBGrid,
    index: u32,
) f32 {
    const data_ptr: [*]const f32 = @ptrCast(@alignCast(picovdb_file.data_buffer.ptr));
    // data_start is in 16-byte units, multiply by 4 to get f32 index (16 bytes = 4 f32s)
    return data_ptr[grid.data_start * 4 + index];
}

// Mutable container for building PicoVDB data
pub const PicoVDBFileMutable = struct {
    header: PicoVDBFileHeader,
    grids: std.ArrayList(PicoVDBGrid),
    roots: std.ArrayList(PicoVDBRoot),
    uppers: std.ArrayList(PicoVDBUpper),
    lowers: std.ArrayList(PicoVDBLower),
    leaves: std.ArrayList(PicoVDBLeaf),
    data_buffer: std.ArrayList(u8),

    pub fn init() PicoVDBFileMutable {
        return PicoVDBFileMutable{
            .header = std.mem.zeroes(PicoVDBFileHeader),
            .grids = .empty,
            .roots = .empty,
            .uppers = .empty,
            .lowers = .empty,
            .leaves = .empty,
            .data_buffer = .empty,
        };
    }

    pub fn deinit(self: *PicoVDBFileMutable, allocator: std.mem.Allocator) void {
        self.grids.deinit(allocator);
        self.roots.deinit(allocator);
        self.uppers.deinit(allocator);
        self.lowers.deinit(allocator);
        self.leaves.deinit(allocator);
        self.data_buffer.deinit(allocator);
    }

    // encode to read-only serialized buffer
    pub fn encode(self: *const PicoVDBFileMutable, allocator: std.mem.Allocator) ![]align(4) const u8 {
        // Calculate sizes with padding
        const root_count = self.roots.items.len;
        const root_count_padded = if (root_count % 2 == 1) root_count + 1 else root_count;
        const data_size = self.data_buffer.items.len;
        const data_size_padded = std.mem.alignForward(usize, data_size, 16);

        const header_size = @sizeOf(PicoVDBFileHeader);
        const grids_size = self.grids.items.len * @sizeOf(PicoVDBGrid);
        const roots_size_padded = root_count_padded * @sizeOf(PicoVDBRoot);
        const uppers_size = self.uppers.items.len * @sizeOf(PicoVDBUpper);
        const lowers_size = self.lowers.items.len * @sizeOf(PicoVDBLower);
        const leaves_size = self.leaves.items.len * @sizeOf(PicoVDBLeaf);

        const total_size = header_size + grids_size + roots_size_padded + uppers_size + lowers_size + leaves_size + data_size_padded;

        // Allocate aligned buffer
        const buffer = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(16), total_size);
        errdefer allocator.free(buffer);

        var offset: usize = 0;

        // Create updated header with final counts
        var header = self.header;
        header.magic = PICOVDB_MAGIC;
        header.grid_count = @intCast(self.grids.items.len);
        header.upper_count = @intCast(self.uppers.items.len);
        header.lower_count = @intCast(self.lowers.items.len);
        header.leaf_count = @intCast(self.leaves.items.len);
        header.data_count = @intCast(data_size_padded / 16); // 16-byte units

        // Copy header
        @memcpy(buffer[offset .. offset + header_size], std.mem.asBytes(&header));
        offset += header_size;

        // Copy grids
        @memcpy(buffer[offset .. offset + grids_size], std.mem.sliceAsBytes(self.grids.items));
        offset += grids_size;

        // Copy roots (with padding for 16-byte alignment)
        const actual_roots_size = root_count * @sizeOf(PicoVDBRoot);
        @memcpy(buffer[offset .. offset + actual_roots_size], std.mem.sliceAsBytes(self.roots.items));
        offset += actual_roots_size;
        if (root_count % 2 == 1) {
            const padding_root = PicoVDBRoot{ .key = [2]u32{ 0, 0 } };
            @memcpy(buffer[offset .. offset + @sizeOf(PicoVDBRoot)], std.mem.asBytes(&padding_root));
            offset += @sizeOf(PicoVDBRoot);
        }

        // Copy uppers
        @memcpy(buffer[offset .. offset + uppers_size], std.mem.sliceAsBytes(self.uppers.items));
        offset += uppers_size;

        // Copy lowers
        @memcpy(buffer[offset .. offset + lowers_size], std.mem.sliceAsBytes(self.lowers.items));
        offset += lowers_size;

        // Copy leaves
        @memcpy(buffer[offset .. offset + leaves_size], std.mem.sliceAsBytes(self.leaves.items));
        offset += leaves_size;

        // Copy data buffer (with padding)
        @memcpy(buffer[offset .. offset + data_size], self.data_buffer.items);
        offset += data_size;
        const data_padding = data_size_padded - data_size;
        if (data_padding > 0) {
            @memset(buffer[offset .. offset + data_padding], 0);
        }

        return buffer;
    }
};

// Read-only container for accessing PicoVDB data from memory buffer
pub const PicoVDBFile = struct {
    header: *const PicoVDBFileHeader,
    grids: []const PicoVDBGrid,
    roots: []const PicoVDBRoot,
    uppers: []const PicoVDBUpper,
    lowers: []const PicoVDBLower,
    leaves: []const PicoVDBLeaf,
    data_buffer: []const u8,

    // Load PicoVDB file from static buffer
    pub fn fromBytes(buffer: []align(4) const u8) error{ BufferTooSmall, InvalidMagic, Misaligned }!PicoVDBFile {
        if (buffer.len < @sizeOf(PicoVDBFileHeader)) {
            return error.BufferTooSmall;
        }

        // Parse header (buffer is aligned to 16, so this is safe)
        const header: *const PicoVDBFileHeader = @ptrCast(buffer.ptr);

        // Verify magic number ('PicoVDB0' little endian)
        if (!std.mem.eql(u32, &header.magic, &PICOVDB_MAGIC)) {
            return error.InvalidMagic;
        }

        var offset: usize = @sizeOf(PicoVDBFileHeader);

        // Parse grids
        const grids_bytes = header.grid_count * @sizeOf(PicoVDBGrid);
        if (offset + grids_bytes > buffer.len) return error.BufferTooSmall;
        std.debug.assert(offset % @alignOf(PicoVDBGrid) == 0);

        const grids_slice: []align(@alignOf(PicoVDBGrid)) const u8 = @alignCast(buffer[offset .. offset + grids_bytes]);
        const grids: []const PicoVDBGrid = std.mem.bytesAsSlice(PicoVDBGrid, grids_slice);
        offset += grids_bytes;

        // Parse roots (upper_count = root_count, padded to even for 16-byte alignment)
        const root_count_padded = if (header.upper_count % 2 == 1) header.upper_count + 1 else header.upper_count;
        const roots_bytes = root_count_padded * @sizeOf(PicoVDBRoot);
        if (offset + roots_bytes > buffer.len) return error.BufferTooSmall;
        std.debug.assert(offset % @alignOf(PicoVDBRoot) == 0);

        const roots_slice: []align(@alignOf(PicoVDBRoot)) const u8 = @alignCast(buffer[offset .. offset + roots_bytes]);
        const roots_all: []const PicoVDBRoot = std.mem.bytesAsSlice(PicoVDBRoot, roots_slice);
        const roots = roots_all[0..header.upper_count]; // Exclude padding root
        offset += roots_bytes;

        // Parse uppers
        const uppers_bytes = header.upper_count * @sizeOf(PicoVDBUpper);
        if (offset + uppers_bytes > buffer.len) return error.BufferTooSmall;
        std.debug.assert(offset % @alignOf(PicoVDBUpper) == 0);

        const uppers_slice: []align(@alignOf(PicoVDBUpper)) const u8 = @alignCast(buffer[offset .. offset + uppers_bytes]);
        const uppers: []const PicoVDBUpper = std.mem.bytesAsSlice(PicoVDBUpper, uppers_slice);
        offset += uppers_bytes;

        // Parse lowers
        const lowers_bytes = header.lower_count * @sizeOf(PicoVDBLower);
        if (offset + lowers_bytes > buffer.len) return error.BufferTooSmall;
        std.debug.assert(offset % @alignOf(PicoVDBLower) == 0);

        const lowers_slice: []align(@alignOf(PicoVDBLower)) const u8 = @alignCast(buffer[offset .. offset + lowers_bytes]);
        const lowers: []const PicoVDBLower = std.mem.bytesAsSlice(PicoVDBLower, lowers_slice);
        offset += lowers_bytes;

        // Parse leaves
        const leaves_bytes = header.leaf_count * @sizeOf(PicoVDBLeaf);
        if (offset + leaves_bytes > buffer.len) return error.BufferTooSmall;
        std.debug.assert(offset % @alignOf(PicoVDBLeaf) == 0);

        const leaves_slice: []align(@alignOf(PicoVDBLeaf)) const u8 = @alignCast(buffer[offset .. offset + leaves_bytes]);
        const leaves: []const PicoVDBLeaf = std.mem.bytesAsSlice(PicoVDBLeaf, leaves_slice);
        offset += leaves_bytes;

        // Parse data (data_count is in 16-byte units)
        const data_size_bytes = header.data_count * 16;
        if (offset + data_size_bytes > buffer.len) return error.BufferTooSmall;
        const data_buffer = buffer[offset .. offset + data_size_bytes];

        return PicoVDBFile{
            .header = header,
            .grids = grids,
            .roots = roots,
            .uppers = uppers,
            .lowers = lowers,
            .leaves = leaves,
            .data_buffer = data_buffer,
        };
    }

    // Get float value from data buffer
    pub fn getGridFloat(self: *const PicoVDBFile, grid: *const PicoVDBGrid, index: u32) f32 {
        const data_ptr: [*]const f32 = @ptrCast(@alignCast(self.data_buffer.ptr));
        // data_start is in 16-byte units, multiply by 4 to get f32 index (16 bytes = 4 f32s)
        return data_ptr[grid.data_start * 4 + index];
    }

    // Access grid by index
    pub fn getGrid(self: *const PicoVDBFile, index: u32) ?*const PicoVDBGrid {
        if (index >= self.grids.len) return null;
        return &self.grids[index];
    }
};

// Rank query implementation
// Returns the number of set bits preceding the specified bit
// value: the mask word containing the target bit
// count: the cumulative count up to the start of this word
// bit_index: the bit position within the word (0-31)
pub fn rankQuery(value: u32, count: u32, bit_index: u5) u32 {
    const bit_mask = (@as(u32, 1) << bit_index) - 1;
    return count + @popCount(value & bit_mask);
}

// Convert coordinate to 64-bit key (matches WGSL pnanovdb_coord_to_key)
pub fn coordToKey(ijk: [3]i32) [2]u32 {
    const iu = @as(u32, @bitCast(ijk[0])) >> 12;
    const ju = @as(u32, @bitCast(ijk[1])) >> 12;
    const ku = @as(u32, @bitCast(ijk[2])) >> 12;
    const key_x = ku | (ju << 21);
    const key_y = (iu << 10) | (ju >> 11);
    return [2]u32{ key_x, key_y };
}

// Convert 3D coordinate to linear offset within leaf node (8^3 = 512)
fn leafCoordToOffset(ijk: [3]i32) u32 {
    const x = (@as(u32, @bitCast(ijk[0])) & 7) >> 0;
    const y = (@as(u32, @bitCast(ijk[1])) & 7) >> 0;
    const z = (@as(u32, @bitCast(ijk[2])) & 7) >> 0;
    return (x << 6) | (y << 3) | z; // x*64 + y*8 + z
}

// Convert 3D coordinate to linear offset within lower node (16^3 = 4096)
fn lowerCoordToOffset(ijk: [3]i32) u32 {
    const x = (@as(u32, @bitCast(ijk[0])) & 127) >> 3;
    const y = (@as(u32, @bitCast(ijk[1])) & 127) >> 3;
    const z = (@as(u32, @bitCast(ijk[2])) & 127) >> 3;
    return (x << 8) | (y << 4) | z; // x*256 + y*16 + z
}

// Convert 3D coordinate to linear offset within upper node (32^3 = 32768)
fn upperCoordToOffset(ijk: [3]i32) u32 {
    const x = (@as(u32, @bitCast(ijk[0])) & 4095) >> 7;
    const y = (@as(u32, @bitCast(ijk[1])) & 4095) >> 7;
    const z = (@as(u32, @bitCast(ijk[2])) & 4095) >> 7;
    return (x << 10) | (y << 5) | z; // x*1024 + y*32 + z
}

test "rank query function" {
    // Test the rank query function with known values
    const value: u32 = 0b11010110; // Binary pattern with bits set at: 1, 2, 4, 6, 7
    const count: u32 = 10; // Base count
    // Test bit 0: no bits before it, should return count (10)
    try std.testing.expectEqual(@as(u32, 10), rankQuery(value, count, 0));
    // Test bit 1: bit 0 is NOT set, should return count + 0 (10)
    try std.testing.expectEqual(@as(u32, 10), rankQuery(value, count, 1));
    // Test bit 3: bits 0,1,2 - bit 0 not set, bits 1,2 are set, should return count + 2 (12)
    try std.testing.expectEqual(@as(u32, 12), rankQuery(value, count, 3));
    // Test bit 5: bits 0-4 - bits 1,2,4 are set, should return count + 3 (13)
    try std.testing.expectEqual(@as(u32, 13), rankQuery(value, count, 5));
}

test "coordinate to key conversion" {
    // Test coordinate to key conversion matches expected bit patterns
    const coord = [3]i32{ 0x12345, 0x67890, 0xABCDE };
    const key = coordToKey(coord);

    // Verify the key is computed correctly based on the WGSL algorithm
    const iu = @as(u32, @bitCast(coord[0])) >> 12;
    const ju = @as(u32, @bitCast(coord[1])) >> 12;
    const ku = @as(u32, @bitCast(coord[2])) >> 12;

    const expected_key_x = ku | (ju << 21);
    const expected_key_y = (iu << 10) | (ju >> 11);

    try std.testing.expectEqual(expected_key_x, key[0]);
    try std.testing.expectEqual(expected_key_y, key[1]);
}

test "upperCoordToOffset with negative coords" {
    // Positive coordinates
    try testing.expectEqual(@as(u32, 0), upperCoordToOffset(.{ 0, 0, 0 }));
    try testing.expectEqual(@as(u32, 1), upperCoordToOffset(.{ 0, 0, 128 }));

    // Negative coordinates (should wrap correctly)
    const neg_result = upperCoordToOffset(.{ -100, 50, 200 });
    // Should not panic and produce valid offset
    try testing.expect(neg_result < 32768); // Within 32^3 range
}

test "coordinate wrapping behavior" {
    // Test that negative coords wrap like C
    const coord: i32 = -1;
    const as_u32 = @as(u32, @bitCast(coord));
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), as_u32);

    const masked = as_u32 & 4095;
    try testing.expectEqual(@as(u32, 4095), masked);
}

test "struct alignments" {
    // Verify all structs are properly sized/aligned for the file format
    std.debug.print("PicoVDBFileHeader: size={} align={}\n", .{ @sizeOf(PicoVDBFileHeader), @alignOf(PicoVDBFileHeader) });
    std.debug.print("PicoVDBGrid: size={} align={}\n", .{ @sizeOf(PicoVDBGrid), @alignOf(PicoVDBGrid) });
    std.debug.print("PicoVDBRoot: size={} align={}\n", .{ @sizeOf(PicoVDBRoot), @alignOf(PicoVDBRoot) });
    std.debug.print("PicoVDBUpper: size={} align={}\n", .{ @sizeOf(PicoVDBUpper), @alignOf(PicoVDBUpper) });
    std.debug.print("PicoVDBLower: size={} align={}\n", .{ @sizeOf(PicoVDBLower), @alignOf(PicoVDBLower) });
    std.debug.print("PicoVDBLeaf: size={} align={}\n", .{ @sizeOf(PicoVDBLeaf), @alignOf(PicoVDBLeaf) });

    // Check that sizes are multiples of alignments (no padding at end)
    try testing.expect(@sizeOf(PicoVDBGrid) % @alignOf(PicoVDBGrid) == 0);
    try testing.expect(@sizeOf(PicoVDBRoot) % @alignOf(PicoVDBRoot) == 0);
    try testing.expect(@sizeOf(PicoVDBUpper) % @alignOf(PicoVDBUpper) == 0);
    try testing.expect(@sizeOf(PicoVDBLower) % @alignOf(PicoVDBLower) == 0);
    try testing.expect(@sizeOf(PicoVDBLeaf) % @alignOf(PicoVDBLeaf) == 0);
}
