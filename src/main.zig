const std = @import("std");
const picovdb = @import("picovdb");

// Import NanoVDB C headers
const c = @cImport({
    @cDefine("PNANOVDB_C", "1");
    @cDefine("PNANOVDB_BUF_BOUNDS_CHECK", "1");
    @cInclude("PNanoVDB.h");
});

// NanoVDBFileHeader structure (16 bytes)
const NanoVDBFileHeader = extern struct {
    magic: u64, // 8 bytes - magic number
    version: u32, // 4 bytes - packed version (major:11, minor:11, patch:10)
    grid_count: u16, // 2 bytes - number of grids in file
    codec: u16, // 2 bytes - compression codec

    // Helper functions to extract version components
    fn getVersionMajor(self: NanoVDBFileHeader) u32 {
        return self.version >> 21;
    }
    fn getVersionMinor(self: NanoVDBFileHeader) u32 {
        return (self.version >> 10) & 0x7ff;
    }
    fn getVersionPatch(self: NanoVDBFileHeader) u32 {
        return self.version & 0x3ff;
    }
};

// NanoVDBFileMetaData structure (176 bytes) - one per grid after FileHeader
const NanoVDBFileMetaData = extern struct {
    grid_size: u64, // 8 bytes - size of grid data in bytes
    file_size: u64, // 8 bytes - total file size (unused for our purposes)
    name_key: u64, // 8 bytes - hash key for grid name
    voxel_count: u64, // 8 bytes - number of active voxels
    grid_type: u32, // 4 bytes - grid data type (float, etc.)
    grid_class: u32, // 4 bytes - grid class (level set, fog volume, etc.)
    world_bbox: [6]f64, // 48 bytes - world space bounding box (min.xyz, max.xyz)
    index_bbox: [6]i32, // 24 bytes - index space bounding box
    voxel_size: [3]f64, // 24 bytes - voxel size in world units
    name_size: u32, // 4 bytes - size of grid name string
    node_count: [4]u32, // 16 bytes - [leaf, lower, upper, root] node counts
    tile_count: [3]u32, // 12 bytes - [leaf, lower, upper] tile counts
    codec: u16, // 2 bytes - compression codec
    _pad1: u16, // 2 bytes - padding
    version: u32, // 4 bytes - grid version
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "convert")) {
        if (args.len != 4) {
            std.debug.print("Error: convert command requires exactly 2 arguments: <src>.nvdb <dst>.pvdb\n", .{});
            try printUsage();
            return;
        }

        const src_path = args[2];
        const dst_path = args[3];

        // Validate file extensions
        if (!std.mem.endsWith(u8, src_path, ".nvdb")) {
            std.debug.print("Error: Source file must have .nvdb extension\n", .{});
            return;
        }

        if (!std.mem.endsWith(u8, dst_path, ".pvdb")) {
            std.debug.print("Error: Destination file must have .pvdb extension\n", .{});
            return;
        }

        try processConversion(allocator, src_path, dst_path);
    } else {
        std.debug.print("Error: Unknown command '{s}'\n", .{command});
        try printUsage();
    }
}

fn printUsage() !void {
    std.debug.print("Usage: picovdb <command> [args]\n", .{});
    std.debug.print("\nCommands:\n", .{});
    std.debug.print("  convert <src>.nvdb <dst>.pvdb    Convert NanoVDB file to PicoVDB format\n", .{});
}

fn processConversion(allocator: std.mem.Allocator, src_path: []const u8, dst_path: []const u8) !void {
    std.debug.print("Converting '{s}' to '{s}'...\n", .{ src_path, dst_path });

    // Open and read the source file
    const src_file = std.fs.cwd().openFile(src_path, .{}) catch |err| {
        std.debug.print("Error: Could not open source file '{s}': {}\n", .{ src_path, err });
        return;
    };
    defer src_file.close();

    const src_stat = try src_file.stat();
    std.debug.print("Source file size: {} bytes ({:.2} MB)\n", .{ src_stat.size, @as(f64, @floatFromInt(src_stat.size)) / 1024.0 / 1024.0 });

    // Read entire file into memory
    const file_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, src_stat.size, 4));
    defer allocator.free(file_buffer);

    _ = try src_file.readAll(file_buffer);
    std.debug.print("File read into memory successfully\n", .{});

    // Convert NanoVDB to PicoVDB format
    var picovdb_file = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, file_buffer, &picovdb_file);

    // Write PicoVDB file
    try writePicoVDBFile(dst_path, &picovdb_file);

    std.debug.print("Conversion completed successfully!\n", .{});
}

fn convertNanoVDBToPicoVDB(allocator: std.mem.Allocator, buffer: []const u8, picovdb_file: *picovdb.PicoVDBFileMutable) !void {
    std.debug.print("\n=== Converting to PicoVDB Format ===\n", .{});

    if (buffer.len < @sizeOf(NanoVDBFileHeader)) {
        return error.FileTooSmall;
    }

    // Parse file header
    const file_header_ptr: *const NanoVDBFileHeader = @ptrCast(@alignCast(buffer.ptr));
    if (file_header_ptr.magic == c.PNANOVDB_MAGIC_FILE) {
        // File format - skip FileMetaData and find grids
        var offset: usize = 16;
        for (0..file_header_ptr.grid_count) |grid_index| {
            std.debug.print("Converting grid {}...\n", .{grid_index});
            offset = try convertGridWithMetadata(allocator, buffer, offset, picovdb_file, @intCast(grid_index));
        }
    } else if (file_header_ptr.magic == c.PNANOVDB_MAGIC_GRID) {
        // Single grid format
        std.debug.print("Converting single grid...\n", .{});
        _ = try convertGrid(allocator, buffer, 0, picovdb_file, 0);
    }

    // Calculate total active voxels we extracted
    const total_extracted_voxels = picovdb_file.data_buffer.items.len / 4; // 4 bytes per float

    std.debug.print("Conversion complete: {} grids, {} roots, {} uppers, {} lowers, {} leaves, {} data\n", .{
        picovdb_file.grids.items.len,
        picovdb_file.roots.items.len,
        picovdb_file.uppers.items.len,
        picovdb_file.lowers.items.len,
        picovdb_file.leaves.items.len,
        picovdb_file.data_buffer.items.len,
    });
    std.debug.print("Total active voxels extracted: {} (vs NanoVDB reported: varies by grid)\n", .{total_extracted_voxels});
}

fn convertRootTiles(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, tree_handle: c.pnanovdb_tree_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid) !void {
    // Get root handle from tree
    const root_handle = c.pnanovdb_tree_get_root(buf, tree_handle);
    const tile_count = c.pnanovdb_root_get_tile_count(buf, root_handle);

    std.debug.print("Converting {} root tiles...\n", .{tile_count});

    // Assume float grid type for now - TODO: Get from grid header
    const grid_type = c.PNANOVDB_GRID_TYPE_FLOAT;

    // Add background value to data buffer
    const backgound_address = c.pnanovdb_root_get_background_address(c.PNANOVDB_GRID_TYPE_FLOAT, buf, root_handle);
    const background_value = c.pnanovdb_read_float(buf, backgound_address);
    const background_bytes = std.mem.asBytes(&background_value);
    try picovdb_file.data_buffer.appendSlice(allocator, background_bytes); // 0
    const inside_value = -background_value;
    const inside_bytes = std.mem.asBytes(&inside_value);
    try picovdb_file.data_buffer.appendSlice(allocator, inside_bytes); // 1

    // Process each root tile
    for (0..tile_count) |i| {
        const tile_handle = c.pnanovdb_root_get_tile(grid_type, root_handle, @intCast(i));

        // Extract tile data
        const key = c.pnanovdb_root_tile_get_key(buf, tile_handle);
        const state = c.pnanovdb_root_tile_get_state(buf, tile_handle);
        const child_offset = c.pnanovdb_root_tile_get_child(buf, tile_handle);

        // Convert to PicoVDB format (roots always have children now)
        const pico_root = picovdb.PicoVDBRoot{
            .key = [2]u32{ @truncate(key), @truncate(key >> 32) }, // Split 64-bit key into 2x32-bit
        };

        _ = state; // No longer stored - roots always have children
        try picovdb_file.roots.append(allocator, pico_root);

        //// Check if this is an active tile (has value but no children)
        //const is_active_tile = (child_offset == 0 and state != 0);
        //if (i < 8) { // Debug output for all tiles
        //    std.debug.print("  Root tile {}: key=0x{X}, state={}, child_offset={} {s}\n", .{ i, key, state, child_offset, if (is_active_tile) "[ACTIVE TILE]" else "" });
        //}

        // If this tile has children (upper nodes), traverse them
        if (child_offset != 0) {
            const upper_handle = c.pnanovdb_root_get_child(grid_type, buf, root_handle, tile_handle);
            try convertUpperNodesFromHandle(allocator, buf, grid_type, upper_handle, picovdb_file, picovdb_grid);
        }
    }
}

fn convertUpperNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, upper_handle: c.pnanovdb_upper_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid) !void {
    // Create interleaved mask array with prefix sum offsets
    var mask_array: [1024]picovdb.PicoVDBNodeMask = undefined;

    // Read value and child masks from NanoVDB
    const value_mask_addr = c.pnanovdb_address_offset(upper_handle.address, c.PNANOVDB_UPPER_OFF_VALUE_MASK);
    const child_mask_addr = c.pnanovdb_address_offset(upper_handle.address, c.PNANOVDB_UPPER_OFF_CHILD_MASK);

    for (0..1024) |i| {
        // Read masks for this word from NanoVDB
        const nano_child_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(child_mask_addr, @intCast(i * 4)));
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        // Current offsets (prefix sums)
        const child_offset = @as(u32, @intCast(picovdb_file.lowers.items.len - picovdb_grid.lower_start));
        const data_elem_size = 4; // f32 hardcoded for now
        const value_offset = @as(u32, @intCast(picovdb_file.data_buffer.items.len / data_elem_size - picovdb_grid.data_start * 16));

        // New encoding: inside and value bitmasks
        // inside=0,value=0 -> outside implicit
        // inside=0,value=1 -> stored value
        // inside=1,value=0 -> inside implicit
        // inside=1,value=1 -> child node
        var inside_word: u32 = 0;
        var value_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;
            const has_nano_child = (nano_child_word >> bit) & 1 != 0;

            if (has_nano_child) {
                // Child: set both inside and value bits
                inside_word |= (@as(u32, 1) << bit);
                value_word |= (@as(u32, 1) << bit);
            } else if (has_nano_value) {
                // Stored value: set only value bit
                value_word |= (@as(u32, 1) << bit);
                const value_address = c.pnanovdb_upper_get_table_address(grid_type, buf, upper_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                const value_bytes = std.mem.asBytes(&value);
                try picovdb_file.data_buffer.appendSlice(allocator, value_bytes);
            } else {
                // No value or child - check if inside
                const value_address = c.pnanovdb_upper_get_table_address(grid_type, buf, upper_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                if (value < 0.0) {
                    // Inside implicit: set only inside bit
                    inside_word |= (@as(u32, 1) << bit);
                }
                // Otherwise: outside implicit (both bits 0)
            }
        }

        // Process children after building the mask (to maintain correct ordering)
        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            if ((nano_child_word >> bit) & 1 != 0) {
                const n = i * 32 + bit_index;
                const lower_handle = c.pnanovdb_upper_get_child(grid_type, buf, upper_handle, @intCast(n));
                try convertLowerNodesFromHandle(allocator, buf, grid_type, lower_handle, picovdb_file, picovdb_grid);
            }
        }

        // Store with new encoding
        mask_array[i] = picovdb.PicoVDBNodeMask{
            .inside = inside_word,
            .value = value_word,
            .value_offset = value_offset,
            .child_offset = child_offset,
        };
    }

    // Convert to PicoVDB format
    const pico_upper = picovdb.PicoVDBUpper{
        .mask = mask_array,
    };

    try picovdb_file.uppers.append(allocator, pico_upper);
}

fn convertLowerNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, lower_handle: c.pnanovdb_lower_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid) !void {
    // Create interleaved mask array with prefix sum offsets
    var mask_array: [128]picovdb.PicoVDBNodeMask = undefined;

    // Read value and child masks from NanoVDB
    const value_mask_addr = c.pnanovdb_address_offset(lower_handle.address, c.PNANOVDB_LOWER_OFF_VALUE_MASK);
    const child_mask_addr = c.pnanovdb_address_offset(lower_handle.address, c.PNANOVDB_LOWER_OFF_CHILD_MASK);

    for (0..128) |i| {
        // Read masks for this word from NanoVDB
        const nano_child_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(child_mask_addr, @intCast(i * 4)));
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        // Current offsets (prefix sums)
        const child_offset = @as(u32, @intCast(picovdb_file.leaves.items.len - picovdb_grid.leaf_start));
        const data_elem_size = 4; // f32 hardcoded for now
        const value_offset = @as(u32, @intCast(picovdb_file.data_buffer.items.len / data_elem_size - picovdb_grid.data_start * 16)); // 4 bytes per float

        // New encoding: inside and value bitmasks
        var inside_word: u32 = 0;
        var value_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;
            const has_nano_child = (nano_child_word >> bit) & 1 != 0;

            if (has_nano_child) {
                // Child: set both inside and value bits
                inside_word |= (@as(u32, 1) << bit);
                value_word |= (@as(u32, 1) << bit);
            } else if (has_nano_value) {
                // Stored value: set only value bit
                value_word |= (@as(u32, 1) << bit);
                const value_address = c.pnanovdb_lower_get_table_address(grid_type, buf, lower_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                const value_bytes = std.mem.asBytes(&value);
                try picovdb_file.data_buffer.appendSlice(allocator, value_bytes);
            } else {
                // No value or child - check if inside
                const value_address = c.pnanovdb_lower_get_table_address(grid_type, buf, lower_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                if (value < 0.0) {
                    // Inside implicit: set only inside bit
                    inside_word |= (@as(u32, 1) << bit);
                }
            }
        }

        // Process children after building the mask
        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            if ((nano_child_word >> bit) & 1 != 0) {
                const n = i * 32 + bit_index;
                const leaf_handle = c.pnanovdb_lower_get_child(grid_type, buf, lower_handle, @intCast(n));
                try convertLeafNodesFromHandle(allocator, buf, grid_type, leaf_handle, picovdb_file, picovdb_grid);
            }
        }

        // Store with new encoding
        mask_array[i] = picovdb.PicoVDBNodeMask{
            .inside = inside_word,
            .value = value_word,
            .value_offset = value_offset,
            .child_offset = child_offset,
        };
    }

    // Convert to PicoVDB format
    const pico_lower = picovdb.PicoVDBLower{
        .mask = mask_array,
    };

    try picovdb_file.lowers.append(allocator, pico_lower);
}

fn convertLeafNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, leaf_handle: c.pnanovdb_leaf_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid) !void {
    // Create interleaved mask array with prefix sum offsets
    var mask_array: [16]picovdb.PicoVDBLeafMask = undefined;

    // Read value mask from NanoVDB
    const value_mask_addr = c.pnanovdb_address_offset(leaf_handle.address, c.PNANOVDB_LEAF_OFF_VALUE_MASK);

    for (0..16) |i| {
        // Read value mask for this word from NanoVDB
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        // Current offset (prefix sum)
        const data_elem_size = 4; // f32 hardcoded for now
        const value_offset = @as(u32, @intCast(picovdb_file.data_buffer.items.len / data_elem_size - picovdb_grid.data_start * 16));

        // New encoding: inside and value bitmasks (no children on leaves)
        var inside_word: u32 = 0;
        var value_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;

            if (has_nano_value) {
                // Stored value: set only value bit
                value_word |= (@as(u32, 1) << bit);
                const value_addr = c.pnanovdb_leaf_get_table_address(grid_type, buf, leaf_handle, n);
                const value = c.pnanovdb_read_float(buf, value_addr);
                const value_bytes = std.mem.asBytes(&value);
                try picovdb_file.data_buffer.appendSlice(allocator, value_bytes);
            } else {
                // No value - check if inside
                const value_addr = c.pnanovdb_leaf_get_table_address(grid_type, buf, leaf_handle, n);
                const value = c.pnanovdb_read_float(buf, value_addr);
                if (value < 0.0) {
                    // Inside implicit: set only inside bit
                    inside_word |= (@as(u32, 1) << bit);
                }
            }
        }

        // Store with new encoding
        mask_array[i] = picovdb.PicoVDBLeafMask{
            .inside = inside_word,
            .value = value_word,
            .value_offset = value_offset,
        };
    }

    // Convert to PicoVDB format
    const pico_leaf = picovdb.PicoVDBLeaf{
        .mask = mask_array,
    };

    try picovdb_file.leaves.append(allocator, pico_leaf);
}

fn convertGrid(allocator: std.mem.Allocator, buffer: []const u8, offset: usize, picovdb_file: *picovdb.PicoVDBFileMutable, grid_index: u32) !usize {
    // Copy remaining buffer from grid offset to end into aligned buffer
    const remaining_len = buffer.len - offset;
    const aligned_len = std.mem.alignForward(usize, remaining_len, @alignOf(c.pnanovdb_grid_t));
    const grid_buffer = try allocator.alloc(u8, aligned_len);
    defer allocator.free(grid_buffer);

    // Copy data and zero-pad if needed
    @memcpy(grid_buffer[0..remaining_len], buffer[offset..]);
    if (aligned_len > remaining_len) {
        @memset(grid_buffer[remaining_len..], 0);
    }

    // Direct cast to C struct
    const grid_ptr: *const c.pnanovdb_grid_t = @ptrCast(@alignCast(grid_buffer.ptr));

    // Verify we have the correct grid magic
    if (grid_ptr.magic != c.PNANOVDB_MAGIC_GRID) {
        std.debug.print("Error: Expected grid magic 0x{X}, got 0x{X}\n", .{ c.PNANOVDB_MAGIC_GRID, grid_ptr.magic });
        return error.InvalidGridMagic;
    }

    // Get tree pointer (located right after the grid struct)
    const tree_offset = @sizeOf(c.pnanovdb_grid_t);
    if (tree_offset >= grid_buffer.len) {
        return error.BufferTooSmall;
    }
    const tree_ptr: *const c.pnanovdb_tree_t = @ptrCast(@alignCast(grid_buffer.ptr + tree_offset));

    const voxel_count = convertU64ToU32(tree_ptr.voxel_count) catch |err| {
        std.debug.print("Too many voxels {}: {}\n", .{ tree_ptr.voxel_count, err });
        return error.VoxelCountOverflow;
    };

    const tree_handle = c.pnanovdb_tree_handle_t{ .address = c.pnanovdb_address_t{ .byte_offset = tree_offset } };
    const pnanovdb_buf = c.pnanovdb_buf_t{
        .data = @ptrCast(@alignCast(grid_buffer.ptr)),
        .size_in_words = @intCast(grid_buffer.len / 4),
    };
    const root_handle = c.pnanovdb_tree_get_root(pnanovdb_buf, tree_handle);

    const index_bbox_min = c.pnanovdb_root_get_bbox_min(pnanovdb_buf, root_handle);
    const index_bbox_max = c.pnanovdb_root_get_bbox_max(pnanovdb_buf, root_handle);

    // Calculate data_start in 16-byte units (current data buffer length / 16)
    const data_start_bytes = picovdb_file.data_buffer.items.len;
    std.debug.assert(data_start_bytes % 16 == 0); // Must be 16-byte aligned

    // Create PicoVDB grid
    var picovdb_grid = picovdb.PicoVDBGrid{
        .grid_index = grid_index,
        .upper_start = @intCast(picovdb_file.uppers.items.len), // Current upper array length (= root start)
        .lower_start = @intCast(picovdb_file.lowers.items.len), // Current lower array length
        .leaf_start = @intCast(picovdb_file.leaves.items.len), // Current leaf array length
        .data_start = @intCast(data_start_bytes / 16), // 16-byte index into data buffer
        .data_elem_count = 0, // Will be set after conversion
        .grid_type = picovdb.GRID_TYPE_SDF_FLOAT, // Assume float grid for now
        ._pad1 = 0,
        .index_bounds_min = [3]i32{
            @intCast(index_bbox_min.x), // min.x
            @intCast(index_bbox_min.y), // min.y
            @intCast(index_bbox_min.z), // min.z
        },
        ._pad2 = 0,
        .index_bounds_max = [3]i32{
            @intCast(index_bbox_max.x), // max.x
            @intCast(index_bbox_max.y), // max.y
            @intCast(index_bbox_max.z), // max.z
        },
        ._pad3 = 0,
    };

    try convertRootTiles(allocator, pnanovdb_buf, tree_handle, picovdb_file, &picovdb_grid);

    // Calculate data_elem_count (number of f32 values for this grid)
    const data_end_bytes = picovdb_file.data_buffer.items.len;
    picovdb_grid.data_elem_count = @intCast((data_end_bytes - data_start_bytes) / 4); // 4 bytes per f32

    // Pad data buffer to 16-byte alignment for next grid
    const data_padding = std.mem.alignForward(usize, data_end_bytes, 16) - data_end_bytes;
    if (data_padding > 0) {
        const padding = [_]u8{0} ** 16;
        try picovdb_file.data_buffer.appendSlice(allocator, padding[0..data_padding]);
    }

    std.debug.print("  Grid version: {}\n", .{grid_ptr.version});
    std.debug.print("  Grid size: {} bytes\n", .{grid_ptr.grid_size});
    std.debug.print("  Voxel count: {}\n", .{voxel_count});
    std.debug.print("  Data buffer size: {} bytes, grid data_start: {} (16B units), data_elem_count: {}\n", .{ picovdb_file.data_buffer.items.len, picovdb_grid.data_start, picovdb_grid.data_elem_count });
    std.debug.print("  Index bbox: [{:.3}, {:.3}, {:.3}] to [{:.3}, {:.3}, {:.3}]\n", .{ index_bbox_min.x, index_bbox_min.y, index_bbox_min.z, index_bbox_max.x, index_bbox_max.y, index_bbox_max.z });

    // Add grid to PicoVDB file
    try picovdb_file.grids.append(allocator, picovdb_grid);

    return offset + grid_ptr.grid_size;
}

fn convertGridWithMetadata(allocator: std.mem.Allocator, buffer: []const u8, offset: usize, picovdb_file: *picovdb.PicoVDBFileMutable, grid_index: u32) !usize {
    if (buffer.len < offset + @sizeOf(NanoVDBFileMetaData)) {
        return error.BufferTooSmall;
    }

    // Skip FileMetaData (160 bytes) and grid name to get to actual grid data
    const metadata_ptr: *const NanoVDBFileMetaData = @ptrCast(@alignCast(buffer.ptr + offset));
    const grid_offset = offset + @sizeOf(NanoVDBFileMetaData) + metadata_ptr.name_size;

    std.debug.print("  Skipping metadata ({} bytes) + name ({} bytes)\n", .{ @sizeOf(NanoVDBFileMetaData), metadata_ptr.name_size });

    // Convert the grid using unified grid parsing - grid will determine its own size
    return try convertGrid(allocator, buffer, grid_offset, picovdb_file, grid_index);
}

fn writePicoVDBFile(dst_path: []const u8, picovdb_file: *picovdb.PicoVDBFileMutable) !void {
    std.debug.print("\n=== Writing PicoVDB File ===\n", .{});

    const dst_file = std.fs.cwd().createFile(dst_path, .{}) catch |err| {
        std.debug.print("Error: Could not create output file '{s}': {}\n", .{ dst_path, err });
        return;
    };
    defer dst_file.close();

    // Calculate padded sizes for alignment
    const root_count = picovdb_file.roots.items.len;
    const root_needs_padding = root_count % 2 == 1;
    const data_size = picovdb_file.data_buffer.items.len;
    const data_size_padded = std.mem.alignForward(usize, data_size, 16);

    // Update file header counts before writing
    picovdb_file.header.magic = [2]u32{ 0x6f636950, 0x30424456 }; // 'PicoVDB0' little endian
    picovdb_file.header.version = 0;
    picovdb_file.header.grid_count = @intCast(picovdb_file.grids.items.len);
    picovdb_file.header.upper_count = @intCast(picovdb_file.uppers.items.len);
    picovdb_file.header.lower_count = @intCast(picovdb_file.lowers.items.len);
    picovdb_file.header.leaf_count = @intCast(picovdb_file.leaves.items.len);
    picovdb_file.header.data_count = @intCast(data_size_padded / 16); // 16-byte unit

    // Write PicoVDB file header
    const header_bytes = std.mem.asBytes(&picovdb_file.header);
    _ = try dst_file.writeAll(header_bytes);

    // Write grids
    const grids_bytes = std.mem.sliceAsBytes(picovdb_file.grids.items);
    _ = try dst_file.writeAll(grids_bytes);

    // Write roots (padded to 16-byte alignment via even count)
    const roots_bytes = std.mem.sliceAsBytes(picovdb_file.roots.items);
    _ = try dst_file.writeAll(roots_bytes);
    if (root_needs_padding) {
        // Add padding root for 16-byte alignment
        const padding_root = picovdb.PicoVDBRoot{ .key = [2]u32{ 0, 0 } };
        _ = try dst_file.writeAll(std.mem.asBytes(&padding_root));
    }

    // Write uppers
    const uppers_bytes = std.mem.sliceAsBytes(picovdb_file.uppers.items);
    _ = try dst_file.writeAll(uppers_bytes);

    // Write lowers
    const lowers_bytes = std.mem.sliceAsBytes(picovdb_file.lowers.items);
    _ = try dst_file.writeAll(lowers_bytes);

    // Write leaves
    const leaves_bytes = std.mem.sliceAsBytes(picovdb_file.leaves.items);
    _ = try dst_file.writeAll(leaves_bytes);

    // Write data buffer (padded to 16 bytes)
    _ = try dst_file.writeAll(picovdb_file.data_buffer.items);
    const data_padding = data_size_padded - data_size;
    if (data_padding > 0) {
        const padding = [_]u8{0} ** 16;
        _ = try dst_file.writeAll(padding[0..data_padding]);
    }

    std.debug.print("PicoVDB file written: {s}\n", .{dst_path});
}

pub fn convertU64ToU32(value: u64) error{Overflow}!u32 {
    return std.math.cast(u32, value) orelse error.Overflow;
}
pub fn bufferToU32Ptr(buffer: []const u8) error{ Misaligned, InvalidLength }![*]const u32 {
    if (buffer.len % 4 != 0) return error.InvalidLength;
    if (@intFromPtr(buffer.ptr) % @alignOf(u32) != 0) return error.Misaligned;

    return @ptrCast(@alignCast(buffer.ptr));
}

test "basic picovdb structures" {
    // Basic test to verify structures compile and have correct size
    const grid = picovdb.PicoVDBGrid{
        .grid_index = 0,
        .upper_start = 0,
        .lower_start = 0,
        .leaf_start = 0,
        .data_start = 0,
        .data_elem_count = 0,
        .grid_type = picovdb.GRID_TYPE_SDF_FLOAT,
        ._pad1 = 0,
        .index_bounds_min = [3]i32{ 0, 0, 0 },
        ._pad2 = 0,
        .index_bounds_max = [3]i32{ 8, 8, 8 },
        ._pad3 = 0,
    };
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(picovdb.PicoVDBGrid));
    try std.testing.expectEqual(@as(u32, 0), grid.grid_index);

    const accessor = picovdb.PicoVDBReadAccessor.init(0);
    try std.testing.expect(accessor.grid == 0);
    try std.testing.expect(accessor.upper == std.math.maxInt(u32));
}

test "picovdb file loader from bytes" {
    const allocator = std.testing.allocator;

    // Convert sphere.nvdb to PicoVDB format in memory
    const test_file = "data/sphere.nvdb";
    const file = try std.fs.cwd().openFile(test_file, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const nvdb_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 4));
    defer allocator.free(nvdb_buffer);
    _ = try file.readAll(nvdb_buffer);

    // Convert to PicoVDB format
    var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file_mutable.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &picovdb_file_mutable);

    // Convert to file format with buffer
    const picovdb_buffer = try picovdb_file_mutable.encode(allocator);
    defer allocator.free(picovdb_buffer);

    // Test loading the PicoVDB file from bytes
    const picovdb_file = try picovdb.PicoVDBFile.fromBytes(picovdb_buffer);

    // Verify file structure
    try std.testing.expectEqual(@as(u32, 1), picovdb_file.header.grid_count);
    try std.testing.expect(picovdb_file.grids.len == 1);
    try std.testing.expect(picovdb_file.roots.len > 0);
    try std.testing.expect(picovdb_file.data_buffer.len > 0);

    // Test accessing a grid
    const grid = picovdb_file.getGrid(0).?;
    try std.testing.expectEqual(picovdb.GRID_TYPE_SDF_FLOAT, grid.grid_type);

    // Test value access
    const test_value = picovdb_file.getGridFloat(grid, 0); // Background value at index 0
    try std.testing.expect(test_value > 0.0); // Should be positive background value

    std.log.info("Successfully loaded PicoVDB file: {} grids, {} roots, {} bytes data", .{ picovdb_file.grids.len, picovdb_file.roots.len, picovdb_file.data_buffer.len });
}

test "read accessor integration with data files" {
    std.testing.log_level = .debug;
    const allocator = std.testing.allocator;

    const test_files = [_][]const u8{
        "data/sphere.nvdb",
        //"data/bunny.nvdb",
    };
    for (test_files) |test_file| {
        const file = try std.fs.cwd().openFile(test_file, .{});
        defer file.close();

        const file_size = try file.getEndPos();
        const buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 16));
        defer allocator.free(buffer);
        _ = try file.readAll(buffer);

        std.log.info("Using test file: {s} ({} bytes)", .{ test_file, file_size });

        // Convert to PicoVDB format
        var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
        defer picovdb_file_mutable.deinit(allocator);

        try convertNanoVDBToPicoVDB(allocator, buffer, &picovdb_file_mutable);

        // Convert to read-only file for testing
        const picovdb_buffer = try picovdb_file_mutable.encode(allocator);
        defer allocator.free(picovdb_buffer);
        const picovdb_file = try picovdb.PicoVDBFile.fromBytes(picovdb_buffer);

        // Verify we have grids to test
        std.debug.assert(picovdb_file.grids.len == 1);

        // Get first grid for testing
        const grid = &picovdb_file.grids[0];

        // Initialize accessors
        var pico_accessor = picovdb.PicoVDBReadAccessor.init(0);
        var pnano_accessor = std.mem.zeroes(c.pnanovdb_readaccessor_t);

        // Parse file header to get correct grid offset
        const file_header_ptr: *const NanoVDBFileHeader = @ptrCast(@alignCast(buffer.ptr));
        std.debug.assert(file_header_ptr.magic == c.PNANOVDB_MAGIC_FILE);
        try std.testing.expect(file_header_ptr.grid_count == 1);

        // Calculate grid offset: FileHeader (16) + FileMetaData (176) + grid name
        var grid_offset: usize = 16; // Skip file header
        const metadata_ptr: *const NanoVDBFileMetaData = @ptrCast(@alignCast(buffer.ptr + grid_offset));
        grid_offset += @sizeOf(NanoVDBFileMetaData) + metadata_ptr.name_size;

        std.log.info("Grid offset calculated: {} bytes", .{grid_offset});

        // Create aligned grid buffer (similar to convertGrid function)
        const remaining_len = buffer.len - grid_offset;
        const aligned_len = std.mem.alignForward(usize, remaining_len, @alignOf(c.pnanovdb_grid_t));
        const grid_buffer = try allocator.alloc(u8, aligned_len);
        defer allocator.free(grid_buffer);

        // Copy grid data to aligned buffer
        @memcpy(grid_buffer[0..remaining_len], buffer[grid_offset..]);
        if (aligned_len > remaining_len) {
            @memset(grid_buffer[remaining_len..], 0);
        }

        // Create PNanoVDB buffer pointing to the aligned grid data
        const pnano_grid_buf = c.pnanovdb_buf_t{
            .data = @ptrCast(@alignCast(grid_buffer.ptr)),
            .size_in_words = @intCast(grid_buffer.len / 4),
        };

        // Grid is now at offset 0 in the aligned buffer
        const grid_handle = c.pnanovdb_grid_handle_t{ .address = c.pnanovdb_address_t{ .byte_offset = 0 } };
        const tree_handle = c.pnanovdb_grid_get_tree(pnano_grid_buf, grid_handle);
        const root_handle = c.pnanovdb_tree_get_root(pnano_grid_buf, tree_handle);
        c.pnanovdb_readaccessor_init(&pnano_accessor, root_handle);

        var matches: u32 = 0;
        var total_tests: u32 = 0;

        // Sample along one full axis of the sphere volume
        // Based on world bbox: [-3.100, -3.100, -3.100] to [3.150, 3.150, 3.150]
        // With transform scale 0.050, this maps to index space roughly [-62, -62, -62] to [63, 63, 63]
        // Let's sample along the Z axis at a fixed X,Y position
        const sample_x: i32 = -30;
        const sample_y: i32 = -30;

        var z_offset: i32 = -65;
        while (z_offset <= 65) : (z_offset += 1) {
            const coord = [_]i32{ sample_x, sample_y, @as(i32, @intCast(z_offset)) };
            total_tests += 1;

            // Get level and count from PicoVDB ReadAccessor
            const pico_result = pico_accessor.getLevelCount(coord, grid, &picovdb_file);
            const pico_value = picovdb_file.getGridFloat(grid, pico_result.count);

            // Get value from PNanoVDB
            const ijk = c.pnanovdb_coord_t{ .x = coord[0], .y = coord[1], .z = coord[2] };
            var pnano_level: u32 = 0;
            const pnano_address = c.pnanovdb_readaccessor_get_value_address_and_level(c.PNANOVDB_GRID_TYPE_FLOAT, pnano_grid_buf, &pnano_accessor, &ijk, &pnano_level);
            const pnano_value = c.pnanovdb_read_float(pnano_grid_buf, pnano_address);

            // Compare values (allow small floating point differences)
            const diff = @abs(pico_value - pnano_value);
            const values_match = diff < 1e-6;
            if (values_match) {
                matches += 1;
                //std.log.warn("Match at [{}, {}, {}]: PicoVDB={d:.6} (level={}, count={}), PNanoVDB={d:.6} (level={})", .{ coord[0], coord[1], coord[2], pico_value, pico_result.level, pico_result.count, pnano_value, pnano_level });
            } else {
                //std.log.warn("Mismatch at [{}, {}, {}]: PicoVDB={d:.6} (level={}, count={}), PNanoVDB={d:.6} (level={}), diff={d:.8}", .{ coord[0], coord[1], coord[2], pico_value, pico_result.level, pico_result.count, pnano_value, pnano_level, diff });
            }
        }
        try std.testing.expectEqual(total_tests, matches);
    }
}
