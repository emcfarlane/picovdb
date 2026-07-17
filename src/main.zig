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

const ValueType = enum {
    f32,
    u8,

    fn elemSize(self: ValueType) usize {
        return switch (self) {
            .f32 => 4,
            .u8 => 1,
        };
    }

    fn gridType(self: ValueType) u32 {
        return switch (self) {
            .f32 => picovdb.GRID_TYPE_SDF_FLOAT,
            .u8 => picovdb.GRID_TYPE_SDF_UINT8,
        };
    }
};

fn appendValue(data_buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, value: f32, value_type: ValueType) !void {
    switch (value_type) {
        .f32 => try data_buffer.appendSlice(allocator, std.mem.asBytes(&value)),
        .u8 => {
            const quantized: u8 = @intFromFloat(std.math.clamp(@round((value / picovdb.LEVEL_SET_HALF_WIDTH + 1.0) * 127.5), 0.0, 255.0));
            try data_buffer.append(allocator, quantized);
        },
    }
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    for (args) |arg| {
        std.log.info("arg: {s}", .{arg});
    }
    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "convert")) {
        // Parse optional flags
        var value_type: ValueType = .f32;
        var positional_start: usize = 2;

        while (positional_start < args.len and std.mem.startsWith(u8, args[positional_start], "--")) {
            if (std.mem.eql(u8, args[positional_start], "--type")) {
                positional_start += 1;
                if (positional_start >= args.len) {
                    std.debug.print("Error: --type requires a value (f32 or u8)\n", .{});
                    return;
                }
                const type_str = args[positional_start];
                if (std.mem.eql(u8, type_str, "f32")) {
                    value_type = .f32;
                } else if (std.mem.eql(u8, type_str, "u8")) {
                    value_type = .u8;
                } else {
                    std.debug.print("Error: Unknown type '{s}'. Use 'f32' or 'u8'\n", .{type_str});
                    return;
                }
                positional_start += 1;
            } else {
                std.debug.print("Error: Unknown flag '{s}'\n", .{args[positional_start]});
                return;
            }
        }

        if (args.len - positional_start != 2) {
            std.debug.print("Error: convert command requires exactly 2 arguments: <src>.nvdb <dst>.pvdb\n", .{});
            try printUsage();
            return;
        }

        const src_path = args[positional_start];
        const dst_path = args[positional_start + 1];

        // Validate file extensions
        if (!std.mem.endsWith(u8, src_path, ".nvdb")) {
            std.debug.print("Error: Source file must have .nvdb extension\n", .{});
            return;
        }

        if (!std.mem.endsWith(u8, dst_path, ".pvdb")) {
            std.debug.print("Error: Destination file must have .pvdb extension\n", .{});
            return;
        }

        try processConversion(init.io, init.gpa, src_path, dst_path, value_type);
    } else if (std.mem.eql(u8, command, "mesh")) {
        try meshCommand(init.io, init.gpa, args[2..]);
    } else {
        std.debug.print("Error: Unknown command '{s}'\n", .{command});
        try printUsage();
    }
}

fn printUsage() !void {
    std.debug.print("Usage: picovdb <command> [args]\n", .{});
    std.debug.print("\nCommands:\n", .{});
    std.debug.print("  convert [--type f32|u8] <src>.nvdb <dst>.pvdb    Convert NanoVDB file to PicoVDB format\n", .{});
    std.debug.print("  mesh --voxel <size> [--width <hw>] [--rotate-x|y|z <deg>]... [--type f32|u8] <src>.stl <dst>.pvdb\n", .{});
    std.debug.print("                                                   Voxelize an STL mesh to a PicoVDB level set\n", .{});
    std.debug.print("                                                   (rotations apply in command-line order)\n", .{});
}

fn meshCommand(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !void {
    const Rotation = struct { axis: picovdb.stl.Axis, degrees: f32 };
    var voxel_size: ?f32 = null;
    var half_width: f32 = picovdb.LEVEL_SET_HALF_WIDTH;
    var rotations: std.ArrayList(Rotation) = .empty;
    defer rotations.deinit(allocator);
    var value_type: picovdb.mesh2ls.ValueType = .f32;
    var positional: [2]?[]const u8 = .{ null, null };
    var positional_count: usize = 0;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--voxel") or std.mem.eql(u8, arg, "--width") or
            std.mem.eql(u8, arg, "--rotate-x") or std.mem.eql(u8, arg, "--rotate-y") or
            std.mem.eql(u8, arg, "--rotate-z") or std.mem.eql(u8, arg, "--type"))
        {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: {s} requires a value\n", .{arg});
                return;
            }
            const value = args[i];
            if (std.mem.eql(u8, arg, "--type")) {
                if (std.mem.eql(u8, value, "f32")) {
                    value_type = .f32;
                } else if (std.mem.eql(u8, value, "u8")) {
                    value_type = .u8;
                } else {
                    std.debug.print("Error: Unknown type '{s}'. Use 'f32' or 'u8'\n", .{value});
                    return;
                }
                continue;
            }
            const parsed = std.fmt.parseFloat(f32, value) catch {
                std.debug.print("Error: Invalid number '{s}' for {s}\n", .{ value, arg });
                return;
            };
            if (std.mem.eql(u8, arg, "--voxel")) {
                voxel_size = parsed;
            } else if (std.mem.eql(u8, arg, "--width")) {
                half_width = parsed;
            } else {
                const axis: picovdb.stl.Axis = switch (arg[arg.len - 1]) {
                    'x' => .x,
                    'y' => .y,
                    else => .z,
                };
                try rotations.append(allocator, .{ .axis = axis, .degrees = parsed });
            }
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("Error: Unknown flag '{s}'\n", .{arg});
            return;
        } else {
            if (positional_count >= 2) {
                std.debug.print("Error: Too many arguments\n", .{});
                try printUsage();
                return;
            }
            positional[positional_count] = arg;
            positional_count += 1;
        }
    }

    if (positional_count != 2 or voxel_size == null) {
        std.debug.print("Error: mesh command requires --voxel <size> plus <src>.stl <dst>.pvdb\n", .{});
        try printUsage();
        return;
    }
    if (voxel_size.? <= 0) {
        std.debug.print("Error: --voxel must be > 0\n", .{});
        return;
    }
    const src_path = positional[0].?;
    const dst_path = positional[1].?;
    if (!std.mem.endsWith(u8, src_path, ".stl")) {
        std.debug.print("Error: Source file must have .stl extension\n", .{});
        return;
    }
    if (!std.mem.endsWith(u8, dst_path, ".pvdb")) {
        std.debug.print("Error: Destination file must have .pvdb extension\n", .{});
        return;
    }

    std.debug.print("Voxelizing '{s}' to '{s}' (voxel: {d}, width: {d}, type: {s})...\n", .{ src_path, dst_path, voxel_size.?, half_width, @tagName(value_type) });

    const cwd = std.Io.Dir.cwd();
    const src_file = cwd.openFile(io, src_path, .{}) catch |err| {
        std.debug.print("Error: Could not open source file '{s}': {}\n", .{ src_path, err });
        return;
    };
    defer src_file.close(io);

    const src_stat = try src_file.stat(io);
    const file_buffer = try allocator.alloc(u8, src_stat.size);
    defer allocator.free(file_buffer);
    _ = try src_file.readPositionalAll(io, file_buffer, 0);

    var mesh = picovdb.stl.parse(allocator, file_buffer) catch |err| {
        std.debug.print("Error: Failed to parse STL: {}\n", .{err});
        return;
    };
    defer mesh.deinit(allocator);

    const b = mesh.bounds();
    std.debug.print("Mesh: {} triangles, {} vertices, bounds [{d:.3}, {d:.3}, {d:.3}] to [{d:.3}, {d:.3}, {d:.3}]\n", .{
        mesh.triangleCount(), mesh.vertexCount(), b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2],
    });

    // Rotations apply in command-line order.
    for (rotations.items) |rotation| {
        std.debug.print("Rotating mesh {d} degrees about {s}\n", .{ rotation.degrees, @tagName(rotation.axis) });
        mesh.rotate(rotation.axis, rotation.degrees * std.math.pi / 180.0);
    }

    var picovdb_file = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file.deinit(allocator);

    const stats = picovdb.mesh2ls.meshToPicoVDB(allocator, &picovdb_file, mesh.vertices, mesh.triangles, .{
        .voxel_size = voxel_size.?,
        .half_width = half_width,
        .value_type = value_type,
    }) catch |err| {
        std.debug.print("Error: Mesh conversion failed: {}\n", .{err});
        return;
    };

    const res = [3]i32{
        stats.index_bounds_max[0] - stats.index_bounds_min[0] + 1,
        stats.index_bounds_max[1] - stats.index_bounds_min[1] + 1,
        stats.index_bounds_max[2] - stats.index_bounds_min[2] + 1,
    };
    std.debug.print("Voxelized: resolution {}x{}x{}, {} active voxels, {} surface voxels\n", .{
        res[0], res[1], res[2], stats.active_voxels, stats.surface_voxels,
    });
    std.debug.print("Tree: {} uppers, {} lowers, {} leaves\n", .{ stats.upper_count, stats.lower_count, stats.leaf_count });
    std.debug.print("Index bbox: [{}, {}, {}] to [{}, {}, {}]\n", .{
        stats.index_bounds_min[0], stats.index_bounds_min[1], stats.index_bounds_min[2],
        stats.index_bounds_max[0], stats.index_bounds_max[1], stats.index_bounds_max[2],
    });

    try writePicoVDBFile(io, dst_path, &picovdb_file);
}

fn processConversion(io: std.Io, allocator: std.mem.Allocator, src_path: []const u8, dst_path: []const u8, value_type: ValueType) !void {
    std.debug.print("Converting '{s}' to '{s}' (type: {s})...\n", .{ src_path, dst_path, @tagName(value_type) });
    const cwd = std.Io.Dir.cwd();

    // Open and read the source file
    const src_file = cwd.openFile(io, src_path, .{}) catch |err| {
        std.debug.print("Error: Could not open source file '{s}': {}\n", .{ src_path, err });
        return;
    };
    defer src_file.close(io);

    const src_stat = try src_file.stat(io);
    std.debug.print("Source file size: {} bytes ({:.2} MB)\n", .{ src_stat.size, @as(f64, @floatFromInt(src_stat.size)) / 1024.0 / 1024.0 });

    // Read entire file into memory
    const file_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, src_stat.size, 4));
    defer allocator.free(file_buffer);

    _ = try src_file.readPositionalAll(io, file_buffer, 0);
    std.debug.print("File read into memory successfully\n", .{});

    // Convert NanoVDB to PicoVDB format
    var picovdb_file = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, file_buffer, &picovdb_file, value_type);

    // Write PicoVDB file
    try writePicoVDBFile(io, dst_path, &picovdb_file);

    std.debug.print("Conversion completed successfully!\n", .{});
}

fn convertNanoVDBToPicoVDB(allocator: std.mem.Allocator, buffer: []const u8, picovdb_file: *picovdb.PicoVDBFileMutable, value_type: ValueType) !void {
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
            offset = try convertGridWithMetadata(allocator, buffer, offset, picovdb_file, value_type);
        }
    } else if (file_header_ptr.magic == c.PNANOVDB_MAGIC_GRID) {
        // Single grid format (no metadata, assume level set)
        std.debug.print("Converting single grid...\n", .{});
        _ = try convertGrid(allocator, buffer, 0, picovdb_file, value_type, false);
    }

    // Calculate total active voxels we extracted
    const total_extracted_voxels = picovdb_file.data_buffer.items.len / value_type.elemSize();

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

fn convertRootTiles(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, tree_handle: c.pnanovdb_tree_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid, value_type: ValueType, voxel_size: f32, is_fog: bool) !void {
    // Get root handle from tree
    const root_handle = c.pnanovdb_tree_get_root(buf, tree_handle);
    const tile_count = c.pnanovdb_root_get_tile_count(buf, root_handle);

    std.debug.print("Converting {} root tiles...\n", .{tile_count});

    // Assume float grid type for now - TODO: Get from grid header
    const grid_type = c.PNANOVDB_GRID_TYPE_FLOAT;

    // Add background and inside values to data buffer.
    // For fog: raw density (no voxel_size division), inside = 1.0 (full density).
    // For level-set: normalize by voxel_size, inside = -background (negative inside).
    const backgound_address = c.pnanovdb_root_get_background_address(c.PNANOVDB_GRID_TYPE_FLOAT, buf, root_handle);
    const raw_background = c.pnanovdb_read_float(buf, backgound_address);
    const background_value = if (is_fog) raw_background else raw_background / voxel_size;
    try appendValue(&picovdb_file.data_buffer, allocator, background_value, value_type); // 0
    const inside_value: f32 = if (is_fog) 1.0 else -background_value;
    try appendValue(&picovdb_file.data_buffer, allocator, inside_value, value_type); // 1

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
            // Recover the 4096-aligned upper origin from the root tile key.
            // coordToKey encodes: iu = x>>12, ju = y>>12, ku = z>>12
            //   key[0] = ku | (ju << 21)
            //   key[1] = (iu << 10) | (ju >> 11)
            const pico_key = pico_root.key;
            const ku = pico_key[0] & 0x1FFFFF;
            const ju = (pico_key[0] >> 21) | ((pico_key[1] & 0x3FF) << 11);
            const iu = pico_key[1] >> 10;
            const upper_origin = [3]i32{
                @bitCast(iu << 12),
                @bitCast(ju << 12),
                @bitCast(ku << 12),
            };
            try convertUpperNodesFromHandle(allocator, buf, grid_type, upper_handle, root_handle, picovdb_file, picovdb_grid, upper_origin, value_type, voxel_size, is_fog);
        }
    }
}

fn convertUpperNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, upper_handle: c.pnanovdb_upper_handle_t, root_handle: c.pnanovdb_root_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid, upper_origin: [3]i32, value_type: ValueType, voxel_size: f32, is_fog: bool) !void {
    var element_array: [1024]picovdb.PicoVDBNodeElement = undefined;

    // Read value and child masks from NanoVDB
    const value_mask_addr = c.pnanovdb_address_offset(upper_handle.address, c.PNANOVDB_UPPER_OFF_VALUE_MASK);
    const child_mask_addr = c.pnanovdb_address_offset(upper_handle.address, c.PNANOVDB_UPPER_OFF_CHILD_MASK);

    // Base offsets at the start of this node (grid-relative)
    const base_child_offset: u32 = @intCast(picovdb_file.lowers.items.len - picovdb_grid.lower_start);
    const data_elem_size = value_type.elemSize();
    const base_value_offset: u32 = @intCast((picovdb_file.data_buffer.items.len - picovdb_grid.data_start * 16) / data_elem_size);

    // Cumulative local counts within this node
    var local_state_count: u32 = 0; // (value & state) = children
    var local_value_count: u32 = 0; // (value & ~state) = values

    for (0..1024) |i| {
        const nano_child_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(child_mask_addr, @intCast(i * 4)));
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        // Build state and value bitmasks
        var state_word: u32 = 0;
        var value_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;
            const has_nano_child = (nano_child_word >> bit) & 1 != 0;

            if (has_nano_child) {
                state_word |= (@as(u32, 1) << bit);
                value_word |= (@as(u32, 1) << bit);
            } else if (has_nano_value) {
                value_word |= (@as(u32, 1) << bit);
                const value_address = c.pnanovdb_upper_get_table_address(grid_type, buf, upper_handle, n);
                const raw = c.pnanovdb_read_float(buf, value_address);
                const value = if (is_fog) raw else raw / voxel_size;
                try appendValue(&picovdb_file.data_buffer, allocator, value, value_type);
            } else {
                const value_address = c.pnanovdb_upper_get_table_address(grid_type, buf, upper_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                if (value < 0.0) {
                    state_word |= (@as(u32, 1) << bit);
                }
            }
        }

        // Store mask with packed local index (counts from preceding words)
        element_array[i] = picovdb.PicoVDBNodeElement{
            .state_mask = state_word,
            .value_mask = value_word,
            .packed_local_index = (local_state_count << 16) | local_value_count,
        };

        // Update cumulative counts after storing (so they reflect preceding words only)
        local_state_count += @popCount(value_word & state_word);
        local_value_count += @popCount(value_word & ~state_word);

        // Process children after building the mask
        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            if ((nano_child_word >> bit) & 1 != 0) {
                const n = i * 32 + bit_index;
                const lower_handle = c.pnanovdb_upper_get_child(grid_type, buf, upper_handle, @intCast(n));
                // Compute lower node origin from upper origin + child offset
                // Upper coord layout: child_x = (n >> 10) & 31, child_y = (n >> 5) & 31, child_z = n & 31
                const lower_origin = [3]i32{
                    upper_origin[0] + @as(i32, @intCast((n >> 10) & 31)) * 128,
                    upper_origin[1] + @as(i32, @intCast((n >> 5) & 31)) * 128,
                    upper_origin[2] + @as(i32, @intCast(n & 31)) * 128,
                };
                try convertLowerNodesFromHandle(allocator, buf, grid_type, lower_handle, root_handle, picovdb_file, picovdb_grid, lower_origin, value_type, voxel_size, is_fog);
            }
        }
    }

    const pico_upper = picovdb.PicoVDBUpper{
        .base_inside_index = base_child_offset,
        .base_active_index = base_value_offset,
        .elements = element_array,
    };

    try picovdb_file.uppers.append(allocator, pico_upper);
}

fn convertLowerNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, lower_handle: c.pnanovdb_lower_handle_t, root_handle: c.pnanovdb_root_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid, lower_origin: [3]i32, value_type: ValueType, voxel_size: f32, is_fog: bool) !void {
    var element_array: [128]picovdb.PicoVDBNodeElement = undefined;

    const value_mask_addr = c.pnanovdb_address_offset(lower_handle.address, c.PNANOVDB_LOWER_OFF_VALUE_MASK);
    const child_mask_addr = c.pnanovdb_address_offset(lower_handle.address, c.PNANOVDB_LOWER_OFF_CHILD_MASK);

    // Base offsets at the start of this node (grid-relative)
    const base_child_offset: u32 = @intCast(picovdb_file.leaves.items.len - picovdb_grid.leaf_start);
    const data_elem_size = value_type.elemSize();
    const base_value_offset: u32 = @intCast((picovdb_file.data_buffer.items.len - picovdb_grid.data_start * 16) / data_elem_size);

    var local_state_count: u32 = 0;
    var local_value_count: u32 = 0;

    for (0..128) |i| {
        const nano_child_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(child_mask_addr, @intCast(i * 4)));
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        var state_word: u32 = 0;
        var value_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;
            const has_nano_child = (nano_child_word >> bit) & 1 != 0;

            if (has_nano_child) {
                state_word |= (@as(u32, 1) << bit);
                value_word |= (@as(u32, 1) << bit);
            } else if (has_nano_value) {
                value_word |= (@as(u32, 1) << bit);
                const value_address = c.pnanovdb_lower_get_table_address(grid_type, buf, lower_handle, n);
                const raw = c.pnanovdb_read_float(buf, value_address);
                const value = if (is_fog) raw else raw / voxel_size;
                try appendValue(&picovdb_file.data_buffer, allocator, value, value_type);
            } else {
                const value_address = c.pnanovdb_lower_get_table_address(grid_type, buf, lower_handle, n);
                const value = c.pnanovdb_read_float(buf, value_address);
                if (value < 0.0) {
                    state_word |= (@as(u32, 1) << bit);
                }
            }
        }

        element_array[i] = picovdb.PicoVDBNodeElement{
            .state_mask = state_word,
            .value_mask = value_word,
            .packed_local_index = (local_state_count << 16) | local_value_count,
        };

        local_state_count += @popCount(value_word & state_word);
        local_value_count += @popCount(value_word & ~state_word);

        // Process children after building the mask
        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            if ((nano_child_word >> bit) & 1 != 0) {
                const n = i * 32 + bit_index;
                const leaf_handle = c.pnanovdb_lower_get_child(grid_type, buf, lower_handle, @intCast(n));
                // Compute leaf origin from lower origin + child offset
                // Lower coord layout: child_x = (n >> 8) & 15, child_y = (n >> 4) & 15, child_z = n & 15
                const leaf_origin = [3]i32{
                    lower_origin[0] + @as(i32, @intCast((n >> 8) & 15)) * 8,
                    lower_origin[1] + @as(i32, @intCast((n >> 4) & 15)) * 8,
                    lower_origin[2] + @as(i32, @intCast(n & 15)) * 8,
                };
                try convertLeafNodesFromHandle(allocator, buf, grid_type, leaf_handle, root_handle, picovdb_file, picovdb_grid, leaf_origin, value_type, voxel_size, is_fog);
            }
        }
    }

    const pico_lower = picovdb.PicoVDBLower{
        .base_inside_index = base_child_offset,
        .base_active_index = base_value_offset,
        .elements = element_array,
    };

    try picovdb_file.lowers.append(allocator, pico_lower);
}

fn convertLeafNodesFromHandle(allocator: std.mem.Allocator, buf: c.pnanovdb_buf_t, grid_type: u32, leaf_handle: c.pnanovdb_leaf_handle_t, root_handle: c.pnanovdb_root_handle_t, picovdb_file: *picovdb.PicoVDBFileMutable, picovdb_grid: *picovdb.PicoVDBGrid, leaf_origin: [3]i32, value_type: ValueType, voxel_size: f32, is_fog: bool) !void {
    var element_array: [16]picovdb.PicoVDBNodeElement = undefined;

    const value_mask_addr = c.pnanovdb_address_offset(leaf_handle.address, c.PNANOVDB_LEAF_OFF_VALUE_MASK);

    // Phase 1: Read all 512 values and build value/state masks
    var values: [512]f32 = undefined;
    var value_bits: [16]u32 = undefined;
    var state_bits: [16]u32 = undefined;

    for (0..16) |i| {
        const nano_value_word = c.pnanovdb_read_uint32(buf, c.pnanovdb_address_offset(value_mask_addr, @intCast(i * 4)));

        var value_word: u32 = 0;
        var state_word: u32 = 0;

        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            const n: u32 = @intCast(i * 32 + bit_index);
            const has_nano_value = (nano_value_word >> bit) & 1 != 0;

            const value_addr = c.pnanovdb_leaf_get_table_address(grid_type, buf, leaf_handle, n);
            const raw = c.pnanovdb_read_float(buf, value_addr);
            const value = if (is_fog) raw else raw / voxel_size;
            values[n] = value;

            if (has_nano_value) {
                value_word |= (@as(u32, 1) << bit);
            } else {
                if (value < 0.0) {
                    state_word |= (@as(u32, 1) << bit);
                }
            }
        }

        value_bits[i] = value_word;
        state_bits[i] = state_word; // Initially only marks inactive negative voxels
    }

    // Phase 2: Cross-over detection — mark surface voxels (value=1, state=1).
    // Skipped for fog grids: fog values are >= 0, so no sign crossings exist.
    // This avoids 7 NanoVDB accessor calls per active voxel for fog.
    if (!is_fog) {
        var nano_accessor = std.mem.zeroes(c.pnanovdb_readaccessor_t);
        c.pnanovdb_readaccessor_init(&nano_accessor, root_handle);

        const neighbor_offsets = [7][3]i32{
            .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 },
            .{ 1, 1, 0 }, .{ 1, 0, 1 }, .{ 0, 1, 1 },
            .{ 1, 1, 1 },
        };

        for (0..16) |i| {
            var surface_word: u32 = 0;
            const value_word = value_bits[i];

            for (0..32) |bit_index| {
                const bit: u5 = @intCast(bit_index);
                if ((value_word >> bit) & 1 == 0) continue;

                const n: u32 = @intCast(i * 32 + bit_index);
                const value = values[n];

                // Decode linear offset n back to local (x, y, z) within 8x8x8 leaf
                // n = x*64 + y*8 + z
                const lx: i32 = @intCast(n >> 6);
                const ly: i32 = @intCast((n >> 3) & 7);
                const lz: i32 = @intCast(n & 7);

                var is_surface = false;
                for (neighbor_offsets) |off| {
                    const nx = lx + off[0];
                    const ny = ly + off[1];
                    const nz = lz + off[2];

                    var neighbor_value: f32 = undefined;
                    if (nx >= 0 and nx < 8 and ny >= 0 and ny < 8 and nz >= 0 and nz < 8) {
                        // Neighbor is within this leaf — read directly from values array
                        const nn: u32 = @intCast(@as(u32, @intCast(nx)) * 64 + @as(u32, @intCast(ny)) * 8 + @as(u32, @intCast(nz)));
                        neighbor_value = values[nn];
                    } else {
                        // Cross-leaf lookup via NanoVDB accessor using origin from tree traversal
                        const global_coord = c.pnanovdb_coord_t{
                            .x = leaf_origin[0] + nx,
                            .y = leaf_origin[1] + ny,
                            .z = leaf_origin[2] + nz,
                        };
                        const neighbor_addr = c.pnanovdb_readaccessor_get_value_address(grid_type, buf, &nano_accessor, &global_coord);
                        neighbor_value = c.pnanovdb_read_float(buf, neighbor_addr) / voxel_size;
                    }

                    const sign_strict = (value < 0.0) != (neighbor_value < 0.0);
                    const sign_nonstrict = (value <= 0.0) != (neighbor_value <= 0.0);
                    if (sign_strict or sign_nonstrict) {
                        is_surface = true;
                        break;
                    }
                }

                if (is_surface) {
                    surface_word |= (@as(u32, 1) << bit);
                }
            }

            // For value voxels: state_bits marks surface voxels (value & state = surface)
            // For non-value voxels: state_bits already marks negative (inside implicit)
            state_bits[i] = (state_bits[i] & ~value_bits[i]) | (surface_word & value_bits[i]);
        }
    }

    // Phase 3: Build elements and write values
    const data_elem_size = value_type.elemSize();
    const base_value_offset: u32 = @intCast((picovdb_file.data_buffer.items.len - picovdb_grid.data_start * 16) / data_elem_size);

    var local_value_count: u32 = 0;
    var local_state_count: u32 = 0;

    for (0..16) |i| {
        element_array[i] = picovdb.PicoVDBNodeElement{
            .state_mask = state_bits[i],
            .value_mask = value_bits[i],
            .packed_local_index = (local_state_count << 16) | local_value_count,
        };

        local_value_count += @popCount(value_bits[i]);
        local_state_count += @popCount(value_bits[i] & state_bits[i]);

        // Write SDF values for ALL value voxels (both surface and non-surface)
        for (0..32) |bit_index| {
            const bit: u5 = @intCast(bit_index);
            if ((value_bits[i] >> bit) & 1 != 0) {
                const n: u32 = @intCast(i * 32 + bit_index);
                const value = values[n];
                try appendValue(&picovdb_file.data_buffer, allocator, value, value_type);
            }
        }
    }

    const pico_leaf = picovdb.PicoVDBLeaf{
        .base_inside_index = 0, // Will be set in post-pass by fixupLeafSurfaceIndices
        .base_active_index = base_value_offset,
        .elements = element_array,
    };

    try picovdb_file.leaves.append(allocator, pico_leaf);
}

fn convertGrid(allocator: std.mem.Allocator, buffer: []const u8, offset: usize, picovdb_file: *picovdb.PicoVDBFileMutable, value_type: ValueType, is_fog: bool) !usize {
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

    // Read voxelSize from the grid's voxel-to-world transform (diagonal element)
    const grid_handle = c.pnanovdb_grid_handle_t{ .address = c.pnanovdb_address_t{ .byte_offset = 0 } };
    const voxel_size: f32 = @floatCast(c.pnanovdb_grid_get_voxel_size(pnanovdb_buf, grid_handle, 0));
    std.debug.print("  Voxel size: {d:.6}\n", .{voxel_size});

    const root_handle = c.pnanovdb_tree_get_root(pnanovdb_buf, tree_handle);

    const index_bbox_min = c.pnanovdb_root_get_bbox_min(pnanovdb_buf, root_handle);
    const index_bbox_max = c.pnanovdb_root_get_bbox_max(pnanovdb_buf, root_handle);

    // Calculate data_start in 16-byte units (current data buffer length / 16)
    const data_start_bytes = picovdb_file.data_buffer.items.len;
    std.debug.assert(data_start_bytes % 16 == 0); // Must be 16-byte aligned

    // Create PicoVDB grid
    var picovdb_grid = picovdb.PicoVDBGrid{
        // The accessor resolves a grid's root-tile range via grids[grid_index + 1],
        // so this must be the position in the output grids array.
        .grid_index = @intCast(picovdb_file.grids.items.len),
        .upper_start = @intCast(picovdb_file.uppers.items.len), // Current upper array length (= root start)
        .lower_start = @intCast(picovdb_file.lowers.items.len), // Current lower array length
        .leaf_start = @intCast(picovdb_file.leaves.items.len), // Current leaf array length
        .data_start = @intCast(data_start_bytes / 16), // 16-byte index into data buffer
        .data_elem_count = 0, // Will be set after conversion
        .grid_type = if (is_fog) picovdb.GRID_TYPE_FOG_FLOAT else value_type.gridType(),
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

    try convertRootTiles(allocator, pnanovdb_buf, tree_handle, picovdb_file, &picovdb_grid, value_type, voxel_size, is_fog);

    // Post-pass: fixup leaf base_inside_index for surface texture indexing
    {
        var surface_count: u64 = 0;
        const leaf_start = picovdb_grid.leaf_start;
        const leaf_end = picovdb_file.leaves.items.len;
        for (leaf_start..leaf_end) |i| {
            picovdb_file.leaves.items[i].base_inside_index = surface_count;
            // Count surface voxels (value & state) in this leaf
            for (0..16) |j| {
                const elem = picovdb_file.leaves.items[i].elements[j];
                surface_count += @popCount(elem.value_mask & elem.state_mask);
            }
        }
        std.debug.print("  Surface voxels: {}\n", .{surface_count});
    }

    // Calculate data_elem_count (number of data elements for this grid)
    const data_end_bytes = picovdb_file.data_buffer.items.len;
    picovdb_grid.data_elem_count = @intCast((data_end_bytes - data_start_bytes) / value_type.elemSize());

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

// NanoVDB grid class constants (from PNanoVDB.h)
const PNANOVDB_GRID_CLASS_FOG_VOLUME = 2;

fn convertGridWithMetadata(allocator: std.mem.Allocator, buffer: []const u8, offset: usize, picovdb_file: *picovdb.PicoVDBFileMutable, value_type: ValueType) !usize {
    if (buffer.len < offset + @sizeOf(NanoVDBFileMetaData)) {
        return error.BufferTooSmall;
    }

    // Skip FileMetaData (160 bytes) and grid name to get to actual grid data
    const metadata_ptr: *const NanoVDBFileMetaData = @ptrCast(@alignCast(buffer.ptr + offset));
    const grid_offset = offset + @sizeOf(NanoVDBFileMetaData) + metadata_ptr.name_size;

    // Detect fog volume from grid class
    const is_fog = (metadata_ptr.grid_class == PNANOVDB_GRID_CLASS_FOG_VOLUME);
    std.debug.print("  Grid class: {} {s}\n", .{ metadata_ptr.grid_class, if (is_fog) "(fog)" else "(level set or other)" });
    std.debug.print("  Skipping metadata ({} bytes) + name ({} bytes)\n", .{ @sizeOf(NanoVDBFileMetaData), metadata_ptr.name_size });

    // Convert the grid using unified grid parsing - grid will determine its own size
    return try convertGrid(allocator, buffer, grid_offset, picovdb_file, value_type, is_fog);
}

fn writePicoVDBFile(io: std.Io, dst_path: []const u8, picovdb_file: *picovdb.PicoVDBFileMutable) !void {
    std.debug.print("\n=== Writing PicoVDB File ===\n", .{});

    const cwd = std.Io.Dir.cwd();
    const dst_file = cwd.createFile(io, dst_path, .{}) catch |err| {
        std.debug.print("Error: Could not create output file '{s}': {}\n", .{ dst_path, err });
        return;
    };
    defer dst_file.close(io);

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
    try dst_file.writeStreamingAll(io, header_bytes);

    // Write grids
    const grids_bytes = std.mem.sliceAsBytes(picovdb_file.grids.items);
    try dst_file.writeStreamingAll(io, grids_bytes);

    // Write roots (padded to 16-byte alignment via even count)
    const roots_bytes = std.mem.sliceAsBytes(picovdb_file.roots.items);
    try dst_file.writeStreamingAll(io, roots_bytes);
    if (root_needs_padding) {
        // Add padding root for 16-byte alignment
        const padding_root = picovdb.PicoVDBRoot{ .key = [2]u32{ 0, 0 } };
        try dst_file.writeStreamingAll(io, std.mem.asBytes(&padding_root));
    }

    // Write uppers
    const uppers_bytes = std.mem.sliceAsBytes(picovdb_file.uppers.items);
    try dst_file.writeStreamingAll(io, uppers_bytes);

    // Write lowers
    const lowers_bytes = std.mem.sliceAsBytes(picovdb_file.lowers.items);
    try dst_file.writeStreamingAll(io, lowers_bytes);

    // Write leaves
    const leaves_bytes = std.mem.sliceAsBytes(picovdb_file.leaves.items);
    try dst_file.writeStreamingAll(io, leaves_bytes);

    // Write data buffer (padded to 16 bytes)
    try dst_file.writeStreamingAll(io, picovdb_file.data_buffer.items);
    const data_padding = data_size_padded - data_size;
    if (data_padding > 0) {
        const padding = [_]u8{0} ** 16;
        try dst_file.writeStreamingAll(io, padding[0..data_padding]);
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
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, test_file, .{});
    defer file.close(io);

    const file_size = (try file.stat(io)).size;
    const nvdb_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 4));
    defer allocator.free(nvdb_buffer);
    _ = try file.readPositionalAll(io, nvdb_buffer, 0);

    // Convert to PicoVDB format
    var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file_mutable.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &picovdb_file_mutable, .f32);

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
        const io = std.testing.io;
        const cwd = std.Io.Dir.cwd();
        const file = try cwd.openFile(io, test_file, .{});
        defer file.close(io);

        const file_size = (try file.stat(io)).size;
        const buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 16));
        defer allocator.free(buffer);
        _ = try file.readPositionalAll(io, buffer, 0);

        std.log.info("Using test file: {s} ({} bytes)", .{ test_file, file_size });

        // Convert to PicoVDB format
        var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
        defer picovdb_file_mutable.deinit(allocator);

        try convertNanoVDBToPicoVDB(allocator, buffer, &picovdb_file_mutable, .f32);

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

        // Read voxelSize for normalization comparison
        const test_voxel_size: f32 = @floatCast(c.pnanovdb_grid_get_voxel_size(pnano_grid_buf, grid_handle, 0));

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
            const pico_result = pico_accessor.getLevelIndex(coord, grid, &picovdb_file);
            const pico_value = picovdb_file.getGridFloat(grid, pico_result.index);

            // Get value from PNanoVDB
            const ijk = c.pnanovdb_coord_t{ .x = coord[0], .y = coord[1], .z = coord[2] };
            var pnano_level: u32 = 0;
            const pnano_address = c.pnanovdb_readaccessor_get_value_address_and_level(c.PNANOVDB_GRID_TYPE_FLOAT, pnano_grid_buf, &pnano_accessor, &ijk, &pnano_level);
            const pnano_value = c.pnanovdb_read_float(pnano_grid_buf, pnano_address) / test_voxel_size;

            // Compare values (allow small floating point differences)
            const diff = @abs(pico_value - pnano_value);
            const values_match = diff < 1e-6;
            if (values_match) {
                matches += 1;
                //std.log.warn("Match at [{}, {}, {}]: PicoVDB={d:.6} (level={}, count={}), PNanoVDB={d:.6} (level={})", .{ coord[0], coord[1], coord[2], pico_value, pico_result.level, pico_result.index, pnano_value, pnano_level });
            } else {
                //std.log.warn("Mismatch at [{}, {}, {}]: PicoVDB={d:.6} (level={}, count={}), PNanoVDB={d:.6} (level={}), diff={d:.8}", .{ coord[0], coord[1], coord[2], pico_value, pico_result.level, pico_result.index, pnano_value, pnano_level, diff });
            }
        }
        try std.testing.expectEqual(total_tests, matches);
    }
}

test "multi-grid file: second grid reads identically to the first" {
    const allocator = std.testing.allocator;

    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, "data/sphere.nvdb", .{});
    defer file.close(io);

    const file_size = (try file.stat(io)).size;
    const nvdb_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 4));
    defer allocator.free(nvdb_buffer);
    _ = try file.readPositionalAll(io, nvdb_buffer, 0);

    // Convert the same source twice into one file: the second grid gets a
    // non-zero data_start, exercising the grid-relative value offsets.
    var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file_mutable.deinit(allocator);
    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &picovdb_file_mutable, .f32);
    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &picovdb_file_mutable, .f32);

    const picovdb_buffer = try picovdb_file_mutable.encode(allocator);
    defer allocator.free(picovdb_buffer);
    const picovdb_file = try picovdb.PicoVDBFile.fromBytes(picovdb_buffer);

    try std.testing.expectEqual(@as(u32, 2), picovdb_file.header.grid_count);
    const grid0 = &picovdb_file.grids[0];
    const grid1 = &picovdb_file.grids[1];
    try std.testing.expectEqual(@as(u32, 0), grid0.grid_index);
    try std.testing.expectEqual(@as(u32, 1), grid1.grid_index);
    try std.testing.expect(grid1.data_start > 0);
    try std.testing.expect(grid1.upper_start > grid0.upper_start);

    // Identical source grids must produce identical values at every coord.
    var acc0 = picovdb.PicoVDBReadAccessor.init(0);
    var acc1 = picovdb.PicoVDBReadAccessor.init(1);
    var x: i32 = -65;
    while (x <= 65) : (x += 5) {
        var y: i32 = -65;
        while (y <= 65) : (y += 5) {
            var z: i32 = -65;
            while (z <= 65) : (z += 1) {
                const coord = [3]i32{ x, y, z };
                const r0 = acc0.getLevelIndex(coord, grid0, &picovdb_file);
                const r1 = acc1.getLevelIndex(coord, grid1, &picovdb_file);
                try std.testing.expectEqual(r0.level, r1.level);
                const v0 = picovdb_file.getGridFloat(grid0, r0.index);
                const v1 = picovdb_file.getGridFloat(grid1, r1.index);
                try std.testing.expectEqual(v0, v1);
            }
        }
    }
}

test "u8 quantization round-trip" {
    const allocator = std.testing.allocator;

    const test_file = "data/sphere.nvdb";
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, test_file, .{});
    defer file.close(io);

    const file_size = (try file.stat(io)).size;
    const nvdb_buffer = try allocator.alloc(u8, std.mem.alignForward(usize, file_size, 16));
    defer allocator.free(nvdb_buffer);
    _ = try file.readPositionalAll(io, nvdb_buffer, 0);

    // Convert with u8 encoding
    var picovdb_file_mutable = picovdb.PicoVDBFileMutable.init();
    defer picovdb_file_mutable.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &picovdb_file_mutable, .u8);

    // Encode and decode
    const picovdb_buffer = try picovdb_file_mutable.encode(allocator);
    defer allocator.free(picovdb_buffer);
    const picovdb_file = try picovdb.PicoVDBFile.fromBytes(picovdb_buffer);

    // Verify grid_type
    try std.testing.expectEqual(@as(u32, 1), picovdb_file.header.grid_count);
    const grid = &picovdb_file.grids[0];
    try std.testing.expectEqual(picovdb.GRID_TYPE_SDF_UINT8, grid.grid_type);

    // Also convert with f32 for ground truth comparison (both normalized by voxelSize)
    var f32_file_mutable = picovdb.PicoVDBFileMutable.init();
    defer f32_file_mutable.deinit(allocator);

    try convertNanoVDBToPicoVDB(allocator, nvdb_buffer, &f32_file_mutable, .f32);

    const f32_buffer = try f32_file_mutable.encode(allocator);
    defer allocator.free(f32_buffer);
    const f32_file = try picovdb.PicoVDBFile.fromBytes(f32_buffer);
    const f32_grid = &f32_file.grids[0];

    // Compare every active voxel
    var pico_accessor = picovdb.PicoVDBReadAccessor.init(0);
    var f32_accessor = picovdb.PicoVDBReadAccessor.init(0);

    const max_quant_error = picovdb.LEVEL_SET_HALF_WIDTH / 127.5;
    var max_observed_error: f32 = 0.0;
    var total_tests: u32 = 0;

    // Sample across the bounding box
    const min = grid.index_bounds_min;
    const max = grid.index_bounds_max;

    var x: i32 = min[0];
    while (x <= max[0]) : (x += 1) {
        var y: i32 = min[1];
        while (y <= max[1]) : (y += 1) {
            var z: i32 = min[2];
            while (z <= max[2]) : (z += 1) {
                const coord = [3]i32{ x, y, z };
                const u8_result = pico_accessor.getLevelIndex(coord, grid, &picovdb_file);
                const f32_result = f32_accessor.getLevelIndex(coord, f32_grid, &f32_file);

                // Only compare active voxels (not background/inside implicit)
                if (u8_result.index < 2 and f32_result.index < 2) continue;

                const u8_value = picovdb_file.getGridValue(grid, u8_result.index);
                const f32_value = f32_file.getGridFloat(f32_grid, f32_result.index);

                const err = @abs(u8_value - f32_value);
                if (err > max_observed_error) {
                    max_observed_error = err;
                }
                total_tests += 1;

                try std.testing.expect(err <= max_quant_error + 1e-6);
            }
        }
    }

    std.log.info("u8 round-trip: {} voxels tested, max error: {d:.6} (limit: {d:.6})", .{ total_tests, max_observed_error, max_quant_error });
    try std.testing.expect(total_tests > 0);
}
