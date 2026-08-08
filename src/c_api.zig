//! C ABI. Compiled to wasm32-freestanding for the browser (`zig build wasm`,
//! consumed by stl.ts) and to Apple static libraries (`zig build xcframework`,
//! consumed from Swift). Struct layouts must match include/picovdb.h and the
//! offsets hardcoded in stl.ts; comptime asserts below pin them.

const std = @import("std");
const builtin = @import("builtin");
const picovdb = @import("picovdb");

// The default panic handler pulls in std.debug.SelfInfo, which references
// _dyld_get_image_header_containing_address and fails to link against the iOS
// SDK (and is useless in freestanding wasm). Trap instead.
pub const panic = std.debug.FullPanic(struct {
    fn trap(_: []const u8, _: ?usize) noreturn {
        @trap();
    }
}.trap);

const gpa: std.mem.Allocator = if (builtin.target.cpu.arch.isWasm())
    std.heap.wasm_allocator // requires single_threaded (set in build.zig)
else
    std.heap.c_allocator; // requires link_libc (set in build.zig)

pub const ABI_VERSION: u32 = 1;

// Error codes: 0 is success, negatives map to strings via picovdb_error_string.
pub const ERROR_PARSE: i32 = -1;
pub const ERROR_OOM: i32 = -2;
pub const ERROR_EMPTY_MESH: i32 = -3;
pub const ERROR_NON_FINITE: i32 = -4;
pub const ERROR_BAD_OPTIONS: i32 = -5;
pub const ERROR_TOO_MANY_VOXELS: i32 = -6;

pub const MeshInfo = extern struct {
    triangle_count: u32,
    bbox_min: [3]f32,
    bbox_max: [3]f32,
};

pub const MeshToGridOptions = extern struct {
    /// Fail with ERROR_TOO_MANY_VOXELS if the voxel estimate (mesh bbox
    /// dilated by the narrow band) exceeds this; 0 = unlimited. Peak memory is
    /// roughly 8 bytes per estimated voxel.
    max_voxels: u64,
    /// Grid resolution in voxels per world unit; required, > 0.
    voxels_per_unit: f32,
    /// Narrow band half-width in voxels; 0 selects the default (3.0).
    half_width: f32,
    /// 0 = f32, 1 = u8.
    value_type: u32,
    /// Rotations in degrees applied to the input points about x, then y, then
    /// z; zeros = none.
    rotate_deg: [3]f32,
};

pub const GridStats = extern struct {
    active_voxels: u64,
    surface_voxels: u64,
    leaf_count: u32,
    lower_count: u32,
    upper_count: u32,
    bbox_min: [3]i32,
    bbox_max: [3]i32,
    world_min: [3]f32,
    world_max: [3]f32,
};

pub const Buffer = extern struct {
    data: ?[*]const u8,
    len: usize,
    stats: GridStats,
};

comptime {
    std.debug.assert(@sizeOf(MeshInfo) == 28);
    std.debug.assert(@sizeOf(MeshToGridOptions) == 32);
    std.debug.assert(@offsetOf(MeshToGridOptions, "voxels_per_unit") == 8);
    if (builtin.target.cpu.arch == .wasm32) {
        // stl.ts hardcodes these offsets.
        std.debug.assert(@offsetOf(Buffer, "stats") == 8);
        std.debug.assert(@sizeOf(Buffer) == 88);
    }
}

export fn picovdb_abi_version() u32 {
    return ABI_VERSION;
}

/// For the wasm caller to stage input bytes inside linear memory. Native
/// callers can pass their own pointers and ignore these.
export fn picovdb_alloc(len: usize) ?[*]u8 {
    const buf = gpa.alloc(u8, len) catch return null;
    return buf.ptr;
}

export fn picovdb_free(ptr: [*]u8, len: usize) void {
    gpa.free(ptr[0..len]);
}

export fn picovdb_buffer_free(buf: *Buffer) void {
    if (buf.data) |data| {
        const slice: []align(4) const u8 = @alignCast(data[0..buf.len]);
        gpa.free(slice);
        buf.data = null;
        buf.len = 0;
    }
}

export fn picovdb_error_string(code: i32) [*:0]const u8 {
    return switch (code) {
        0 => "ok",
        ERROR_PARSE => "failed to parse STL",
        ERROR_OOM => "out of memory",
        ERROR_EMPTY_MESH => "mesh has no triangles or no active voxels",
        ERROR_NON_FINITE => "mesh contains non-finite vertices",
        ERROR_BAD_OPTIONS => "invalid options",
        ERROR_TOO_MANY_VOXELS => "estimated voxel count exceeds max_voxels",
        else => "unknown error",
    };
}

/// Cheap pre-voxelization pass over an STL: triangle count + world bounds.
export fn picovdb_stl_info(bytes: [*]const u8, len: usize, out: *MeshInfo) i32 {
    var mesh = picovdb.stl.parse(gpa, bytes[0..len]) catch |err| return mapStlError(err);
    defer mesh.deinit(gpa);
    if (mesh.vertices.len < 3) return ERROR_EMPTY_MESH;
    const b = mesh.bounds();
    out.* = .{
        .triangle_count = @intCast(mesh.triangleCount()),
        .bbox_min = b[0],
        .bbox_max = b[1],
    };
    return 0;
}

/// Rasterize a triangle mesh into an encoded .pvdb narrow-band SDF grid.
/// `points` are xyz triples (world units), `triangles` are vertex index
/// triples. On success out.data/out.len hold the encoded file (release with
/// picovdb_buffer_free) and out.stats is filled.
export fn picovdb_mesh_to_grid(
    points: [*]const f32,
    point_count: u32,
    triangles: [*]const u32,
    triangle_count: u32,
    opts: *const MeshToGridOptions,
    out: *Buffer,
) i32 {
    out.* = std.mem.zeroes(Buffer);
    if (!validOptions(opts)) return ERROR_BAD_OPTIONS;
    const vertices = points[0 .. @as(usize, point_count) * 3];
    const indices = triangles[0 .. @as(usize, triangle_count) * 3];
    if (std.mem.eql(f32, &opts.rotate_deg, &.{ 0, 0, 0 }))
        return meshToGrid(vertices, indices, opts, out);
    const rotated = gpa.dupe(f32, vertices) catch return ERROR_OOM;
    defer gpa.free(rotated);
    rotate(rotated, opts.rotate_deg);
    return meshToGrid(rotated, indices, opts, out);
}

/// One-shot STL (binary or ASCII) -> encoded .pvdb; see picovdb_mesh_to_grid.
export fn picovdb_stl_to_grid(bytes: [*]const u8, len: usize, opts: *const MeshToGridOptions, out: *Buffer) i32 {
    out.* = std.mem.zeroes(Buffer);
    if (!validOptions(opts)) return ERROR_BAD_OPTIONS;
    var mesh = picovdb.stl.parse(gpa, bytes[0..len]) catch |err| return mapStlError(err);
    defer mesh.deinit(gpa);
    rotate(mesh.vertices, opts.rotate_deg);
    return meshToGrid(mesh.vertices, mesh.triangles, opts, out);
}

fn validOptions(opts: *const MeshToGridOptions) bool {
    return opts.voxels_per_unit > 0 and opts.half_width >= 0 and opts.value_type <= 1;
}

fn rotate(vertices: []f32, rotate_deg: [3]f32) void {
    for ([3]picovdb.stl.Axis{ .x, .y, .z }, rotate_deg) |axis, deg| {
        if (deg != 0) picovdb.stl.rotateVertices(vertices, axis, deg * std.math.pi / 180.0);
    }
}

fn meshToGrid(vertices: []const f32, triangles: []const u32, opts: *const MeshToGridOptions, out: *Buffer) i32 {
    if (triangles.len < 3 or vertices.len < 9) return ERROR_EMPTY_MESH;

    const hw = if (opts.half_width > 0) opts.half_width else picovdb.LEVEL_SET_HALF_WIDTH;
    const b = picovdb.stl.vertexBounds(vertices);
    if (opts.max_voxels > 0) {
        // Upper bound on the voxel count: bbox dilated by the narrow band
        // (without the dilation the bound is exceeded at coarse resolutions).
        var estimate: f64 = 1;
        for (0..3) |axis| {
            const dim: f64 = b[1][axis] - b[0][axis];
            estimate *= @ceil(dim * opts.voxels_per_unit) + 1 + 2 * @as(f64, hw);
        }
        if (estimate > @as(f64, @floatFromInt(opts.max_voxels))) return ERROR_TOO_MANY_VOXELS;
    }

    var file = picovdb.PicoVDBFileMutable.init();
    defer file.deinit(gpa);

    const stats = picovdb.mesh_to_grid.meshToGrid(gpa, &file, vertices, triangles, .{
        .voxel_size = 1.0 / opts.voxels_per_unit,
        .half_width = hw,
        .value_type = if (opts.value_type == 1) .u8 else .f32,
    }) catch |err| return switch (err) {
        error.EmptyMesh, error.NoActiveVoxels => ERROR_EMPTY_MESH,
        error.NonFiniteVertex => ERROR_NON_FINITE,
        error.OutOfMemory => ERROR_OOM,
    };

    const encoded = file.encode(gpa) catch return ERROR_OOM;
    out.* = .{
        .data = encoded.ptr,
        .len = encoded.len,
        .stats = .{
            .active_voxels = stats.active_voxels,
            .surface_voxels = stats.surface_voxels,
            .leaf_count = stats.leaf_count,
            .lower_count = stats.lower_count,
            .upper_count = stats.upper_count,
            .bbox_min = stats.index_bounds_min,
            .bbox_max = stats.index_bounds_max,
            .world_min = b[0],
            .world_max = b[1],
        },
    };
    return 0;
}

fn mapStlError(err: picovdb.stl.Error) i32 {
    return switch (err) {
        error.OutOfMemory => ERROR_OOM,
        else => ERROR_PARSE,
    };
}

// A minimal one-triangle binary STL: 80-byte header, u32 count, one 50-byte
// triangle record (normal + 3 vertices + attribute).
fn writeTestStl(buf: *[134]u8) void {
    @memset(buf, 0);
    std.mem.writeInt(u32, buf[80..84], 1, .little);
    const verts = [9]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };
    var off: usize = 84 + 12;
    for (verts) |v| {
        std.mem.writeInt(u32, buf[off..][0..4], @bitCast(v), .little);
        off += 4;
    }
}

const test_options = MeshToGridOptions{
    .max_voxels = 0,
    .voxels_per_unit = 4,
    .half_width = 0,
    .value_type = 0,
    .rotate_deg = .{ 0, 0, 0 },
};

test "info and convert roundtrip" {
    var stl_buf: [134]u8 = undefined;
    writeTestStl(&stl_buf);

    var info: MeshInfo = undefined;
    try std.testing.expectEqual(@as(i32, 0), picovdb_stl_info(&stl_buf, stl_buf.len, &info));
    try std.testing.expectEqual(@as(u32, 1), info.triangle_count);
    try std.testing.expectEqual(@as(f32, 1), info.bbox_max[0]);

    var out: Buffer = undefined;
    try std.testing.expectEqual(@as(i32, 0), picovdb_stl_to_grid(&stl_buf, stl_buf.len, &test_options, &out));
    defer picovdb_buffer_free(&out);

    try std.testing.expect(out.stats.active_voxels > 0);
    const data = out.data.?;
    try std.testing.expectEqual(@as(u32, 0x6f636950), std.mem.readInt(u32, data[0..4], .little));
    try std.testing.expectEqual(@as(u32, 0x30424456), std.mem.readInt(u32, data[4..8], .little));
}

test "mesh_to_grid matches stl_to_grid" {
    var stl_buf: [134]u8 = undefined;
    writeTestStl(&stl_buf);
    var from_stl: Buffer = undefined;
    try std.testing.expectEqual(@as(i32, 0), picovdb_stl_to_grid(&stl_buf, stl_buf.len, &test_options, &from_stl));
    defer picovdb_buffer_free(&from_stl);

    const points = [9]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0 };
    const triangles = [3]u32{ 0, 1, 2 };
    var from_mesh: Buffer = undefined;
    try std.testing.expectEqual(@as(i32, 0), picovdb_mesh_to_grid(&points, 3, &triangles, 1, &test_options, &from_mesh));
    defer picovdb_buffer_free(&from_mesh);

    try std.testing.expectEqualSlices(u8, from_stl.data.?[0..from_stl.len], from_mesh.data.?[0..from_mesh.len]);
}

test "error codes" {
    var stl_buf: [134]u8 = undefined;
    writeTestStl(&stl_buf);
    var out: Buffer = undefined;

    const garbage = "not an stl at all, definitely not";
    var info: MeshInfo = undefined;
    try std.testing.expectEqual(ERROR_PARSE, picovdb_stl_info(garbage.ptr, garbage.len, &info));

    var opts = test_options;
    opts.voxels_per_unit = 0;
    try std.testing.expectEqual(ERROR_BAD_OPTIONS, picovdb_stl_to_grid(&stl_buf, stl_buf.len, &opts, &out));
    try std.testing.expect(out.data == null);

    // The unit triangle at vpu=4 dilates to ~12^3 voxels; a limit of 10 must
    // reject it, a generous one must not.
    opts = test_options;
    opts.max_voxels = 10;
    try std.testing.expectEqual(ERROR_TOO_MANY_VOXELS, picovdb_stl_to_grid(&stl_buf, stl_buf.len, &opts, &out));
    try std.testing.expect(out.data == null);
    opts.max_voxels = 1 << 20;
    try std.testing.expectEqual(@as(i32, 0), picovdb_stl_to_grid(&stl_buf, stl_buf.len, &opts, &out));
    picovdb_buffer_free(&out);
}
