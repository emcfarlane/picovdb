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

pub const PV_ABI_VERSION: u32 = 1;

// Error codes: 0 is success, negatives map to strings via pv_error_string.
pub const PV_ERROR_PARSE: i32 = -1;
pub const PV_ERROR_OOM: i32 = -2;
pub const PV_ERROR_EMPTY_MESH: i32 = -3;
pub const PV_ERROR_NON_FINITE: i32 = -4;
pub const PV_ERROR_BAD_OPTIONS: i32 = -5;
pub const PV_ERROR_TOO_MANY_VOXELS: i32 = -6;

pub const PvStlInfo = extern struct {
    triangle_count: u32,
    bbox_min: [3]f32,
    bbox_max: [3]f32,
};

pub const PvMeshOptions = extern struct {
    /// Fail with PV_ERROR_TOO_MANY_VOXELS if the voxel estimate (mesh bbox
    /// dilated by the narrow band) exceeds this; 0 = unlimited. Peak memory is
    /// roughly 8 bytes per estimated voxel.
    max_voxels: u64,
    /// Grid resolution in voxels per world unit; required, > 0.
    voxels_per_unit: f32,
    /// Narrow band half-width in voxels; 0 selects the default (3.0).
    half_width: f32,
    /// 0 = f32, 1 = u8.
    value_type: u32,
    /// Rotations in degrees, applied about x, then y, then z; zeros = none.
    rotate_deg: [3]f32,
};

pub const PvStats = extern struct {
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

pub const PvBuffer = extern struct {
    data: ?[*]const u8,
    len: usize,
    stats: PvStats,
};

comptime {
    std.debug.assert(@sizeOf(PvStlInfo) == 28);
    std.debug.assert(@sizeOf(PvMeshOptions) == 32);
    std.debug.assert(@offsetOf(PvMeshOptions, "voxels_per_unit") == 8);
    if (builtin.target.cpu.arch == .wasm32) {
        // stl.ts hardcodes these offsets.
        std.debug.assert(@offsetOf(PvBuffer, "stats") == 8);
        std.debug.assert(@sizeOf(PvBuffer) == 88);
    }
}

export fn pv_abi_version() u32 {
    return PV_ABI_VERSION;
}

/// For the wasm caller to stage input bytes inside linear memory. Native
/// callers can pass their own pointers and ignore these.
export fn pv_alloc(len: usize) ?[*]u8 {
    const buf = gpa.alloc(u8, len) catch return null;
    return buf.ptr;
}

export fn pv_dealloc(ptr: [*]u8, len: usize) void {
    gpa.free(ptr[0..len]);
}

export fn pv_buffer_free(buf: *PvBuffer) void {
    if (buf.data) |data| {
        const slice: []align(4) const u8 = @alignCast(data[0..buf.len]);
        gpa.free(slice);
        buf.data = null;
        buf.len = 0;
    }
}

export fn pv_error_string(code: i32) [*:0]const u8 {
    return switch (code) {
        0 => "ok",
        PV_ERROR_PARSE => "failed to parse STL",
        PV_ERROR_OOM => "out of memory",
        PV_ERROR_EMPTY_MESH => "mesh has no triangles or no active voxels",
        PV_ERROR_NON_FINITE => "mesh contains non-finite vertices",
        PV_ERROR_BAD_OPTIONS => "invalid options",
        PV_ERROR_TOO_MANY_VOXELS => "estimated voxel count exceeds max_voxels",
        else => "unknown error",
    };
}

/// Cheap pre-voxelization pass: triangle count + world bounds, for UI preview
/// and the voxel-count estimate (docs/stl-c-library-plan.md section 1a).
export fn pv_stl_get_info(bytes: [*]const u8, len: usize, out: *PvStlInfo) i32 {
    var mesh = picovdb.stl.parse(gpa, bytes[0..len]) catch |err| return mapStlError(err);
    defer mesh.deinit(gpa);
    if (mesh.vertices.len < 3) return PV_ERROR_EMPTY_MESH;
    const b = mesh.bounds();
    out.* = .{
        .triangle_count = @intCast(mesh.triangleCount()),
        .bbox_min = b[0],
        .bbox_max = b[1],
    };
    return 0;
}

/// One-shot STL -> encoded .pvdb. On success out.data/out.len hold the encoded
/// file (release with pv_buffer_free) and out.stats is filled.
export fn pv_stl_to_pvdb(bytes: [*]const u8, len: usize, opts: *const PvMeshOptions, out: *PvBuffer) i32 {
    out.* = std.mem.zeroes(PvBuffer);
    if (!(opts.voxels_per_unit > 0) or opts.half_width < 0 or opts.value_type > 1)
        return PV_ERROR_BAD_OPTIONS;

    var mesh = picovdb.stl.parse(gpa, bytes[0..len]) catch |err| return mapStlError(err);
    defer mesh.deinit(gpa);
    if (mesh.triangles.len < 3 or mesh.vertices.len < 9) return PV_ERROR_EMPTY_MESH;

    for ([3]picovdb.stl.Axis{ .x, .y, .z }, opts.rotate_deg) |axis, deg| {
        if (deg != 0) mesh.rotate(axis, deg * std.math.pi / 180.0);
    }

    const hw = if (opts.half_width > 0) opts.half_width else picovdb.LEVEL_SET_HALF_WIDTH;
    if (opts.max_voxels > 0) {
        // Upper bound on the voxel count: bbox dilated by the narrow band
        // (without the dilation the bound is exceeded at coarse resolutions).
        const b = mesh.bounds();
        var estimate: f64 = 1;
        for (0..3) |axis| {
            const dim: f64 = b[1][axis] - b[0][axis];
            estimate *= @ceil(dim * opts.voxels_per_unit) + 1 + 2 * @as(f64, hw);
        }
        if (estimate > @as(f64, @floatFromInt(opts.max_voxels))) return PV_ERROR_TOO_MANY_VOXELS;
    }

    var file = picovdb.PicoVDBFileMutable.init();
    defer file.deinit(gpa);

    const stats = picovdb.mesh2ls.meshToPicoVDB(gpa, &file, mesh.vertices, mesh.triangles, .{
        .voxel_size = 1.0 / opts.voxels_per_unit,
        .half_width = hw,
        .value_type = if (opts.value_type == 1) .u8 else .f32,
    }) catch |err| return switch (err) {
        error.EmptyMesh, error.NoActiveVoxels => PV_ERROR_EMPTY_MESH,
        error.NonFiniteVertex => PV_ERROR_NON_FINITE,
        error.OutOfMemory => PV_ERROR_OOM,
    };

    const b = mesh.bounds(); // post-rotation, matches the emitted grid
    const encoded = file.encode(gpa) catch return PV_ERROR_OOM;
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
        error.OutOfMemory => PV_ERROR_OOM,
        else => PV_ERROR_PARSE,
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

test "info and convert roundtrip" {
    var stl_buf: [134]u8 = undefined;
    writeTestStl(&stl_buf);

    var info: PvStlInfo = undefined;
    try std.testing.expectEqual(@as(i32, 0), pv_stl_get_info(&stl_buf, stl_buf.len, &info));
    try std.testing.expectEqual(@as(u32, 1), info.triangle_count);
    try std.testing.expectEqual(@as(f32, 1), info.bbox_max[0]);

    const opts = PvMeshOptions{
        .max_voxels = 0,
        .voxels_per_unit = 4,
        .half_width = 0,
        .value_type = 0,
        .rotate_deg = .{ 0, 0, 0 },
    };
    var out: PvBuffer = undefined;
    try std.testing.expectEqual(@as(i32, 0), pv_stl_to_pvdb(&stl_buf, stl_buf.len, &opts, &out));
    defer pv_buffer_free(&out);

    try std.testing.expect(out.stats.active_voxels > 0);
    const data = out.data.?;
    try std.testing.expectEqual(@as(u32, 0x6f636950), std.mem.readInt(u32, data[0..4], .little));
    try std.testing.expectEqual(@as(u32, 0x30424456), std.mem.readInt(u32, data[4..8], .little));
}

test "error codes" {
    var stl_buf: [134]u8 = undefined;
    writeTestStl(&stl_buf);
    var out: PvBuffer = undefined;

    const garbage = "not an stl at all, definitely not";
    var info: PvStlInfo = undefined;
    try std.testing.expectEqual(PV_ERROR_PARSE, pv_stl_get_info(garbage.ptr, garbage.len, &info));

    const bad = PvMeshOptions{ .max_voxels = 0, .voxels_per_unit = 0, .half_width = 0, .value_type = 0, .rotate_deg = .{ 0, 0, 0 } };
    try std.testing.expectEqual(PV_ERROR_BAD_OPTIONS, pv_stl_to_pvdb(&stl_buf, stl_buf.len, &bad, &out));
    try std.testing.expect(out.data == null);

    // The unit triangle at vpu=4 dilates to ~12^3 voxels; a limit of 10 must
    // reject it, a generous one must not.
    var limited = PvMeshOptions{ .max_voxels = 10, .voxels_per_unit = 4, .half_width = 0, .value_type = 0, .rotate_deg = .{ 0, 0, 0 } };
    try std.testing.expectEqual(PV_ERROR_TOO_MANY_VOXELS, pv_stl_to_pvdb(&stl_buf, stl_buf.len, &limited, &out));
    try std.testing.expect(out.data == null);
    limited.max_voxels = 1 << 20;
    try std.testing.expectEqual(@as(i32, 0), pv_stl_to_pvdb(&stl_buf, stl_buf.len, &limited, &out));
    pv_buffer_free(&out);
}
