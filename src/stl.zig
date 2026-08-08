//! STL mesh parsing (binary and ASCII).
//! Produces flat triangle-soup arrays suitable for voxelization; no vertex
//! deduplication is performed.

const std = @import("std");

pub const Error = error{
    InvalidFormat,
    UnsupportedPolygon,
    InvalidCharacter,
} || std.mem.Allocator.Error;

pub const Axis = enum { x, y, z };

pub const Mesh = struct {
    vertices: []f32, // xyz triples
    triangles: []u32, // vertex index triples

    pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
        allocator.free(self.triangles);
        self.* = undefined;
    }

    pub fn vertexCount(self: *const Mesh) usize {
        return self.vertices.len / 3;
    }

    pub fn triangleCount(self: *const Mesh) usize {
        return self.triangles.len / 3;
    }

    /// Axis-aligned bounds over all vertices as .{ min, max }.
    /// Asserts the mesh is non-empty.
    pub fn bounds(self: *const Mesh) [2][3]f32 {
        return vertexBounds(self.vertices);
    }

    /// Rotate all vertices about `axis` by `radians` (right-handed).
    /// E.g. rotate(.x, -pi/2) re-orients a Z-up mesh to the Y-up convention.
    pub fn rotate(self: *Mesh, axis: Axis, radians: f32) void {
        rotateVertices(self.vertices, axis, radians);
    }
};

/// Axis-aligned bounds over xyz vertex triples as .{ min, max }.
/// Asserts the slice is non-empty.
pub fn vertexBounds(vertices: []const f32) [2][3]f32 {
    std.debug.assert(vertices.len >= 3);
    var min = [3]f32{ vertices[0], vertices[1], vertices[2] };
    var max = min;
    var i: usize = 3;
    while (i < vertices.len) : (i += 3) {
        for (0..3) |axis| {
            min[axis] = @min(min[axis], vertices[i + axis]);
            max[axis] = @max(max[axis], vertices[i + axis]);
        }
    }
    return .{ min, max };
}

/// Rotate xyz vertex triples about `axis` by `radians` (right-handed).
pub fn rotateVertices(vertices: []f32, axis: Axis, radians: f32) void {
    const c = @cos(radians);
    const s = @sin(radians);
    // The two rotated components in cyclic order; the axis component is
    // unchanged: x -> (y, z), y -> (z, x), z -> (x, y).
    const u: usize, const v: usize = switch (axis) {
        .x => .{ 1, 2 },
        .y => .{ 2, 0 },
        .z => .{ 0, 1 },
    };
    var i: usize = 0;
    while (i < vertices.len) : (i += 3) {
        const a = vertices[i + u];
        const b = vertices[i + v];
        vertices[i + u] = a * c - b * s;
        vertices[i + v] = a * s + b * c;
    }
}

/// Parse an STL file from memory. Detects binary vs ASCII automatically.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) Error!Mesh {
    // Binary detection first: the size formula is exact, whereas "solid" can
    // legitimately appear in a binary file's 80-byte header.
    if (bytes.len >= 84) {
        const tri_count = std.mem.readInt(u32, bytes[80..84], .little);
        if (bytes.len == 84 + 50 * @as(u64, tri_count)) {
            return parseBinary(allocator, bytes, tri_count);
        }
    }
    if (std.mem.startsWith(u8, bytes, "solid")) {
        return parseAscii(allocator, bytes);
    }
    return error.InvalidFormat;
}

fn parseBinary(allocator: std.mem.Allocator, bytes: []const u8, tri_count: u32) Error!Mesh {
    const vertices = try allocator.alloc(f32, @as(usize, tri_count) * 9);
    errdefer allocator.free(vertices);
    const triangles = try allocator.alloc(u32, @as(usize, tri_count) * 3);
    errdefer allocator.free(triangles);

    for (0..tri_count) |i| {
        // 50-byte record: normal (3xf32, skipped), 3 vertices (9xf32), u16 attribute.
        const record = bytes[84 + 50 * i ..][0..50];
        for (0..9) |j| {
            const bits = std.mem.readInt(u32, record[12 + j * 4 ..][0..4], .little);
            vertices[i * 9 + j] = @bitCast(bits);
        }
        for (0..3) |j| {
            triangles[i * 3 + j] = @intCast(i * 3 + j);
        }
    }

    return Mesh{ .vertices = vertices, .triangles = triangles };
}

fn parseAscii(allocator: std.mem.Allocator, bytes: []const u8) Error!Mesh {
    var vertices: std.ArrayList(f32) = .empty;
    defer vertices.deinit(allocator);
    var triangles: std.ArrayList(u32) = .empty;
    defer triangles.deinit(allocator);

    var facet_start: usize = 0;
    var it = std.mem.tokenizeAny(u8, bytes, " \t\r\n");
    while (it.next()) |token| {
        if (std.mem.eql(u8, token, "vertex")) {
            for (0..3) |_| {
                const num = it.next() orelse return error.InvalidFormat;
                const value = std.fmt.parseFloat(f32, num) catch return error.InvalidCharacter;
                try vertices.append(allocator, value);
            }
        } else if (std.mem.eql(u8, token, "outer")) {
            facet_start = vertices.items.len / 3;
        } else if (std.mem.eql(u8, token, "endfacet")) {
            const base: u32 = @intCast(facet_start);
            switch (vertices.items.len / 3 - facet_start) {
                3 => try triangles.appendSlice(allocator, &.{ base, base + 1, base + 2 }),
                // Quad facets (as produced by some exporters): fan-triangulate.
                4 => try triangles.appendSlice(allocator, &.{ base, base + 1, base + 2, base, base + 2, base + 3 }),
                else => return error.UnsupportedPolygon,
            }
        }
    }

    return Mesh{
        .vertices = try vertices.toOwnedSlice(allocator),
        .triangles = try triangles.toOwnedSlice(allocator),
    };
}

fn appendBinaryTriangle(buffer: *std.ArrayList(u8), allocator: std.mem.Allocator, verts: [9]f32) !void {
    // normal (unused by the parser)
    for (0..3) |_| try buffer.appendSlice(allocator, &std.mem.toBytes(@as(f32, 0)));
    for (verts) |v| try buffer.appendSlice(allocator, &std.mem.toBytes(v));
    try buffer.appendSlice(allocator, &.{ 0, 0 }); // attribute byte count
}

test "parse binary STL" {
    const allocator = std.testing.allocator;

    var buffer: std.ArrayList(u8) = .empty;
    defer buffer.deinit(allocator);
    try buffer.appendNTimes(allocator, 0, 80); // header
    try buffer.appendSlice(allocator, &std.mem.toBytes(@as(u32, 2))); // triangle count
    try appendBinaryTriangle(&buffer, allocator, .{ 0, 0, 0, 1, 0, 0, 0, 1, 0 });
    try appendBinaryTriangle(&buffer, allocator, .{ 0, 0, 1, 1, 0, 1, 0, 1, 1 });

    var mesh = try parse(allocator, buffer.items);
    defer mesh.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), mesh.triangleCount());
    try std.testing.expectEqual(@as(usize, 6), mesh.vertexCount());
    try std.testing.expectEqual(@as(f32, 1.0), mesh.vertices[3]); // second vertex x
    try std.testing.expectEqual(@as(u32, 5), mesh.triangles[5]);

    const b = mesh.bounds();
    try std.testing.expectEqual([3]f32{ 0, 0, 0 }, b[0]);
    try std.testing.expectEqual([3]f32{ 1, 1, 1 }, b[1]);
}

test "parse ASCII STL with triangle and quad facets" {
    const allocator = std.testing.allocator;

    const text =
        \\solid test
        \\  facet normal 0 0 1
        \\    outer loop
        \\      vertex 0.0 0.0 0.0
        \\      vertex 1.0 0.0 0.0
        \\      vertex 0.0 1.0 0.0
        \\    endloop
        \\  endfacet
        \\  facet normal 0 0 1
        \\    outer loop
        \\      vertex 0.0 0.0 1.0
        \\      vertex 1.0 0.0 1.0
        \\      vertex 1.0 1.0 1.0
        \\      vertex 0.0 1.0 1.0
        \\    endloop
        \\  endfacet
        \\endsolid test
    ;

    var mesh = try parse(allocator, text);
    defer mesh.deinit(allocator);

    // 1 triangle facet + 1 quad facet (split into 2 triangles)
    try std.testing.expectEqual(@as(usize, 3), mesh.triangleCount());
    try std.testing.expectEqual(@as(usize, 7), mesh.vertexCount());
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 3, 5, 6 }, mesh.triangles);
}

test "rotate -90 degrees about X maps Z-up to Y-up" {
    const allocator = std.testing.allocator;

    var mesh = Mesh{
        .vertices = try allocator.dupe(f32, &.{ 1, 2, 3 }),
        .triangles = try allocator.dupe(u32, &.{ 0, 0, 0 }),
    };
    defer mesh.deinit(allocator);

    mesh.rotate(.x, -std.math.pi / 2.0);
    // old Z -> new Y, old Y -> new -Z
    try std.testing.expectApproxEqAbs(@as(f32, 1), mesh.vertices[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), mesh.vertices[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2), mesh.vertices[2], 1e-6);
}

test "rotations about each axis are right-handed" {
    const allocator = std.testing.allocator;

    var mesh = Mesh{
        .vertices = try allocator.dupe(f32, &.{ 1, 0, 0, 0, 1, 0, 0, 0, 1 }),
        .triangles = try allocator.dupe(u32, &.{ 0, 1, 2 }),
    };
    defer mesh.deinit(allocator);

    // +90 about Z: x -> y.
    mesh.rotate(.z, std.math.pi / 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), mesh.vertices[1], 1e-6);
    // +90 about Y: z -> x (vertex 2 still points +z).
    mesh.rotate(.y, std.math.pi / 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), mesh.vertices[6], 1e-6);
}

test "reject garbage input" {
    try std.testing.expectError(error.InvalidFormat, parse(std.testing.allocator, "not an stl"));
}
