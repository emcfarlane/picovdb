//! Mesh -> narrow-band signed distance field -> PicoVDB conversion.
//!
//! Replaces the OpenVDB `meshToLevelSet` + NanoVDB conversion pipeline with a
//! pure Zig implementation that writes PicoVDB structures directly.
//!
//! Pipeline (all in index space, distances in voxel units):
//!   1. Rasterize each triangle's half-width-dilated bounding box, keeping the
//!      minimum unsigned distance per voxel in a sparse map of 8^3 leaf blocks.
//!   2. Record ray crossings along +Z for every voxel column the mesh projects
//!      onto (with a consistent tie-break rule so shared edges/vertices count
//!      exactly once), giving inside/outside by parity. Assumes a watertight
//!      mesh; a flood-fill based sign pass is a possible future alternative for
//!      open meshes.
//!   3. Sign the narrow band, classify empty regions, and emit the picovdb
//!      tree (roots/uppers/lowers/leaves + value buffer).

const std = @import("std");
const picovdb = @import("picovdb.zig");

pub const ValueType = enum {
    f32,
    u8,

    pub fn elemSize(self: ValueType) usize {
        return switch (self) {
            .f32 => 4,
            .u8 => 1,
        };
    }

    pub fn gridType(self: ValueType) u32 {
        return switch (self) {
            .f32 => picovdb.GRID_TYPE_SDF_FLOAT,
            .u8 => picovdb.GRID_TYPE_SDF_UINT8,
        };
    }
};

pub const Options = struct {
    /// World units per voxel.
    voxel_size: f32,
    /// Narrow band half-width in voxels.
    half_width: f32 = picovdb.LEVEL_SET_HALF_WIDTH,
    value_type: ValueType = .f32,
};

pub const Stats = struct {
    active_voxels: u64,
    surface_voxels: u64,
    leaf_count: u32,
    lower_count: u32,
    upper_count: u32,
    index_bounds_min: [3]i32,
    index_bounds_max: [3]i32,
};

pub const Error = error{
    EmptyMesh,
    NoActiveVoxels,
    NonFiniteVertex,
} || std.mem.Allocator.Error;

const V3 = @Vector(3, f32);

inline fn dot(a: V3, b: V3) f32 {
    return @reduce(.Add, a * b);
}

inline fn dot2(a: V3) f32 {
    return dot(a, a);
}

/// Squared distance from point p to segment ab.
fn distSqPointSegment(p: V3, a: V3, b: V3) f32 {
    const ab = b - a;
    const denom = dot2(ab);
    if (denom <= 0) return dot2(p - a);
    const t = std.math.clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    return dot2(p - (a + ab * @as(V3, @splat(t))));
}

/// Squared distance from point p to triangle abc (Ericson, "Real-Time
/// Collision Detection" closest-point-on-triangle).
fn distSqPointTriangle(p: V3, a: V3, b: V3, c: V3) f32 {
    const ab = b - a;
    const ac = c - a;
    const ap = p - a;
    const d1 = dot(ab, ap);
    const d2 = dot(ac, ap);
    if (d1 <= 0 and d2 <= 0) return dot2(ap); // vertex a

    const bp = p - b;
    const d3 = dot(ab, bp);
    const d4 = dot(ac, bp);
    if (d3 >= 0 and d4 <= d3) return dot2(bp); // vertex b

    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 and d1 >= 0 and d3 <= 0) {
        const denom = d1 - d3;
        if (denom > 0) {
            const v = d1 / denom;
            return dot2(ap - ab * @as(V3, @splat(v))); // edge ab
        }
    }

    const cp = p - c;
    const d5 = dot(ab, cp);
    const d6 = dot(ac, cp);
    if (d6 >= 0 and d5 <= d6) return dot2(cp); // vertex c

    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 and d2 >= 0 and d6 <= 0) {
        const denom = d2 - d6;
        if (denom > 0) {
            const w = d2 / denom;
            return dot2(ap - ac * @as(V3, @splat(w))); // edge ac
        }
    }

    const va = d3 * d6 - d5 * d4;
    if (va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0) {
        const denom = (d4 - d3) + (d5 - d6);
        if (denom > 0) {
            const w = (d4 - d3) / denom;
            return dot2(bp - (c - b) * @as(V3, @splat(w))); // edge bc
        }
    }

    const denom = va + vb + vc;
    if (denom <= 0) {
        // Degenerate (collinear) triangle: fall back to edge distances.
        const e0 = distSqPointSegment(p, a, b);
        const e1 = distSqPointSegment(p, a, c);
        const e2 = distSqPointSegment(p, b, c);
        return @min(e0, @min(e1, e2));
    }
    const inv = 1.0 / denom;
    const v = vb * inv;
    const w = vc * inv;
    return dot2(ap - (ab * @as(V3, @splat(v)) + ac * @as(V3, @splat(w)))); // face
}

// Sparse narrow-band scratch storage: one block per 8^3 leaf.
const Leaf = struct {
    // Squared unsigned distance during rasterization; signed distance (voxel
    // units) for every voxel after the sign pass (non-band voxels hold +-hw).
    values: [512]f32,
    // Bit n set if voxel n is within the narrow band (active). Filled by the
    // sign pass.
    band: [16]u32,
    band_count: u32,

    fn create(allocator: std.mem.Allocator) !*Leaf {
        const leaf = try allocator.create(Leaf);
        @memset(&leaf.values, std.math.inf(f32));
        @memset(&leaf.band, 0);
        leaf.band_count = 0;
        return leaf;
    }
};

const LeafMap = std.AutoHashMapUnmanaged([3]i32, *Leaf);
const OriginSet = std.AutoHashMapUnmanaged([3]i32, void);

inline fn leafOrigin(ijk: [3]i32) [3]i32 {
    return .{ ijk[0] & ~@as(i32, 7), ijk[1] & ~@as(i32, 7), ijk[2] & ~@as(i32, 7) };
}

/// Rasterize one triangle: for every voxel center within `hw` of the triangle,
/// min-update the squared distance in the leaf map.
fn rasterizeTriangle(arena: std.mem.Allocator, map: *LeafMap, a: V3, b: V3, c: V3, hw: f32) !void {
    const hw2 = hw * hw;
    const lo_f: [3]f32 = @min(a, @min(b, c)) - @as(V3, @splat(hw));
    const hi_f: [3]f32 = @max(a, @max(b, c)) + @as(V3, @splat(hw));
    var lo: [3]i32 = undefined;
    var hi: [3]i32 = undefined;
    for (0..3) |axis| {
        lo[axis] = @intFromFloat(@ceil(lo_f[axis]));
        hi[axis] = @intFromFloat(@floor(hi_f[axis]));
    }

    // Iterate leaf-aligned blocks so the hash lookup happens once per leaf,
    // not once per voxel.
    var lx = lo[0] & ~@as(i32, 7);
    while (lx <= hi[0]) : (lx += 8) {
        var ly = lo[1] & ~@as(i32, 7);
        while (ly <= hi[1]) : (ly += 8) {
            var lz = lo[2] & ~@as(i32, 7);
            while (lz <= hi[2]) : (lz += 8) {
                const gop = try map.getOrPut(arena, .{ lx, ly, lz });
                if (!gop.found_existing) {
                    gop.value_ptr.* = try Leaf.create(arena);
                }
                const leaf = gop.value_ptr.*;

                var x = @max(lo[0], lx);
                const x_end = @min(hi[0], lx + 7);
                while (x <= x_end) : (x += 1) {
                    var y = @max(lo[1], ly);
                    const y_end = @min(hi[1], ly + 7);
                    while (y <= y_end) : (y += 1) {
                        var z = @max(lo[2], lz);
                        const z_end = @min(hi[2], lz + 7);
                        while (z <= z_end) : (z += 1) {
                            const p = V3{
                                @floatFromInt(x),
                                @floatFromInt(y),
                                @floatFromInt(z),
                            };
                            const d2 = distSqPointTriangle(p, a, b, c);
                            if (d2 <= hw2) {
                                const n = picovdb.leafCoordToOffset(.{ x, y, z });
                                if (d2 < leaf.values[n]) leaf.values[n] = d2;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Per-column +Z ray crossings for parity-based inside/outside signing.
const ColumnGrid = struct {
    min_x: i32,
    min_y: i32,
    nx: usize,
    ny: usize,
    lists: []std.ArrayList(f64),

    fn init(arena: std.mem.Allocator, min_x: i32, min_y: i32, max_x: i32, max_y: i32) !ColumnGrid {
        const nx: usize = @intCast(max_x - min_x + 1);
        const ny: usize = @intCast(max_y - min_y + 1);
        const lists = try arena.alloc(std.ArrayList(f64), nx * ny);
        @memset(lists, .empty);
        return .{ .min_x = min_x, .min_y = min_y, .nx = nx, .ny = ny, .lists = lists };
    }

    fn columnIndex(self: *const ColumnGrid, x: i32, y: i32) ?usize {
        if (x < self.min_x or y < self.min_y) return null;
        const ix: usize = @intCast(x - self.min_x);
        const iy: usize = @intCast(y - self.min_y);
        if (ix >= self.nx or iy >= self.ny) return null;
        return ix * self.ny + iy;
    }

    /// Record the Z crossings of one triangle for every lattice column its XY
    /// projection covers. Boundary hits use a consistent tie-break rule
    /// (equivalent to shifting the sample point by an infinitesimal (+dx,+dx^2))
    /// so an edge shared by two triangles is counted exactly once and XY-
    /// degenerate (vertical) triangles contribute nothing.
    fn binTriangle(self: *ColumnGrid, arena: std.mem.Allocator, a: V3, b: V3, c: V3) !void {
        const ax: f64 = a[0];
        const ay: f64 = a[1];
        const bx: f64 = b[0];
        const by: f64 = b[1];
        const cx: f64 = c[0];
        const cy: f64 = c[1];

        // Twice the signed XY area; constant over sample points.
        const signed_area = edgeFn(ax, ay, bx, by, cx, cy);
        if (signed_area == 0) return; // XY-degenerate (vertical) triangle
        const flip: f64 = if (signed_area < 0) -1 else 1;
        const area = flip * signed_area;

        const x0: i32 = @intFromFloat(@ceil(@min(ax, @min(bx, cx))));
        const x1: i32 = @intFromFloat(@floor(@max(ax, @max(bx, cx))));
        const y0: i32 = @intFromFloat(@ceil(@min(ay, @min(by, cy))));
        const y1: i32 = @intFromFloat(@floor(@max(ay, @max(by, cy))));

        var x = x0;
        while (x <= x1) : (x += 1) {
            var y = y0;
            while (y <= y1) : (y += 1) {
                const px: f64 = @floatFromInt(x);
                const py: f64 = @floatFromInt(y);

                const w0 = flip * edgeFn(bx, by, cx, cy, px, py);
                const w1 = flip * edgeFn(cx, cy, ax, ay, px, py);
                const w2 = flip * edgeFn(ax, ay, bx, by, px, py);

                // Edge vectors (in normalized orientation) for the tie-break.
                const inside =
                    accept(w0, flip * (cx - bx), flip * (cy - by)) and
                    accept(w1, flip * (ax - cx), flip * (ay - cy)) and
                    accept(w2, flip * (bx - ax), flip * (by - ay));
                if (inside) {
                    const z = (w0 * @as(f64, a[2]) + w1 * @as(f64, b[2]) + w2 * @as(f64, c[2])) / area;
                    const index = self.columnIndex(x, y) orelse continue;
                    try self.lists[index].append(arena, z);
                }
            }
        }
    }

    inline fn edgeFn(px: f64, py: f64, qx: f64, qy: f64, sx: f64, sy: f64) f64 {
        return (qx - px) * (sy - py) - (qy - py) * (sx - px);
    }

    inline fn accept(w: f64, ex: f64, ey: f64) bool {
        if (w > 0) return true;
        if (w < 0) return false;
        return ey < 0 or (ey == 0 and ex > 0);
    }

    fn sortAll(self: *ColumnGrid) void {
        for (self.lists) |*list| {
            std.mem.sort(f64, list.items, {}, std.sort.asc(f64));
        }
    }

    /// Parity test: odd number of surface crossings below the voxel center.
    fn isInside(self: *const ColumnGrid, ijk: [3]i32) bool {
        const index = self.columnIndex(ijk[0], ijk[1]) orelse return false;
        const z: f64 = @floatFromInt(ijk[2]);
        var count: usize = 0;
        for (self.lists[index].items) |crossing| {
            if (crossing < z) count += 1 else break;
        }
        return count % 2 == 1;
    }
};

// Emits the picovdb tree for one grid from the signed leaf map, mirroring the
// encoding produced by the NanoVDB converter in main.zig.
const Builder = struct {
    allocator: std.mem.Allocator,
    out: *picovdb.PicoVDBFileMutable,
    grid: *picovdb.PicoVDBGrid,
    map: *const LeafMap,
    columns: *const ColumnGrid,
    lower_set: *const OriginSet,
    hw: f32,
    value_type: ValueType,
    surface_count: u64 = 0,

    fn dataElems(self: *const Builder) u32 {
        const grid_bytes = self.out.data_buffer.items.len - @as(usize, self.grid.data_start) * 16;
        return @intCast(grid_bytes / self.value_type.elemSize());
    }

    fn appendValue(self: *Builder, value: f32) !void {
        switch (self.value_type) {
            .f32 => try self.out.data_buffer.appendSlice(self.allocator, std.mem.asBytes(&value)),
            .u8 => {
                const quantized: u8 = @intFromFloat(std.math.clamp(@round((value / picovdb.LEVEL_SET_HALF_WIDTH + 1.0) * 127.5), 0.0, 255.0));
                try self.out.data_buffer.append(self.allocator, quantized);
            },
        }
    }

    /// Signed value at any coordinate: stored narrow-band/leaf value, or the
    /// implicit +-hw classified by column parity.
    fn valueAt(self: *const Builder, ijk: [3]i32) f32 {
        if (self.map.get(leafOrigin(ijk))) |leaf| {
            return leaf.values[picovdb.leafCoordToOffset(ijk)];
        }
        return if (self.columns.isInside(ijk)) -self.hw else self.hw;
    }

    fn emitUpper(self: *Builder, origin: [3]i32) !void {
        var elements: [1024]picovdb.PicoVDBNodeElement = undefined;
        const base_lower: u32 = @intCast(self.out.lowers.items.len - self.grid.lower_start);
        const base_value: u32 = self.dataElems();

        var local_state_count: u32 = 0;
        var local_value_count: u32 = 0;
        for (0..1024) |word| {
            var state_word: u32 = 0;
            var value_word: u32 = 0;
            for (0..32) |bit_index| {
                const bit: u5 = @intCast(bit_index);
                const n: u32 = @intCast(word * 32 + bit_index);
                const child_origin = [3]i32{
                    origin[0] + @as(i32, @intCast((n >> 10) & 31)) * 128,
                    origin[1] + @as(i32, @intCast((n >> 5) & 31)) * 128,
                    origin[2] + @as(i32, @intCast(n & 31)) * 128,
                };
                if (self.lower_set.contains(child_origin)) {
                    state_word |= @as(u32, 1) << bit;
                    value_word |= @as(u32, 1) << bit;
                } else {
                    // Empty 128^3 region: inside-implicit if its center is
                    // inside the mesh (uniform sign for watertight meshes).
                    const center = [3]i32{ child_origin[0] + 64, child_origin[1] + 64, child_origin[2] + 64 };
                    if (self.columns.isInside(center)) {
                        state_word |= @as(u32, 1) << bit;
                    }
                }
            }
            elements[word] = .{
                .state_mask = state_word,
                .value_mask = value_word,
                .packed_local_index = (local_state_count << 16) | local_value_count,
            };
            local_state_count += @popCount(value_word & state_word);
            local_value_count += @popCount(value_word & ~state_word);
        }

        // Children in slot order so child index = base + preceding count.
        for (0..32768) |n| {
            const child_origin = [3]i32{
                origin[0] + @as(i32, @intCast((n >> 10) & 31)) * 128,
                origin[1] + @as(i32, @intCast((n >> 5) & 31)) * 128,
                origin[2] + @as(i32, @intCast(n & 31)) * 128,
            };
            if (self.lower_set.contains(child_origin)) {
                try self.emitLower(child_origin);
            }
        }

        try self.out.uppers.append(self.allocator, .{
            .base_inside_index = base_lower,
            .base_active_index = base_value,
            .elements = elements,
        });
    }

    fn emitLower(self: *Builder, origin: [3]i32) !void {
        var elements: [128]picovdb.PicoVDBNodeElement = undefined;
        const base_leaf: u32 = @intCast(self.out.leaves.items.len - self.grid.leaf_start);
        const base_value: u32 = self.dataElems();

        var local_state_count: u32 = 0;
        var local_value_count: u32 = 0;
        for (0..128) |word| {
            var state_word: u32 = 0;
            var value_word: u32 = 0;
            for (0..32) |bit_index| {
                const bit: u5 = @intCast(bit_index);
                const n: u32 = @intCast(word * 32 + bit_index);
                const child_origin = [3]i32{
                    origin[0] + @as(i32, @intCast((n >> 8) & 15)) * 8,
                    origin[1] + @as(i32, @intCast((n >> 4) & 15)) * 8,
                    origin[2] + @as(i32, @intCast(n & 15)) * 8,
                };
                if (self.isBandLeaf(child_origin)) {
                    state_word |= @as(u32, 1) << bit;
                    value_word |= @as(u32, 1) << bit;
                } else {
                    const center = [3]i32{ child_origin[0] + 4, child_origin[1] + 4, child_origin[2] + 4 };
                    if (self.columns.isInside(center)) {
                        state_word |= @as(u32, 1) << bit;
                    }
                }
            }
            elements[word] = .{
                .state_mask = state_word,
                .value_mask = value_word,
                .packed_local_index = (local_state_count << 16) | local_value_count,
            };
            local_state_count += @popCount(value_word & state_word);
            local_value_count += @popCount(value_word & ~state_word);
        }

        for (0..4096) |n| {
            const child_origin = [3]i32{
                origin[0] + @as(i32, @intCast((n >> 8) & 15)) * 8,
                origin[1] + @as(i32, @intCast((n >> 4) & 15)) * 8,
                origin[2] + @as(i32, @intCast(n & 15)) * 8,
            };
            if (self.map.get(child_origin)) |leaf| {
                if (leaf.band_count > 0) {
                    try self.emitLeaf(child_origin, leaf);
                }
            }
        }

        try self.out.lowers.append(self.allocator, .{
            .base_inside_index = base_leaf,
            .base_active_index = base_value,
            .elements = elements,
        });
    }

    fn isBandLeaf(self: *const Builder, origin: [3]i32) bool {
        const leaf = self.map.get(origin) orelse return false;
        return leaf.band_count > 0;
    }

    fn emitLeaf(self: *Builder, origin: [3]i32, leaf: *const Leaf) !void {
        var elements: [16]picovdb.PicoVDBNodeElement = undefined;
        const base_value: u32 = self.dataElems();

        const neighbor_offsets = [7][3]i32{
            .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 },
            .{ 1, 1, 0 }, .{ 1, 0, 1 }, .{ 0, 1, 1 },
            .{ 1, 1, 1 },
        };

        var local_state_count: u32 = 0;
        var local_value_count: u32 = 0;
        var leaf_surface_count: u64 = 0;
        for (0..16) |word| {
            const value_word = leaf.band[word];
            var state_word: u32 = 0;
            for (0..32) |bit_index| {
                const bit: u5 = @intCast(bit_index);
                const n: u32 = @intCast(word * 32 + bit_index);
                const value = leaf.values[n];

                if ((value_word >> bit) & 1 == 0) {
                    // Inactive voxel: inside-implicit if negative.
                    if (value < 0) state_word |= @as(u32, 1) << bit;
                    continue;
                }

                // Active voxel: mark as surface if the SDF sign changes toward
                // any +1 neighbor (same rule as the NanoVDB converter).
                const lx: i32 = @intCast(n >> 6);
                const ly: i32 = @intCast((n >> 3) & 7);
                const lz: i32 = @intCast(n & 7);
                for (neighbor_offsets) |off| {
                    const nx = lx + off[0];
                    const ny = ly + off[1];
                    const nz = lz + off[2];
                    var neighbor: f32 = undefined;
                    if (nx < 8 and ny < 8 and nz < 8) {
                        neighbor = leaf.values[@intCast(nx * 64 + ny * 8 + nz)];
                    } else {
                        neighbor = self.valueAt(.{ origin[0] + nx, origin[1] + ny, origin[2] + nz });
                    }
                    const sign_strict = (value < 0) != (neighbor < 0);
                    const sign_nonstrict = (value <= 0) != (neighbor <= 0);
                    if (sign_strict or sign_nonstrict) {
                        state_word |= @as(u32, 1) << bit;
                        break;
                    }
                }
            }

            elements[word] = .{
                .state_mask = state_word,
                .value_mask = value_word,
                .packed_local_index = (local_state_count << 16) | local_value_count,
            };
            local_value_count += @popCount(value_word);
            local_state_count += @popCount(value_word & state_word);
            leaf_surface_count += @popCount(value_word & state_word);

            // Append SDF values for all active voxels in bit order.
            for (0..32) |append_bit| {
                const abit: u5 = @intCast(append_bit);
                if ((value_word >> abit) & 1 != 0) {
                    try self.appendValue(leaf.values[word * 32 + append_bit]);
                }
            }
        }

        try self.out.leaves.append(self.allocator, .{
            .base_inside_index = self.surface_count,
            .base_active_index = base_value,
            .elements = elements,
        });
        self.surface_count += leaf_surface_count;
    }
};

fn lessThanOrigin(_: void, a: [3]i32, b: [3]i32) bool {
    if (a[0] != b[0]) return a[0] < b[0];
    if (a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

/// Convert a triangle mesh (world units) into a narrow-band signed distance
/// grid appended to `out`. Vertices are xyz triples, triangles are vertex
/// index triples.
pub fn meshToPicoVDB(
    allocator: std.mem.Allocator,
    out: *picovdb.PicoVDBFileMutable,
    vertices: []const f32,
    triangles: []const u32,
    opts: Options,
) Error!Stats {
    if (triangles.len < 3 or vertices.len < 9) return error.EmptyMesh;
    std.debug.assert(opts.voxel_size > 0);
    std.debug.assert(opts.half_width > 0);
    const hw = opts.half_width;

    // All intermediate storage lives in the arena; only `out` uses `allocator`.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Transform vertices to index space (voxel units).
    const pts = try arena.alloc(f32, vertices.len);
    for (vertices, 0..) |v, i| {
        if (!std.math.isFinite(v)) return error.NonFiniteVertex;
        pts[i] = v / opts.voxel_size;
    }

    var mesh_min = V3{ pts[0], pts[1], pts[2] };
    var mesh_max = mesh_min;
    var vi: usize = 3;
    while (vi < pts.len) : (vi += 3) {
        const p = V3{ pts[vi], pts[vi + 1], pts[vi + 2] };
        mesh_min = @min(mesh_min, p);
        mesh_max = @max(mesh_max, p);
    }

    var columns = try ColumnGrid.init(
        arena,
        @as(i32, @intFromFloat(@floor(mesh_min[0]))) - 1,
        @as(i32, @intFromFloat(@floor(mesh_min[1]))) - 1,
        @as(i32, @intFromFloat(@ceil(mesh_max[0]))) + 1,
        @as(i32, @intFromFloat(@ceil(mesh_max[1]))) + 1,
    );

    // Pass 1: unsigned squared distances into the sparse leaf map + column
    // crossings for parity signing.
    var map: LeafMap = .empty;
    var ti: usize = 0;
    while (ti < triangles.len) : (ti += 3) {
        const a = V3{ pts[triangles[ti] * 3], pts[triangles[ti] * 3 + 1], pts[triangles[ti] * 3 + 2] };
        const b = V3{ pts[triangles[ti + 1] * 3], pts[triangles[ti + 1] * 3 + 1], pts[triangles[ti + 1] * 3 + 2] };
        const c = V3{ pts[triangles[ti + 2] * 3], pts[triangles[ti + 2] * 3 + 1], pts[triangles[ti + 2] * 3 + 2] };
        try rasterizeTriangle(arena, &map, a, b, c, hw);
        try columns.binTriangle(arena, a, b, c);
    }
    columns.sortAll();

    // Pass 2: sign the band (parity), fill non-band voxels with +-hw, and
    // collect the active bounds and node origin sets.
    const hw2 = hw * hw;
    var active_min = [3]i32{ std.math.maxInt(i32), std.math.maxInt(i32), std.math.maxInt(i32) };
    var active_max = [3]i32{ std.math.minInt(i32), std.math.minInt(i32), std.math.minInt(i32) };
    var active_voxels: u64 = 0;
    var lower_set: OriginSet = .empty;
    var upper_set: OriginSet = .empty;

    var it = map.iterator();
    while (it.next()) |entry| {
        const origin = entry.key_ptr.*;
        const leaf = entry.value_ptr.*;
        for (0..512) |n| {
            const ijk = [3]i32{
                origin[0] + @as(i32, @intCast(n >> 6)),
                origin[1] + @as(i32, @intCast((n >> 3) & 7)),
                origin[2] + @as(i32, @intCast(n & 7)),
            };
            const inside = columns.isInside(ijk);
            const d2 = leaf.values[n];
            if (d2 <= hw2) {
                const d = @sqrt(d2);
                leaf.values[n] = if (inside) -d else d;
                leaf.band[n / 32] |= @as(u32, 1) << @intCast(n % 32);
                leaf.band_count += 1;
                for (0..3) |axis| {
                    active_min[axis] = @min(active_min[axis], ijk[axis]);
                    active_max[axis] = @max(active_max[axis], ijk[axis]);
                }
            } else {
                leaf.values[n] = if (inside) -hw else hw;
            }
        }
        if (leaf.band_count > 0) {
            active_voxels += leaf.band_count;
            try lower_set.put(arena, .{
                origin[0] & ~@as(i32, 127),
                origin[1] & ~@as(i32, 127),
                origin[2] & ~@as(i32, 127),
            }, {});
            try upper_set.put(arena, .{
                origin[0] & ~@as(i32, 4095),
                origin[1] & ~@as(i32, 4095),
                origin[2] & ~@as(i32, 4095),
            }, {});
        }
    }
    if (active_voxels == 0) return error.NoActiveVoxels;

    const upper_origins = try arena.alloc([3]i32, upper_set.count());
    {
        var i: usize = 0;
        var uit = upper_set.keyIterator();
        while (uit.next()) |key| : (i += 1) upper_origins[i] = key.*;
        std.mem.sort([3]i32, upper_origins, {}, lessThanOrigin);
    }

    // Pass 3: emit the grid.
    const data_start_bytes = out.data_buffer.items.len;
    std.debug.assert(data_start_bytes % 16 == 0);

    var grid = picovdb.PicoVDBGrid{
        .grid_index = @intCast(out.grids.items.len),
        .upper_start = @intCast(out.uppers.items.len),
        .lower_start = @intCast(out.lowers.items.len),
        .leaf_start = @intCast(out.leaves.items.len),
        .data_start = @intCast(data_start_bytes / 16),
        .data_elem_count = 0,
        .grid_type = opts.value_type.gridType(),
        ._pad1 = 0,
        .index_bounds_min = active_min,
        ._pad2 = 0,
        .index_bounds_max = active_max,
        ._pad3 = 0,
    };

    var builder = Builder{
        .allocator = allocator,
        .out = out,
        .grid = &grid,
        .map = &map,
        .columns = &columns,
        .lower_set = &lower_set,
        .hw = hw,
        .value_type = opts.value_type,
    };

    // Data indices 0 and 1 hold the implicit background/inside values.
    try builder.appendValue(hw);
    try builder.appendValue(-hw);

    for (upper_origins) |origin| {
        try out.roots.append(allocator, .{ .key = picovdb.coordToKey(origin) });
        try builder.emitUpper(origin);
    }

    grid.data_elem_count = builder.dataElems();

    // Pad the data buffer to 16-byte alignment for any following grid.
    const data_end_bytes = out.data_buffer.items.len;
    const data_padding = std.mem.alignForward(usize, data_end_bytes, 16) - data_end_bytes;
    if (data_padding > 0) {
        const padding = [_]u8{0} ** 16;
        try out.data_buffer.appendSlice(allocator, padding[0..data_padding]);
    }

    const stats = Stats{
        .active_voxels = active_voxels,
        .surface_voxels = builder.surface_count,
        .leaf_count = @intCast(out.leaves.items.len - grid.leaf_start),
        .lower_count = @intCast(out.lowers.items.len - grid.lower_start),
        .upper_count = @intCast(out.uppers.items.len - grid.upper_start),
        .index_bounds_min = active_min,
        .index_bounds_max = active_max,
    };

    try out.grids.append(allocator, grid);
    return stats;
}

const TestMesh = struct {
    vertices: []f32,
    triangles: []u32,
};

/// UV sphere triangle mesh (world units), consistent outward winding.
fn makeUvSphere(allocator: std.mem.Allocator, center: [3]f32, radius: f32, stacks: u32, slices: u32) !TestMesh {
    var vertices: std.ArrayList(f32) = .empty;
    defer vertices.deinit(allocator);
    var triangles: std.ArrayList(u32) = .empty;
    defer triangles.deinit(allocator);

    // Vertex layout: [top pole, ring 1 .. ring stacks-1 (slices each), bottom pole]
    try vertices.appendSlice(allocator, &.{ center[0], center[1] + radius, center[2] });
    for (1..stacks) |i| {
        const theta = std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(stacks));
        const y = center[1] + radius * @cos(theta);
        const ring_radius = radius * @sin(theta);
        for (0..slices) |j| {
            const phi = 2.0 * std.math.pi * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(slices));
            try vertices.appendSlice(allocator, &.{
                center[0] + ring_radius * @cos(phi),
                y,
                center[2] + ring_radius * @sin(phi),
            });
        }
    }
    const bottom: u32 = @intCast(vertices.items.len / 3);
    try vertices.appendSlice(allocator, &.{ center[0], center[1] - radius, center[2] });

    const ring = struct {
        fn index(i: usize, j: usize, slices_: u32) u32 {
            return @intCast(1 + (i - 1) * slices_ + (j % slices_));
        }
    }.index;

    for (0..slices) |j| {
        // Top and bottom fans.
        try triangles.appendSlice(allocator, &.{ 0, ring(1, j + 1, slices), ring(1, j, slices) });
        try triangles.appendSlice(allocator, &.{ bottom, ring(stacks - 1, j, slices), ring(stacks - 1, j + 1, slices) });
    }
    for (1..stacks - 1) |i| {
        for (0..slices) |j| {
            const a = ring(i, j, slices);
            const b = ring(i, j + 1, slices);
            const c = ring(i + 1, j + 1, slices);
            const d = ring(i + 1, j, slices);
            try triangles.appendSlice(allocator, &.{ a, b, c });
            try triangles.appendSlice(allocator, &.{ a, c, d });
        }
    }

    return .{
        .vertices = try vertices.toOwnedSlice(allocator),
        .triangles = try triangles.toOwnedSlice(allocator),
    };
}

test "sphere mesh to picovdb matches analytic SDF" {
    const allocator = std.testing.allocator;

    // World-space sphere; at voxel_size 0.05 the index-space radius is 20.3
    // voxels. Center is off-lattice to avoid degenerate alignments.
    const voxel_size: f32 = 0.05;
    const center = [3]f32{ 0.015, -0.02, 0.01 };
    const radius: f32 = 1.015;
    const mesh = try makeUvSphere(allocator, center, radius, 64, 128);
    defer allocator.free(mesh.vertices);
    defer allocator.free(mesh.triangles);

    var file_mutable = picovdb.PicoVDBFileMutable.init();
    defer file_mutable.deinit(allocator);

    const stats = try meshToPicoVDB(allocator, &file_mutable, mesh.vertices, mesh.triangles, .{
        .voxel_size = voxel_size,
    });

    // Surface area heuristic: ~4*pi*r^2 voxels per band layer, 2*hw+1 layers.
    const index_radius = radius / voxel_size;
    const shell: f64 = 4.0 * std.math.pi * @as(f64, index_radius) * @as(f64, index_radius);
    try std.testing.expect(@as(f64, @floatFromInt(stats.active_voxels)) > shell * 5);
    try std.testing.expect(@as(f64, @floatFromInt(stats.active_voxels)) < shell * 9);
    try std.testing.expect(stats.surface_voxels > 0);
    try std.testing.expect(stats.upper_count >= 1);

    const buffer = try file_mutable.encode(allocator);
    defer allocator.free(buffer);
    const file = try picovdb.PicoVDBFile.fromBytes(buffer);
    try std.testing.expectEqual(@as(u32, 1), file.header.grid_count);
    const grid = &file.grids[0];

    // Compare against the analytic sphere SDF over a 3D sample lattice.
    const hw = picovdb.LEVEL_SET_HALF_WIDTH;
    const index_center = V3{
        center[0] / voxel_size,
        center[1] / voxel_size,
        center[2] / voxel_size,
    };
    var accessor = picovdb.PicoVDBReadAccessor.init(0);
    var tested: u32 = 0;
    var x: i32 = -32;
    while (x <= 32) : (x += 1) {
        var y: i32 = -32;
        while (y <= 32) : (y += 1) {
            var z: i32 = -32;
            while (z <= 32) : (z += 1) {
                const p = V3{ @floatFromInt(x), @floatFromInt(y), @floatFromInt(z) };
                const analytic = @sqrt(dot2(p - index_center)) - index_radius;
                const result = accessor.getLevelIndex(.{ x, y, z }, grid, &file);
                const value = file.getGridFloat(grid, result.index);

                // Margins skip voxels near the band boundary, where activation
                // differs by rasterization epsilon, and near the surface facets.
                if (analytic > hw + 0.6) {
                    try std.testing.expectEqual(@as(u32, 0), result.index); // background
                } else if (analytic < -hw - 0.6) {
                    try std.testing.expectEqual(@as(u32, 1), result.index); // inside implicit
                } else if (@abs(analytic) < hw - 0.6) {
                    // Active band voxel: value tracks the analytic distance
                    // (tolerance covers facet chord error + f32 rounding).
                    try std.testing.expect(result.index >= 2);
                    try std.testing.expectApproxEqAbs(analytic, value, 0.15);
                }
                tested += 1;
            }
        }
    }
    try std.testing.expect(tested > 0);

    // Far away from the mesh there is no root tile: background.
    var far_accessor = picovdb.PicoVDBReadAccessor.init(0);
    const far = far_accessor.getLevelIndex(.{ 5000, 5000, 5000 }, grid, &file);
    try std.testing.expectEqual(@as(u32, 0), far.index);
    try std.testing.expectEqual(@as(u32, 4), far.level);
}

test "degenerate triangle distance falls back to edges" {
    const p = V3{ 0, 1, 0 };
    const a = V3{ -1, 0, 0 };
    const b = V3{ 1, 0, 0 };
    const c = V3{ 3, 0, 0 }; // collinear with a-b
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distSqPointTriangle(p, a, b, c), 1e-6);
}

test "empty mesh is rejected" {
    const allocator = std.testing.allocator;
    var file_mutable = picovdb.PicoVDBFileMutable.init();
    defer file_mutable.deinit(allocator);
    try std.testing.expectError(error.EmptyMesh, meshToPicoVDB(allocator, &file_mutable, &.{}, &.{}, .{ .voxel_size = 1 }));
}
