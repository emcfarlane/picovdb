//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Include tests from picovdb.zig
test {
    std.testing.refAllDecls(@import("picovdb.zig"));
}

// Re-export PicoVDB structures from picovdb.zig
pub const PicoVDBGrid = @import("picovdb.zig").PicoVDBGrid;
pub const PicoVDBRoot = @import("picovdb.zig").PicoVDBRoot;
pub const PicoVDBUpper = @import("picovdb.zig").PicoVDBUpper;
pub const PicoVDBLower = @import("picovdb.zig").PicoVDBLower;
pub const PicoVDBLeaf = @import("picovdb.zig").PicoVDBLeaf;
pub const PicoVDBReadAccessor = @import("picovdb.zig").PicoVDBReadAccessor;
pub const PicoVDBNodeMask = @import("picovdb.zig").PicoVDBNodeMask;
pub const PicoVDBLeafMask = @import("picovdb.zig").PicoVDBLeafMask;
pub const PicoVDBFile = @import("picovdb.zig").PicoVDBFile;
pub const PicoVDBFileMutable = @import("picovdb.zig").PicoVDBFileMutable;
pub const coordToKey = @import("picovdb.zig").coordToKey;
pub const getGridFloat = @import("picovdb.zig").getGridFloat;

// Grid type constants
pub const GRID_TYPE_SDF_FLOAT = @import("picovdb.zig").GRID_TYPE_SDF_FLOAT;
pub const GRID_TYPE_SDF_UINT8 = @import("picovdb.zig").GRID_TYPE_SDF_UINT8;
