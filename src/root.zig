//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// Include tests from picovdb.zig
test {
    std.testing.refAllDecls(@import("picovdb.zig"));
}

const picovdb = @import("picovdb.zig");

// Re-export PicoVDB structures from picovdb.zig
pub const PicoVDBGrid = picovdb.PicoVDBGrid;
pub const PicoVDBRoot = picovdb.PicoVDBRoot;
pub const PicoVDBUpper = picovdb.PicoVDBUpper;
pub const PicoVDBLower = picovdb.PicoVDBLower;
pub const PicoVDBLeaf = picovdb.PicoVDBLeaf;
pub const PicoVDBReadAccessor = picovdb.PicoVDBReadAccessor;
pub const PicoVDBNodeMask = picovdb.PicoVDBNodeMask;
pub const PicoVDBLeafMask = picovdb.PicoVDBLeafMask;
pub const PicoVDBFile = picovdb.PicoVDBFile;
pub const PicoVDBFileMutable = picovdb.PicoVDBFileMutable;
pub const coordToKey = picovdb.coordToKey;
pub const getGridFloat = picovdb.getGridFloat;

// Grid type constants
pub const GRID_TYPE_SDF_FLOAT = picovdb.GRID_TYPE_SDF_FLOAT;
pub const GRID_TYPE_SDF_UINT8 = picovdb.GRID_TYPE_SDF_UINT8;
