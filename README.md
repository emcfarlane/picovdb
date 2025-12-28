# PicoVDB

Compact sparse volumetric data format optimized for WebGPU real-time rendering.

**[Live Demo →](https://emcfarlane.github.io/picovdb/demo/)**

> [!WARNING]
> This project is under active development. The data format and API are subject to change.

## Overview

- **50%+ smaller volumes** than NanoVDB ([bunny](http://graphics.stanford.edu/data/3Dscanrep/): 64MB → 28MB)
- **WebGPU-native** data layout with WGSL shader library
- **32-bit addressing** for better GPU compatibility
- **Fast traversal** with hierarchical raymarching (HDDA)

This repository includes:
- `picovdb.wgsl` - WGSL shader library
- `picovdb.ts` - TypeScript loader
- `src/main.zig` - NanoVDB → PicoVDB converter

## How It Works

PicoVDB compresses NanoVDB files through:
- **Rank query compression**: Bit masks + counts eliminate inactive voxel storage
- **32-bit offsets**: Replace 64-bit pointers with computed indices (limits to 4 billion active voxels)
- **GPU-aligned structs**: Minimize padding, maximize cache efficiency

## Usage

```wgsl
// Include the library in your compute shader
// (concatenate picovdb.wgsl with your shader code)

@group(0) @binding(2) var<storage> picovdb_grids: array<PicoVDBGrid>;
@group(0) @binding(3) var<storage> picovdb_roots: array<PicoVDBRoot>;
@group(0) @binding(4) var<storage> picovdb_uppers: array<PicoVDBUpper>;
@group(0) @binding(5) var<storage> picovdb_lowers: array<PicoVDBLower>;
@group(0) @binding(6) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
@group(0) @binding(7) var<storage> picovdb_buffer: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let grid = picovdb_grids[0];

    // Initialize read accessor
    var accessor: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&accessor);

    // Sample voxel data using HDDA traversal
    var hit_t: f32;
    var hit_value: f32;
    let hit = picovdbHDDAZeroCrossing(
        &accessor, grid, ray_origin, t_near, ray_direction, t_far, &hit_t, &hit_value
    );
}
```

## Converting Files
```bash
# Build converter
zig build

# Convert NanoVDB to PicoVDB
./zig-out/bin/picovdb input.nvdb output.picovdb
```

## Related Projects

- **[OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb)** - Industry standard sparse volume library
- **[NanoVDB](https://developer.nvidia.com/nanovdb)** - GPU-optimized sparse volumes
- **[WebGPU NanoVDB](https://github.com/emcfarlane/webgpu-nanovdb)** - WebGPU port of NanoVDB
