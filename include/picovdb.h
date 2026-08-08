/* C ABI for the PicoVDB library (src/c_api.zig).
 * Struct layouts must stay in sync with src/c_api.zig and stl.ts. */
#ifndef PICOVDB_H
#define PICOVDB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PICOVDB_ABI_VERSION 1

/* Error codes: 0 is success; map to strings with picovdb_error_string(). */
#define PICOVDB_ERROR_PARSE (-1)
#define PICOVDB_ERROR_OOM (-2)
#define PICOVDB_ERROR_EMPTY_MESH (-3)
#define PICOVDB_ERROR_NON_FINITE (-4)
#define PICOVDB_ERROR_BAD_OPTIONS (-5)
#define PICOVDB_ERROR_TOO_MANY_VOXELS (-6)

typedef enum { PICOVDB_VALUE_F32 = 0, PICOVDB_VALUE_U8 = 1 } picovdb_value_type;

/* Cheap pre-voxelization pass: triangle count + world bounds. */
typedef struct {
    uint32_t triangle_count;
    float bbox_min[3];
    float bbox_max[3];
} picovdb_mesh_info;

typedef struct {
    /* Fail with PICOVDB_ERROR_TOO_MANY_VOXELS if the voxel estimate (mesh bbox
     * dilated by the narrow band) exceeds this; 0 = unlimited. Peak memory is
     * roughly 8 bytes per estimated voxel. */
    uint64_t max_voxels;
    float voxels_per_unit; /* grid resolution, voxels per world unit; > 0 */
    float half_width;      /* narrow band half-width in voxels; 0 => 3.0 */
    uint32_t value_type;   /* picovdb_value_type */
    float rotate_deg[3];   /* applied about x, then y, then z; zeros = none */
} picovdb_mesh_to_grid_options;

typedef struct {
    uint64_t active_voxels;
    uint64_t surface_voxels;
    uint32_t leaf_count;
    uint32_t lower_count;
    uint32_t upper_count;
    int32_t bbox_min[3]; /* index-space bounds of active voxels */
    int32_t bbox_max[3];
    float world_min[3]; /* post-rotation mesh bounds, world units */
    float world_max[3];
} picovdb_grid_stats;

typedef struct {
    const uint8_t *data; /* encoded .pvdb; release with picovdb_buffer_free */
    size_t len;
    picovdb_grid_stats stats;
} picovdb_buffer;

uint32_t picovdb_abi_version(void);

/* Allocation helpers for the wasm caller to stage input bytes inside linear
 * memory; native callers can pass their own pointers and ignore these. */
void *picovdb_alloc(size_t len);
void picovdb_free(void *ptr, size_t len);

int32_t picovdb_stl_info(const uint8_t *stl, size_t len, picovdb_mesh_info *out);

/* Rasterize a triangle mesh into an encoded .pvdb narrow-band SDF grid.
 * points are xyz triples (world units), triangles are vertex index triples. */
int32_t picovdb_mesh_to_grid(const float *points, uint32_t point_count,
                             const uint32_t *triangles, uint32_t triangle_count,
                             const picovdb_mesh_to_grid_options *opts,
                             picovdb_buffer *out);

/* One-shot STL (binary or ASCII) -> encoded .pvdb; see picovdb_mesh_to_grid. */
int32_t picovdb_stl_to_grid(const uint8_t *stl, size_t len,
                            const picovdb_mesh_to_grid_options *opts,
                            picovdb_buffer *out);

void picovdb_buffer_free(picovdb_buffer *buf);

/* Static string, never freed. */
const char *picovdb_error_string(int32_t code);

#ifdef __cplusplus
}
#endif

#endif /* PICOVDB_H */
