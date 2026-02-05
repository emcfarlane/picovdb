struct Input {
    camera_matrix: mat4x4f,
    fov_scale: f32, // tan(fov * 0.5)
    time_delta: f32,
    pixel_radius: f32, // Cone spread per unit distance: 1 / (resolution.y * focal_length)
    debug_iterations: u32, // 0 = normal rendering, 1 = debug iteration heatmap
    transform_matrix: mat4x4f,
    transform_inverse_matrix: mat4x4f,
}

@group(0) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> input: Input;
@group(0) @binding(2) var<storage> picovdb_grids: array<PicoVDBGrid>;
@group(0) @binding(3) var<storage> picovdb_roots: array<PicoVDBRoot>;
@group(0) @binding(4) var<storage> picovdb_uppers: array<PicoVDBUpper>;
@group(0) @binding(5) var<storage> picovdb_lowers: array<PicoVDBLower>;
@group(0) @binding(6) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
@group(0) @binding(7) var<storage> picovdb_buffer: array<u32>;

struct RayHit {
    distance: f32,
    normal: vec3f,
    iterations: u32,
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

fn intersect_picovdb(
    world_ray: Ray,
    world_to_index: mat4x4f,
    index_to_world: mat4x4f,
) -> RayHit {
    let grid = picovdb_grids[0];
    let idx_origin = (world_to_index * vec4f(world_ray.origin, 1.0)).xyz;
    let idx_dir_unnorm = (world_to_index * vec4f(world_ray.direction, 0.0)).xyz;
    let idx_dir_len = length(idx_dir_unnorm);
    let idx_direction = idx_dir_unnorm / idx_dir_len;

    let index_ray = Ray(idx_origin, idx_direction);
    let tmin = 0.0;
    let tmax = 10000.0;

    var accessor: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&accessor, 0);

    // Inside Check (Works even if camera is in background space)
    let start_val = picovdbSampleTrilinear(&accessor, grid, idx_origin);
    if start_val < 0.0 {
        return RayHit(0.01, -world_ray.direction, 0u);
    }

    var hit_distance: f32;
    var hit_normal: vec3f;
    var iterations: u32;
    let hit = picovdbHDDAZeroCrossing(
        &accessor, grid, index_ray.origin, tmin, index_ray.direction, tmax, input.pixel_radius, &hit_distance, &hit_normal, &iterations,
    );
    if !hit { return RayHit(-1.0, vec3f(0), iterations); }

    let index_hit_point = index_ray.origin + index_ray.direction * hit_distance;
    let world_hit_point = (index_to_world * vec4f(index_hit_point, 1.0)).xyz;
    let world_distance = length(world_hit_point - world_ray.origin);

    let normal = normalize((index_to_world * vec4f(hit_normal, 0.0)).xyz);
    return RayHit(world_distance, normal, iterations);
}

// Ground plane intersection (only visible from above)
fn intersect_ground_plane(ray: Ray, plane_y: f32) -> f32 {
    if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {
        return -1.0;
    }
    let t = (plane_y - ray.origin.y) / ray.direction.y;
    return select(-1.0, t, t > 0.001);
}

fn raymarch_scene_graph(ray: Ray, iterations: ptr<function, u32>) -> vec3f {
    let volume_hit = intersect_picovdb(ray, input.transform_matrix, input.transform_inverse_matrix);
    *iterations = volume_hit.iterations;
    let ground_t = intersect_ground_plane(ray, -2.0);

    let background = vec3f(0.95, 0.95, 1.0);
    let light_pos = vec3f(20.0, 30.0, 10.0);

    // Determine primary hit
    var t = 1e6f;
    var is_volume = false;
    if volume_hit.distance > 0.0 {
        t = volume_hit.distance;
        is_volume = true;
    }
    if ground_t > 0.0 && ground_t < t {
        t = ground_t;
        is_volume = false;
    }
    if t > 1e5f { return background; }

    let hit_point = ray.origin + ray.direction * t;
    let light_vec = light_pos - hit_point;
    let light_dir = normalize(light_vec);
    let light_dist = length(light_vec);
    if is_volume {
        var diffuse = 0.5 + 0.5 * dot(volume_hit.normal, light_dir);
        diffuse = diffuse * diffuse;

        // Use the optimized occlusion check for shadows
        var acc: PicoVDBReadAccessor; picovdbReadAccessorInit(&acc, 0);
        let in_shadow = picovdbHDDAIsOccluded(
            &acc, picovdb_grids[0],
            (input.transform_matrix * vec4f(hit_point + volume_hit.normal * 0.1, 1.0)).xyz,
            0.0,
            (input.transform_matrix * vec4f(light_dir, 0.0)).xyz,
            light_dist
        );
        let shadow = select(1.0, 0.8, in_shadow);
        return vec3f(0.2, 0.5, 1.0) * (diffuse * shadow);
    } else {
        // Simple ground shadow
        var acc: PicoVDBReadAccessor; picovdbReadAccessorInit(&acc, 0);
        let in_shadow = picovdbHDDAIsOccluded(
            &acc, picovdb_grids[0],
            (input.transform_matrix * vec4f(hit_point + vec3f(0, 0.1, 0), 1.0)).xyz,
            0.0,
            (input.transform_matrix * vec4f(light_dir, 0.0)).xyz,
            light_dist
        );
        let fade = 1.0 - smoothstep(20.0, 30.0, length(hit_point.xz));
        return mix(background, vec3f(0.3), select(0.0, fade * 0.7, in_shadow));
    }
}

fn generate_camera_ray(screen_coord: vec2f, screen_size: vec2f) -> Ray {
    // Convert to normalized coordinates [-1, 1]
    let uv = (screen_coord / screen_size) * 2.0 - 1.0;

    // Calculate aspect ratio
    let aspect_ratio = screen_size.x / screen_size.y;

    // Extract camera basis vectors from view matrix
    let right: vec3f = input.camera_matrix[0].xyz;
    let up: vec3f = input.camera_matrix[1].xyz;
    let forward: vec3f = -input.camera_matrix[2].xyz;

    // Extract camera position
    let camera_pos: vec3f = input.camera_matrix[3].xyz;

    // Calculate ray direction
    let ray_direction = normalize(
        forward + uv.x * right * aspect_ratio * input.fov_scale + uv.y * up * input.fov_scale
    );
    return Ray(camera_pos, ray_direction);
}

@compute @workgroup_size(16, 16)
fn computeMain(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
) {
    let screen_size = textureDimensions(outputTexture);

    // Early exit for out-of-bounds threads
    if (global_id.x >= screen_size.x || global_id.y >= screen_size.y) {
        return;
    }

    // Generate ray for this pixel
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(screen_size));
    var iterations = u32(0);
    var color = raymarch_scene_graph(ray, &iterations);

    // Debug iteration visualization: override color with heatmap
    if (input.debug_iterations == 1u) {
        // Scale coarse by 32 (typical range 0-32), fine by 128 (typical range 0-128)
        let heat = clamp(f32(iterations) / 128.0, 0.0, 1.0);
        color = vec3f(0.0, heat, 0.0);
    }

    // Write result
    textureStore(outputTexture, global_id.xy, vec4f(color, 1.0));
}
