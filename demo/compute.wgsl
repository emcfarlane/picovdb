struct Input {
    camera_matrix: mat4x4f,
    fov_scale: f32, // tan(fov * 0.5)
    time_delta: f32,
    _pad: vec2u,
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
}

// Define a structure for an Axis-Aligned Bounding Box (AABB)
struct AABB {
    min_bounds: vec3<f32>,
    max_bounds: vec3<f32>,
};


// Define a structure to hold the intersection results
struct IntersectionInterval {
    // The parametric distance to the nearest intersection point
    t_near: f32,
    // The parametric distance to the farthest intersection point
    t_far: f32,
    // True if a valid intersection occurs in front of the ray origin
    hit: bool,
};

/**
 * Checks if a ray intersects an AABB and returns the min/max distances.
 *
 * @param box The AABB to check for intersection.
 * @param ray The ray used for intersection testing.
 * @returns An IntersectionInterval structure with hit status and distances.
 */
fn intersect_ray_aabb_interval(box: AABB, ray: Ray) -> IntersectionInterval {
    // Calculate intersection distances for the x, y, and z slabs
    let ray_inv_direction = 1.0 / ray.direction; // TODO: precompute?
    let t0 = (box.min_bounds - ray.origin) * ray_inv_direction;
    let t1 = (box.max_bounds - ray.origin) * ray_inv_direction;

    // Order the distances for each axis to get t_min and t_max for each slab
    let t_min_xyz = min(t0, t1);
    let t_max_xyz = max(t0, t1);

    // Find the maximum of all near intersections (t_near) and the minimum of all far intersections (t_far)
    let t_near_final = max(max(t_min_xyz.x, t_min_xyz.y), t_min_xyz.z);
    let t_far_final = min(min(t_max_xyz.x, t_max_xyz.y), t_max_xyz.z);

    // Check for a valid intersection:
    // 1. The near point must be less than or equal to the far point (intervals overlap).
    // 2. The far point must not be behind the ray origin (t_far_final >= 0.0).
    let hit_valid = t_near_final <= t_far_final && t_far_final >= 0.0;

    // If a hit occurred, ensure t_near_final is >= 0 if we only care about hits in front of the origin.
    // If the origin is inside the box, t_near_final might be negative, 
    // but the intersection technically starts "at the origin" (t=0) for visual rendering purposes.
    let final_t_near = max(0.0, t_near_final);

    return IntersectionInterval(final_t_near, t_far_final, hit_valid);
}

fn intersect_picovdb(
    world_ray: Ray,
    world_to_index: mat4x4f,
    index_to_world: mat4x4f,
) -> RayHit {
    let grid = picovdb_grids[0];

    // Transform World Ray directly to Index Space
    // Direction stays unnormalized initially to preserve T-metric scale
    let idx_origin = (world_to_index * vec4f(world_ray.origin, 1.0)).xyz;
    let idx_dir_unnorm = (world_to_index * vec4f(world_ray.direction, 0.0)).xyz;
    let idx_dir_len = length(idx_dir_unnorm);
    let idx_direction = idx_dir_unnorm / idx_dir_len;

    let index_ray = Ray(idx_origin, idx_direction);

    let bbox = AABB(vec3f(grid.indexBoundsMin), vec3f(grid.indexBoundsMax));
    let intersection = intersect_ray_aabb_interval(bbox, index_ray);
    if !intersection.hit {
        return RayHit(-1.0, vec3f(0.0));
    }

    var accessor: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&accessor, 0);

    // Use HDDA zero crossing detection (all parameters in index space)
    var hit_t_index: f32;
    var hit_value: f32;
     let hit = picovdbHDDAZeroCrossing(
        &accessor,
        grid,
        index_ray.origin,
        intersection.t_near,
        index_ray.direction,
        intersection.t_far,
        &hit_t_index,
        &hit_value
    );
    if !hit {
        return RayHit(-1.0, vec3f(0));
    }

    // Calculate World Space Data
    let index_hit_point = index_ray.origin + index_ray.direction * hit_t_index;
    let world_hit_point = (index_to_world * vec4f(index_hit_point, 1.0)).xyz;

    // Distance in world space
    let world_distance = length(world_hit_point - world_ray.origin);

    // Normal Calculation (Gradient in Index Space -> World Space)
    let ijk_base = vec3i(floor(index_hit_point));
    let uvw = fract(index_hit_point);
    let stencil = picovdbSampleStencil(&accessor, grid, ijk_base);
    let index_gradient = picovdbTrilinearGradient(uvw, stencil);
    
    // Normals transform by the Transpose of the Inverse (World-to-Index)
    // Or more simply: transform the gradient by the inverse-world matrix
    let world_gradient = (index_to_world * vec4f(index_gradient, 0.0)).xyz;
    let normal = normalize(world_gradient);

    return RayHit(world_distance, normal);
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

// Ground plane intersection (only visible from above)
fn intersect_ground_plane(ray: Ray, plane_y: f32) -> f32 {
    if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {
        return -1.0;
    }
    let t = (plane_y - ray.origin.y) / ray.direction.y;
    return select(-1.0, t, t > 0.001);
}

// Shadow ray test
fn is_in_shadow(point: vec3f, normal: vec3f, light_dir: vec3f, light_distance: f32) -> bool {
    // Offset ray origin along surface normal to avoid self-intersection
    let shadow_ray = Ray(point + normal * 0.01, light_dir);
    let shadow_hit = intersect_picovdb(
        shadow_ray,
        input.transform_matrix,
        input.transform_inverse_matrix,
    );
    return shadow_hit.distance > 0.0 && shadow_hit.distance < light_distance;
}

fn raymarch_scene_graph(ray: Ray) -> vec3f {
    let ground_y = -2.0;
    
    // Check volume intersection
    let volume_hit = intersect_picovdb(
        ray,
        input.transform_matrix,
        input.transform_inverse_matrix,
    );
    
    // Check ground plane intersection  
    let ground_t = intersect_ground_plane(ray, ground_y);
    
    let light_pos = vec3f(20.0, 30.0, 10.0);
    let ambient = vec3f(0.15);
    
    // Determine closest hit
    var closest_t = 10000.0;
    var hit_volume = false;
    var hit_ground = false;
    
    if volume_hit.distance > 0.0 && volume_hit.distance < closest_t {
        closest_t = volume_hit.distance;
        hit_volume = true;
    }
    
    if ground_t > 0.0 && ground_t < closest_t {
        closest_t = ground_t;
        hit_ground = true;
        hit_volume = false;
    }
    
    if hit_volume {
        // Blue bunny material
        let hit_point = ray.origin + ray.direction * volume_hit.distance;
        let light_dir = normalize(light_pos - hit_point);
        let light_distance = length(light_pos - hit_point);
        
        // Diffuse lighting
        let diffuse = max(dot(volume_hit.normal, light_dir), 0.0);
        
        // Shadow test
        let shadow_factor = select(1.0, 0.3, is_in_shadow(hit_point, volume_hit.normal, light_dir, light_distance));
        
        // Blue bunny color
        let base_color = vec3f(0.2, 0.5, 1.0);
        return base_color * (ambient + diffuse * shadow_factor * 0.8);
        
    } else if hit_ground {
        // Shadow-only ground plane
        let hit_point = ray.origin + ray.direction * ground_t;
        let ground_normal = vec3f(0.0, 1.0, 0.0);
        let light_dir = normalize(light_pos - hit_point);
        let light_distance = length(light_pos - hit_point);
        
        // Distance fade from center
        let distance_from_center = length(hit_point.xz);
        let fade_radius = 30.0;
        let fade_factor = 1.0 - smoothstep(fade_radius - 10.0, fade_radius, distance_from_center);
        
        // If completely faded, return background
        if fade_factor <= 0.001 {
            return vec3f(0.95, 0.95, 1.0);
        }
        
        // Shadow test - only show ground if in shadow
        let in_shadow = is_in_shadow(hit_point, ground_normal, light_dir, light_distance);
        
        if in_shadow {
            // Dark shadow color
            let shadow_color = vec3f(0.3, 0.3, 0.35);
            let background_color = vec3f(0.95, 0.95, 1.0);
            return mix(background_color, shadow_color, fade_factor * 0.7);
        }
    }
    return vec3f(0.95, 0.95, 1.0); // Background color
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
fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let screen_size = textureDimensions(outputTexture);
    
    // Early exit for out-of-bounds threads
    if global_id.x >= screen_size.x || global_id.y >= screen_size.y {
        return;
    }

    // Generate ray
    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(screen_size));
    
    // Raymarching using scene graph
    let color = raymarch_scene_graph(ray);
    
    // Write result
    textureStore(outputTexture, global_id.xy, vec4f(color, 1.0));
}
