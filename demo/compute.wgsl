struct Input {
    camera_matrix: mat4x4f,
    fov_scale: f32, // tan(fov * 0.5)
    time_delta: f32,
    pixel_radius: f32, // Cone spread per unit distance: 1 / (resolution.y * focal_length)
    debug_iterations: u32, // 0 = normal rendering, 1 = debug iteration heatmap
}

// --- Object types ---
const OBJECT_TYPE_UNKNOWN: u32 = 0u;
const OBJECT_TYPE_VDB: u32 = 1u;
const OBJECT_TYPE_SDF: u32 = 2u;

struct Object { // 144
    object_type: u32,
    type_index: u32,
    material_index: u32,
    _pad: u32,
    transform: mat4x4f,
    transform_inverse: mat4x4f,
}

struct Material { // 32
    color: vec3f,
    albedo: f32,
    metallic: f32,
    roughness: f32,
    _pad: array<f32, 2>,
}

// --- Bind group 0: per-frame ---
@group(0) @binding(0) var<uniform> input: Input;
@group(0) @binding(1) var<storage> objects: array<Object>;

// -- Bind group 1: data ---
@group(1) @binding(0) var<storage> picovdb_grids: array<PicoVDBGrid>;
@group(1) @binding(1) var<storage> picovdb_roots: array<PicoVDBRoot>;
@group(1) @binding(2) var<storage> picovdb_uppers: array<PicoVDBUpper>;
@group(1) @binding(3) var<storage> picovdb_lowers: array<PicoVDBLower>;
@group(1) @binding(4) var<storage> picovdb_leaves: array<PicoVDBLeaf>;
@group(1) @binding(5) var<storage> picovdb_buffer: array<u32>;

// --- Bind group 2: pass ---
@group(2) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;

const MAX_DIST: f32 = 1e7;

struct Intersection {
    distance: f32,
    object_index: i32,
    iterations: u32,
    normal: vec3f,
}

fn no_intersection() -> Intersection {
    return Intersection(MAX_DIST, -1, 0, vec3f(0));
}

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

fn intersect_picovdb(
    ray: Ray,
    grid_index: u32,
    hit_distance: ptr<function, f32>,
    hit_normal: ptr<function, vec3f>,
    hit_iterations: ptr<function, u32>,
) -> bool {
    let tmin = 0.0;
    let tmax = 10000.0;

    let grid = picovdb_grids[grid_index];
    var accessor: PicoVDBReadAccessor;
    picovdbReadAccessorInit(&accessor, grid_index);

    // Inside Check (Works even if camera is in background space)
    let start_val = picovdbSampleTrilinear(&accessor, grid, ray.origin);
    if start_val < 0.0 {
        *hit_distance = tmin;
        *hit_normal = -ray.direction;
        return true;
    }

    return picovdbHDDAZeroCrossing(
        &accessor, grid, ray.origin, tmin, ray.direction, tmax, input.pixel_radius, hit_distance, hit_normal, hit_iterations,
    );
}

fn intersect_sdf(
    ray: Ray,
    index: u32,
    hit_distance: ptr<function, f32>,
    hit_normal: ptr<function, vec3f>,
    iterations: ptr<function, u32>,
) -> bool {
    switch index {
        case 0u: { // ground plane at y=0 in index space
            if ray.direction.y >= 0.0 || abs(ray.direction.y) < 0.001 {
                return false;
            }
            let t = -ray.origin.y / ray.direction.y;
            if t < 0.001 {
                return false;
            }
            *hit_distance = t;
            *hit_normal = vec3f(0, 1, 0);
            return true;
        }
        case default: { return false; }
    }
}

fn intersect_scene(world_ray: Ray, iterations: ptr<function, u32>) -> Intersection {
    var min_hit = no_intersection();
    for (var i = 0i; i < i32(arrayLength(&objects)); i++) {
        let obj = objects[i];
        let idx_origin = (obj.transform * vec4f(world_ray.origin, 1.0)).xyz;
        let idx_dir_unnorm = (obj.transform * vec4f(world_ray.direction, 0.0)).xyz;
        let idx_direction = normalize(idx_dir_unnorm);
        let index_ray = Ray(idx_origin, idx_direction);

        var hit = false;
        var hit_distance = MAX_DIST;
        var hit_normal = vec3f(0);
        var hit_iterations = 0u;
        switch obj.object_type {
            case OBJECT_TYPE_VDB: {
                hit = intersect_picovdb(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);
            }
            case OBJECT_TYPE_SDF: {
                hit = intersect_sdf(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);
            }
            case default: { 
                hit = false;
            }
        }
        *iterations += hit_iterations;
        if !hit {
            continue;
        }
        let index_hit_point = index_ray.origin + index_ray.direction * hit_distance;
        let world_hit_point = (obj.transform_inverse * vec4f(index_hit_point, 1.0)).xyz;
        let world_distance = length(world_hit_point - world_ray.origin);
        if world_distance >= min_hit.distance {
            continue;
        }

        min_hit.distance = world_distance;
        min_hit.object_index = i;
        min_hit.normal = (obj.transform_inverse * vec4f(hit_normal, 0.0)).xyz;
    }
    min_hit.normal = normalize(min_hit.normal);
    return min_hit;
}

fn generate_camera_ray(screen_coord: vec2f, screen_size: vec2f) -> Ray {
    // Convert to normalized coordinates [-1, 1k
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

fn get_material(hit: Intersection, obj: Object) -> Material {
    switch obj.material_index {
        case 0u: {
            return Material(vec3f(0.2, 0.5, 1.0), 0, 0, 0, array(0,0));
        }
        case 1u: {
            return Material(vec3f(0.8, 0.8, 0.8), 0, 0, 0, array(0,0));
        }
        default: {
            return Material(vec3f(0.95, 0.95, 1.0), 0, 0, 0, array(0,0));
        }
    }
}

fn compute_color(ray: Ray, hit: Intersection) -> vec3f {
    if hit.object_index < 0 {
        return vec3f(0.95, 0.95, 1.0);
    }
    let obj = objects[hit.object_index];
    let material = get_material(hit, obj);
    let hit_point = ray.origin + ray.direction * hit.distance;
    let light_pos = vec3f(20.0, 30.0, 10.0);
    let light_dir = normalize(light_pos - hit_point);
    let diffuse = max(dot(hit.normal, light_dir), 0.0);
    let ambient = 0.15;
    return material.color * (ambient + diffuse * 0.85);
}

@compute @workgroup_size(8, 8)
fn computeMain(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }

    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(dims));
    var iterations: u32;
    let hit = intersect_scene(ray, &iterations);
    var color = compute_color(ray, hit);

    // Debug iteration visualization: override color with heatmap
    if (input.debug_iterations == 1u) {
        let heat = clamp(f32(iterations) / 128.0, 0.0, 1.0);
        color = vec3f(0.0, heat, 0.0);
    }

    textureStore(output_texture, global_id.xy, vec4f(color, 1.0));
}

// --- PBR ---

const PI: f32 = 3.14159265359;

fn distributionGGX(n: vec3f, h: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let nDotH = max(dot(n, h), 0.0);
  let nDotH2 = nDotH * nDotH;
  var denom = (nDotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;
  return a2 / denom;
}

fn geometrySchlickGGX(nDotV: f32, roughness: f32) -> f32 {
  let r = (roughness + 1.0);
  let k = (r * r) / 8.0;
  return nDotV / (nDotV * (1.0 - k) + k);
}

fn geometrySmith(n: vec3f, v: vec3f, l: vec3f, roughness: f32) -> f32 {
  let nDotV = max(dot(n, v), 0.0);
  let nDotL = max(dot(n, l), 0.0);
  let ggx2 = geometrySchlickGGX(nDotV, roughness);
  let ggx1 = geometrySchlickGGX(nDotL, roughness);
  return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelSchlickRoughness(cosTheta: f32, f0: vec3f, roughness: f32) -> vec3f {
  return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
fn radicalInverseVdC(bits: u32) -> f32 {
  var result = bits;
  result = (bits << 16u) | (bits >> 16u);
  result = ((result & 0x55555555u) << 1u) | ((result & 0xAAAAAAAAu) >> 1u);
  result = ((result & 0x33333333u) << 2u) | ((result & 0xCCCCCCCCu) >> 2u);
  result = ((result & 0x0F0F0F0Fu) << 4u) | ((result & 0xF0F0F0F0u) >> 4u);
  result = ((result & 0x00FF00FFu) << 8u) | ((result & 0xFF00FF00u) >> 8u);
  return f32(result) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2f {
  return vec2f(f32(i) / f32(n), radicalInverseVdC(i));
}

fn importanceSampleGGX(xi: vec2f, n: vec3f, roughness: f32) -> vec3f {
  let a = roughness * roughness;

  let phi = 2.0 * PI * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  // from spherical coordinates to cartesian coordinates - halfway vector
  let h = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  // from tangent-space H vector to world-space sample vector
  let up: vec3f = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);

  let sampleVec = tangent * h.x + bitangent * h.y + n * h.z;
  return normalize(sampleVec);
}

  // toneMapping implements lottes.
fn toneMapping(color: vec3f) -> vec3f {
    let a = vec3f(1.6);
    let d = vec3f(0.977);
    let hdrMax = vec3f(8.0);
    let midIn = vec3f(0.18);
    let midOut = vec3f(0.267);

    let b = (-pow(midIn, a) + pow(hdrMax, a) * midOut) / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    let c = (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(color, a) / (pow(color, a * d) * b + c);
}
