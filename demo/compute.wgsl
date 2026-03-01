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
@group(0) @binding(2) var<storage, read> skyState: SkyState;

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
                // Skip fog grids during surface intersection — they use volumetric marching instead
                let vdb_grid = picovdb_grids[obj.type_index];
                if (vdb_grid.gridType != GRID_TYPE_FOG_FLOAT) {
                    hit = intersect_picovdb(index_ray, obj.type_index, &hit_distance, &hit_normal, &hit_iterations);
                }
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
            return Material(vec3f(0.0, 0.1, 1.0), 0.0, 0.0, 0.1, array(0,0));
        }
        case 1u: {
            return Material(vec3f(0.2, 0.2, 0.2), 1.0, 1.0, 1.0, array(0,0));
        }
        default: {
            return Material(vec3f(0.0, 0.0, 0.0), 0, 0, 0, array(0,0));
        }
    }
}

fn traceShadowRay(origin: vec3f, normal: vec3f) -> f32 {
    // Offset origin slightly along normal to avoid self-intersection
    let shadowOrigin = origin + normal * 0.01;
    let shadowRay = Ray(shadowOrigin, skyState.sunDirection);
    var iterations: u32;
    let hit = intersect_scene(shadowRay, &iterations);
    if hit.object_index >= 0 {
        return 0.0;  // Fully shadowed
    }
    return 1.0;  // Fully lit
}

fn applyFog(color: vec3f, distance: f32, rayDir: vec3f, fogDensity: f32) -> vec3f {
    let fogAmount = 1.0 - exp(-distance * fogDensity);
    // Blend between sky color and slight blue haze
    let skyFog = skyRadianceRGB(rayDir, false);
    let hazeTint = vec3f(0.7, 0.8, 1.0);  // Subtle blue
    let fogColor = mix(skyFog, skyFog * hazeTint, 0.3);
    return mix(color, fogColor, fogAmount);
}

fn computeColor(ray: Ray, hit: Intersection) -> vec3f {
    if hit.object_index < 0 {
        return skyRadianceRGB(ray.direction, true);
    }

    let obj = objects[hit.object_index];
    let material = get_material(hit, obj);
    let hitPoint = ray.origin + ray.direction * hit.distance;

    let albedo = material.color;
    let metallic = material.metallic;
    let roughness = max(material.roughness, 0.04);  // Clamp to avoid division issues
    let ao = 1.0;  // Could come from material or SSAO later

    let n = normalize(hit.normal);
    let v = normalize(-ray.direction);
    let l = skyState.sunDirection;
    let h = normalize(v + l);
    let r = reflect(-v, n);

    let f0 = mix(vec3f(0.04), albedo, metallic);

    // Direct sun lighting
    let nDotL = max(dot(n, l), 0.0);
    var lo = vec3f(0.0);

    if nDotL > 0.0 {
        let shadow = traceShadowRay(hitPoint, n);
        if shadow > 0.0 {
            let sunRadiance = sunIrradiance();

            let d = distributionGGX(n, h, roughness);
            let g = geometrySmith(n, v, l, roughness);
            let f = fresnelSchlick(max(dot(h, v), 0.0), f0);

            let numerator = d * g * f;
            let denominator = 4.0 * max(dot(n, v), 0.0) * nDotL + 0.0001;
            let specular = numerator / denominator;

            let kS = f;
            var kD = vec3f(1.0) - kS;
            kD *= 1.0 - metallic;

            lo = (kD * albedo / PI + specular) * sunRadiance * nDotL;
        }
    }

    // Ambient / environment lighting
    let f = fresnelSchlickRoughness(max(dot(n, v), 0.0001), f0, roughness);
    let kS = f;
    var kD = vec3f(1.0) - kS;
    kD *= 1.0 - metallic;

    // Diffuse irradiance from sky hemisphere
    let irradiance = skyIrradiance(n);
    let diffuse = irradiance * albedo / PI;

    // Specular reflection, sample sky in reflection direction.
    // For rough surfaces, we'd ideally blur/filter, but single sample works ok
    let prefilteredColor = skyRadianceRGB(r, true);
    
    // Approximate the BRDF integration (simplified - no LUT)
    // This is a rough approximation of the split-sum BRDF
    let nDotV = max(dot(n, v), 0.0);
    let envBRDF = vec2f(
        1.0 - roughness * 0.5,  // Approximate F scale
        roughness * 0.5         // Approximate F bias
    );
    let specular = prefilteredColor * (f * envBRDF.x + envBRDF.y);

    let ambient = (kD * diffuse + specular) * ao;

    var color = ambient + lo;
    if hit.distance > 10 {
        color = applyFog(color, hit.distance-10, ray.direction, 0.01);
    }
    return color;
}

// toneMapping implements ACES
fn toneMapping(color: vec3f) -> vec3f {
    let exposure = 0.05; // Tuneable
    let exposed = color * exposure;
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return (exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e);
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(output_texture);
    if global_id.x >= dims.x || global_id.y >= dims.y { return; }

    let ray = generate_camera_ray(vec2f(global_id.xy) + 0.5, vec2f(dims));
    var iterations: u32;
    let hit = intersect_scene(ray, &iterations);

    var color = computeColor(ray, hit);

    //// Fog volume compositing (HDR, before tone mapping)
    //for (var i = 0i; i < i32(arrayLength(&objects)); i++) {
    //    let obj = objects[i];
    //    if (obj.object_type != OBJECT_TYPE_VDB) { continue; }
    //    let fog_grid = picovdb_grids[obj.type_index];
    //    if (fog_grid.gridType != GRID_TYPE_FOG_FLOAT) { continue; }

    //    let idx_origin     = (obj.transform * vec4f(ray.origin, 1.0)).xyz;
    //    let idx_dir_unnorm = (obj.transform * vec4f(ray.direction, 0.0)).xyz;
    //    let idx_direction  = normalize(idx_dir_unnorm);

    //    var fog_acc: PicoVDBReadAccessor;
    //    picovdbReadAccessorInit(&fog_acc, u32(obj.type_index));

    //    let fog = picovdbFogMarch(&fog_acc, fog_grid, idx_origin, idx_direction, 0.0, 1e6);
    //    color = fog.color + fog.transmittance * color;
    //}

    color = toneMapping(color);
    color = pow(color, vec3f(1.0 / 2.2));  // Gamma correction

    if input.debug_iterations == 1u {
        let heat = clamp(f32(iterations) / 128.0, 0.0, 1.0);
        color = vec3f(0.0, heat, 0.0);
    }
    textureStore(output_texture, global_id.xy, vec4f(color, 1.0));
}

// ============================================================================
// PBR
// ============================================================================

const PI = 3.14159265359;

fn distributionGGX(n: vec3f, h: vec3f, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let nDotH = max(dot(n, h), 0.0);
    let nDotH2 = nDotH * nDotH;
    let num = a2;
    let denom = nDotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX(nDotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
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


// ============================================================================
// Sky Model
// ============================================================================
const CHANNEL_R = 0u;
const CHANNEL_G = 1u;
const CHANNEL_B = 2u;
const SOLAR_RADIUS_RADIANS = 0.004450589; // 0.255 degrees

struct SkyState {
    sunDirection: vec3<f32>,
    params: array<f32, 27>,
    skyRadiances: array<f32, 3>,
    solarRadiances: array<f32, 3>,
}

fn radiance(theta: f32, gamma: f32, channel: u32, includeSun: bool) -> f32 {
    let r = skyState.skyRadiances[channel];
    let idx = 9u * channel;
    let p0 = skyState.params[idx + 0u];
    let p1 = skyState.params[idx + 1u];
    let p2 = skyState.params[idx + 2u];
    let p3 = skyState.params[idx + 3u];
    let p4 = skyState.params[idx + 4u];
    let p5 = skyState.params[idx + 5u];
    let p6 = skyState.params[idx + 6u];
    let p7 = skyState.params[idx + 7u];
    let p8 = skyState.params[idx + 8u];

    let cosGamma = cos(gamma);
    let cosGamma2 = cosGamma * cosGamma;
    let cosTheta = abs(cos(theta));

    let expM = exp(p4 * gamma);
    let rayM = cosGamma2;
    let mieMLhs = 1.0 + cosGamma2;
    let mieMRhs = pow(1.0 + p8 * p8 - 2.0 * p8 * cosGamma, 1.5f);
    let mieM = mieMLhs / mieMRhs;
    let zenith = sqrt(cosTheta);
    let radianceLhs = 1.0 + p0 * exp(p1 / (cosTheta + 0.01));
    let radianceRhs = p2 + p3 * expM + p5 * rayM + p6 * mieM + p7 * zenith;
    let radianceDist = radianceLhs * radianceRhs;

    let solarDiskRadius = gamma / SOLAR_RADIUS_RADIANS;
    let solarRadiance = select(0.0, skyState.solarRadiances[channel], includeSun && solarDiskRadius <= 1.0);

    return r * radianceDist + solarRadiance;
}

fn skyRadianceRGB(direction: vec3f, includeSun: bool) -> vec3f {
    let v = normalize(direction);
    let s = skyState.sunDirection;
    let theta = acos(clamp(v.y, -1.0, 1.0));
    let gamma = acos(clamp(dot(v, s), -1.0, 1.0));
    return vec3f(
        radiance(theta, gamma, CHANNEL_R, includeSun),
        radiance(theta, gamma, CHANNEL_G, includeSun),
        radiance(theta, gamma, CHANNEL_B, includeSun)
    );
}

fn sunIrradiance() -> vec3f {
    // Solar radiance * solid angle of sun disk
    let sunSolidAngle = PI * SOLAR_RADIUS_RADIANS * SOLAR_RADIUS_RADIANS;
    return vec3f(
        skyState.solarRadiances[0],
        skyState.solarRadiances[1],
        skyState.solarRadiances[2]
    ) * sunSolidAngle;
}

fn skyIrradiance(n: vec3f) -> vec3f {
    var irradiance = vec3f(0.0);
    let SAMPLE_COUNT = 16u;
    
    for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let xi = hammersley(i, SAMPLE_COUNT);
        
        // Cosine-weighted hemisphere sampling
        let phi = 2.0 * PI * xi.x;
        let cosTheta = sqrt(1.0 - xi.y);  // Cosine-weighted
        let sinTheta = sqrt(xi.y);

        // To world space
        let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
        let tangent = normalize(cross(up, n));
        let bitangent = cross(n, tangent);

        let sampleDir = normalize(
            tangent * cos(phi) * sinTheta +
            bitangent * sin(phi) * sinTheta +
            n * cosTheta
        );
        irradiance += skyRadianceRGB(sampleDir, false);
    }
    return irradiance * PI / f32(SAMPLE_COUNT);
}
