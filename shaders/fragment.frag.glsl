#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;

    vec4 camera_position;
    vec4 resolution;
} ubo;

layout(binding = 1) uniform usampler3D voxelTexture;

// Output color
layout(location = 0) out vec4 outColor;

const int MAX_STEPS = 100;
const float MAX_DISTANCE = 100.0;
const float EPSILON = 0.001;

float sampleVoxel(vec3 pos, vec3 gridMin, vec3 gridMax) {
    if (!all(greaterThanEqual(pos, gridMin)) || !all(lessThanEqual(pos, gridMax))) {
        return 0.0;  // Outside grid
    }

    vec3 texCoord = (pos - gridMin) / (gridMax - gridMin);
    ivec3 voxelCoord = ivec3(texCoord * float(32));
    uint voxelValue = texelFetch(voxelTexture, voxelCoord, 0).r;

    return float(voxelValue);  // Convert to float for gradient calculation
}

vec3 calculateVoxelNormal(vec3 pos, vec3 gridMin, vec3 gridMax) {
    float delta = 0.5;  // Small offset for sampling neighbors

    // Sample voxel density at 6 neighboring positions
    // Convert each position to voxel coordinate and sample

    vec3 texCoord = (pos - gridMin) / (gridMax - gridMin);

    // Sample in each axis direction
    float dx_pos = sampleVoxel(pos + vec3(delta, 0, 0), gridMin, gridMax);
    float dx_neg = sampleVoxel(pos - vec3(delta, 0, 0), gridMin, gridMax);
    float dy_pos = sampleVoxel(pos + vec3(0, delta, 0), gridMin, gridMax);
    float dy_neg = sampleVoxel(pos - vec3(0, delta, 0), gridMin, gridMax);
    float dz_pos = sampleVoxel(pos + vec3(0, 0, delta), gridMin, gridMax);
    float dz_neg = sampleVoxel(pos - vec3(0, 0, delta), gridMin, gridMax);

    // Gradient = difference in each direction
    vec3 normal = vec3(
        dx_pos - dx_neg,
        dy_pos - dy_neg,
        dz_pos - dz_neg
    );

    return normalize(normal);
}

bool rayMarchVoxels(vec3 origin, vec3 direction, vec3 gridMin, vec3 gridMax, out vec3 hitPos) {
    float t = 0.0;
    float tMax = 50.0;  // Max distance to march
    float stepSize = 0.1;  // Step size (smaller = more accurate but slower)

    for (int i = 0; i < 500; i++) {
        vec3 pos = origin + direction * t;

        // Check if we're inside the grid bounds
        if (all(greaterThanEqual(pos, gridMin)) && all(lessThanEqual(pos, gridMax))) {
            // Convert world position to texture coordinates [0, 1]
            vec3 texCoord = (pos - gridMin) / (gridMax - gridMin);

            // Convert to voxel indices [0, 32)
            ivec3 voxelCoord = ivec3(texCoord * float(32));  // CHUNK_SIZE = 32

            // Sample the voxel
            uint voxelValue = texelFetch(voxelTexture, voxelCoord, 0).r;

            if (voxelValue > 0u) {  // Non-empty voxel
                hitPos = pos;
                return true;
            }
        }

        t += stepSize;
        if (t > tMax) {
            break;
        }
    }

    return false;
}

void main() {
    float aspect_ratio = ubo.resolution.z;
    vec2 screen_size = ubo.resolution.xy;

    vec2 uv = (gl_FragCoord.xy / screen_size) * 2.0 - 1.0;
    uv.x *= aspect_ratio;

    mat3 camera_rotation = transpose(mat3(ubo.view));

    vec3 ray_dir_camera = normalize(vec3(uv, -1.0));  // Assuming a simple camera looking down -Z

    vec3 ray_dir_world = camera_rotation * ray_dir_camera;

    vec3 ray_origin = ubo.camera_position.xyz;

    // Position the grid in world space
    vec3 gridMin = vec3(-16.0, -16.0, -16.0);  // Grid center
    vec3 gridMax = vec3(16.0, 16.0, 16.0);

    vec3 hitPos;
    if (rayMarchVoxels(ray_origin, ray_dir_world, gridMin, gridMax, hitPos)) {
        // Hit something - calculate normal and light it

        vec3 light_dir = normalize(vec3(1.0, 1.0, -1.0));

        // vec3 normal = calculateNormal(hitPos);
        vec3 normal = calculateVoxelNormal(hitPos, gridMin, gridMax);

        float ambient = 0.1;
        float diffuse = max(0.0, dot(normal, light_dir));
        float brightness = ambient + (1.0 - ambient) * diffuse;

        outColor = vec4(
            clamp(vec3(0.4, 0.6, 0.8) * brightness, 0.0, 1.0),
            1.0
        );
    } else {
        outColor = vec4(0.1, 0.1, 0.1, 1.0);  // Miss: black color
    }
}
