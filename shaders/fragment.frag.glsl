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

// Output color
layout(location = 0) out vec4 outColor;

const int MAX_STEPS = 100;
const float MAX_DISTANCE = 100.0;
const float EPSILON = 0.001;

float sphereSDF(vec3 p, vec3 center, float radius) {
    return length(p - center) - radius;
}

float sceneSDF(vec3 p) {
    // Example: single sphere at origin with radius 1.0
    return sphereSDF(p, vec3(0.0, 0.0, -5.0), 1.0);
}

float rayMarch(vec3 origin, vec3 direction) {
    float t = 0.0;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 pos = origin + direction * t;
        float dist = sceneSDF(pos);

        if (dist < EPSILON) {
            return t;  // Hit
        }

        t += dist;

        if (t > MAX_DISTANCE) {
            break;  // Exceeded max distance
        }
    }

    return -1.0; // Miss
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

    float t = rayMarch(ray_origin, ray_dir_world);

    if (t > 0.0) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);  // Hit: red color
    } else {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);  // Miss: black color
    }

}
