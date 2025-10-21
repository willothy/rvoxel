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

struct OctreeNode {
    uint data;
    uint children[8];
};

layout(binding = 1) readonly buffer OctreeBuffer {
    OctreeNode nodes[];
};

// Output color
layout(location = 0) out vec4 outColor;

bool isLeafNode(uint index) {
    return (nodes[index].data >> 7) == 0u;
}

bool isBranchNode(uint index) {
    return (nodes[index].data >> 7) == 1u;
}

bool rayBoxIntersection(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, out float tEnter, out float tExit) {
    vec3 t0s = (boxMin - rayOrigin) / rayDir;
    vec3 t1s = (boxMax - rayOrigin) / rayDir;

    vec3 tMin = min(t0s, t1s);
    vec3 tMax = max(t0s, t1s);

    tEnter = max(max(tMin.x, tMin.y), max(tMin.z, 0.0));
    tExit = min(min(tMax.x, tMax.y), tMax.z);

    return tEnter < tExit && tExit > 0;
}

void sortChildrenByOrder(vec3 ray, uint children[8], out uint traversalOrderedChildren[8]) {
    uint i = 1;
    uint dirMask = (ray.x < 0 ? 4 : 0) | (ray.y < 0 ? 2 : 0) | (ray.z < 0 ? 1 : 0);

    for (uint i = 0; i < 8; i++) {
        uint childIndex = i ^ dirMask;

        // visit child
    }
}

struct StackEntry {
    uint nodeIndex;
    vec3 boxMin;
    vec3 boxMax;
};

void subdivide(vec3 parentMin, vec3 parentMax, vec3 center, uint childIdx, out vec3 childMin, out vec3 childMax) {
    bool xUpper = (childIdx & 4) != 0;
    bool yUpper = (childIdx & 2) != 0;
    bool zUpper = (childIdx & 1) != 0;

    childMin = vec3(
            xUpper ? center.x : parentMin.x,
            yUpper ? center.y : parentMin.y,
            zUpper ? center.z : parentMin.z
        );
    childMax = vec3(
            xUpper ? parentMax.x : center.x,
            yUpper ? parentMax.y : center.y,
            zUpper ? parentMax.z : center.z
        );
}

bool traverseOctree(
    vec3 rayOrigin,
    vec3 rayDir,
    vec3 gridMin,
    vec3 gridMax,
    out vec3 hitPos
) {
    StackEntry stack[64];
    uint stackPtr = 0;

    uint dirMask = (rayOrigin.x < 0 ? 4 : 0)
            | (rayOrigin.y < 0 ? 2 : 0)
            | (rayOrigin.z < 0 ? 1 : 0);

    stack[stackPtr++] = StackEntry(0, gridMin, gridMax);

    while (stackPtr > 0) {
        StackEntry entry = stack[--stackPtr];
        OctreeNode node = nodes[entry.nodeIndex];

        float tEnter, tExit;
        if (!rayBoxIntersection(rayOrigin, rayDir, entry.boxMin, entry.boxMax, tEnter, tExit)) {
            continue;
        }

        vec3 center = (entry.boxMin + entry.boxMax) * 0.5;
        if (isLeafNode(entry.nodeIndex)) {
            if ((node.data & 0x7f) != 0) {
                hitPos = center;
                return true;
            }

            continue;
        }

        for (uint i = 0; i < 8; i++) {
            uint childIdx = i ^ dirMask;

            vec3 childMin, childMax;
            subdivide(entry.boxMin, entry.boxMax, center, childIdx, childMin, childMax);

            stack[stackPtr++] = StackEntry(
                    node.children[childIdx],
                    childMin,
                    childMax
                );
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

    vec3 ray_dir_camera = normalize(vec3(uv, -1.0)); // Assuming a simple camera looking down -Z

    vec3 ray_dir_world = camera_rotation * ray_dir_camera;

    vec3 ray_origin = ubo.camera_position.xyz;

    // Position the grid in world space
    vec3 gridMin = vec3(-16.0, -16.0, -16.0); // Grid center
    vec3 gridMax = vec3(16.0, 16.0, 16.0);

    vec3 hitPos;
    // if (rayMarchVoxels(ray_origin, ray_dir_world, gridMin, gridMax, hitPos)) {
    //     // Hit something - calculate normal and light it
    //
    //     vec3 light_dir = normalize(vec3(1.0, 1.0, -1.0));
    //
    //     // vec3 normal = calculateNormal(hitPos);
    //     vec3 normal = calculateVoxelNormal(hitPos, gridMin, gridMax);
    //
    //     float ambient = 0.1;
    //     float diffuse = max(0.0, dot(normal, light_dir));
    //     float brightness = ambient + (1.0 - ambient) * diffuse;
    //
    //     outColor = vec4(
    //         clamp(vec3(0.4, 0.6, 0.8) * brightness, 0.0, 1.0),
    //         1.0
    //     );
    if (traverseOctree(ray_origin, ray_dir_world, gridMin, gridMax, hitPos)) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        outColor = vec4(0.1, 0.1, 0.1, 1.0); // Miss: black color
    }
}
