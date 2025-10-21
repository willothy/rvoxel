#version 450

// Input from vertex buffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 projection;

    vec3 camera_position;
    vec3 resolution;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} push;

// Output to fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;


vec2 verts(uint i) {
    // 0: (-1,-1), 1: ( 3,-1), 2: (-1, 3)
    const vec2 pos[3] = vec2[](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );
    return pos[i];
}

// void main() {
//     vec2 p = verts(gl_VertexIndex);
//     gl_Position = vec4(p, 0.0, 1.0);
//     // Map NDC -> UV (flip Y if your texture is top-left origin)
//     v_uv = p * 0.5 + 0.5;
// }

void main() {
    vec2 p = verts(gl_VertexIndex);
    gl_Position = vec4(p, 0.0, 1.0);

    // // Transform vertex position to clip space
    // gl_Position = vec4(inPosition, 1.0);

    // Pass color and normal to fragment shader
    // fragColor = inColor;
    // fragNormal = inNormal;
}
