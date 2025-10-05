#version 450

// Input from vertex buffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 projection;
} ubo;


// Output to fragment shader
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;

void main() {
    // Transform vertex position to clip space
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // Pass color and normal to fragment shader
    fragColor = inColor;
    fragNormal = inNormal;
}
