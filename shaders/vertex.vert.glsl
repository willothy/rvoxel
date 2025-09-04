#version 450

// Input from vertex buffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

// Output to fragment shader
layout(location = 0) out vec3 fragColor;

void main() {
    // Transform vertex position to clip space
    gl_Position = vec4(inPosition, 1.0);
    
    // Pass color through to fragment shader
    fragColor = inColor;
}
