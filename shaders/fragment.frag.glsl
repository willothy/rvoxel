#version 450

// Input from vertex shader (interpolated)
layout(location = 0) in vec3 fragColor;

// Output color
layout(location = 0) out vec4 outColor;

void main() {
    // Just use the interpolated color
    outColor = vec4(fragColor, 1.0);
}
