#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;

// Output color
layout(location = 0) out vec4 outColor;

void main() {
    // Simple lighting: use normal to modulate brightness
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));  // Light from top-right
    float brightness = max(dot(normalize(fragNormal), lightDir), 0.2);  // Min 0.2 (ambient)
    
    outColor = vec4(fragColor * brightness, 1.0);
}
