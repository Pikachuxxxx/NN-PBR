#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D gTexture0;
layout(binding = 1) uniform sampler2D gTexture1;
layout(binding = 2) uniform sampler2D gTexture2;
layout(binding = 3) uniform sampler2D gTexture3;

void main()
{
    // Determine which quadrant we're in based on screen coordinates
    vec2 quadUV = fract(inUV * 2.0);  // 0..1 in current quadrant

    vec3 color;

    if (inUV.x < 0.5 && inUV.y < 0.5) {
        // Bottom-left: Texture 0
        color = texture(gTexture0, quadUV).rgb;
    } else if (inUV.x >= 0.5 && inUV.y < 0.5) {
        // Bottom-right: Texture 1
        color = texture(gTexture1, quadUV).rgb;
    } else if (inUV.x < 0.5 && inUV.y >= 0.5) {
        // Top-left: Texture 2
        color = texture(gTexture2, quadUV).rgb;
    } else {
        // Top-right: Texture 3
        color = texture(gTexture3, quadUV).rgb;
    }

    outColor = vec4(color, 1.0);
}
