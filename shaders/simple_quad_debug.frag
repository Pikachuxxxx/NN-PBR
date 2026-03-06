#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Textures are at bindings 1-4 (binding 0 is decoder weights)
layout(binding = 1) uniform sampler2D gTexture0;
layout(binding = 2) uniform sampler2D gTexture1;
layout(binding = 3) uniform sampler2D gTexture2;
layout(binding = 4) uniform sampler2D gTexture3;

void main()
{
    // First, let's output the UVs as a debug pattern
    // If we see this gradient, the shader is running
    outColor = vec4(inUV.x, inUV.y, 0.5, 1.0);

    // Determine which quadrant we're in based on screen coordinates
    float x = inUV.x * 2.0;
    float y = inUV.y * 2.0;
    vec2 quadUV = vec2(fract(x), fract(y));

    vec3 color = vec3(0.0);

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

    // If textures are black, output will still show gradient debug pattern
    // If textures have data, it will override the gradient
    if (length(color) > 0.0) {
        outColor = vec4(color, 1.0);
    }
}
