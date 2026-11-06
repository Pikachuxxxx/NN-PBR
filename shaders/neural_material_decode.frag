#version 450
#extension GL_GOOGLE_include_directive : require
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;
#include "material_decoder.glsl"
void main()
{
    vec2 dx=dFdx(inUV*2.0), dy=dFdy(inUV*2.0);
    // Divide viewport into 2x2 grid
    // Determine which quadrant we're in based on fragment coordinates
    vec2 quadUV = fract(inUV * 2.0);  // UV within current quadrant (0-1)
    vec2 quadIdx = floor(inUV * 2.0); // Which quadrant (0 or 1 in each dimension)

    // Determine actual sampling UV (scale back to 0-1)
    vec2 sampleUV = quadUV;

    // Decode material at this sample point
    NeuralPBR pbr = DecodeNeuralMaterial(sampleUV,dx,dy);

    // Select output based on quadrant position
    vec4 result;

    if (quadIdx.x < 0.5) {
        // Left side
        if (quadIdx.y < 0.5) {
            // Top-left: Albedo (RGB)
            result = vec4(pbr.albedo, 1.0);
        } else {
            // Bottom-left: AO (grayscale)
            result = vec4(vec3(pbr.ao), 1.0);
        }
    } else {
        // Right side
        if (quadIdx.y < 0.5) {
            // Top-right: Normal in tangent space (RGB, remapped to [0,1])
            vec3 normalDisplay = pbr.normalTS * 0.5 + 0.5;
            result = vec4(normalDisplay, 1.0);
        } else {
            // Bottom-right: Roughness + Metallic as RG, AO as B (ORM-like)
            result = vec4(pbr.ao, pbr.roughness, pbr.metallic, 1.0);
        }
    }

    outColor = result;
}
