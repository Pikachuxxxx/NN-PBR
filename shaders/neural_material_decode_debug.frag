#version 450
#extension GL_GOOGLE_include_directive : require
layout(location=0) in vec2 inUV;
layout(location=0) out vec4 outColor;
#include "material_decoder.glsl"
void main()
{
    vec2 dx=dFdx(inUV*vec2(4.0,2.0)), dy=dFdy(inUV*vec2(4.0,2.0));
    // TOP ROW: Show 4 latent textures with normal dynamic mip selection
    if (inUV.y > 0.5) {
        float x = inUV.x * 4.0;
        float y = (inUV.y - 0.5) * 2.0;  // remap [0.5,1.0] -> [0,1]
        vec2 cell_uv = vec2(fract(x), y);  // 0..1 within cell

        vec3 color = vec3(0.0);
        if (x < 1.0) {
            // Latent 0
            color = textureGrad(gLatentTex0,cell_uv,dx,dy).xyz;
        } else if (x < 2.0) {
            // Latent 1
            color = textureGrad(gLatentTex1,cell_uv,dx,dy).xyz;
        } else if (x < 3.0) {
            // Latent 2
            color = textureGrad(gLatentTex2,cell_uv,dx,dy).xyz;
        } else {
            // Latent 3
            color = textureGrad(gLatentTex3,cell_uv,dx,dy).xyz;
        }
        outColor = vec4(color, 1.0);
        return;
    }

    // BOTTOM ROW: Debug outputs in 4 regions
    float x = inUV.x * 4.0;
    int region = int(floor(x));
    vec2 panelUV = vec2(fract(x), inUV.y * 2.0);

    NeuralPBR pbr = DecodeNeuralMaterial(panelUV,dx,dy);

    vec3 display = vec3(0.0);

    if (region == 0) {
        // Albedo
        display = pbr.albedo;

    } else if (region == 1) {
        // Normal (remap [-1,1] -> [0,1])
        display = pbr.normalTS * 0.5 + 0.5;

    } else if (region == 2) {
        // ORM packed for visualization
        display = vec3(pbr.ao, pbr.roughness, pbr.metallic);

    } else {
        // Region 3: empty
        display = vec3(0.0);
    }

outColor = vec4(display, 1.0);
}
