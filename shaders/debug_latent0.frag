#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Latent texture 0 (BC6H format)
layout(binding = 1) uniform sampler2D gLatentTex0;

void main()
{
    // Simple display: just show latent 0 sampled at mip 0
    vec3 latent = textureLod(gLatentTex0, inUV, 0.0).xyz;

    // Clamp to [0, 1] for visualization (latents are [-1, 1] so remap)
    latent = latent * 0.5 + 0.5;
    latent = clamp(latent, 0.0, 1.0);

    outColor = vec4(latent, 1.0);
}
