// Generated material usage (pixel shader):
// Bind latent_00.bc6.dds through latent_03.bc6.dds, in order, to t1..t4.
// Keep all mip levels, use unsigned BC6H and a clamp-to-edge trilinear sampler at s0.
// Include the generated material.hlsl, then call:
//   NeuralPBR p = DecodeNeuralMaterial(input.uv);
// For tiled/remapped UVs, take derivatives before frac() and pass them to
// DecodeNeuralMaterialWithGrad. Texture dimensions already account for mip bias.
// Define NNPBR_VIEWER to compile the Vulkan sample's four-panel fragment entry point.

#ifndef NNPBR_MATERIAL_INCLUDED
#define NNPBR_MATERIAL_INCLUDED
struct NeuralPBR {
    float3 albedo;
    float3 normalTS;
    float ao, roughness, metallic;
};

Texture2D<float4> latent0 : register(t1);
Texture2D<float4> latent1 : register(t2);
Texture2D<float4> latent2 : register(t3);
Texture2D<float4> latent3 : register(t4);
SamplerState linearClamp : register(s0);

// The exporter inserts exact FP16 values promoted to float, in PyTorch row order.
static const float weights[344] = {
    /* NNPBR_WEIGHTS */
};

NeuralPBR DecodeNeuralMaterialWithGrad(float2 uv, float2 dx, float2 dy)
{
    float3 a = latent0.SampleGrad(linearClamp, uv, dx, dy).rgb;
    float3 b = latent1.SampleGrad(linearClamp, uv, dx, dy).rgb;
    float3 c = latent2.SampleGrad(linearClamp, uv, dx, dy).rgb;
    float3 d = latent3.SampleGrad(linearClamp, uv, dx, dy).rgb;
    float x[12] = {a.x,a.y,a.z, b.x,b.y,b.z, c.x,c.y,c.z, d.x,d.y,d.z};

    const uint fc1WeightBase = 0u;
    const uint fc1BiasBase   = fc1WeightBase + (16 * 12);
    const uint fc2WeightBase = fc1BiasBase + 16;
    const uint fc2BiasBase   = fc2WeightBase + (8 * 16);

    float h[16];
    [unroll]
    for (uint j = 0u; j < 16; ++j)
    {
        float s = weights[fc1BiasBase + j];
        [unroll]
        for (uint i = 0u; i < 12; ++i)
        {
            float w = weights[fc1WeightBase + j * 12 + i];
            s += w * x[i];
        }
        h[j] = max(s, 0.0);
    }

    float y[8];
    [unroll]
    for (uint k = 0u; k < 8; ++k)
    {
        float s = weights[fc2BiasBase + k];
        [unroll]
        for (uint j = 0u; j < 16; ++j)
        {
            float w = weights[fc2WeightBase + k * 16 + j];
            s += w * h[j];
        }
        y[k] = s;
    }

    NeuralPBR o;
    // Reconstructed albedo is sRGB encoded. Convert to linear before lighting.
    // Other outputs (normal, ORM) are in [-1,1] and need the 0.5 + 0.5 conversion.
    o.albedo = saturate(float3(y[0], y[1], y[2]));

    float nx = y[3];
    float ny = y[4];
    float nz = sqrt(saturate(1.0 - nx * nx - ny * ny));
    o.normalTS = normalize(float3(nx, ny, nz));

    o.ao        = saturate(y[5] * 0.5 + 0.5);
    o.roughness = saturate(y[6] * 0.5 + 0.5);
    o.metallic  = saturate(y[7] * 0.5 + 0.5);
    return o;
}

NeuralPBR DecodeNeuralMaterial(float2 uv)
{
    return DecodeNeuralMaterialWithGrad(uv, ddx(uv), ddy(uv));
}

#ifdef NNPBR_VIEWER
float4 main(float2 uv : TEXCOORD0) : SV_Target
{
    float2 tiled = uv * 2.0;
    NeuralPBR p = DecodeNeuralMaterialWithGrad(frac(tiled), ddx(tiled), ddy(tiled));
    if (tiled.x < 1.0)
        return float4(tiled.y < 1.0 ? p.albedo : p.ao.xxx, 1.0);
    return float4(tiled.y < 1.0 ? p.normalTS * 0.5 + 0.5 : float3(p.ao, p.roughness, p.metallic), 1.0);
}
#endif
#endif
