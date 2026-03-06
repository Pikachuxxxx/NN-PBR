// Drop-in neural material decode shader.
// - Samples BC-compressed latent textures with hardware filtering.
// - Runs exported MLP weights from decoder_fp16.bin (12 -> 16 -> 8 by default).
// - Outputs standard PBR parameters.

#ifndef N_LATENT
#define N_LATENT 4
#endif

#ifndef LATENT_CHANNELS_PER_TEX
#define LATENT_CHANNELS_PER_TEX 3
#endif

#ifndef IN_DIM
#define IN_DIM (N_LATENT * LATENT_CHANNELS_PER_TEX)
#endif

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 16
#endif

#ifndef OUT_DIM
#define OUT_DIM 8
#endif

struct NeuralPBR
{
    float3 albedo;     // [0,1]
    float3 normalTS;   // tangent-space normal
    float  ao;         // [0,1]
    float  roughness;  // [0,1]
    float  metallic;   // [0,1]
};

// Layout follows exported decoder_fp16.bin:
// [fc1.weight (HIDDEN_DIM * IN_DIM)]
// [fc1.bias   (HIDDEN_DIM)]
// [fc2.weight (OUT_DIM * HIDDEN_DIM)]
// [fc2.bias   (OUT_DIM)]
ByteAddressBuffer gDecoderWeightsFP16 : register(t0);

Texture2D<float3> gLatentTex0 : register(t1);
Texture2D<float3> gLatentTex1 : register(t2);
Texture2D<float3> gLatentTex2 : register(t3);
Texture2D<float3> gLatentTex3 : register(t4);

SamplerState gLinearWrapSampler : register(s0);

cbuffer NeuralMaterialCB : register(b0)
{
    // From training export metadata "lod_biases"
    float gLodBias0;
    float gLodBias1;
    float gLodBias2;
    float gLodBias3;
};

float LoadFP16(ByteAddressBuffer src, uint halfIndex)
{
    uint word = src.Load((halfIndex >> 1u) << 2u);
    uint h = ((halfIndex & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16u);
    return f16tof32(h);
}

float3 SampleLatent(Texture2D<float3> tex, float2 uv, float2 duvdx, float2 duvdy, float lodBias)
{
    // Equivalent to adding LOD bias by scaling derivatives.
    float s = exp2(lodBias);
    return tex.SampleGrad(gLinearWrapSampler, uv, duvdx * s, duvdy * s).xyz;
}

NeuralPBR DecodeNeuralMaterialWithGrad(float2 uv, float2 duvdx, float2 duvdy)
{
    float x[IN_DIM];

    float3 l0 = SampleLatent(gLatentTex0, uv, duvdx, duvdy, gLodBias0);
    float3 l1 = SampleLatent(gLatentTex1, uv, duvdx, duvdy, gLodBias1);
    float3 l2 = SampleLatent(gLatentTex2, uv, duvdx, duvdy, gLodBias2);
    float3 l3 = SampleLatent(gLatentTex3, uv, duvdx, duvdy, gLodBias3);

    x[0]  = l0.x; x[1]  = l0.y; x[2]  = l0.z;
    x[3]  = l1.x; x[4]  = l1.y; x[5]  = l1.z;
    x[6]  = l2.x; x[7]  = l2.y; x[8]  = l2.z;
    x[9]  = l3.x; x[10] = l3.y; x[11] = l3.z;

    const uint fc1WeightBase = 0u;
    const uint fc1BiasBase   = fc1WeightBase + (HIDDEN_DIM * IN_DIM);
    const uint fc2WeightBase = fc1BiasBase + HIDDEN_DIM;
    const uint fc2BiasBase   = fc2WeightBase + (OUT_DIM * HIDDEN_DIM);

    float h[HIDDEN_DIM];
    [unroll]
    for (uint j = 0u; j < HIDDEN_DIM; ++j)
    {
        float s = LoadFP16(gDecoderWeightsFP16, fc1BiasBase + j);
        [unroll]
        for (uint i = 0u; i < IN_DIM; ++i)
        {
            float w = LoadFP16(gDecoderWeightsFP16, fc1WeightBase + j * IN_DIM + i);
            s += w * x[i];
        }
        h[j] = max(s, 0.0);
    }

    float y[OUT_DIM];
    [unroll]
    for (uint k = 0u; k < OUT_DIM; ++k)
    {
        float s = LoadFP16(gDecoderWeightsFP16, fc2BiasBase + k);
        [unroll]
        for (uint j = 0u; j < HIDDEN_DIM; ++j)
        {
            float w = LoadFP16(gDecoderWeightsFP16, fc2WeightBase + k * HIDDEN_DIM + j);
            s += w * h[j];
        }
        y[k] = s;
    }

    NeuralPBR o;
    // Albedo is trained in [0,1] range (physical reflectance), just saturate.
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
