#define N_LATENT 4
#define IN_DIM 12
#define HIDDEN_DIM 16
#define OUT_DIM 8

struct PSIn {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

struct PushData {
    uint debugMode;
};

struct NeuralCB {
    float lodBias0;
    float lodBias1;
    float lodBias2;
    float lodBias3;
};

[[vk::binding(0, 0)]] ByteAddressBuffer gDecoderWeightsFP16;
[[vk::binding(1, 0)]] Texture2D<float4> gLatentTex0;
[[vk::binding(2, 0)]] Texture2D<float4> gLatentTex1;
[[vk::binding(3, 0)]] Texture2D<float4> gLatentTex2;
[[vk::binding(4, 0)]] Texture2D<float4> gLatentTex3;
[[vk::binding(5, 0)]] SamplerState gLinearWrapSampler;
[[vk::binding(6, 0)]] cbuffer NeuralMaterialCB { NeuralCB gCB; };

[[vk::push_constant]] PushData gPush;

float LoadFP16(ByteAddressBuffer src, uint halfIndex)
{
    uint word = src.Load((halfIndex >> 1u) << 2u);
    uint h = ((halfIndex & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16u);
    return f16tof32(h);
}

float3 SampleLatent(Texture2D<float4> tex, float2 uv, float2 duvdx, float2 duvdy, float lodBias)
{
    float s = exp2(lodBias);
    return tex.SampleGrad(gLinearWrapSampler, uv, duvdx * s, duvdy * s).xyz;
}

float4 main(PSIn IN) : SV_Target0
{
    float x[IN_DIM];

    float2 du = ddx(IN.uv);
    float2 dv = ddy(IN.uv);

    float3 l0 = SampleLatent(gLatentTex0, IN.uv, du, dv, gCB.lodBias0);
    float3 l1 = SampleLatent(gLatentTex1, IN.uv, du, dv, gCB.lodBias1);
    float3 l2 = SampleLatent(gLatentTex2, IN.uv, du, dv, gCB.lodBias2);
    float3 l3 = SampleLatent(gLatentTex3, IN.uv, du, dv, gCB.lodBias3);

    x[0] = l0.x;  x[1] = l0.y;  x[2] = l0.z;
    x[3] = l1.x;  x[4] = l1.y;  x[5] = l1.z;
    x[6] = l2.x;  x[7] = l2.y;  x[8] = l2.z;
    x[9] = l3.x;  x[10] = l3.y; x[11] = l3.z;

    const uint fc1WeightBase = 0u;
    const uint fc1BiasBase = fc1WeightBase + (HIDDEN_DIM * IN_DIM);
    const uint fc2WeightBase = fc1BiasBase + HIDDEN_DIM;
    const uint fc2BiasBase = fc2WeightBase + (OUT_DIM * HIDDEN_DIM);

    float h[HIDDEN_DIM];
    [unroll]
    for (uint j = 0; j < HIDDEN_DIM; ++j) {
        float s = LoadFP16(gDecoderWeightsFP16, fc1BiasBase + j);
        [unroll]
        for (uint i = 0; i < IN_DIM; ++i) {
            float w = LoadFP16(gDecoderWeightsFP16, fc1WeightBase + j * IN_DIM + i);
            s += w * x[i];
        }
        h[j] = max(s, 0.0);
    }

    float y[OUT_DIM];
    [unroll]
    for (uint k = 0; k < OUT_DIM; ++k) {
        float s = LoadFP16(gDecoderWeightsFP16, fc2BiasBase + k);
        [unroll]
        for (uint j = 0; j < HIDDEN_DIM; ++j) {
            float w = LoadFP16(gDecoderWeightsFP16, fc2WeightBase + k * HIDDEN_DIM + j);
            s += w * h[j];
        }
        y[k] = s;
    }

    float3 albedo = saturate(float3(y[0], y[1], y[2]) * 0.5 + 0.5);
    float nx = y[3];
    float ny = y[4];
    float nz = sqrt(saturate(1.0 - nx * nx - ny * ny));
    float3 normalTS = normalize(float3(nx, ny, nz));
    float ao = saturate(y[5] * 0.5 + 0.5);
    float roughness = saturate(y[6] * 0.5 + 0.5);
    float metallic = saturate(y[7] * 0.5 + 0.5);

    uint mode = gPush.debugMode;
    if (mode == 1) {
        return float4(normalTS * 0.5 + 0.5, 1.0);
    }
    if (mode == 2) {
        return float4(ao, roughness, metallic, 1.0);
    }
    if (mode == 3) {
        float l = dot(albedo, float3(0.2126, 0.7152, 0.0722));
        return float4(l, l, l, 1.0);
    }
    return float4(albedo, 1.0);
}
