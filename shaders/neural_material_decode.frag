#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

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
    vec3 albedo;
    vec3 normalTS;
    float ao;
    float roughness;
    float metallic;
};

// Decoder weights (FP16 as floats)
layout(std140, binding = 0) readonly buffer DecoderWeights {
    float weights[];
} gDecoderWeights;

// Latent textures (BC6H format)
layout(binding = 1) uniform sampler2D gLatentTex0;
layout(binding = 2) uniform sampler2D gLatentTex1;
layout(binding = 3) uniform sampler2D gLatentTex2;
layout(binding = 4) uniform sampler2D gLatentTex3;

// LOD biases
layout(std140, binding = 5) uniform NeuralMaterialCB {
    float gLodBias0;
    float gLodBias1;
    float gLodBias2;
    float gLodBias3;
};

vec3 SampleLatent(sampler2D tex, vec2 uv, float lodBias)
{
    // Convert LOD bias to mip level
    float mip = -lodBias;
    return textureLod(tex, uv, mip).xyz;
}

NeuralPBR DecodeNeuralMaterial(vec2 uv)
{
    float x[IN_DIM];

    vec3 l0 = SampleLatent(gLatentTex0, uv, gLodBias0);
    vec3 l1 = SampleLatent(gLatentTex1, uv, gLodBias1);
    vec3 l2 = SampleLatent(gLatentTex2, uv, gLodBias2);
    vec3 l3 = SampleLatent(gLatentTex3, uv, gLodBias3);

    x[0]  = l0.x; x[1]  = l0.y; x[2]  = l0.z;
    x[3]  = l1.x; x[4]  = l1.y; x[5]  = l1.z;
    x[6]  = l2.x; x[7]  = l2.y; x[8]  = l2.z;
    x[9]  = l3.x; x[10] = l3.y; x[11] = l3.z;

    const uint fc1WeightBase = 0u;
    const uint fc1BiasBase   = fc1WeightBase + (HIDDEN_DIM * IN_DIM);
    const uint fc2WeightBase = fc1BiasBase + HIDDEN_DIM;
    const uint fc2BiasBase   = fc2WeightBase + (OUT_DIM * HIDDEN_DIM);

    // First layer (12 -> 16)
    float h[HIDDEN_DIM];
    for (uint j = 0u; j < HIDDEN_DIM; ++j)
    {
        float s = gDecoderWeights.weights[fc1BiasBase + j];
        for (uint i = 0u; i < IN_DIM; ++i)
        {
            float w = gDecoderWeights.weights[fc1WeightBase + j * IN_DIM + i];
            s += w * x[i];
        }
        h[j] = max(s, 0.0); // ReLU
    }

    // Second layer (16 -> 8)
    float y[OUT_DIM];
    for (uint k = 0u; k < OUT_DIM; ++k)
    {
        float s = gDecoderWeights.weights[fc2BiasBase + k];
        for (uint j = 0u; j < HIDDEN_DIM; ++j)
        {
            float w = gDecoderWeights.weights[fc2WeightBase + k * HIDDEN_DIM + j];
            s += w * h[j];
        }
        y[k] = s;
    }

    NeuralPBR o;
    // Convert [-1, 1] to [0, 1]
    o.albedo = clamp(vec3(y[0], y[1], y[2]) * 0.5 + 0.5, 0.0, 1.0);

    float nx = y[3];
    float ny = y[4];
    float nz = sqrt(clamp(1.0 - nx * nx - ny * ny, 0.0, 1.0));
    o.normalTS = normalize(vec3(nx, ny, nz));

    o.ao        = clamp(y[5] * 0.5 + 0.5, 0.0, 1.0);
    o.roughness = clamp(y[6] * 0.5 + 0.5, 0.0, 1.0);
    o.metallic  = clamp(y[7] * 0.5 + 0.5, 0.0, 1.0);
    return o;
}

void main()
{
    NeuralPBR pbr = DecodeNeuralMaterial(inUV);

    // Output albedo for now (you can switch to normal or orm)
    outColor = vec4(pbr.albedo, 1.0);
}
