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

// Decoder weights uploaded as tightly packed FP32 values.
layout(std430, binding = 0) readonly buffer DecoderWeights {
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

vec3 SampleLatent(sampler2D tex, vec2 uv, vec2 duvdx, vec2 duvdy, float lodBias)
{
    float s = exp2(lodBias);
    return textureGrad(tex, uv, duvdx * s, duvdy * s).xyz;
}

NeuralPBR DecodeNeuralMaterial(vec2 uv)
{
    float x[IN_DIM];
    vec2 duvdx = dFdx(uv);
    vec2 duvdy = dFdy(uv);

    vec3 l0 = SampleLatent(gLatentTex0, uv, duvdx, duvdy, gLodBias0);
    vec3 l1 = SampleLatent(gLatentTex1, uv, duvdx, duvdy, gLodBias1);
    vec3 l2 = SampleLatent(gLatentTex2, uv, duvdx, duvdy, gLodBias2);
    vec3 l3 = SampleLatent(gLatentTex3, uv, duvdx, duvdy, gLodBias3);

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
    o.albedo = clamp(vec3(y[0], y[1], y[2]), 0.0, 1.0);

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
    // Divide viewport into 2x2 grid
    // Determine which quadrant we're in based on fragment coordinates
    vec2 quadUV = fract(inUV * 2.0);  // UV within current quadrant (0-1)
    vec2 quadIdx = floor(inUV * 2.0); // Which quadrant (0 or 1 in each dimension)

    // Determine actual sampling UV (scale back to 0-1)
    vec2 sampleUV = quadUV;

    // Decode material at this sample point
    NeuralPBR pbr = DecodeNeuralMaterial(sampleUV);

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
            result = vec4(pbr.roughness, pbr.metallic, pbr.ao, 1.0);
        }
    }

    outColor = result;
}
