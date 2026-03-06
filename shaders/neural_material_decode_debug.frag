#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

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

struct NeuralPBR
{
    vec3 albedo;
    vec3 normalTS;
    float ao;
    float roughness;
    float metallic;
};

#define N_LATENT 4
#define LATENT_CHANNELS_PER_TEX 3
#define IN_DIM (N_LATENT * LATENT_CHANNELS_PER_TEX)
#define HIDDEN_DIM 16
#define OUT_DIM 8

vec3 SampleLatent(sampler2D tex, vec2 uv, float lodBias)
{
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

    float h[HIDDEN_DIM];
    for (uint j = 0u; j < HIDDEN_DIM; ++j)
    {
        float s = gDecoderWeights.weights[fc1BiasBase + j];
        for (uint i = 0u; i < IN_DIM; ++i)
        {
            float w = gDecoderWeights.weights[fc1WeightBase + j * IN_DIM + i];
            s += w * x[i];
        }
        h[j] = max(s, 0.0);
    }

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
    // Albedo is trained in [0, 1] (physical reflectance), just clamp
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
    vec2 uv = inUV;

    // TOP ROW: Show 4 latent textures (mip0)
    if (inUV.y > 0.5) {
        float x = inUV.x * 4.0;
        float y = (inUV.y - 0.5) * 2.0;  // remap [0.5,1.0] -> [0,1]
        vec2 cell_uv = vec2(fract(x), y);  // 0..1 within cell

        vec3 color = vec3(0.0);
        if (x < 1.0) {
            // Latent 0
            color = SampleLatent(gLatentTex0, cell_uv, 0.0);
        } else if (x < 2.0) {
            // Latent 1
            color = SampleLatent(gLatentTex1, cell_uv, 0.0);
        } else if (x < 3.0) {
            // Latent 2
            color = SampleLatent(gLatentTex2, cell_uv, 0.0);
        } else {
            // Latent 3
            color = SampleLatent(gLatentTex3, cell_uv, 0.0);
        }
        outColor = vec4(color, 1.0);
        return;
    }

    // BOTTOM ROW: Debug outputs in 4 regions
    float x = inUV.x * 4.0;
    int region = int(floor(x));

    NeuralPBR pbr = DecodeNeuralMaterial(uv);

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
