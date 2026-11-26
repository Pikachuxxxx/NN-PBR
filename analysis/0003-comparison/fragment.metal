#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

struct Decoder
{
    uint4 _packed[21];
};

struct main0_out
{
    float4 albedoNormalX [[color(0)]];
    float4 normalYAndORM [[color(1)]];
};

struct main0_in
{
    float2 uv [[user(locn0)]];
};

static inline __attribute__((always_inline))
float2 parameterPair(thread const int& halfIndex, constant Decoder& _29)
{
    int word = halfIndex >> 1;
    float2 _42 = float2(as_type<half2>(_29._packed[word >> 2][word & 3]));
    return _42;
}

static inline __attribute__((always_inline))
float4 parameters4(thread const int& index, constant Decoder& _29)
{
    int param = index;
    int param_1 = index + 2;
    return float4(parameterPair(param, _29), parameterPair(param_1, _29));
}

fragment main0_out main0(main0_in in [[stage_in]], constant Decoder& _29 [[buffer(0)]], texture2d<float> latent [[texture(0)]], sampler latentSmplr [[sampler(0)]])
{
    main0_out out = {};
    float4 x = (latent.sample(latentSmplr, in.uv, gradient2d(dfdx(in.uv) * 1.0, dfdy(in.uv) * 1.0)) * 2.0) - float4(1.0);
    spvUnsafeArray<float4, 3> h;
    for (int j = 0; j < 3; j++)
    {
        int offset = 16 * j;
        int param = offset;
        int param_1 = offset + 4;
        int param_2 = offset + 8;
        int param_3 = offset + 12;
        float4 sum = float4(dot(parameters4(param, _29), x), dot(parameters4(param_1, _29), x), dot(parameters4(param_2, _29), x), dot(parameters4(param_3, _29), x));
        int param_4 = 48 + (4 * j);
        h[j] = fast::max(sum + parameters4(param_4, _29), float4(0.0));
    }
    spvUnsafeArray<float4, 2> y;
    for (int group = 0; group < 2; group++)
    {
        int param_5 = 156 + (4 * group);
        float4 value = parameters4(param_5, _29);
        for (int lane = 0; lane < 4; lane++)
        {
            int row = 60 + (12 * ((4 * group) + lane));
            int param_6 = row;
            int param_7 = row + 4;
            int param_8 = row + 8;
            value[lane] += ((dot(parameters4(param_6, _29), h[0]) + dot(parameters4(param_7, _29), h[1])) + dot(parameters4(param_8, _29), h[2]));
        }
        y[group] = value;
    }
    out.albedoNormalX = y[0];
    out.normalYAndORM = y[1];
    return out;
}

