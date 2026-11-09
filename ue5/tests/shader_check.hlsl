#include "NNPBRMaterial.ush"
StructuredBuffer<float4> Input : register(t0);
RWStructuredBuffer<float4> Output : register(u1);
[numthreads(1,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float4 w[86];
    [unroll] for(int i=0;i<86;++i) w[i]=Input[i];
    float3 normal,orm;
    float3 rgb=NNPBRDecode(Input[86].xyz,Input[87].xyz,Input[88].xyz,Input[89].xyz,w,-1,normal,orm);
    Output[0]=float4(rgb,1);Output[1]=float4(normal,1);Output[2]=float4(orm,1);
}
