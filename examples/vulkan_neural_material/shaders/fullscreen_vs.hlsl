struct VSOut {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

VSOut main(uint vid : SV_VertexID)
{
    VSOut o;
    float2 p;
    if (vid == 0) p = float2(-1.0, -1.0);
    else if (vid == 1) p = float2(3.0, -1.0);
    else p = float2(-1.0, 3.0);

    o.pos = float4(p, 0.0, 1.0);
    o.uv = p * 0.5 + 0.5;
    return o;
}
