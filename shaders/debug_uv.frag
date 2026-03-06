#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main()
{
    // Output UV coordinates as RGB colors
    // U (red) ranges 0-1 left to right
    // V (green) ranges 0-1 bottom to top
    // B is constant
    outColor = vec4(inUV.x, inUV.y, 0.5, 1.0);
}
