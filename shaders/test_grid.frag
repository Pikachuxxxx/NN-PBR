#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

void main()
{
    // Divide viewport into 2x2 grid based on UV coordinates
    vec2 quadUV = fract(inUV * 2.0);  // UV within current quadrant (0-1)
    vec2 quadIdx = floor(inUV * 2.0); // Which quadrant (0 or 1 in each dimension)

    vec4 result;

    if (quadIdx.x < 0.5) {
        // Left side
        if (quadIdx.y < 0.5) {
            // Top-left: Red
            result = vec4(1.0, 0.0, 0.0, 1.0);
        } else {
            // Bottom-left: Green
            result = vec4(0.0, 1.0, 0.0, 1.0);
        }
    } else {
        // Right side
        if (quadIdx.y < 0.5) {
            // Top-right: Blue
            result = vec4(0.0, 0.0, 1.0, 1.0);
        } else {
            // Bottom-right: Yellow
            result = vec4(1.0, 1.0, 0.0, 1.0);
        }
    }

    // Draw grid lines at boundaries
    float gridWidth = 0.01;
    if (abs(quadUV.x - 0.5) < gridWidth || abs(quadUV.y - 0.5) < gridWidth) {
        result = vec4(1.0, 1.0, 1.0, 1.0); // White grid lines
    }

    outColor = result;
}
