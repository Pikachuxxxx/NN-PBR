# Vulkan Neural Material Demo

A real-time neural material decoder using Vulkan 1.3 with dynamic rendering.

## Features

- **Loads DDS BC6H latent textures** directly from export artifacts
- **Decodes neural material MLPs** in-shader using FP16 weights
- **2×2 quad layout** showing 3 predicted channels (albedo, normal, orm)
- **Vulkan 1.3 dynamic rendering** (no renderpass overhead)
- **LOD bias sampling** matching training configuration

## Building

### Prerequisites

- **Vulkan SDK 1.3+** (with glslc compiler)
- **CMake 3.16+**
- **GLFW3**
- **GLM**

### macOS (Homebrew)

```bash
brew install vulkan-headers molten-vk glfw3 glm cmake
```

### Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Run

```bash
./vk_neural_demo
```

The demo loads exported artifacts from `/Users/phanisrikar/Desktop/Projects/NN-PBR/runs/iter65k/export`.

## Architecture

### File Layout

- `vk_neural_material_demo.cpp` — Main Vulkan app
- `shaders/neural_material_decode.vert` — Vertex shader (quad positions)
- `shaders/neural_material_decode.frag` — Fragment shader (neural decoder)
- `CMakeLists.txt` — Build configuration

### Shader Design

The fragment shader:
1. **Samples 4 latent textures** (BC6H) with LOD bias
2. **Concatenates** into 12D latent vector
3. **Evaluates 2-layer MLP** (12 → 16 → 8)
4. **Converts output** to standard PBR ranges

```
input: [L0.rgb, L1.rgb, L2.rgb, L3.rgb]  (12D)
  ↓ (fc1.weight @ 192 FP16 params + fc1.bias @ 16)
hidden: 16D with ReLU
  ↓ (fc2.weight @ 128 FP16 params + fc2.bias @ 8)
output: [Albedo.rgb, Normal.xy, AO, Roughness, Metallic]  (8D)
```

### Layout Overview

- **2×2 grid of quads** (0-3) at fixed viewport coordinates
- Each quad shows the decoded material
- Textures are BC6H UF16 (4 mips per pyramid)

## Known Limitations

- **Simplified render** (no perspective correction, static quad layout)
- **Single material** (hardcoded iter65k export)
- **No frame timing** or perf metrics
- **Placeholder command buffer** (draw logic not fully implemented)

## TODO

- [ ] Implement full dynamic rendering command buffer recording
- [ ] Add rotation/animation to quads
- [ ] Toggle between albedo/normal/orm display modes
- [ ] Real-time MLP weight reloading
- [ ] Performance metrics overlay

## License

Part of NN-PBR research implementation.
