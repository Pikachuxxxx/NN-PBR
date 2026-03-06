# Rendering Debug Guide

## Issue: Pink Flashing Output

### Root Cause Analysis

The pink/flashing output indicates one of these problems:

1. **Shader not executing** - Would show solid pink (clear color)
2. **Textures not loaded/bound** - Would show black or noise
3. **Sampler misconfigured** - Would show wrong colors or NaN artifacts
4. **Descriptor binding incorrect** - Would show undefined behavior
5. **Clear color is pink** - Would fill screen with solid magenta

### Debug Visualization Shader

The debug shader (`neural_material_decode_debug.frag`) provides **visual diagnostics** by splitting the viewport:

```
TOP HALF (inUV.y > 0.5):
  [0.00-0.25] | [0.25-0.50] | [0.50-0.75] | [0.75-1.00]
   Latent 0  |   Latent 1  |   Latent 2  |   Latent 3
   (mip0)    |    (mip0)   |    (mip0)   |    (mip0)

BOTTOM HALF (inUV.y < 0.5):
  [0.00-0.33]    |    [0.33-0.67]    |    [0.67-1.00]
    Albedo       |     Normal Map    |   Roughness/Metal/AO
  (decoded R,G,B)|  (visualized X,Y,Z) | (R=rough, G=metal, B=AO)
```

### How to Interpret Results

#### Scenario 1: All Black
**Diagnosis**: Textures not loaded or samplers not bound

**Check**:
- Verify `loadNeuralMaterialAssets()` completes without errors
- Ensure `latentImages[0-3]` have data
- Check sampler is created: `vkCreateSampler()`
- Verify descriptor writes include image views with correct layouts

**Fix**: Run with validation layer to see descriptor binding errors

---

#### Scenario 2: Pink/Magenta Blocks
**Diagnosis**: Textures present but decoded values out of range

**Possible causes**:
- Decoder weights not loaded (`decoder_fp16.bin` missing/corrupted)
- Shader not running (would show solid color instead)
- Neural network weights are NaN
- Output clamping failing

**Check**:
```
✓ decoderBuffer created
✓ decoderMemory mapped
✓ decoder_fp16.bin (688 bytes) loaded
```

---

#### Scenario 3: Noise/Random Colors
**Diagnosis**: Descriptor pointers are valid but data is garbage

**Possible causes**:
- Texture images in UNDEFINED layout (should be SHADER_READ_ONLY_OPTIMAL)
- Descriptor updates not executed
- Wrong mip level being sampled
- Sampler addressing mode issues

**Check**:
- `transitionImageLayout()` called for each texture
- Descriptor pool has correct size
- `vkUpdateDescriptorSets()` called successfully
- Sampler created with CLAMP_TO_EDGE

---

#### Scenario 4: Black on Top, Pink on Bottom
**Diagnosis**: Textures load but decoder fails

**Possible causes**:
- Decoder weight buffer not bound
- MLP computation has NaN values
- Output range incorrect (should be -1..1 then clamped to 0..1)

**Check**:
- `decoderBuffer` VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
- `gDecoderWeights` binding 0 is storage buffer
- Weight clamping at output: `clamp(..., 0.0, 1.0)`

---

#### Scenario 5: Flashing/Changing
**Diagnosis**: Frame synchronization or command buffer recording issue

**Possible causes**:
- Command buffer not recorded correctly
- Viewport/scissor changed between frames
- Descriptor set corrupted on next frame
- Fence not waited on

**Check**:
- Triple buffering working (MAX_FRAMES_IN_FLIGHT = 3)
- Fence wait before acquire
- Command buffer reset before recording
- Same descriptors used for all frames

---

## Smart Rendering Approach

Instead of multiple draw calls, the debug shader uses **UV space splitting**:

```glsl
// For latent textures (top row)
float x = inUV.x * 4.0;              // Divide into 4 cells
vec2 cell_uv = vec2(fract(x), inUV.y);  // Convert to 0..1 in cell

if (x < 1.0) color = texture(gLatentTex0, cell_uv);
else if (x < 2.0) color = texture(gLatentTex1, cell_uv);
// ... etc
```

**Benefits**:
- ✅ Single fragment shader
- ✅ No extra draw calls
- ✅ One descriptor set
- ✅ Efficient visualization
- ✅ No descriptor setup overhead

---

## Production Rendering Path

Once debug validation passes, switch back to `neural_material_decode.frag`:

```cpp
// In createPipeline()
std::vector<char> fragShaderCode = loadShaderFile("neural_material_decode.frag.spv");
// NOT: loadShaderFile("neural_material_decode_debug.frag.spv");
```

The production shader:
- Runs full MLP decoder
- Outputs 8 material channels (albedo, normal, AO, roughness, metallic)
- Single full-screen quad render
- Efficient for real-time use

---

## Diagnostic Checklist

Run through these checks if you see incorrect rendering:

- [ ] Clear color is not pink (check `VkClearValue`)
- [ ] Latent textures visible in top row (4 blocks)
- [ ] Colors are sensible ranges (not pure pink/black noise)
- [ ] Albedo section shows plausible colors
- [ ] Normal section shows ~0.5 gray (tangent space neutral)
- [ ] Roughness/Metal/AO show varied values
- [ ] No flashing/corruption between frames
- [ ] No validation errors in console

---

## Expected Debug Output

With proper setup, you should see:

**Top Row**:
- Latent 0: Reddish-tan colors (base material features)
- Latent 1-3: Greenish/blueish (higher-level features)

**Bottom Row**:
- Albedo: Orange/red fabric texture
- Normal: Grayish with texture detail
- Roughness/Metal/AO: Mixed values indicating material properties

---

## Next Steps

1. **Run windowed mode** on a display-equipped system
2. **Verify debug output** shows correct texture sampling
3. **Check decoder execution** by examining bottom row colors
4. **Validate frame sync** - no flashing or corruption
5. **Switch to production shader** for final rendering
6. **Profile performance** - should sustain 60+ FPS

---

**Status**: Debug shader ready for visualization and diagnostics
**Next**: Use on actual display to validate rendering pipeline
