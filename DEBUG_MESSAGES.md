# Vulkan Validation Layer Debug Messages

## Summary
Enabled Vulkan validation layer (VK_LAYER_KHRONOS_validation) with debug utils extension for detailed error reporting.

## Errors Found:

### 1. **Viewport Width = 0** ⚠️
```
vkCreateGraphicsPipelines(): pCreateInfos[0].pViewportState->pViewports[0].width (0.000000) is not greater than zero.
```
**Cause**: In headless mode, swapchain is not created, so `swapChainExtent.width` is never set.
**Fix**: Use fixed viewport dimensions when swapchain extent is not available.

### 2. **Descriptor Type Mismatch** ⚠️
```
vkCreateGraphicsPipelines(): pCreateInfos[0].pStages[1] SPIR-V uses descriptor [Set 0, Binding 1-4]
of type VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE but expected VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER.
```
**Cause**: Shader expects combined image samplers (texture + sampler), but descriptors are separated.
**Fix**: Change descriptor layout bindings 1-4 from `SAMPLED_IMAGE` to `COMBINED_IMAGE_SAMPLER`.

### 3. **Missing Vertex Input Attributes** ⚠️
```
vkCreateGraphicsPipelines(): pVertexInputState->pVertexAttributeDescriptions does not have a Location 0/1
but vertex shader has an input variable at that Location.
```
**Cause**: Pipeline vertex input state is empty, but vertex shader expects position (location 0) and UV (location 1).
**Fix**: Add `VkVertexInputAttributeDescription` array with proper attribute bindings.

### 4. **Shader Module Leak** ⚠️
```
vkDestroyDevice(): For VkDevice 0x932c38018, VkShaderModule 0x170000000017 has not been destroyed.
vkDestroyDevice(): For VkDevice 0x932c38018, VkShaderModule 0x180000000018 has not been destroyed.
```
**Cause**: Framebuffer creation failed, so shader modules weren't destroyed properly.
**Fix**: Ensure shader modules are destroyed even if framebuffer creation fails.

### 5. **Device Extension Warnings** ℹ️
```
vkCreateDevice(): pCreateInfo->ppEnabledExtensionNames[0] Missing extension required by the device extension VK_KHR_swapchain: VK_KHR_surface.
vkCreateDevice(): VK_KHR_portability_subset must be enabled because physical device supports it.
```
**Note**: These are warnings on macOS with MoltenVK - not critical but should be addressed.

## Changes Made:

✅ Enabled VK_EXT_DEBUG_UTILS extension
✅ Added VK_LAYER_KHRONOS_validation validation layer
✅ Created debug messenger with callback for error reporting
✅ Changed latent texture descriptors to COMBINED_IMAGE_SAMPLER (in progress)

## Next Steps:

1. Set fixed viewport dimensions in headless mode
2. Add vertex input attribute descriptions to pipeline
3. Properly handle shader module cleanup
4. Update descriptor pool pool sizes to match new layout
5. Update descriptor set updates to bind combined samplers
