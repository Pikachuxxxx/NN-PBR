#include "NNPBRFormat.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <string>

static void Check(VkResult r)
{
    if (r != VK_SUCCESS)
        throw std::runtime_error("Vulkan error " + std::to_string(r));
}

struct Buffer
{
    VkBuffer handle;
    VkDeviceMemory memory;
    void *mapped;
};

static Buffer Allocate(VkDevice dev, VkPhysicalDevice gpu, VkDeviceSize size)
{
    VkBufferCreateInfo desc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    Buffer result{};
    Check(vkCreateBuffer(dev, &desc, nullptr, &result.handle));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, result.handle, &req);
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(gpu, &props);
    const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = 0;
    for (; type < props.memoryTypeCount; ++type)
        if ((req.memoryTypeBits & (1u << type)) && (props.memoryTypes[type].propertyFlags & flags) == flags)
            break;
    if (type == props.memoryTypeCount)
        throw std::runtime_error("No host-visible coherent storage memory");
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    Check(vkAllocateMemory(dev, &alloc, nullptr, &result.memory));
    Check(vkBindBufferMemory(dev, result.handle, result.memory, 0));
    Check(vkMapMemory(dev, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped));
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 3)
        return 2;
    VkInstanceCreateInfo ic{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    VkInstance inst;
    Check(vkCreateInstance(&ic, nullptr, &inst));
    uint32_t count = 0;
    Check(vkEnumeratePhysicalDevices(inst, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    Check(vkEnumeratePhysicalDevices(inst, &count, devices.data()));
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    uint32_t q = 0;
    for (auto dev : devices)
    {
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
        for (q = 0; q < count; ++q)
            if (families[q].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                gpu = dev;
                break;
            }
        if (gpu)
            break;
    }
    if (!gpu)
        return 77;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    std::cout << props.deviceName << std::endl;
    const float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = q;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    VkDevice dev;
    Check(vkCreateDevice(gpu, &di, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, q, 0, &queue);
    auto in = Allocate(dev, gpu, 90 * 16);
    auto out = Allocate(dev, gpu, 3 * 16);
    auto floats = static_cast<float *>(in.mapped);
    std::ifstream bin(argv[2], std::ios::binary);
    std::vector<uint8_t> blob{std::istreambuf_iterator<char>(bin), std::istreambuf_iterator<char>()};
    std::array<float, 344> weights;
    if (!NNPBR::DecodeWeights(blob.data(), blob.size(), weights))
        return 3;
    std::memcpy(floats, weights.data(), sizeof(weights));
    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo ld{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ld.bindingCount = 2;
    ld.pBindings = bindings;
    VkDescriptorSetLayout layout;
    Check(vkCreateDescriptorSetLayout(dev, &ld, nullptr, &layout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pd{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pd.maxSets = 1;
    pd.poolSizeCount = 1;
    pd.pPoolSizes = &ps;
    VkDescriptorPool pool;
    Check(vkCreateDescriptorPool(dev, &pd, nullptr, &pool));
    VkDescriptorSetAllocateInfo sa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sa.descriptorPool = pool;
    sa.descriptorSetCount = 1;
    sa.pSetLayouts = &layout;
    VkDescriptorSet set;
    Check(vkAllocateDescriptorSets(dev, &sa, &set));
    VkDescriptorBufferInfo bi[2] = {{in.handle, 0, VK_WHOLE_SIZE}, {out.handle, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 2; ++i)
    {
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = i;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bi[i];
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    }
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1;
    pi.pSetLayouts = &layout;
    VkPipelineLayout pl;
    Check(vkCreatePipelineLayout(dev, &pi, nullptr, &pl));
    std::ifstream shader(argv[1], std::ios::binary | std::ios::ate);
    const auto length = shader.tellg();
    if (length <= 0 || size_t(length) % 4)
        return 4;
    std::vector<uint32_t> spirv(size_t(length) / 4);
    shader.seekg(0);
    shader.read(reinterpret_cast<char *>(spirv.data()), length);
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = length;
    si.pCode = spirv.data();
    VkShaderModule sm;
    Check(vkCreateShaderModule(dev, &si, nullptr, &sm));
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.layout = pl;
    cp.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = sm;
    cp.stage.pName = "main";
    VkPipeline pipe;
    Check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, nullptr, &pipe));
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = q;
    VkCommandPool commands;
    Check(vkCreateCommandPool(dev, &pci, nullptr, &commands));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commands;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    Check(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    Check(vkBeginCommandBuffer(cb, &begin));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, nullptr);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
    Check(vkEndCommandBuffer(cb));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    float maxError = 0;
    for (int sample = 0; sample < 64; ++sample)
    {
        float x[12], h[16], y[8];
        for (int i = 0; i < 12; ++i)
            floats[344 + (i / 3) * 4 + i % 3] = x[i] = float((sample * 17 + i * 7) % 43) / 43.f;
        for (int j = 0; j < 16; ++j)
        {
            float sum = weights[192 + j];
            for (int i = 0; i < 12; ++i)
                sum += weights[j * 12 + i] * x[i];
            h[j] = std::max(0.f, sum);
        }
        for (int j = 0; j < 8; ++j)
        {
            float sum = weights[336 + j];
            for (int i = 0; i < 16; ++i)
                sum += weights[208 + j * 16 + i] * h[i];
            y[j] = sum;
        }
        float z = std::sqrt(std::max(0.f, 1 - y[3] * y[3] - y[4] * y[4]));
        float n = std::sqrt(y[3] * y[3] + y[4] * y[4] + z * z);
        float expected[12] = {0, 0, 0, 1, y[3] / n, -y[4] / n, z / n, 1, 0, 0, 0, 1};
        for (int i = 0; i < 3; ++i)
        {
            float s = std::clamp(y[i], 0.f, 1.f);
            expected[i] = s <= .04045f ? s / 12.92f : std::pow((s + .055f) / 1.055f, 2.4f);
            expected[i + 8] = std::clamp(.5f * y[i + 5] + .5f, 0.f, 1.f);
        }
        Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        Check(vkQueueWaitIdle(queue));
        for (int i = 0; i < 12; ++i)
        {
            float error = std::abs(static_cast<float *>(out.mapped)[i] - expected[i]);
            if (!std::isfinite(error))
                return 5;
            maxError = std::max(maxError, error);
        }
    }
    std::cout << "Max error: " << maxError << std::endl;
    vkDestroyCommandPool(dev, commands, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyDescriptorSetLayout(dev, layout, nullptr);
    for (auto b : {in, out})
    {
        vkUnmapMemory(dev, b.memory);
        vkDestroyBuffer(dev, b.handle, nullptr);
        vkFreeMemory(dev, b.memory, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return maxError < 2e-5f ? 0 : 6;
}
