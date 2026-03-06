#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cstring>
#include <memory>
#include <map>

// Simple math utilities
struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
};

// DDS header parsing for BC6H textures
struct DDSHeader {
    uint32_t magic;
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitchOrLinearSize;
    uint32_t depth;
    uint32_t mipMapCount;
    uint32_t reserved[11];
};

struct DDSPixelFormat {
    uint32_t size;
    uint32_t flags;
    uint32_t fourCC;
    uint32_t rgbBitCount;
    uint32_t rBitMask;
    uint32_t gBitMask;
    uint32_t bBitMask;
    uint32_t aBitMask;
};

struct DDSFile {
    std::vector<uint8_t> data;
    uint32_t width, height;
    uint32_t mipCount;
    bool isBc6h;
};

// Load DDS file (BC6H only)
DDSFile loadDDS(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open DDS: " + path);
    }

    DDSHeader header;
    DDSPixelFormat pixelFormat;

    file.read(reinterpret_cast<char*>(&header.magic), 4);
    if (header.magic != 0x20534444) { // "DDS "
        throw std::runtime_error("Not a valid DDS file: " + path);
    }

    file.read(reinterpret_cast<char*>(&header.size), 120);
    file.read(reinterpret_cast<char*>(&pixelFormat), 32);

    uint32_t caps, caps2, caps3, caps4, reserved;
    file.read(reinterpret_cast<char*>(&caps), 4);
    file.read(reinterpret_cast<char*>(&caps2), 4);
    file.read(reinterpret_cast<char*>(&caps3), 4);
    file.read(reinterpret_cast<char*>(&caps4), 4);
    file.read(reinterpret_cast<char*>(&reserved), 4);

    // DX10 header
    uint32_t dxgiFormat;
    file.read(reinterpret_cast<char*>(&dxgiFormat), 4);

    // Read remaining pixel data
    std::vector<uint8_t> pixelData;
    char byte;
    while (file.read(&byte, 1)) {
        pixelData.push_back(byte);
    }

    DDSFile result;
    result.width = header.width;
    result.height = header.height;
    result.mipCount = header.mipMapCount > 0 ? header.mipMapCount : 1;
    result.isBc6h = (dxgiFormat == 95); // DXGI_FORMAT_BC6H_UF16
    result.data = pixelData;

    return result;
}

// Vulkan utility functions
class VulkanApp {
public:
    VulkanApp();
    ~VulkanApp();

    void run();

private:
    // Window
    GLFWwindow* window = nullptr;

    // Vulkan
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Descriptor resources
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // Textures
    std::array<VkImage, 4> latentImages;
    std::array<VkImageView, 4> latentImageViews;
    std::array<VkDeviceMemory, 4> latentMemory;
    VkSampler sampler = VK_NULL_HANDLE;

    // Buffers
    VkBuffer decoderBuffer = VK_NULL_HANDLE;
    VkDeviceMemory decoderMemory = VK_NULL_HANDLE;

    VkBuffer lodBiasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lodBiasMemory = VK_NULL_HANDLE;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;

    uint32_t indexCount = 0;

    // Sync
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // Methods
    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void loadNeuralMaterialAssets();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSet();
    void createPipeline();
    void createBuffers();
    void createSyncObjects();
    void mainLoop();
    void recordCommandBuffer(uint32_t imageIndex);
    void draw();
    void cleanup();

    // Helper functions
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& memory);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void createImage(uint32_t width, uint32_t height, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image,
                     VkDeviceMemory& memory);
    void createImageView(VkImage image, VkFormat format, VkImageView& imageView);
};

VulkanApp::VulkanApp() {}

VulkanApp::~VulkanApp() {
    cleanup();
}

void VulkanApp::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to init GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(1280, 800, "NN-PBR Vulkan Demo", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create window");
    }
}

void VulkanApp::initVulkan() {
    std::cout << "\n=== Initializing Vulkan ===" << std::endl;
    createInstance();
    createSurface();
    selectPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    loadNeuralMaterialAssets();
    createDescriptorSetLayout();
    createDescriptorPool();
    createDescriptorSet();
    createPipeline();
    createBuffers();
    createSyncObjects();
    std::cout << "=== Vulkan Initialization Complete ===" << std::endl;
}

void VulkanApp::createInstance() {
    std::cout << "Creating Vulkan instance..." << std::endl;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "NN-PBR Demo";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;  // Use 1.2 for broader compatibility

    // Get GLFW-required extensions (must call after glfwInit())
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);

    std::vector<const char*> extensions;

    if (glfwExts && glfwExtCount > 0) {
        extensions.insert(extensions.begin(), glfwExts, glfwExts + glfwExtCount);
        std::cout << "GLFW extensions required:" << std::endl;
        for (uint32_t i = 0; i < glfwExtCount; i++) {
            std::cout << "  - " << glfwExts[i] << std::endl;
        }
    }

    // Add platform-specific extensions
#ifdef __APPLE__
    // On macOS, add MoltenVK portability enumeration
    extensions.push_back("VK_KHR_portability_enumeration");
    std::cout << "Added (macOS): VK_KHR_portability_enumeration" << std::endl;
#endif

    std::cout << "Total extensions: " << extensions.size() << std::endl;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = 0;

#ifdef __APPLE__
    // Enable portability enumeration on macOS
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        std::cerr << "ERROR: vkCreateInstance failed with code " << static_cast<int>(result) << std::endl;
        throw std::runtime_error("Failed to create Vulkan instance");
    }

    std::cout << "✓ Vulkan instance created" << std::endl;
}

void VulkanApp::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create surface");
    }
}

void VulkanApp::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No GPUs available");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    physicalDevice = devices[0];
}

void VulkanApp::createLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t graphicsFamily = 0;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsFamily = i;
            break;
        }
    }

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = graphicsFamily;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledExtensionCount = deviceExtensions.size();
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);
}

void VulkanApp::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    VkSurfaceFormatKHR surfaceFormat = formats[0];

    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = 1280;
        extent.height = 800;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = 3;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swapchain");
    }

    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
}

void VulkanApp::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = VK_FORMAT_B8G8R8A8_SRGB;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image view");
        }
    }
}

void VulkanApp::loadNeuralMaterialAssets() {
    std::string exportDir = "/Users/phanisrikar/Desktop/Projects/NN-PBR/runs/iter65k/export";

    // Load DDS latent textures
    for (int i = 0; i < 4; i++) {
        std::string ddsPath = exportDir + "/latent_" + (i < 10 ? "0" : "") + std::to_string(i) + ".bc6.dds";
        DDSFile ddsFile = loadDDS(ddsPath);

        std::cout << "Loaded latent_" << i << ": " << ddsFile.width << "x" << ddsFile.height
                  << " mips=" << ddsFile.mipCount << std::endl;

        // For now, just create simple placeholder images
        // In a full impl, you'd upload the BC6H data directly to GPU
        createImage(ddsFile.width, ddsFile.height, VK_FORMAT_BC6H_UFLOAT_BLOCK,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    latentImages[i], latentMemory[i]);

        createImageView(latentImages[i], VK_FORMAT_BC6H_UFLOAT_BLOCK, latentImageViews[i]);
    }

    // Load decoder weights
    std::string decoderPath = exportDir + "/decoder_fp16.bin";
    std::ifstream decoderFile(decoderPath, std::ios::binary);
    if (!decoderFile) {
        throw std::runtime_error("Failed to open decoder: " + decoderPath);
    }

    decoderFile.seekg(0, std::ios::end);
    size_t decoderSize = decoderFile.tellg();
    decoderFile.seekg(0, std::ios::beg);

    std::vector<char> decoderData(decoderSize);
    decoderFile.read(decoderData.data(), decoderSize);
    decoderFile.close();

    createBuffer(decoderSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 decoderBuffer, decoderMemory);

    void* data;
    vkMapMemory(device, decoderMemory, 0, decoderSize, 0, &data);
    std::memcpy(data, decoderData.data(), decoderSize);
    vkUnmapMemory(device, decoderMemory);

    std::cout << "Loaded decoder: " << decoderSize << " bytes" << std::endl;

    // Load LOD biases (from metadata)
    float lodBiases[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    createBuffer(sizeof(lodBiases), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 lodBiasBuffer, lodBiasMemory);

    vkMapMemory(device, lodBiasMemory, 0, sizeof(lodBiases), 0, &data);
    std::memcpy(data, lodBiases, sizeof(lodBiases));
    vkUnmapMemory(device, lodBiasMemory);
}

void VulkanApp::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};

    // Decoder weights
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Latent textures
    for (int i = 0; i < 4; i++) {
        bindings[1 + i].binding = 1 + i;
        bindings[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[1 + i].descriptorCount = 1;
        bindings[1 + i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    // LOD biases
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Sampler
    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void VulkanApp::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[1].descriptorCount = 4;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 1;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[3].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool");
    }
}

void VulkanApp::createDescriptorSet() {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    // Write descriptors
    std::array<VkWriteDescriptorSet, 7> descriptorWrites{};

    VkDescriptorBufferInfo decoderInfo{};
    decoderInfo.buffer = decoderBuffer;
    decoderInfo.offset = 0;
    decoderInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].pBufferInfo = &decoderInfo;

    std::array<VkDescriptorImageInfo, 4> latentInfos{};
    for (int i = 0; i < 4; i++) {
        latentInfos[i].imageView = latentImageViews[i];
        latentInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        descriptorWrites[1 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1 + i].dstSet = descriptorSet;
        descriptorWrites[1 + i].dstBinding = 1 + i;
        descriptorWrites[1 + i].descriptorCount = 1;
        descriptorWrites[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        descriptorWrites[1 + i].pImageInfo = &latentInfos[i];
    }

    VkDescriptorBufferInfo lodBiasInfo{};
    lodBiasInfo.buffer = lodBiasBuffer;
    lodBiasInfo.offset = 0;
    lodBiasInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[5].dstSet = descriptorSet;
    descriptorWrites[5].dstBinding = 5;
    descriptorWrites[5].descriptorCount = 1;
    descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[5].pBufferInfo = &lodBiasInfo;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 8.0f;

    if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler");
    }

    VkDescriptorImageInfo samplerInfo2{};
    samplerInfo2.sampler = sampler;

    descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[6].dstSet = descriptorSet;
    descriptorWrites[6].dstBinding = 6;
    descriptorWrites[6].descriptorCount = 1;
    descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptorWrites[6].pImageInfo = &samplerInfo2;

    vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

void VulkanApp::createPipeline() {
    // Load shader
    std::ifstream fragShaderFile("neural_material_decode.frag.spv", std::ios::binary);
    if (!fragShaderFile) {
        throw std::runtime_error("Failed to load fragment shader");
    }

    std::vector<char> fragShaderCode((std::istreambuf_iterator<char>(fragShaderFile)),
                                     std::istreambuf_iterator<char>());

    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    // Simple vertex shader (quad positions)
    const std::string vertShaderSource = R"glsl(
        #version 450
        layout(location = 0) out vec2 outUV;

        void main() {
            vec2 uv = vec2((gl_VertexIndex & 1), (gl_VertexIndex >> 1) & 1);
            gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            outUV = uv;
        }
    )glsl";

    // For simplicity, assume vert shader is pre-compiled
    // In real code, you'd use glslc to compile

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
}

void VulkanApp::createBuffers() {
    // Create sphere mesh (simplified: 2x2 quads)
    std::vector<Vertex> vertices = {
        // Quad 0
        {{-0.9f, 0.9f, 0.0f}, {0.0f, 0.0f}},
        {{-0.5f, 0.9f, 0.0f}, {1.0f, 0.0f}},
        {{-0.9f, 0.5f, 0.0f}, {0.0f, 1.0f}},
        {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}},

        // Quad 1
        {{-0.4f, 0.9f, 0.0f}, {0.0f, 0.0f}},
        {{0.0f, 0.9f, 0.0f}, {1.0f, 0.0f}},
        {{-0.4f, 0.5f, 0.0f}, {0.0f, 1.0f}},
        {{0.0f, 0.5f, 0.0f}, {1.0f, 1.0f}},

        // Quad 2
        {{0.1f, 0.9f, 0.0f}, {0.0f, 0.0f}},
        {{0.5f, 0.9f, 0.0f}, {1.0f, 0.0f}},
        {{0.1f, 0.5f, 0.0f}, {0.0f, 1.0f}},
        {{0.5f, 0.5f, 0.0f}, {1.0f, 1.0f}},
    };

    std::vector<uint16_t> indices = {
        0, 1, 2, 1, 3, 2,
        4, 5, 6, 5, 7, 6,
        8, 9, 10, 9, 11, 10,
    };

    indexCount = indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    createBuffer(sizeof(Vertex) * vertices.size(),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);

    void* data;
    vkMapMemory(device, stagingMemory, 0, sizeof(Vertex) * vertices.size(), 0, &data);
    std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
    vkUnmapMemory(device, stagingMemory);

    createBuffer(sizeof(Vertex) * vertices.size(),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer, vertexMemory);

    copyBuffer(stagingBuffer, vertexBuffer, sizeof(Vertex) * vertices.size());

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    createBuffer(sizeof(uint16_t) * indices.size(),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);

    vkMapMemory(device, indexMemory, 0, sizeof(uint16_t) * indices.size(), 0, &data);
    std::memcpy(data, indices.data(), sizeof(uint16_t) * indices.size());
    vkUnmapMemory(device, indexMemory);
}

void VulkanApp::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }
}

void VulkanApp::recordCommandBuffer(uint32_t imageIndex) {
    // Placeholder - would record actual drawing commands
}

void VulkanApp::draw() {
    // Placeholder - would implement actual draw call
}

void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        draw();
    }

    vkDeviceWaitIdle(device);
}

void VulkanApp::run() {
    initWindow();
    initVulkan();
    mainLoop();
}

uint32_t VulkanApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module");
    }

    return shaderModule;
}

void VulkanApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags properties, VkBuffer& buffer,
                             VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanApp::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanApp::createImage(uint32_t width, uint32_t height, VkFormat format,
                            VkImageTiling tiling, VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkImage& image,
                            VkDeviceMemory& memory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory");
    }

    vkBindImageMemory(device, image, memory, 0);
}

void VulkanApp::createImageView(VkImage image, VkFormat format, VkImageView& imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }
}

void VulkanApp::cleanup() {
    if (device) {
        vkDeviceWaitIdle(device);

        for (auto semaphore : imageAvailableSemaphores) {
            if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (auto semaphore : renderFinishedSemaphores) {
            if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (auto fence : inFlightFences) {
            if (fence) vkDestroyFence(device, fence, nullptr);
        }

        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        if (sampler) vkDestroySampler(device, sampler, nullptr);

        if (vertexBuffer) vkDestroyBuffer(device, vertexBuffer, nullptr);
        if (vertexMemory) vkFreeMemory(device, vertexMemory, nullptr);
        if (indexBuffer) vkDestroyBuffer(device, indexBuffer, nullptr);
        if (indexMemory) vkFreeMemory(device, indexMemory, nullptr);
        if (decoderBuffer) vkDestroyBuffer(device, decoderBuffer, nullptr);
        if (decoderMemory) vkFreeMemory(device, decoderMemory, nullptr);
        if (lodBiasBuffer) vkDestroyBuffer(device, lodBiasBuffer, nullptr);
        if (lodBiasMemory) vkFreeMemory(device, lodBiasMemory, nullptr);

        for (auto& imageView : latentImageViews) {
            if (imageView) vkDestroyImageView(device, imageView, nullptr);
        }
        for (int i = 0; i < 4; i++) {
            if (latentImages[i]) vkDestroyImage(device, latentImages[i], nullptr);
            if (latentMemory[i]) vkFreeMemory(device, latentMemory[i], nullptr);
        }

        for (auto& imageView : swapchainImageViews) {
            if (imageView) vkDestroyImageView(device, imageView, nullptr);
        }
        if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);

        vkDestroyDevice(device, nullptr);
    }

    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);

    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

int main() {
    try {
        VulkanApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
