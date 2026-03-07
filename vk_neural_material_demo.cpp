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
#include <filesystem>
#include <regex>
#include <sstream>

// CMAKE build directory (set during compilation)
#ifndef CMAKE_BINARY_DIR
#define CMAKE_BINARY_DIR "."
#endif

// FP16 to FP32 conversion
inline float fp16_to_fp32(uint16_t h) {
    uint32_t h32 = h;
    uint32_t sign = (h32 >> 15) & 0x1;
    uint32_t exponent = (h32 >> 10) & 0x1F;
    uint32_t mantissa = h32 & 0x3FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal number
        return (sign ? -1.0f : 1.0f) * (mantissa / 1024.0f) * (1.0f / 16384.0f);
    }

    if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }

    // Normalized number
    uint32_t f32_exp = exponent + 112; // 127 - 15 = 112
    uint32_t f32_mantissa = mantissa << 13;
    uint32_t f32 = (sign << 31) | (f32_exp << 23) | f32_mantissa;

    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

// Simple math utilities
struct Vertex {
    glm::vec3 position;
    glm::vec2 uv;
};

// DDS file layout (all offsets from start of file):
// 0x00: magic "DDS " (4 bytes)
// 0x04: DDS_HEADER (124 bytes)
//   0x04: size, 0x08: flags, 0x0C: height, 0x10: width
//   0x14: pitchOrLinearSize, 0x18: depth, 0x1C: mipMapCount
//   0x20: reserved[11] (44 bytes)
//   0x4C: DDS_PIXELFORMAT (32 bytes)  <- fourCC at 0x54
//   0x6C: caps, caps2, caps3, caps4, reserved2
// 0x80: DDS_HEADER_DXT10 (if fourCC=="DX10", 20 bytes)
//   0x80: dxgiFormat, 0x84: resourceDimension, ...
// 0x94: pixel data (if DX10) OR 0x80 (if not DX10)

struct DDSFile {
    std::vector<uint8_t> data;
    uint32_t width, height;
    uint32_t mipCount;
    bool isBc6h;
    bool isRgba16f;  // Added for RGBA16F format support
};

struct ExportMetadata {
    std::filesystem::path exportDir;
    uint32_t latentCount = 0;
    uint32_t hiddenDim = 0;
    uint32_t outDim = 0;
    std::string bc6Format;
    std::array<float, 4> lodBiases{{-1.0f, -2.0f, -3.0f, -4.0f}};
};

static std::filesystem::path defaultExportDir() {
    return std::filesystem::path(CMAKE_BINARY_DIR).parent_path() / "runs" / "mode10_paper512_512_cpu" / "export";
}

static std::string loadTextFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open text file: " + path.string());
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

static uint32_t extractJsonUInt(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Failed to find integer key in metadata.json: " + key);
    }
    return static_cast<uint32_t>(std::stoul(match[1].str()));
}

static std::string extractJsonString(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Failed to find string key in metadata.json: " + key);
    }
    return match[1].str();
}

static std::array<float, 4> extractJsonFloatArray4(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("Failed to find float array key in metadata.json: " + key);
    }

    std::array<float, 4> values{};
    std::stringstream ss(match[1].str());
    std::string token;
    size_t index = 0;
    while (std::getline(ss, token, ',')) {
        if (index >= values.size()) {
            throw std::runtime_error("Unexpected lod_biases length in metadata.json");
        }
        values[index++] = std::stof(token);
    }
    if (index != values.size()) {
        throw std::runtime_error("Expected exactly 4 lod_biases entries in metadata.json");
    }
    return values;
}

static ExportMetadata loadExportMetadata(const std::filesystem::path& exportDir) {
    ExportMetadata metadata;
    metadata.exportDir = exportDir;

    const auto metadataPath = exportDir / "metadata.json";
    const std::string json = loadTextFile(metadataPath);

    metadata.latentCount = extractJsonUInt(json, "latent_count");
    metadata.hiddenDim = extractJsonUInt(json, "hidden_dim");
    metadata.outDim = extractJsonUInt(json, "out_dim");
    metadata.bc6Format = extractJsonString(json, "bc6_format");
    metadata.lodBiases = extractJsonFloatArray4(json, "lod_biases");
    return metadata;
}

// Load DDS file, handles DX10 extended header (required for BC6H)
DDSFile loadDDS(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open DDS: " + path);
    }

    // Read entire file into memory
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> raw(fileSize);
    file.read(reinterpret_cast<char*>(raw.data()), fileSize);
    file.close();

    if (fileSize < 128) throw std::runtime_error("DDS file too small: " + path);

    auto read32 = [&](size_t off) -> uint32_t {
        uint32_t v; std::memcpy(&v, raw.data() + off, 4); return v;
    };

    if (read32(0) != 0x20534444) throw std::runtime_error("Not a valid DDS file: " + path);

    uint32_t height    = read32(0x0C);
    uint32_t width     = read32(0x10);
    uint32_t mipCount  = read32(0x1C);
    uint32_t fourCC    = read32(0x54); // DDS_PIXELFORMAT.fourCC at offset 0x54

    bool isDX10 = (fourCC == 0x30315844); // "DX10"
    size_t pixelDataOffset = isDX10 ? 148 : 128; // 128 + 20 bytes DX10 header

    uint32_t dxgiFormat = 0;
    if (isDX10 && fileSize >= 148) {
        dxgiFormat = read32(0x80); // DX10 header starts at 0x80
    }

    DDSFile result;
    result.width = width;
    result.height = height;
    result.mipCount = mipCount > 0 ? mipCount : 1;
    result.isBc6h = isDX10 && (dxgiFormat == 95); // DXGI_FORMAT_BC6H_UF16 = 95
    result.isRgba16f = isDX10 && (dxgiFormat == 10); // DXGI_FORMAT_R16G16B16A16_FLOAT = 10
    result.data.assign(raw.begin() + pixelDataOffset, raw.end());

    return result;
}

// Debug messenger callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    std::string severity;
    std::string color = "\033[0m";  // Reset color

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity = "ERROR";
        color = "\033[91m";  // Red
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity = "WARNING";
        color = "\033[93m";  // Yellow
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        severity = "INFO";
        color = "\033[92m";  // Green
    } else {
        severity = "VERBOSE";
        color = "\033[94m";  // Blue
    }

    std::string type;
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) {
        type = "GENERAL";
    } else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
        type = "VALIDATION";
    } else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
        type = "PERFORMANCE";
    }

    std::cerr << color << "[Vulkan " << severity << " - " << type << "] "
              << pCallbackData->pMessage << "\033[0m" << std::endl;

    return VK_FALSE;
}

// Vulkan utility functions
class VulkanApp {
public:
    VulkanApp(std::filesystem::path exportDir, bool headless = false, bool debugView = true);
    ~VulkanApp();

    void run();

private:
    // Configuration
    bool headless = false;
    bool debugView = true;
    std::filesystem::path exportDir;
    ExportMetadata exportMetadata;

    // Window
    GLFWwindow* window = nullptr;

    // Vulkan
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapChainFormat = VK_FORMAT_UNDEFINED;  // Store actual swapchain format
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkExtent2D swapChainExtent{};

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

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

    // Sync - Triple buffering (per frame in flight, NOT per swapchain image)
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 3;

    // Cleanup guard
    bool cleanedUp = false;

    // Methods
    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void selectPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void createCommandPool();
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
    std::vector<char> loadShaderFile(const std::string& filename);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& memory);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
    void createImage(uint32_t width, uint32_t height, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image,
                     VkDeviceMemory& memory, uint32_t mipLevels = 1);
    void createImageView(VkImage image, VkFormat format, VkImageView& imageView, uint32_t mipLevels = 1);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                               VkImageLayout newLayout, uint32_t mipLevels = 1);
};

VulkanApp::VulkanApp(std::filesystem::path export_dir, bool headless_mode, bool debug_view)
    : headless(headless_mode), debugView(debug_view), exportDir(std::move(export_dir)) {
    std::cout << "[Init] Headless mode: " << (headless ? "ON" : "OFF") << std::endl;
    std::cout << "[Init] Debug view: " << (debugView ? "ON" : "OFF") << std::endl;
    std::cout << "[Init] Export dir: " << exportDir << std::endl;
}

VulkanApp::~VulkanApp() {
    cleanup();
}

void VulkanApp::initWindow() {
    std::cout << "[Window] Initializing GLFW..." << std::endl;
    if (!glfwInit()) {
        throw std::runtime_error("Failed to init GLFW");
    }

    std::cout << "[Window] GLFW initialized successfully" << std::endl;
    std::cout << "[Window] GLFW version: " << glfwGetVersionString() << std::endl;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    std::cout << "[Window] Creating window (1280x800)..." << std::endl;
    window = glfwCreateWindow(1280, 800, "NN-PBR Vulkan Demo", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create window");
    }

    std::cout << "[Window] Window created successfully at " << window << std::endl;
}

void VulkanApp::initVulkan() {
    std::cout << "\n=== Initializing Vulkan ===" << std::endl;
    try {
        std::cout << "[Init] Creating Vulkan instance..." << std::endl;
        createInstance();

        if (!headless) {
            std::cout << "[Init] Creating surface..." << std::endl;
            createSurface();
        } else {
            std::cout << "[Init] Skipping surface creation (headless mode)" << std::endl;
        }

        std::cout << "[Init] Selecting physical device..." << std::endl;
        selectPhysicalDevice();

        std::cout << "[Init] Creating logical device..." << std::endl;
        createLogicalDevice();

        std::cout << "[Init] Creating command pool..." << std::endl;
        createCommandPool();

        if (!headless) {
            std::cout << "[Init] Creating swapchain..." << std::endl;
            createSwapchain();

            std::cout << "[Init] Creating image views..." << std::endl;
            createImageViews();
        } else {
            std::cout << "[Init] Skipping swapchain and image views (headless mode)" << std::endl;
            // Set a default format for headless mode (needed for render pass)
            swapChainFormat = VK_FORMAT_B8G8R8A8_UNORM;
            std::cout << "[Init] Using default format for headless: B8G8R8A8_UNORM" << std::endl;
        }

        std::cout << "[Init] Loading neural material assets..." << std::endl;
        loadNeuralMaterialAssets();

        std::cout << "[Init] Creating descriptor set layout..." << std::endl;
        createDescriptorSetLayout();

        std::cout << "[Init] Creating descriptor pool..." << std::endl;
        createDescriptorPool();

        std::cout << "[Init] Creating descriptor set..." << std::endl;
        createDescriptorSet();

        std::cout << "[Init] Creating pipeline..." << std::endl;
        createPipeline();

        std::cout << "[Init] Creating buffers..." << std::endl;
        createBuffers();

        if (!headless) {
            std::cout << "[Init] Creating sync objects..." << std::endl;
            createSyncObjects();
        } else {
            std::cout << "[Init] Skipping sync objects (headless mode)" << std::endl;
        }

        std::cout << "=== Vulkan Initialization Complete ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Init] FATAL: " << e.what() << std::endl;
        throw;
    }
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
    } else {
        std::cout << "WARNING: glfwGetRequiredInstanceExtensions returned NULL or 0 extensions" << std::endl;
        // Fallback for headless or when GLFW doesn't return extensions
        // Always add VK_KHR_surface as it's needed for window surfaces
        extensions.push_back("VK_KHR_surface");
        std::cout << "Added (fallback): VK_KHR_surface" << std::endl;
#ifdef __APPLE__
        extensions.push_back("VK_EXT_metal_surface");
        std::cout << "Added (fallback): VK_EXT_metal_surface" << std::endl;
#endif
    }

    // Add platform-specific extensions
#ifdef __APPLE__
    // On macOS, add MoltenVK portability enumeration
    extensions.push_back("VK_KHR_portability_enumeration");
    std::cout << "Added (macOS): VK_KHR_portability_enumeration" << std::endl;
#endif

    // Add debug utilities extension for better error messages
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    std::cout << "Added: VK_EXT_DEBUG_UTILS_EXTENSION_NAME" << std::endl;

    std::cout << "Total extensions: " << extensions.size() << std::endl;
    std::cout << "All extensions to be enabled:" << std::endl;
    for (const auto& ext : extensions) {
        std::cout << "  - " << ext << std::endl;
    }

    // Enable validation layers
    const char* validationLayers[] = {"VK_LAYER_KHRONOS_validation"};

    std::cout << "Enabled validation layers:" << std::endl;
    for (const auto& layer : validationLayers) {
        std::cout << "  - " << layer << std::endl;
    }

    // Create debug messenger info for instance creation
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = debugCallback;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.enabledLayerCount = 1;
    createInfo.ppEnabledLayerNames = validationLayers;
    createInfo.pNext = &debugCreateInfo;

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

    // Create debug messenger
    auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");

    if (vkCreateDebugUtilsMessengerEXT) {
        result = vkCreateDebugUtilsMessengerEXT(instance, &debugCreateInfo, nullptr, &debugMessenger);
        if (result == VK_SUCCESS) {
            std::cout << "✓ Debug messenger created" << std::endl;
        } else {
            std::cerr << "WARNING: Failed to create debug messenger" << std::endl;
        }
    } else {
        std::cerr << "WARNING: vkCreateDebugUtilsMessengerEXT not available" << std::endl;
    }
}

void VulkanApp::createSurface() {
    if (!window) {
        throw std::runtime_error("Window is NULL");
    }

    if (!instance) {
        throw std::runtime_error("Vulkan instance is NULL");
    }

    VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (result != VK_SUCCESS) {
        std::string errorMsg = "Failed to create surface (Vulkan error code: " + std::to_string(result) + ")";
        std::cerr << "ERROR: " << errorMsg << std::endl;
        throw std::runtime_error(errorMsg);
    }

    std::cout << "✓ Surface created successfully" << std::endl;
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
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        "VK_KHR_portability_subset",  // Required on macOS
    };

    // Enable dynamic rendering feature
    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &dynamicRenderingFeatures;
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
    swapChainFormat = surfaceFormat.format;  // SAVE the actual format!

    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width = 1280;
        extent.height = 800;
    }

    swapChainExtent = extent;

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
        createInfo.format = swapChainFormat;  // Use actual swapchain format!
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

void VulkanApp::createCommandPool() {
    std::cout << "[CommandPool] Creating command pool..." << std::endl;

    // Find graphics queue family index
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

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    std::cout << "[CommandPool] ✓ Command pool created" << std::endl;
}

void VulkanApp::loadNeuralMaterialAssets() {
    exportMetadata = loadExportMetadata(exportDir);
    if (exportMetadata.latentCount != 4) {
        throw std::runtime_error("Vulkan sample currently supports exactly 4 latent textures.");
    }
    if (exportMetadata.hiddenDim != 16 || exportMetadata.outDim != 8) {
        throw std::runtime_error("Vulkan sample currently supports decoder dims 12 -> 16 -> 8 only.");
    }
    if (exportMetadata.bc6Format != "DXGI_FORMAT_BC6H_UF16") {
        throw std::runtime_error("Vulkan sample currently supports DXGI_FORMAT_BC6H_UF16 only.");
    }

    std::cout << "[Assets] Loading from: " << exportDir << std::endl;

    // Load DDS latent textures
    for (int i = 0; i < 4; i++) {
        std::cout << "[Assets] Loading latent texture " << i << "..." << std::endl;
        std::string idx = (i < 10 ? "0" : "") + std::to_string(i);
        std::string ddsPath = (exportDir / ("latent_" + idx + ".bc6.dds")).string();

        std::cout << "  - Path: " << ddsPath << std::endl;
        DDSFile ddsFile = loadDDS(ddsPath);

        if (!ddsFile.isBc6h) {
            throw std::runtime_error("Unsupported DDS format in: " + ddsPath);
        }
        VkFormat texFormat = VK_FORMAT_BC6H_UFLOAT_BLOCK;
        std::cout << "  - Loaded: " << ddsFile.width << "x" << ddsFile.height
                  << " mips=" << ddsFile.mipCount
                  << " format=BC6H_UF16" << std::endl;
        
        createImage(ddsFile.width, ddsFile.height, texFormat,
                    VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    latentImages[i], latentMemory[i], ddsFile.mipCount);

        createImageView(latentImages[i], texFormat, latentImageViews[i], ddsFile.mipCount);

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingMemory;
        createBuffer(ddsFile.data.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingMemory);

        void* stagingData;
        vkMapMemory(device, stagingMemory, 0, ddsFile.data.size(), 0, &stagingData);
        std::memcpy(stagingData, ddsFile.data.data(), ddsFile.data.size());
        vkUnmapMemory(device, stagingMemory);

        transitionImageLayout(latentImages[i], texFormat,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             ddsFile.mipCount);

        // Build copy regions for all mip levels
        std::vector<VkBufferImageCopy> copyRegions;
        size_t bufOffset = 0;
        for (uint32_t mip = 0; mip < ddsFile.mipCount; ++mip) {
            uint32_t mipW = std::max(1u, ddsFile.width  >> mip);
            uint32_t mipH = std::max(1u, ddsFile.height >> mip);
            uint32_t bw = std::max(1u, (mipW + 3) / 4);
            uint32_t bh = std::max(1u, (mipH + 3) / 4);
            size_t mipBytes = (size_t)bw * bh * 16;
            VkBufferImageCopy region{};
            region.bufferOffset      = bufOffset;
            region.bufferRowLength   = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel       = mip;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount     = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {mipW, mipH, 1};
            copyRegions.push_back(region);
            bufOffset += mipBytes;
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmdBuf;
        vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuf, &beginInfo);

        vkCmdCopyBufferToImage(cmdBuf, stagingBuffer, latentImages[i],
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(copyRegions.size()), copyRegions.data());

        vkEndCommandBuffer(cmdBuf);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuf;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);

        transitionImageLayout(latentImages[i], texFormat,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             ddsFile.mipCount);

        std::cout << "  ✓ Latent texture " << i << " loaded (" << ddsFile.mipCount << " mips)" << std::endl;
    }

    // Load decoder weights
    std::cout << "[Assets] Loading decoder weights..." << std::endl;
    std::string decoderPath = (exportDir / "decoder_fp16.bin").string();
    std::cout << "  - Path: " << decoderPath << std::endl;

    std::ifstream decoderFile(decoderPath, std::ios::binary);
    if (!decoderFile) {
        throw std::runtime_error("Failed to open decoder: " + decoderPath);
    }

    decoderFile.seekg(0, std::ios::end);
    size_t decoderSize = decoderFile.tellg();
    decoderFile.seekg(0, std::ios::beg);
    std::cout << "  - Size: " << decoderSize << " bytes" << std::endl;

    std::vector<char> decoderData(decoderSize);
    decoderFile.read(decoderData.data(), decoderSize);
    decoderFile.close();

    // Convert FP16 to FP32
    std::cout << "  - Converting FP16 to FP32..." << std::endl;
    uint16_t* fp16Data = (uint16_t*)decoderData.data();
    size_t numValues = decoderSize / 2;  // FP16 is 2 bytes per value
    const size_t expectedValues =
        (12 * exportMetadata.hiddenDim)
        + exportMetadata.hiddenDim
        + (exportMetadata.outDim * exportMetadata.hiddenDim)
        + exportMetadata.outDim;
    if (numValues != expectedValues) {
        throw std::runtime_error(
            "decoder_fp16.bin size mismatch; expected "
            + std::to_string(expectedValues) + " FP16 values, got "
            + std::to_string(numValues)
        );
    }
    std::vector<float> fp32Data(numValues);
    for (size_t i = 0; i < numValues; i++) {
        fp32Data[i] = fp16_to_fp32(fp16Data[i]);
    }

    // Debug: print first few values
    std::cout << "  - First 10 converted values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), numValues); i++) {
        std::cout << "    [" << i << "] FP16=0x" << std::hex << fp16Data[i] << std::dec << " -> FP32=" << fp32Data[i] << std::endl;
    }

    size_t fp32Size = fp32Data.size() * sizeof(float);
    std::cout << "  - Creating buffer (" << fp32Size << " bytes for " << numValues << " values)..." << std::endl;
    createBuffer(fp32Size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 decoderBuffer, decoderMemory);

    std::cout << "  - Mapping memory..." << std::endl;
    void* data;
    vkMapMemory(device, decoderMemory, 0, fp32Size, 0, &data);
    std::memcpy(data, fp32Data.data(), fp32Size);
    vkUnmapMemory(device, decoderMemory);
    std::cout << "  ✓ Decoder loaded (converted " << numValues << " FP16 → FP32)" << std::endl;

    // Load LOD biases (from metadata)
    std::cout << "[Assets] Setting LOD biases..." << std::endl;
    float lodBiases[4] = {
        exportMetadata.lodBiases[0],
        exportMetadata.lodBiases[1],
        exportMetadata.lodBiases[2],
        exportMetadata.lodBiases[3],
    };

    std::cout << "  - Creating buffer..." << std::endl;
    createBuffer(sizeof(lodBiases), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 lodBiasBuffer, lodBiasMemory);

    std::cout << "  - Mapping memory..." << std::endl;
    vkMapMemory(device, lodBiasMemory, 0, sizeof(lodBiases), 0, &data);
    std::memcpy(data, lodBiases, sizeof(lodBiases));
    vkUnmapMemory(device, lodBiasMemory);
    std::cout << "  ✓ LOD biases set" << std::endl;
}

void VulkanApp::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};

    // Decoder weights (binding 0)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Latent textures (bindings 1-4) as combined image samplers - texture + sampler together
    for (int i = 0; i < 4; i++) {
        bindings[1 + i].binding = 1 + i;
        bindings[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[1 + i].descriptorCount = 1;
        bindings[1 + i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    // LOD biases (binding 5)
    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout");
    }
}

void VulkanApp::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 4;  // 4 latent textures with samplers
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 1;

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

    // Create sampler
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

    // Write descriptors
    std::array<VkWriteDescriptorSet, 6> descriptorWrites{};

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
        latentInfos[i].sampler = sampler;  // Include sampler with image
        latentInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        descriptorWrites[1 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1 + i].dstSet = descriptorSet;
        descriptorWrites[1 + i].dstBinding = 1 + i;
        descriptorWrites[1 + i].descriptorCount = 1;
        descriptorWrites[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
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

    vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

std::vector<char> VulkanApp::loadShaderFile(const std::string& filename) {
    // Try multiple search paths
    std::vector<std::string> searchPaths = {
        filename,
        "shaders/" + filename,
        "../shaders/" + filename,
        "../../shaders/" + filename,
        std::string(CMAKE_BINARY_DIR) + "/shaders/" + filename,
    };

    for (const auto& path : searchPaths) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<char> buffer(size);
            if (file.read(buffer.data(), size)) {
                std::cout << "[Shader] Loaded " << filename << " from: " << path << " (" << size << " bytes)" << std::endl;
                return buffer;
            }
        }
    }

    // Print all attempted paths for debugging
    std::string errorMsg = "Failed to load shader: " + filename + "\nSearched paths:\n";
    for (const auto& path : searchPaths) {
        errorMsg += "  - " + path + "\n";
    }
    throw std::runtime_error(errorMsg);
}

void VulkanApp::createPipeline() {
    std::cout << "[Pipeline] Starting pipeline creation..." << std::endl;

    // Load compiled shaders from SPIR-V
    const char* fragShaderName = debugView
        ? "neural_material_decode_debug.frag.spv"
        : "neural_material_decode.frag.spv";
    std::vector<char> fragShaderCode = loadShaderFile(fragShaderName);
    std::cout << "[Pipeline] Fragment shader loaded: " << fragShaderCode.size() << " bytes" << std::endl;

    std::vector<char> vertShaderCode = loadShaderFile("neural_material_decode.vert.spv");
    std::cout << "[Pipeline] Vertex shader loaded: " << vertShaderCode.size() << " bytes" << std::endl;

    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    std::cout << "[Pipeline] Shader modules created successfully" << std::endl;

    // Pipeline shader stages
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // Vertex input state
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
    // Position attribute (location 0)
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    // UV attribute (location 1)
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, uv);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // Use swapchain extent, or default to 1280x800 if in headless mode
    VkExtent2D renderExtent = swapChainExtent;
    if (renderExtent.width == 0 || renderExtent.height == 0) {
        renderExtent.width = 1280;
        renderExtent.height = 800;
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)renderExtent.width;
    viewport.height = (float)renderExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = renderExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout");
    }
    std::cout << "[Pipeline] Pipeline layout created" << std::endl;
    std::cout << "[Pipeline] Using VK_KHR_dynamic_rendering (no render pass needed)" << std::endl;

    // Create graphics pipeline with dynamic states
    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // Change viewport state to use NULL pointers since they're now dynamic
    VkPipelineViewportStateCreateInfo dynamicViewportState{};
    dynamicViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    dynamicViewportState.viewportCount = 1;
    dynamicViewportState.pViewports = nullptr;  // Dynamic
    dynamicViewportState.scissorCount = 1;
    dynamicViewportState.pScissors = nullptr;   // Dynamic

    // Setup color attachment format for dynamic rendering
    VkFormat colorFormat = swapChainFormat;
    if (colorFormat == VK_FORMAT_UNDEFINED) {
        colorFormat = VK_FORMAT_B8G8R8A8_UNORM;  // Fallback
    }

    VkPipelineRenderingCreateInfoKHR pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pNext = &pipelineRenderingInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &dynamicViewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;  // No render pass with dynamic rendering
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create graphics pipeline");
    }
    std::cout << "[Pipeline] Graphics pipeline created" << std::endl;

    // Create command buffers
    std::cout << "[Pipeline] Creating command buffers..." << std::endl;
    commandBuffers.resize(swapchainImages.size());
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
    std::cout << "[Pipeline] Command buffers allocated" << std::endl;

    // Store render pass and framebuffers for use in recordCommandBuffer
    // Note: In a real implementation, these would be member variables
    // For now, we'll store them in a temporary way

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    // Clean up render pass and framebuffers after storing references
    // These will be cleaned up in cleanup() function
    // For now, we store as member variables

    std::cout << "[Pipeline] Pipeline creation completed" << std::endl;
}

void VulkanApp::createBuffers() {
    std::cout << "[Buffers] Creating vertex and index buffers..." << std::endl;

    // Create full-screen quad
    std::cout << "[Buffers] Setting up full-screen quad vertices..." << std::endl;
    std::vector<Vertex> vertices = {
        // Full-screen quad (normalized device coordinates)
        // Note: UV Y is flipped because Vulkan viewport Y increases downward
        {{-1.0f,  1.0f, 0.0f}, {0.0f, 0.0f}},  // Top-left
        {{ 1.0f,  1.0f, 0.0f}, {1.0f, 0.0f}},  // Top-right
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},  // Bottom-left
        {{ 1.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},  // Bottom-right
    };

    std::cout << "[Buffers] Setting up quad indices..." << std::endl;
    std::vector<uint16_t> indices = {
        0, 2, 1,  // First triangle (clockwise for front-face)
        1, 2, 3,  // Second triangle (clockwise for front-face)
    };

    indexCount = indices.size();
    std::cout << "[Buffers] Index count: " << indexCount << std::endl;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    std::cout << "[Buffers] Creating staging buffer for vertices..." << std::endl;
    createBuffer(sizeof(Vertex) * vertices.size(),
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);

    std::cout << "[Buffers] Mapping and copying vertex data..." << std::endl;
    void* data;
    vkMapMemory(device, stagingMemory, 0, sizeof(Vertex) * vertices.size(), 0, &data);
    std::memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
    vkUnmapMemory(device, stagingMemory);

    std::cout << "[Buffers] Creating vertex buffer..." << std::endl;
    createBuffer(sizeof(Vertex) * vertices.size(),
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer, vertexMemory);

    std::cout << "[Buffers] Copying vertex data to GPU..." << std::endl;
    copyBuffer(stagingBuffer, vertexBuffer, sizeof(Vertex) * vertices.size());

    std::cout << "[Buffers] Destroying staging buffer..." << std::endl;
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    std::cout << "[Buffers] Creating index buffer..." << std::endl;
    createBuffer(sizeof(uint16_t) * indices.size(),
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 indexBuffer, indexMemory);

    std::cout << "[Buffers] Mapping and copying index data..." << std::endl;
    vkMapMemory(device, indexMemory, 0, sizeof(uint16_t) * indices.size(), 0, &data);
    std::memcpy(data, indices.data(), sizeof(uint16_t) * indices.size());
    vkUnmapMemory(device, indexMemory);

    std::cout << "[Buffers] ✓ All buffers created successfully" << std::endl;
}

void VulkanApp::createSyncObjects() {
    // Triple buffering: one semaphore pair + fence per frame in flight
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Create semaphore pair + fence for each frame in flight
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects");
        }
    }
}

void VulkanApp::recordCommandBuffer(uint32_t frameIndex) {
    VkCommandBuffer commandBuffer = commandBuffers[frameIndex];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    // Transition swapchain image to COLOR_ATTACHMENT_OPTIMAL
    uint32_t swapImageIndex = frameIndex % swapchainImages.size();
    if (!swapchainImages.empty()) {
        VkImageMemoryBarrier imageBarrier{};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarrier.srcAccessMask = 0;
        imageBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imageBarrier.image = swapchainImages[swapImageIndex];
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    }

    // Get color attachment for dynamic rendering
    VkFormat colorFormat = swapChainFormat;
    if (colorFormat == VK_FORMAT_UNDEFINED) {
        colorFormat = VK_FORMAT_B8G8R8A8_UNORM;
    }

    // Setup color attachment for dynamic rendering
    VkRenderingAttachmentInfoKHR colorAttachmentInfo{};
    colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colorAttachmentInfo.imageView = swapchainImageViews.empty() ? VK_NULL_HANDLE : swapchainImageViews[swapImageIndex];
    colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentInfo.clearValue = {{0.1f, 0.1f, 0.1f, 1.0f}};

    VkRenderingInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = swapChainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;

    // Use vkCmdBeginRenderingKHR for dynamic rendering
    auto vkCmdBeginRenderingKHR = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdBeginRenderingKHR");
    if (vkCmdBeginRenderingKHR) {
        vkCmdBeginRenderingKHR(commandBuffer, &renderingInfo);
    }

    // Bind pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Bind vertex buffer
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);

    // Bind index buffer
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    // Bind descriptor set
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    // Set viewport and scissor
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Draw call with indices
    if (indexCount > 0) {
        vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
    }

    // End dynamic rendering
    auto vkCmdEndRenderingKHR = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(device, "vkCmdEndRenderingKHR");
    if (vkCmdEndRenderingKHR) {
        vkCmdEndRenderingKHR(commandBuffer);
    }

    // Transition swapchain image to PRESENT_SRC_KHR
    if (!swapchainImages.empty()) {
        VkImageMemoryBarrier imageBarrier{};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        imageBarrier.dstAccessMask = 0;
        imageBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        imageBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imageBarrier.image = swapchainImages[swapImageIndex];
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &imageBarrier);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer");
    }
}

void VulkanApp::draw() {
    // Wait for this frame to complete before reusing its resources
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquire next image using this frame's semaphore
    uint32_t imageIndex;
    VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                          imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (acquireResult != VK_SUCCESS) {
        return;  // Handle swapchain recreation if needed
    }

    // Reset fence after acquiring (not before!)
    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    // Reset and record command buffer for the acquired image
    vkResetCommandBuffer(commandBuffers[imageIndex], 0);
    recordCommandBuffer(imageIndex);

    // Submit work using this frame's semaphores
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw command buffer");
    }

    // Present with this frame's semaphore
    VkSwapchainKHR swapchains[] = {swapchain};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(graphicsQueue, &presentInfo);

    // Move to next frame in flight
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Handle ESC key to close window
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        draw();
    }

    vkDeviceWaitIdle(device);
}

void VulkanApp::run() {
    try {
        if (!headless) {
            std::cout << "[Run] Initializing window..." << std::endl;
            initWindow();
        } else {
            std::cout << "[Run] Skipping window initialization (headless mode)" << std::endl;
        }

        std::cout << "[Run] Initializing Vulkan..." << std::endl;
        initVulkan();

        if (!headless) {
            std::cout << "[Run] Starting main loop..." << std::endl;
            mainLoop();
        } else {
            std::cout << "[Run] Skipping main loop (headless mode)" << std::endl;
            std::cout << "[Run] ✓ Headless mode complete - all systems initialized!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[Run] FATAL ERROR: " << e.what() << std::endl;
        // Don't call cleanup() here - it will be called by destructor
        throw;
    }
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
                            VkDeviceMemory& memory, uint32_t mipLevels) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
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

void VulkanApp::createImageView(VkImage image, VkFormat format, VkImageView& imageView, uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view");
    }
}

void VulkanApp::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                                       VkImageLayout newLayout, uint32_t mipLevels) {
    // Create one-time command buffer
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

    // Prepare image memory barrier
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::runtime_error("Unsupported layout transition");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    vkEndCommandBuffer(commandBuffer);

    // Submit and wait
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanApp::cleanup() {
    if (cleanedUp) {
        std::cout << "[Cleanup] Already cleaned up, skipping..." << std::endl;
        return;
    }
    cleanedUp = true;

    std::cout << "[Cleanup] Starting cleanup..." << std::endl;
    try {
        if (device) {
            std::cout << "[Cleanup] Waiting for device idle..." << std::endl;
            vkDeviceWaitIdle(device);

            std::cout << "[Cleanup] Destroying semaphores..." << std::endl;
            for (auto semaphore : imageAvailableSemaphores) {
                if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
            }
            for (auto semaphore : renderFinishedSemaphores) {
                if (semaphore) vkDestroySemaphore(device, semaphore, nullptr);
            }
            for (auto fence : inFlightFences) {
                if (fence) vkDestroyFence(device, fence, nullptr);
            }

            std::cout << "[Cleanup] Destroying pipeline resources..." << std::endl;
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            if (sampler) vkDestroySampler(device, sampler, nullptr);

            std::cout << "[Cleanup] Destroying buffers..." << std::endl;
            if (vertexBuffer) vkDestroyBuffer(device, vertexBuffer, nullptr);
            if (vertexMemory) vkFreeMemory(device, vertexMemory, nullptr);
            if (indexBuffer) vkDestroyBuffer(device, indexBuffer, nullptr);
            if (indexMemory) vkFreeMemory(device, indexMemory, nullptr);
            if (decoderBuffer) vkDestroyBuffer(device, decoderBuffer, nullptr);
            if (decoderMemory) vkFreeMemory(device, decoderMemory, nullptr);
            if (lodBiasBuffer) vkDestroyBuffer(device, lodBiasBuffer, nullptr);
            if (lodBiasMemory) vkFreeMemory(device, lodBiasMemory, nullptr);

            std::cout << "[Cleanup] Destroying latent image views..." << std::endl;
            for (auto& imageView : latentImageViews) {
                if (imageView) vkDestroyImageView(device, imageView, nullptr);
            }
            std::cout << "[Cleanup] Destroying latent images..." << std::endl;
            for (int i = 0; i < 4; i++) {
                if (latentImages[i]) vkDestroyImage(device, latentImages[i], nullptr);
                if (latentMemory[i]) vkFreeMemory(device, latentMemory[i], nullptr);
            }

            std::cout << "[Cleanup] Destroying swapchain image views..." << std::endl;
            for (auto& imageView : swapchainImageViews) {
                if (imageView) vkDestroyImageView(device, imageView, nullptr);
            }
            std::cout << "[Cleanup] Destroying swapchain..." << std::endl;
            if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);

            std::cout << "[Cleanup] Destroying device..." << std::endl;
            vkDestroyDevice(device, nullptr);
        }

        std::cout << "[Cleanup] Destroying surface..." << std::endl;
        if (surface && instance) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
        }

        std::cout << "[Cleanup] Destroying debug messenger..." << std::endl;
        if (debugMessenger && instance) {
            auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
                instance, "vkDestroyDebugUtilsMessengerEXT");
            if (vkDestroyDebugUtilsMessengerEXT) {
                vkDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
            }
        }

        std::cout << "[Cleanup] Destroying instance..." << std::endl;
        if (instance) {
            vkDestroyInstance(instance, nullptr);
        }

        std::cout << "[Cleanup] Terminating GLFW..." << std::endl;
        if (window) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
        std::cout << "[Cleanup] Cleanup complete" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[Cleanup] Error during cleanup: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        bool headless = false;
        bool debugView = true;
        std::filesystem::path exportDir = defaultExportDir();
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--headless") {
                headless = true;
            } else if (std::string(argv[i]) == "--final-view") {
                debugView = false;
            } else if (std::string(argv[i]) == "--debug-view") {
                debugView = true;
            } else if (std::string(argv[i]) == "--export-dir" && (i + 1) < argc) {
                exportDir = argv[++i];
            }
        }

        std::cout << "NN-PBR Vulkan Demo" << std::endl;
        if (headless) {
            std::cout << "Mode: HEADLESS (testing)" << std::endl;
        } else {
            std::cout << "Mode: WINDOWED (Press ESC to exit)" << std::endl;
        }
        std::cout << "Export dir: " << exportDir << std::endl;
        std::cout << "View: " << (debugView ? "DEBUG" : "FINAL") << std::endl;
        std::cout << std::endl;

        VulkanApp app(exportDir, headless, debugView);
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
