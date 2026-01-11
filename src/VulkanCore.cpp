#include "VulkanCore.hpp"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan_hpp_macros.hpp>
#include <format>
#include <iostream>
#include <stdexcept>

// Enable validation layers and debug utils in Debug builds
#ifdef _DEBUG
    #define ENABLE_VALIDATION_LAYERS
#endif

namespace {
#ifdef ENABLE_VALIDATION_LAYERS
VKAPI_ATTR auto VKAPI_CALL
debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT /*severity*/,
              const vk::DebugUtilsMessageTypeFlagsEXT type,
              const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* /*unused*/) -> vk::Bool32 {
    std::cerr << std::format("[{}]: {}", to_string(type), pCallbackData->pMessage)
        << '\n';

    return vk::False;
}

const std::vector kRequiredValidationLayers = {"VK_LAYER_KHRONOS_validation"};
#endif

// Note the following extensions are built-in since 1.2
// VK_KHR_spirv_1_4, VK_KHR_shader_float_controls, VK_KHR_maintenance3, VK_KHR_buffer_device_address
const std::vector kRequiredDeviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRAccelerationStructureExtensionName,
    vk::KHRRayQueryExtensionName,
    vk::KHRDeferredHostOperationsExtensionName,
};
} // namespace

VulkanCore::VulkanCore(GLFWwindow* window) {
    createInstance();
#ifdef ENABLE_VALIDATION_LAYERS
    setupDebugMessenger();
#endif
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
}

auto VulkanCore::findSupportedFormat(
    const std::vector<vk::Format>& candidates,
    const vk::ImageTiling tiling,
    const vk::FormatFeatureFlags features
    ) const -> vk::Format {
    for (const auto format : candidates) {
        vk::FormatProperties props = m_physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (
                props.linearTilingFeatures & features) == features) {
            return format;
        }

        if (tiling == vk::ImageTiling::eOptimal && (
                props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::Format VulkanCore::findDepthFormat() const {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
}

vk::SampleCountFlagBits VulkanCore::findMsaaSamples() const {
    const vk::PhysicalDeviceProperties physicalDeviceProperties = m_physicalDevice.getProperties();
    const vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
                                        physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) {
        return vk::SampleCountFlagBits::e64;
    }
    if (counts & vk::SampleCountFlagBits::e32) {
        return vk::SampleCountFlagBits::e32;
    }
    if (counts & vk::SampleCountFlagBits::e16) {
        return vk::SampleCountFlagBits::e16;
    }
    if (counts & vk::SampleCountFlagBits::e8) {
        return vk::SampleCountFlagBits::e8;
    }
    if (counts & vk::SampleCountFlagBits::e4) {
        return vk::SampleCountFlagBits::e4;
    }
    if (counts & vk::SampleCountFlagBits::e2) {
        return vk::SampleCountFlagBits::e2;
    }

    return vk::SampleCountFlagBits::e1;
}

std::uint32_t VulkanCore::findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
    const auto memProperties = m_physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

std::uint64_t VulkanCore::getAvailableVRAM() const {
    const auto memProperties = m_physicalDevice.getMemoryProperties();
    
    std::uint64_t totalVRAM = 0;
    
    // Sum up all device-local heaps (VRAM)
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; i++) {
        if (memProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            totalVRAM += memProperties.memoryHeaps[i].size;
        }
    }
    
    return totalVRAM;
}

void VulkanCore::createInstance() {
    vk::ApplicationInfo appInfo{
        .pApplicationName = "Cyberpunk City Demo",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = version(),
    };

    std::uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifdef ENABLE_VALIDATION_LAYERS
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    // Enable advanced validation features for more detailed feedback.
    const std::vector<vk::ValidationFeatureEnableEXT> enabledValidationFeatures = {
        // vk::ValidationFeatureEnableEXT::eBestPractices,  // Disabled: generates performance warnings for sub-1MB buffers
        vk::ValidationFeatureEnableEXT::eGpuAssisted,
        vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
    };

    const vk::ValidationFeaturesEXT validationFeatures{
        .enabledValidationFeatureCount = static_cast<uint32_t>(enabledValidationFeatures.size()),
        .pEnabledValidationFeatures = enabledValidationFeatures.data(),
    };

    const vk::InstanceCreateInfo createInfo{
        .pNext = &validationFeatures,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(kRequiredValidationLayers.size()),
        .ppEnabledLayerNames = kRequiredValidationLayers.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
#else
    const vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
#endif

    m_instance = vk::raii::Instance(m_context, createInfo);
}

void VulkanCore::setupDebugMessenger() {
#ifdef ENABLE_VALIDATION_LAYERS
    constexpr vk::DebugUtilsMessengerCreateInfoEXT createInfo{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                           | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                           | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                       | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                       | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback
    };

    m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(createInfo);
#endif
}

void VulkanCore::createSurface(GLFWwindow* window) {
    VkSurfaceKHR _surface = nullptr;

    if (glfwCreateWindowSurface(*m_instance, window, nullptr, &_surface) != 0) {
        throw std::runtime_error("failed to create window surface!");
    }

    m_surface = vk::raii::SurfaceKHR(m_instance, _surface);
}

void VulkanCore::pickPhysicalDevice() {
    const auto physicalDevices = m_instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const auto& device : physicalDevices) {
        if (isDeviceSuitable(device, kRequiredDeviceExtensions)) {
            m_physicalDevice = device;
            break;
        }
    }

    if (m_physicalDevice == nullptr) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VulkanCore::createLogicalDevice() {
    m_queueFamilyIndices = findQueueFamilies(m_physicalDevice);

    auto featureChain = buildFeatureChain();
    auto queueCreateInfos = buildQueueInfos(m_queueFamilyIndices);

    vk::DeviceCreateInfo const deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount =
        static_cast<std::uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(kRequiredDeviceExtensions
            .size()),
        .ppEnabledExtensionNames = kRequiredDeviceExtensions.data(),
    };

    m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
    m_graphicsQueue = vk::raii::Queue(m_device, m_queueFamilyIndices.graphicsFamily.value(), 0);
    m_presentQueue = vk::raii::Queue(m_device, m_queueFamilyIndices.presentFamily.value(), 0);

    vk::detail::defaultDispatchLoaderDynamic.init(*m_instance, *m_device);
}

auto VulkanCore::buildFeatureChain() -> vk::StructureChain<
    vk::PhysicalDeviceFeatures2,
    vk::PhysicalDeviceVulkan13Features,
    vk::PhysicalDeviceVulkan12Features,
    vk::PhysicalDeviceVulkan11Features,
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
    vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV,
    vk::PhysicalDeviceRayQueryFeaturesKHR> {
    vk::PhysicalDeviceFeatures2 deviceFeatures2{
        .features = {
            .multiDrawIndirect = true,
            .drawIndirectFirstInstance = true,
            .samplerAnisotropy = true,
            .vertexPipelineStoresAndAtomics = true,
            .fragmentStoresAndAtomics = true,
            .shaderInt64 = true,
        },
    };
    vk::PhysicalDeviceVulkan13Features vulkan13Features{
        .synchronization2 = true,
        .dynamicRendering = true,
    };
    vk::PhysicalDeviceVulkan12Features vulkan12Features{
        .storageBuffer8BitAccess = true,
        .shaderSampledImageArrayNonUniformIndexing = true,
        .descriptorBindingSampledImageUpdateAfterBind = true,
        .descriptorBindingPartiallyBound = true,
        .descriptorBindingVariableDescriptorCount = true,
        .runtimeDescriptorArray = true,
        .timelineSemaphore = true,
        .bufferDeviceAddress = true,
        .vulkanMemoryModel = true,
        .vulkanMemoryModelDeviceScope = true,
    };
    vk::PhysicalDeviceVulkan11Features vulkan11Features{
        .shaderDrawParameters = true,
    };

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedDynamicStateFeatures{
        .extendedDynamicState = true,
    };
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{
        .accelerationStructure = true,
    };
    vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV clusterAccelFeatures{
        .clusterAccelerationStructure = true,
    };
    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
        .rayQuery = true,
    };

    return {
        deviceFeatures2,
        vulkan13Features,
        vulkan12Features,
        vulkan11Features,
        extendedDynamicStateFeatures,
        accelFeatures,
        clusterAccelFeatures,
        rayQueryFeatures
    };
}

auto VulkanCore::buildQueueInfos(
    const QueueFamilyIndices& queueFamilyIndices) -> std::vector<
    vk::DeviceQueueCreateInfo> {
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

    queueCreateInfos.emplace_back(
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
            .queueCount = 1,
            .pQueuePriorities = &queueFamilyIndices.graphicsFamilyPriority,
        }
        );

    if (queueFamilyIndices.graphicsFamily.value() != queueFamilyIndices.
                                                     presentFamily.value()) {
        queueCreateInfos.emplace_back(
            vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = queueFamilyIndices.presentFamily.value(),
                .queueCount = 1,
                .pQueuePriorities = &queueFamilyIndices.presentationFamilyPriority,
            }
            );
    }

    return queueCreateInfos;
}

auto VulkanCore::isDeviceSuitable(const vk::raii::PhysicalDevice& physicalDevice,
                                  const std::vector<const char*>& extensions) const -> bool {
    auto queueFamilies = findQueueFamilies(physicalDevice);

    if (!queueFamilies.isComplete()) {
        return false;
    }

    if (!checkDeviceExtensionSupport(physicalDevice, extensions)) {
        return false;
    }

    auto featureChain = physicalDevice.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR,
        vk::PhysicalDeviceBufferDeviceAddressFeatures
    >();

    const auto& accelFeatures = featureChain.get<
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
    const auto& rayQueryFeatures = featureChain.get<
        vk::PhysicalDeviceRayQueryFeaturesKHR>();
    const auto& bufferAddressFeatures = featureChain.get<
        vk::PhysicalDeviceBufferDeviceAddressFeatures>();

    return accelFeatures.accelerationStructure &&
           rayQueryFeatures.rayQuery &&
           bufferAddressFeatures.bufferDeviceAddress;
}

auto VulkanCore::findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const -> QueueFamilyIndices {
    QueueFamilyIndices queueFamilyIndices{};

    const auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
        const auto& queueFamily = queueFamilies[i];

        if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && !
            queueFamilyIndices.graphicsFamily.has_value()) {
            queueFamilyIndices.graphicsFamily = i;
        }

        if (physicalDevice.getSurfaceSupportKHR(i, *m_surface) && !queueFamilyIndices.presentFamily.has_value()) {
            queueFamilyIndices.presentFamily = i;
        }
    }

    return queueFamilyIndices;
}

auto VulkanCore::checkDeviceExtensionSupport(
    const vk::raii::PhysicalDevice& physicalDevice,
    const std::vector<const char*>& extensions) -> bool {
    const auto deviceProperties = physicalDevice.getProperties();
    auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

    for (const auto& requiredExtName : extensions) {
        auto matcher = [&](const vk::ExtensionProperties& extension) {
            return std::string(extension.extensionName.data()) == std::string(requiredExtName);
        };

        if (!std::ranges::any_of(availableExtensions, matcher)) {
            std::cout << std::format(
                "Extension {} not found on device {}\n",
                std::string(requiredExtName),
                std::string(deviceProperties.deviceName.data()));
            return false;
        }
    }

    return true;
}
