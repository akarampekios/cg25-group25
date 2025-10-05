#pragma once

#include "utils/window_utils.hpp"
#include "utils/physical_device_utils.hpp"
#include "utils/logical_device_utils.hpp"
#include "utils/swap_chain_utils.hpp"
#include "utils/shader_utils.hpp"
#include "utils/image_utils.hpp"

#include "Camera.h"
#include "structs.hpp"
#include "vulkan_transfer_context.hpp"

#include "GLTFLoader.h"  // TODO: move

const std::vector kRequiredValidationLayers = {"VK_LAYER_KHRONOS_validation"};

const std::vector kRequiredDeviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRAccelerationStructureExtensionName,
    // vk::KHRRayTracingPipelineExtensionName,
    vk::KHRDeferredHostOperationsExtensionName,
    vk::KHRPipelineLibraryExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRBufferDeviceAddressExtensionName,
    vk::KHRShaderFloatControlsExtensionName,
    vk::KHRMaintenance3ExtensionName,
};

class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    void drawFrame(const Camera& camera);
private:
    GLFWwindow* m_window;
    WindowUtils m_windowUtils;

    std::uint32_t m_currentFrame = 0;
    std::uint32_t m_semaphoreIndex = 0;

    VulkanTransferContext m_vulkanTransferContext;
    PhysicalDeviceUtils m_physicalDeviceUtils;
    LogicalDeviceUtils m_deviceUtils;
    SwapChainUtils m_swapChainUtils;
    ImageUtils m_imageUtils;

    vk::raii::Context m_context;
    vk::raii::Instance m_instance = nullptr;
    vk::raii::SurfaceKHR m_surface = nullptr;

    vk::raii::PhysicalDevice m_physicalDevice = nullptr;
    vk::raii::Device m_device = nullptr;

    QueueFamilyIndices m_queueFamilyIndices;
    vk::raii::Queue m_graphicsQueue = nullptr;
    vk::raii::Queue m_presentQueue = nullptr;

    vk::raii::SwapchainKHR m_swapChain = nullptr;
    std::vector<vk::Image> m_swapChainImages;
    std::vector<vk::raii::ImageView> m_swapChainImageViews;
    vk::Format m_swapChainImageFormat;
    vk::Extent2D m_swapChainExtent;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool m_descriptorPool = nullptr;

    vk::SampleCountFlagBits m_msaaSamples = vk::SampleCountFlagBits::e1;
    vk::raii::PipelineLayout m_pipelineLayout = nullptr;
    vk::raii::Pipeline m_graphicsPipeline = nullptr;

    vk::raii::Image m_colorImage = nullptr;
    vk::raii::DeviceMemory m_colorImageMemory = nullptr;
    vk::raii::ImageView m_colorImageView = nullptr;

    vk::raii::Image m_depthImage = nullptr;
    vk::raii::DeviceMemory m_depthImageMemory = nullptr;
    vk::raii::ImageView m_depthImageView = nullptr;

    vk::raii::CommandPool m_commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;
    std::vector<vk::raii::Semaphore> m_presentationCompleteSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::raii::Fence> m_inFlightFences;

    vk::raii::Sampler m_baseColorTextureSampler = nullptr;
    vk::raii::Sampler m_metallicRoughnessTextureSampler = nullptr;
    vk::raii::Sampler m_normalTextureSampler = nullptr;
    vk::raii::Sampler m_emissiveTextureSampler = nullptr;
    vk::raii::Sampler m_occlusionTextureSampler = nullptr;

    Scene scene;

    void createInstance();

    void createSurface();

    void pickPhysicalDevice();

    void pickMsaaSamples();

    void createLogicalDevice();

    void createSwapChain();

    void createImageViews();

    void createDescriptorSetLayout();

    void createGraphicsPipeline();

    void createCommandPool();

    void createColorResources();

    void createDepthResources();

    void createCommandBuffers();

    void createSyncObjects();

    void createTextureSamplers();

    void createDescriptorPool();

    void loadScene();

    void updateUniformBuffers(const Camera& camera);

    void recordCommandBuffer(std::uint32_t imageIndex);
};
