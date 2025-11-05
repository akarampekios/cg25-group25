#pragma once

#include <vector>
#include <cstdint>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "Shader.hpp"

class VulkanCore;
class ResourceManager;
class CommandManager;
class SwapChain;
class ImageManager;
class BufferManager;
class PostProcessingStack;
struct Scene;

class RayQueryPipeline {
public:
    explicit RayQueryPipeline(VulkanCore& vulkanCore,
                              ResourceManager& resourceManager,
                              CommandManager& commandManager,
                              SwapChain& swapChain,
                              ImageManager& imageManager,
                              BufferManager& bufferManager,
                              PostProcessingStack& postProcessingPipeline);

    ~RayQueryPipeline() = default;

    void drawFrame(const Scene& scene);

private:
    VulkanCore& m_vulkanCore;
    ResourceManager& m_resourceManager;
    CommandManager& m_commandManager;
    SwapChain& m_swapChain;
    ImageManager& m_imageManager;
    BufferManager& m_bufferManager;
    PostProcessingStack& m_postProcessingPipeline;

    std::vector<Shader> m_shaders;

    std::uint32_t m_currentFrame{0};
    std::uint32_t m_semaphoreIndex{0};

    vk::SampleCountFlagBits m_msaaSamples = vk::SampleCountFlagBits::e1;
    vk::raii::PipelineLayout m_pipelineLayout = nullptr;
    vk::raii::Pipeline m_opaquePipeline = nullptr;
    vk::raii::Pipeline m_transparentPipeline = nullptr;

    vk::raii::Image m_colorImage = nullptr;
    vk::raii::DeviceMemory m_colorImageMemory = nullptr;
    vk::raii::ImageView m_colorImageView = nullptr;

    std::vector<vk::raii::Image> m_resolveImages;
    std::vector<vk::raii::DeviceMemory> m_resolveImageMemories;
    std::vector<vk::raii::ImageView> m_resolveImageViews;

    vk::raii::Image m_depthImage = nullptr;
    vk::raii::DeviceMemory m_depthImageMemory = nullptr;
    vk::raii::ImageView m_depthImageView = nullptr;

    std::vector<vk::raii::Semaphore> m_presentationCompleteSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::raii::Fence> m_inFlightFences;

    void createShaderModules();
    void pickMsaaSamples();
    void createGraphicsPipeline();
    void createColorResources();
    void createResolveResources();
    void createDepthResources();
    void initializeImageLayouts();
    void createSyncObjects();

    void recordCommandBuffer(const Scene& scene, std::uint32_t imageIndex);
};
