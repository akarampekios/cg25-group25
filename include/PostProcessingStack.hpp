#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

#include "VulkanCore.hpp"
#include "ResourceManager.hpp"
#include "SwapChain.hpp"
#include "ImageManager.hpp"
#include "BufferManager.hpp"
#include "SharedTypes.hpp"
#include "Shader.hpp"
#include "FSR2Pass.hpp"

class PostProcessingStack {
public:
    PostProcessingStack(VulkanCore& vulkanCore,
                        ResourceManager& resourceManager,
                        SwapChain& swapChain,
                        ImageManager& imageManager,
                        BufferManager& bufferManager);

    ~PostProcessingStack() = default;

    void recordCommandBuffer(const vk::raii::Image& resolvedImage,
                             const vk::raii::ImageView& resolvedImageView,
                             const vk::raii::Image& depthImage, const vk::raii::ImageView& depthImageView,
                             const vk::raii::Image& velocityImage, const vk::raii::ImageView& velocityImageView,  
                             const vk::Image& targetImage,
                             const vk::raii::ImageView& targetImageView,
                             vk::raii::CommandBuffer const& cmd,
                             BloomParameters bloomParams,
                             uint32_t frameIndex,
                             float deltaTime, float nearPlane, float farPlane, float fov, glm::vec2 jitter,
                             vk::Extent2D renderExtent);

    void updateDescriptorSets(const vk::raii::ImageView& resolvedImageView, 
                              const vk::raii::ImageView& velocityImageView,  // TAA: velocity buffer
                              uint32_t frameIndex);

private:
    VulkanCore& m_vulkanCore;
    ResourceManager& m_resourceManager;
    SwapChain& m_swapChain;
    ImageManager& m_imageManager;
    BufferManager& m_bufferManager;

    vk::raii::DescriptorPool m_descriptorPool = nullptr;

    // HDR Images - receive output from HDR transfer shader pass
    std::vector<vk::raii::Image> m_hdrImages;
    std::vector<vk::raii::DeviceMemory> m_hdrImageMemories;
    std::vector<vk::raii::ImageView> m_hdrImageViews;

    // Shaders
    std::unique_ptr<Shader> m_fullscreenVertexShader = nullptr;
    std::unique_ptr<Shader> m_hdrFragmentShader = nullptr;
    std::unique_ptr<Shader> m_brightPassFragmentShader = nullptr;
    std::unique_ptr<Shader> m_blurFragmentShader = nullptr;
    std::unique_ptr<Shader> m_compositeFragmentShader = nullptr;

    // Bright pass
    std::vector<vk::raii::Image> m_brightPassImages;
    std::vector<vk::raii::DeviceMemory> m_brightPassImageMemories;
    std::vector<vk::raii::ImageView> m_brightPassImageViews;
    vk::raii::DescriptorSetLayout m_brightPassDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_hdrTransferDescriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> m_hdrTransferDescriptorSets;
    std::vector<vk::raii::DescriptorSet> m_brightPassDescriptorSets;
    std::vector<vk::raii::DescriptorSet> m_compositeDescriptorSets;
    vk::raii::PipelineLayout m_hdrTransferPipelineLayout = nullptr;
    vk::raii::PipelineLayout m_brightPassPipelineLayout = nullptr;
    vk::raii::Pipeline m_hdrTransferPipeline = nullptr;
    vk::raii::Pipeline m_brightPassPipeline = nullptr;

    // Blur stuff
    std::array<std::vector<vk::raii::Image>, 2> m_blurImages;
    std::array<std::vector<vk::raii::ImageView>, 2> m_blurImageViews;
    std::array<std::vector<vk::raii::DeviceMemory>, 2> m_blurImageMemories;
    vk::raii::DescriptorSetLayout m_blurDescriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> m_blurDescriptorSets;
    vk::raii::PipelineLayout m_blurPipelineLayout = nullptr;
    vk::raii::Pipeline m_blurPipeline = nullptr;

    // Composite
    vk::raii::DescriptorSetLayout m_compositeDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSet m_compositeDescriptorSet = nullptr;
    vk::raii::PipelineLayout m_compositePipelineLayout = nullptr;
    vk::raii::Pipeline m_compositePipeline = nullptr;

    vk::raii::Sampler m_sampler = nullptr;
    
    // FSR 2
    FSR2Pass m_fsr2Pass;
    
    // Output of FSR 2 (Display Resolution) - replaces TAA output
    std::vector<vk::raii::Image> m_fsr2OutputImages;
    std::vector<vk::raii::DeviceMemory> m_fsr2OutputImageMemories;
    std::vector<vk::raii::ImageView> m_fsr2OutputImageViews;
    
    // TAA Resources - REMOVED/DEPRECATED
    // ... (Old TAA members removed)

    void createShaderModules();

    void createPipelines();

    void createImages();

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void createDescriptorSets();

    void createPipelineLayouts();
    
    // FSR 2 Dispatch
    void recordFSR2Pass(const vk::raii::CommandBuffer& cmd,
                       const vk::raii::Image& colorImage, const vk::raii::ImageView& colorView,
                       const vk::raii::Image& depthImage, const vk::raii::ImageView& depthView,
                       const vk::raii::Image& velocityImage, const vk::raii::ImageView& velocityView,
                       uint32_t frameIndex,
                       float deltaTime, float nearPlane, float farPlane, float fov,
                       glm::vec2 jitterOffset);

    vk::raii::Pipeline createPostProcessPipeline(const Shader& fragmentShader, 
                                                  vk::raii::PipelineLayout& outPipelineLayout,
                                                  vk::Format targetFormat);
};
