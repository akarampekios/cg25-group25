#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>

#include "VulkanCore.hpp"
#include "ResourceManager.hpp"
#include "SwapChain.hpp"
#include "ImageManager.hpp"
#include "BufferManager.hpp"
#include "Shader.hpp"

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
                             const vk::raii::ImageView& velocityImageView,  // TAA: velocity buffer
                             const vk::Image& targetImage,
                             const vk::raii::ImageView& targetImageView,
                             vk::raii::CommandBuffer const& cmd,
                             BloomParameters bloomParams,
                             uint32_t frameIndex);

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
    
    // TAA Resources
    std::unique_ptr<Shader> m_taaFragmentShader = nullptr;
    
    // TAA History buffers (double-buffered: current output becomes next frame's history)
    std::vector<vk::raii::Image> m_taaHistoryImages;
    std::vector<vk::raii::DeviceMemory> m_taaHistoryImageMemories;
    std::vector<vk::raii::ImageView> m_taaHistoryImageViews;
    
    // TAA output (anti-aliased result, fed to bloom/composite)
    std::vector<vk::raii::Image> m_taaOutputImages;
    std::vector<vk::raii::DeviceMemory> m_taaOutputImageMemories;
    std::vector<vk::raii::ImageView> m_taaOutputImageViews;
    
    vk::raii::DescriptorSetLayout m_taaDescriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> m_taaDescriptorSets;
    vk::raii::PipelineLayout m_taaPipelineLayout = nullptr;
    vk::raii::Pipeline m_taaPipeline = nullptr;
    
    // Track which history buffer to use (ping-pong)
    std::uint32_t m_taaHistoryIndex{0};
    bool m_taaFirstFrame{true};

    void createShaderModules();

    void createPipelines();

    void createImages();

    void createDescriptorPool();

    void createDescriptorSetLayouts();

    void createDescriptorSets();

    void createPipelineLayouts();
    
    // TAA: Record TAA resolve pass
    void recordTAAPass(const vk::raii::CommandBuffer& cmd,
                       const vk::raii::ImageView& currentColorView,
                       const vk::raii::ImageView& velocityView,
                       uint32_t frameIndex);

    vk::raii::Pipeline createPostProcessPipeline(const Shader& fragmentShader, 
                                                  vk::raii::PipelineLayout& outPipelineLayout,
                                                  vk::Format targetFormat);
};
