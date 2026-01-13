#include <iostream>
#include <array>

#include "PostProcessingStack.hpp"
#include "VulkanCore.hpp"
#include "SwapChain.hpp"
#include "ImageManager.hpp"
#include "BufferManager.hpp"
#include "Shader.hpp"
#include "SharedTypes.hpp"
#include "constants.hpp"

static const std::uint32_t RESOLVED_IMAGE_BINDING = 0;
static const std::uint32_t BLOOM_BINDING = 1;

PostProcessingStack::PostProcessingStack(VulkanCore& vulkanCore,
                                         ResourceManager& resourceManager,
                                         SwapChain& swapChain,
                                         ImageManager& imageManager,
                                         BufferManager& bufferManager) 
    : m_vulkanCore(vulkanCore),
      m_resourceManager(resourceManager),
      m_swapChain(swapChain),
      m_imageManager(imageManager),
      m_bufferManager(bufferManager),
      m_fsr2Pass(vulkanCore) {
    createShaderModules();
    createImages();
    createDescriptorPool();
    createDescriptorSetLayouts();
    createDescriptorSets();
    createPipelineLayouts();
    createPipelines();
}

void PostProcessingStack::createShaderModules() {
    m_fullscreenVertexShader = std::make_unique<Shader>(m_vulkanCore.device(), vk::ShaderStageFlagBits::eVertex,
                                                        "shaders/postprocessing/fullscreen.vert.spv");
    m_hdrFragmentShader = std::make_unique<Shader>(m_vulkanCore.device(), vk::ShaderStageFlagBits::eFragment,
                                                   "shaders/postprocessing/hdr.frag.spv");
    m_brightPassFragmentShader = std::make_unique<Shader>(m_vulkanCore.device(), vk::ShaderStageFlagBits::eFragment,
                                                          "shaders/postprocessing/bright_pass.frag.spv");
    m_blurFragmentShader = std::make_unique<Shader>(m_vulkanCore.device(), vk::ShaderStageFlagBits::eFragment,
                                                    "shaders/postprocessing/gaussian_blur.frag.spv");
    m_compositeFragmentShader = std::make_unique<Shader>(m_vulkanCore.device(), vk::ShaderStageFlagBits::eFragment,
                                                         "shaders/postprocessing/composite.frag.spv");
    
}

void PostProcessingStack::createPipelines() {
    // TAA pipeline removed.
    
    m_hdrTransferPipeline = createPostProcessPipeline(*m_hdrFragmentShader, m_hdrTransferPipelineLayout, POST_PROCESSING_IMAGE_FORMAT);
    m_brightPassPipeline = createPostProcessPipeline(*m_brightPassFragmentShader, m_brightPassPipelineLayout, POST_PROCESSING_IMAGE_FORMAT);
    m_blurPipeline = createPostProcessPipeline(*m_blurFragmentShader, m_blurPipelineLayout, POST_PROCESSING_IMAGE_FORMAT);
    m_compositePipeline = createPostProcessPipeline(*m_compositeFragmentShader, m_compositePipelineLayout, m_swapChain.getFormat());
}

void PostProcessingStack::createImages() {
    const auto extent = m_swapChain.getExtent();
    m_sampler = m_imageManager.createPostProcessingSampler();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // FSR 2 Output Images
        vk::raii::Image fsr2OutputImage{nullptr};
        vk::raii::DeviceMemory fsr2OutputMemory{nullptr};

        // FSR 2 requires UNORDERED_ACCESS (Storage) and SAMPLED (for Bloom/Composite) 
        // We also use it as TransferSrc/Dst for transitions if needed
        m_imageManager.createImage(
            extent.width,
            extent.height,
            1,
            vk::SampleCountFlagBits::e1,
            POST_PROCESSING_IMAGE_FORMAT,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            fsr2OutputImage,
            fsr2OutputMemory);

        auto fsr2OutputView = m_imageManager.createImageView(
            fsr2OutputImage,
            POST_PROCESSING_IMAGE_FORMAT,
            vk::ImageAspectFlagBits::eColor,
            1);

        m_fsr2OutputImages.emplace_back(std::move(fsr2OutputImage));
        m_fsr2OutputImageMemories.emplace_back(std::move(fsr2OutputMemory));
        m_fsr2OutputImageViews.emplace_back(std::move(fsr2OutputView));
        
        // HDR Image - receives output from HDR transfer pass
        vk::raii::Image hdrImage{nullptr};
        vk::raii::DeviceMemory hdrImageMemory{nullptr};

        m_imageManager.createImage(
            extent.width,
            extent.height,
            1,
            vk::SampleCountFlagBits::e1,
            POST_PROCESSING_IMAGE_FORMAT,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            hdrImage,
            hdrImageMemory);

        auto hdrImageView = m_imageManager.createImageView(
            hdrImage,
            POST_PROCESSING_IMAGE_FORMAT,
            vk::ImageAspectFlagBits::eColor,
            1
            );

        m_hdrImages.emplace_back(std::move(hdrImage));
        m_hdrImageViews.emplace_back(std::move(hdrImageView));
        m_hdrImageMemories.emplace_back(std::move(hdrImageMemory));

        // Bright Pass Image
        vk::raii::Image brightPassImage{nullptr};
        vk::raii::DeviceMemory brightPassImageMemory{nullptr};

        m_imageManager.createImage(
            extent.width,
            extent.height,
            1,
            vk::SampleCountFlagBits::e1,
            POST_PROCESSING_IMAGE_FORMAT,  // Use HDR format, not swap chain format
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            brightPassImage,
            brightPassImageMemory);

        auto brightPassImageView = m_imageManager.createImageView(
            brightPassImage,
            POST_PROCESSING_IMAGE_FORMAT,  // Use HDR format, not swap chain format
            vk::ImageAspectFlagBits::eColor,
            1
            );

        m_brightPassImages.emplace_back(std::move(brightPassImage));
        m_brightPassImageViews.emplace_back(std::move(brightPassImageView));
        m_brightPassImageMemories.emplace_back(std::move(brightPassImageMemory));

        // Blur pass images
        for (std::size_t pass = 0; pass < 2; pass++) {
            vk::raii::Image blurImage{nullptr};
            vk::raii::DeviceMemory blurImageMemory{nullptr};

            m_imageManager.createImage(
                extent.width,
                extent.height,
                1,
                vk::SampleCountFlagBits::e1,
                POST_PROCESSING_IMAGE_FORMAT,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                blurImage,
                blurImageMemory);

            auto blurImageView = m_imageManager.createImageView(
                blurImage,
                POST_PROCESSING_IMAGE_FORMAT,
                vk::ImageAspectFlagBits::eColor,
                1
                );

            m_blurImages[pass].emplace_back(std::move(blurImage));
            m_blurImageViews[pass].emplace_back(std::move(blurImageView));
            m_blurImageMemories[pass].emplace_back(std::move(blurImageMemory));
        }
    }
}

void PostProcessingStack::createDescriptorPool() {
    // TAA descriptor counts removed.
    
    std::array poolSizes = {
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT * 4 + POST_PROCESSING_BLUR_STAGES + 3, // + taaDescriptorCount,
        },
    };

    const vk::DescriptorPoolCreateInfo poolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_FRAMES_IN_FLIGHT * 5 + POST_PROCESSING_BLUR_STAGES + 2, // + taaSetsCount,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };

    m_descriptorPool = vk::raii::DescriptorPool(m_vulkanCore.device(), poolCreateInfo);
}

void PostProcessingStack::createDescriptorSetLayouts() {
    constexpr vk::DescriptorSetLayoutBinding hdrTraansferResolvedImageBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array hdrTransferBindings = {hdrTraansferResolvedImageBinding};

    const vk::DescriptorSetLayoutCreateInfo hdrTransferLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(hdrTransferBindings.size()),
        .pBindings = hdrTransferBindings.data(),
    };

    m_hdrTransferDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        hdrTransferLayoutCreateInfo
        );

    constexpr vk::DescriptorSetLayoutBinding brightPassHdrImageBinding{
        .binding = RESOLVED_IMAGE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array brightPassBindings = {brightPassHdrImageBinding};

    const vk::DescriptorSetLayoutCreateInfo brightPassLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(brightPassBindings.size()),
        .pBindings = brightPassBindings.data(),
    };

    m_brightPassDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        brightPassLayoutCreateInfo
        );

    constexpr vk::DescriptorSetLayoutBinding blurImageBinding{
        .binding = RESOLVED_IMAGE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array blurBindings = {blurImageBinding};

    const vk::DescriptorSetLayoutCreateInfo blurLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(blurBindings.size()),
        .pBindings = blurBindings.data(),
    };

    m_blurDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        blurLayoutCreateInfo
        );

    constexpr vk::DescriptorSetLayoutBinding compositeHdrImageBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

     constexpr vk::DescriptorSetLayoutBinding compositeBlurImageBinding{
        .binding = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array compositeBindings = {compositeHdrImageBinding, compositeBlurImageBinding};

    const vk::DescriptorSetLayoutCreateInfo compositeLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(compositeBindings.size()),
        .pBindings = compositeBindings.data(),
    };

    m_compositeDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        compositeLayoutCreateInfo
        );
    
    // TAA descriptor set layout removed.
}

void PostProcessingStack::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> hdrTransferLayouts(MAX_FRAMES_IN_FLIGHT, *m_hdrTransferDescriptorSetLayout);
    
    const vk::DescriptorSetAllocateInfo hdrTransferDescriptorSetAllocInfo{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = hdrTransferLayouts.data(),
    };

    // Allocate descriptor sets but don't initialize them yet - they'll be updated per-frame
    // with the current resolved image view
    m_hdrTransferDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(hdrTransferDescriptorSetAllocInfo);

    std::vector<vk::DescriptorSetLayout> brightPassLayouts(MAX_FRAMES_IN_FLIGHT, *m_brightPassDescriptorSetLayout);
    
    const vk::DescriptorSetAllocateInfo brightDescriptorSetAllocInfo{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = brightPassLayouts.data(),
    };

    m_brightPassDescriptorSets =  m_vulkanCore.device().allocateDescriptorSets(brightDescriptorSetAllocInfo);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorImageInfo hdrImageIinfo{
            .sampler = *m_sampler,
            .imageView = *m_hdrImageViews[i],  // Fixed: use HDR image, not HDR transfer image
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        std::array writeDescriptors = {
            vk::WriteDescriptorSet{
                .dstSet = m_brightPassDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &hdrImageIinfo,
            },
        };

        m_vulkanCore.device().updateDescriptorSets(writeDescriptors, {});
    }

    // We need descriptor sets for 2 blur passes (horizontal and vertical) for each frame
    std::vector<vk::DescriptorSetLayout> blurLayouts(MAX_FRAMES_IN_FLIGHT * 2, *m_blurDescriptorSetLayout);

    const vk::DescriptorSetAllocateInfo blurDescriptorSetAllocInfo{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT * 2,
        .pSetLayouts = blurLayouts.data(),
    };

    m_blurDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(blurDescriptorSetAllocInfo);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // Descriptor set for horizontal blur pass (reads from bright pass)
        vk::DescriptorImageInfo brightPassImageInfo{
            .sampler = *m_sampler,
            .imageView = *m_brightPassImageViews[i],
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        std::array horizontalWriteDescriptors = {
            vk::WriteDescriptorSet{
                .dstSet = m_blurDescriptorSets[i * 2 + 0], // Horizontal blur descriptor set
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &brightPassImageInfo,
            },
        };

        m_vulkanCore.device().updateDescriptorSets(horizontalWriteDescriptors, {});

        // Descriptor set for vertical blur pass (reads from horizontal blur output)
        vk::DescriptorImageInfo horizontalBlurImageInfo{
            .sampler = *m_sampler,
            .imageView = *m_blurImageViews[0][i], // Output from horizontal blur
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        std::array verticalWriteDescriptors = {
            vk::WriteDescriptorSet{
                .dstSet = m_blurDescriptorSets[i * 2 + 1], // Vertical blur descriptor set
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &horizontalBlurImageInfo,
            },
        };

        m_vulkanCore.device().updateDescriptorSets(verticalWriteDescriptors, {});
    }

    std::vector<vk::DescriptorSetLayout> compositeLayouts(MAX_FRAMES_IN_FLIGHT, *m_compositeDescriptorSetLayout);
    
    const vk::DescriptorSetAllocateInfo compositeDescriptorSetAllocInfo{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = compositeLayouts.data(),
    };

    m_compositeDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(compositeDescriptorSetAllocInfo);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vk::DescriptorImageInfo compositeHdrImageInfo{
            .sampler = *m_sampler,
            .imageView = *m_hdrImageViews[i],
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        vk::DescriptorImageInfo compositeBlurImageInfo{
            .sampler = *m_sampler,
            .imageView = *m_blurImageViews[1][i], // Fixed: use final vertical blur output
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        };

        std::array writeDescriptors = {
            vk::WriteDescriptorSet{
                .dstSet = m_compositeDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &compositeHdrImageInfo,
            },
            vk::WriteDescriptorSet{
                .dstSet = m_compositeDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &compositeBlurImageInfo,  // Fixed: use final vertical blur output
            },
        };

        m_vulkanCore.device().updateDescriptorSets(writeDescriptors, {});
    }
    
    // TAA descriptor sets removed.
}

void PostProcessingStack::createPipelineLayouts() {
    const vk::PipelineLayoutCreateInfo hdrTransferInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*m_hdrTransferDescriptorSetLayout,
    };

    m_hdrTransferPipelineLayout = vk::raii::PipelineLayout(m_vulkanCore.device(), hdrTransferInfo);

    constexpr vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(BloomPushConstant),
    };
    
    const vk::PipelineLayoutCreateInfo brightPassInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*m_brightPassDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };

    m_brightPassPipelineLayout = vk::raii::PipelineLayout(m_vulkanCore.device(), brightPassInfo);

    const vk::PipelineLayoutCreateInfo blurInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*m_blurDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };

    m_blurPipelineLayout = vk::raii::PipelineLayout(m_vulkanCore.device(), blurInfo);

    const vk::PipelineLayoutCreateInfo compositeInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*m_compositeDescriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };

    m_compositePipelineLayout = vk::raii::PipelineLayout(m_vulkanCore.device(), compositeInfo);
    
    // TAA pipeline layout removed.
}


vk::raii::Pipeline PostProcessingStack::createPostProcessPipeline(
    const Shader& fragmentShader,
    vk::raii::PipelineLayout& pipelineLayout,
    vk::Format targetFormat) {
    std::array shaderStages = {
        m_fullscreenVertexShader->getStage(),
        fragmentShader.getStage()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
    };

    constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasSlopeFactor = 1.0F,
        .lineWidth = 1.0F,
    };

    vk::PipelineMultisampleStateCreateInfo const multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = vk::False,
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment,
    };

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo const dynamicState{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    constexpr vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1
    };

    // Empty vertex input state for fullscreen triangle (no vertex buffers needed)
    constexpr vk::PipelineVertexInputStateCreateInfo vertexInputState{
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr
    };

    auto colorAttachmentFormat = targetFormat; // Use the passed target format
    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorAttachmentFormat,
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputState, // Use empty vertex input state
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = *pipelineLayout
    };

    return vk::raii::Pipeline(m_vulkanCore.device(), nullptr, pipelineInfo);
}

void PostProcessingStack::updateDescriptorSets(const vk::raii::ImageView& resolvedImageView,
                                               const vk::raii::ImageView& velocityImageView,
                                               uint32_t frameIndex) {
    std::vector<vk::WriteDescriptorSet> descriptorWrites;
    
    // All DescriptorImageInfo objects must outlive the updateDescriptorSets call
    // so we declare them at function scope
    
    // TAA descriptor infos (only used when TAA_ENABLED)
    // vk::DescriptorImageInfo taaCurrentInfo{};
    // vk::DescriptorImageInfo taaHistoryInfo{};
    // vk::DescriptorImageInfo taaVelocityInfo{};
    
    // TAA descriptor set update
    // if constexpr (TAA_ENABLED) {
    //     // Current color (from scene render)
    //     taaCurrentInfo = {
    //         .sampler = m_sampler,
    //         .imageView = resolvedImageView,
    //         .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    //     };
        
    //     // History buffer (previous frame's TAA output, or current if first frame)
    //     taaHistoryInfo = {
    //         .sampler = m_sampler,
    //         .imageView = m_taaFirstFrame ? *resolvedImageView : *m_taaHistoryImageViews[frameIndex],
    //         .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    //     };
        
    //     // Velocity buffer
    //     taaVelocityInfo = {
    // HDR transfer and bright pass descriptor set updates...

    // Update HDR transfer descriptor set
    // Use FSR 2 output image
    const vk::DescriptorImageInfo hdrTransferInfo{
        .sampler = m_sampler,
        .imageView = *m_fsr2OutputImageViews[frameIndex],
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::WriteDescriptorSet hdrTransferWrite{
        .dstSet = m_hdrTransferDescriptorSets[frameIndex],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &hdrTransferInfo,
    };

    descriptorWrites.emplace_back(hdrTransferWrite);

    // Update bright pass descriptor set
    const vk::DescriptorImageInfo brightInfo{
        .sampler = m_sampler,
        .imageView = *m_fsr2OutputImageViews[frameIndex],
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    vk::WriteDescriptorSet brightWrite{
        .dstSet = m_brightPassDescriptorSets[frameIndex],
        .dstBinding = RESOLVED_IMAGE_BINDING,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &brightInfo,
    };

    descriptorWrites.emplace_back(brightWrite);
    m_vulkanCore.device().updateDescriptorSets(descriptorWrites, {});
}

void PostProcessingStack::recordCommandBuffer(const vk::raii::Image& resolvedImage,
                                              const vk::raii::ImageView& resolvedImageView,
                                              const vk::raii::Image& depthImage, const vk::raii::ImageView& depthImageView,
                                              const vk::raii::Image& velocityImage, const vk::raii::ImageView& velocityImageView,
                                              const vk::Image& targetImage,
                                              const vk::raii::ImageView& targetImageView,
                                              vk::raii::CommandBuffer const& cmd,
                                              BloomParameters bloomParams,
                                              uint32_t frameIndex,
                                              float deltaTime, float nearPlane, float farPlane, float fov, glm::vec2 jitter,
                                              vk::Extent2D renderExtent) {

    const auto extent = m_swapChain.getExtent();

    // FSR 2 Pass
    
    // Resize FSR 2 context if needed - render resolution is lower than display resolution
    // renderExtent = render resolution (e.g. 67% of display for Quality mode)
    // extent = display resolution (swapchain size)
    m_fsr2Pass.onResize(renderExtent.width, renderExtent.height, extent.width, extent.height);
    
    // FSR 2 Output image needs to be in General layout for Compute Write
    // FSR 2 usually handles internal transitions but we should provide it in a usable state if needed
    // Assuming FSR 2 library handles internal barriers for its resources.
    
    // Call FSR 2
    recordFSR2Pass(cmd, resolvedImage, resolvedImageView, 
                   depthImage, depthImageView, 
                   velocityImage, velocityImageView, 
                   frameIndex, deltaTime, nearPlane, farPlane, fov, jitter);
                   
    // Transition FSR 2 Output to Shader Read for subsequent passes (HDR/Bloom)
    // FSR 2 dispatch leaves output in UNORDERED_ACCESS state usually.
    m_imageManager.transitionImageLayout(
        m_fsr2OutputImages[frameIndex],
        cmd,
        vk::ImageLayout::eUndefined, // Don't care previous
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eShaderWrite, // Compute write
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
    );

    // Note: On first frame, images transition from Undefined. On subsequent frames,
    // they're already in the correct layout. We use CLEAR load op which makes
    // preserving old contents unnecessary, so Undefined is safe and fast.
    
    // Note: When TAA is enabled, resolved image is already in ShaderReadOnlyOptimal
    // from the RayQueryPipeline. When TAA is disabled, we need to transition it.

    // Render resolved image to internal HDR image
    const vk::RenderingAttachmentInfo hdrTransferColorAttachmentInfo = {
        .imageView = *m_hdrImageViews[frameIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F)
    };

    const vk::RenderingInfo hdrTransferRenderingInfo = {
        .renderArea = {
            .offset = {.x = 0, .y = 0},
            .extent = extent,
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &hdrTransferColorAttachmentInfo,
    };

    cmd.beginRendering(hdrTransferRenderingInfo);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_hdrTransferPipeline);
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width),
                                    static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_hdrTransferPipelineLayout, 0, *m_hdrTransferDescriptorSets[frameIndex],
                           {});
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Transition HDR image to shader read layout
    m_imageManager.transitionImageLayout(
        m_hdrImages[frameIndex],
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
        );

    // Prepare bright pass image for rendering
    // Note: Image persists between frames, transitioned back to ColorAttachment at end of previous frame
    m_imageManager.transitionImageLayout(
        m_brightPassImages[frameIndex],
        cmd,
        vk::ImageLayout::eShaderReadOnlyOptimal,  // From previous frame's final layout
        vk::ImageLayout::eColorAttachmentOptimal, 
        vk::AccessFlagBits2::eShaderRead,  // From previous use
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,  // From previous stage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    // Render bright pass
    const vk::RenderingAttachmentInfo colorAttachmentInfo = {
        .imageView = *m_brightPassImageViews[frameIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F)
    };

    const vk::RenderingInfo brightPassRendering = {
        .renderArea = {
            .offset = {.x = 0, .y = 0},
            .extent = extent,
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentInfo,
    };

    // Prepare push constants for bright pass
    BloomPushConstant bloomPushConstant {
        .textureSize = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height)),
        .direction = glm::vec2(1.0f, 0.0f),
        .blurStrength = bloomParams.blurStrength,
        .exposure = bloomParams.exposure,
        .threshold = bloomParams.threshold,
        .scale = bloomParams.scale,
    };

    cmd.beginRendering(brightPassRendering);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_brightPassPipeline);
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width),
                                    static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_brightPassPipelineLayout, 0, *m_brightPassDescriptorSets[frameIndex],
                           {});
    cmd.pushConstants<BloomPushConstant>(*m_brightPassPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, bloomPushConstant);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Transition bright pass image to shader read layout for blur pass
    m_imageManager.transitionImageLayout(
        m_brightPassImages[frameIndex],
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
        );

    // Transition the first (horizontal) blur pass image for rendering
    // Note: Blur images persist between frames
    m_imageManager.transitionImageLayout(
        *m_blurImages[0][frameIndex],
        cmd,
        vk::ImageLayout::eShaderReadOnlyOptimal,  // From previous frame
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eShaderRead,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    const vk::RenderingAttachmentInfo horizontalBlurAttachment = {
        .imageView = *m_blurImageViews[0][frameIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F)
    };

    const vk::RenderingInfo horizontalBlurRendering = {
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &horizontalBlurAttachment,
    };

    bloomPushConstant.direction = glm::vec2(1.0f, 0.0f); // Horizontal
    
    cmd.beginRendering(horizontalBlurRendering);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_blurPipeline);
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width),
                                    static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_blurPipelineLayout, 0, *m_blurDescriptorSets[frameIndex * 2 + 0], {});
    cmd.pushConstants<BloomPushConstant>(*m_blurPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, bloomPushConstant);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // transition the resulring horizontal blur image for vertical blur rendering
    m_imageManager.transitionImageLayout(
        m_blurImages[0][frameIndex],
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
        );

    // Prepare vertical blur image for rendering
    m_imageManager.transitionImageLayout(
        m_blurImages[1][frameIndex],
        cmd,
        vk::ImageLayout::eShaderReadOnlyOptimal,  // From previous frame
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eShaderRead,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    const vk::RenderingAttachmentInfo verticalBlurAttachment = {
        .imageView = *m_blurImageViews[1][frameIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F)
    };

    const vk::RenderingInfo verticalBlurRendering = {
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &verticalBlurAttachment,
    };

    bloomPushConstant.direction = glm::vec2(0.0f, 1.0f); // Vertical

    cmd.beginRendering(verticalBlurRendering);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_blurPipeline);
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width),
                                    static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_blurPipelineLayout, 0, *m_blurDescriptorSets[frameIndex * 2 + 1], {});
    cmd.pushConstants<BloomPushConstant>(*m_blurPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, bloomPushConstant);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();

    // Transition resulting vertical blur image to shader read layout for composite
    m_imageManager.transitionImageLayout(
        m_blurImages[1][frameIndex],
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor
        );

    const vk::RenderingAttachmentInfo compositeAttachmentInfo = {
        .imageView = targetImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F)
    };

    const vk::RenderingInfo compositeRendering = {
        .renderArea = {.offset = {.x = 0, .y = 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &compositeAttachmentInfo,
    };

    cmd.beginRendering(compositeRendering);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_compositePipeline);
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), extent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(extent.width),
                                    static_cast<float>(extent.height), 0.0f, 1.0f));
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_compositePipelineLayout, 0, *m_compositeDescriptorSets[frameIndex], {});
    cmd.pushConstants<BloomPushConstant>(*m_compositePipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, bloomPushConstant);
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
    
    // TAA: Copy current TAA output to history buffer for next frame removed.
}

void PostProcessingStack::recordFSR2Pass(const vk::raii::CommandBuffer& cmd,
                                       const vk::raii::Image& colorImage, const vk::raii::ImageView& colorView,
                                       const vk::raii::Image& depthImage, const vk::raii::ImageView& depthView,
                                       const vk::raii::Image& velocityImage, const vk::raii::ImageView& velocityView,
                                       uint32_t frameIndex,
                                       float deltaTime, float nearPlane, float farPlane, float fov, glm::vec2 jitterOffset) {
    
    m_fsr2Pass.dispatch(cmd,
                        colorImage, colorView,
                        depthImage, depthView,
                        velocityImage, velocityView,
                        nullptr, // Exposure
                        m_fsr2OutputImages[frameIndex], m_fsr2OutputImageViews[frameIndex],
                        nullptr, // Reactive mask
                        nullptr, // Composition mask
                        deltaTime,
                        nearPlane, farPlane, fov,
                        jitterOffset);
}
