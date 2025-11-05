#include <chrono>
#include <vector>
#include <iostream>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "constants.hpp"
#include "SharedTypes.hpp"
#include "RayQueryPipeline.hpp"
#include "VulkanCore.hpp"
#include "Shader.hpp"
#include "SwapChain.hpp"
#include "ImageManager.hpp"
#include "BufferManager.hpp"
#include "CommandManager.hpp"
#include "ResourceManager.hpp"
#include "PostProcessingStack.hpp"
#include "Scene.hpp"

RayQueryPipeline::RayQueryPipeline(VulkanCore& vulkanCore,
                                   ResourceManager& resourceManager,
                                   CommandManager& commandManager,
                                   SwapChain& swapChain,
                                   ImageManager& imageManager,
                                   BufferManager& bufferManager,
                                   PostProcessingStack& postProcessingPipeline)
    : m_vulkanCore(vulkanCore),
      m_resourceManager(resourceManager),
      m_commandManager(commandManager),
      m_swapChain{swapChain},
      m_imageManager{imageManager},
      m_bufferManager{bufferManager},
      m_postProcessingPipeline{postProcessingPipeline} {
    createShaderModules();
    pickMsaaSamples();
    createGraphicsPipeline();
    createColorResources();
    createResolveResources();
    createDepthResources();
    createSyncObjects();
}

void RayQueryPipeline::createShaderModules() {
    m_shaders.emplace_back(m_vulkanCore.device(), vk::ShaderStageFlagBits::eVertex, "shaders/vertex_shader.vert.spv");
    m_shaders.emplace_back(m_vulkanCore.device(), vk::ShaderStageFlagBits::eFragment, "shaders/fragment_shader.frag.spv");
}

void RayQueryPipeline::pickMsaaSamples() {
    m_msaaSamples = m_vulkanCore.findMsaaSamples();
}


void RayQueryPipeline::createGraphicsPipeline() {
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    for (const auto& shader : m_shaders) {
        shaderStages.push_back(shader.getStage());
    }

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    vk::PipelineDynamicStateCreateInfo const dynamicState{
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    auto [globalLayout, materialLayout, lightingLayout] = m_resourceManager.getDescriptorSetLayouts();
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts = {
        globalLayout,
        materialLayout,
        lightingLayout,
    };

    const vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size()),
        .pSetLayouts = descriptorSetLayouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    m_pipelineLayout = vk::raii::PipelineLayout(m_vulkanCore.device(), pipelineLayoutInfo);

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    const vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data()
    };

    constexpr vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = false,
    };

    constexpr vk::PipelineViewportStateCreateInfo viewportState{
        .viewportCount = 1,
        .scissorCount = 1
    };

    constexpr vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .depthBiasSlopeFactor = 1.0F,
        .lineWidth = 1.0F,
    };

    vk::PipelineMultisampleStateCreateInfo const multisampling{
        .rasterizationSamples = m_msaaSamples,
        .sampleShadingEnable = vk::False,
    };

    // Opaque blend attachment - no blending
    vk::PipelineColorBlendAttachmentState opaqueBlendAttachment{
        .blendEnable = vk::False,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo opaqueBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &opaqueBlendAttachment,
    };

    // Transparent blend attachment - alpha blending enabled
    vk::PipelineColorBlendAttachmentState transparentBlendAttachment{
        .blendEnable = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp = vk::BlendOp::eAdd,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                          | vk::ColorComponentFlagBits::eG
                          | vk::ColorComponentFlagBits::eB
                          | vk::ColorComponentFlagBits::eA,
    };

    vk::PipelineColorBlendStateCreateInfo transparentBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = 1,
        .pAttachments = &transparentBlendAttachment,
    };

    auto swapChainFormat = m_swapChain.getFormat();
    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &swapChainFormat,
        .depthAttachmentFormat = m_vulkanCore.findDepthFormat(),
    };

    // Opaque depth stencil - depth writes enabled
    vk::PipelineDepthStencilStateCreateInfo opaqueDepthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLessOrEqual,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    // Transparent depth stencil - depth writes disabled, depth test enabled
    vk::PipelineDepthStencilStateCreateInfo transparentDepthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eLessOrEqual,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    // Create opaque pipeline
    vk::GraphicsPipelineCreateInfo opaquePipelineInfo{
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &opaqueDepthStencil,
        .pColorBlendState = &opaqueBlending,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
        .renderPass = nullptr, // enable dynamic rendering
    };

    m_opaquePipeline = vk::raii::Pipeline(m_vulkanCore.device(), nullptr, opaquePipelineInfo);

    // Create transparent pipeline
    vk::GraphicsPipelineCreateInfo transparentPipelineInfo{
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &transparentDepthStencil,
        .pColorBlendState = &transparentBlending,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
        .renderPass = nullptr, // enable dynamic rendering
    };

    m_transparentPipeline = vk::raii::Pipeline(m_vulkanCore.device(), nullptr, transparentPipelineInfo);
}

void RayQueryPipeline::createColorResources() {
    m_imageManager.createImage(
        m_swapChain.getExtent().width,
        m_swapChain.getExtent().height,
        1,
        m_msaaSamples,
        m_swapChain.getFormat(),
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_colorImage,
        m_colorImageMemory
        );

    m_colorImageView = m_imageManager.createImageView(m_colorImage, m_swapChain.getFormat(),
                                                      vk::ImageAspectFlagBits::eColor, 1);
}

void RayQueryPipeline::createResolveResources() {
    m_resolveImages.clear();
    m_resolveImageMemories.clear();
    m_resolveImageViews.clear();

    const auto extent = m_swapChain.getExtent();
    const auto format = m_swapChain.getFormat();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Image resolveImage{nullptr};
        vk::raii::DeviceMemory resolveImageMemory{nullptr};
    
        m_imageManager.createImage(
            extent.width,
            extent.height,
            1,
            vk::SampleCountFlagBits::e1,
            format,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            resolveImage,
            resolveImageMemory
            );

        auto resolveImageView = m_imageManager.createImageView(
            resolveImage,
            format,
            vk::ImageAspectFlagBits::eColor,
            1
            );

        m_resolveImages.push_back(std::move(resolveImage));
        m_resolveImageMemories.push_back(std::move(resolveImageMemory));
        m_resolveImageViews.push_back(std::move(resolveImageView));
    }
}

void RayQueryPipeline::createDepthResources() {
    const vk::Format depthFormat = m_vulkanCore.findDepthFormat();

    m_imageManager.createImage(
        m_swapChain.getExtent().width,
        m_swapChain.getExtent().height,
        1,
        m_msaaSamples,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_depthImage,
        m_depthImageMemory
        );

    m_depthImageView = m_imageManager.createImageView(
        m_depthImage,
        depthFormat,
        vk::ImageAspectFlagBits::eDepth,
        1
        );
}


void RayQueryPipeline::createSyncObjects() {
    m_presentationCompleteSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    const auto imageCount = m_swapChain.getImages().size();
    
    // One acquire semaphore per image - signaled when that image is available from presentation
    for (std::size_t i = 0; i < imageCount; i++) {
        m_presentationCompleteSemaphores.emplace_back(m_vulkanCore.device(), vk::SemaphoreCreateInfo());
    }
    
    // One render finished semaphore per image - signaled when rendering to that image is complete
    for (std::size_t i = 0; i < imageCount; i++) {
        m_renderFinishedSemaphores.emplace_back(m_vulkanCore.device(), vk::SemaphoreCreateInfo());
    }

    // One fence per frame in flight
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_inFlightFences.emplace_back(m_vulkanCore.device(),
                                      vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

void RayQueryPipeline::recordCommandBuffer(const Scene& scene, const std::uint32_t imageIndex) {
    const auto& cmd = m_commandManager.getCommandBuffer(m_currentFrame);

    cmd.begin({});

    // Update TLAS if scene has animated objects - this happens BEFORE rendering
    // so the updated acceleration structure is ready for ray queries
    m_resourceManager.recordTLASUpdate(*cmd, scene, false);
    
    // transition multisampled color image
    m_imageManager.transitionImageLayout(
        m_colorImage,
        cmd,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    // transition depth image
    m_imageManager.transitionImageLayout(
        m_depthImage,
        cmd,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth
        );

    constexpr vk::ClearValue clearColor = vk::ClearColorValue(0.0F, 0.0F, 0.0F, 1.0F);
    constexpr vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0F, 0);

    const vk::RenderingAttachmentInfo colorAttachmentInfo = {
        .imageView = m_colorImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .resolveMode = vk::ResolveModeFlagBits::eAverage,
        .resolveImageView = m_resolveImageViews[m_currentFrame],
        .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };

    const vk::RenderingAttachmentInfo depthAttachmentInfo = {
        .imageView = m_depthImageView,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth
    };

    const vk::RenderingInfo renderingInfo = {
        .renderArea = {
            .offset = {.x = 0, .y = 0},
            .extent = m_swapChain.getExtent(),
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo,
    };

    cmd.beginRendering(renderingInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_opaquePipeline);

    const auto swapChainExtent = m_swapChain.getExtent();
    cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                                    static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));

    // Bind vertex and index buffers (only need the buffer handles, not memory)
    const auto [vertexBuffer, _] = m_resourceManager.getVertexBuffer();
    const auto [indexBuffer, __] = m_resourceManager.getIndexBuffer();
    cmd.bindVertexBuffers(0, *vertexBuffer, {0});
    cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);

    // Bind all descriptor sets in a single call - more efficient than 3 separate calls
    const auto descriptorSets = m_resourceManager.getDescriptorSets();
    const std::array<vk::DescriptorSet, 3> allDescriptorSets = {
        *descriptorSets.globalSets[m_currentFrame],
        *descriptorSets.materialSets[m_currentFrame],
        *descriptorSets.lightSets[m_currentFrame]
    };
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, 
        m_pipelineLayout, 
        0,  // first set
        allDescriptorSets,
        nullptr  // dynamic offsets
    );

    // Get indirect draw buffer and draw counts
    const auto [indirectBuffer, ___] = m_resourceManager.getIndirectDrawBuffer(m_currentFrame);
    const std::uint32_t opaqueDrawCount = m_resourceManager.getOpaqueDrawCount();
    const std::uint32_t transparentDrawCount = m_resourceManager.getTransparentDrawCount();
    
    // Early exit optimization: if nothing to draw, skip binding and draw calls
    if (opaqueDrawCount == 0 && transparentDrawCount == 0) {
        cmd.endRendering();
        // Continue to post-processing even with empty scene
    } else {
        const vk::DeviceSize transparentOffset = m_resourceManager.getTransparentDrawOffset();

        // First pass: Render ALL opaque objects with a SINGLE multi-draw indirect call!
        if (opaqueDrawCount > 0) {
            cmd.drawIndexedIndirect(
                *indirectBuffer,
                0, // offset = 0 (opaque commands start at the beginning)
                opaqueDrawCount, // draw all opaque commands at once
                sizeof(DrawIndexedIndirectCommand) // stride between commands
            );
        }

        // Second pass: Render ALL transparent objects with a SINGLE multi-draw indirect call!
        if (transparentDrawCount > 0) {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_transparentPipeline);
            
            cmd.drawIndexedIndirect(
                *indirectBuffer,
                transparentOffset, // offset = after opaque commands
                transparentDrawCount, // draw all transparent commands at once
                sizeof(DrawIndexedIndirectCommand) // stride between commands
            );
        }

        cmd.endRendering();
    }

    // Transition resolved image
    m_imageManager.transitionImageLayout(
        m_resolveImages[m_currentFrame],
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor
    );

    // transition swap chain image
    m_imageManager.transitionImageLayout(
        m_swapChain.getImage(imageIndex),
        cmd,
        vk::ImageLayout::eUndefined,  // Don't preserve contents
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eNone,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,  // Matches semaphore wait stage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    // Post processing pass
    m_postProcessingPipeline.recordCommandBuffer(
        m_resolveImages[m_currentFrame],
        m_resolveImageViews[m_currentFrame],
        m_swapChain.getImage(imageIndex),
        m_swapChain.getImageView(imageIndex),
        cmd,
        scene.bloom,
        m_currentFrame
    );

    // Transition swap chain image for presentation
    m_imageManager.transitionImageLayout(
        m_swapChain.getImage(imageIndex),
        cmd,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eNone,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor
        );


    cmd.end();
}

void RayQueryPipeline::drawFrame(const Scene& scene) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float>(currentTime - startTime).count();
    
    // Wait for the current frame's fence (ensures we don't have more than MAX_FRAMES_IN_FLIGHT in flight)
    while (vk::Result::eTimeout == m_vulkanCore.device().waitForFences(*m_inFlightFences[m_currentFrame], vk::True,
        UINT64_MAX)) {
        // wait
    }

    // Acquire the next available swap chain image
    // We use m_semaphoreIndex to rotate through acquire semaphores
    auto [result, imageIndex] = m_swapChain.getSwapChain().acquireNextImage(
        UINT64_MAX, *m_presentationCompleteSemaphores[m_semaphoreIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) {
        // recreateSwapChain();
        return;
    }

    m_resourceManager.updateSceneResources(scene, time, m_currentFrame);

    // Update post-processing descriptor sets with the current frame's resolve image
    m_postProcessingPipeline.updateDescriptorSets(m_resolveImageViews[m_currentFrame], m_currentFrame);

    m_vulkanCore.device().resetFences(*m_inFlightFences[m_currentFrame]);


    const auto& cmd = m_commandManager.getCommandBuffer(m_currentFrame);

    cmd.reset();
    recordCommandBuffer(scene, imageIndex);

    constexpr vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);

    // Wait on the acquire semaphore - this ensures the presentation engine has released the image
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_presentationCompleteSemaphores[m_semaphoreIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*m_renderFinishedSemaphores[imageIndex],
    };

    m_vulkanCore.graphicsQueue().submit(submitInfo, *m_inFlightFences[m_currentFrame]);

    vk::PresentInfoKHR const presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*m_swapChain.getSwapChain(),
        .pImageIndices = &imageIndex,
    };

    auto presentationResult = m_vulkanCore.presentQueue().presentKHR(presentInfoKHR);

    switch (presentationResult) {
        case vk::Result::eSuccess:
            break;
        // TODO: implement swap chain recreation!
        case vk::Result::eSuboptimalKHR:
            std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
            break;
        default:
            break;
    }

    m_semaphoreIndex = (m_semaphoreIndex + 1) % m_presentationCompleteSemaphores.size();
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
