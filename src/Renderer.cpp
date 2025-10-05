#include "Renderer.hpp"
#include "Camera.h"
#include "loader.hpp"
#include "structs.hpp"

Renderer::Renderer(GLFWwindow* window) :
    m_window{window},
    m_windowUtils{m_window},
    m_physicalDeviceUtils{m_vulkanTransferContext},
    m_deviceUtils{m_vulkanTransferContext},
    m_swapChainUtils{m_vulkanTransferContext},
    m_imageUtils{m_vulkanTransferContext},
    m_vulkanTransferContext{.window = m_window} {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    pickMsaaSamples();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createCommandBuffers();
    createSyncObjects();
    createTextureSamplers();
    createDescriptorPool();

    loadScene();

    // createVertexBuffer();
    // createIndexBuffer();
    // createDescriptorPool();
    // createDescriptorSets();
}

Renderer::~Renderer() {
    m_device.waitIdle();
}


void Renderer::createInstance() {
    constexpr vk::ApplicationInfo appInfo{
        .pApplicationName = "Cyberpunk City Demo",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_4,
    };

    auto requiredInstanceExtensions =
        WindowUtils::getRequiredInstanceExtensions();

    const vk::InstanceCreateInfo createInfo{
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(kRequiredValidationLayers.
            size()),
        .ppEnabledLayerNames = kRequiredValidationLayers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(
            requiredInstanceExtensions.size()),
        .ppEnabledExtensionNames = requiredInstanceExtensions.data(),
    };

    m_instance = vk::raii::Instance(m_context, createInfo);
    m_vulkanTransferContext.instance = &m_instance;
}

void Renderer::createSurface() {
    VkSurfaceKHR _surface = nullptr;

    if (glfwCreateWindowSurface(*m_instance, m_window, nullptr, &_surface) !=
        0) {
        throw std::runtime_error("failed to create window surface!");
    }

    m_surface = vk::raii::SurfaceKHR(m_instance, _surface);
    m_vulkanTransferContext.surface = &m_surface;
}

void Renderer::pickPhysicalDevice() {
    const auto physicalDevices = m_instance.enumeratePhysicalDevices();

    if (physicalDevices.empty()) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    for (const auto& device : physicalDevices) {
        if (m_physicalDeviceUtils.isDeviceSuitable(device, kRequiredDeviceExtensions)) {
            m_physicalDevice = device;
            m_vulkanTransferContext.physicalDevice = &m_physicalDevice;
            break;
        }
    }

    if (m_physicalDevice == nullptr) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void Renderer::pickMsaaSamples() {
    m_msaaSamples = m_physicalDeviceUtils.findMsaaSamples();
}

void Renderer::createLogicalDevice() {
    m_queueFamilyIndices = m_physicalDeviceUtils.findQueueFamilies();

    auto featureChain = LogicalDeviceUtils::buildFeatureChain();
    auto queueCreateInfos = LogicalDeviceUtils::buildQueueInfos(
        m_queueFamilyIndices);

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
    m_vulkanTransferContext.device = &m_device;

    m_graphicsQueue = vk::raii::Queue(m_device, m_queueFamilyIndices.graphicsFamily.value(), 0);
    m_vulkanTransferContext.graphicsQueue = &m_graphicsQueue;

    m_presentQueue = vk::raii::Queue(m_device, m_queueFamilyIndices.presentFamily.value(), 0);
    m_vulkanTransferContext.presentationQueue = &m_presentQueue;
}

void Renderer::createSwapChain() {
    auto [format, colorSpace] = m_swapChainUtils.chooseSurfaceFormat();
    const auto presentationMode = m_swapChainUtils.choosePresentationMode();
    const auto transform = m_swapChainUtils.chooseTransform();
    const auto imageCount = m_swapChainUtils.chooseMinImageCount();
    const auto extent = m_swapChainUtils.chooseExtent();

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .flags = vk::SwapchainCreateFlagsKHR(),
        .surface = m_surface,
        .minImageCount = imageCount,
        .imageFormat = format,
        .imageColorSpace = colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = transform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentationMode,
        .clipped = vk::True,
        .oldSwapchain = nullptr,
    };

    if (m_queueFamilyIndices.graphicsFamily.value() != m_queueFamilyIndices.presentFamily.value()) {
        const std::vector queueFamilyIndices = {
            m_queueFamilyIndices.graphicsFamily.value(),
            m_queueFamilyIndices.presentFamily.value(),
        };

        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapChainCreateInfo.queueFamilyIndexCount = queueFamilyIndices.size();
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    }

    m_swapChainImageFormat = format;
    m_swapChainExtent = extent;

    m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);
    m_vulkanTransferContext.swapChain = &m_swapChain;

    m_swapChainImages = m_swapChain.getImages();
}

void Renderer::createImageViews() {
    m_swapChainImageViews.clear();

    vk::ImageViewCreateInfo imageViewCreateInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = m_swapChainImageFormat,
        .subresourceRange =
        {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    for (const auto& image : m_swapChainImages) {
        imageViewCreateInfo.image = image;
        m_swapChainImageViews.emplace_back(m_device, imageViewCreateInfo);
    }
}

void Renderer::createDescriptorSetLayout() {
    std::vector bindings = {
        vk::DescriptorSetLayoutBinding{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags =
                                       vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,
                                       .descriptorCount = 1,
                                       .stageFlags =
                                       vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 6, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment}
    };

    const vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };

    m_descriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_device,
        descriptorSetLayoutCreateInfo
        );

    m_vulkanTransferContext.descriptorSetLayout = &m_descriptorSetLayout;
}

void Renderer::createGraphicsPipeline() {
    auto vertShaderModule = ShaderUtils::loadShader(m_device, "vert.vert.spv");
    auto fragShaderModule = ShaderUtils::loadShader(m_device, "frag.frag.spv");

    vk::PipelineShaderStageCreateInfo const vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex,
        .module = vertShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo const fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment,
        .module = fragShaderModule,
        .pName = "main",
    };

    vk::PipelineShaderStageCreateInfo const shaderStages[] = {
        vertShaderStageInfo,
        fragShaderStageInfo,
    };

    std::vector dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };
    vk::PipelineDynamicStateCreateInfo const dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &*m_descriptorSetLayout,
        .pushConstantRangeCount = 0,
    };

    m_pipelineLayout = vk::raii::PipelineLayout(m_device, pipelineLayoutInfo);

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo const vertexInputInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions = attributeDescriptions.data(),
    };

    vk::PipelineInputAssemblyStateCreateInfo const inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
        .primitiveRestartEnable = vk::False,
    };

    vk::PipelineViewportStateCreateInfo const viewportState{
        .viewportCount = 1,
        .scissorCount = 1,
    };

    vk::PipelineRasterizationStateCreateInfo const rasterizer{
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

    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &m_swapChainImageFormat,
        .depthAttachmentFormat = m_physicalDeviceUtils.findDepthFormat(),
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext = &pipelineRenderingCreateInfo,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlending,
        .pDynamicState = &dynamicState,
        .layout = m_pipelineLayout,
        .renderPass = nullptr, // enable dynamic rendering
    };

    m_graphicsPipeline = vk::raii::Pipeline(m_device, nullptr, pipelineInfo);
}

void Renderer::createCommandPool() {
    const vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = m_queueFamilyIndices.graphicsFamily.value(),
    };

    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);
    m_vulkanTransferContext.commandPool = &m_commandPool;
}

void Renderer::createColorResources() {
    m_imageUtils.createImage(
        m_swapChainExtent.width,
        m_swapChainExtent.height,
        1,
        m_msaaSamples,
        m_swapChainImageFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_colorImage,
        m_colorImageMemory
        );

    m_colorImageView = m_imageUtils.createImageView(m_colorImage, m_swapChainImageFormat,
                                                    vk::ImageAspectFlagBits::eColor, 1);
}

void Renderer::createDepthResources() {
    const vk::Format depthFormat = m_physicalDeviceUtils.findDepthFormat();

    m_imageUtils.createImage(
        m_swapChainExtent.width,
        m_swapChainExtent.height,
        1,
        m_msaaSamples,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_depthImage,
        m_depthImageMemory
        );

    m_depthImageView = m_imageUtils.createImageView(
        m_depthImage,
        depthFormat,
        vk::ImageAspectFlagBits::eDepth,
        1
        );
}

void Renderer::createCommandBuffers() {
    m_commandBuffers.clear();

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    m_commandBuffers = vk::raii::CommandBuffers(m_device, allocInfo);
}

void Renderer::createSyncObjects() {
    m_presentationCompleteSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    for (size_t i = 0; i < m_swapChainImages.size(); i++) {
        m_presentationCompleteSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo());
        m_renderFinishedSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        m_inFlightFences.emplace_back(m_device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

void Renderer::createTextureSamplers() {
    m_baseColorTextureSampler = m_imageUtils.createSampler(true);
    m_metallicRoughnessTextureSampler = m_imageUtils.createSampler(false);
    m_normalTextureSampler = m_imageUtils.createSampler(false);
    m_emissiveTextureSampler = m_imageUtils.createSampler(true);
    m_occlusionTextureSampler = m_imageUtils.createSampler(false);

    m_vulkanTransferContext.baseColorTextureSampler = &m_baseColorTextureSampler;
    m_vulkanTransferContext.metallicRoughnessTextureSampler = &m_metallicRoughnessTextureSampler;
    m_vulkanTransferContext.normalTextureSampler = &m_normalTextureSampler;
    m_vulkanTransferContext.emissiveTextureSampler = &m_emissiveTextureSampler;
    m_vulkanTransferContext.occlusionTextureSampler = &m_occlusionTextureSampler;
}

void Renderer::createDescriptorPool() {
    constexpr vk::DescriptorPoolSize uboPoolSize(
        vk::DescriptorType::eUniformBuffer,
        MAX_FRAMES_IN_FLIGHT * MAX_SCENE_OBJECTS
        );

    constexpr vk::DescriptorPoolSize combinedImageSamplerPoolSize(
        vk::DescriptorType::eCombinedImageSampler,
        MAX_FRAMES_IN_FLIGHT * MAX_SCENE_OBJECTS * 5 // we have 5 textures per material
        );

    std::vector poolSizes = {uboPoolSize, combinedImageSamplerPoolSize};

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = MAX_FRAMES_IN_FLIGHT * MAX_SCENE_OBJECTS,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    m_descriptorPool = vk::raii::DescriptorPool(m_device, poolInfo);
    m_vulkanTransferContext.descriptorPool = &m_descriptorPool;
}

void Renderer::loadScene() {
    Loader loader(m_vulkanTransferContext);

    scene = loader.loadGltfScene("assets/helmet_with_camera.glb");
}

void Renderer::updateUniformBuffers(const Camera& camera) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    const auto currentTime = std::chrono::high_resolution_clock::now();
    const float time = std::chrono::duration<float>(currentTime - startTime).count();

    for (auto& mesh : scene.meshes) {
        const auto cameraPos = camera.getPosition();

        // const auto view = camera.getViewMatrix();
        const auto view = scene.camera.getView();
        const auto proj = scene.camera.getProjection();

        // auto model = glm::mat4(1.0f);

        // auto view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        // auto proj = glm::perspective(glm::radians(45.0f), static_cast<float>(m_swapChainExtent.width) / static_cast<float>(m_swapChainExtent.height), 0.1f, 100.0f);
        // proj[1][1] *= -1;

        glm::vec3 camPos = glm::vec3(0.0f, 0.0f, 3.0f); // 3 units in front of origin
        glm::vec3 camTarget = glm::vec3(0.0f, 0.0f, 0.0f); // look at origin
        glm::vec3 camUp = glm::vec3(0.0f, 1.0f, 0.0f); // y-up

        // auto view = glm::lookAt(camPos, camTarget, camUp);
        // auto view = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, 0.1f, 10.0f);
        // scene.camera.update(time);
        auto model = glm::rotate(mesh.model, time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        // scene.camera.rotation = rotate(scene.camera.rotation, time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        // view = rotate(view, time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        // std::cout << "CAM POS2 " << scene.camera.getPosition()[0] << "\n";
        // std::cout << "CAM POS2 " << scene.camera.getPosition()[1] << "\n";
        // std::cout << "CAM POS2 " << scene.camera.getPosition()[2] << "\n";
        // std::cout << "---- " << "\n";

        UniformBufferObject ubo{
            .model = model,
            .view = view,
            .proj = proj,
            .viewInverse = glm::inverse(view),
            .projInverse = glm::inverse(proj),
            .cameraPos = scene.camera.getPosition(),
            .time = time,
        };

        MaterialBufferObject material {
            .baseColorFactor = mesh.material->baseColorFactor,
            .metallicFactor = mesh.material->metallicFactor,
            .roughnessFactor = mesh.material->roughnessFactor,
            .emissiveFactor = mesh.material->emissiveFactor,
        };

        memcpy(mesh.uniformBuffersMapped[m_currentFrame], &ubo, sizeof(ubo));
        memcpy(mesh.materialBuffersMapped[m_currentFrame], &material, sizeof(material));
    }
}

void Renderer::recordCommandBuffer(std::uint32_t const imageIndex) {
    m_commandBuffers[m_currentFrame].begin({});

    // transition swap chain image
    m_imageUtils.transitionImageLayout(
        m_swapChainImages[imageIndex],
        m_commandBuffers[m_currentFrame],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        // do not wait for previous operations
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    // transition multisampled color image
    m_imageUtils.transitionImageLayout(
        m_colorImage,
        m_commandBuffers[m_currentFrame],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor
        );

    // transition depth image
    m_imageUtils.transitionImageLayout(
        m_depthImage,
        m_commandBuffers[m_currentFrame],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        {},
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests,
        vk::ImageAspectFlagBits::eDepth
        );

    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo colorAttachmentInfo = {
        .imageView = m_colorImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .resolveMode = vk::ResolveModeFlagBits::eAverage,
        .resolveImageView = m_swapChainImageViews[imageIndex],
        .resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };

    vk::RenderingAttachmentInfo depthAttachmentInfo = {
        .imageView = m_depthImageView,
        .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth
    };

    const vk::RenderingInfo renderingInfo = {
        .renderArea = {
            .offset = {.x = 0, .y = 0},
            .extent = m_swapChainExtent,
        },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentInfo,
        .pDepthAttachment = &depthAttachmentInfo,
    };

    m_commandBuffers[m_currentFrame].beginRendering(renderingInfo);
    m_commandBuffers[m_currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, *m_graphicsPipeline);
    m_commandBuffers[m_currentFrame].setViewport(0, vk::Viewport(
                                                     0.0f, 0.0f, static_cast<float>(m_swapChainExtent.width),
                                                     static_cast<float>(m_swapChainExtent.height), 0.0f,
                                                     1.0f));
    m_commandBuffers[m_currentFrame].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), m_swapChainExtent));

    // draw the scene meshes
    for (const auto& mesh : scene.meshes) {
        m_commandBuffers[m_currentFrame].bindVertexBuffers(0, *mesh.vertexBuffer, {0});
        m_commandBuffers[m_currentFrame].bindIndexBuffer(*mesh.indexBuffer, 0, vk::IndexType::eUint32);
        m_commandBuffers[m_currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0,
                                                            *mesh.descriptorSets[m_currentFrame], nullptr);
        m_commandBuffers[m_currentFrame].drawIndexed(mesh.indices.size(), 1, 0, 0, 0);
    }

    m_commandBuffers[m_currentFrame].endRendering();

    m_imageUtils.transitionImageLayout(
        m_swapChainImages[imageIndex],
        m_commandBuffers[m_currentFrame],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor
        );

    m_commandBuffers[m_currentFrame].end();
}

void Renderer::drawFrame(const Camera& camera) {
    while (vk::Result::eTimeout == m_device.waitForFences(*m_inFlightFences[m_currentFrame], vk::True, UINT64_MAX)) {
        // wait
    }

    auto [result, imageIndex] = m_swapChain.acquireNextImage(
        UINT64_MAX, *m_presentationCompleteSemaphores[m_semaphoreIndex], nullptr
        );

    updateUniformBuffers(camera);

    m_device.resetFences(*m_inFlightFences[m_currentFrame]);
    m_commandBuffers[m_currentFrame].reset();

    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);

    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_presentationCompleteSemaphores[m_semaphoreIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*m_commandBuffers[m_currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*m_renderFinishedSemaphores[imageIndex],
    };

    m_graphicsQueue.submit(submitInfo, *m_inFlightFences[m_currentFrame]);

    vk::PresentInfoKHR const presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*m_renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*m_swapChain,
        .pImageIndices = &imageIndex,
    };

    auto presentationResult = m_presentQueue.presentKHR(presentInfoKHR);

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
