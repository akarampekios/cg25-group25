#pragma once

struct VulkanTransferContext {
    GLFWwindow* window = nullptr;
    vk::raii::Instance* instance = nullptr;
    vk::raii::SurfaceKHR* surface = nullptr;
    vk::raii::Device* device = nullptr;
    vk::raii::PhysicalDevice* physicalDevice = nullptr;
    vk::raii::CommandPool* commandPool = nullptr;
    vk::raii::Queue* graphicsQueue = nullptr;
    vk::raii::Queue* presentationQueue = nullptr;
    vk::raii::SwapchainKHR* swapChain = nullptr;
    vk::raii::DescriptorSetLayout* descriptorSetLayout = nullptr;
    vk::raii::DescriptorPool* descriptorPool = nullptr;
    vk::raii::Sampler* baseColorTextureSampler = nullptr;
    vk::raii::Sampler* metallicRoughnessTextureSampler = nullptr;
    vk::raii::Sampler* normalTextureSampler = nullptr;
    vk::raii::Sampler* emissiveTextureSampler = nullptr;
    vk::raii::Sampler* occlusionTextureSampler = nullptr;
};
