#pragma once

#include <tiny_gltf.h>

#include <cstdint>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_raii.hpp>

#include "SharedTypes.hpp"

class VulkanCore;
class CommandManager;
class BufferManager;

class ImageManager {
public:
    explicit ImageManager(VulkanCore& vulkanCore, CommandManager& commandManager, BufferManager& bufferManager);

    void createImage(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t mipLevels,
        vk::SampleCountFlagBits numSamples,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Image& image,
        vk::raii::DeviceMemory& imageMemory
        ) const;

    vk::raii::ImageView createImageView(
        const vk::raii::Image& image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        std::uint32_t mipLevels
        ) const;

    vk::raii::Sampler createSampler(bool anisotropy) const;
    vk::raii::Sampler createSkyboxSampler() const;
    vk::raii::Sampler createPostProcessingSampler() const;

    void createImageFromTexture(
        const Texture& texture,
        vk::raii::Image& image,
        vk::raii::ImageView& imageView,
        vk::raii::DeviceMemory& imageMemory
        ) const;

    void transitionImageLayout(
        const vk::raii::Image& image,
        const vk::raii::CommandBuffer& commandBuffer,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags2 srcAccessMask,
        vk::AccessFlags2 dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask,
        vk::ImageAspectFlags aspectMask
        );

    void transitionImageLayout(
        const vk::Image& image,
        const vk::raii::CommandBuffer& commandBuffer,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::AccessFlags2 srcAccessMask,
        vk::AccessFlags2 dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask,
        vk::ImageAspectFlags aspectMask
        );

    void transitionImageLayout(
        const vk::raii::Image& image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels
        ) const;

    void generateMipmaps(
        const vk::raii::Image& image,
        vk::Format imageFormat,
        int32_t texWidth,
        int32_t texHeight,
        uint32_t mipLevels
        ) const;

private:
    VulkanCore& m_vulkanCore;
    CommandManager& m_commandManager;
    BufferManager& m_bufferManager;
};
