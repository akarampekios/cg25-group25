#pragma once

#include "utils/physical_device_utils.hpp"
#include "utils/buffer_utils.hpp"
#include "vulkan_transfer_context.hpp"

class ImageUtils {
public:
    explicit ImageUtils(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext},
        m_physicalDeviceUtils{vulkanTransferContext},
        m_bufferUtils{vulkanTransferContext} {
    }

    void createImage(
        const std::uint32_t width,
        const std::uint32_t height,
        const uint32_t mipLevels,
        const vk::SampleCountFlagBits numSamples,
        const vk::Format format,
        const vk::ImageTiling tiling,
        const vk::ImageUsageFlags usage,
        const vk::MemoryPropertyFlags properties,
        vk::raii::Image& image,
        vk::raii::DeviceMemory& imageMemory
        ) const {
        const vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {.width = width, .height = height, .depth = 1},
            .mipLevels = mipLevels,
            .arrayLayers = 1,
            .samples = numSamples,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        image = vk::raii::Image(*m_vulkanTransferContext.device, imageInfo);

        const vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        const vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = m_physicalDeviceUtils.findMemoryType(
                memRequirements.memoryTypeBits,
                properties
                ),
        };

        imageMemory = vk::raii::DeviceMemory(*m_vulkanTransferContext.device, allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    vk::raii::ImageView createImageView(
        const vk::raii::Image& image,
        const vk::Format format,
        const vk::ImageAspectFlags aspectFlags,
        const std::uint32_t mipLevels
        ) const {
        vk::ImageViewCreateInfo viewInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        return vk::raii::ImageView(*m_vulkanTransferContext.device, viewInfo);
    }

    void generateMipmaps(
        const vk::raii::Image& image,
        const vk::Format imageFormat,
        const int32_t texWidth,
        const int32_t texHeight,
        const uint32_t mipLevels
        ) const {
        auto formatProperties = m_vulkanTransferContext.physicalDevice->getFormatProperties(imageFormat);

        // todo: smart way to choose the format
        if (!(formatProperties.optimalTilingFeatures &
              vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
            throw std::runtime_error(
                "texture image format does not support linear blitting!");
        }

        auto commandBuffer = m_bufferUtils.beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eTransfer,
                {},
                {},
                {},
                barrier
                );

            vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;

            offsets[0] = vk::Offset3D(0, 0, 0);
            offsets[1] = vk::Offset3D(mipWidth, mipHeight, 1);

            dstOffsets[0] = vk::Offset3D(0, 0, 0);
            dstOffsets[1] = vk::Offset3D(mipWidth > 1 ? mipWidth / 2 : 1,
                                         mipHeight > 1 ? mipHeight / 2 : 1, 1);

            vk::ImageBlit blit = {
                .srcSubresource = vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .srcOffsets = offsets,
                .dstSubresource = vk::ImageSubresourceLayers{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = i,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .dstOffsets = dstOffsets,
            };

            // must be submitted to the graphics and not the transger queue
            commandBuffer.blitImage(
                image,
                vk::ImageLayout::eTransferSrcOptimal,
                image,
                vk::ImageLayout::eTransferDstOptimal,
                {blit},
                vk::Filter::eLinear
                );

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer,
                vk::PipelineStageFlagBits::eFragmentShader,
                {},
                {},
                {},
                barrier
                );

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {},
            {},
            {},
            barrier
            );

        m_bufferUtils.endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(
        const vk::raii::Image& image,
        const vk::ImageLayout oldLayout,
        const vk::ImageLayout newLayout,
        const uint32_t mipLevels
        ) const {
        const auto commandBuffer = m_bufferUtils.beginSingleTimeCommands();

        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout ==
            vk::ImageLayout::eTransferDstOptimal) {
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            barrier.srcAccessMask = {};

            destinationStage = vk::PipelineStageFlagBits::eTransfer;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout ==
                   vk::ImageLayout::eShaderReadOnlyOptimal) {
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;

            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(
            sourceStage,
            destinationStage,
            {},
            {},
            nullptr,
            barrier
            );

        m_bufferUtils.endSingleTimeCommands(commandBuffer);
    }

    // void transitionImageLayout(
    //     const vk::raii::Image& image,
    //     const vk::raii::CommandBuffer& commandBuffer,
    //     vk::ImageLayout oldLayout,
    //     vk::ImageLayout newLayout,
    //     vk::AccessFlags2 srcAccessMask,
    //     vk::AccessFlags2 dstAccessMask,
    //     vk::PipelineStageFlags2 srcStageMask,
    //     vk::PipelineStageFlags2 dstStageMask
    //     ) {
    //   vk::ImageMemoryBarrier2 barrier = {
    //       .srcStageMask = srcStageMask,
    //       .srcAccessMask = srcAccessMask,
    //       .dstStageMask = dstStageMask,
    //       .dstAccessMask = dstAccessMask,
    //       .oldLayout = oldLayout,
    //       .newLayout = newLayout,
    //       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    //       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    //       .image = image,
    //       .subresourceRange = {
    //           .aspectMask = vk::ImageAspectFlagBits::eColor,
    //           .baseMipLevel = 0,
    //           .levelCount = 1,
    //           .baseArrayLayer = 0,
    //           .layerCount = 1
    //       }
    //   };
    //   vk::DependencyInfo dependencyInfo = {
    //       .dependencyFlags = {},
    //       .imageMemoryBarrierCount = 1,
    //       .pImageMemoryBarriers = &barrier
    //   };
    //   commandBuffer.pipelineBarrier2(dependencyInfo);
    // }

    void transitionImageLayout(
        const vk::raii::Image& image,
        const vk::raii::CommandBuffer& commandBuffer,
        const vk::ImageLayout oldLayout,
        const vk::ImageLayout newLayout,
        const vk::AccessFlags2 srcAccessMask,
        const vk::AccessFlags2 dstAccessMask,
        const vk::PipelineStageFlags2 srcStageMask,
        const vk::PipelineStageFlags2 dstStageMask,
        const vk::ImageAspectFlags aspectMask
        ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = aspectMask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
    }

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
        ) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {
                .aspectMask = aspectMask,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier
        };
        commandBuffer.pipelineBarrier2(dependencyInfo);
    }

    vk::raii::Sampler createSampler(const bool anisotropy) const {
        const auto properties = m_vulkanTransferContext.physicalDevice->getProperties();

        const vk::SamplerCreateInfo samplerCreateInfo {
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = anisotropy ? vk::True : vk::False,
            .maxAnisotropy = anisotropy ? properties.limits.maxSamplerAnisotropy : 1.0F,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
          };

        return vk::raii::Sampler(*m_vulkanTransferContext.device, samplerCreateInfo);
    }

private:
    const VulkanTransferContext& m_vulkanTransferContext;
    PhysicalDeviceUtils m_physicalDeviceUtils;
    BufferUtils m_bufferUtils;
};
