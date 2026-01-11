#include "ImageManager.hpp"
#include "VulkanCore.hpp"
#include "SharedTypes.hpp"
#include "CommandManager.hpp"
#include "BufferManager.hpp"
#include <iostream>

// Global memory tracking
namespace {
    std::size_t g_totalAllocatedMemory = 0;
    std::size_t g_peakMemoryUsage = 0;
}

ImageManager::ImageManager(VulkanCore& vulkanCore, CommandManager& commandManager, BufferManager& bufferManager) :
    m_vulkanCore{vulkanCore},
    m_commandManager{commandManager},
    m_bufferManager{bufferManager} {
}

void ImageManager::createImage(
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

    image = vk::raii::Image(m_vulkanCore.device(), imageInfo);

    const vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    const vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = m_vulkanCore.findMemoryType(memRequirements.memoryTypeBits, properties),
    };

    try {
        imageMemory = vk::raii::DeviceMemory(m_vulkanCore.device(), allocInfo);
        image.bindMemory(imageMemory, 0);
        
        // Track memory usage
        g_totalAllocatedMemory += memRequirements.size;
        if (g_totalAllocatedMemory > g_peakMemoryUsage) {
            g_peakMemoryUsage = g_totalAllocatedMemory;
        }
    } catch (const vk::OutOfDeviceMemoryError& e) {
        const float sizeMB = static_cast<float>(memRequirements.size) / (1024.0f * 1024.0f);
        std::cerr << "[GPU Memory] ERROR: Out of device memory while allocating image!" << std::endl;
        std::cerr << "  - Image size: " << width << "x" << height << std::endl;
        std::cerr << "  - Mip levels: " << mipLevels << std::endl;
        std::cerr << "  - Memory required: " << sizeMB << " MB" << std::endl;
        std::cerr << "  - Total memory already allocated: " 
                  << (static_cast<float>(g_totalAllocatedMemory) / (1024.0f * 1024.0f)) << " MB" << std::endl;
        throw;
    }
}

vk::raii::ImageView ImageManager::createImageView(
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

    return vk::raii::ImageView(m_vulkanCore.device(), viewInfo);
}

vk::raii::Sampler ImageManager::createSampler(const bool anisotropy) const {
    const auto properties = m_vulkanCore.physicalDevice().getProperties();

    const vk::SamplerCreateInfo samplerCreateInfo{
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

    return vk::raii::Sampler(m_vulkanCore.device(), samplerCreateInfo);
}

vk::raii::Sampler ImageManager::createSkyboxSampler() const {
    const auto properties = m_vulkanCore.physicalDevice().getProperties();

    const vk::SamplerCreateInfo samplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = true,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
    };

    return vk::raii::Sampler(m_vulkanCore.device(), samplerCreateInfo);
}

vk::raii::Sampler ImageManager::createPostProcessingSampler() const {
    const auto properties = m_vulkanCore.physicalDevice().getProperties();

    const vk::SamplerCreateInfo samplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = false,
        .maxAnisotropy = 1.0F,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
    };

    return vk::raii::Sampler(m_vulkanCore.device(), samplerCreateInfo);
}

void ImageManager::createImageFromTexture(const Texture& texture,
                                          vk::raii::Image& image,
                                          vk::raii::ImageView& imageView,
                                          vk::raii::DeviceMemory& imageMemory) const {
    createImage(
        texture.width,
        texture.height,
        texture.mipLevels,
        vk::SampleCountFlagBits::e1,
        texture.format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc
        | vk::ImageUsageFlagBits::eTransferDst
        | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        image,
        imageMemory
        );

    transitionImageLayout(
        image,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        texture.mipLevels
        );

    // Use explicit scope to ensure staging buffers are freed immediately after upload
    {
        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        const vk::DeviceSize imageSize = texture.image.size();

        m_bufferManager.createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
            );

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, texture.image.data(), imageSize);
        stagingBufferMemory.unmapMemory();

        m_bufferManager.copyBufferToImage(
            stagingBuffer,
            image,
            texture.width,
            texture.height
            );
        
        // stagingBuffer and stagingBufferMemory are destroyed here when leaving scope
    }

    if (texture.mipLevels > 1) {
        generateMipmaps(
            image,
            texture.format,
            texture.width,
            texture.height,
            texture.mipLevels
            );
    } else {
        // For textures with only 1 mip level, we need to transition from TransferDst to ShaderReadOnly
        transitionImageLayout(
            image,
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            texture.mipLevels
            );
    }

    imageView = createImageView(
        image,
        texture.format,
        vk::ImageAspectFlagBits::eColor,
        texture.mipLevels
        );
}

void ImageManager::transitionImageLayout(
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
    const vk::DependencyInfo dependencyInfo = {
        .dependencyFlags = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier
    };

    commandBuffer.pipelineBarrier2(dependencyInfo);
}

void ImageManager::transitionImageLayout(
    const vk::Image& image,
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

void ImageManager::transitionImageLayout(
    const vk::raii::Image& image,
    const vk::ImageLayout oldLayout,
    const vk::ImageLayout newLayout,
    const uint32_t mipLevels
    ) const {
    m_commandManager.immediateSubmit([&](vk::CommandBuffer cmd) {
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

        cmd.pipelineBarrier(
            sourceStage,
            destinationStage,
            {},
            {},
            nullptr,
            barrier
            );
    });
}

void ImageManager::generateMipmaps(
    const vk::raii::Image& image,
    const vk::Format imageFormat,
    const std::int32_t texWidth,
    const std::int32_t texHeight,
    const std::uint32_t mipLevels
    ) const {
    auto formatProperties = m_vulkanCore.physicalDevice().getFormatProperties(imageFormat);

    // todo: smart way to choose the format
    if (!(formatProperties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
        throw std::runtime_error(
            "texture image format does not support linear blitting!");
    }

    m_commandManager.immediateSubmit([&](vk::CommandBuffer cmd) {
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

            cmd.pipelineBarrier(
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
            cmd.blitImage(
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

            cmd.pipelineBarrier(
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

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            {},
            {},
            {},
            barrier
            );
    });
}
