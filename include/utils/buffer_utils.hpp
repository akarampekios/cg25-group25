#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "utils/physical_device_utils.hpp"
#include "vulkan_transfer_context.hpp"

class BufferUtils {
public:
    explicit BufferUtils(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext},
        m_physicalDeviceUtils{vulkanTransferContext} {
    }

    vk::raii::CommandBuffer beginSingleTimeCommands() const {
        const vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *m_vulkanTransferContext.commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandBuffer = std::move(
            m_vulkanTransferContext.device->allocateCommandBuffers(allocInfo).front());

        commandBuffer.begin(
            vk::CommandBufferBeginInfo{
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
            }
            );

        return commandBuffer;
    }

    void endSingleTimeCommands(
        const vk::raii::CommandBuffer& commandBuffer) const {
        commandBuffer.end();

        m_vulkanTransferContext.graphicsQueue->submit(
            vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
        m_vulkanTransferContext.graphicsQueue->waitIdle();
    }

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
        ) {
        const vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        buffer = vk::raii::Buffer(*m_vulkanTransferContext.device, bufferInfo);

        auto memoryRequirements = buffer.getMemoryRequirements();

        const vk::MemoryAllocateInfo memoryAllocateInfo{
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = m_physicalDeviceUtils.findMemoryType(
                memoryRequirements.memoryTypeBits,
                properties),
        };

        bufferMemory = vk::raii::DeviceMemory(*m_vulkanTransferContext.device, memoryAllocateInfo);

        buffer.bindMemory(*bufferMemory, 0);
    }

    void copyBufferToImage(const vk::raii::Buffer& buffer,
                           const vk::raii::Image& image,
                           const uint32_t width, const uint32_t height) const {
        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {
                .x = 0,
                .y = 0,
                .z = 0
            },
            .imageExtent = {
                .width = width,
                .height = height,
                .depth = 1,
            },
        };

        commandBuffer.copyBufferToImage(buffer, image,
                                        vk::ImageLayout::eTransferDstOptimal,
                                        {region});

        endSingleTimeCommands(commandBuffer);
    }

    void copyBuffer(const vk::raii::Buffer& srcBuffer, const vk::raii::Buffer& dstBuffer,
                    const vk::DeviceSize size) const {
        const vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
        endSingleTimeCommands(commandCopyBuffer);
    }

private:
    const VulkanTransferContext m_vulkanTransferContext;
    PhysicalDeviceUtils m_physicalDeviceUtils;
};
