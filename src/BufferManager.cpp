#include "BufferManager.hpp"
#include "VulkanCore.hpp"
#include "CommandManager.hpp"

BufferManager::BufferManager(
    VulkanCore& vulkanCore,
    CommandManager& commandManager)
    : m_vulkanCore(vulkanCore),
      m_commandManager(commandManager) {
}


void BufferManager::createBuffer(
    const vk::DeviceSize size,
    const vk::BufferUsageFlags usage,
    const vk::MemoryPropertyFlags properties,
    vk::raii::Buffer& buffer,
    vk::raii::DeviceMemory& bufferMemory,
    const void* data
    ) {
    vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    if (data != nullptr) {
        bufferInfo.usage = usage | vk::BufferUsageFlagBits::eTransferDst;
    }

    buffer = vk::raii::Buffer(m_vulkanCore.device(), bufferInfo);

    const auto memoryRequirements = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = m_vulkanCore.findMemoryType(
            memoryRequirements.memoryTypeBits,
            properties),
    };

    vk::MemoryAllocateFlagsInfo memoryAllocateFlagsInfo;
    if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
        memoryAllocateFlagsInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
        memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    }

    bufferMemory = vk::raii::DeviceMemory(m_vulkanCore.device(), memoryAllocateInfo);
    buffer.bindMemory(*bufferMemory, 0);

    if (data != nullptr) {
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createStagingBuffer(size, stagingBuffer, stagingBufferMemory, data);
        copyBuffer(stagingBuffer, buffer, size);
    }
}

void BufferManager::createBuffer(
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::raii::Buffer& buffer,
    vk::raii::DeviceMemory& bufferMemory
    ) {
    createBuffer(
        size,
        usage,
        properties,
        buffer,
        bufferMemory,
        nullptr
        );
}

void BufferManager::createStagingBuffer(
    vk::DeviceSize size,
    vk::raii::Buffer& buffer,
    vk::raii::DeviceMemory& bufferMemory,
    const void* data
    ) {
    const vk::BufferCreateInfo bufferInfo{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc,
        .sharingMode = vk::SharingMode::eExclusive,
    };

    buffer = vk::raii::Buffer(m_vulkanCore.device(), bufferInfo);

    const auto memoryRequirements = buffer.getMemoryRequirements();

    const vk::MemoryAllocateInfo memoryAllocateInfo{
        .allocationSize = memoryRequirements.size,
        .memoryTypeIndex = m_vulkanCore.findMemoryType(
            memoryRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    };

    bufferMemory = vk::raii::DeviceMemory(m_vulkanCore.device(), memoryAllocateInfo);
    buffer.bindMemory(*bufferMemory, 0);

    if (data != nullptr) {
        void* dataMemory = bufferMemory.mapMemory(0, size);
        memcpy(dataMemory, data, size);
        bufferMemory.unmapMemory();
    }
}

void BufferManager::createStagingBuffer(
    const vk::DeviceSize size,
    vk::raii::Buffer& buffer,
    vk::raii::DeviceMemory& bufferMemory
    ) {
    createStagingBuffer(size, buffer, bufferMemory, nullptr);
}

void BufferManager::copyBuffer(const vk::raii::Buffer& srcBuffer,
                               const vk::raii::Buffer& dstBuffer,
                               const vk::DeviceSize size) const {
    m_commandManager.immediateSubmit([&](vk::CommandBuffer cmd) {
        cmd.copyBuffer(srcBuffer, dstBuffer, vk::BufferCopy(0, 0, size));
    });
}

void BufferManager::copyBufferToImage(const vk::raii::Buffer& buffer,
                                      const vk::raii::Image& image,
                                      const uint32_t width, const uint32_t height) const {
    m_commandManager.immediateSubmit([&](const vk::CommandBuffer cmd) {
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

        cmd.copyBufferToImage(buffer, image,
                              vk::ImageLayout::eTransferDstOptimal,
                              {region});
    });
}
