#pragma once

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

class VulkanCore;
class CommandManager;

class BufferManager {
public:
    explicit BufferManager(VulkanCore& vulkanCore, CommandManager& commandManager);

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory,
        const void* data
        );

    void createBuffer(
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
        );

    void createStagingBuffer(
        vk::DeviceSize size,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory,
        const void* data
        );

    void createStagingBuffer(
        vk::DeviceSize size,
        vk::raii::Buffer& buffer,
        vk::raii::DeviceMemory& bufferMemory
        );

    void copyBuffer(const vk::raii::Buffer& srcBuffer,
                    const vk::raii::Buffer& dstBuffer,
                    vk::DeviceSize size) const;

    void copyBufferToImage(const vk::raii::Buffer& buffer,
                           const vk::raii::Image& image,
                           std::uint32_t width,
                           std::uint32_t height) const;

private:
    VulkanCore& m_vulkanCore;
    CommandManager& m_commandManager;
};
