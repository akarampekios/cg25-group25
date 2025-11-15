#include "constants.hpp"
#include "CommandManager.hpp"
#include "VulkanCore.hpp"

CommandManager::CommandManager(VulkanCore& vulkanCore) : m_vulkanCore{vulkanCore} {
    createCommandPool();
    createCommandBuffers();
}

void CommandManager::immediateSubmit(std::function<void(vk::CommandBuffer)>&& function) const {
    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = *m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };

    const vk::raii::CommandBuffers commandBuffers(m_vulkanCore.device(), allocInfo);
    vk::CommandBuffer cmd = *commandBuffers.front();

    constexpr vk::CommandBufferBeginInfo beginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    };

    cmd.begin(beginInfo);

    function(cmd);

    cmd.end();

    const vk::SubmitInfo submitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };

    m_vulkanCore.graphicsQueue().submit(submitInfo);
    m_vulkanCore.graphicsQueue().waitIdle();
}

void CommandManager::createCommandPool() {
    const vk::CommandPoolCreateInfo poolInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = m_vulkanCore.queueFamilyIndices().graphicsFamily.value(),
    };

    m_commandPool = vk::raii::CommandPool(m_vulkanCore.device(), poolInfo);
}

void CommandManager::createCommandBuffers() {
    m_commandBuffers.clear();

    const vk::CommandBufferAllocateInfo allocInfo{
        .commandPool = m_commandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };

    m_commandBuffers = vk::raii::CommandBuffers(m_vulkanCore.device(), allocInfo);
}
