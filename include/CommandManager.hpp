#pragma once

#include <functional>
#include <vulkan/vulkan_raii.hpp>

class VulkanCore;

class CommandManager {
public:
    explicit CommandManager(VulkanCore& vulkanCore);
    ~CommandManager() = default;

    CommandManager(const CommandManager&) = delete;
    auto operator=(CommandManager&) -> CommandManager& = delete;
    CommandManager(CommandManager&&) = delete;
    auto operator=(CommandManager&&) -> CommandManager&& = delete;

    void immediateSubmit(std::function<void(vk::CommandBuffer)>&& function) const;

    auto getCommandBuffer(const std::uint32_t index) const -> const vk::raii::CommandBuffer& {
        return m_commandBuffers[index];
    }

private:
    void createCommandPool();
    void createCommandBuffers();

    VulkanCore& m_vulkanCore;

    vk::raii::CommandPool m_commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;
};
