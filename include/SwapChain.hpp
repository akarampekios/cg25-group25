#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include <vulkan/vulkan_raii.hpp>

class VulkanCore;
struct GLFWwindow;

class SwapChain {
public:
    explicit SwapChain(VulkanCore& vulkanCore, GLFWwindow* window);

    [[nodiscard]] auto getImages() const -> const std::vector<vk::Image>& { return m_swapChainImages; }
    [[nodiscard]] auto getFormat() const -> vk::Format { return m_swapChainImageFormat; }
    [[nodiscard]] auto getExtent() const -> vk::Extent2D { return m_swapChainExtent; }
    [[nodiscard]] auto getImage(const std::uint32_t index) const -> vk::Image { return m_swapChainImages[index]; }
    [[nodiscard]] auto getSwapChain() const -> const vk::raii::SwapchainKHR& { return m_swapChain; }

    [[nodiscard]] auto getImageView(const std::uint32_t index) const -> const vk::raii::ImageView& {
        return m_swapChainImageViews[index];
    }

private:
    VulkanCore& m_vulkanCore;

    vk::raii::SwapchainKHR m_swapChain = nullptr;
    std::vector<vk::Image> m_swapChainImages;
    std::vector<vk::raii::ImageView> m_swapChainImageViews;
    vk::Format m_swapChainImageFormat;
    vk::Extent2D m_swapChainExtent;

    void createSwapChain(GLFWwindow* window);

    void createImageViews();

    [[nodiscard]] auto chooseSurfaceFormat() const -> vk::SurfaceFormatKHR;
    [[nodiscard]] auto choosePresentationMode() const -> vk::PresentModeKHR;
    [[nodiscard]] auto chooseTransform() const -> vk::SurfaceTransformFlagBitsKHR;
    [[nodiscard]] auto chooseMinImageCount() const -> std::uint32_t;
    auto chooseExtent(GLFWwindow* window) const -> vk::Extent2D;
};
