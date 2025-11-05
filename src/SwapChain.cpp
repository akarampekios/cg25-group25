#include <vector>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "constants.hpp"
#include "SwapChain.hpp"
#include "VulkanCore.hpp"

SwapChain::SwapChain(VulkanCore& vulkanCore, GLFWwindow* window) : m_vulkanCore{vulkanCore} {
    createSwapChain(window);
    createImageViews();
}

void SwapChain::createSwapChain(GLFWwindow* window) {
    auto [format, colorSpace] = chooseSurfaceFormat();
    const auto presentationMode = choosePresentationMode();
    const auto transform = chooseTransform();
    const auto imageCount = chooseMinImageCount();
    const auto extent = chooseExtent(window);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .flags = vk::SwapchainCreateFlagsKHR(),
        .surface = m_vulkanCore.surface(),
        .minImageCount = imageCount,
        .imageFormat = format,
        .imageColorSpace = colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = transform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = presentationMode,
        .clipped = vk::True,
        .oldSwapchain = nullptr,
    };

    const auto graphicsIndex = m_vulkanCore.queueFamilyIndices().graphicsFamily.value();
    const auto presentIndex = m_vulkanCore.queueFamilyIndices().presentFamily.value();

    if (graphicsIndex != presentIndex) {
        const std::vector queueFamilyIndices = {graphicsIndex, presentIndex};
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapChainCreateInfo.queueFamilyIndexCount = queueFamilyIndices.size();
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    }

    m_swapChainImageFormat = format;
    m_swapChainExtent = extent;

    m_swapChain = vk::raii::SwapchainKHR(m_vulkanCore.device(), swapChainCreateInfo);

    m_swapChainImages = m_swapChain.getImages();
}

void SwapChain::createImageViews() {
    m_swapChainImageViews.clear();

    vk::ImageViewCreateInfo imageViewCreateInfo{
        .viewType = vk::ImageViewType::e2D,
        .format = m_swapChainImageFormat,
        .subresourceRange =
        {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    for (const auto& image : m_swapChainImages) {
        imageViewCreateInfo.image = image;
        m_swapChainImageViews.emplace_back(m_vulkanCore.device(), imageViewCreateInfo);
    }
}

auto SwapChain::chooseSurfaceFormat() const -> vk::SurfaceFormatKHR {
    const auto colorFormats = m_vulkanCore.physicalDevice().getSurfaceFormatsKHR(*m_vulkanCore.surface());

    for (const auto& surfaceColorFormat : colorFormats) {
        if (surfaceColorFormat.format == PREFERRED_COLOR_FORMAT &&
            surfaceColorFormat.colorSpace == PREFERRED_COLOR_SPACE
        ) {
            return surfaceColorFormat;
        }
    }

    return colorFormats[0];
}

auto SwapChain::choosePresentationMode() const -> vk::PresentModeKHR {
    auto presentationModes = m_vulkanCore.physicalDevice().getSurfacePresentModesKHR(*m_vulkanCore.surface());

    for (const auto& presentationMode : presentationModes) {
        if (presentationMode == PREFERRED_PRESENTATION_MODE) {
            return presentationMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

auto SwapChain::chooseTransform() const -> vk::SurfaceTransformFlagBitsKHR {
    const auto capabilities = m_vulkanCore.physicalDevice().getSurfaceCapabilitiesKHR(*m_vulkanCore.surface());
    return capabilities.currentTransform;
}

auto SwapChain::chooseMinImageCount() const -> std::uint32_t {
    auto capabilities = m_vulkanCore.physicalDevice().getSurfaceCapabilitiesKHR(*m_vulkanCore.surface());
    return std::min(std::max(PREFERRED_IMAGE_COUNT, capabilities.minImageCount),
                    capabilities.maxImageCount);
}

auto SwapChain::chooseExtent(GLFWwindow* window) const -> vk::Extent2D {
    const auto capabilities = m_vulkanCore.physicalDevice().getSurfaceCapabilitiesKHR(*m_vulkanCore.surface());
    const auto isFixedWindowSize = capabilities.currentExtent.width !=
                                   std::numeric_limits<uint32_t>::max();

    if (isFixedWindowSize) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(window, &width, &height);

    return {
        .width = std::clamp<std::uint32_t>(width,
                                           capabilities.minImageExtent.width,
                                           capabilities.maxImageExtent.width),
        .height = std::clamp<std::uint32_t>(height,
                                            capabilities.minImageExtent.height,
                                            capabilities.maxImageExtent.height),
    };
}
