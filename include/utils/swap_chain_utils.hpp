#pragma once

#include "utils/window_utils.hpp"
#include "vulkan_transfer_context.hpp"

static constexpr auto PREFERRED_COLOR_FORMAT = vk::Format::eB8G8R8A8Srgb;
static constexpr auto PREFERRED_COLOR_SPACE = vk::ColorSpaceKHR::eSrgbNonlinear;
static constexpr auto PREFERRED_PRESENTATION_MODE =
    vk::PresentModeKHR::eMailbox;
static constexpr auto PREFERRED_IMAGE_COUNT = 3U;

class SwapChainUtils {
public:
    explicit SwapChainUtils(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext} {
    }

    auto chooseSurfaceFormat() const -> vk::SurfaceFormatKHR {
        const auto pPhysicalDevice = m_vulkanTransferContext.physicalDevice;
        const auto pSurface = m_vulkanTransferContext.surface;

        const auto colorFormats = pPhysicalDevice->getSurfaceFormatsKHR(**pSurface);

        for (const auto& surfaceColorFormat : colorFormats) {
            if (surfaceColorFormat.format == PREFERRED_COLOR_FORMAT &&
                surfaceColorFormat.colorSpace == PREFERRED_COLOR_SPACE
            ) {
                return surfaceColorFormat;
            }
        }

        return colorFormats[0];
    }

    auto choosePresentationMode() const -> vk::PresentModeKHR {
        const auto pPhysicalDevice = m_vulkanTransferContext.physicalDevice;
        const auto pSurface = m_vulkanTransferContext.surface;

        auto presentationModes = pPhysicalDevice->getSurfacePresentModesKHR(**pSurface);

        for (const auto& presentationMode : presentationModes) {
            if (presentationMode == PREFERRED_PRESENTATION_MODE) {
                return presentationMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    auto chooseTransform() const -> vk::SurfaceTransformFlagBitsKHR {
        const auto pPhysicalDevice = m_vulkanTransferContext.physicalDevice;
        const auto pSurface = m_vulkanTransferContext.surface;

        const auto capabilities = pPhysicalDevice->getSurfaceCapabilitiesKHR(*pSurface);

        return capabilities.currentTransform;
    }

    auto chooseMinImageCount() const -> std::uint32_t {
        const auto pPhysicalDevice = m_vulkanTransferContext.physicalDevice;
        const auto pSurface = m_vulkanTransferContext.surface;

        auto capabilities = pPhysicalDevice->getSurfaceCapabilitiesKHR(*pSurface);
        return std::min(std::max(PREFERRED_IMAGE_COUNT, capabilities.minImageCount),
                        capabilities.maxImageCount);
    }

    auto chooseExtent() -> vk::Extent2D {
        const auto pPhysicalDevice = m_vulkanTransferContext.physicalDevice;
        const auto pSurface = m_vulkanTransferContext.surface;
        const auto pWindow = m_vulkanTransferContext.window;

        const auto capabilities = pPhysicalDevice->getSurfaceCapabilitiesKHR(*pSurface);
        auto isFixedWindowSize = capabilities.currentExtent.width !=
                                 std::numeric_limits<uint32_t>::max();

        if (isFixedWindowSize) {
            return capabilities.currentExtent;
        }

        auto [width, height] = WindowUtils::getPixelSize(pWindow);

        return {
            .width = std::clamp<std::uint32_t>(width,
                                               capabilities.minImageExtent.width,
                                               capabilities.maxImageExtent.width),
            .height = std::clamp<std::uint32_t>(height,
                                                capabilities.minImageExtent.height,
                                                capabilities.maxImageExtent.height),
        };
    }

private:
    const VulkanTransferContext& m_vulkanTransferContext;
};
