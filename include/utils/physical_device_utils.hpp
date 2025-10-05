#pragma once

#include <optional>
#include <iostream>
#include <vector>

#include "vulkan_transfer_context.hpp"

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    float graphicsFamilyPriority = 0.0F;
    float presentationFamilyPriority = 0.0F;

    bool isComplete() const {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class PhysicalDeviceUtils {
public:
    explicit PhysicalDeviceUtils(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext} {
    }

    ~PhysicalDeviceUtils() = default;

    PhysicalDeviceUtils(const PhysicalDeviceUtils&) = delete;
    auto operator=(PhysicalDeviceUtils&) -> PhysicalDeviceUtils& = delete;
    PhysicalDeviceUtils(PhysicalDeviceUtils&&) = delete;
    auto operator=(PhysicalDeviceUtils&&) -> PhysicalDeviceUtils&& = delete;

    [[nodiscard]] auto isDeviceSuitable(const vk::raii::PhysicalDevice& physicalDevice,
                                        const std::vector<const char*>& extensions) const -> bool {
        auto queueFamilies = findQueueFamilies(physicalDevice);

        if (!queueFamilies.isComplete()) {
            return false;
        }


        if (!checkDeviceExtensionSupport(physicalDevice, extensions)) {
            return false;
        }

        auto featureChain = physicalDevice.getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceBufferDeviceAddressFeatures,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        >();

        const auto& bufferAddressFeatures = featureChain.get<
            vk::PhysicalDeviceBufferDeviceAddressFeatures>();
        const auto& rtPipelineFeatures = featureChain.get<
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>();
        const auto& accelFeatures = featureChain.get<
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();

        return bufferAddressFeatures.bufferDeviceAddress &&
               rtPipelineFeatures.rayTracingPipeline &&
               accelFeatures.accelerationStructure;
    }

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const {
        QueueFamilyIndices queueFamilyIndices{};

        const auto queueFamilies = physicalDevice.getQueueFamilyProperties();

        for (std::uint32_t i = 0; i < queueFamilies.size(); ++i) {
            const auto& queueFamily = queueFamilies[i];

            if ((queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) && !
                queueFamilyIndices.graphicsFamily.has_value()) {
                queueFamilyIndices.graphicsFamily = i;
            }

            auto ppSurface = m_vulkanTransferContext.surface;
            if (physicalDevice.getSurfaceSupportKHR(i, **ppSurface) && !queueFamilyIndices.presentFamily.has_value()) {
                queueFamilyIndices.presentFamily = i;
            }
        }

        return queueFamilyIndices;
    }

    auto findQueueFamilies() const -> QueueFamilyIndices {
        return findQueueFamilies(*m_vulkanTransferContext.physicalDevice);
    }

    auto findSupportedFormat(
        const std::vector<vk::Format>& candidates,
        const vk::ImageTiling tiling,
        const vk::FormatFeatureFlags features
        ) const -> vk::Format {
        for (const auto format : candidates) {
            vk::FormatProperties props = m_vulkanTransferContext.physicalDevice->getFormatProperties(format);

            if (tiling == vk::ImageTiling::eLinear && (
                    props.linearTilingFeatures & features) == features) {
                return format;
            }

            if (tiling == vk::ImageTiling::eOptimal && (
                    props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    vk::Format findDepthFormat() const {
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
             vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
            );
    }

    vk::SampleCountFlagBits findMsaaSamples() const {
        const vk::PhysicalDeviceProperties physicalDeviceProperties = m_vulkanTransferContext.physicalDevice->
            getProperties();

        const vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
                                            physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & vk::SampleCountFlagBits::e64) {
            return vk::SampleCountFlagBits::e64;
        }
        if (counts & vk::SampleCountFlagBits::e32) {
            return vk::SampleCountFlagBits::e32;
        }
        if (counts & vk::SampleCountFlagBits::e16) {
            return vk::SampleCountFlagBits::e16;
        }
        if (counts & vk::SampleCountFlagBits::e8) {
            return vk::SampleCountFlagBits::e8;
        }
        if (counts & vk::SampleCountFlagBits::e4) {
            return vk::SampleCountFlagBits::e4;
        }
        if (counts & vk::SampleCountFlagBits::e2) {
            return vk::SampleCountFlagBits::e2;
        }

        return vk::SampleCountFlagBits::e1;
    }

    std::uint32_t findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const {
        const auto memProperties = m_vulkanTransferContext.physicalDevice->getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

private:
    const VulkanTransferContext& m_vulkanTransferContext;

    static bool checkDeviceExtensionSupport(
        const vk::raii::PhysicalDevice& physicalDevice,
        const std::vector<const char*>& extensions) {
        const auto deviceProperties = physicalDevice.getProperties();
        auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        for (const auto& requiredExtName : extensions) {
            auto matcher = [&](const vk::ExtensionProperties& extension) {
                return std::string(extension.extensionName.data()) == std::string(
                           requiredExtName);
            };

            if (!std::ranges::any_of(availableExtensions, matcher)) {
                std::cout << std::format(
                    "Extension {} not found on device {}\n",
                    std::string(requiredExtName),
                    std::string(deviceProperties.deviceName.data()));
                return false;
            }
        }

        return true;
    }
};
