#pragma once

#include "utils/physical_device_utils.hpp"
#include "vulkan_transfer_context.hpp"

class LogicalDeviceUtils {
public:
    explicit
    LogicalDeviceUtils(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext},
        m_device{*vulkanTransferContext.device} {
    }

    static auto buildQueueInfos(
        const QueueFamilyIndices& queueFamilyIndices) -> std::vector<
        vk::DeviceQueueCreateInfo> {
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

        queueCreateInfos.emplace_back(
            vk::DeviceQueueCreateInfo{
                .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
                .queueCount = 1,
                .pQueuePriorities = &queueFamilyIndices.graphicsFamilyPriority,
            }
            );

        if (queueFamilyIndices.graphicsFamily.value() != queueFamilyIndices.
                                                         presentFamily.value()) {
            queueCreateInfos.emplace_back(
                vk::DeviceQueueCreateInfo{
                    .queueFamilyIndex = queueFamilyIndices.presentFamily.value(),
                    .queueCount = 1,
                    .pQueuePriorities = &queueFamilyIndices.presentationFamilyPriority,
                }
                );
        }

        return queueCreateInfos;
    }

    static vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceBufferDeviceAddressFeatures,
        vk::PhysicalDeviceDescriptorIndexingFeatures,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR> buildFeatureChain() {
        vk::PhysicalDeviceFeatures2 deviceFeatures2{
            .features = {
                .samplerAnisotropy = vk::True
            },
        };
        vk::PhysicalDeviceVulkan13Features vulkan13Features{
            .synchronization2 = vk::True,
            .dynamicRendering = vk::True,
        };
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extendedStateFeatures{
            .extendedDynamicState = vk::True,
        };
        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferAddressFeatures{
            .bufferDeviceAddress = vk::True,
        };
        vk::PhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{
            .shaderSampledImageArrayNonUniformIndexing = vk::True,
            .descriptorBindingPartiallyBound = vk::True,
            .descriptorBindingVariableDescriptorCount = vk::True,
            .runtimeDescriptorArray = vk::True,
        };
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{
            .accelerationStructure = vk::True,
        };
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
            .rayTracingPipeline = vk::False,
        };

        return {
            deviceFeatures2,
            vulkan13Features,
            extendedStateFeatures,
            bufferAddressFeatures,
            descriptorIndexing,
            accelFeatures,
            rayTracingPipelineFeatures
        };
    }

private:
    const VulkanTransferContext m_vulkanTransferContext;
    const vk::raii::Device& m_device;
};
