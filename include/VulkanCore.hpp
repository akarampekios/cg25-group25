#pragma once

#include <optional>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>

struct GLFWwindow;

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    float graphicsFamilyPriority = 0.0F;
    float presentationFamilyPriority = 0.0F;

    auto isComplete() const -> bool {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class VulkanCore {
public:
    explicit VulkanCore(GLFWwindow* window);

    static std::uint32_t version() { return VK_API_VERSION_1_4; }
    vk::raii::Instance& instance() { return m_instance; }
    vk::raii::SurfaceKHR& surface() { return m_surface; }
    const vk::raii::Device& device() const { return m_device; }
    vk::raii::PhysicalDevice& physicalDevice() { return m_physicalDevice; }
    QueueFamilyIndices queueFamilyIndices() const { return m_queueFamilyIndices; }
    vk::raii::Queue& graphicsQueue() { return m_graphicsQueue; }
    vk::raii::Queue& presentQueue() { return m_presentQueue; }

    auto findSupportedFormat(
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features
        ) const -> vk::Format;

    vk::Format findDepthFormat() const;

    vk::SampleCountFlagBits findMsaaSamples() const;

    std::uint32_t findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) const;

    std::uint64_t getAvailableVRAM() const;

private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface(GLFWwindow* window);
    void pickPhysicalDevice();
    void createLogicalDevice();

    auto findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const -> QueueFamilyIndices;

    auto isDeviceSuitable(const vk::raii::PhysicalDevice& physicalDevice,
                          const std::vector<const char*>& extensions) const -> bool;

    static auto checkDeviceExtensionSupport(
        const vk::raii::PhysicalDevice& physicalDevice,
        const std::vector<const char*>& extensions) -> bool;

    static auto buildQueueInfos(
        const QueueFamilyIndices& queueFamilyIndices) -> std::vector<
        vk::DeviceQueueCreateInfo>;

    static auto buildFeatureChain() -> vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceClusterAccelerationStructureFeaturesNV,
        vk::PhysicalDeviceRayQueryFeaturesKHR>;

    vk::raii::Context m_context;
    vk::raii::Instance m_instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger = nullptr;
    vk::raii::SurfaceKHR m_surface = nullptr;
    vk::raii::PhysicalDevice m_physicalDevice = nullptr;
    QueueFamilyIndices m_queueFamilyIndices;
    vk::raii::Queue m_graphicsQueue = nullptr;
    vk::raii::Queue m_presentQueue = nullptr;
    vk::raii::Device m_device = nullptr;
};
