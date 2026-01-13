#pragma once

#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

// FSR 2 Includes
#include <ffx_fsr2.h>
#include <ffx_fsr2_vk.h>

class VulkanCore;

class FSR2Pass {
public:
    explicit FSR2Pass(VulkanCore& vulkanCore);
    ~FSR2Pass();

    void initialize(uint32_t renderWidth, uint32_t renderHeight, 
                   uint32_t displayWidth, uint32_t displayHeight);
    
    // Call this every frame to dispatch FSR 2
    void dispatch(const vk::raii::CommandBuffer& cmd,
                 const vk::raii::Image& colorImage, const vk::raii::ImageView& colorView,
                 const vk::raii::Image& depthImage, const vk::raii::ImageView& depthView,
                 const vk::raii::Image& motionVectorsImage, const vk::raii::ImageView& motionVectorsView,
                 const vk::raii::ImageView* exposure, // Mutable pointer as it can be null
                 const vk::raii::Image& outputImage, const vk::raii::ImageView& outputView,
                 const vk::raii::ImageView* reactiveMask,
                 const vk::raii::ImageView* transparencyAndCompositionMask,
                 float deltaTimeInMilliseconds,
                 float nearPlane,
                 float farPlane,
                 float fovV,
                 glm::vec2 jitterOffset,
                 bool reset = false
                 );

    void destroy();

    void onResize(uint32_t renderWidth, uint32_t renderHeight, 
                  uint32_t displayWidth, uint32_t displayHeight);

private:
    void createContext(uint32_t renderWidth, uint32_t renderHeight, 
                       uint32_t displayWidth, uint32_t displayHeight);
    void destroyContext();

    VulkanCore& m_vulkanCore;
    FfxFsr2Context m_context = {};
    FfxFsr2ContextDescription m_contextDescription = {};
    bool m_initialized = false;
    void* m_scratchBuffer = nullptr;
    
    // Cached resolution for validation/reset
    uint32_t m_renderWidth{0};
    uint32_t m_renderHeight{0};
    uint32_t m_displayWidth{0};
    uint32_t m_displayHeight{0};
};
