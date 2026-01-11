#pragma once

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vulkan/vulkan.hpp>

// Set to true to enable verbose debug output during loading
constexpr bool VERBOSE_DEBUG_OUTPUT = false;

constexpr int WINDOW_WIDTH = 1920;
constexpr int WINDOW_HEIGHT = 1080;
constexpr const char* WINDOW_TITLE = "Cyberpunk City Demo";

constexpr std::uint32_t MAX_FRAMES_IN_FLIGHT = 2;
constexpr std::uint32_t MAX_SCENE_OBJECTS = 100;
constexpr std::size_t MAX_TEXTURES_PER_TYPE = 1024;

static constexpr auto PREFERRED_COLOR_FORMAT = vk::Format::eB8G8R8A8Srgb;
static constexpr auto PREFERRED_COLOR_SPACE = vk::ColorSpaceKHR::eSrgbNonlinear;
static constexpr auto PREFERRED_PRESENTATION_MODE = vk::PresentModeKHR::eMailbox;
static constexpr auto PREFERRED_IMAGE_COUNT = 3U;

constexpr std::size_t POST_PROCESSING_BLUR_STAGES = 2; // should be 2
static constexpr std::uint32_t POST_PROCESSING_BLUR_PASSES = 4; // More passes = stronger, smoother blur (reduced from 3 with better kernel)
static constexpr vk::Format POST_PROCESSING_IMAGE_FORMAT = vk::Format::eR16G16B16A16Sfloat;

// TAA (Temporal Anti-Aliasing) Configuration
constexpr bool TAA_ENABLED = true;  // Enable TAA (disables MSAA when true)
constexpr float TAA_BLEND_FACTOR = 0.1f;  // α: 0.1 = 90% history, 10% current (higher = more responsive but more aliasing)
constexpr std::uint32_t TAA_JITTER_SEQUENCE_LENGTH = 16;  // Halton sequence length before repeat
static constexpr vk::Format VELOCITY_BUFFER_FORMAT = vk::Format::eR16G16Sfloat;  // RG16F for motion vectors

constexpr float GLTF_DIRECTIONAL_LIGHT_INTENSITY_CONVERSION_FACTOR = 50000.0;
constexpr float GLTF_POINT_LIGHT_INTENSITY_CONVERSION_FACTOR = 500.0;
constexpr float GLTF_SPOT_LIGHT_INTENSITY_CONVERSION_FACTOR = 500.0;

// Texture memory management - dynamic configuration
struct TextureMemoryConfig {
    std::uint32_t maxMipLevels;
    std::uint32_t maxTextureDimension;
    bool enableDownscaling;
    std::uint32_t tdrPreventionBatchSize;  // Flush GPU every N textures to prevent TDR
    std::uint32_t tdrPreventionDelayMs;    // Sleep time between batches (milliseconds)
    bool skipEmissiveTextures;             // Skip emissive textures for problematic GPUs
};

// Global texture configuration (initialized at startup based on VRAM)
inline TextureMemoryConfig g_textureConfig = {
    .maxMipLevels = 16,             // Full quality default
    .maxTextureDimension = 8192,    // No limit default
    .enableDownscaling = false,
    .tdrPreventionBatchSize = 0,    // 0 = disabled (high-end GPU)
    .tdrPreventionDelayMs = 0,
    .skipEmissiveTextures = false
};

// Initialize texture settings based on available VRAM
inline void initializeTextureSettings(std::uint64_t availableVRAM_bytes) {
    const std::uint64_t vramGB = availableVRAM_bytes / (1024ULL * 1024ULL * 1024ULL);
    const std::uint64_t vramMB = availableVRAM_bytes / (1024ULL * 1024ULL);
    
    const char* profileName = "HIGH";
    std::uint32_t maxDim = 8192;
    
    if (vramGB < 4) {
        g_textureConfig.maxMipLevels = 8;
        g_textureConfig.maxTextureDimension = 512;
        g_textureConfig.enableDownscaling = true;
        g_textureConfig.tdrPreventionBatchSize = 10;
        g_textureConfig.tdrPreventionDelayMs = 150;
        g_textureConfig.skipEmissiveTextures = false;
        profileName = "VERY LOW"; maxDim = 512;
    } else if (vramGB < 6) {
        g_textureConfig.maxMipLevels = 9;
        g_textureConfig.maxTextureDimension = 512;
        g_textureConfig.enableDownscaling = true;
        g_textureConfig.tdrPreventionBatchSize = 15;
        g_textureConfig.tdrPreventionDelayMs = 100;
        g_textureConfig.skipEmissiveTextures = false;
        profileName = "LOW"; maxDim = 512;
    } else if (vramGB < 8) {
        g_textureConfig.maxMipLevels = 10;
        g_textureConfig.maxTextureDimension = 1024;
        g_textureConfig.enableDownscaling = true;
        g_textureConfig.tdrPreventionBatchSize = 30;
        g_textureConfig.tdrPreventionDelayMs = 50;
        profileName = "MEDIUM"; maxDim = 1024;
    } else if (vramGB < 12) {
        g_textureConfig.maxMipLevels = 10;
        g_textureConfig.maxTextureDimension = 2048;
        g_textureConfig.enableDownscaling = true;
        g_textureConfig.tdrPreventionBatchSize = 50;
        g_textureConfig.tdrPreventionDelayMs = 25;
        profileName = "MEDIUM-HIGH"; maxDim = 2048;
    } else {
        g_textureConfig.maxMipLevels = 16;
        g_textureConfig.maxTextureDimension = 8192;
        g_textureConfig.enableDownscaling = false;
        g_textureConfig.tdrPreventionBatchSize = 0;
        g_textureConfig.tdrPreventionDelayMs = 0;
        profileName = "HIGH"; maxDim = 8192;
    }
    
    std::cout << "[GPU] VRAM: " << vramMB << " MB | Profile: " << profileName 
              << " | Max texture: " << maxDim << "x" << maxDim << std::endl;
}