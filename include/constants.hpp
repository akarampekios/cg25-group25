#pragma once

#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.hpp>

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

constexpr float GLTF_DIRECTIONAL_LIGHT_INTENSITY_CONVERSION_FACTOR = 50000.0;
constexpr float GLTF_POINT_LIGHT_INTENSITY_CONVERSION_FACTOR = 500.0;
constexpr float GLTF_SPOT_LIGHT_INTENSITY_CONVERSION_FACTOR = 500.0;
