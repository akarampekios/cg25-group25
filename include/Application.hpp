#pragma once

#include <memory>
#include <stdexcept>
#include <GLFW/glfw3.h>

// Prevent Windows min/max macro conflicts with glm
#ifndef NOMINMAX
#define NOMINMAX
#endif

struct ma_engine;
struct ma_sound;

#include "constants.hpp"
#include "VulkanCore.hpp"
#include "ResourceManager.hpp"
#include "CommandManager.hpp"
#include "GLTFLoader.hpp"

class Application {
public:
    explicit Application();

    ~Application();

    void run();
private:
    GLFWwindow* m_window;
    std::unique_ptr<VulkanCore> m_vulkanCore = nullptr;
    
    ma_engine* m_audioEngine;
    ma_sound* m_backgroundMusic;

    void createWindow();

    void initVulkanCore();
};
