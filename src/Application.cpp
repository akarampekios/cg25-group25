#include <iostream>
#include <glm/gtc/matrix_transform.hpp>

// Include miniaudio implementation in this ONE cpp file only
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

// Undefine min/max macros if they were defined by miniaudio
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include "Application.hpp"
#include "SwapChain.hpp"
#include "RayQueryPipeline.hpp"
#include "ImageManager.hpp"
#include "BufferManager.hpp"
#include "PostProcessingStack.hpp"
#include "Animator.hpp"

Application::Application() : m_audioEngine(nullptr), m_backgroundMusic(nullptr) {
    createWindow();
    initVulkanCore();
    
    m_audioEngine = new ma_engine();
    ma_engine_config engineConfig = ma_engine_config_init();
    
    if (ma_engine_init(&engineConfig, m_audioEngine) != MA_SUCCESS) {
        std::cerr << "Failed to initialize audio engine" << std::endl;
        delete m_audioEngine;
        m_audioEngine = nullptr;
    }

    m_backgroundMusic = new ma_sound();
}

Application::~Application() {
    if (m_backgroundMusic) {
        ma_sound_uninit(m_backgroundMusic);
        delete m_backgroundMusic;
    }

    if (m_audioEngine) {
        ma_engine_uninit(m_audioEngine);
        delete m_audioEngine;
    }
    
    if (m_window) {
        glfwDestroyWindow(m_window);
    }

    glfwTerminate();
}

void Application::run() {
    GLTFLoader gltfLoader;
    Animator animator;
    CommandManager commandManager(*m_vulkanCore);
    BufferManager bufferManager(*m_vulkanCore, commandManager);
    ImageManager imageManager(*m_vulkanCore, commandManager, bufferManager);
    ResourceManager resourceManager(*m_vulkanCore, commandManager, bufferManager, imageManager);
    SwapChain swapChain(*m_vulkanCore, m_window);
    
    PostProcessingStack postProcessingStack(
        *m_vulkanCore, 
        resourceManager, 
        swapChain, 
        imageManager, 
        bufferManager
    );

    RayQueryPipeline rayQueryPipeline(
        *m_vulkanCore,
        resourceManager, 
        commandManager, 
        swapChain, 
        imageManager,
        bufferManager,
        postProcessingStack
        );
    
    auto loaded = gltfLoader.load("assets/scene_full.glb");

    resourceManager.allocateSceneResources(loaded->scene);

    if (m_audioEngine && m_backgroundMusic) {
        if (ma_sound_init_from_file(m_audioEngine, "assets/soundtrack_2.mp3", 
                                     MA_SOUND_FLAG_STREAM, NULL, NULL, 
                                     m_backgroundMusic) == MA_SUCCESS) {
            ma_sound_set_looping(m_backgroundMusic, MA_TRUE);  // Loop forever
            ma_sound_start(m_backgroundMusic);
        } else {
            std::cerr << "Failed to load music file (assets/soundtrack_2.mp3)" << std::endl;
        }
    }

    const double startTime = glfwGetTime();
    double lastTime = startTime;
    double lastFPSTime = startTime;
    int frameCount = 0;
    
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        
        const double currentTime = glfwGetTime();
        const double deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        
        const float animationTime = static_cast<float>(currentTime - startTime);
        
        animator.animate(loaded->model, loaded->scene, animationTime);
        rayQueryPipeline.drawFrame(loaded->scene);
        
        // FPS Counter (update every second)
        // We should prob delete this later before presentation build
        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            const double fps = frameCount / (currentTime - lastFPSTime);
            const double avgFrameTime = (currentTime - lastFPSTime) / frameCount * 1000.0;
            std::cout << "FPS: " << static_cast<int>(fps) 
                      << " | Avg Frame Time: " << avgFrameTime << "ms" 
                      << " | Last Delta: " << deltaTime * 1000.0 << "ms" << std::endl;
            frameCount = 0;
            lastFPSTime = currentTime;
        }
    }

    m_vulkanCore->device().waitIdle();
}

void Application::createWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    // Prevent Windows from throttling when unfocused
    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);

    m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);

    if (!m_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    // Request focus immediately after creation
    glfwFocusWindow(m_window);
}

void Application::initVulkanCore() {
    m_vulkanCore = std::make_unique<VulkanCore>(m_window);
}
