#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>
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
    
    const std::string scenePath = "assets/scene_full.glb";
    if (!std::filesystem::exists(scenePath)) {
        throw std::runtime_error("Scene file not found: " + scenePath);
    }

    auto loaded = gltfLoader.load(scenePath);
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
    
    // Frame pacing: target 60 FPS (16.67ms per frame)
    const double targetFrameTime = 1.0 / 60.0;
    double frameStartTime = startTime;
    
    // Initialize free camera
    m_freeCamera.setPosition(loaded->scene.camera.getPosition());

    std::cout << "[Render] Entering render loop..." << std::endl;
    
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        
        const double currentTime = glfwGetTime();
        const double deltaTime = currentTime - lastTime;
        lastTime = currentTime;
        
        // Frame pacing: limit frame rate to prevent queue buildup
        // The fence wait in drawFrame() already handles GPU sync, but we pace CPU-side
        // to prevent submitting too many frames ahead of the GPU
        const double elapsedSinceFrameStart = currentTime - frameStartTime;
        if (elapsedSinceFrameStart < targetFrameTime) {
            // Sleep to avoid busy-waiting (only if significant time remaining)
            const double sleepTime = (targetFrameTime - elapsedSinceFrameStart) * 1000.0;
            if (sleepTime > 1.0) {  // Only sleep if more than 1ms
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(sleepTime * 1000)));
            }
        }
        frameStartTime = currentTime;
        
        // Toggle camera mode with F key (debounced)
        bool fKeyDown = glfwGetKey(m_window, GLFW_KEY_F) == GLFW_PRESS;
        if (fKeyDown && !m_fKeyPressed) {
            m_useFreeCam = !m_useFreeCam;
            
            if (m_useFreeCam) {
                // Switched to free camera
                m_freeCamera.setPosition(loaded->scene.camera.getPosition());
                // Capture mouse
                glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                std::cout << "Free Camera: ENABLED (WASD=move, Mouse=look, Shift=sprint, F=toggle)" << std::endl;
                m_freeCamera.resetMouse(m_window);
            } else {
                // Switched back to animated camera
                glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                std::cout << "Free Camera: DISABLED (cinematic path active)" << std::endl;
            }
        }
        m_fKeyPressed = fKeyDown;

        const float animationTime = static_cast<float>(currentTime - startTime) * 0.5f; // Quarter speed (half of previous half speed)
        
        if (m_useFreeCam) {
            m_freeCamera.update(m_window, static_cast<float>(deltaTime));
            loaded->scene.camera.model = m_freeCamera.getModelMatrix();
        } else {
            animator.animate(loaded->model, loaded->scene, animationTime);
        }
        
        rayQueryPipeline.drawFrame(loaded->scene, animationTime);
        
        // FPS Counter (update every second)
        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            const double fps = frameCount / (currentTime - lastFPSTime);
            const double avgFrameTime = (currentTime - lastFPSTime) / frameCount * 1000.0;
            const double lastDeltaMs = deltaTime * 1000.0;
            
            std::cout << "FPS: " << static_cast<int>(fps) 
                      << " | Avg: " << avgFrameTime << "ms" 
                      << " | Last: " << lastDeltaMs << "ms" << std::endl;
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
    
    // Initialize texture settings based on available VRAM
    const std::uint64_t availableVRAM = m_vulkanCore->getAvailableVRAM();
    initializeTextureSettings(availableVRAM);
}
