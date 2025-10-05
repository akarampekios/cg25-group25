#include <iostream>
#include <chrono>
#include <thread>

#include "Application.h"
// #include "SimpleRenderer.h"
#include "Renderer.hpp"
#include "Camera.h"
// #include "Scene.h"


Application::Application() 
    : m_window(nullptr)
    , m_running(false)
    , m_lastTime(0.0f)
    , m_deltaTime(0.0f) {
  initWindow();
  initVulkan();
}

Application::~Application() {
    cleanup();
}

void Application::run() {
    // initWindow();
    // initVulkan();
    mainLoop();
}

void Application::initWindow() {
    glfwInit();
    
    // Don't create OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    
    m_window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, nullptr, nullptr);
    if (!m_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    // std::cout << "Window created successfully: " << WINDOW_WIDTH << "x" << WINDOW_HEIGHT << std::endl;
    
    glfwSetWindowUserPointer(m_window, this);
    
    // Input callbacks
    glfwSetKeyCallback(m_window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
            app->m_running = false;
        }
    });
    
    glfwSetCursorPosCallback(m_window, [](GLFWwindow* window, double xpos, double ypos) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        // Camera will handle mouse input
    });
    
    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* window, int button, int action, int mods) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            app->m_dragging = true;
            // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
            app->m_dragging = false;
            // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    });
}

void Application::initVulkan() {
    // std::cout << "Creating renderer..." << std::endl;
    m_renderer = std::make_unique<Renderer>(m_window);
    
    // std::cout << "Creating camera..." << std::endl;
    m_camera = std::make_unique<Camera>(WINDOW_WIDTH, WINDOW_HEIGHT);

    // std::cout << "Creating scene..." << std::endl;
    // m_scene = std::make_unique<Scene>();
    //
    // // Initialize scene with basic geometry
    // std::cout << "Initializing scene..." << std::endl;
    // m_scene->init();
    //
    // // Initialize renderer geometry with scene data
    // std::cout << "Initializing renderer geometry..." << std::endl;
    // // m_renderer->initGeometry(m_scene.get());
    //
    // std::cout << "Vulkan initialization complete!" << std::endl;
}

void Application::mainLoop() {
    m_running = true;
    m_lastTime = static_cast<float>(glfwGetTime());
    
    // std::cout << "Starting main loop..." << std::endl;
    
    while (m_running && !glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        
        // Check if window should close
        if (glfwWindowShouldClose(m_window)) {
            std::cout << "Window should close" << std::endl;
            break;
        }
        
        float currentTime = static_cast<float>(glfwGetTime());
        m_deltaTime = currentTime - m_lastTime;
        m_lastTime = currentTime;
        
        update(m_deltaTime);
        drawFrame();
        
        // Add a small delay to prevent excessive CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
    
    std::cout << "Main loop ended." << std::endl;
    // m_renderer->getDevice().waitIdle();
}

void Application::drawFrame() {
    try {
        // m_renderer->beginFrame();
        m_renderer->drawFrame(*m_camera);
        // m_renderer->endFrame();
    } catch (const std::exception& e) {
        std::cerr << "Error in drawFrame: " << e.what() << std::endl;
        m_running = false;
    }
}

void Application::update(float deltaTime) {
    handleInput(deltaTime);
    m_camera->update(deltaTime);
    // m_scene->update(deltaTime);
}

void Application::handleInput(float deltaTime) {
    // Camera movement
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
        m_camera->moveForward(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
        m_camera->moveForward(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) {
        m_camera->moveBackward(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) {
        m_camera->moveLeft(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) {
        m_camera->moveRight(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        m_camera->moveUp(deltaTime);
    }
    if (glfwGetKey(m_window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        m_camera->moveDown(deltaTime);
    }
    
    // Mouse look
    double xpos, ypos;
    glfwGetCursorPos(m_window, &xpos, &ypos);

    if (m_dragging) {
        m_camera->handleMouseMovement(static_cast<float>(xpos), static_cast<float>(ypos));
    }
}

void Application::cleanup() {
    if (m_renderer) {
        m_renderer.reset();
    }
    
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    
    glfwTerminate();
}
