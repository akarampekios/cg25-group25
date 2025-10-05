#pragma once

#include <memory>

#include "Renderer.hpp"

class SimpleRenderer;
class Camera;
class Scene;

class Application {
public:
    Application();
    ~Application();

    void run();
    void cleanup();

private:
    void initWindow();
    void initVulkan();
    void mainLoop();
    void drawFrame();
    void update(float deltaTime);
    void handleInput(float deltaTime);

    GLFWwindow* m_window;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<Camera> m_camera;
    // std::unique_ptr<Scene> m_scene;

    bool m_running;
    bool m_dragging = false;
    float m_lastTime;
    float m_deltaTime;

    // Window properties
    static constexpr int WINDOW_WIDTH = 1920;
    static constexpr int WINDOW_HEIGHT = 1080;
    static constexpr const char* WINDOW_TITLE = "Cyberpunk City Demo";
};
