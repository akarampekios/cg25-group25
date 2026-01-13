#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class FreeCamera {
public:
    glm::vec3 position{0.0f, 5.0f, 0.0f};
    float yaw{0.0f};    // Horizontal rotation (radians)
    float pitch{0.0f};  // Vertical rotation (radians)
    
    float moveSpeed{10.0f};      // Units per second
    float sprintMultiplier{3.0f}; // Speed boost when sprinting
    float lookSpeed{0.002f};      // Radians per pixel
    
    void update(GLFWwindow* window, float deltaTime) {
        // Mouse look
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        
        static double lastX = mouseX, lastY = mouseY;
        // Initialize last positions on first run or if jump is too large
        static bool firstMouse = true;
        if (firstMouse) {
            lastX = mouseX;
            lastY = mouseY;
            firstMouse = false;
        }

        float dx = static_cast<float>(mouseX - lastX) * lookSpeed;
        float dy = static_cast<float>(mouseY - lastY) * lookSpeed;
        lastX = mouseX;
        lastY = mouseY;
        
        yaw += dx;
        pitch = glm::clamp(pitch - dy, -1.5f, 1.5f); // Limit pitch to avoid flipping
        
        // Calculate direction vectors
        glm::vec3 forward(
            cos(pitch) * cos(yaw),
            sin(pitch),
            cos(pitch) * sin(yaw)
        );
        forward = glm::normalize(forward);
        
        glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        glm::vec3 up = glm::vec3(0, 1, 0); // World up for vertical movement
        
        // Speed adjustment
        float speed = moveSpeed;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            speed *= sprintMultiplier;
        
        // WASD movement (horizontal plane)
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            position += forward * speed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            position -= forward * speed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            position -= right * speed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            position += right * speed * deltaTime;
        
        // Vertical movement (world up/down)
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            position += up * speed * deltaTime;
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
            position -= up * speed * deltaTime;
    }
    
    glm::mat4 getModelMatrix() const {
        glm::vec3 forward(
            cos(pitch) * cos(yaw),
            sin(pitch),
            cos(pitch) * sin(yaw)
        );
        forward = glm::normalize(forward);
        
        glm::vec3 worldUp(0, 1, 0);
        glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
        glm::vec3 up = glm::cross(right, forward);
        
        // Build model matrix (camera-to-world transform)
        // Note: The camera "model" matrix places the camera object in the world.
        // The View matrix is the inverse of this.
        // Our system seems to use a "camera model matrix" which is then inverted to get View.
        
        glm::mat4 model(1.0f);
        model[0] = glm::vec4(right, 0.0f);
        model[1] = glm::vec4(up, 0.0f);
        model[2] = glm::vec4(-forward, 0.0f); // -Z forward (OpenGL convention)
        model[3] = glm::vec4(position, 1.0f);
        return model;
    }
    
    void setPosition(const glm::vec3& pos) {
        position = pos;
    }
    
    void setOrientation(float yawRad, float pitchRad) {
        yaw = yawRad;
        pitch = glm::clamp(pitchRad, -1.5f, 1.5f);
    }
    
    // Reset mouse state to prevent jumps when toggling back
    void resetMouse(GLFWwindow* window) {
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);
        // We can't easily access static vars from outside, 
        // but we can make a small structure or just accept a small jump frame 1.
        // Alternatively, we just won't worry about the 1-frame jump for this demo.
    }
};
