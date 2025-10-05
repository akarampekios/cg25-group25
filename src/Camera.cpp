#include "Camera.h"
#include <algorithm>

Camera::Camera(int width, int height)
    : m_position(2.0f, 2.0f, 2.0f)
    , m_worldUp(0.0f, 1.0f, 0.0f)
    , m_yaw(-90.0f)
    , m_pitch(-60.0f)
    , m_movementSpeed(5.0f)
    , m_mouseSensitivity(0.1f)
    , m_lastX(static_cast<float>(width) / 2.0f)
    , m_lastY(static_cast<float>(height) / 2.0f)
    , m_firstMouse(true)
    , m_width(width)
    , m_height(height)
    , m_fov(45.0f)
    , m_nearPlane(0.1f)
    , m_farPlane(1000.0f) {
    updateCameraVectors();
}

void Camera::update(float deltaTime) {
    // Update any camera-specific logic here
}

void Camera::handleMouseMovement(float xpos, float ypos) {
    if (m_firstMouse) {
        m_lastX = xpos;
        m_lastY = ypos;
        m_firstMouse = false;
    }
    
    float xoffset = xpos - m_lastX;
    float yoffset = m_lastY - ypos; // Reversed since y-coordinates go from bottom to top
    
    m_lastX = xpos;
    m_lastY = ypos;
    
    xoffset *= m_mouseSensitivity;
    yoffset *= m_mouseSensitivity;
    
    m_yaw += xoffset;
    m_pitch += yoffset;
    
    // Constrain pitch
    m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
    
    updateCameraVectors();
}

void Camera::moveForward(float deltaTime) {
    m_position += m_front * m_movementSpeed * deltaTime;
}

void Camera::moveBackward(float deltaTime) {
    m_position -= m_front * m_movementSpeed * deltaTime;
}

void Camera::moveLeft(float deltaTime) {
    m_position -= m_right * m_movementSpeed * deltaTime;
}

void Camera::moveRight(float deltaTime) {
    m_position += m_right * m_movementSpeed * deltaTime;
}

void Camera::moveUp(float deltaTime) {
    m_position += m_worldUp * m_movementSpeed * deltaTime;
}

void Camera::moveDown(float deltaTime) {
    m_position -= m_worldUp * m_movementSpeed * deltaTime;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    glm::mat4 proj = glm::perspective(glm::radians(m_fov),
                                  float(m_width) / float(m_height),
                                  m_nearPlane, m_farPlane);

    // Flip Y for Vulkan (GLM has a helper)
    proj[1][1] *= -1;

    return proj;
}

void Camera::updateCameraVectors() {
    // Calculate the new front vector
    glm::vec3 front;
    front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front.y = sin(glm::radians(m_pitch));
    front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);
    
    // Re-calculate the right and up vector
    m_right = glm::normalize(glm::cross(m_front, m_worldUp));
    m_up = glm::normalize(glm::cross(m_right, m_front));
}
