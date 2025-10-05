#pragma once

class WindowUtils {
public:
  explicit WindowUtils(GLFWwindow* window) : m_window{window} {
  }

  static std::vector<const char*> getRequiredInstanceExtensions() {
    std::uint32_t glfwExtensionCount = 0;
    auto* const glfwExtensions = glfwGetRequiredInstanceExtensions(
        &glfwExtensionCount);
    std::vector const requiredInstanceExtensions(glfwExtensions,
                                                 glfwExtensions +
                                                 glfwExtensionCount);
    return requiredInstanceExtensions;
  }

  static auto getPixelSize(GLFWwindow* window) -> vk::Extent2D {
    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(window, &width, &height);

    return {
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
    };
  }

private:
  GLFWwindow* m_window;
};
