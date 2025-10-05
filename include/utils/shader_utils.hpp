#pragma once

#include "utils/file_utils.hpp"

namespace ShaderUtils {
inline vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device, const std::vector<char>& code) {
  vk::ShaderModuleCreateInfo const createInfo {
    .codeSize = code.size() * sizeof(char),
    .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };

  return {device, createInfo};
}

inline std::filesystem::path getShaderPath(const std::string& filename) {
  const std::vector<std::filesystem::path> searchDirectories = {
    std::filesystem::path("./build/bin/Release/shaders"),
    std::filesystem::path("../bin/Release/shaders"),
    std::filesystem::path("./build/bin/Debug/shaders"),
    std::filesystem::path("../bin/Debug/shaders"),
    std::filesystem::path("./build/bin/shaders"),
    std::filesystem::path("../bin/shaders"),
    std::filesystem::path("./shaders"),
    std::filesystem::path("../shaders"),
    std::filesystem::path("../../shaders"),
    std::filesystem::path("../../bin/shaders")
};

  for (const auto& dir : searchDirectories) {
    std::error_code ec;
    auto candidate = dir / filename;
    if (std::filesystem::exists(candidate, ec)) {
      auto resolved = std::filesystem::weakly_canonical(candidate, ec);
      if (!ec) {
        return resolved;
      }
      return candidate;
    }
  }

  throw std::runtime_error("Shader not found: " + filename);
}

inline vk::raii::ShaderModule loadShader(const vk::raii::Device& device, const std::string& filename) {
  const std::filesystem::path shaderPath = getShaderPath(filename);
  const auto code = FileUtils::readFile(shaderPath.string());
  return createShaderModule(device, code);
}
}