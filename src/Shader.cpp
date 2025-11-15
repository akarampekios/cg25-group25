#include "Shader.hpp"
#include "FileUtils.hpp"
#include "VulkanCore.hpp"

Shader::Shader(const vk::raii::Device& device, vk::ShaderStageFlagBits stage, const std::string& filename) {
    createShaderModule(device, filename);
    createShaderStage(stage);
}

void Shader::createShaderModule(const vk::raii::Device& device, const std::string& filename) {
    const auto code = FileUtils::readFile(filename);

    vk::ShaderModuleCreateInfo const createInfo{
        .codeSize = code.size() * sizeof(char),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    m_module = {device, createInfo};
}

void Shader::createShaderStage(const vk::ShaderStageFlagBits stage) {
    m_stage = {
        .stage = stage,
        .module = m_module,
        .pName = "main",
    };
}
