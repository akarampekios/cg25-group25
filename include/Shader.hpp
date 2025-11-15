#pragma once

#include <string>
#include <vulkan/vulkan_raii.hpp>
#include "vulkan/vulkan_enums.hpp"

class VulkanCore;

class Shader {
public:
    explicit Shader(const vk::raii::Device& device, vk::ShaderStageFlagBits stage, const std::string& filename);

    const vk::raii::ShaderModule& getModule() const { return m_module; }
    const vk::PipelineShaderStageCreateInfo& getStage() const { return m_stage; }

private:
    vk::raii::ShaderModule m_module = nullptr;
    vk::PipelineShaderStageCreateInfo m_stage;

    void createShaderModule(const vk::raii::Device& device, const std::string& filename);
    void createShaderStage(vk::ShaderStageFlagBits stage);
};
