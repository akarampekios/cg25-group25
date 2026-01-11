#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <array>
#include <algorithm>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>

#include "constants.hpp"
#include "Scene.hpp"
#include "ResourceManager.hpp"
#include "VulkanCore.hpp"
#include "BufferManager.hpp"
#include "CommandManager.hpp"
#include "SharedTypes.hpp"
#include "ImageManager.hpp"
#include "FrustumCulling.hpp"

constexpr std::uint32_t AS_REFLECTIVE_OBJECT_MASK = 0x01;
constexpr std::uint32_t AS_SHADOW_OBJECT_MASK = 0x02;
constexpr std::uint32_t AS_UNKNOWN_OBJ_MASK = 0x00;

constexpr std::uint32_t DS_UBO_BINDING = 0;
constexpr std::uint32_t DS_TLAS_BINDING = 1;
constexpr std::uint32_t DS_INSTANCES_BINDING = 2;
constexpr std::uint32_t DS_MESHES_BINDING = 3;
constexpr std::uint32_t DS_UVS_BINDING = 4;
constexpr std::uint32_t DS_INDEX_BINDING = 5;
constexpr std::uint32_t DS_VERTEX_BINDING = 6;

constexpr std::uint32_t DS_MATERIALS_BINDING = 0;
constexpr std::uint32_t DS_BASE_COLOR_TEXTURE_BINDING = 1;
constexpr std::uint32_t DS_METALLIC_ROUGHNESS_TEXTURE_BINDING = 2;
constexpr std::uint32_t DS_NORMAL_TEXTURE_BINDING = 3;
constexpr std::uint32_t DS_EMISSIVE_TEXTURE_BINDING = 4;
constexpr std::uint32_t DS_OCCLUSION_TEXTURE_BINDING = 5;
constexpr std::uint32_t DS_SKYBOX_TEXTURE_BINDING = 6;

constexpr std::uint32_t DS_POINT_LIGHTS_BINDING = 0;
constexpr std::uint32_t DS_SPOT_LIGHTS_BINDING = 1;

namespace {
void appendTextureWrites(
    std::vector<vk::WriteDescriptorSet>& descriptorWrites,
    std::vector<vk::DescriptorImageInfo>& allImageInfos,
    const vk::DescriptorSet dstSet,
    const std::uint32_t dstBinding,
    const vk::raii::Sampler& sampler,
    const std::vector<AllocatedTextureImage>& textureImages
    ) {
    for (std::size_t i = 0; i < textureImages.size(); ++i) {
        allImageInfos.emplace_back(vk::DescriptorImageInfo{
            .sampler = sampler,
            .imageView = textureImages[i].imageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        });
        descriptorWrites.emplace_back(vk::WriteDescriptorSet{
            .dstSet = dstSet,
            .dstBinding = dstBinding,
            .dstArrayElement = static_cast<uint32_t>(i),
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &allImageInfos.back(),
        });
    }
}
}


ResourceManager::ResourceManager(VulkanCore& vulkanCore, CommandManager& commandManager, BufferManager& bufferManager,
                                 ImageManager& imageManager)
    : m_vulkanCore{vulkanCore},
      m_commandManager{commandManager},
      m_bufferManager{bufferManager},
      m_imageManager{imageManager},
      m_cachedCameraViewProj(MAX_FRAMES_IN_FLIGHT, glm::mat4(0.0f)),
      m_indirectDrawBuffersInitialized(MAX_FRAMES_IN_FLIGHT, false) {
    createDescriptorPool();
    createDescriptorSetLayouts();
    createTextureSamplers();
}

void ResourceManager::createDescriptorPool() {
    constexpr vk::DescriptorPoolSize uboPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT);
    constexpr vk::DescriptorPoolSize tlasPoolSize(vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT);

    // We have 5 texture arrays, and we allocate a descriptor set for each frame in flight.
    // In addition, we have one skybox with one descriptor set per frame.
    constexpr std::uint32_t texturesPerFrame = (MAX_TEXTURES_PER_TYPE * 5) + 1; // 5 arrays + 1 skybox
    constexpr vk::DescriptorPoolSize texturesPoolSize(vk::DescriptorType::eCombinedImageSampler, texturesPerFrame * MAX_FRAMES_IN_FLIGHT);

    // Global (5) + Material (1) + Light (2) = 9 SSBOs per frame.
    // We allocate descriptor sets for each type per frame.
    constexpr std::uint32_t storageBuffersPerFrame = 5 + 1 + 2;
    constexpr vk::DescriptorPoolSize
        storagePoolSize(vk::DescriptorType::eStorageBuffer, storageBuffersPerFrame * MAX_FRAMES_IN_FLIGHT);

    std::vector poolSizes = {uboPoolSize, tlasPoolSize, texturesPoolSize, storagePoolSize};

    const vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                 vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = MAX_FRAMES_IN_FLIGHT * 3, // We have 3 descriptor sets per frame
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    m_descriptorPool = vk::raii::DescriptorPool(m_vulkanCore.device(), poolInfo);
}

void ResourceManager::createDescriptorSetLayouts() {
    constexpr vk::DescriptorSetLayoutBinding uboBinding{
        .binding = DS_UBO_BINDING,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding tlasBinding{
        .binding = DS_TLAS_BINDING,
        .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding instancesBinding{
        .binding = DS_INSTANCES_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding meshesBinding{
        .binding = DS_MESHES_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding uvBinding{
        .binding = DS_UVS_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding indexBinding{
        .binding = DS_INDEX_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding vertexBinding{
        .binding = DS_VERTEX_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array globalBindings = {
        uboBinding,
        tlasBinding,
        instancesBinding,
        meshesBinding,
        uvBinding,
        indexBinding,
        vertexBinding,
    };

    const vk::DescriptorSetLayoutCreateInfo globalLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(globalBindings.size()),
        .pBindings = globalBindings.data(),
    };

    m_globalDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        globalLayoutCreateInfo
        );

    constexpr vk::DescriptorSetLayoutBinding materialsBinding{
        .binding = DS_MATERIALS_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding baseColorTextureBinding{
        .binding = DS_BASE_COLOR_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_TEXTURES_PER_TYPE,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding metallicTextureBinding{
        .binding = DS_METALLIC_ROUGHNESS_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_TEXTURES_PER_TYPE,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding normalTextureBinding{
        .binding = DS_NORMAL_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_TEXTURES_PER_TYPE,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding emissiveTextureBinding{
        .binding = DS_EMISSIVE_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_TEXTURES_PER_TYPE,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding occlusionTextureBinding{
        .binding = DS_OCCLUSION_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = MAX_TEXTURES_PER_TYPE,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding skyboxTextureBinding{
        .binding = DS_SKYBOX_TEXTURE_BINDING,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array materialBindings = {
        materialsBinding,
        baseColorTextureBinding,
        metallicTextureBinding,
        normalTextureBinding,
        emissiveTextureBinding,
        occlusionTextureBinding,
        skyboxTextureBinding,
    };

    std::array bindingFlags = {
        vk::DescriptorBindingFlags(0),                                              // materials
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // baseColor
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // metallicRoughness
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // normal
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // emissive
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // occlusion
        vk::DescriptorBindingFlags(vk::DescriptorBindingFlagBits::ePartiallyBound), // skybox (may be skipped on low VRAM)
    };

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .bindingCount = static_cast<uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };

    const vk::DescriptorSetLayoutCreateInfo materialsLayoutCreateInfo{
        .pNext = &flagsInfo,
        .bindingCount = static_cast<std::uint32_t>(materialBindings.size()),
        .pBindings = materialBindings.data(),
    };

    m_materialDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        materialsLayoutCreateInfo
        );

    constexpr vk::DescriptorSetLayoutBinding pointLightsBinding{
        .binding = DS_POINT_LIGHTS_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    constexpr vk::DescriptorSetLayoutBinding spotLightsBinding{
        .binding = DS_SPOT_LIGHTS_BINDING,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .pImmutableSamplers = nullptr,
    };

    std::array lightBindings = {pointLightsBinding, spotLightsBinding};

    const vk::DescriptorSetLayoutCreateInfo lightsLayoutCreateInfo{
        .bindingCount = static_cast<std::uint32_t>(lightBindings.size()),
        .pBindings = lightBindings.data(),
    };

    m_lightDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        m_vulkanCore.device(),
        lightsLayoutCreateInfo
        );
}

void ResourceManager::createTextureSamplers() {
    m_skyboxSampler = m_imageManager.createSkyboxSampler();
    m_baseColorTextureSampler = m_imageManager.createSampler(true);
    m_metallicRoughnessTextureSampler = m_imageManager.createSampler(false);
    m_normalTextureSampler = m_imageManager.createSampler(false);
    m_emissiveTextureSampler = m_imageManager.createSampler(true);
    m_occlusionTextureSampler = m_imageManager.createSampler(false);
}

void ResourceManager::allocateSceneResources(const Scene& scene) {
    allocateVertexBuffer(scene.vertices);
    allocateIndexBuffer(scene.indices);

    createUniformBuffers();
    createInstanceBuffers(scene);
    createMeshesBuffer(scene);
    createUVBuffer(scene);
    createMaterialBuffers(scene);
    createLightBuffers(scene);
    createIndirectDrawBuffers(scene);
    
    createTextureImages(scene);
    createSkyboxImage(scene);

    createAccelerationStructures(scene);
    
    createGlobalDescriptorSets(scene);
    createMaterialDescriptorSets(scene);
    createLightingDescriptorSets(scene);
}

void ResourceManager::updateSceneResources(const Scene& scene,
                                           const float time,
                                           const std::uint32_t frameIdx,
                                           glm::vec2 jitterOffset) {
    updateUniformBuffer(scene, time, frameIdx, jitterOffset);
    updateInstanceBuffers(scene, frameIdx);
    updateLightBuffers(scene, frameIdx);
    updateIndirectDrawBuffers(scene, frameIdx);
    
    // TLAS update is now recorded directly into the command buffer via recordTLASUpdate()
    // This eliminates the waitIdle() stall!
    
    // we do not animate materials (blender export limitations), but it saves us some fps      
    // updateMaterialBuffers(scene, frameIdx);
}


void ResourceManager::allocateVertexBuffer(const std::vector<Vertex>& vertices) {
    const vk::DeviceSize bufferSize = vertices.empty() ? sizeof(Vertex) : sizeof(Vertex) * vertices.size();
    const void* data = vertices.empty() ? nullptr : vertices.data();
    m_bufferManager.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst
        | vk::BufferUsageFlagBits::eVertexBuffer
        | vk::BufferUsageFlagBits::eShaderDeviceAddress
        | vk::BufferUsageFlagBits::eStorageBuffer
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_vertexBuffer,
        m_vertexBufferMemory,
        data
        );
}

void ResourceManager::allocateIndexBuffer(const std::vector<std::uint32_t>& indices) {
    const vk::DeviceSize bufferSize = indices.empty() ? sizeof(std::uint32_t) : sizeof(std::uint32_t) * indices.size();
    const void* data = indices.empty() ? nullptr : indices.data();
    m_bufferManager.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst
        | vk::BufferUsageFlagBits::eIndexBuffer
        | vk::BufferUsageFlagBits::eShaderDeviceAddress
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR
        | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_indexBuffer,
        m_indexBufferMemory,
        data
        );
}

void ResourceManager::createUniformBuffers() {
    m_uniformBuffers.clear();
    m_uniformBuffersMemory.clear();
    m_uniformBuffersMapped.clear();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});

        m_bufferManager.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            buffer,
            bufferMem
            );

        m_uniformBuffers.emplace_back(std::move(buffer));
        m_uniformBuffersMemory.emplace_back(std::move(bufferMem));
        m_uniformBuffersMapped.emplace_back(m_uniformBuffersMemory[i].mapMemory(0, bufferSize));
    }
}

void ResourceManager::createInstanceBuffers(const Scene& scene) {
    m_instanceBuffers.clear();
    m_instanceBuffersMemory.clear();
    m_instanceBuffersMapped.clear();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DeviceSize bufferSize = scene.instances.empty()
                                              ? sizeof(Instance)
                                              : sizeof(Instance) * scene.instances.size();

        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});

        m_bufferManager.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            buffer,
            bufferMem
            );

        m_instanceBuffers.emplace_back(std::move(buffer));
        m_instanceBuffersMemory.emplace_back(std::move(bufferMem));
        m_instanceBuffersMapped.emplace_back(m_instanceBuffersMemory[i].mapMemory(0, bufferSize));
        
        // Initialize buffer with instance data
        if (!scene.instances.empty()) {
            memcpy(m_instanceBuffersMapped[i], scene.instances.data(), sizeof(Instance) * scene.instances.size());
        }
    }
}

void ResourceManager::createMeshesBuffer(const Scene& scene) {
    const vk::DeviceSize bufferSize = scene.meshes.empty() ? sizeof(Mesh) : sizeof(Mesh) * scene.meshes.size();
    const void* data = scene.meshes.empty() ? nullptr : scene.meshes.data();
    m_bufferManager.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_meshesBuffer,
        m_meshesBufferMemory,
        data
        );
}

void ResourceManager::createUVBuffer(const Scene& scene) {
    const vk::DeviceSize bufferSize = scene.uvs.empty() ? sizeof(glm::vec2) : sizeof(glm::vec2) * scene.uvs.size();
    const void* data = scene.uvs.empty() ? nullptr : scene.uvs.data();
    m_bufferManager.createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        m_uvBuffer,
        m_uvBufferMemory,
        data
        );
}

void ResourceManager::createMaterialBuffers(const Scene& scene) {
    m_materialBuffers.clear();
    m_materialBuffersMemory.clear();
    m_materialBuffersMapped.clear();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DeviceSize bufferSize = scene.materials.empty()
                                              ? sizeof(Material)
                                              : sizeof(Material) * scene.materials.size();

        vk::raii::Buffer buffer({});
        vk::raii::DeviceMemory bufferMem({});

        m_bufferManager.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            buffer,
            bufferMem
            );

        m_materialBuffers.emplace_back(std::move(buffer));
        m_materialBuffersMemory.emplace_back(std::move(bufferMem));
        m_materialBuffersMapped.emplace_back(m_materialBuffersMemory[i].mapMemory(0, bufferSize));
    }

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        updateMaterialBuffers(scene, i);
    }
}

void ResourceManager::createLightBuffers(const Scene& scene) {
    m_pointLightBuffers.clear();
    m_pointLightBuffersMemory.clear();
    m_pointLightBuffersMapped.clear();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DeviceSize pointLightBufferSize = scene.pointLights.empty()
                                                        ? sizeof(PointLight)
                                                        : sizeof(PointLight) * scene.pointLights.size();

        vk::raii::Buffer pointLightBuffer({});
        vk::raii::DeviceMemory pointLightBufferMemory({});

        m_bufferManager.createBuffer(
            pointLightBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            pointLightBuffer,
            pointLightBufferMemory
            );

        m_pointLightBuffers.emplace_back(std::move(pointLightBuffer));
        m_pointLightBuffersMemory.emplace_back(std::move(pointLightBufferMemory));
        m_pointLightBuffersMapped.emplace_back(m_pointLightBuffersMemory[i].mapMemory(0, pointLightBufferSize));
    }

    m_spotLightBuffers.clear();
    m_spotLightBuffersMemory.clear();
    m_spotLightBuffersMapped.clear();

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DeviceSize spotLightBufferSize = scene.spotLights.empty()
                                                       ? sizeof(SpotLight)
                                                       : sizeof(SpotLight) * scene.spotLights.size();

        vk::raii::Buffer spotLightBuffer({});
        vk::raii::DeviceMemory spotLightBufferMemory({});

        m_bufferManager.createBuffer(
            spotLightBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            spotLightBuffer,
            spotLightBufferMemory
            );

        m_spotLightBuffers.emplace_back(std::move(spotLightBuffer));
        m_spotLightBuffersMemory.emplace_back(std::move(spotLightBufferMemory));
        m_spotLightBuffersMapped.emplace_back(m_spotLightBuffersMemory[i].mapMemory(0, spotLightBufferSize));
    }

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        updateLightBuffers(scene, i);
    }
}

void ResourceManager::createTextureImages(const Scene& scene) {
    if (scene.baseColorTextures.empty() && scene.metallicRoughnessTextures.empty() && scene.normalTextures.empty() &&
        scene.emissiveTextures.empty() && scene.occlusionTextures.empty()) {
        return;
    }

    m_baseColorTextureImages.resize(scene.baseColorTextures.size());
    m_metallicTextureImages.resize(scene.metallicRoughnessTextures.size());
    m_normalTextureImages.resize(scene.normalTextures.size());
    m_emissiveTextureImages.resize(scene.emissiveTextures.size());
    m_occlusionTextureImages.resize(scene.occlusionTextures.size());

    // TDR prevention: track texture count and periodically flush GPU
    std::uint32_t textureCounter = 0;
    auto tdrPreventionCheck = [this, &textureCounter]() {
        textureCounter++;
        if (g_textureConfig.tdrPreventionBatchSize > 0 && 
            textureCounter % g_textureConfig.tdrPreventionBatchSize == 0) {
            m_vulkanCore.device().waitIdle();
            if (g_textureConfig.tdrPreventionDelayMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(g_textureConfig.tdrPreventionDelayMs));
            }
        }
    };

    for (std::size_t i = 0; i < scene.baseColorTextures.size(); i++) {
        const auto& texture = scene.baseColorTextures[i];
        m_imageManager.createImageFromTexture(
            texture,
            m_baseColorTextureImages[i].image,
            m_baseColorTextureImages[i].imageView,
            m_baseColorTextureImages[i].imageMemory
            );
        tdrPreventionCheck();
    }

    for (std::size_t i = 0; i < scene.metallicRoughnessTextures.size(); i++) {
        const auto& texture = scene.metallicRoughnessTextures[i];
        m_imageManager.createImageFromTexture(
            texture,
            m_metallicTextureImages[i].image,
            m_metallicTextureImages[i].imageView,
            m_metallicTextureImages[i].imageMemory
            );
        tdrPreventionCheck();
    }

    for (std::size_t i = 0; i < scene.normalTextures.size(); i++) {
        const auto& texture = scene.normalTextures[i];
        m_imageManager.createImageFromTexture(
            texture,
            m_normalTextureImages[i].image,
            m_normalTextureImages[i].imageView,
            m_normalTextureImages[i].imageMemory
            );
        tdrPreventionCheck();
    }

    if (g_textureConfig.skipEmissiveTextures) {
        m_emissiveTextureImages.clear();
    } else {
        for (std::size_t i = 0; i < scene.emissiveTextures.size(); i++) {
            auto texture = scene.emissiveTextures[i];
            texture.mipLevels = 1;
            texture.format = vk::Format::eR8G8B8A8Srgb;
            
            m_imageManager.createImageFromTexture(
                texture,
                m_emissiveTextureImages[i].image,
                m_emissiveTextureImages[i].imageView,
                m_emissiveTextureImages[i].imageMemory
                );
            
            // TDR prevention: periodic flush
            tdrPreventionCheck();
        }
    }

    for (std::size_t i = 0; i < scene.occlusionTextures.size(); i++) {
        const auto& texture = scene.occlusionTextures[i];
        m_imageManager.createImageFromTexture(
            texture,
            m_occlusionTextureImages[i].image,
            m_occlusionTextureImages[i].imageView,
            m_occlusionTextureImages[i].imageMemory
            );
        tdrPreventionCheck();
    }
}

void ResourceManager::createSkyboxImage(const Scene& scene) {
    if (g_textureConfig.skipEmissiveTextures) {
        return;
    }

    if (scene.skySphereTextureIndex < 0) {
        return;
    }

    auto texture = scene.emissiveTextures[scene.skySphereTextureIndex];
    
    // Skybox texture: limit mip levels and force safe RGBA format
    texture.mipLevels = 1;
    texture.format = vk::Format::eR8G8B8A8Srgb;

    m_imageManager.createImageFromTexture(
        texture,
        m_skyboxImage.image,
        m_skyboxImage.imageView,
        m_skyboxImage.imageMemory
        );
}


void ResourceManager::createAccelerationStructures(const Scene& scene) {
    createBLAS(scene);
    createBLASInstances(scene);
    createTLAS();
}

void ResourceManager::createGlobalDescriptorSets(const Scene& scene) {
    m_globalDescriptorSets.clear();

    std::vector<vk::DescriptorSetLayout> globalLayouts(MAX_FRAMES_IN_FLIGHT, m_globalDescriptorSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfoGlobal{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(globalLayouts.size()),
        .pSetLayouts = globalLayouts.data()
    };

    m_globalDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(allocInfoGlobal);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DescriptorBufferInfo uboInfo{
            .buffer = m_uniformBuffers[i],
            .offset = 0,
            .range = sizeof(UniformBufferObject)
        };

        const vk::WriteDescriptorSet uboWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_UBO_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &uboInfo
        };

        const vk::WriteDescriptorSetAccelerationStructureKHR asInfo{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &*m_tlasHandles[i],
        };

        const vk::WriteDescriptorSet asWrite{
            .pNext = &asInfo,
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_TLAS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR
        };

        const vk::DescriptorBufferInfo instancesInfo{
            .buffer = m_instanceBuffers[i],
            .offset = 0,
            .range = scene.instances.empty() ? VK_WHOLE_SIZE : sizeof(Instance) * scene.instances.size(),
        };

        const vk::WriteDescriptorSet instancesWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_INSTANCES_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &instancesInfo
        };

        const vk::DescriptorBufferInfo meshesInfo{
            .buffer = m_meshesBuffer,
            .offset = 0,
            .range = scene.meshes.empty() ? VK_WHOLE_SIZE : sizeof(Mesh) * scene.meshes.size(),
        };

        const vk::WriteDescriptorSet meshesWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_MESHES_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &meshesInfo
        };

        const vk::DescriptorBufferInfo uvInfo{
            .buffer = m_uvBuffer,
            .offset = 0,
            .range = scene.uvs.empty() ? VK_WHOLE_SIZE : sizeof(glm::vec2) * scene.uvs.size()
        };

        const vk::WriteDescriptorSet uvWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_UVS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &uvInfo
        };

        const vk::DescriptorBufferInfo indexInfo{
            .buffer = m_indexBuffer,
            .offset = 0,
            .range = scene.indices.empty() ? VK_WHOLE_SIZE : sizeof(std::uint32_t) * scene.indices.size()
        };

        const vk::WriteDescriptorSet indexWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_INDEX_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &indexInfo
        };

        const vk::DescriptorBufferInfo vertexInfo{
            .buffer = m_vertexBuffer,
            .offset = 0,
            .range = scene.vertices.empty() ? VK_WHOLE_SIZE : sizeof(Vertex) * scene.vertices.size()
        };

        const vk::WriteDescriptorSet vertexWrite{
            .dstSet = m_globalDescriptorSets[i],
            .dstBinding = DS_VERTEX_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &vertexInfo
        };

        const std::vector descriptorWrites{
            uboWrite,
            asWrite,
            instancesWrite,
            meshesWrite,
            uvWrite,
            indexWrite,
            vertexWrite,
        };

        m_vulkanCore.device().updateDescriptorSets(descriptorWrites, {});
    }
}

void ResourceManager::createMaterialDescriptorSets(const Scene& scene) {
    m_materialDescriptorSets.clear();

    std::vector<vk::DescriptorSetLayout> materialLayouts(MAX_FRAMES_IN_FLIGHT, m_materialDescriptorSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfoMaterial{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(materialLayouts.size()),
        .pSetLayouts = materialLayouts.data()
    };

    m_materialDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(allocInfoMaterial);

    std::vector<vk::DescriptorImageInfo> allImageInfos;
    allImageInfos.reserve(
        (scene.baseColorTextures.size() + scene.metallicRoughnessTextures.size() + scene.normalTextures.size() +
         scene.emissiveTextures.size() + scene.occlusionTextures.size()) * MAX_FRAMES_IN_FLIGHT + MAX_FRAMES_IN_FLIGHT);

    for (std::size_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; frameIdx++) {
        const vk::DescriptorBufferInfo materialsInfo{
            .buffer = m_materialBuffers[frameIdx],
            .offset = 0,
            .range = scene.materials.empty() ? VK_WHOLE_SIZE : sizeof(Material) * scene.materials.size(),
        };

        const vk::WriteDescriptorSet materialsWrite{
            .dstSet = m_materialDescriptorSets[frameIdx],
            .dstBinding = DS_MATERIALS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &materialsInfo
        };

        std::vector descriptorWrites{materialsWrite};

        appendTextureWrites(descriptorWrites, allImageInfos, m_materialDescriptorSets[frameIdx],
                            DS_BASE_COLOR_TEXTURE_BINDING, m_baseColorTextureSampler, m_baseColorTextureImages);
        appendTextureWrites(descriptorWrites, allImageInfos, m_materialDescriptorSets[frameIdx],
                            DS_METALLIC_ROUGHNESS_TEXTURE_BINDING, m_metallicRoughnessTextureSampler,
                            m_metallicTextureImages);
        appendTextureWrites(descriptorWrites, allImageInfos, m_materialDescriptorSets[frameIdx],
                            DS_NORMAL_TEXTURE_BINDING, m_normalTextureSampler, m_normalTextureImages);
        appendTextureWrites(descriptorWrites, allImageInfos, m_materialDescriptorSets[frameIdx],
                            DS_EMISSIVE_TEXTURE_BINDING, m_emissiveTextureSampler, m_emissiveTextureImages);
        appendTextureWrites(descriptorWrites, allImageInfos, m_materialDescriptorSets[frameIdx],
                            DS_OCCLUSION_TEXTURE_BINDING, m_occlusionTextureSampler, m_occlusionTextureImages);

        if (*m_skyboxImage.imageView) {
            allImageInfos.emplace_back(vk::DescriptorImageInfo{
                .sampler = m_skyboxSampler,
                .imageView = m_skyboxImage.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            });

            descriptorWrites.emplace_back(vk::WriteDescriptorSet{
                .dstSet = m_materialDescriptorSets[frameIdx],
                .dstBinding = DS_SKYBOX_TEXTURE_BINDING,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &allImageInfos.back()
            });
        }

        m_vulkanCore.device().updateDescriptorSets(descriptorWrites, {});
    }
}

void ResourceManager::createLightingDescriptorSets(const Scene& scene) {
    m_lightingDescriptorSets.clear();

    std::vector<vk::DescriptorSetLayout> lightingLayouts(MAX_FRAMES_IN_FLIGHT, m_lightDescriptorSetLayout);

    const vk::DescriptorSetAllocateInfo allocInfoLighting{
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(lightingLayouts.size()),
        .pSetLayouts = lightingLayouts.data()
    };

    m_lightingDescriptorSets = m_vulkanCore.device().allocateDescriptorSets(allocInfoLighting);

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        const vk::DeviceSize pointLightBufferSize = scene.pointLights.empty()
                                                        ? sizeof(PointLight)
                                                        : sizeof(PointLight) * scene.pointLights.size();

        const vk::DescriptorBufferInfo pointLightInfo{
            .buffer = m_pointLightBuffers[i],
            .offset = 0,
            .range = pointLightBufferSize,
        };

        const vk::WriteDescriptorSet pointLightWrite{
            .dstSet = m_lightingDescriptorSets[i],
            .dstBinding = DS_POINT_LIGHTS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &pointLightInfo
        };

        const vk::DeviceSize spotLightBufferSize = scene.spotLights.empty()
                                                       ? sizeof(SpotLight)
                                                       : sizeof(SpotLight) * scene.spotLights.size();

        const vk::DescriptorBufferInfo spotLightInfo{
            .buffer = m_spotLightBuffers[i],
            .offset = 0,
            .range = spotLightBufferSize,
        };

        const vk::WriteDescriptorSet spotLightWrite{
            .dstSet = m_lightingDescriptorSets[i],
            .dstBinding = DS_SPOT_LIGHTS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &spotLightInfo
        };

        const std::vector descriptorWrites{pointLightWrite, spotLightWrite};
        m_vulkanCore.device().updateDescriptorSets(descriptorWrites, {});
    }
}


void ResourceManager::createBLAS(const Scene& scene) {
    vk::BufferDeviceAddressInfo const vertexAddressInfo{.buffer = m_vertexBuffer};
    vk::DeviceAddress const vertexAddress = m_vulkanCore.device().getBufferAddress(vertexAddressInfo);

    vk::BufferDeviceAddressInfo const indexAddressInfo{.buffer = m_indexBuffer};
    vk::DeviceAddress const indexAddress = m_vulkanCore.device().getBufferAddress(indexAddressInfo);

    m_blasBuffers.reserve(scene.meshes.size());
    m_blasMemories.reserve(scene.meshes.size());
    m_blasHandles.reserve(scene.meshes.size());

    for (std::size_t i = 0; i < scene.meshes.size(); ++i) {
        const auto& mesh = scene.meshes[i];

        const vk::AccelerationStructureGeometryTrianglesDataKHR trianglesData{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = vertexAddress + (mesh.baseVertex * sizeof(Vertex)),
            .vertexStride = sizeof(Vertex),
            .maxVertex = mesh.vertexCount,
            .indexType = vk::IndexType::eUint32,
            .indexData = indexAddress + (mesh.baseIndex * sizeof(std::uint32_t)),
        };

        vk::AccelerationStructureGeometryDataKHR const geometryData(trianglesData);

        vk::AccelerationStructureGeometryKHR blasGeometry{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = geometryData,
            .flags = vk::GeometryFlagsKHR(0),
        };

        if (mesh.materialIndex != -1) {
            if (scene.materials[mesh.materialIndex].alphaMode == 0) {
                blasGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque;
            }
        }

        vk::AccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &blasGeometry,
        };

        auto primitiveCount = mesh.indexCount / 3U;

        vk::AccelerationStructureBuildSizesInfoKHR blasBuildSizes =
            m_vulkanCore.device().getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice,
                blasBuildGeometryInfo,
                {primitiveCount}
                );

        vk::raii::Buffer scratchBuffer = nullptr;
        vk::raii::DeviceMemory scratchMemory = nullptr;
        m_bufferManager.createBuffer(
            blasBuildSizes.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            scratchBuffer,
            scratchMemory
            );

        vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *scratchBuffer};
        vk::DeviceAddress scratchAddress = m_vulkanCore.device().getBufferAddress(scratchAddressInfo);
        blasBuildGeometryInfo.scratchData.deviceAddress = scratchAddress;

        vk::raii::Buffer blasBuffer = nullptr;
        vk::raii::DeviceMemory blasMemory = nullptr;

        m_blasBuffers.emplace_back(std::move(blasBuffer));
        m_blasMemories.emplace_back(std::move(blasMemory));

        m_bufferManager.createBuffer(
            blasBuildSizes.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            m_blasBuffers[i],
            m_blasMemories[i]
            );

        vk::AccelerationStructureCreateInfoKHR const blasCreateInfo{
            .buffer = m_blasBuffers[i],
            .offset = 0,
            .size = blasBuildSizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        };

        m_blasHandles.emplace_back(m_vulkanCore.device().createAccelerationStructureKHR(blasCreateInfo));
        blasBuildGeometryInfo.dstAccelerationStructure = m_blasHandles[i];

        vk::AccelerationStructureBuildRangeInfoKHR blasRangeInfo{
            .primitiveCount = primitiveCount,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
        };

        m_commandManager.immediateSubmit([&](const vk::CommandBuffer cmd) {
            cmd.buildAccelerationStructuresKHR({blasBuildGeometryInfo}, {&blasRangeInfo});
        });
    }
}

void ResourceManager::createBLASInstances(const Scene& scene) {
    m_blasInstances.reserve(scene.instances.size());

    for (std::size_t i = 0; i < scene.instances.size(); ++i) {
        const auto& instance = scene.instances[i];

        vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{
            .accelerationStructure = *m_blasHandles[instance.meshIndex]
        };

        const vk::DeviceAddress blasDeviceAddr = m_vulkanCore.device().getAccelerationStructureAddressKHR(addrInfo);

        const auto& t = instance.transform;
        vk::TransformMatrixKHR transformMatrix{};
        transformMatrix.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{t[0][0], t[1][0], t[2][0], t[3][0]},
            std::array<float,4>{t[0][1], t[1][1], t[2][1], t[3][1]},
            std::array<float,4>{t[0][2], t[1][2], t[2][2], t[3][2]}
        }};

        // Start with no mask (invisible to all ray types)
        std::uint32_t mask = 0;

        // Check instance-level properties first
        bool instanceReflective = instance.reflective != 0;
        bool instanceCastsShadows = instance.castsShadows != 0;

        // Check material-level properties
        bool materialReflective = true;
        bool materialCastsShadows = true;

        if (instance.meshIndex >= 0 && instance.meshIndex < static_cast<std::int32_t>(scene.meshes.size())) {
            const auto& mesh = scene.meshes[instance.meshIndex];
            if (mesh.materialIndex >= 0 && mesh.materialIndex < static_cast<std::int32_t>(scene.materials.size())) {
                const auto& material = scene.materials[mesh.materialIndex];
                materialReflective = material.reflective != 0;
                materialCastsShadows = material.castsShadows != 0;
            }
        }

        if (instanceReflective || materialReflective) {
            mask |= AS_REFLECTIVE_OBJECT_MASK;
        }
        if (instanceCastsShadows || materialCastsShadows) {
            mask |= AS_SHADOW_OBJECT_MASK;
        }

        vk::AccelerationStructureInstanceKHR asInstance{
            .transform = transformMatrix,
            .instanceCustomIndex = static_cast<uint32_t>(i),
            .mask = mask,
            .instanceShaderBindingTableRecordOffset = 0,
            .accelerationStructureReference = blasDeviceAddr,
        };

        m_blasInstances.push_back(asInstance);
    }

    const vk::DeviceSize instBufferSize = sizeof(vk::AccelerationStructureInstanceKHR) * m_blasInstances.size();

    m_blasInstancesBuffers.clear();
    m_blasInstancesMemories.clear();
    m_blasInstancesBuffersMapped.clear();

    m_blasInstancesBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    m_blasInstancesMemories.reserve(MAX_FRAMES_IN_FLIGHT);
    m_blasInstancesBuffersMapped.reserve(MAX_FRAMES_IN_FLIGHT);

    for (std::uint32_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; ++frameIdx) {
        vk::raii::Buffer instancesBuffer{nullptr};
        vk::raii::DeviceMemory instancesMemory{nullptr};

        // Host-visible + coherent so we can update instance transforms every frame without staging.
        m_bufferManager.createBuffer(
            instBufferSize,
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            instancesBuffer,
            instancesMemory,
            m_blasInstances.data()
        );

        void* mapped = instancesMemory.mapMemory(0, instBufferSize);

        m_blasInstancesBuffers.emplace_back(std::move(instancesBuffer));
        m_blasInstancesMemories.emplace_back(std::move(instancesMemory));
        m_blasInstancesBuffersMapped.emplace_back(mapped);
    }
}

void ResourceManager::createTLAS() {
    m_tlasBuffers.clear();
    m_tlasMemories.clear();
    m_tlasScratchBuffers.clear();
    m_tlasScratchMemories.clear();
    m_tlasHandles.clear();

    m_tlasBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    m_tlasMemories.reserve(MAX_FRAMES_IN_FLIGHT);
    m_tlasScratchBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    m_tlasScratchMemories.reserve(MAX_FRAMES_IN_FLIGHT);
    m_tlasHandles.reserve(MAX_FRAMES_IN_FLIGHT);

    for (std::uint32_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; ++frameIdx) {
        vk::BufferDeviceAddressInfo instanceAddrInfo{.buffer = m_blasInstancesBuffers[frameIdx]};
        vk::DeviceAddress instanceAddr = m_vulkanCore.device().getBufferAddress(instanceAddrInfo);

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
            .arrayOfPointers = false,
            .data = instanceAddr,
        };

        vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

        vk::AccelerationStructureGeometryKHR tlasGeometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = geometryData
        };

        vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &tlasGeometry,
        };

        auto primitiveCount = static_cast<uint32_t>(m_blasInstances.size());

        vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizes =
            m_vulkanCore.device().getAccelerationStructureBuildSizesKHR(
                vk::AccelerationStructureBuildTypeKHR::eDevice,
                tlasBuildGeometryInfo,
                {primitiveCount}
            );

        // IMPORTANT: updateScratchSize can be > buildScratchSize (driver-dependent).
        const vk::DeviceSize scratchSize = std::max(tlasBuildSizes.buildScratchSize, tlasBuildSizes.updateScratchSize);

        vk::raii::Buffer tlasScratchBuffer{nullptr};
        vk::raii::DeviceMemory tlasScratchMemory{nullptr};
        m_bufferManager.createBuffer(
            scratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasScratchBuffer,
            tlasScratchMemory
        );

        vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *tlasScratchBuffer};
        vk::DeviceAddress scratchAddr = m_vulkanCore.device().getBufferAddress(scratchAddressInfo);
        tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

        vk::raii::Buffer tlasBuffer{nullptr};
        vk::raii::DeviceMemory tlasMemory{nullptr};
        m_bufferManager.createBuffer(
            tlasBuildSizes.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            tlasBuffer,
            tlasMemory
        );

        vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{
            .buffer = tlasBuffer,
            .offset = 0,
            .size = tlasBuildSizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        };

        auto tlasHandle = m_vulkanCore.device().createAccelerationStructureKHR(tlasCreateInfo);
        tlasBuildGeometryInfo.dstAccelerationStructure = tlasHandle;

        vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
            .primitiveCount = primitiveCount,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0
        };

        m_commandManager.immediateSubmit([&](const vk::CommandBuffer cmd) {
            cmd.buildAccelerationStructuresKHR({tlasBuildGeometryInfo}, {&tlasRangeInfo});
        });

        m_tlasScratchBuffers.emplace_back(std::move(tlasScratchBuffer));
        m_tlasScratchMemories.emplace_back(std::move(tlasScratchMemory));
        m_tlasBuffers.emplace_back(std::move(tlasBuffer));
        m_tlasMemories.emplace_back(std::move(tlasMemory));
        m_tlasHandles.emplace_back(std::move(tlasHandle));
    }
}

void ResourceManager::updateUniformBuffer(const Scene& scene, const float time, const std::uint32_t frameIdx, glm::vec2 jitterOffset) {
    const auto view = scene.camera.getView();
    auto proj = scene.camera.getProjection();
    
    // TAA: Apply jitter to projection matrix
    if constexpr (TAA_ENABLED) {
        // Convert jitter from pixels to NDC
        const float jitterX = (jitterOffset.x * 2.0f) / static_cast<float>(WINDOW_WIDTH);
        const float jitterY = (jitterOffset.y * 2.0f) / static_cast<float>(WINDOW_HEIGHT);
        
        // Apply jitter to projection matrix (affects clip space position)
        proj[2][0] += jitterX;
        proj[2][1] += jitterY;
    }
    
    // TAA: Get previous frame matrices (or current if first frame)
    const glm::mat4 prevView = m_firstFrame ? view : m_prevViewMatrix;
    const glm::mat4 prevProj = m_firstFrame ? scene.camera.getProjection() : m_prevProjMatrix;

    const UniformBufferObject ubo{
        .view = view,
        .proj = proj,
        .viewInverse = glm::inverse(view),
        .projInverse = glm::inverse(proj),
        .prevView = prevView,
        .prevProj = prevProj,
        .cameraPos = scene.camera.getPosition(),
        .time = time,
        .pointLightsCount = static_cast<std::uint32_t>(scene.pointLights.size()),
        .spotLightsCount = static_cast<std::uint32_t>(scene.spotLights.size()),
        .directionalLight = scene.directionalLight,
        .skySphereInstanceIndex = scene.skySphereInstanceIndex,
        .skySphereTextureIndex = scene.skySphereTextureIndex,
        .jitterOffset = jitterOffset,
        .fogColor = scene.fog.fogColor,
        .fogDensity = scene.fog.fogDensity,
        .screenSize = glm::vec2(static_cast<float>(WINDOW_WIDTH), static_cast<float>(WINDOW_HEIGHT)),
    };

    memcpy(m_uniformBuffersMapped[frameIdx], &ubo, sizeof(ubo));
    
    // TAA: Store current matrices for next frame
    m_prevViewMatrix = view;
    m_prevProjMatrix = scene.camera.getProjection();  // Store unjittered projection
    m_firstFrame = false;
}

void ResourceManager::updateTopLevelAccelerationStructures(const Scene& scene, bool initialBuild, std::uint32_t frameIdx) {
    auto primitiveCount = static_cast<uint32_t>(scene.instances.size());

    // Get pointer to persistently mapped buffer for this frame
    auto* instancesPtr = static_cast<vk::AccelerationStructureInstanceKHR*>(m_blasInstancesBuffersMapped[frameIdx]);
    
    // Only update transforms for animated instances
    std::uint32_t updatedCount = 0;
    for (std::size_t i = 0; i < primitiveCount; ++i) {
        const auto& instance = scene.instances[i];

        // Skip static instances unless it's the initial build
        if (instance.animated == 0 && !initialBuild) {
            continue;
        }
    
        const auto& t = instance.transform;
        vk::TransformMatrixKHR transformMatrix{};
        transformMatrix.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{t[0][0], t[1][0], t[2][0], t[3][0]},
            std::array<float,4>{t[0][1], t[1][1], t[2][1], t[3][1]},
            std::array<float,4>{t[0][2], t[1][2], t[2][2], t[3][2]}
        }};
        
        // Update both the CPU-side cache and GPU-mapped memory directly
        m_blasInstances[i].setTransform(transformMatrix);
        instancesPtr[i].transform = transformMatrix;
        updatedCount++;
    }
    
    // Early exit if nothing was updated and this isn't the initial build
    if (updatedCount == 0 && !initialBuild) {
        return;
    }
    
    // No need to map/unmap - memory is persistently mapped and HostCoherent
    // so writes are automatically visible to the GPU

    vk::BufferDeviceAddressInfo instanceAddrInfo{.buffer = m_blasInstancesBuffers[frameIdx]};
    vk::DeviceAddress instanceAddr = m_vulkanCore.device().getBufferAddress(instanceAddrInfo);

    auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR{
        .arrayOfPointers = false,
        .data = instanceAddr
    };

    vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

    vk::AccelerationStructureGeometryKHR tlasGeometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = geometryData
    };

    vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        .mode = vk::BuildAccelerationStructureModeKHR::eUpdate,
        .srcAccelerationStructure = m_tlasHandles[frameIdx],
        .dstAccelerationStructure = m_tlasHandles[frameIdx],
        .geometryCount = 1,
        .pGeometries = &tlasGeometry
    };

    vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *m_tlasScratchBuffers[frameIdx]};
    vk::DeviceAddress scratchAddr = m_vulkanCore.device().getBufferAddress(scratchAddressInfo);
    tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

    vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };

    m_commandManager.immediateSubmit([&](const vk::CommandBuffer cmd) {
        const vk::MemoryBarrier preBarrier{
            .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR | vk::AccessFlagBits::eTransferWrite |
                             vk::AccessFlagBits::eShaderRead,
            .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR |
                             vk::AccessFlagBits::eAccelerationStructureWriteKHR
        };

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eTransfer |
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {},
            preBarrier,
            {},
            {}
            );

        cmd.buildAccelerationStructuresKHR({tlasBuildGeometryInfo}, {&tlasRangeInfo});

        const vk::MemoryBarrier postBarrier{
            .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
            .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead
        };

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eFragmentShader,
            {},
            postBarrier,
            {},
            {}
            );
    });
}

void ResourceManager::recordTLASUpdate(const vk::CommandBuffer& cmd, const Scene& scene, bool initialBuild, std::uint32_t frameIdx) {
    auto primitiveCount = static_cast<uint32_t>(scene.instances.size());

    // Get pointer to persistently mapped buffer
    auto* instancesPtr = static_cast<vk::AccelerationStructureInstanceKHR*>(m_blasInstancesBuffersMapped[frameIdx]);
    
    // Only update transforms for animated instances
    std::uint32_t updatedCount = 0;
    for (std::size_t i = 0; i < primitiveCount; ++i) {
        const auto& instance = scene.instances[i];

        // Skip static instances unless it's the initial build
        if (instance.animated == 0 && !initialBuild) {
            continue;
        }
    
        const auto& t = instance.transform;
        vk::TransformMatrixKHR transformMatrix{};
        transformMatrix.matrix = std::array<std::array<float,4>,3>{{
            std::array<float,4>{t[0][0], t[1][0], t[2][0], t[3][0]},
            std::array<float,4>{t[0][1], t[1][1], t[2][1], t[3][1]},
            std::array<float,4>{t[0][2], t[1][2], t[2][2], t[3][2]}
        }};
        
        // Update both the CPU-side cache and GPU-mapped memory directly
        m_blasInstances[i].setTransform(transformMatrix);
        instancesPtr[i].transform = transformMatrix;
        updatedCount++;
    }
    
    // Early exit if nothing was updated and this isn't the initial build
    if (updatedCount == 0 && !initialBuild) {
        return;
    }
    
    // Memory is persistently mapped and HostCoherent - writes are visible to GPU
    
    vk::BufferDeviceAddressInfo instanceAddrInfo{.buffer = m_blasInstancesBuffers[frameIdx]};
    vk::DeviceAddress instanceAddr = m_vulkanCore.device().getBufferAddress(instanceAddrInfo);

    auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR{
        .arrayOfPointers = false,
        .data = instanceAddr
    };

    vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

    vk::AccelerationStructureGeometryKHR tlasGeometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = geometryData
    };

    vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        .mode = vk::BuildAccelerationStructureModeKHR::eUpdate,
        .srcAccelerationStructure = m_tlasHandles[frameIdx],
        .dstAccelerationStructure = m_tlasHandles[frameIdx],
        .geometryCount = 1,
        .pGeometries = &tlasGeometry
    };

    vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *m_tlasScratchBuffers[frameIdx]};
    vk::DeviceAddress scratchAddr = m_vulkanCore.device().getBufferAddress(scratchAddressInfo);
    tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

    vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{
        .primitiveCount = primitiveCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0
    };

    // Pre-build barrier: Wait for previous AS reads/writes to finish
    const vk::MemoryBarrier preBarrier{
        .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR | vk::AccessFlagBits::eTransferWrite |
                         vk::AccessFlagBits::eShaderRead,
        .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR |
                         vk::AccessFlagBits::eAccelerationStructureWriteKHR
    };

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eTransfer |
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        {},
        preBarrier,
        {},
        {}
    );

    // Build/update the TLAS
    cmd.buildAccelerationStructuresKHR({tlasBuildGeometryInfo}, {&tlasRangeInfo});

    // Post-build barrier: Make TLAS available for shader reads
    const vk::MemoryBarrier postBarrier{
        .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
        .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR | vk::AccessFlagBits::eShaderRead
    };

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR | vk::PipelineStageFlagBits::eFragmentShader,
        {},
        postBarrier,
        {},
        {}
    );
}

void ResourceManager::updateInstanceBuffers(const Scene& scene, const std::uint32_t frameIdx) {
    if (scene.instances.empty()) {
        return;
    }
    
    // On first frame (or when frameIdx wraps), copy all instances
    static bool initialCopyDone[MAX_FRAMES_IN_FLIGHT] = {false};
    
    if (!initialCopyDone[frameIdx]) {
        memcpy(m_instanceBuffersMapped[frameIdx], scene.instances.data(), sizeof(Instance) * scene.instances.size());
        initialCopyDone[frameIdx] = true;
        return;
    }
    
    // For subsequent frames, only update animated instances
    auto* bufferPtr = static_cast<Instance*>(m_instanceBuffersMapped[frameIdx]);
    std::uint32_t updatedCount = 0;
    
    for (std::size_t i = 0; i < scene.instances.size(); ++i) {
        if (scene.instances[i].animated != 0) {
            bufferPtr[i] = scene.instances[i];
            updatedCount++;
        }
    }
}

void ResourceManager::updateMaterialBuffers(const Scene& scene, const std::uint32_t frameIdx) {
    if (!scene.materials.empty()) {
        memcpy(m_materialBuffersMapped[frameIdx], scene.materials.data(), sizeof(Material) * scene.materials.size());
    }
}

void ResourceManager::updateLightBuffers(const Scene& scene, const std::uint32_t frameIdx) {
    static bool initialCopyDone[MAX_FRAMES_IN_FLIGHT] = {false};
    
    if (!initialCopyDone[frameIdx]) {
        if (!scene.pointLights.empty()) {
            memcpy(m_pointLightBuffersMapped[frameIdx], scene.pointLights.data(),
                   sizeof(PointLight) * scene.pointLights.size());
        }
        if (!scene.spotLights.empty()) {
            memcpy(m_spotLightBuffersMapped[frameIdx], scene.spotLights.data(),
                   sizeof(SpotLight) * scene.spotLights.size());
        }
        initialCopyDone[frameIdx] = true;
        return;
    }
    
    // For subsequent frames, only update animated lights
    std::uint32_t pointLightsUpdated = 0;
    std::uint32_t spotLightsUpdated = 0;
    
    if (!scene.pointLights.empty()) {
        auto* bufferPtr = static_cast<PointLight*>(m_pointLightBuffersMapped[frameIdx]);
        for (std::size_t i = 0; i < scene.pointLights.size(); ++i) {
            if (scene.pointLights[i].animated != 0) {
                bufferPtr[i] = scene.pointLights[i];
                pointLightsUpdated++;
            }
        }
    }
    
    if (!scene.spotLights.empty()) {
        auto* bufferPtr = static_cast<SpotLight*>(m_spotLightBuffersMapped[frameIdx]);
        for (std::size_t i = 0; i < scene.spotLights.size(); ++i) {
            if (scene.spotLights[i].animated != 0) {
                bufferPtr[i] = scene.spotLights[i];
                spotLightsUpdated++;
            }
        }
    }
}

void ResourceManager::createIndirectDrawBuffers(const Scene& scene) {
    m_indirectDrawBuffers.clear();
    m_indirectDrawBuffersMemory.clear();
    m_indirectDrawBuffersMapped.clear();

    // Allocate for one draw command per instance (worst case)
    // This ensures we have enough space even if all instances use different meshes
    m_indirectDrawCount = static_cast<std::uint32_t>(scene.instances.size());

    const vk::DeviceSize bufferSize = sizeof(DrawIndexedIndirectCommand) * m_indirectDrawCount;

    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vk::raii::Buffer indirectDrawBuffer{nullptr};
        vk::raii::DeviceMemory indirectDrawBufferMemory{nullptr};
        void* indirectDrawBufferMapped = nullptr;

        m_bufferManager.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            indirectDrawBuffer,
            indirectDrawBufferMemory,
            nullptr
        );

        indirectDrawBufferMapped = indirectDrawBufferMemory.mapMemory(0, bufferSize);

        m_indirectDrawBuffers.push_back(std::move(indirectDrawBuffer));
        m_indirectDrawBuffersMemory.push_back(std::move(indirectDrawBufferMemory));
        m_indirectDrawBuffersMapped.push_back(indirectDrawBufferMapped);
    }

    for (std::uint32_t frameIdx = 0; frameIdx < MAX_FRAMES_IN_FLIGHT; frameIdx++) {
        updateIndirectDrawBuffers(scene, frameIdx);
    }
}

void ResourceManager::updateIndirectDrawBuffers(const Scene& scene, const std::uint32_t frameIdx) {
    if (m_indirectDrawBuffersMapped.empty() || m_indirectDrawCount == 0) {
        return;
    }
    
    // Get current camera view-projection
    const glm::mat4 currentViewProj = scene.camera.getViewProjection();
    
    // Check if camera has moved significantly (use epsilon to avoid tiny movements)
    bool cameraChanged = !m_indirectDrawBuffersInitialized[frameIdx];
    if (!cameraChanged) {
        // Check if any matrix element changed by more than threshold
        constexpr float CAMERA_CHANGE_THRESHOLD = 0.0001f;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (std::abs(currentViewProj[i][j] - m_cachedCameraViewProj[frameIdx][i][j]) > CAMERA_CHANGE_THRESHOLD) {
                    cameraChanged = true;
                    break;
                }
            }
            if (cameraChanged) break;
        }
    }
    
    // Cache scene data pointers to reduce pointer chasing
    const Instance* instances = scene.instances.data();
    const Mesh* meshes = scene.meshes.data();
    const Material* materials = scene.materials.data();
    const std::uint32_t instanceCount = static_cast<std::uint32_t>(scene.instances.size());
    const std::uint32_t meshCount = static_cast<std::uint32_t>(scene.meshes.size());
    const std::uint32_t materialCount = static_cast<std::uint32_t>(scene.materials.size());

    // Get frustum for culling (only if camera changed)
    Frustum frustum;
    if (cameraChanged) {
        frustum = scene.camera.getFrustum();
    }

    // Write directly to mapped buffer - NO intermediate vectors
    auto* bufferPtr = static_cast<DrawIndexedIndirectCommand*>(m_indirectDrawBuffersMapped[frameIdx]);
    
    if (cameraChanged) {
        // FULL REBUILD: Camera moved or first frame - rebuild entire buffer
        std::uint32_t opaqueCount = 0;
        std::uint32_t transparentCount = 0;
        
        const std::uint32_t maxTransparent = std::min(instanceCount, 500u);
        std::vector<DrawIndexedIndirectCommand> tempTransparent;
        tempTransparent.reserve(maxTransparent);
            
        const auto& planes = frustum.planes;

        // Process all instances
        for (std::uint32_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++) {
            const auto& instance = instances[instanceIdx];
            
            const std::int32_t meshIdx = instance.meshIndex;
            if (meshIdx < 0 || meshIdx >= static_cast<std::int32_t>(meshCount)) {
                continue;
            }
            
            const auto& mesh = meshes[meshIdx];
            const std::int32_t matIdx = mesh.materialIndex;
            
            if (matIdx < 0 || matIdx >= static_cast<std::int32_t>(materialCount)) {
                continue;
            }
            
            // Frustum culling
            const glm::vec3 localCenter = (mesh.boundingBoxMin + mesh.boundingBoxMax) * 0.5f;
            const glm::vec3 boxExtents = mesh.boundingBoxMax - mesh.boundingBoxMin;
            const float localRadius = glm::length(boxExtents) * 0.5f;
            const glm::vec3 worldCenter = glm::vec3(instance.transform * glm::vec4(localCenter, 1.0f));
            
            const glm::vec3 col0 = instance.transform[0];
            const glm::vec3 col1 = instance.transform[1];
            const glm::vec3 col2 = instance.transform[2];
            const float scale0Sq = glm::dot(col0, col0);
            const float scale1Sq = glm::dot(col1, col1);
            const float scale2Sq = glm::dot(col2, col2);
            const float maxScaleSq = glm::max(scale0Sq, glm::max(scale1Sq, scale2Sq));
            const float worldRadius = localRadius * std::sqrt(maxScaleSq);
            
            bool visible = true;
            for (int i = 0; i < 5; ++i) {
                const float dist = glm::dot(planes[i].normal, worldCenter) + planes[i].distance;
                if (dist < -worldRadius) {
                    visible = false;
                    break;
                }
            }
            
            if (!visible) {
                continue;
            }
            
            const DrawIndexedIndirectCommand cmd{
                .indexCount = mesh.indexCount,
                .instanceCount = 1,
                .firstIndex = mesh.baseIndex,
                .vertexOffset = static_cast<std::int32_t>(mesh.baseVertex),
                .firstInstance = instanceIdx
            };
            
            if (materials[matIdx].alphaMode == 1) {
                if (transparentCount < maxTransparent) {
                    tempTransparent.push_back(cmd);
                    transparentCount++;
                }
            } else {
                bufferPtr[opaqueCount++] = cmd;
            }
        }
        
        // Copy transparent commands after opaque
        if (transparentCount > 0) {
            std::memcpy(bufferPtr + opaqueCount, tempTransparent.data(), 
                       transparentCount * sizeof(DrawIndexedIndirectCommand));
        }

        m_opaqueDrawCount = opaqueCount;
        m_transparentDrawCount = transparentCount;
        m_indirectDrawCount = m_opaqueDrawCount + m_transparentDrawCount;
        
        // Update cache
        m_cachedCameraViewProj[frameIdx] = currentViewProj;
        m_indirectDrawBuffersInitialized[frameIdx] = true;
        
    } else {
        // PARTIAL UPDATE: Camera hasn't moved - only update animated instances
        // This is a simplified optimization - we rebuild only animated objects
        // A more complex approach would be to track and update individual commands,
        // but that requires maintaining a mapping structure
        
        // For now, we check if there are any animated instances
        bool hasAnimated = false;
        for (std::uint32_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++) {
            if (instances[instanceIdx].animated != 0) {
                hasAnimated = true;
                break;
            }
        }
        
        // If we have animated instances, we need to rebuild (because their positions changed)
        // This could be further optimized by only updating their specific draw commands,
        // but that's more complex and requires tracking which buffer positions they occupy
        if (hasAnimated) {
            // For simplicity, do a full rebuild if any animated objects exist
            // This is still better than the original, because we skip this entirely
            // when nothing is animated and camera hasn't moved
            frustum = scene.camera.getFrustum();
            const auto& planes = frustum.planes;
            
            std::uint32_t opaqueCount = 0;
            std::uint32_t transparentCount = 0;
            
            const std::uint32_t maxTransparent = std::min(instanceCount, 500u);
            std::vector<DrawIndexedIndirectCommand> tempTransparent;
            tempTransparent.reserve(maxTransparent);

            for (std::uint32_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++) {
                const auto& instance = instances[instanceIdx];
                
                const std::int32_t meshIdx = instance.meshIndex;
                if (meshIdx < 0 || meshIdx >= static_cast<std::int32_t>(meshCount)) {
                    continue;
                }
                
                const auto& mesh = meshes[meshIdx];
                const std::int32_t matIdx = mesh.materialIndex;
                
                if (matIdx < 0 || matIdx >= static_cast<std::int32_t>(materialCount)) {
                    continue;
                }
                
                // Frustum culling
                const glm::vec3 localCenter = (mesh.boundingBoxMin + mesh.boundingBoxMax) * 0.5f;
                const glm::vec3 boxExtents = mesh.boundingBoxMax - mesh.boundingBoxMin;
                const float localRadius = glm::length(boxExtents) * 0.5f;
                const glm::vec3 worldCenter = glm::vec3(instance.transform * glm::vec4(localCenter, 1.0f));
                
                const glm::vec3 col0 = instance.transform[0];
                const glm::vec3 col1 = instance.transform[1];
                const glm::vec3 col2 = instance.transform[2];
                const float scale0Sq = glm::dot(col0, col0);
                const float scale1Sq = glm::dot(col1, col1);
                const float scale2Sq = glm::dot(col2, col2);
                const float maxScaleSq = glm::max(scale0Sq, glm::max(scale1Sq, scale2Sq));
                const float worldRadius = localRadius * std::sqrt(maxScaleSq);
                
                bool visible = true;
                for (int i = 0; i < 5; ++i) {
                    const float dist = glm::dot(planes[i].normal, worldCenter) + planes[i].distance;
                    if (dist < -worldRadius) {
                        visible = false;
                        break;
                    }
                }
                
                if (!visible) {
                    continue;
                }
                
                const DrawIndexedIndirectCommand cmd{
                    .indexCount = mesh.indexCount,
                    .instanceCount = 1,
                    .firstIndex = mesh.baseIndex,
                    .vertexOffset = static_cast<std::int32_t>(mesh.baseVertex),
                    .firstInstance = instanceIdx
                };
                
                if (materials[matIdx].alphaMode == 1) {
                    if (transparentCount < maxTransparent) {
                        tempTransparent.push_back(cmd);
                        transparentCount++;
                    }
                } else {
                    bufferPtr[opaqueCount++] = cmd;
                }
            }
            
            if (transparentCount > 0) {
                std::memcpy(bufferPtr + opaqueCount, tempTransparent.data(), 
                           transparentCount * sizeof(DrawIndexedIndirectCommand));
            }

            m_opaqueDrawCount = opaqueCount;
            m_transparentDrawCount = transparentCount;
            m_indirectDrawCount = m_opaqueDrawCount + m_transparentDrawCount;
        }
        // If no animated instances, we skip the update entirely - huge win!
    }
}
