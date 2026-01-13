#pragma once

#include <cstddef>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "SharedTypes.hpp"
#include "Scene.hpp"

class VulkanCore;
class CommandManager;
class BufferManager;
class ImageManager;

struct AllocatedBuffer {
    vk::raii::Buffer& buffer;
    vk::raii::DeviceMemory& memory;
};

struct AllocatedTextureImage {
    vk::raii::Image image{nullptr};
    vk::raii::ImageView imageView{nullptr};
    vk::raii::DeviceMemory imageMemory{nullptr};
};

struct AllocatedDescriptorSetLayouts {
    vk::raii::DescriptorSetLayout& globalLayout;
    vk::raii::DescriptorSetLayout& materialLayout;
    vk::raii::DescriptorSetLayout& lightingLayout;
};

struct AllocatedDescriptorSets {
    std::vector<vk::raii::DescriptorSet>& globalSets;
    std::vector<vk::raii::DescriptorSet>& materialSets;
    std::vector<vk::raii::DescriptorSet>& lightSets;
};

class ResourceManager {
public:
    ResourceManager(VulkanCore& vulkanCore,
                    CommandManager& commandManager,
                    BufferManager& bufferManager,
                    ImageManager& imageManager);

    ~ResourceManager() = default;

    [[nodiscard]] auto getVertexBuffer() -> AllocatedBuffer {
        return {.buffer = m_vertexBuffer, .memory = m_vertexBufferMemory};
    }

    [[nodiscard]] auto getIndexBuffer() -> AllocatedBuffer {
        return {.buffer = m_indexBuffer, .memory = m_indexBufferMemory};
    }

    [[nodiscard]] auto getDescriptorSetLayouts() -> AllocatedDescriptorSetLayouts {
        return {
            .globalLayout = m_globalDescriptorSetLayout,
            .materialLayout = m_materialDescriptorSetLayout,
            .lightingLayout = m_lightDescriptorSetLayout,
        };
    }

    [[nodiscard]] auto getDescriptorSets() -> AllocatedDescriptorSets {
        return {
            .globalSets = m_globalDescriptorSets,
            .materialSets = m_materialDescriptorSets,
            .lightSets = m_lightingDescriptorSets,
        };
    }

    [[nodiscard]] auto getDescriptorPool() const -> const vk::raii::DescriptorPool& {
        return m_descriptorPool;
    }

    [[nodiscard]] auto getIndirectDrawBuffer(std::uint32_t frameIdx) -> AllocatedBuffer {
        return {.buffer = m_indirectDrawBuffers[frameIdx], .memory = m_indirectDrawBuffersMemory[frameIdx]};
    }

    [[nodiscard]] auto getIndirectDrawCount() const -> std::uint32_t {
        return m_indirectDrawCount;
    }

    [[nodiscard]] auto getOpaqueDrawCount() const -> std::uint32_t {
        return m_opaqueDrawCount;
    }

    [[nodiscard]] auto getTransparentDrawCount() const -> std::uint32_t {
        return m_transparentDrawCount;
    }

    [[nodiscard]] auto getTransparentDrawOffset() const -> vk::DeviceSize {
        return m_opaqueDrawCount * sizeof(DrawIndexedIndirectCommand);
    }

    void allocateSceneResources(const Scene& scene);
    void updateSceneResources(const Scene& scene, float time, std::uint32_t frameIdx, glm::vec2 jitterOffset = glm::vec2(0.0f));
    
    // Record TLAS update commands into the provided command buffer (if needed)
    void recordTLASUpdate(const vk::CommandBuffer& cmd, const Scene& scene, bool initialBuild, std::uint32_t frameIdx);

private:
    VulkanCore& m_vulkanCore;
    CommandManager& m_commandManager;
    BufferManager& m_bufferManager;
    ImageManager& m_imageManager;

    vk::raii::Sampler m_skyboxSampler = nullptr;
    vk::raii::Sampler m_baseColorTextureSampler = nullptr;
    vk::raii::Sampler m_metallicRoughnessTextureSampler = nullptr;
    vk::raii::Sampler m_normalTextureSampler = nullptr;
    vk::raii::Sampler m_emissiveTextureSampler = nullptr;
    vk::raii::Sampler m_occlusionTextureSampler = nullptr;

    AllocatedTextureImage m_skyboxImage;
    std::vector<AllocatedTextureImage> m_baseColorTextureImages;
    std::vector<AllocatedTextureImage> m_metallicTextureImages;
    std::vector<AllocatedTextureImage> m_normalTextureImages;
    std::vector<AllocatedTextureImage> m_emissiveTextureImages;
    std::vector<AllocatedTextureImage> m_occlusionTextureImages;

    vk::raii::DescriptorPool m_descriptorPool = nullptr;

    vk::raii::DescriptorSetLayout m_globalDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_materialDescriptorSetLayout = nullptr;
    vk::raii::DescriptorSetLayout m_lightDescriptorSetLayout = nullptr;

    std::vector<vk::raii::DescriptorSet> m_globalDescriptorSets;
    std::vector<vk::raii::DescriptorSet> m_materialDescriptorSets;
    std::vector<vk::raii::DescriptorSet> m_lightingDescriptorSets;

    vk::raii::Buffer m_vertexBuffer = nullptr;
    vk::raii::DeviceMemory m_vertexBufferMemory = nullptr;

    vk::raii::Buffer m_indexBuffer = nullptr;
    vk::raii::DeviceMemory m_indexBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> m_uniformBuffers;
    std::vector<vk::raii::DeviceMemory> m_uniformBuffersMemory;
    std::vector<void*> m_uniformBuffersMapped;

    std::vector<vk::raii::Buffer> m_instanceBuffers;
    std::vector<vk::raii::DeviceMemory> m_instanceBuffersMemory;
    std::vector<void*> m_instanceBuffersMapped;

    vk::raii::Buffer m_meshesBuffer = nullptr;
    vk::raii::DeviceMemory m_meshesBufferMemory = nullptr;

    vk::raii::Buffer m_uvBuffer = nullptr;
    vk::raii::DeviceMemory m_uvBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> m_materialBuffers;
    std::vector<vk::raii::DeviceMemory> m_materialBuffersMemory;
    std::vector<void*> m_materialBuffersMapped;

    std::vector<vk::raii::Buffer> m_pointLightBuffers;
    std::vector<vk::raii::DeviceMemory> m_pointLightBuffersMemory;
    std::vector<void*> m_pointLightBuffersMapped;
    std::vector<vk::raii::Buffer> m_spotLightBuffers;
    std::vector<vk::raii::DeviceMemory> m_spotLightBuffersMemory;
    std::vector<void*> m_spotLightBuffersMapped;

    std::vector<vk::raii::Buffer> m_indirectDrawBuffers;
    std::vector<vk::raii::DeviceMemory> m_indirectDrawBuffersMemory;
    std::vector<void*> m_indirectDrawBuffersMapped;
    std::uint32_t m_indirectDrawCount{0};
    std::uint32_t m_opaqueDrawCount{0};
    std::uint32_t m_transparentDrawCount{0};
    
    std::vector<glm::mat4> m_cachedCameraViewProj;
    std::vector<bool> m_indirectDrawBuffersInitialized;
    
    // TAA: Previous frame matrices for velocity calculation (per-frame to handle multiple frames in flight)
    // Each frame index stores its own previous matrices to avoid cross-frame interference
    std::vector<glm::mat4> m_prevViewMatrices;
    std::vector<glm::mat4> m_prevProjMatrices;
    std::vector<bool> m_frameInitialized;

    std::vector<vk::AccelerationStructureInstanceKHR> m_blasInstances;

    // IMPORTANT: TLAS builds/updates must not share instance buffers / scratch buffers across frames in flight.
    // With multiple frames in flight, sharing these can cause GPU races and VK_ERROR_DEVICE_LOST.
    std::vector<vk::raii::Buffer> m_blasInstancesBuffers;
    std::vector<vk::raii::DeviceMemory> m_blasInstancesMemories;
    std::vector<void*> m_blasInstancesBuffersMapped;

    std::vector<vk::raii::Buffer> m_blasBuffers;
    std::vector<vk::raii::DeviceMemory> m_blasMemories;
    std::vector<vk::raii::AccelerationStructureKHR> m_blasHandles;

    std::vector<vk::raii::Buffer> m_tlasBuffers;
    std::vector<vk::raii::DeviceMemory> m_tlasMemories;
    std::vector<vk::raii::Buffer> m_tlasScratchBuffers;
    std::vector<vk::raii::DeviceMemory> m_tlasScratchMemories;
    std::vector<vk::raii::AccelerationStructureKHR> m_tlasHandles;

    void createDescriptorPool();
    void createDescriptorSetLayouts();
    void createTextureSamplers();

    void allocateVertexBuffer(const std::vector<Vertex>& vertices);
    void allocateIndexBuffer(const std::vector<std::uint32_t>& indices);
    void createUniformBuffers();

    void createInstanceBuffers(const Scene& scene);
    void createMeshesBuffer(const Scene& scene);
    void createUVBuffer(const Scene& scene);
    void createMaterialBuffers(const Scene& scene);
    void createLightBuffers(const Scene& scene);
    void createIndirectDrawBuffers(const Scene& scene);
    void createTextureImages(const Scene& scene);
    void createSkyboxImage(const Scene& scene);

    void createGlobalDescriptorSets(const Scene& scene);
    void createMaterialDescriptorSets(const Scene& scene);
    void createLightingDescriptorSets(const Scene& scene);
    void createAccelerationStructures(const Scene& scene);
    void createBLAS(const Scene& scene);
    void createBLASInstances(const Scene& scene);
    void createTLAS();

    void updateUniformBuffer(const Scene& scene, float time, std::uint32_t frameIdx, glm::vec2 jitterOffset);
    void updateTopLevelAccelerationStructures(const Scene& scene, bool initialBuild, std::uint32_t frameIdx);
    void updateInstanceBuffers(const Scene& scene, std::uint32_t frameIdx);
    void updateMaterialBuffers(const Scene& scene, std::uint32_t frameIdx);
    void updateLightBuffers(const Scene& scene, std::uint32_t frameIdx);
    void updateIndirectDrawBuffers(const Scene& scene, std::uint32_t frameIdx);
};
