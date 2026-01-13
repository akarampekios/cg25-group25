#pragma once

#include <vector>
#include <cstdint>
#include <vulkan/vulkan.hpp>
#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>

struct DrawIndexedIndirectCommand {
    std::uint32_t indexCount;
    std::uint32_t instanceCount;
    std::uint32_t firstIndex;
    std::int32_t vertexOffset;
    std::uint32_t firstInstance;
};

struct BloomPushConstant {
    glm::vec2 textureSize;
    glm::vec2 direction;
    float blurStrength;
    float exposure;
    float threshold;
    float scale;
};

struct BloomParameters {
    float blurStrength{4.0};
    float exposure{1.0};
    float threshold{0.5};
    float scale{2.0};
};

struct TAAPushConstant {
    glm::vec2 screenSize;
    float blendFactor;
    float _padding;
};

struct alignas(16) Material {
    glm::vec4 baseColorFactor;

    glm::vec3 emissiveFactor;
    std::int32_t padding;

    float metallicFactor;
    float roughnessFactor;
    std::int32_t baseColorTexIndex = -1;
    std::int32_t metallicRoughnessTexIndex = -1;

    std::int32_t normalTexIndex = -1;
    std::int32_t emissiveTexIndex = -1;
    std::int32_t occlusionTexIndex = -1;
    std::int32_t alphaMode = 0; // 0 - opaque, 1 - BLEND, 2 - MASK

    std::int32_t reflective = 1; // 0 - no reflections, 1 - reflective (default)
    std::int32_t castsShadows = 1; // 0 - no shadow casting, 1 - casts shadows (default)
    std::int32_t receivesLighting = 1; // 0 - no lighting, 1 - receives lighting (default)
    std::int32_t padding1;
};

struct Texture {
    vk::Format format;
    std::uint32_t mipLevels;
    std::uint32_t width;
    std::uint32_t height;
    std::vector<unsigned char> image;
    std::vector<float> imagef; // For HDR images
    bool skyTexture{false};
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    static auto getBindingDescription() -> vk::VertexInputBindingDescription {
        return {
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };
    }

    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() {
        constexpr vk::VertexInputAttributeDescription posDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, position),
        };

        constexpr vk::VertexInputAttributeDescription normalDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, normal),
        };

        constexpr vk::VertexInputAttributeDescription texDescription{
            .location = 2,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(Vertex, texCoord),
        };

        constexpr vk::VertexInputAttributeDescription tangentDescription{
            .location = 3,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = offsetof(Vertex, tangent),
        };

    return {posDescription, normalDescription, texDescription, tangentDescription};
    }
};

struct Geometry {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) Mesh {
    glm::vec3 boundingBoxMin{0.0f};
    std::int32_t padding;

    glm::vec3 boundingBoxMax{0.0f};
    std::uint32_t baseVertex;

    std::uint32_t baseIndex;
    std::uint32_t vertexCount;
    std::uint32_t indexCount;
    std::int32_t materialIndex{-1};
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) Instance {
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    std::int32_t meshIndex{-1};
    std::int32_t reflective{0}; // 0 - no reflections, 1 - reflective (default)
    std::int32_t castsShadows{0}; // 0 - no shadow casting, 1 - casts shadows (default)
    std::int32_t receivesLighting{1}; // 0 - no lighting, 1 - receives lighting (default)
    std::int32_t animated{0}; // 0 - static (default), 1 - animated (updated by animator)
    std::int32_t _padding[3];
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) DirectionalLight {
    glm::vec3 direction;
    float intensity;
    glm::vec3 color;
    float padding;
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) PointLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
    float radius;
    std::int32_t castsShadows{1}; // 0 - no shadows, 1 - casts shadows (default)
    std::int32_t animated{0}; // 0 - static (default), 1 - animated
    std::int32_t _padding[2];
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) SpotLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 direction;
    float cutoff;
    glm::vec3 color;
    float outerCutoff;
    std::int32_t castsShadows{1}; // 0 - no shadows, 1 - casts shadows (default)
    std::int32_t animated{0}; // 0 - static (default), 1 - animated
    std::int32_t _padding[2];
};

// padding confirmed, do not touch or it will break! ✅
struct alignas(16) UniformBufferObject {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::mat4 prevView;      // TAA: Previous frame view matrix
    glm::mat4 prevProj;      // TAA: Previous frame projection matrix
    glm::vec3 cameraPos;
    float time;
    std::uint32_t pointLightsCount;
    std::uint32_t spotLightsCount;
    DirectionalLight directionalLight;
    std::int32_t skySphereInstanceIndex{-1};
    std::int32_t skySphereTextureIndex{-1};
    glm::vec2 jitterOffset;  // TAA: Sub-pixel jitter in pixels
    glm::vec3 fogColor;
    float fogDensity;
    glm::vec2 screenSize;    // TAA: Screen dimensions for velocity calculation
    glm::vec2 _padding3;
};
