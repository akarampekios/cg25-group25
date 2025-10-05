#pragma once

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 color;
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

        constexpr vk::VertexInputAttributeDescription colorDescription{
            .location = 3,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(Vertex, color),
        };

        constexpr vk::VertexInputAttributeDescription tangentDescription{
            .location = 4,
            .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = offsetof(Vertex, tangent),
        };

        return {posDescription, normalDescription, texDescription, colorDescription, tangentDescription};
    }
};

struct Texture {
    vk::raii::Image image = nullptr;
    vk::raii::DeviceMemory imageMemory = nullptr;
    vk::raii::ImageView imageView = nullptr;
    vk::raii::Sampler sampler = nullptr;
};

struct Material {
    Texture baseColorTex;
    Texture metallicRoughnessTex;
    Texture normalTex;
    Texture emissiveTex;
    Texture occlusionTex;

    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
};

struct Mesh {
    glm::mat4 model;

    Material* material = nullptr;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // in theory, we do not need per frame buffers, but it is just easier for now
    std::vector<vk::raii::Buffer> materialBuffers;
    std::vector<vk::raii::DeviceMemory> materialBuffersMemory;
    std::vector<void*> materialBuffersMapped;

    std::vector<vk::raii::DescriptorSet> descriptorSets;
};

// TODO: add support for animation
struct CameraParameters {
    float yfov;
    float aspectRatio;
    float znear;
    float zfar;

    glm::mat4 model;

    glm::vec3 getUp() {
        return glm::normalize(glm::vec3(model * glm::vec4(0, 1, 0, 0)));
    }

    auto getForward() -> glm::vec3 {
        return glm::normalize(glm::vec3(model * glm::vec4(0, 0, -1, 0)));
    }

    auto getPosition() -> glm::vec3 {
        return glm::vec3(model[3]);
    }

    auto getView() -> glm::mat4 {
        return glm::lookAt(getPosition(), getPosition() + getForward(), getUp());
    }

    glm::mat4 getProjection() const {
        auto projection = glm::perspective(yfov, aspectRatio, znear, zfar);
        projection[1][1] *= -1.0f;
        return projection;
    }
};

struct Scene {
    std::vector<Material> materials;
    std::vector<Mesh> meshes;
    CameraParameters camera;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec3 cameraPos;
    float time;
};

struct MaterialBufferObject {
    glm::vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    glm::vec3 emissiveFactor;
    float padding; // align to 16 bytes
};
