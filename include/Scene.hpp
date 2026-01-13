#pragma once

#include <cstdint>
#include <vector>
#include <glm/ext/vector_float2.hpp>
#include <glm/ext/matrix_float4x4.hpp>

#include "SharedTypes.hpp"
#include "FrustumCulling.hpp"

struct CameraParameters {
    float yfov;
    float aspectRatio;
    float znear;
    float zfar;

    glm::mat4 model;

    glm::vec3 getUp() const {
        return glm::normalize(glm::vec3(model * glm::vec4(0, 1, 0, 0)));
    }

    auto getForward() const -> glm::vec3 {
        return glm::normalize(glm::vec3(model * glm::vec4(0, 0, -1, 0)));
    }

    auto getPosition() const -> glm::vec3 {
        return glm::vec3(model[3]);
    }

    auto getView() const -> glm::mat4 {
        return glm::lookAt(getPosition(), getPosition() + getForward(), getUp());
    }

    glm::mat4 getProjection() const {
        auto projection = glm::perspective(yfov, aspectRatio, znear, zfar);
        projection[1][1] *= -1.0f;
        return projection;
    }

    glm::mat4 getViewProjection() const {
        return getProjection() * getView();
    }

    Frustum getFrustum() const {
        return Frustum::fromViewProjection(getViewProjection());
    }
};

struct FogParameters {
    glm::vec3 fogColor{0.0f, 0.11f, 0.11f};
    // current models are large, so distance between camera in objects is also large, 
    // meaning we need lower density, as it gets hugely amplified over distance
    float fogDensity{0.035f};
};

struct Scene {
    std::vector<Mesh> meshes;
    std::vector<Instance> instances;
    std::vector<Material> materials;
    std::vector<Texture> baseColorTextures;
    std::vector<Texture> metallicRoughnessTextures;
    std::vector<Texture> normalTextures;
    std::vector<Texture> emissiveTextures;
    std::vector<Texture> occlusionTextures;

    CameraParameters camera;
    BloomParameters bloom;
    FogParameters fog;

    std::int32_t skySphereInstanceIndex{-1};
    std::int32_t skySphereTextureIndex{-1};

    DirectionalLight directionalLight;
    std::vector<PointLight> pointLights;
    std::vector<SpotLight> spotLights;

    std::vector<glm::vec2> uvs;
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    // Animation support: maps glTF node index to first instance index for that node
    // Each node with a mesh may have multiple instances (one per primitive)
    std::vector<std::int32_t> nodeToInstanceIndex;

    // For indirect drawing: track which instances use which mesh
    // Key: meshIndex, Value: vector of instance indices
    std::vector<std::vector<std::uint32_t>> meshToInstanceIndices;
};
