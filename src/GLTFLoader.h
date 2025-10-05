#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

struct GLTFVertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 color;
    
    static vk::VertexInputBindingDescription getBindingDescription();
    static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions();
};

struct GLTFMesh {
    std::vector<GLTFVertex> vertices;
    std::vector<std::uint32_t> indices;
    glm::mat4 transform;
    std::string name;
    glm::vec3 baseColor;
    float metallic;
    float roughness;
    bool hasEmission;
    glm::vec3 emissionColor;
    float emissionStrength;
};

struct GLTFModel {
    std::vector<GLTFMesh> meshes;
    glm::vec3 minBounds;
    glm::vec3 maxBounds;
    std::string name;
};

class GLTFLoader {
public:
    static bool loadModel(const std::string& filepath, GLTFModel& model);
    
private:
    static void processNode(const void* node, const void* scene, GLTFModel& model);
    static GLTFMesh processMesh(const void* mesh, const void* scene);
    static GLTFVertex processVertex(const void* vertex, const void* accessor);
    static glm::mat4 getNodeTransform(const void* node);
    static glm::vec3 getVec3FromAccessor(const void* accessor, size_t index);
    static glm::vec2 getVec2FromAccessor(const void* accessor, size_t index);
    
    // Helper functions for creating simple geometry
    static void createSimpleCityMesh(GLTFMesh& mesh);
    static void createBuilding(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
    static void createNeonStrip(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
};
