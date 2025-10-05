#pragma once

#include <vector>
#include <memory>
#include <array>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "GLTFLoader.h"

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;
    
    static VkVertexInputBindingDescription getBindingDescription();
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions();
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::mat4 transform;
    glm::vec3 color;
};

class Scene {
public:
    Scene();
    ~Scene();
    
    void init();
    void update(float deltaTime);
    
    const std::vector<GLTFMesh>& getMeshes() const { return m_meshes; }
    const std::vector<GLTFVertex>& getVertices() const { return m_vertices; }
    const std::vector<uint32_t>& getIndices() const { return m_indices; }

private:
    bool loadCityModel();
    void createBasicCity();
    void createBuilding(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
    void createGround();
    void createNeonLights();
    
    // Helper functions for mesh creation
    void createGroundMesh(GLTFMesh& mesh);
    void createBuildingMesh(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
    void createNeonMesh(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color);
    
    std::vector<GLTFMesh> m_meshes;
    std::vector<GLTFVertex> m_vertices;
    std::vector<uint32_t> m_indices;
    
    GLTFModel m_cityModel;
    float m_time;
};
