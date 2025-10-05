#include "Scene.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

Scene::Scene() : m_time(0.0f) {
}

Scene::~Scene() {
}

void Scene::init() {
    // Try to load the city model from glTF file
    if (!loadCityModel()) {
        std::cout << "Failed to load city model, creating basic city..." << std::endl;
        createBasicCity();
    }
    
    // Combine all meshes into single vertex/index buffers
    m_vertices.clear();
    m_indices.clear();
    
    uint32_t vertexOffset = 0;
    for (const auto& mesh : m_meshes) {
        for (const auto& vertex : mesh.vertices) {
            m_vertices.push_back(vertex);
        }
        
        for (const auto& index : mesh.indices) {
            m_indices.push_back(index + vertexOffset);
        }
        
        vertexOffset += static_cast<uint32_t>(mesh.vertices.size());
    }
}

bool Scene::loadCityModel() {
    std::string modelPath = "assets/cyberpunk_city.glb";
    if (GLTFLoader::loadModel(modelPath, m_cityModel)) {
        std::cout << "Successfully loaded city model: " << m_cityModel.name << std::endl;
        std::cout << "Loaded " << m_cityModel.meshes.size() << " meshes" << std::endl;
        
        // Copy meshes from the loaded model
        m_meshes = m_cityModel.meshes;
        return true;
    }
    return false;
}

void Scene::update(float deltaTime) {
    m_time += deltaTime;
    
    // Animate neon lights
    for (auto& mesh : m_meshes) {
        if (mesh.hasEmission) {
            float intensity = 0.5f + 0.5f * sin(m_time * 2.0f);
            mesh.emissionColor = glm::vec3(intensity, intensity * 0.8f, intensity * 1.2f);
        }
    }
}

void Scene::createBasicCity() {
    // Create a simple city using the GLTFLoader helper
    GLTFMesh cityMesh;
    cityMesh.name = "BasicCity";
    cityMesh.baseColor = glm::vec3(0.1f, 0.1f, 0.2f);
    cityMesh.metallic = 0.2f;
    cityMesh.roughness = 0.8f;
    cityMesh.hasEmission = false;
    cityMesh.transform = glm::mat4(1.0f);
    
    // Create ground
    createGroundMesh(cityMesh);
    
    // Create some buildings
    for (int i = 0; i < 20; ++i) {
        float x = (i % 5 - 2) * 15.0f;
        float z = (i / 5 - 2) * 15.0f;
        float height = 5.0f + (i % 3) * 8.0f;
        
        createBuildingMesh(cityMesh, glm::vec3(x, 0.0f, z), glm::vec3(4.0f, height, 4.0f), 
                          glm::vec3(0.1f + (i % 3) * 0.1f, 0.1f, 0.2f + (i % 2) * 0.2f));
    }
    
    // Add some neon elements
    for (int i = 0; i < 10; ++i) {
        float x = (i % 5 - 2) * 15.0f;
        float z = (i / 5 - 2) * 15.0f;
        float height = 8.0f + (i % 3) * 5.0f;
        
        createNeonMesh(cityMesh, glm::vec3(x, height, z), glm::vec3(6.0f, 0.2f, 0.2f),
                      glm::vec3(1.0f, 0.2f, 0.8f));
    }
    
    m_meshes.push_back(cityMesh);
}

void Scene::createGroundMesh(GLTFMesh& mesh) {
    // Ground plane
    uint32_t baseIndex = static_cast<uint32_t>(mesh.vertices.size());
    
    mesh.vertices.push_back({{-50.0f, -1.0f, -50.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}, {0.1f, 0.1f, 0.15f}});
    mesh.vertices.push_back({{ 50.0f, -1.0f, -50.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}, {0.1f, 0.1f, 0.15f}});
    mesh.vertices.push_back({{ 50.0f, -1.0f,  50.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}, {0.1f, 0.1f, 0.15f}});
    mesh.vertices.push_back({{-50.0f, -1.0f,  50.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}, {0.1f, 0.1f, 0.15f}});
    
    mesh.indices.push_back(baseIndex + 0);
    mesh.indices.push_back(baseIndex + 1);
    mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 3);
    mesh.indices.push_back(baseIndex + 0);
}

void Scene::createBuildingMesh(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color) {
    float halfWidth = size.x * 0.5f;
    float halfHeight = size.y * 0.5f;
    float halfDepth = size.z * 0.5f;
    
    uint32_t baseIndex = static_cast<uint32_t>(mesh.vertices.size());
    
    // Create building vertices (simplified box - just front and back faces for now)
    mesh.vertices.push_back({{-halfWidth, -halfHeight,  halfDepth}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth, -halfHeight,  halfDepth}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth,  halfHeight,  halfDepth}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, color});
    mesh.vertices.push_back({{-halfWidth,  halfHeight,  halfDepth}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, color});
    
    mesh.vertices.push_back({{-halfWidth, -halfHeight, -halfDepth}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth, -halfHeight, -halfDepth}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth,  halfHeight, -halfDepth}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}, color});
    mesh.vertices.push_back({{-halfWidth,  halfHeight, -halfDepth}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}, color});
    
    // Apply position offset
    for (size_t i = baseIndex; i < mesh.vertices.size(); ++i) {
        mesh.vertices[i].position += position;
    }
    
    // Create indices for the building (front and back faces)
    mesh.indices.push_back(baseIndex + 0); mesh.indices.push_back(baseIndex + 1); mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 2); mesh.indices.push_back(baseIndex + 3); mesh.indices.push_back(baseIndex + 0);
    
    mesh.indices.push_back(baseIndex + 4); mesh.indices.push_back(baseIndex + 5); mesh.indices.push_back(baseIndex + 6);
    mesh.indices.push_back(baseIndex + 6); mesh.indices.push_back(baseIndex + 7); mesh.indices.push_back(baseIndex + 4);
}

void Scene::createNeonMesh(GLTFMesh& mesh, const glm::vec3& position, const glm::vec3& size, const glm::vec3& color) {
    float halfWidth = size.x * 0.5f;
    float halfHeight = size.y * 0.5f;
    
    uint32_t baseIndex = static_cast<uint32_t>(mesh.vertices.size());
    
    // Create neon strip vertices
    mesh.vertices.push_back({{-halfWidth, -halfHeight, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth, -halfHeight, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}, color});
    mesh.vertices.push_back({{ halfWidth,  halfHeight, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}, color});
    mesh.vertices.push_back({{-halfWidth,  halfHeight, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, color});
    
    // Apply position offset
    for (size_t i = baseIndex; i < mesh.vertices.size(); ++i) {
        mesh.vertices[i].position += position;
    }
    
    // Create indices for the neon strip
    mesh.indices.push_back(baseIndex + 0); mesh.indices.push_back(baseIndex + 1); mesh.indices.push_back(baseIndex + 2);
    mesh.indices.push_back(baseIndex + 2); mesh.indices.push_back(baseIndex + 3); mesh.indices.push_back(baseIndex + 0);
}

// Placeholder implementations for the old interface
void Scene::createBuilding(const glm::vec3& position, const glm::vec3& size, const glm::vec3& color) {
    // This is now handled by createBuildingMesh above
}

void Scene::createGround() {
    // This is now handled by createGroundMesh above
}

void Scene::createNeonLights() {
    // This is now handled by createNeonMesh above
}

VkVertexInputBindingDescription Vertex::getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

std::array<VkVertexInputAttributeDescription, 4> Vertex::getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    
    // Position
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    
    // Color
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    
    // Texture coordinates
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
    
    // Normal
    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Vertex, normal);
    
    return attributeDescriptions;
}
