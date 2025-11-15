#pragma once

#include <cstdint>
#include <memory>
#include <map>
#include <tiny_gltf.h>

#include "SharedTypes.hpp"
#include "Scene.hpp"

struct LoadedGLTF {
    Scene scene;
    tinygltf::Model model;
};

class GLTFLoader {
public:
    std::unique_ptr<LoadedGLTF> load(const std::string& path);

    explicit GLTFLoader();

private:
    // [gltfd mesh idx][gltff mesh primitive idx] => our mesh
    std::vector<std::vector<std::uint32_t>> m_gltfPrimitiveToEngineGeometry;

    std::map<std::uint32_t, std::uint32_t> m_gltfBaseColorTextureMap;
    std::map<std::uint32_t, std::uint32_t> m_gltfMetallicTextureMap;
    std::map<std::uint32_t, std::uint32_t> m_gltfNormalTextureMap;
    std::map<std::uint32_t, std::uint32_t> m_gltfEmissiveTextureMap;
    std::map<std::uint32_t, std::uint32_t> m_gltfOcclusionTextureMap;

    std::vector<glm::mat4> m_nodeWorldMatrices;

    void computeWorldMatrices(const tinygltf::Model& model);

    void computePrimitiveToGeometryMapping(const tinygltf::Model& model);

    void loadMeshes(const tinygltf::Model& model, Scene& scene);

    void loadMaterialsAndTextures(const tinygltf::Model& model, Scene& scene);

    Texture loadTexture(const tinygltf::Texture& texture, const tinygltf::Model& model);

    template <typename T>
    void loadTextureMap(int gltfTexIndex,
                        std::map<std::uint32_t, std::uint32_t>& gltfTextureMap,
                        std::vector<T>& sceneTextures,
                        std::int32_t& parsedMaterialTexIndex,
                        const tinygltf::Model& model);

    void loadNodes(const tinygltf::Model& model, Scene& scene);

    void loadMeshNode(const tinygltf::Node& node, std::size_t nodeIdx, Scene& scene);

    void loadCameraNode(const tinygltf::Camera& cam, std::size_t nodeIdx, Scene& scene);

    void loadLightNode(const tinygltf::Light& light, std::size_t nodeIdx, const tinygltf::Node& node, Scene& scene);

    void loadSkySphereNode(const tinygltf::Node& node, const tinygltf::Model& model, Scene& scene);

    auto loadPrimitive(const tinygltf::Primitive& prim, const tinygltf::Model& model) -> Geometry;

    void buildMeshToInstanceMapping(Scene& scene);

    void computeNodeWorldMatrix(const tinygltf::Model& model,
                                int nodeIndex,
                                const glm::mat4& parentMatrix,
                                std::vector<glm::mat4>& outMatrices);

    auto getLocalTransform(const tinygltf::Node& node) -> glm::mat4;
};
