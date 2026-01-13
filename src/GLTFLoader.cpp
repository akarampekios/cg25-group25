#include <iostream>
#include <memory>
#include <cstdint>
#include <limits>
#include <glm/glm.hpp>

#include "constants.hpp"
#include "SharedTypes.hpp"
#include "GLTFLoader.hpp"

inline float lux_to_radiance(float lux, float radius) {
    constexpr float lumenToWatt = 683.0f;

    if (radius <= 0.0f) {
        return lux / lumenToWatt;
    }

    const float area = 4.0f * static_cast<float>(3.14) * radius * radius;
    return lux / (lumenToWatt * area);
}

GLTFLoader::GLTFLoader() {
}

std::unique_ptr<LoadedGLTF> GLTFLoader::load(const std::string& path) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;

    std::string err;
    std::string warn;

    LoadedGLTF loaded;

    loader.LoadBinaryFromFile(&model, &err, &warn, path);

    computeWorldMatrices(model);
    computePrimitiveToGeometryMapping(model);

    loadMaterialsAndTextures(model, loaded.scene);
    loadMeshes(model, loaded.scene);
    loadNodes(model, loaded.scene);

    // Build mesh-to-instance mapping for indirect drawing
    buildMeshToInstanceMapping(loaded.scene);

    // Store the model for animation
    loaded.model = std::move(model);

    return std::make_unique<LoadedGLTF>(std::move(loaded));
}

void GLTFLoader::computeWorldMatrices(const tinygltf::Model& model) {
    m_nodeWorldMatrices = std::vector(model.nodes.size(), glm::mat4(1.0f));

    // Find root nodes (nodes that are not children of any other node)
    std::vector<bool> isChild(model.nodes.size(), false);
    for (const auto& node : model.nodes) {
        for (const int childIdx : node.children) {
            if (childIdx >= 0 && static_cast<std::size_t>(childIdx) < model.nodes.size()) {
                isChild[childIdx] = true;
            }
        }
    }

    // Only compute world matrices starting from root nodes
    for (std::size_t nodeIdx = 0; nodeIdx < model.nodes.size(); nodeIdx++) {
        if (!isChild[nodeIdx]) {
            computeNodeWorldMatrix(model, static_cast<int>(nodeIdx), glm::mat4(1.0f), m_nodeWorldMatrices);
        }
    }
}

void GLTFLoader::computePrimitiveToGeometryMapping(const tinygltf::Model& model) {
    m_gltfPrimitiveToEngineGeometry.clear();
    m_gltfPrimitiveToEngineGeometry.resize(model.meshes.size());

    std::uint32_t geometryIndex = 0;

    for (std::size_t meshIdx = 0; meshIdx < model.meshes.size(); meshIdx++) {
        const auto& mesh = model.meshes[meshIdx];

        m_gltfPrimitiveToEngineGeometry[meshIdx].resize(mesh.primitives.size());
        for (std::size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++) {
            m_gltfPrimitiveToEngineGeometry[meshIdx][primIdx] = geometryIndex;
            geometryIndex++;
        }
    }
}

void GLTFLoader::loadMeshes(const tinygltf::Model& model, Scene& scene) {
    std::vector<Geometry> geometry;

    // Extract geometry from glTF
    for (std::size_t meshIdx = 0; meshIdx < model.meshes.size(); meshIdx++) {
        const auto& mesh = model.meshes[meshIdx];
        for (std::size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++) {
            const auto& prim = mesh.primitives[primIdx];
            auto parsedMesh = loadPrimitive(prim, model);
            geometry.emplace_back(parsedMesh);
        }
    }


    // Transform geometry into Meshes
    std::uint32_t currentBaseVertex = 0;
    std::uint32_t currentBaseIndex = 0;

    for (std::size_t meshIdx = 0; meshIdx < model.meshes.size(); meshIdx++) {
        const auto& mesh = model.meshes[meshIdx];
        for (std::size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++) {
            const auto parsedMeshIdx = m_gltfPrimitiveToEngineGeometry[meshIdx][primIdx];
            const auto& parsedMesh = geometry[parsedMeshIdx];

            const auto vertexCount = static_cast<std::uint32_t>(parsedMesh.vertices.size());
            const auto indexCount = static_cast<std::uint32_t>(parsedMesh.indices.size());

            // Compute bounding box for this mesh
            glm::vec3 boundingBoxMin(std::numeric_limits<float>::max());
            glm::vec3 boundingBoxMax(std::numeric_limits<float>::lowest());
            
            for (const auto& vertex : parsedMesh.vertices) {
                boundingBoxMin = glm::min(boundingBoxMin, vertex.position);
                boundingBoxMax = glm::max(boundingBoxMax, vertex.position);
            }

            scene.meshes.emplace_back(Mesh{
                .boundingBoxMin = boundingBoxMin,
                .boundingBoxMax = boundingBoxMax,
                .baseVertex = currentBaseVertex,
                .baseIndex = currentBaseIndex,
                .vertexCount = static_cast<std::uint32_t>(parsedMesh.vertices.size()),
                .indexCount = static_cast<std::uint32_t>(parsedMesh.indices.size()),
                .materialIndex = mesh.primitives[primIdx].material,
            });

            currentBaseVertex += vertexCount;
            currentBaseIndex += indexCount;
        }
    }

    // Populate vertices/indices
    for (const auto& [vertices, indices] : geometry) {
        scene.vertices.insert(scene.vertices.end(), vertices.begin(), vertices.end());
        scene.indices.insert(scene.indices.end(), indices.begin(), indices.end());
    }
}

void GLTFLoader::loadMaterialsAndTextures(const tinygltf::Model& model, Scene& scene) {
    scene.baseColorTextures.clear();
    scene.metallicRoughnessTextures.clear();
    scene.normalTextures.clear();
    scene.emissiveTextures.clear();
    scene.occlusionTextures.clear();

    m_gltfBaseColorTextureMap.clear();
    m_gltfMetallicTextureMap.clear();
    m_gltfNormalTextureMap.clear();
    m_gltfEmissiveTextureMap.clear();
    m_gltfOcclusionTextureMap.clear();

    scene.materials.resize(model.materials.size());

    for (std::size_t matIdx = 0; matIdx < model.materials.size(); matIdx++) {
        const auto& gltfMat = model.materials[matIdx];
        auto& parsedMaterial = scene.materials[matIdx];

        const auto& pbr = gltfMat.pbrMetallicRoughness;

        parsedMaterial.metallicFactor = pbr.metallicFactor;
        parsedMaterial.roughnessFactor = pbr.roughnessFactor;
        parsedMaterial.baseColorFactor = glm::vec4(
            pbr.baseColorFactor[0],
            pbr.baseColorFactor[1],
            pbr.baseColorFactor[2],
            pbr.baseColorFactor[3]
            );
        parsedMaterial.emissiveFactor = glm::vec3(
            gltfMat.emissiveFactor[0],
            gltfMat.emissiveFactor[1],
            gltfMat.emissiveFactor[2]
            );

        if (gltfMat.alphaMode == "MASK") {
            parsedMaterial.alphaMode = 2;
        } else if (gltfMat.alphaMode == "BLEND") {
            parsedMaterial.alphaMode = 1;
        } else {
            parsedMaterial.alphaMode = 0;
        }

        // Read custom properties from glTF extras
        if (gltfMat.extras.Has("reflective")) {
            auto value = gltfMat.extras.Get("reflective");
            if (value.IsBool()) {
                parsedMaterial.reflective = value.Get<bool>() ? 1 : 0;
            } else if (value.IsNumber()) {
                parsedMaterial.reflective = value.GetNumberAsInt() != 0 ? 1 : 0;
            }
        }

        if (gltfMat.extras.Has("castsShadows")) {
            auto value = gltfMat.extras.Get("castsShadows");
            if (value.IsBool()) {
                parsedMaterial.castsShadows = value.Get<bool>() ? 1 : 0;
            } else if (value.IsNumber()) {
                parsedMaterial.castsShadows = value.GetNumberAsInt() != 0 ? 1 : 0;
            }
        }

        if (gltfMat.extras.Has("receivesLighting")) {
            auto value = gltfMat.extras.Get("receivesLighting");
            if (value.IsBool()) {
                parsedMaterial.receivesLighting = value.Get<bool>() ? 1 : 0;
            } else if (value.IsNumber()) {
                parsedMaterial.receivesLighting = value.GetNumberAsInt() != 0 ? 1 : 0;
            }
        }

        loadTextureMap(pbr.baseColorTexture.index, m_gltfBaseColorTextureMap, scene.baseColorTextures, parsedMaterial.baseColorTexIndex, model);
        loadTextureMap(pbr.metallicRoughnessTexture.index, m_gltfMetallicTextureMap, scene.metallicRoughnessTextures, parsedMaterial.metallicRoughnessTexIndex, model);
        loadTextureMap(gltfMat.normalTexture.index, m_gltfNormalTextureMap, scene.normalTextures, parsedMaterial.normalTexIndex, model);
        
        // Skip emissive textures if configured (GPU compatibility mode)
        if (g_textureConfig.skipEmissiveTextures) {
            parsedMaterial.emissiveTexIndex = -1;
        } else {
            loadTextureMap(gltfMat.emissiveTexture.index, m_gltfEmissiveTextureMap, scene.emissiveTextures, parsedMaterial.emissiveTexIndex, model);
        }
        
        loadTextureMap(gltfMat.occlusionTexture.index, m_gltfOcclusionTextureMap, scene.occlusionTextures, parsedMaterial.occlusionTexIndex, model);
    }
}

Texture GLTFLoader::loadTexture(const tinygltf::Texture& texture, const tinygltf::Model& model) {
    const auto& image = model.images[texture.source];
    bool const isHDRISource = (image.bits > 8);

    auto format = vk::Format::eR8G8B8A8Srgb;
    if (image.component == 4) {
        format = vk::Format::eR8G8B8A8Srgb;
    } else if (image.component == 3) {
        format = vk::Format::eR8G8B8Srgb;
    } else if (image.component == 1) {
        format = vk::Format::eR8Unorm;
    }

    if (isHDRISource) {
        if (image.component >= 3) {
            format = vk::Format::eR16G16B16A16Unorm;
        }
    }

    // Apply texture dimension limits to reduce VRAM usage (dynamic based on available VRAM)
    std::uint32_t width = static_cast<std::uint32_t>(image.width);
    std::uint32_t height = static_cast<std::uint32_t>(image.height);
    std::vector<unsigned char> processedImage = image.image;

    if (g_textureConfig.enableDownscaling && (width > g_textureConfig.maxTextureDimension || height > g_textureConfig.maxTextureDimension)) {
        // Calculate downscale factor
        const float scale = static_cast<float>(g_textureConfig.maxTextureDimension) / static_cast<float>(std::max(width, height));
        const std::uint32_t newWidth = static_cast<std::uint32_t>(width * scale);
        const std::uint32_t newHeight = static_cast<std::uint32_t>(height * scale);

        // Simple box filter downsampling
        processedImage = downscaleImage(image.image, width, height, newWidth, newHeight, image.component);
        
        width = newWidth;
        height = newHeight;
    }

    // Calculate mip levels with limit to reduce memory usage (dynamic based on available VRAM)
    const auto fullMipLevels = static_cast<std::uint32_t>(std::floor(
                               std::log2(std::max(width, height)))) + 1;
    const auto mipLevels = std::min(fullMipLevels, g_textureConfig.maxMipLevels);

    return {
        .format = format,
        .mipLevels = mipLevels,
        .width = width,
        .height = height,
        .image = processedImage,
    };
}

template <typename T>
void GLTFLoader::loadTextureMap(const int gltfTexIndex,
                                std::map<std::uint32_t, std::uint32_t>& gltfTextureMap,
                                std::vector<T>& sceneTextures,
                                std::int32_t& parsedMaterialTexIndex,
                                const tinygltf::Model& model) {
    if (gltfTexIndex < 0) {
        return;
    }

    if (auto it = gltfTextureMap.find(gltfTexIndex); it != gltfTextureMap.end()) {
        parsedMaterialTexIndex = it->second;
    } else {
        const auto newArrayIndex = sceneTextures.size();
        gltfTextureMap[gltfTexIndex] = newArrayIndex;
        parsedMaterialTexIndex = newArrayIndex;
        sceneTextures.emplace_back(loadTexture(model.textures[gltfTexIndex], model));
    }
}


void GLTFLoader::loadNodes(const tinygltf::Model& model, Scene& scene) {
    scene.pointLights.clear();
    scene.spotLights.clear();
    
    // Initialize node-to-instance mapping (-1 means no instance for this node)
    scene.nodeToInstanceIndex.clear();
    scene.nodeToInstanceIndex.resize(model.nodes.size(), -1);

    for (std::size_t nodeIdx = 0; nodeIdx < model.nodes.size(); nodeIdx++) {
        const auto& node = model.nodes[nodeIdx];

        if (node.mesh >= 0) {
            loadMeshNode(node, nodeIdx, scene);
        }

        if (node.camera >= 0) {
            const tinygltf::Camera& cam = model.cameras[node.camera];
            loadCameraNode(cam, nodeIdx, scene);
        }

        if (node.light >= 0) {
            const tinygltf::Light& light = model.lights[node.light];
            loadLightNode(light, nodeIdx, node, scene);
        }

        if (node.mesh >= 0 and node.name == "__SkySphere__") {
            loadSkySphereNode(node, model, scene);
        }
    }
}

void GLTFLoader::loadMeshNode(const tinygltf::Node& node, const std::size_t nodeIdx, Scene& scene) {
    scene.nodeToInstanceIndex[nodeIdx] = static_cast<std::int32_t>(scene.instances.size());
    
    std::int32_t reflective = 0;
    std::int32_t castsShadows = 0;
    std::int32_t receivesLighting = 1;
    std::int32_t animated = 0;

    if (node.extras.Has("reflective")) {
        auto value = node.extras.Get("reflective");
        if (value.IsBool()) {
            reflective = value.Get<bool>() ? 1 : 0;
        } else if (value.IsNumber()) {
            reflective = value.GetNumberAsInt() != 0 ? 1 : 0;
        }
    }

    if (node.extras.Has("castsShadows")) {
        auto value = node.extras.Get("castsShadows");
        if (value.IsBool()) {
            castsShadows = value.Get<bool>() ? 1 : 0;
        } else if (value.IsNumber()) {
            castsShadows = value.GetNumberAsInt() != 0 ? 1 : 0;
        }
    }

    if (node.extras.Has("receivesLighting")) {
        auto value = node.extras.Get("receivesLighting");
        if (value.IsBool()) {
            receivesLighting = value.Get<bool>() ? 1 : 0;
        } else if (value.IsNumber()) {
            receivesLighting = value.GetNumberAsInt() != 0 ? 1 : 0;
        }
    }

    if (node.extras.Has("animated")) {
        auto value = node.extras.Get("animated");
        if (value.IsBool()) {
            animated = value.Get<bool>() ? 1 : 0;
        } else if (value.IsNumber()) {
            animated = value.GetNumberAsInt() != 0 ? 1 : 0;
        }
    }
    
    auto prims = m_gltfPrimitiveToEngineGeometry[node.mesh];
    for (const auto& meshIndex : m_gltfPrimitiveToEngineGeometry[node.mesh]) {
        scene.instances.emplace_back(Instance{
            .transform = m_nodeWorldMatrices[nodeIdx],
            .inverseTransform = glm::inverse(m_nodeWorldMatrices[nodeIdx]),
            .meshIndex = static_cast<int>(meshIndex),
            .reflective = reflective,
            .castsShadows = castsShadows,
            .receivesLighting = receivesLighting,
            .animated = animated,
        });
    }
}

void GLTFLoader::loadCameraNode(const tinygltf::Camera& cam, const std::size_t nodeIdx, Scene& scene) {
    scene.camera = {
        .yfov = static_cast<float>(cam.perspective.yfov),
        .aspectRatio = static_cast<float>(cam.perspective.aspectRatio),
        .znear = static_cast<float>(cam.perspective.znear),
        .zfar = static_cast<float>(cam.perspective.zfar),
        .model = m_nodeWorldMatrices[nodeIdx],
    };
}

void GLTFLoader::loadLightNode(const tinygltf::Light& light, std::size_t nodeIdx, const tinygltf::Node& node, Scene& scene) {
    const glm::mat4& worldMat = m_nodeWorldMatrices[nodeIdx];

    if (light.type == "directional") {
        glm::vec3 forward = glm::normalize(glm::vec3(worldMat * glm::vec4(0, 0, -1, 0)));
        scene.directionalLight = {
            .direction = -forward,
            .intensity = static_cast<float>(light.intensity / GLTF_DIRECTIONAL_LIGHT_INTENSITY_CONVERSION_FACTOR),
            .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
        };
    }

    if (light.type == "point") {
        float radius = static_cast<float>(light.range);
        if (radius <= 0.0f) {
            radius = sqrt(light.intensity) * 2.0f; // Heuristic: brighter lights have larger radius
        }

        std::int32_t castsShadows = 0; // Default: cast shadows
        if (node.extras.Has("castsShadows")) {
            auto value = node.extras.Get("castsShadows");
            if (value.IsBool()) {
                castsShadows = value.Get<bool>() ? 1 : 0;
            } else if (value.IsInt()) {
                castsShadows = value.Get<int>();
            }
        }

        std::int32_t animated = 0; // Default: static
        if (node.extras.Has("animated")) {
            auto value = node.extras.Get("animated");
            if (value.IsBool()) {
                animated = value.Get<bool>() ? 1 : 0;
            } else if (value.IsInt()) {
                animated = value.Get<int>();
            }
        }

        PointLight pointLight{
            .position = glm::vec3(worldMat[3]),
            .intensity = static_cast<float>(light.intensity / GLTF_POINT_LIGHT_INTENSITY_CONVERSION_FACTOR),
            .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
            .radius = radius,
            .castsShadows = castsShadows,
            .animated = animated,
        };

        scene.pointLights.emplace_back(pointLight);
    }

    if (light.type == "spot") {
        glm::vec3 forward = glm::normalize(glm::vec3(worldMat * glm::vec4(0, 0, -1, 0)));

        std::int32_t castsShadows = 0; // Default: no shadows
        if (node.extras.Has("castsShadows")) {
            auto value = node.extras.Get("castsShadows");
            if (value.IsBool()) {
                castsShadows = value.Get<bool>() ? 1 : 0;
            } else if (value.IsInt()) {
                castsShadows = value.Get<int>();
            }
        }

        std::int32_t animated = 0; // Default: static
        if (node.extras.Has("animated")) {
            auto value = node.extras.Get("animated");
            if (value.IsBool()) {
                animated = value.Get<bool>() ? 1 : 0;
            } else if (value.IsInt()) {
                animated = value.Get<int>();
            }
        }

        SpotLight spotLight{
            .position = glm::vec3(worldMat[3]),
            .intensity = static_cast<float>(light.intensity / GLTF_SPOT_LIGHT_INTENSITY_CONVERSION_FACTOR),
            .direction = -forward,
            .cutoff = static_cast<float>(light.spot.innerConeAngle),
            .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
            .outerCutoff = static_cast<float>(light.spot.outerConeAngle),
            .castsShadows = castsShadows,
            .animated = animated,
        };

        scene.spotLights.emplace_back(spotLight);
    }
}

void GLTFLoader::loadSkySphereNode(const tinygltf::Node& node, const tinygltf::Model& model, Scene& scene) {
    const auto instanceIndex = m_gltfPrimitiveToEngineGeometry[node.mesh][0];
    scene.skySphereInstanceIndex = instanceIndex;

    // Skip sky sphere texture when emissive textures are disabled
    if (g_textureConfig.skipEmissiveTextures) {
        scene.skySphereTextureIndex = -1;
        return;
    }

    const auto& prim = model.meshes[node.mesh].primitives[0];
    const auto& gltfMat = model.materials[prim.material];
    const auto gltfTexIdx = gltfMat.emissiveTexture.index;

    const auto textureIndex = m_gltfEmissiveTextureMap[gltfTexIdx];

    scene.skySphereTextureIndex = textureIndex;

    scene.emissiveTextures[textureIndex].skyTexture = true;
}

Geometry GLTFLoader::loadPrimitive(const tinygltf::Primitive& prim, const tinygltf::Model& model) {
    Geometry parsedMesh;
    
    auto posIt = prim.attributes.find("POSITION");
    auto normIt = prim.attributes.find("NORMAL");
    auto uvIt = prim.attributes.find("TEXCOORD_0");
    auto tanIt = prim.attributes.find("TANGENT");

    if (posIt == prim.attributes.end()) {
        std::cerr << "WARNING: Primitive missing POSITION attribute, skipping" << std::endl;
        return parsedMesh;
    }
    if (normIt == prim.attributes.end()) {
        return parsedMesh;
    }
    if (uvIt == prim.attributes.end()) {
        return parsedMesh;
    }
    if (tanIt == prim.attributes.end()) {
        return parsedMesh;
    }


    const tinygltf::Accessor& posAccessor = model.accessors[prim.attributes.find("POSITION")->second];
    const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
    const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];

    const tinygltf::Accessor& normalAccessor = model.accessors[prim.attributes.find("NORMAL")->second];
    const tinygltf::BufferView& normalView = model.bufferViews[normalAccessor.bufferView];
    const tinygltf::Buffer& normalBuffer = model.buffers[normalView.buffer];

    const tinygltf::Accessor& uvAccessor = model.accessors[prim.attributes.find("TEXCOORD_0")->second];
    const tinygltf::BufferView& uvView = model.bufferViews[uvAccessor.bufferView];
    const tinygltf::Buffer& uvBuffer = model.buffers[uvView.buffer];

    const std::size_t posStride = posAccessor.ByteStride(posView)
                                      ? posAccessor.ByteStride(posView)
                                      : sizeof(float) * 3;
    const std::size_t normalStride = normalAccessor.ByteStride(normalView)
                                         ? normalAccessor.ByteStride(normalView)
                                         : sizeof(float) * 3;
    const std::size_t uvStride = uvAccessor.ByteStride(uvView) ? uvAccessor.ByteStride(uvView) : sizeof(float) * 2;

    parsedMesh.vertices.resize(posAccessor.count);
    for (std::size_t i = 0; i < posAccessor.count; i++) {
        const auto p = reinterpret_cast<const float*>(
            posBuffer.data.data() + posView.byteOffset + posAccessor.byteOffset + i * posStride);
        const auto n = reinterpret_cast<const float*>(
            normalBuffer.data.data() + normalView.byteOffset + normalAccessor.byteOffset + i *
            normalStride);
        const auto uv = reinterpret_cast<const float*>(
            uvBuffer.data.data() + uvView.byteOffset + uvAccessor.byteOffset + i * uvStride);

        glm::vec4 tangent(0.0f);
        if (auto it = prim.attributes.find("TANGENT"); it != prim.attributes.end()) {
            const tinygltf::Accessor& tanAccessor = model.accessors[it->second];
            const tinygltf::BufferView& tanView = model.bufferViews[tanAccessor.bufferView];
            const tinygltf::Buffer& tanBuffer = model.buffers[tanView.buffer];

            const std::size_t tanStride = tanAccessor.ByteStride(tanView)
                                              ? tanAccessor.ByteStride(tanView)
                                              : sizeof(float) * 4;
            const auto t = reinterpret_cast<const float*>(
                tanBuffer.data.data() + tanView.byteOffset + tanAccessor.byteOffset + i * tanStride);

            tangent = glm::vec4(t[0], t[1], t[2], t[3]);
        }

        parsedMesh.vertices[i] = {
            .position = glm::vec3(p[0], p[1], p[2]),
            .normal = glm::normalize(glm::vec3(n[0], n[1], n[2])),
            .texCoord = glm::vec2(uv[0], uv[1]),
            .tangent = tangent,
        };
    }

    // Index data
    const tinygltf::Accessor& indexAccessor = model.accessors[prim.indices];
    const tinygltf::BufferView& indexView = model.bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer& indexBuffer = model.buffers[indexView.buffer];
    const void* dataPtr = indexBuffer.data.data() + indexView.byteOffset + indexAccessor.byteOffset;

    parsedMesh.indices.resize(indexAccessor.count);
    switch (indexAccessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            auto buf = static_cast<const uint32_t*>(dataPtr);
            for (size_t i = 0; i < indexAccessor.count; i++) {
                parsedMesh.indices[i] = buf[i];
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            auto buf = static_cast<const uint16_t*>(dataPtr);
            for (size_t i = 0; i < indexAccessor.count; i++) {
                parsedMesh.indices[i] = buf[i];
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: {
            auto buf = static_cast<const uint8_t*>(dataPtr);
            for (size_t i = 0; i < indexAccessor.count; i++) {
                parsedMesh.indices[i] = buf[i];
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported index component type");
    }

    return parsedMesh;
}

void GLTFLoader::computeNodeWorldMatrix(const tinygltf::Model& model,
                                        const int nodeIndex,
                                        const glm::mat4& parentMatrix,
                                        std::vector<glm::mat4>& outMatrices) {
    const auto& node = model.nodes[nodeIndex];

    const glm::mat4 local = getLocalTransform(node);
    const glm::mat4 world = parentMatrix * local;

    outMatrices[nodeIndex] = world;

    for (const int childIndex : node.children) {
        computeNodeWorldMatrix(model, childIndex, world, outMatrices);
    }
}

glm::mat4 GLTFLoader::getLocalTransform(const tinygltf::Node& node) {
    glm::mat4 T(1.0f);
    glm::mat4 R(1.0f);
    glm::mat4 S(1.0f);

    if (node.translation.size() == 3) {
        T = glm::translate(glm::mat4(1.0f),
                           glm::vec3(node.translation[0], node.translation[1], node.translation[2]));
    }

    if (node.rotation.size() == 4) {
        R = glm::mat4_cast(glm::quat(
            static_cast<float>(node.rotation[3]), // w
            static_cast<float>(node.rotation[0]), // x
            static_cast<float>(node.rotation[1]), // y
            static_cast<float>(node.rotation[2]) // z
            ));
    }

    if (node.scale.size() == 3) {
        S = glm::scale(glm::mat4(1.0f),
                       glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    }

    glm::mat4 M(1.0f);
    if (node.matrix.size() == 16) {
        M = glm::make_mat4x4(node.matrix.data());
    } else {
        M = T * R * S;
    }

    return M;
}

void GLTFLoader::buildMeshToInstanceMapping(Scene& scene) {
    scene.meshToInstanceIndices.clear();
    scene.meshToInstanceIndices.resize(scene.meshes.size());

    // Build the mapping without reordering instances
    // This preserves the original instance order and keeps nodeToInstanceIndex valid
    for (std::uint32_t instanceIdx = 0; instanceIdx < scene.instances.size(); instanceIdx++) {
        const auto& instance = scene.instances[instanceIdx];
        if (instance.meshIndex >= 0 && instance.meshIndex < static_cast<std::int32_t>(scene.meshes.size())) {
            scene.meshToInstanceIndices[instance.meshIndex].push_back(instanceIdx);
        }
    }
}

std::vector<unsigned char> GLTFLoader::downscaleImage(const std::vector<unsigned char>& srcImage,
                                                       const std::uint32_t srcWidth,
                                                       const std::uint32_t srcHeight,
                                                       const std::uint32_t dstWidth,
                                                       const std::uint32_t dstHeight,
                                                       const int components) {
    std::vector<unsigned char> dstImage(dstWidth * dstHeight * components);

    const float xRatio = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    const float yRatio = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);

    // Simple box filter downsampling
    for (std::uint32_t y = 0; y < dstHeight; y++) {
        for (std::uint32_t x = 0; x < dstWidth; x++) {
            // Calculate the source region
            const std::uint32_t srcX = static_cast<std::uint32_t>(x * xRatio);
            const std::uint32_t srcY = static_cast<std::uint32_t>(y * yRatio);

            // Sample from source
            for (int c = 0; c < components; c++) {
                const std::uint32_t srcIndex = (srcY * srcWidth + srcX) * components + c;
                const std::uint32_t dstIndex = (y * dstWidth + x) * components + c;
                dstImage[dstIndex] = srcImage[srcIndex];
            }
        }
    }

    return dstImage;
}
