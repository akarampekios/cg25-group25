#pragma once

#include <vector>
#include <tiny_gltf.h>

#include "utils/image_utils.hpp"
#include "utils/buffer_utils.hpp"
#include "constants.hpp"
#include "structs.hpp"
#include "vulkan_transfer_context.hpp"

class Loader {
public:
    explicit Loader(const VulkanTransferContext& vulkanTransferContext) :
        m_vulkanTransferContext{vulkanTransferContext},
        m_device{vulkanTransferContext.device},
        m_physicalDevice{vulkanTransferContext.physicalDevice},
        m_imageUtils{vulkanTransferContext},
        m_bufferUtils{vulkanTransferContext} {
    }

    auto loadGltfScene(const std::string& path) -> Scene {
        tinygltf::TinyGLTF loader;
        tinygltf::Model scene;

        std::string err;
        std::string warn;

        loader.LoadBinaryFromFile(&scene, &err, &warn, path);

        auto materials = loadMaterials(scene);
        auto nodeWorldMatrices = loadNodeWorldMatrices(scene);
        auto meshes = loadMeshes(scene, materials, nodeWorldMatrices);
        auto camera = loadCamera(scene, nodeWorldMatrices);

        for (auto& mesh : meshes) {
            computeTangents(mesh.vertices, mesh.indices);

            createVertexBuffer(mesh);
            createIndexBuffer(mesh);
            createUniformBuffers(mesh);
            createMaterialBuffers(mesh);
            createDescriptorSets(mesh);
        }

        return {
            .materials = std::move(materials),
            .meshes = std::move(meshes),
            .camera = std::move(camera),
        };
    }

private:
    ImageUtils m_imageUtils;
    BufferUtils m_bufferUtils;

    const VulkanTransferContext& m_vulkanTransferContext;
    const vk::raii::Device* m_device;
    const vk::raii::PhysicalDevice* m_physicalDevice;

    auto loadMaterials(const tinygltf::Model& scene) -> std::vector<Material> {
        std::vector<Material> parsedMaterials(scene.materials.size());

        for (std::size_t i = 0; i < scene.materials.size(); i++) {
            const auto& gltfMat = scene.materials[i];
            Material& mat = parsedMaterials[i];

            const auto& pbr = gltfMat.pbrMetallicRoughness;
            mat.baseColorFactor = glm::vec4(
                pbr.baseColorFactor[0],
                pbr.baseColorFactor[1],
                pbr.baseColorFactor[2],
                pbr.baseColorFactor[3]
                );
            mat.metallicFactor = pbr.metallicFactor;
            mat.roughnessFactor = pbr.roughnessFactor;
            mat.emissiveFactor = glm::vec3(
                gltfMat.emissiveFactor[0],
                gltfMat.emissiveFactor[1],
                gltfMat.emissiveFactor[2]
                );

            if (pbr.baseColorTexture.index >= 0) {
                mat.baseColorTex = createTextureFromGLBImage(scene.images[pbr.baseColorTexture.index]);
            }

            if (pbr.metallicRoughnessTexture.index >= 0) {
                mat.metallicRoughnessTex = createTextureFromGLBImage(scene.images[pbr.metallicRoughnessTexture.index]);
            }

            if (gltfMat.normalTexture.index >= 0) {
                mat.normalTex = createTextureFromGLBImage(scene.images[gltfMat.normalTexture.index]);
            }

            if (gltfMat.emissiveTexture.index >= 0) {
                mat.emissiveTex = createTextureFromGLBImage(scene.images[gltfMat.emissiveTexture.index]);
            }

            if (gltfMat.occlusionTexture.index >= 0) {
                mat.occlusionTex = createTextureFromGLBImage(scene.images[gltfMat.occlusionTexture.index]);
            }
        }

        return parsedMaterials;
    }

    std::vector<glm::mat4> loadNodeWorldMatrices(const tinygltf::Model& scene) {
        std::vector meshWorldMatrices(scene.nodes.size(), glm::mat4(1.0f));

        for (std::size_t rootIdx = 0; rootIdx < scene.nodes.size(); rootIdx++) {
            computeNodeWorldMatrix(scene, rootIdx, glm::mat4(1.0f), meshWorldMatrices);
        }

        return meshWorldMatrices;
    }

    void computeNodeWorldMatrix(const tinygltf::Model& model,
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

    glm::mat4 getLocalTransform(const tinygltf::Node& node) {
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

    void computeTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
        // Temporary accumulators
        std::vector<glm::vec3> tan1(vertices.size(), glm::vec3(0.0f));
        std::vector<glm::vec3> tan2(vertices.size(), glm::vec3(0.0f));

        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            const glm::vec3& v0 = vertices[i0].position;
            const glm::vec3& v1 = vertices[i1].position;
            const glm::vec3& v2 = vertices[i2].position;

            const glm::vec2& uv0 = vertices[i0].texCoord;
            const glm::vec2& uv1 = vertices[i1].texCoord;
            const glm::vec2& uv2 = vertices[i2].texCoord;

            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;

            glm::vec2 deltaUV1 = uv1 - uv0;
            glm::vec2 deltaUV2 = uv2 - uv0;

            float r = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
            glm::vec3 sdir = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * r;
            glm::vec3 tdir = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) * r;

            tan1[i0] += sdir;
            tan1[i1] += sdir;
            tan1[i2] += sdir;
            tan2[i0] += tdir;
            tan2[i1] += tdir;
            tan2[i2] += tdir;
        }

        // Orthogonalize and store tangents with handedness
        for (size_t i = 0; i < vertices.size(); ++i) {
            const glm::vec3& n = vertices[i].normal;
            const glm::vec3& t = tan1[i];

            // Gram-Schmidt orthogonalize
            glm::vec3 tangent = glm::normalize(t - n * glm::dot(n, t));

            // Compute handedness
            float w = (glm::dot(glm::cross(n, t), tan2[i]) < 0.0f) ? -1.0f : 1.0f;

            vertices[i].tangent = glm::vec4(tangent, w);
        }
    }

    auto loadMeshes(const tinygltf::Model& scene,
                    std::vector<Material>& materials,
                    const std::vector<glm::mat4>& nodeWorldMatrices) -> std::vector<Mesh> {
        std::vector<Mesh> meshes;

        for (std::size_t nodeIdx = 0; nodeIdx < scene.nodes.size(); nodeIdx++) {
            const auto& node = scene.nodes[nodeIdx];

            if (node.mesh < 0) {
                continue;
            }

            const auto& mesh = scene.meshes[node.mesh];
            for (const auto& prim : mesh.primitives) {
                Mesh parsedMesh;

                const tinygltf::Accessor& posAccessor = scene.accessors[prim.attributes.find("POSITION")->second];
                const tinygltf::BufferView& posView = scene.bufferViews[posAccessor.bufferView];
                const tinygltf::Buffer& posBuffer = scene.buffers[posView.buffer];

                const tinygltf::Accessor& normalAccessor = scene.accessors[prim.attributes.find("NORMAL")->second];
                const tinygltf::BufferView& normalView = scene.bufferViews[normalAccessor.bufferView];
                const tinygltf::Buffer& normalBuffer = scene.buffers[normalView.buffer];

                const tinygltf::Accessor& uvAccessor = scene.accessors[prim.attributes.find("TEXCOORD_0")->second];
                const tinygltf::BufferView& uvView = scene.bufferViews[uvAccessor.bufferView];
                const tinygltf::Buffer& uvBuffer = scene.buffers[uvView.buffer];

                // const tinygltf::Accessor& tangentAccessor = scene.accessors[prim.attributes.find("TANGENT")->second];
                // const tinygltf::BufferView& tangentView = scene.bufferViews[tangentAccessor.bufferView];
                // const tinygltf::Buffer& tangentBuffer = scene.buffers[tangentView.buffer];

                size_t posStride = posAccessor.ByteStride(posView)
                                       ? posAccessor.ByteStride(posView)
                                       : sizeof(float) * 3;
                size_t normalStride = normalAccessor.ByteStride(normalView)
                                          ? normalAccessor.ByteStride(normalView)
                                          : sizeof(float) * 3;
                size_t uvStride = uvAccessor.ByteStride(uvView) ? uvAccessor.ByteStride(uvView) : sizeof(float) * 2;
                // size_t tangentStride = tangentAccessor.ByteStride(tangentView)
                //                            ? tangentAccessor.ByteStride(tangentView)
                //                            : sizeof(float) * 4;

                parsedMesh.vertices.resize(posAccessor.count);
                for (std::size_t i = 0; i < posAccessor.count; i++) {
                    const float* p = reinterpret_cast<const float*>(
                        posBuffer.data.data() + posView.byteOffset + posAccessor.byteOffset + i * posStride);
                    const float* n = reinterpret_cast<const float*>(
                        normalBuffer.data.data() + normalView.byteOffset + normalAccessor.byteOffset + i *
                        normalStride);
                    const float* uv = reinterpret_cast<const float*>(
                        uvBuffer.data.data() + uvView.byteOffset + uvAccessor.byteOffset + i * uvStride);
                    // const float* t = reinterpret_cast<const float*>(
                    //     tangentBuffer.data.data() + tangentView.byteOffset + tangentAccessor.byteOffset + i *
                    //     tangentStride);


                    parsedMesh.vertices[i] = {
                        .position = glm::vec3(p[0], p[1], p[2]),
                        .normal = glm::vec3(n[0], n[1], n[2]),
                        .texCoord = glm::vec2(uv[0], uv[1]),
                    };
                }

                // Index data
                const tinygltf::Accessor& indexAccessor = scene.accessors[prim.indices];
                const tinygltf::BufferView& indexView = scene.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = scene.buffers[indexView.buffer];
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

                // FIXME: think what to do if there is no material assigned - in this case descriptor set creation will fail
                if (prim.material >= 0) {
                    parsedMesh.material = &materials[prim.material];
                }

                parsedMesh.model = nodeWorldMatrices[nodeIdx];

                meshes.push_back(std::move(parsedMesh));
            }
        }

        return meshes;
    }

    CameraParameters loadCamera(const tinygltf::Model& scene, const std::vector<glm::mat4>& nodeWorldMatrices) {
        for (std::size_t nodeIdx = 0; nodeIdx < scene.nodes.size(); nodeIdx++) {
            const auto& node = scene.nodes[nodeIdx];

            if (node.camera < 0) {
                continue;
            }

            const tinygltf::Camera& cam = scene.cameras[node.camera];

            CameraParameters camera{
                .yfov = static_cast<float>(cam.perspective.yfov),
                .aspectRatio = static_cast<float>(cam.perspective.aspectRatio),
                .znear = static_cast<float>(cam.perspective.znear),
                .zfar = static_cast<float>(cam.perspective.zfar),
            };

            camera.model = nodeWorldMatrices[nodeIdx];

            return camera;
        }

        throw std::runtime_error("No cameras found");
    }

    void createVertexBuffer(Mesh& mesh) {
        const vk::DeviceSize bufferSize = sizeof(mesh.vertices[0]) * mesh.vertices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        m_bufferUtils.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
            );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, mesh.vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        m_bufferUtils.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            mesh.vertexBuffer,
            mesh.vertexBufferMemory
            );

        m_bufferUtils.copyBuffer(stagingBuffer, mesh.vertexBuffer, bufferSize);
    }

    void createIndexBuffer(Mesh& mesh) {
        const vk::DeviceSize bufferSize = sizeof(mesh.indices[0]) * mesh.indices.size();

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        m_bufferUtils.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
            );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, mesh.indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        m_bufferUtils.createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            mesh.indexBuffer,
            mesh.indexBufferMemory
            );

        m_bufferUtils.copyBuffer(stagingBuffer, mesh.indexBuffer, bufferSize);
    }

    void createUniformBuffers(Mesh& mesh) {
        mesh.uniformBuffers.clear();
        mesh.uniformBuffersMemory.clear();
        mesh.uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            constexpr vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});

            m_bufferUtils.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                       vk::MemoryPropertyFlagBits::eHostVisible |
                                       vk::MemoryPropertyFlagBits::eHostCoherent,
                                       buffer, bufferMem);

            mesh.uniformBuffers.emplace_back(std::move(buffer));
            mesh.uniformBuffersMemory.emplace_back(std::move(bufferMem));
            mesh.uniformBuffersMapped.emplace_back(mesh.uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createMaterialBuffers(Mesh& mesh) {
        mesh.materialBuffers.clear();
        mesh.materialBuffers.clear();
        mesh.materialBuffers.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            constexpr vk::DeviceSize bufferSize = sizeof(MaterialBufferObject);

            vk::raii::Buffer buffer({});
            vk::raii::DeviceMemory bufferMem({});

            m_bufferUtils.createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                                       vk::MemoryPropertyFlagBits::eHostVisible |
                                       vk::MemoryPropertyFlagBits::eHostCoherent,
                                       buffer, bufferMem);

            mesh.materialBuffers.emplace_back(std::move(buffer));
            mesh.materialBuffersMemory.emplace_back(std::move(bufferMem));
            mesh.materialBuffersMapped.emplace_back(mesh.materialBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorSets(Mesh& mesh) const {
        std::vector<vk::DescriptorSetLayout>
            layouts(MAX_FRAMES_IN_FLIGHT, *m_vulkanTransferContext.descriptorSetLayout);

        const vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = *m_vulkanTransferContext.descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        mesh.descriptorSets.clear();
        mesh.descriptorSets = m_device->allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo uboBufferInfo{
                .buffer = *mesh.uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };
            vk::DescriptorBufferInfo materialBufferInfo{
                .buffer = *mesh.materialBuffers[i],
                .offset = 0,
                .range = sizeof(MaterialBufferObject)
            };

            vk::DescriptorImageInfo baseColorImageInfo{
                .sampler = *m_vulkanTransferContext.baseColorTextureSampler,
                .imageView = *mesh.material->baseColorTex.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo metallicRoughnessImageInfo{
                .sampler = *m_vulkanTransferContext.baseColorTextureSampler,
                .imageView = *mesh.material->metallicRoughnessTex.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo normalImageInfo{
                .sampler = *m_vulkanTransferContext.normalTextureSampler,
                .imageView = *mesh.material->normalTex.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo emissiveImageInfo{
                .sampler = *m_vulkanTransferContext.emissiveTextureSampler,
                .imageView = *mesh.material->emissiveTex.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };
            vk::DescriptorImageInfo occlusionImageInfo{
                .sampler = *m_vulkanTransferContext.occlusionTextureSampler,
                .imageView = *mesh.material->occlusionTex.imageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };

            std::vector<vk::WriteDescriptorSet> descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &uboBufferInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo = &materialBufferInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 2,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &baseColorImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 3,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &metallicRoughnessImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 4,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &normalImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 5,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &emissiveImageInfo
                },
                vk::WriteDescriptorSet{
                    .dstSet = *mesh.descriptorSets[i],
                    .dstBinding = 6,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo = &occlusionImageInfo
                }
            };
            m_device->updateDescriptorSets(descriptorWrites, {});
        }
    }

    Texture createTextureFromGLBImage(const tinygltf::Image& image) {
        Texture texture;

        auto format = vk::Format::eR8G8B8A8Srgb;
        if (image.component == 4) {
            format = vk::Format::eR8G8B8A8Srgb;
        } else if (image.component == 3) {
            format = vk::Format::eR8G8B8Srgb;
        } else if (image.component == 1) {
            format = vk::Format::eR8Unorm;
        }

        const auto mipLevels = static_cast<uint32_t>(std::floor(
                                   std::log2(std::max(image.width, image.height)))) + 1;

        m_imageUtils.createImage(
            image.width,
            image.height,
            mipLevels,
            vk::SampleCountFlagBits::e1,
            format,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferSrc
            | vk::ImageUsageFlagBits::eTransferDst
            | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            texture.image,
            texture.imageMemory
            );

        m_imageUtils.transitionImageLayout(
            texture.image,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal,
            mipLevels
            );

        vk::raii::Buffer stagingBuffer({});
        vk::raii::DeviceMemory stagingBufferMemory({});
        const vk::DeviceSize imageSize = image.image.size();

        m_bufferUtils.createBuffer(
            imageSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory
            );

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, image.image.data(), imageSize);
        stagingBufferMemory.unmapMemory();

        m_bufferUtils.copyBufferToImage(
            stagingBuffer,
            texture.image,
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.width)
            );

        m_imageUtils.generateMipmaps(
            texture.image,
            format,
            image.width,
            image.width,
            mipLevels
            );

        texture.imageView = m_imageUtils.createImageView(
            texture.image,
            format,
            vk::ImageAspectFlagBits::eColor,
            mipLevels
            );

        const auto properties = m_physicalDevice->getProperties();
        const vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0F,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
        };

        texture.sampler = vk::raii::Sampler(*m_device, samplerInfo);

        return texture;
    }
};
