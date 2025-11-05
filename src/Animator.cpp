#include <algorithm>
#include <cmath>
#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Animator.hpp"
#include "constants.hpp"

namespace {
inline void decomposeTRS(const glm::mat4& M, glm::vec3& T, glm::quat& R, glm::vec3& S) {
    T = glm::vec3(M[3]);

    glm::vec3 col0 = glm::vec3(M[0]);
    glm::vec3 col1 = glm::vec3(M[1]);
    glm::vec3 col2 = glm::vec3(M[2]);

    float sx = glm::length(col0);
    float sy = glm::length(col1);
    float sz = glm::length(col2);

    if (sx == 0.0f) sx = 1.0f;
    if (sy == 0.0f) sy = 1.0f;
    if (sz == 0.0f) sz = 1.0f;

    S = glm::vec3(sx, sy, sz);

    glm::mat3 rot(
        glm::vec3(col0 / sx),
        glm::vec3(col1 / sy),
        glm::vec3(col2 / sz)
    );
    R = glm::quat_cast(rot);
}

inline float getAnimationDuration(const tinygltf::Animation& anim, const tinygltf::Model& model) {
    float maxTime = 0.0f;
    for (const auto& samp : anim.samplers) {
        const auto& inAcc = model.accessors[samp.input];
        if (inAcc.count == 0) continue;
        const auto& inView = model.bufferViews[inAcc.bufferView];
        const auto& inBuf = model.buffers[inView.buffer];
        const std::uint8_t* base = inBuf.data.data() + inView.byteOffset + inAcc.byteOffset;
        const std::size_t stride = inAcc.ByteStride(inView) ? inAcc.ByteStride(inView) : sizeof(float);
        const float* last = reinterpret_cast<const float*>(base + (inAcc.count - 1) * stride);
        maxTime = std::max(maxTime, *last);
    }
    return maxTime;
}

inline void getSamplerTimes(const tinygltf::AnimationSampler& samp, const tinygltf::Model& model,
                            const float*& tBase, std::size_t& count, std::size_t& stride) {
    const auto& inAcc = model.accessors[samp.input];
    const auto& inView = model.bufferViews[inAcc.bufferView];
    const auto& inBuf = model.buffers[inView.buffer];
    tBase = reinterpret_cast<const float*>(inBuf.data.data() + inView.byteOffset + inAcc.byteOffset);
    count = inAcc.count;
    stride = inAcc.ByteStride(inView) ? inAcc.ByteStride(inView) : sizeof(float);
}

inline bool isCubicSpline(const tinygltf::AnimationSampler& samp) {
    return samp.interpolation == "CUBICSPLINE";
}

template <int N>
inline const float* getOutputPtrAt(const tinygltf::AnimationSampler& samp, const tinygltf::Model& model, std::size_t keyIndex) {
    const auto& outAcc = model.accessors[samp.output];
    const auto& outView = model.bufferViews[outAcc.bufferView];
    const auto& outBuf = model.buffers[outView.buffer];
    const bool cubic = isCubicSpline(samp);
    const std::size_t elemStride = outAcc.ByteStride(outView)
        ? outAcc.ByteStride(outView)
        : sizeof(float) * N * (cubic ? 3 : 1);
    const std::size_t valueOffset = cubic ? sizeof(float) * N /*middle value*/ : 0;
    const std::uint8_t* base = outBuf.data.data() + outView.byteOffset + outAcc.byteOffset + keyIndex * elemStride + valueOffset;
    return reinterpret_cast<const float*>(base);
}

inline std::size_t findKeyframeIndex(const float* tBase, std::size_t count, std::size_t strideBytes, float t) {
    if (count == 0) return 0;
    if (count == 1) return 0;

    auto timeAt = [&](std::size_t i) {
        return *reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(tBase) + i * strideBytes);
    };

    if (t <= timeAt(0)) return 0;
    if (t >= timeAt(count - 1)) return count - 1;

    for (std::size_t i = 0; i + 1 < count; ++i) {
        const float t0 = timeAt(i);
        const float t1 = timeAt(i + 1);
        if (t >= t0 && t <= t1) {
            return i;
        }
    }
    return count - 1;
}

inline glm::vec3 sampleVec3(const tinygltf::AnimationSampler& samp, const tinygltf::Model& model, float t) {
    const float* tBase = nullptr; std::size_t keyCount = 0; std::size_t tStride = 0;
    getSamplerTimes(samp, model, tBase, keyCount, tStride);
    if (keyCount == 0) return glm::vec3(0.0f);
    if (keyCount == 1) {
        const float* v = getOutputPtrAt<3>(samp, model, 0);
        return glm::vec3(v[0], v[1], v[2]);
    }

    const std::size_t i = findKeyframeIndex(tBase, keyCount, tStride, t);
    const auto timeAt = [&](std::size_t idx) {
        return *reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(tBase) + idx * tStride);
    };

    if (i == keyCount - 1) {
        const float* v = getOutputPtrAt<3>(samp, model, i);
        return glm::vec3(v[0], v[1], v[2]);
    }

    const float t0 = timeAt(i);
    const float t1 = timeAt(i + 1);
    const float u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;

    if (samp.interpolation == "STEP") {
        const float* v = getOutputPtrAt<3>(samp, model, i);
        return glm::vec3(v[0], v[1], v[2]);
    }

    // LINEAR or CUBICSPLINE (use only value term for now)
    const float* v0 = getOutputPtrAt<3>(samp, model, i);
    const float* v1 = getOutputPtrAt<3>(samp, model, i + 1);
    return glm::mix(glm::vec3(v0[0], v0[1], v0[2]), glm::vec3(v1[0], v1[1], v1[2]), u);
}

inline glm::quat sampleQuat(const tinygltf::AnimationSampler& samp, const tinygltf::Model& model, float t) {
    const float* tBase = nullptr; std::size_t keyCount = 0; std::size_t tStride = 0;
    getSamplerTimes(samp, model, tBase, keyCount, tStride);
    if (keyCount == 0) return glm::quat(1, 0, 0, 0);
    if (keyCount == 1) {
        const float* v = getOutputPtrAt<4>(samp, model, 0);
        return glm::normalize(glm::quat(v[3], v[0], v[1], v[2])); // glTF stores (x,y,z,w)
    }

    const std::size_t i = findKeyframeIndex(tBase, keyCount, tStride, t);
    const auto timeAt = [&](std::size_t idx) {
        return *reinterpret_cast<const float*>(reinterpret_cast<const std::uint8_t*>(tBase) + idx * tStride);
    };

    if (i == keyCount - 1) {
        const float* v = getOutputPtrAt<4>(samp, model, i);
        return glm::normalize(glm::quat(v[3], v[0], v[1], v[2]));
    }

    const float t0 = timeAt(i);
    const float t1 = timeAt(i + 1);
    const float u = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;

    if (samp.interpolation == "STEP") {
        const float* v = getOutputPtrAt<4>(samp, model, i);
        return glm::normalize(glm::quat(v[3], v[0], v[1], v[2]));
    }

    const float* v0 = getOutputPtrAt<4>(samp, model, i);
    const float* v1 = getOutputPtrAt<4>(samp, model, i + 1);
    glm::quat q0 = glm::normalize(glm::quat(v0[3], v0[0], v0[1], v0[2]));
    glm::quat q1 = glm::normalize(glm::quat(v1[3], v1[0], v1[1], v1[2]));
    if (glm::dot(q0, q1) < 0.0f) q1 = -q1; // shortest path
    return glm::normalize(glm::slerp(q0, q1, u));
}

void computeNodeWorldMatrixAnimated(const tinygltf::Model& model,
                                    const int nodeIndex,
                                    const glm::mat4& parentMatrix,
                                    const std::vector<glm::mat4>& localMatrices,
                                    std::vector<glm::mat4>& outMatrices) {
    const auto& node = model.nodes[nodeIndex];

    const glm::mat4 local = localMatrices[static_cast<std::size_t>(nodeIndex)];
    const glm::mat4 world = parentMatrix * local;

    outMatrices[static_cast<std::size_t>(nodeIndex)] = world;

    for (const int childIndex : node.children) {
        computeNodeWorldMatrixAnimated(model, childIndex, world, localMatrices, outMatrices);
    }
}

// Find root nodes (nodes that are not children of any other node)
std::vector<int> findRootNodes(const tinygltf::Model& model) {
    std::vector<bool> isChild(model.nodes.size(), false);
    for (const auto& node : model.nodes) {
        for (const int childIdx : node.children) {
            if (childIdx >= 0 && static_cast<std::size_t>(childIdx) < model.nodes.size()) {
                isChild[childIdx] = true;
            }
        }
    }
    std::vector<int> roots;
    for (std::size_t i = 0; i < model.nodes.size(); ++i) {
        if (!isChild[i]) {
            roots.push_back(static_cast<int>(i));
        }
    }
    return roots;
}
} // namespace

void Animator::animate(const tinygltf::Model& model, Scene& scene, float time) {
    const std::size_t nodeCount = model.nodes.size();
    if (nodeCount == 0) return;

    // Check if there are any animations to process
    if (model.animations.empty()) {
        return; // No animations to apply
    }

    // Prepare default TRS per node (from node's static transform)
    std::vector<glm::vec3> translations(nodeCount, glm::vec3(0.0f));
    std::vector<glm::quat> rotations(nodeCount, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    std::vector<glm::vec3> scales(nodeCount, glm::vec3(1.0f));
    
    for (std::size_t i = 0; i < nodeCount; ++i) {
        const auto& node = model.nodes[i];
        glm::vec3 T(0.0f), S(1.0f);
        glm::quat R(1.0f, 0.0f, 0.0f, 0.0f);

        // Match GLTFLoader behavior: matrix takes precedence over TRS
        if (node.matrix.size() == 16) {
            // Node has explicit matrix - decompose it
            glm::mat4 M = glm::make_mat4x4(node.matrix.data());
            decomposeTRS(M, T, R, S);
        } else {
            // Node uses TRS components
            if (node.translation.size() == 3) {
                T = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
            }
            if (node.rotation.size() == 4) {
                R = glm::quat(
                    static_cast<float>(node.rotation[3]), // w
                    static_cast<float>(node.rotation[0]), // x
                    static_cast<float>(node.rotation[1]), // y
                    static_cast<float>(node.rotation[2])  // z
                );
            }
            if (node.scale.size() == 3) {
                S = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
            }
        }

        translations[i] = T;
        rotations[i] = glm::normalize(R);
        scales[i] = S;
    }

    // Find the maximum duration across all animations for synchronized looping
    float maxDuration = 0.0f;
    for (const auto& animation : model.animations) {
        const float duration = getAnimationDuration(animation, model);
        maxDuration = std::max(maxDuration, duration);
    }

    // Use global time synchronized across all animations
    const float globalTime = maxDuration > 0.0f ? std::fmod(std::max(time, 0.0f), maxDuration) : 0.0f;

    // Apply animations: sample each channel at synchronized global time
    for (const auto& animation : model.animations) {
        for (const auto& channel : animation.channels) {
            const int nodeIndex = channel.target_node;
            if (nodeIndex < 0 || static_cast<std::size_t>(nodeIndex) >= nodeCount) continue;
            const auto& samp = animation.samplers[channel.sampler];

            if (channel.target_path == "translation") {
                translations[nodeIndex] = sampleVec3(samp, model, globalTime);
            } else if (channel.target_path == "rotation") {
                rotations[nodeIndex] = sampleQuat(samp, model, globalTime);
            } else if (channel.target_path == "scale") {
                scales[nodeIndex] = sampleVec3(samp, model, globalTime);
            } else {
                // weights not supported currently
            }
        }
    }

    // Build local matrices
    std::vector<glm::mat4> localMats(nodeCount);
    for (std::size_t i = 0; i < nodeCount; ++i) {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), translations[i]);
        glm::mat4 R = glm::mat4_cast(glm::normalize(rotations[i]));
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scales[i]);
        localMats[i] = T * R * S;
    }

    // Compute world matrices - only process actual root nodes
    std::vector<glm::mat4> worldMats(nodeCount, glm::mat4(1.0f));
    const auto rootNodes = findRootNodes(model);
    for (const int rootIdx : rootNodes) {
        computeNodeWorldMatrixAnimated(model, rootIdx, glm::mat4(1.0f), localMats, worldMats);
    }

    // Update mesh instances using the node-to-instance mapping
    for (std::size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
        const auto& node = model.nodes[nodeIdx];
        if (node.mesh >= 0 && static_cast<std::size_t>(nodeIdx) < scene.nodeToInstanceIndex.size()) {
            const std::int32_t firstInstanceIdx = scene.nodeToInstanceIndex[nodeIdx];
            if (firstInstanceIdx >= 0) {
                // Update all instances (primitives) for this node
                const auto primCount = model.meshes[static_cast<std::size_t>(node.mesh)].primitives.size();
                for (std::size_t p = 0; p < primCount; ++p) {
                    const std::size_t instanceIdx = static_cast<std::size_t>(firstInstanceIdx) + p;
                    if (instanceIdx < scene.instances.size()) {
                        scene.instances[instanceIdx].transform = worldMats[nodeIdx];
                        scene.instances[instanceIdx].inverseTransform = glm::inverse(worldMats[nodeIdx]);
                    }
                }
            }
        }
    }

    // Update camera
    for (std::size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
        const auto& node = model.nodes[nodeIdx];
        if (node.camera >= 0) {
            const tinygltf::Camera& cam = model.cameras[node.camera];
            scene.camera = {
                .yfov = static_cast<float>(cam.perspective.yfov),
                .aspectRatio = static_cast<float>(cam.perspective.aspectRatio),
                .znear = static_cast<float>(cam.perspective.znear),
                .zfar = static_cast<float>(cam.perspective.zfar),
                .model = worldMats[nodeIdx],
            };
        }
    }

    // Update lights preserving order
    std::size_t pointIdx = 0;
    std::size_t spotIdx = 0;
    for (std::size_t nodeIdx = 0; nodeIdx < nodeCount; ++nodeIdx) {
        const auto& node = model.nodes[nodeIdx];
        if (node.light < 0) continue;
        const glm::mat4& world = worldMats[nodeIdx];
        const tinygltf::Light& light = model.lights[node.light];

        if (light.type == "directional") {
            glm::vec3 forward = glm::normalize(glm::vec3(world * glm::vec4(0, 0, -1, 0)));
            scene.directionalLight = {
                .direction = -forward,
                .intensity = static_cast<float>(light.intensity / GLTF_DIRECTIONAL_LIGHT_INTENSITY_CONVERSION_FACTOR),
                .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
            };
        } else if (light.type == "point") {
            if (pointIdx < scene.pointLights.size()) {
                auto& pointLight = scene.pointLights[pointIdx];
                // Preserve castsShadows and animated properties
                const std::int32_t castsShadows = pointLight.castsShadows;
                const std::int32_t animated = pointLight.animated;
                
                pointLight = {
                    .position = glm::vec3(world[3]),
                    .intensity = static_cast<float>(light.intensity / GLTF_POINT_LIGHT_INTENSITY_CONVERSION_FACTOR),
                    .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
                    .radius = static_cast<float>(light.range),
                    .castsShadows = castsShadows,
                    .animated = animated,
                };
            }
            ++pointIdx;
        } else if (light.type == "spot") {
            if (spotIdx < scene.spotLights.size()) {
                auto& spotLight = scene.spotLights[spotIdx];
                // Preserve castsShadows and animated properties
                const std::int32_t castsShadows = spotLight.castsShadows;
                const std::int32_t animated = spotLight.animated;
                
                glm::vec3 forward = glm::normalize(glm::vec3(world * glm::vec4(0, 0, -1, 0)));
                spotLight = {
                    .position = glm::vec3(world[3]),
                    .intensity = static_cast<float>(light.intensity / GLTF_SPOT_LIGHT_INTENSITY_CONVERSION_FACTOR),
                    .direction = -forward,
                    .cutoff = static_cast<float>(light.spot.innerConeAngle),
                    .color = glm::vec3(light.color[0], light.color[1], light.color[2]),
                    .outerCutoff = static_cast<float>(light.spot.outerConeAngle),
                    .castsShadows = castsShadows,
                    .animated = animated,
                };
            }
            ++spotIdx;
        }
    }
}
