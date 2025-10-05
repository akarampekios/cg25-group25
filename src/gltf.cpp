// #include <iostream>
// #include <tiny_gltf.h>
//
// #include "utils/image_utils.hpp"
// #include "utils/buffer_utils.hpp"
//
// struct Texture {
//   vk::raii::Image image;
//   vk::raii::DeviceMemory imageMemory;
//   vk::raii::ImageView imageView;
//   vk::raii::Sampler sampler;
// };
// //
// struct Material {
//   VkPipeline pipeline;
//   VkPipelineLayout layout;
//   VkDescriptorSet descriptorSet;
//
//   Texture* baseColorTex = nullptr;
//   Texture* metallicRoughnessTex = nullptr;
//   Texture* normalTex = nullptr;
//   Texture* emissiveTex = nullptr;
//   Texture* occlusionTex = nullptr;
//
//   glm::vec4 baseColorFactor = glm::vec4(1.0f);
//   float metallicFactor = 1.0f;
//   float roughnessFactor = 1.0f;
//   glm::vec3 emissiveFactor = glm::vec3(0.0f);
// };
//
// Texture createTextureFromGLBImage(tinygltf::Image& image) {
//   vk::Format format;
//   Texture texture;
//
//   if (image.component == 4) {
//     format = vk::Format::eR8G8B8A8Srgb;
//   } else if (image.component == 3) {
//     format = vk::Format::eR8G8B8Srgb;
//   } else if (image.component == 1) {
//     format = vk::Format::eR8Unorm;
//   }
//
//   auto mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(image.width, image.height)))) + 1;
//
//   createImage(
//     device,
//     image.width,
//     image.height,
//     mipLevels,
//     vk::SampleCountFlagBits::e1,
//     format,
//     vk::ImageTiling::eOptimal,
//     vk::ImageUsageFlagBits::eTransferSrc
//       | vk::ImageUsageFlagBits::eTransferDst
//       | vk::ImageUsageFlagBits::eSampled,
//     vk::MemoryPropertyFlagBits::eDeviceLocal,
//     texture.image,
//     texture.imageMemory
//   );
//
//   transitionImageLayout(
//     texture.image,
//     vk::ImageLayout::eUndefined,
//     vk::ImageLayout::eTransferDstOptimal,
//     mipLevels
//   );
//
//   copyBufferToImage(
//     stagingBuffer,
//     texture.image,
//     static_cast<uint32_t>(image.width),
//     static_cast<uint32_t>(image.width)
//   );
//
//   generateMipmaps(
//     texture.image,
//     format,
//     image.width,
//     image.width,
//     mipLevels
//   );
//
//   texture.imageView = createImageView(
//     texture.image,
//     format,
//     vk::ImageAspectFlagBits::eColor,
//     mipLevels
//   );
//
//   vk::SamplerCreateInfo samplerInfo {
//     .magFilter = vk::Filter::eLinear,
//     .minFilter = vk::Filter::eLinear,
//     .mipmapMode = vk::SamplerMipmapMode::eLinear,
//     .addressModeU = vk::SamplerAddressMode::eRepeat,
//     .addressModeV = vk::SamplerAddressMode::eRepeat,
//     .addressModeW = vk::SamplerAddressMode::eRepeat,
//     .mipLodBias = 0.0f,
//     .anisotropyEnable = vk::True,
//     .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
//     .compareEnable = vk::False,
//     .compareOp = vk::CompareOp::eAlways,
//   };
//
//   texture.sampler = vk::raii::Sampler(m_device, samplerInfo);
//
//   return texture;
// }
//
// std::vector<Material> parseMaterials(tinygltf::Model& model) {
//   std::vector<Material> materials(model.materials.size());
//
//   for (std::size_t i = 0; i < model.materials.size(); i++) {
//     const auto &gltfMat = model.materials[i];
//     Material &mat = materials[i];
//
//     const auto &pbr = gltfMat.pbrMetallicRoughness;
//     mat.baseColorFactor = glm::vec4(
//         pbr.baseColorFactor[0],
//         pbr.baseColorFactor[1],
//         pbr.baseColorFactor[2],
//         pbr.baseColorFactor[3]
//     );
//     mat.metallicFactor = pbr.metallicFactor;
//     mat.roughnessFactor = pbr.roughnessFactor;
//     mat.emissiveFactor = glm::vec3(
//         gltfMat.emissiveFactor[0],
//         gltfMat.emissiveFactor[1],
//         gltfMat.emissiveFactor[2]
//     );
//
//     if (pbr.baseColorTexture.index >= 0) {
//       mat.baseColorTex = createTextureFromGLBImage(model.images[pbr.baseColorTexture.index]);
//     }
//
//     if (pbr.metallicRoughnessTexture.index >= 0) {
//       mat.metallicRoughnessTex = createTextureFromGLBImage(model.images[pbr.metallicRoughnessTexture.index]);
//     }
//
//     if (gltfMat.normalTexture.index >= 0) {
//       mat.normalTex = createTextureFromGLBImage(model.images[gltfMat.normalTexture.index]);
//     }
//
//     if (gltfMat.emissiveTexture.index >= 0) {
//       mat.emissiveTex = createTextureFromGLBImage(model.images[gltfMat.emissiveTexture.index]);
//     }
//
//     if (gltfMat.occlusionTexture.index >= 0) {
//       mat.occlusionTex = createTextureFromGLBImage(model.images[gltfMat.occlusionTexture.index]);
//     }
//   }
//
//   return materials;
// }
//
// int main() {
//   tinygltf::TinyGLTF loader;
//   tinygltf::Model scene;
//   std::string err, warn;
//
//   loader.LoadBinaryFromFile(&scene, &err, &warn, "assets/mustang.glb");
//
//   if (!warn.empty()) std::cout << "Warning: " << warn << std::endl;
//   if (!err.empty())  std::cerr << "Error: " << err << std::endl;
//
//
//   auto materials = parseMaterials(scene);
//
//   std::cout << materials.size() << std::endl;
//
//   return 0;
// }
