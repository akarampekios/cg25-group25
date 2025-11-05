import types;
import constants;

struct SceneData {
    ConstantBuffer<SceneDataUBO> scene;
    RaytracingAccelerationStructure tlas;
    StructuredBuffer<Instance> instances;
    StructuredBuffer<Mesh> meshes;
    StructuredBuffer<float2> uvs;
    StructuredBuffer<uint> indices;
    StructuredBuffer<Vertex> vertices;
};

struct MaterialData {
    StructuredBuffer<Material> materials;
    Sampler2D baseColorTextures[MAX_TEXTURE_ARRAY_SIZE];
    Sampler2D metallicRoughnessTextures[MAX_TEXTURE_ARRAY_SIZE];
    Sampler2D normalTextures[MAX_TEXTURE_ARRAY_SIZE];
    Sampler2D emissiveTextures[MAX_TEXTURE_ARRAY_SIZE];
    Sampler2D occlusionTextures[MAX_TEXTURE_ARRAY_SIZE];
    Sampler2D skyboxTexture;
};

struct LightData {
    StructuredBuffer<PointLight> pointLights;
    StructuredBuffer<SpotLight> spotLights;
};
