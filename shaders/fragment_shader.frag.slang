import "common/types";
import "common/parameters";
import "shading/pbr";


[shader("fragment")]
float4 main(
    VsOutput IN,
    ParameterBlock<SceneData> g_sceneData,
    ParameterBlock<MaterialData> g_materialData,
    ParameterBlock<LightData> g_lightData,
) : SV_Target {
    // Use the instance index passed from vertex shader
    if (IN.instanceIndex == g_sceneData.scene.skySphereInstanceIndex) {
        return float4(g_materialData.skyboxTexture.Sample(IN.inTexCoord).rgb, 1.0);
    }

    Instance instance = g_sceneData.instances[IN.instanceIndex];
    Mesh mesh = g_sceneData.meshes[instance.meshIndex];
    int uniformMaterialIndex = NonUniformResourceIndex(mesh.materialIndex);
    Material material = g_materialData.materials[uniformMaterialIndex];

    // OPTIMIZATION: Early alpha test before expensive calculations
    // For MASK mode, sample base color alpha early and discard if needed
    if (material.alphaMode == ALPHA_MODE_MASK) {
        float earlyAlpha = 1.0;
        if (material.baseColorTexIndex >= 0) {
            earlyAlpha = g_materialData.baseColorTextures[NonUniformResourceIndex(material.baseColorTexIndex)].Sample(IN.inTexCoord).a;
        } else {
            earlyAlpha = material.baseColorFactor.a;
        }
        
        if (earlyAlpha < 0.5) {
            discard;
        }
    }

    float3 V = normalize(g_sceneData.scene.cameraPos - IN.inWorldPos);
    float3 N = normalize(IN.inWorldNormal);
    float3 T = normalize(IN.inWorldTangent);

    ResolvedSurfaceParameters surfaceParams = resolveSurfaceParameters(
        material,
        g_materialData,
        IN.inTexCoord,
        N,
        T,
        IN.handedness,
    );

    // Check if material or instance should receive lighting
    bool shouldReceiveLighting = (material.receivesLighting != 0) && (instance.receivesLighting != 0);

    if (!shouldReceiveLighting) {
        return float4(surfaceParams.albedo, surfaceParams.alpha);
    }

    // Check if reflections are enabled for this material and instance
    bool enableReflections = (material.reflective != 0) || (instance.reflective != 0);

    float3 color = shadeFragment(
        surfaceParams,
        g_sceneData,
        g_materialData,
        g_lightData,
        IN.inWorldPos,
        V,
        N,
        enableReflections
    );

    // Alpha mode handled by render passes (opaque vs transparent)
    return float4(color, surfaceParams.alpha);
}
