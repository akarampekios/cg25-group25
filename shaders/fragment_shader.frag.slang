import "common/types";
import "common/parameters";
import "shading/pbr";

// TAA: Fragment shader output structure for MRT (Multiple Render Targets)
struct FragmentOutput {
    float4 color    : SV_Target0;  // Main color output
    float2 velocity : SV_Target1;  // TAA: Screen-space velocity (motion vector)
};

// TAA: Calculate screen-space velocity from clip positions
float2 calculateVelocity(float4 currClipPos, float4 prevClipPos) {
    // Perspective divide to get NDC coordinates
    float2 currNDC = currClipPos.xy / currClipPos.w;
    float2 prevNDC = prevClipPos.xy / prevClipPos.w;
    
    // Convert from NDC [-1, 1] to UV [0, 1] space
    float2 currUV = currNDC * 0.5 + 0.5;
    float2 prevUV = prevNDC * 0.5 + 0.5;
    
    // Velocity = where we are now - where we were
    // This gives us the motion vector to look up in history
    return currUV - prevUV;
}

[shader("fragment")]
FragmentOutput main(
    VsOutput IN,
    ParameterBlock<SceneData> g_sceneData,
    ParameterBlock<MaterialData> g_materialData,
    ParameterBlock<LightData> g_lightData,
) {
    FragmentOutput output;
    
    // TAA: Calculate velocity for this fragment
    output.velocity = calculateVelocity(IN.currClipPos, IN.prevClipPos);
    
    // Use the instance index passed from vertex shader
    if (IN.instanceIndex == g_sceneData.scene.skySphereInstanceIndex) {
        output.color = float4(g_materialData.skyboxTexture.Sample(IN.inTexCoord).rgb, 1.0);
        output.velocity = float2(0.0, 0.0);  // Sky has no motion
        return output;
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
        output.color = float4(surfaceParams.albedo, surfaceParams.alpha);
        return output;
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
    output.color = float4(color, surfaceParams.alpha);
    return output;
}
