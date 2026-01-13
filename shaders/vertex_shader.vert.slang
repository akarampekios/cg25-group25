import "common/types";
import "common/parameters";

[shader("vertex")]
VsOutput main(
    Vertex vertex,
    uint instanceID : SV_StartInstanceLocation,  // Built-in instance index from Vulkan
    ParameterBlock<SceneData> sceneData
) {
    Instance instance = sceneData.instances[instanceID];

    float4x4 modelMatrix = instance.transform;
    float4x4 modelMatrixInverse = instance.inverseTransform;
    float3x3 normalMatrix = (float3x3)transpose(modelMatrixInverse);

    float4 worldPos = mul(modelMatrix, float4(vertex.position, 1.0));

    VsOutput result;

    result.inTexCoord = vertex.texCoord;
    result.inWorldPos = worldPos.xyz;
    result.instanceIndex = instanceID;
    
    // Current frame clip position (with jitter applied via projection matrix)
    result.position = mul(sceneData.scene.proj, mul(sceneData.scene.view, worldPos));
    result.currClipPos = result.position;
    
    // Previous frame clip position (for TAA velocity calculation)
    // Note: For static objects, we use the same world position
    // For animated objects, you would need to store previous frame transforms
    float4 prevClipPos = mul(sceneData.scene.prevProj, mul(sceneData.scene.prevView, worldPos));
    result.prevClipPos = prevClipPos;
    
    result.inWorldNormal = normalize(mul(normalMatrix, vertex.normal));
    result.inWorldTangent = normalize(mul((float3x3)modelMatrix, vertex.tangent.xyz));
    result.handedness = vertex.tangent.w;

    // Force SkySphere to the far plane
    if (instanceID == sceneData.scene.skySphereInstanceIndex) {
        result.position.z = -result.position.w;
        // Sky sphere has no motion (it follows camera)
        result.prevClipPos = result.currClipPos;
    }

    return result;
}
