public struct VsOutput {
   public float4 position      : SV_Position;
   public float2 inTexCoord    : TEXCOORD0;
   public float3 inWorldPos    : TEXCOORD1;
   public float3 inWorldNormal : TEXCOORD2;
   public float3 inWorldTangent: TEXCOORD3;
   public float  handedness    : TEXCOORD4;
   public nointerpolation uint instanceIndex : TEXCOORD5; // Pass instance index to fragment shader
   public float4 currClipPos   : TEXCOORD6;
   public float4 prevClipPos   : TEXCOORD7;
};

public struct Vertex {
    public float3 position;
    public float3 normal;
    public float2 texCoord;
    public float4 tangent;
}

public struct Instance {
    public float4x4 transform;
    public float4x4 inverseTransform;
    public int meshIndex;
    public int reflective; // 0 - no reflections, 1 - reflective (default)
    public int castsShadows; // 0 - no shadow casting, 1 - casts shadows (default)
    public int receivesLighting; // 0 - no lighting, 1 - receives lighting (default)
    public int animated; // 0 - static (default), 1 - animated (updated by animator)
    // Shader compiler automatically pads to 16-byte alignment
};

public struct Mesh {
    public float3 boundingBoxMin;
    public float3 boundingBoxMax;
    public uint baseVertex;
    public uint baseIndex;
    public uint vertexCount;
    public uint indexCount;
    public int materialIndex;
    // Shader compiler automatically pads to 16-byte alignment for std140
};

public struct DirectionalLight {
    public float3 direction;
    public float intensity;
    public float3 color;
    // Shader compiler automatically pads to 16-byte alignment
};

public struct PointLight {
    public float3 position;
    public float intensity;
    public float3 color;
    public float radius;
    public int castsShadows; // 0 - no shadows, 1 - casts shadows (default)
    public int animated; // 0 - static (default), 1 - animated
    // Shader compiler automatically pads to 16-byte alignment
};

public struct SpotLight {
    public float3 position;
    public float intensity;
    public float3 direction;
    public float cutoff;
    public float3 color;
    public float outerCutoff;
    public int castsShadows; // 0 - no shadows, 1 - casts shadows (default)
    public int animated; // 0 - static (default), 1 - animated
    // Shader compiler automatically pads to 16-byte alignment
};

public static const int ALPHA_MODE_OPAQUE = 0;
public static const int ALPHA_MODE_BLEND = 1;
public static const int ALPHA_MODE_MASK = 2;

public struct Material {
    public float4 baseColorFactor;
    public float3 emissiveFactor;

    public float padding_emissive;

    public float metallicFactor;
    public float roughnessFactor;

    public int baseColorTexIndex;
    public int metallicRoughnessTexIndex;
    public int normalTexIndex;
    public int emissiveTexIndex;
    public int occlusionTexIndex;

    public int alphaMode; // 0 - opaque, 1 - MASK, 2 - blend
    public int reflective; // 0 - no reflections, 1 - reflective (default)
    public int castsShadows; // 0 - no shadow casting, 1 - casts shadows (default)
    public int receivesLighting; // 0 - no lighting, 1 - receives lighting (default)
};

public struct SceneDataUBO {
    public float4x4 view;
    public float4x4 proj;
    public float4x4 viewInverse;
    public float4x4 projInverse;
    public float4x4 prevView;
    public float4x4 prevProj;
    public float3 cameraPos;
    public float time;
    public uint pointLightsCount;
    public uint spotLightsCount;
    public DirectionalLight directionalLight;
    public int skySphereInstanceIndex;
    public int skySphereTextureIndex;
    public float2 jitterOffset;
    public float3 fogColor;
    public float fogDensity;
    public float2 screenSize;
    public float2 _padding3;
};

public struct ResolvedSurfaceParameters {
    public float occlusion;
    public float4 baseColor;
    public float3 albedo;
    public float roughness;
    public float metallic;
    public float3 emissive;
    public float3 normal;
    public float alpha;
    public int alphaMode;
};
