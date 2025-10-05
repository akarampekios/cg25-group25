#version 450

const float PI = 3.14159265359;

layout(set = 0, binding = 0) uniform ObjectUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 cameraPos;
    float time;
} ubo;

layout(set = 0, binding = 1) uniform MaterialUBO {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    vec3 emissiveFactor;
} material;

layout(set = 0, binding = 2) uniform sampler2D baseColorTex;
layout(set = 0, binding = 3) uniform sampler2D metallicRoughnessTex;
layout(set = 0, binding = 4) uniform sampler2D normalTex;
layout(set = 0, binding = 5) uniform sampler2D emissiveTex;
layout(set = 0, binding = 6) uniform sampler2D occlusionTex;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 fragWorldPos;
layout(location = 4) in vec3 fragTangent;
layout(location = 5) in vec3 fragBitangent;

layout(location = 0) out vec4 outColor; // color attachment

void main() {
    // Construct TBN matrix
    mat3 TBN = mat3(normalize(fragTangent),
                    normalize(fragBitangent),
                    normalize(fragNormal));

    // Sample normal map and transform to world space
    vec3 normalMap = texture(normalTex, fragTexCoord).rgb;
    normalMap = normalize(normalMap * 2.0 - 1.0); // [0,1] -> [-1,1]
    vec3 N = normalize(TBN * normalMap);

    // View and light vectors
    vec3 V = normalize(ubo.cameraPos - fragWorldPos);
    vec3 L = normalize(vec3(1.0, 1.0, 1.0)); // Replace with actual light
    vec3 H = normalize(V + L);

    // Metallic/Roughness
    vec2 mrSample = texture(metallicRoughnessTex, fragTexCoord).rg;
    float metallic = mrSample.x * material.metallicFactor;
    float roughness = mrSample.y * material.roughnessFactor;

    // Base color
    vec3 albedo = (texture(baseColorTex, fragTexCoord) * material.baseColorFactor).rgb;

    // Fresnel (Schlick)
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);

    // GGX specular
    float alpha = roughness * roughness;
    float NdotH = max(dot(N,H), 0.0);
    float D = alpha*alpha / (PI * pow((NdotH*NdotH)*(alpha*alpha -1.0)+1.0,2.0));

    float k = (roughness+1.0)*(roughness+1.0)/8.0;
    float NdotV = max(dot(N,V),0.0);
    float NdotL = max(dot(N,L),0.0);
    float G = NdotL/(NdotL*(1.0-k)+k) * NdotV/(NdotV*(1.0-k)+k);

    vec3 spec = D * G * F / (4.0*NdotV*NdotL + 0.001);
    vec3 kD = (1.0 - F)*(1.0-metallic);

    // Combine
    vec3 ambient = 0.03 * albedo;
    vec3 color = ambient + (kD*albedo/PI + spec) * NdotL;

    // Emissive
    color += texture(emissiveTex, fragTexCoord).rgb * material.emissiveFactor;

    outColor = vec4(color, 1.0);
}
