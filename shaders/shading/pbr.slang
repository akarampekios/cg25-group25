import brdf;

import "../common/constants";
import "../common/parameters";
import "../common/types";

float calculateFogFactor(float3 worldPos, SceneData sceneData) {
    float fogDistance = distance(worldPos, sceneData.scene.cameraPos);
    float fogAmount = 1.0 - exp(-pow(fogDistance * sceneData.scene.fogDensity, 2.0));
    return saturate(fogAmount);
}

public float3 shadeFragment(
    ResolvedSurfaceParameters surfaceParams,
    SceneData sceneData,
    MaterialData materialData,
    LightData lightData,
    float3 worldPos,
    float3 V,
    float3 shadingNormal,
    float3 geometricNormal,
    bool enableReflections = true
) {
    // Use shading normal (normal-mapped) for direct lighting
    // Use geometric normal for reflection direction to avoid artifacts
    float3 N = shadingNormal;
    float3 directionalLight = computeDirectionalLightRadiance(
        sceneData.scene.directionalLight,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 pointLights = computePointLightRadiance(
        lightData.pointLights,
        sceneData.scene.pointLightsCount,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 spotLights = computeSpotLightRadiance(
        lightData.spotLights,
        sceneData.scene.spotLightsCount,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 indirectLighting = computeIndirectLighting(
        sceneData,
        materialData,
        lightData,
        surfaceParams,
        worldPos,
        V,
        geometricNormal,
        enableReflections
    );

    float3 lighting = directionalLight + pointLights + spotLights;

    float3 finalColor = (lighting * surfaceParams.occlusion) + indirectLighting + surfaceParams.emissive;

    // Fog - reduce impact on emissive materials
    float fogAmount = calculateFogFactor(worldPos, sceneData);
    
    // Calculate emissive strength to reduce fog impact on glowing objects
    float emissiveStrength = saturate(length(surfaceParams.emissive) / 1.5);
    // Reduce fog by up to 50% for highly emissive materials
    float adjustedFogAmount = fogAmount * (1.0 - emissiveStrength * 0.5);

    return lerp(finalColor, sceneData.scene.fogColor, adjustedFogAmount);
}

float3 shadeHitPoint(
    SceneData sceneData,
    MaterialData materialData,
    LightData lightData,
    ResolvedSurfaceParameters surfaceParams,
    float3 worldPos,
    float3 V,
    float3 N,
    bool computeDirectLighting = false
) {
    // For reflections, we only want the material's base appearance to avoid double-counting direct lighting
    // NOTE: with lights on, reflections do not look much better, so we skip it entirely
    if (!computeDirectLighting) {
        // Return albedo with a subtle ambient factor plus clamped emissive
        // Clamp emissive to prevent fireflies from very bright emissive surfaces
        float3 clampedEmissive = min(surfaceParams.emissive, float3(2.0, 2.0, 2.0));
        return surfaceParams.albedo * 0.5 + clampedEmissive;
    }

    // Full lighting path (used for primary rays, not reflections)
    float3 directionalLight = computeDirectionalLightRadiance(
        sceneData.scene.directionalLight,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 pointLights = computePointLightRadiance(
        lightData.pointLights,
        sceneData.scene.pointLightsCount,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 spotLights = computeSpotLightRadiance(
        lightData.spotLights,
        sceneData.scene.spotLightsCount,
        surfaceParams,
        sceneData.tlas,
        worldPos,
        N,
        V,
        sceneData,
        materialData,
        true
    );

    float3 lighting = directionalLight + pointLights + spotLights;

    return (lighting * surfaceParams.occlusion) + surfaceParams.emissive;
}

float3 shadeInstance(
    SceneData sceneData,
    MaterialData materialData,
    LightData lightData,
    float3 worldPos,
    float3 V,
    uint instanceId,
    uint primitiveIndex,
    float2 barycentrics,
    bool computeDirectLighting = false
) {
    Instance instance = sceneData.instances[NonUniformResourceIndex(instanceId)];
    Mesh mesh = sceneData.meshes[NonUniformResourceIndex(instance.meshIndex)];
    Material material = materialData.materials[NonUniformResourceIndex(mesh.materialIndex)];

    uint i0 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + primitiveIndex * 3 + 0)];
    uint i1 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + primitiveIndex * 3 + 1)];
    uint i2 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + primitiveIndex * 3 + 2)];

    Vertex v0 = sceneData.vertices[NonUniformResourceIndex(i0)];
    Vertex v1 = sceneData.vertices[NonUniformResourceIndex(i1)];
    Vertex v2 = sceneData.vertices[NonUniformResourceIndex(i2)];

    float2 uv0 = v0.texCoord;
    float2 uv1 = v1.texCoord;
    float2 uv2 = v2.texCoord;

    float3 n0 = v0.normal;
    float3 n1 = v1.normal;
    float3 n2 = v2.normal;

    float4 t0 = v0.tangent;
    float4 t1 = v1.tangent;
    float4 t2 = v2.tangent;

    float w0 = 1.0 - barycentrics.x - barycentrics.y;
    float w1 = barycentrics.x;
    float w2 = barycentrics.y;

    float3 N = normalize((w0 * n0) + (w1 * n1) + (w2 * n2));
    float4 T_interp = (w0 * t0) + (w1 * t1) + (w2 * t2);
    float3 T = normalize(T_interp.xyz);
    float handedness = T_interp.w;

    float2 texCoord = w0 * uv0 + w1 * uv1 + w2 * uv2;

    ResolvedSurfaceParameters surfaceParams = resolveSurfaceParameters(
        material,
        materialData,
        texCoord,
        N,
        T,
        handedness
    );

    return shadeHitPoint(
        sceneData,
        materialData,
        lightData,
        surfaceParams,
        worldPos,
        V,
        N,
        computeDirectLighting
    );
}

public ResolvedSurfaceParameters resolveSurfaceParameters(
    Material material,
    MaterialData materialData,
    float2 uv,
    float3 N,
    float3 T,
    float handedness
) {
    ResolvedSurfaceParameters resolved;

    resolved.occlusion = 1.0;
    resolved.baseColor = material.baseColorFactor;
    resolved.albedo = resolved.baseColor.rgb;
    resolved.metallic = material.metallicFactor;
    // Clamp roughness to a minimum value to avoid numerical issues
    // Very smooth surfaces (roughness near 0) need a small epsilon to prevent
    // division by zero and delta function behavior in the BRDF
    resolved.roughness = max(material.roughnessFactor, 0.04);
    resolved.emissive = material.emissiveFactor;
    resolved.normal = N;
    resolved.alphaMode = material.alphaMode;

    resolved.alpha = material.baseColorFactor.a;
    if (material.baseColorTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.baseColorTexIndex);
        resolved.alpha *= materialData.baseColorTextures[uniformIndex].Sample(uv).a;
    }

    if (material.baseColorTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.baseColorTexIndex);
        resolved.baseColor *= materialData.baseColorTextures[uniformIndex].Sample(uv);
        resolved.albedo = resolved.baseColor.rgb;
    }

    if (material.metallicRoughnessTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.metallicRoughnessTexIndex);
        float4 mrSample = materialData.metallicRoughnessTextures[uniformIndex].Sample(uv);
        resolved.metallic = mrSample.b;
        // Clamp texture roughness to minimum value to prevent specular aliasing/fireflies
        // Very low roughness (near 0) causes GGX distribution to spike, creating bright patches
        resolved.roughness = max(mrSample.g, 0.04);
    }

    if (material.occlusionTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.occlusionTexIndex);
        resolved.occlusion = materialData.occlusionTextures[uniformIndex].Sample(uv).r;
    }

    if (material.emissiveTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.emissiveTexIndex);
        resolved.emissive *= materialData.emissiveTextures[uniformIndex].Sample(uv).rgb;
    }

    if (material.normalTexIndex >= 0) {
        int uniformIndex = NonUniformResourceIndex(material.normalTexIndex);
        float3 tangentNormal = materialData.normalTextures[uniformIndex].Sample(uv).xyz * 2.0 - 1.0;

        float3 orthoT = normalize(T - dot(T, N) * N);
        float3 B = cross(N, orthoT) * handedness;
        
        resolved.normal = normalize(tangentNormal.x * orthoT + tangentNormal.y * B + tangentNormal.z * N);
    }

    return resolved;
}

float3 computeDirectionalLightRadiance(
    DirectionalLight light,
    ResolvedSurfaceParameters surface,
    RaytracingAccelerationStructure tlas,
    float3 fragmentWorldPosition,
    float3 fragmentNormal,
    float3 viewDirection,
    SceneData sceneData,
    MaterialData materialData,
    bool traceRays = true
) {
    float3 F0 = computeReflectance(surface);

    float3 V = viewDirection;
    float3 N = fragmentNormal;
    float3 L = normalize(light.direction);
    float3 H = normalize(viewDirection + L);
    float NdotL = max(dot(fragmentNormal, L), 0.0);

    float3 radiance = light.color * light.intensity * NdotL;

    float D = D_GGX(N, H, surface.roughness);
    float G = G_Smith(N, V, L, surface.roughness);
    float3 F = F_Schlick(max(dot(H, V), 0.0), F0);

    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - surface.metallic;

    float3 numerator = D * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    float3 specular = numerator / denominator;
    
    // Clamp specular to prevent fireflies at grazing angles
    // Even with roughness clamped, D*G/denom can spike when NdotV or NdotL are near 0
    specular = min(specular, float3(10.0, 10.0, 10.0));

    float3 result = (kD * surface.albedo / PI + specular) * radiance;

    if (traceRays) {
        float shadow = calculateShadow(tlas, fragmentWorldPosition, L, 1000.0, sceneData, materialData);
        result *= shadow;
    }

    return result;
}

float3 computePointLightRadiance(
    StructuredBuffer<PointLight> lights,
    uint count,
    ResolvedSurfaceParameters surface,
    RaytracingAccelerationStructure tlas,
    float3 fragmentWorldPosition,
    float3 fragmentNormal,
    float3 viewDirection,
    SceneData sceneData,
    MaterialData materialData,
    bool traceRays = true
) {
    float3 F0 = computeReflectance(surface);
    float3 Lo = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < count; ++i) {
        uint lightUniformIndex = NonUniformResourceIndex(i);
        PointLight light = lights[lightUniformIndex];

        float3 V = viewDirection;
        float3 N = fragmentNormal;
        float3 lightToFragment = light.position - fragmentWorldPosition;
        float distance = length(lightToFragment);
        float3 L = normalize(lightToFragment);
        float3 H = normalize(V + L);

        // OPTIMIZATION: Early out if light is too far or has no radius
        // Only consider lights within their effective radius
        if (light.radius > 0.0 && distance > light.radius * 1.1) {
            continue;
        }

        // Distance-based attenuation with optional radius cutoff
        float attenuation = 1.0 / (distance * distance);
        
        // If light has a radius, apply smooth falloff near the edge
        if (light.radius > 0.0) {
            float falloff = saturate(1.0 - (distance / light.radius));
            falloff = falloff * falloff; // Smooth squared falloff
            attenuation *= falloff;
        }

        // OPTIMIZATION: Skip this light if contribution is negligible
        if (attenuation < 0.001) {
            continue;
        }

        float3 radiance = light.color * light.intensity * attenuation;

        float NdotL = max(dot(N, L), 0.0);
        radiance *= NdotL;

        float D = D_GGX(N, H, surface.roughness);
        float G = G_Smith(N, V, L, surface.roughness);
        float3 F = F_Schlick(max(dot(H, V), 0.0), F0);

        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - surface.metallic;

        float3 numerator = D * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
        float3 specular = numerator / denominator;
        
        // Clamp specular to prevent fireflies at grazing angles
        specular = min(specular, float3(0.1, 0.1, 0.1));

        float3 result = (kD * surface.albedo / PI + specular) * radiance;

        // OPTIMIZATION: Only trace shadow rays if the light contribution is significant
        // AND the light casts shadows AND we're tracing rays
        if (traceRays && light.castsShadows != 0 && attenuation > 0.001) {
            float shadow = calculateShadow(tlas, fragmentWorldPosition, L, distance, sceneData, materialData);
            result *= shadow;
        }

        Lo += result;
    }

    return Lo;
}

float3 computeSpotLightRadiance(
    StructuredBuffer<SpotLight> lights,
    uint count,
    ResolvedSurfaceParameters surface,
    RaytracingAccelerationStructure tlas,
    float3 fragmentWorldPosition,
    float3 fragmentNormal,
    float3 viewDirection,
    SceneData sceneData,
    MaterialData materialData,
    bool traceRays = true
) {
    float3 F0 = computeReflectance(surface);
    float3 Lo = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < count; ++i) {
        uint lightUniformIndex = NonUniformResourceIndex(i);
        SpotLight light = lights[lightUniformIndex];

        float3 V = viewDirection;
        float3 N = fragmentNormal;
        
        // Calculate direction and distance BEFORE normalization
        float3 lightToFragment = light.position - fragmentWorldPosition;
        float distance = length(lightToFragment);
        float3 L = normalize(lightToFragment);
        float3 H = normalize(V + L);

        // Spot light cone calculation - EARLY OUT
        float3 spotDirection = normalize(light.direction);
        float theta = dot(L, spotDirection);
        float outerCutoffCos = cos(light.outerCutoff);
        
        // OPTIMIZATION: Early out if completely outside the cone
        if (theta < outerCutoffCos) {
            continue;
        }

        // Distance attenuation (inverse square law)
        float attenuation = 1.0 / (distance * distance + 0.01);
        
        // Convert angles to cosine
        float innerCutoffCos = cos(light.cutoff);
        
        // Calculate NdotL (surface facing the light)
        float NdotL = max(dot(N, L), 0.0);
        
        // OPTIMIZATION: Early out if surface facing away from light
        if (NdotL <= 0.0) {
            continue;
        }
        
        // Smooth transition from inner to outer cone
        float epsilon = abs(innerCutoffCos - outerCutoffCos) + 0.0001;
        float spotIntensity = saturate((theta - outerCutoffCos) / epsilon);
        
        // Combine attenuation with spot cone falloff
        float3 radiance = light.color * light.intensity * attenuation * spotIntensity;

        // Lambert term (surface facing the light)
        radiance *= NdotL;
        
        // OPTIMIZATION: Skip if contribution is negligible
        float radianceIntensity = dot(radiance, float3(0.299, 0.587, 0.114)); // luminance
        if (radianceIntensity < 0.001) {
            continue;
        }

        // PBR calculations
        float D = D_GGX(N, H, surface.roughness);
        float G = G_Smith(N, V, L, surface.roughness);
        float3 F = F_Schlick(max(dot(H, V), 0.0), F0);

        float3 kS = F;
        float3 kD = float3(1.0, 1.0, 1.0) - kS;
        kD *= 1.0 - surface.metallic;

        float3 numerator = D * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.001;
        float3 specular = numerator / denominator;
        
        // Clamp specular to prevent fireflies at grazing angles
        specular = min(specular, float3(10.0, 10.0, 10.0));

        float3 result = (kD * surface.albedo / PI + specular) * radiance;

        // OPTIMIZATION: Only trace shadows for significant contributions
        if (traceRays && light.castsShadows != 0 && radianceIntensity > 0.01) {
            // Spot lights need a larger bias than other light types to avoid self-intersection
            // This is likely due to the grazing angle when light points straight down
            float3 shadowOrigin = fragmentWorldPosition + fragmentNormal * 0.01;
            float shadowMaxDist = max(distance - 0.01, 0.002);
            float shadow = calculateShadow(tlas, shadowOrigin, L, shadowMaxDist, sceneData, materialData);
            result *= shadow;
        }

        Lo += result;
    }

    return Lo;
}

float3 computeIndirectLighting(
    SceneData sceneData,
    MaterialData materialData,
    LightData lightData,
    ResolvedSurfaceParameters surfaceParams,
    float3 worldPos,
    float3 V,
    float3 N,
    bool enableReflections = true
) {
    float3 R = reflect(-V, N);
    // Only perturb reflection direction slightly for glossy reflections
    // This keeps reflections sharp enough to show texture detail
    R = normalize(lerp(R, N, surfaceParams.roughness * surfaceParams.roughness * 0.5));

    // Sample skybox for irradiance (this is used in both reflection and non-reflection cases)
    float3 irradiance = sampleEquirectangularTexture(materialData.skyboxTexture, R);

    float3 F0 = computeReflectance(surfaceParams);
    float3 F = F_Schlick(max(dot(N, V), 0.0), F0);

    float3 kS = F;
    float3 kD = float3(1.0, 1.0, 1.0) - kS;
    kD *= 1.0 - surfaceParams.metallic;

    // For diffuse indirect lighting, sample the skybox based on the normal direction (not reflection)
    float3 diffuseIrradiance = sampleEquirectangularTexture(materialData.skyboxTexture, N);

    // OPTIMIZATION: More aggressive reflection culling
    // Only trace reflection rays for very smooth and/or highly metallic surfaces
    // roughness < 0.3 AND (metallic > 0.5 OR roughness < 0.15)
    bool shouldTraceReflection = enableReflections && 
                                 (surfaceParams.roughness < 0.3) &&
                                 ((surfaceParams.metallic > 0.5) || (surfaceParams.roughness < 0.15));
        
    // Only add specular reflections for materials with low roughness or high metallic values
    // Use a much more aggressive cutoff to prevent rough surfaces from being reflective
    float smoothness = 1.0 - surfaceParams.roughness;

    // Cubic falloff makes rough surfaces essentially non-reflective
    float reflectionMask = smoothness * smoothness * smoothness * smoothness;
    float3 specularWeight = kS * reflectionMask;
    
    // For specular indirect, use the traced reflection result
    // Weight by Fresnel and metallic properties
    float3 specularContribution = irradiance * specularWeight;

    if (shouldTraceReflection) {
        RayDesc reflectionRayDesc;
        reflectionRayDesc.Origin = worldPos;
        reflectionRayDesc.Direction = R;
        reflectionRayDesc.TMin = 0.01;

        // OPTIMIZATION: Significantly reduced max reflection distance
        // Only trace nearby reflections - distant ones use skybox
        reflectionRayDesc.TMax = 50.0;  // Reduced from 1e4 to 50 units

        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
        let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

        query.TraceRayInline(sceneData.tlas, rayFlags, AS_LIT_OBJECT_MASK, reflectionRayDesc);

        while (query.Proceed()) {
            if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
                continue;
            }

            // Get hit information
            uint hitInstanceId = query.CandidateRayInstanceCustomIndex();
            uint hitPrimitiveIndex = query.CandidatePrimitiveIndex();
            float2 hitBarycentrics = query.CandidateTriangleBarycentrics();

            if (hitPrimitiveIndex == uint(-1)) {
                continue;
            }

            // Get instance and mesh data
            Instance instance = sceneData.instances[NonUniformResourceIndex(hitInstanceId)];
            Mesh mesh = sceneData.meshes[NonUniformResourceIndex(instance.meshIndex)];
            Material material = materialData.materials[NonUniformResourceIndex(mesh.materialIndex)];

            // If material is opaque, commit the hit
            if (material.alphaMode == ALPHA_MODE_OPAQUE) {
                query.CommitNonOpaqueTriangleHit();
                break;
            }

            // For alpha-tested or blended materials, check the actual alpha value
            if (material.alphaMode == ALPHA_MODE_MASK || material.alphaMode == ALPHA_MODE_BLEND) {
                // Get triangle indices
                uint i0 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 0)];
                uint i1 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 1)];
                uint i2 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 2)];

                // Get vertex data
                Vertex v0 = sceneData.vertices[NonUniformResourceIndex(i0)];
                Vertex v1 = sceneData.vertices[NonUniformResourceIndex(i1)];
                Vertex v2 = sceneData.vertices[NonUniformResourceIndex(i2)];

                // Interpolate UV coordinates
                float w0 = 1.0 - hitBarycentrics.x - hitBarycentrics.y;
                float w1 = hitBarycentrics.x;
                float w2 = hitBarycentrics.y;
                float2 uv = (w0 * v0.texCoord) + (w1 * v1.texCoord) + (w2 * v2.texCoord);

                // Sample alpha from base color texture
                float alpha = material.baseColorFactor.a;
                if (material.baseColorTexIndex >= 0) {
                    alpha *= materialData.baseColorTextures[NonUniformResourceIndex(material.baseColorTexIndex)].Sample(uv).a;
                }

                // Apply alpha test
                if (material.alphaMode == ALPHA_MODE_MASK) {
                    if (alpha >= 0.5) {
                        query.CommitNonOpaqueTriangleHit();
                        break;
                    }
                    // If alpha < 0.5, continue tracing (transparent)
                } else { // ALPHA_MODE_BLEND
                    if (alpha >= 0.5) {
                        query.CommitNonOpaqueTriangleHit();
                        break;
                    }
                }
            }
        }

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            uint hitInstanceId = query.CommittedInstanceID();
            uint hitPrimitiveIndex = query.CommittedPrimitiveIndex();
            float2 hitBarycentrics = query.CommittedTriangleBarycentrics();
            float3 hitWorldPos = worldPos + query.CommittedRayT() * R;
            float3 hitReflectionView = -R;

            irradiance = shadeInstance(
                sceneData,
                materialData,
                lightData,
                hitWorldPos,
                hitReflectionView,
                hitInstanceId,
                hitPrimitiveIndex,
                hitBarycentrics,
                // Don't compute direct lighting for reflections
                // it is cheaper and does not add much visual quality
                false 
            );

            // Clamp reflection irradiance to prevent fireflies from bright surfaces
            // This limits the maximum brightness a single reflection sample can contribute
            irradiance = min(irradiance, float3(3.0, 3.0, 3.0));

            specularContribution = irradiance * specularWeight;
        }
    }

    // Apply different scaling based on surface properties
    // Diffuse indirect is always subtle ambient (3% of skybox)
    float diffuseAmbient = 0.03;
    float3 diffuse = kD * diffuseIrradiance * surfaceParams.albedo * diffuseAmbient;
    
    // Specular indirect scaling with more aggressive roughness response
    // For rough surfaces (roughness > 0.5), reflections should be nearly invisible
    // Scale from 0.01 (rough) to 0.25 (mirror-smooth)
    float specularScale = lerp(0.04, 1.0, surfaceParams.metallic) * (1.0 - surfaceParams.roughness * 0.7);
    float3 specular = specularContribution * specularScale;

    return diffuse + specular;
}

float3 computeReflectance(ResolvedSurfaceParameters surfaceParams) {
    float3 F0 = float3(0.04, 0.04, 0.04);
    return lerp(F0, surfaceParams.albedo, surfaceParams.metallic);
}

float3 sampleEquirectangularTexture(Sampler2D texture, float3 direction) {
    // Standard equirectangular mapping
    // Horizontal: atan2(z, x) maps to [0, 1]
    // Vertical: asin(y) maps to [-π/2, π/2], normalized to [0, 1]
    // Note: negating asin(y) flips the vertical
    float2 uv = float2(
        atan2(direction.z, direction.x) / (2.0 * PI) + 0.5,
        -asin(direction.y) / PI + 0.5
    );
    return texture.Sample(uv).rgb;
}

float calculateShadow(
    RaytracingAccelerationStructure tlas, 
    float3 origin, 
    float3 direction, 
    float maxDist,
    SceneData sceneData,
    MaterialData materialData
) {
    // Skip shadow rays for very distant surfaces
    float distanceFromCamera = distance(origin, sceneData.scene.cameraPos);
    if (distanceFromCamera > 100.0) {
        return 1.0;
    }
    
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

    RayDesc shadowRayDesc;
    shadowRayDesc.Origin = origin;
    shadowRayDesc.Direction = direction;
    shadowRayDesc.TMin = 0.001;
    shadowRayDesc.TMax = max(maxDist, shadowRayDesc.TMin + 0.001);

    query.TraceRayInline(tlas, rayFlags, AS_SHADOW_OBJECT_MASK, shadowRayDesc);

    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            continue;
        }

        // Get hit information
        uint hitInstanceId = query.CandidateRayInstanceCustomIndex();
        uint hitPrimitiveIndex = query.CandidatePrimitiveIndex();
        float2 hitBarycentrics = query.CandidateTriangleBarycentrics();

        if (hitPrimitiveIndex == uint(-1)) {
            continue;
        }

        // Get instance and mesh data
        Instance instance = sceneData.instances[NonUniformResourceIndex(hitInstanceId)];
        Mesh mesh = sceneData.meshes[NonUniformResourceIndex(instance.meshIndex)];
        Material material = materialData.materials[NonUniformResourceIndex(mesh.materialIndex)];

        // If material is opaque, commit the hit
        if (material.alphaMode == ALPHA_MODE_OPAQUE) {
            query.CommitNonOpaqueTriangleHit();
            break;
        }

        // For alpha-tested or blended materials, check the actual alpha value
        if (material.alphaMode == ALPHA_MODE_MASK || material.alphaMode == ALPHA_MODE_BLEND) {
            // Get triangle indices
            uint i0 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 0)];
            uint i1 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 1)];
            uint i2 = sceneData.indices[NonUniformResourceIndex(mesh.baseIndex + hitPrimitiveIndex * 3 + 2)];

            // Get vertex data
            Vertex v0 = sceneData.vertices[NonUniformResourceIndex(i0)];
            Vertex v1 = sceneData.vertices[NonUniformResourceIndex(i1)];
            Vertex v2 = sceneData.vertices[NonUniformResourceIndex(i2)];

            // Interpolate UV coordinates
            float w0 = 1.0 - hitBarycentrics.x - hitBarycentrics.y;
            float w1 = hitBarycentrics.x;
            float w2 = hitBarycentrics.y;
            float2 uv = (w0 * v0.texCoord) + (w1 * v1.texCoord) + (w2 * v2.texCoord);

            // Sample alpha from base color texture
            float alpha = material.baseColorFactor.a;
            if (material.baseColorTexIndex >= 0) {
                alpha *= materialData.baseColorTextures[NonUniformResourceIndex(material.baseColorTexIndex)].Sample(uv).a;
            }

            // Apply alpha test
            if (material.alphaMode == ALPHA_MODE_MASK) {
                if (alpha >= 0.5) {
                    query.CommitNonOpaqueTriangleHit();
                    break;
                }
                // If alpha < 0.5, continue tracing (transparent, no shadow)
            } else {
                // For blended materials, we could weight the shadow by alpha
                // For simplicity, treat as transparent if alpha < 0.5
                if (alpha >= 0.5) {
                    query.CommitNonOpaqueTriangleHit();
                    break;
                }
            }
        }
    }

    // if we hit geometry - we are in a shadow
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        return 0.0;
    }

    return 1.0;
}
