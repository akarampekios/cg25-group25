import bloom_params;

struct Buffers {
    Sampler2D hdrBuffer;
};

[vk::push_constant] BloomPushConstant bloomParams;

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<Buffers> buffers
) : SV_Target {
    float3 color = buffers.hdrBuffer.Sample(vsOutput.texCoord).rgb;
    
    float maxComponent = max(color.r, max(color.g, color.b));
    
    float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
    
    float minComponent = min(color.r, min(color.g, color.b));
    float saturation = (maxComponent > 0.0) ? (maxComponent - minComponent) / maxComponent : 0.0;
    

    float brightness = lerp(luminance, maxComponent, 0.6) + saturation * 0.3;

    // Soft knee threshold - smooth transition instead of hard cutoff
    // This prevents blocky edges in the bloom
    float threshold = bloomParams.threshold;
    float knee = threshold * 0.5;  // Soft knee width
    float softThreshold = threshold - knee;
    
    // Smooth quadratic falloff from softThreshold to threshold
    float contribution = clamp((brightness - softThreshold) / (2.0 * knee + 0.0001), 0.0, 1.0);
    contribution *= contribution;  // Quadratic curve for smoother falloff
    
    return float4(color * contribution, 1.0);
}
