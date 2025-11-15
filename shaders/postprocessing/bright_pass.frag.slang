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

    if (brightness > bloomParams.threshold) {
        return float4(color, 1.0);
    }
    
    return float4(0.0, 0.0, 0.0, 1.0);
}
