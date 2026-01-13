import bloom_params;

struct Buffers {
    Sampler2D hdrBuffer;
    Sampler2D bloomBuffer;
};

[vk::push_constant] BloomPushConstant bloomParams;

// ACES Filmic Tonemapping
// Attempt to approximate the Academy Color Encoding System (ACES) curve
// This compresses HDR values into displayable range while preserving contrast
float3 ACESFilm(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// Alternative: Reinhard Extended tonemapping with white point
// Preserves more detail in highlights compared to basic Reinhard
float3 ReinhardExtended(float3 color, float whitePoint) {
    float3 numerator = color * (1.0 + color / (whitePoint * whitePoint));
    return numerator / (1.0 + color);
}

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<Buffers> buffers
) : SV_Target
{
    float3 hdrColor = buffers.hdrBuffer.Sample(vsOutput.texCoord).rgb;
    float3 bloomColor = buffers.bloomBuffer.Sample(vsOutput.texCoord).rgb;

    bloomColor *= bloomParams.scale;
    hdrColor += bloomColor;
    hdrColor *= bloomParams.exposure;
    
    // Apply ACES tonemapping to compress HDR range while preserving contrast
    // This ensures bright emissive surfaces don't clip to pure white,
    // maintaining texture detail on neon signs, advertisements, etc.
    float3 tonemapped = ACESFilm(hdrColor);
    
    return float4(tonemapped, 1.0);
}
