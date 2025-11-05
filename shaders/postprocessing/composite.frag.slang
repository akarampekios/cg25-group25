import bloom_params;

struct Buffers {
    Sampler2D hdrBuffer;
    Sampler2D bloomBuffer;
};

[vk::push_constant] BloomPushConstant bloomParams;

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
    
    return float4(hdrColor, 1.0);
}
