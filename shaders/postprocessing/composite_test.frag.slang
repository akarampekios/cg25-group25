struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

struct Buffers {
    Sampler2D bloomBuffer;
};

struct BloomPushConstant {
    float2 textureSize;
    float2 direction; // (1, 0) for horizontal, (0, 1) for vertical
    float blurStrength;
    float exposure;
    float threshold;
    float scale;
};


[vk::push_constant] BloomPushConstant bloomParams;

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<Buffers> buffers
) : SV_Target
{
    float3 bloomColor = buffers.bloomBuffer.Sample(vsOutput.texCoord).rgb;
    return float4(bloomColor.r, 0.0, 0.0, 1.0);
}
