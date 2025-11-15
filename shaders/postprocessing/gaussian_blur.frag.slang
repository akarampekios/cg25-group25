import bloom_params;

struct Buffers {
    Sampler2D inputTexture;
};

[vk::push_constant] BloomPushConstant bloomParams;

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<Buffers> buffers
) : SV_Target
{
    float weight[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216 };
    
    float2 texelSize = (1.0 / bloomParams.textureSize) * bloomParams.blurStrength;
    float3 result = buffers.inputTexture.Sample(vsOutput.texCoord).rgb * weight[0];

    for (int i = 1; i < 5; i++) {
        float2 offset = bloomParams.direction * texelSize * float(i);
        result += buffers.inputTexture.Sample(vsOutput.texCoord + offset).rgb * weight[i];
        result += buffers.inputTexture.Sample(vsOutput.texCoord - offset).rgb * weight[i];
    }

    return float4(result, 1.0);
}
