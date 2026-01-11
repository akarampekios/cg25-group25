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
    float weight[7] = { 
        0.1964825501511404,
        0.2969069646728344,
        0.0944703978504317,
        0.0103813596594438,
        0.0009570318253683,
        0.0000744786707024,
        0.0000048849321570
    };
    
    float2 texelSize = (1.0 / bloomParams.textureSize) * bloomParams.blurStrength;
    float3 result = buffers.inputTexture.Sample(vsOutput.texCoord).rgb * weight[0];

    // Sample in both directions with 6-pixel radius (13-tap total)
    for (int i = 1; i < 7; i++) {
        float2 offset = bloomParams.direction * texelSize * float(i);
        result += buffers.inputTexture.Sample(vsOutput.texCoord + offset).rgb * weight[i];
        result += buffers.inputTexture.Sample(vsOutput.texCoord - offset).rgb * weight[i];
    }

    return float4(result, 1.0);
}
