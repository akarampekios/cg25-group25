struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

struct Buffers {
    Sampler2D image;
};

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<Buffers> buffers
) : SV_Target {
    return buffers.image.Sample(vsOutput.texCoord).rgba;
}
