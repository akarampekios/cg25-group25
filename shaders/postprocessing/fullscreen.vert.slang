struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

[shader("vertex")]
VSOutput main(uint vertexId : SV_VertexID) {
    VSOutput outVSOutput;

    outVSOutput.texCoord = float2((vertexId << 1) & 2, vertexId & 2);
    outVSOutput.position = float4(outVSOutput.texCoord * 2.0f - 1.0f, 0.0f, 1.0f);

    return outVSOutput;
}
