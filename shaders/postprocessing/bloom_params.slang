struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

// todo: merge with the above
struct BloomPushConstant {
    float2 textureSize;
    float2 direction; // (1, 0) for horizontal, (0, 1) for vertical
    float blurStrength;
    float exposure;
    float threshold;
    float scale;
};
