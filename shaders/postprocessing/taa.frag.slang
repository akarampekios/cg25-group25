// TAA (Temporal Anti-Aliasing) Resolve Shader
// Implements temporal reprojection with variance clipping for anti-aliasing

struct VSOutput {
    float4 position : SV_Position;
    float2 texCoord : TEXCOORD0;
};

struct TAABuffers {
    Sampler2D currentColor;   // Current frame color (jittered)
    Sampler2D historyColor;   // Previous frame's TAA output
    Sampler2D velocityBuffer; // Screen-space motion vectors
};

struct TAAPushConstant {
    float2 screenSize;
    float blendFactor;      // α: typically 0.1 (10% current, 90% history)
    float _padding;
};

[vk::push_constant] TAAPushConstant taaParams;

// Convert RGB to YCoCg color space (better for clamping)
float3 RGBToYCoCg(float3 rgb) {
    float Y  = dot(rgb, float3(0.25, 0.5, 0.25));
    float Co = dot(rgb, float3(0.5, 0.0, -0.5));
    float Cg = dot(rgb, float3(-0.25, 0.5, -0.25));
    return float3(Y, Co, Cg);
}

// Convert YCoCg back to RGB
float3 YCoCgToRGB(float3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    
    float R = Y + Co - Cg;
    float G = Y + Cg;
    float B = Y - Co - Cg;
    return float3(R, G, B);
}

// Clip color towards AABB center (better than simple clamp)
float3 clipToAABB(float3 color, float3 minimum, float3 maximum) {
    float3 center = 0.5 * (maximum + minimum);
    float3 extents = 0.5 * (maximum - minimum);
    
    float3 offset = color - center;
    float3 ts = abs(extents) / max(abs(offset), float3(0.0001, 0.0001, 0.0001));
    float t = saturate(min(min(ts.x, ts.y), ts.z));
    
    return center + offset * t;
}

[shader("fragment")]
float4 main(
    VSOutput vsOutput,
    ParameterBlock<TAABuffers> buffers
) : SV_Target {
    float2 texelSize = 1.0 / taaParams.screenSize;
    float2 uv = vsOutput.texCoord;
    
    // Sample current frame color
    float3 currentColor = buffers.currentColor.Sample(uv).rgb;
    
    // Sample velocity and calculate history UV
    float2 velocity = buffers.velocityBuffer.Sample(uv).rg;
    float2 historyUV = uv - velocity;
    
    // Check if history UV is valid (inside screen)
    bool validHistory = all(historyUV >= float2(0.0, 0.0)) && all(historyUV <= float2(1.0, 1.0));
    
    if (!validHistory) {
        // No valid history, use current frame only
        return float4(currentColor, 1.0);
    }
    
    // Sample history (use bilinear filtering for smoother results)
    float3 historyColor = buffers.historyColor.Sample(historyUV).rgb;
    
    // Gather 3x3 neighborhood for variance clipping
    float3 neighbors[9];
    int idx = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float2 sampleUV = uv + float2(x, y) * texelSize;
            neighbors[idx] = buffers.currentColor.Sample(sampleUV).rgb;
            idx++;
        }
    }
    
    // Convert to YCoCg for better clipping
    float3 currentYCoCg = RGBToYCoCg(currentColor);
    float3 historyYCoCg = RGBToYCoCg(historyColor);
    
    // Calculate neighborhood mean and variance in YCoCg
    float3 mean = float3(0.0, 0.0, 0.0);
    float3 variance = float3(0.0, 0.0, 0.0);
    
    for (int i = 0; i < 9; i++) {
        float3 sampleYCoCg = RGBToYCoCg(neighbors[i]);
        mean += sampleYCoCg;
    }
    mean /= 9.0;
    
    for (int i = 0; i < 9; i++) {
        float3 sampleYCoCg = RGBToYCoCg(neighbors[i]);
        float3 diff = sampleYCoCg - mean;
        variance += diff * diff;
    }
    variance /= 9.0;
    float3 stdDev = sqrt(variance);
    
    // Variance clipping: constrain history to μ ± γσ
    float gamma = 1.25;  // Slightly larger than 1.0 to reduce flickering
    float3 minBound = mean - gamma * stdDev;
    float3 maxBound = mean + gamma * stdDev;
    
    // Clip history color to valid range
    float3 clippedHistoryYCoCg = clipToAABB(historyYCoCg, minBound, maxBound);
    float3 clippedHistory = YCoCgToRGB(clippedHistoryYCoCg);
    
    // Calculate blend factor based on velocity magnitude (reduce ghosting on fast motion)
    float velocityMagnitude = length(velocity * taaParams.screenSize);
    // Increased sensitivity (0.1 -> 0.3) and max blend (0.5 -> 0.8) for faster response during camera movement
    float motionBlendFactor = lerp(taaParams.blendFactor, 0.8, saturate(velocityMagnitude * 0.3));
    
    // Blend current with clamped history
    float3 result = lerp(clippedHistory, currentColor, motionBlendFactor);
    
    // Prevent negative colors (can happen with aggressive clipping)
    result = max(result, float3(0.0, 0.0, 0.0));
    
    return float4(result, 1.0);
}
