#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace clover_diffusion {
// Single-sequence Clover preview convolution. qkv/output are [T,C], history is [C,3],
// forward weights are [C,4], right weights [C,3]. History is read-only. T <= 32.
// Output stays FP32; the caller must apply the checkpoint's required BF16 rounding.
__global__ void preview_conv_silu(const float* __restrict__ qkv, const float* __restrict__ history,
                                  const float* __restrict__ forward, const float* __restrict__ right,
                                  float* __restrict__ output, int channels, int width) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= channels * width) return;
    const int t = idx / channels, c = idx % channels;
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int source = t + j - 3;
        const float x = source < 0 ? history[c * 3 + source + 3] : qkv[source * channels + c];
        sum += x * forward[c * 4 + j];
    }
    #pragma unroll
    for (int d = 1; d <= 3; ++d) {
        if (t + d < width) sum += qkv[(t + d) * channels + c] * right[c * 3 + d - 1];
    }
    output[idx] = sum / (1.f + expf(-sum));
}
} // namespace clover_diffusion
