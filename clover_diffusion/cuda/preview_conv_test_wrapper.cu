#include "preview_conv.cuh"
extern "C" int clover_test_preview_conv(const float* qkv, const float* history,
                                        const float* forward, const float* right,
                                        float* out, int channels, int width) {
    clover_diffusion::preview_conv_silu<<<(channels * width + 255) / 256, 256>>>(
        qkv, history, forward, right, out, channels, width);
    return (int)cudaDeviceSynchronize();
}
