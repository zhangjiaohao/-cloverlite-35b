#include "preview_gdn.cuh"
extern "C" int clover_test_preview_gdn(const float* qkv, const float* state, const float* gate,
                                       const float* scale, const float* bias, const float* history_c,
                                       float* out, int T, int HK, int HV) {
    clover_preview_gdn<<<dim3(T, HV), 128>>>(qkv, state, gate, scale, bias, history_c, out, T, HK, HV);
    return (int)cudaDeviceSynchronize();
}
extern "C" int clover_test_preview_gdn_warp(const float* qkv, const float* state, const float* gate,
                                            const float* scale, const float* bias, const float* history_c,
                                            float* out, int T, int HK, int HV) {
    clover_preview_gdn_warp<<<dim3(T, HV), 128>>>(qkv, state, gate, scale, bias, history_c, out, T, HK, HV);
    return (int)cudaDeviceSynchronize();
}
extern "C" float clover_bench_preview_gdn(const float* qkv, const float* state, const float* gate,
                                          const float* scale, const float* bias, const float* history_c,
                                          float* out, int T, int HK, int HV, int use_warp, int repeats) {
    cudaEvent_t a, b;
    if (cudaEventCreate(&a) != cudaSuccess || cudaEventCreate(&b) != cudaSuccess) return -1.f;
    cudaEventRecord(a);
    for (int i = 0; i < repeats; ++i) {
        if (use_warp) clover_preview_gdn_warp<<<dim3(T, HV), 128>>>(qkv, state, gate, scale, bias, history_c, out, T, HK, HV);
        else clover_preview_gdn<<<dim3(T, HV), 128>>>(qkv, state, gate, scale, bias, history_c, out, T, HK, HV);
    }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = -1.f;
    if (cudaGetLastError() == cudaSuccess) cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / repeats;
}
