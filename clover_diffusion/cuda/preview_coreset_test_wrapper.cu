#include "preview_coreset.cuh"
extern "C" int clover_test_preview_coreset(const float* logits, float* output, int T, int stride, int top_k, float top_p) {
    clover_preview_coreset<<<1, 256>>>(logits, output, T, stride, top_k, top_p);
    return (int)cudaDeviceSynchronize();
}
extern "C" int clover_test_preview_coreset_parallel(const float* logits, float* output, int T, int stride, int top_k, float top_p) {
    float* scores = nullptr;
    cudaError_t err = cudaMalloc(&scores, (size_t)T * 256 * sizeof(float));
    if (err != cudaSuccess) return (int)err;
    clover_preview_coreset_rows<<<T, 256>>>(logits, scores, stride, top_k);
    clover_preview_coreset_finish<<<1, 256>>>(scores, logits, output, T, stride, top_p);
    err = cudaDeviceSynchronize();
    cudaFree(scores);
    return (int)err;
}
