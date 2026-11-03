#include "preview_penalties.cuh"
__global__ void apply(float* logits, const int32_t* counts, int T, int vocab, int committed, bool closed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T * vocab) logits[i] = clover_logit(logits[i], i % vocab, i / vocab, counts, committed, closed);
}
extern "C" int clover_test_penalties(float* logits, const int32_t* counts, int T, int vocab, int committed, bool closed) {
    apply<<<(T * vocab + 255) / 256, 256>>>(logits, counts, T, vocab, committed, closed);
    return (int)cudaDeviceSynchronize();
}
