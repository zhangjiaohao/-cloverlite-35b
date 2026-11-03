#include "preview_accept.cuh"
extern "C" int clover_test_accept(const float* logits, int vocab, const int32_t* draft, int T, float* accepted) {
    k_clover_accept_entropy<<<T - 1, 256>>>(logits, vocab, draft, T, 0.6f, accepted);
    return (int)cudaDeviceSynchronize();
}
