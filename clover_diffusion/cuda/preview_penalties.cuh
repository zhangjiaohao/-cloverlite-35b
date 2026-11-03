#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Clover's default generation penalties apply to committed response tokens,
// not prompt tokens or the tokens currently visible in the preview block.
__device__ float clover_logit(float x, int id, int row, const int32_t* counts, int committed, bool closed) {
    if (counts[id] > 0) {
        x -= 0.9f;
        x = x < 0.f ? x * 1.05f : x / 1.05f;
    }
    const float position = (float) (committed + row + 1);
    if (id == 248044 || id == 248046)
        x += 10.f * fminf(1.f, fmaxf(0.f, (position - 8192.f) / 8192.f));
    if (id == 248069 && !closed)
        x += 10.f * fminf(1.f, fmaxf(0.f, (position - 4096.f) / 4096.f));
    return x;
}

__global__ void k_clover_count(const int32_t* tokens, int T, int32_t* counts) {
    int i = threadIdx.x;
    if (i < T) atomicAdd(counts + tokens[i], 1);
}

