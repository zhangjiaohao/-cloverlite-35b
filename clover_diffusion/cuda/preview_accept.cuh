#pragma once
#include <cuda_runtime.h>
#include <math.h>
__device__ float clover_accept_sum(float v, float* sh) {
    sh[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    const float r = sh[0]; __syncthreads(); return r;
}
__device__ float clover_accept_max(float v, float* sh) {
    sh[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]); __syncthreads(); }
    const float r = sh[0]; __syncthreads(); return r;
}
// Official llama.cpp Clover entropy-threshold acceptance, using raw target
// logits before penalties. A non-argmax draft may be accepted by design.
__global__ void k_clover_accept_entropy(const float* logits, int vocab, const int32_t* draft,
                                        int T, float delta, float* accepted) {
    __shared__ float sh[256];
    const int row = blockIdx.x;
    if (row >= T - 1) return;
    const float* x = logits + (size_t)row * vocab;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) mx = fmaxf(mx, x[i]);
    mx = clover_accept_max(mx, sh);
    float sum = 0.f, weighted = 0.f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float d = x[i] - mx;
        if (isfinite(d)) {
            float p = expf(d);
            sum += p;
            weighted += p * d;
        }
    }
    sum = clover_accept_sum(sum, sh);
    weighted = clover_accept_sum(weighted, sh);
    if (threadIdx.x == 0) {
        const float entropy = logf(sum) - weighted / sum;
        const float p = expf(x[draft[row + 1]] - mx) / sum;
        accepted[row] = p >= fminf(1.f, delta * expf(-entropy)) ? 1.f : 0.f;
    }
}

