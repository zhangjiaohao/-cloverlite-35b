#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Single-request, all-valid Clover bidirectional GDN preview. qkv is [T,C]
// in q | k | v order; state is Strata's [K,HV,V] committed causal state.
// GGUF stores value heads in tiled order; the corresponding Q/K head is h % HK.
// The history is read-only. qkv must already have the model's convolution
// and BF16 rounding applied; output is FP32 before the model's BF16 cast.
__global__ void clover_preview_gdn(const float* qkv, const float* state,
                                   const float* gate, const float* logit_scale,
                                   const float* history_bias, const float* history_c,
                                   float* out, int T, int HK, int HV) {
    constexpr int D = 128;
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int C = (2 * HK + HV) * D;
    const int qh = h % HK;
    __shared__ float q[D], part[8], weights[33];
    float qi = qkv[(size_t)t * C + qh * D + i];
    float sum = qi * qi;
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if ((i & 31) == 0) part[i / 32] = sum;
    __syncthreads();
    qi *= rsqrtf(part[0] + part[1] + part[2] + part[3] + 1e-6f);
    q[i] = qi;
    __syncthreads();
    for (int s = 0; s < T; ++s) {
        float ki = qkv[(size_t)s * C + (HK + qh) * D + i];
        float dot = qi * ki, kn = ki * ki;
        for (int d = 16; d; d >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, d);
            kn += __shfl_down_sync(0xffffffff, kn, d);
        }
        if ((i & 31) == 0) { part[i / 32] = dot; part[4 + i / 32] = kn; }
        __syncthreads();
        if (i == 0) weights[s] = (part[0] + part[1] + part[2] + part[3]) *
            rsqrtf(part[4] + part[5] + part[6] + part[7] + 1e-6f) * logit_scale[h];
        __syncthreads();
    }
    if (i == 0) {
        weights[T] = history_bias[h] + history_c[h] * logf((float)T / 32.0f);
        float m = weights[0];
        for (int s = 1; s <= T; ++s) m = fmaxf(m, weights[s]);
        float z = 0;
        for (int s = 0; s <= T; ++s) { weights[s] = expf(weights[s] - m); z += weights[s]; }
        for (int s = 0; s <= T; ++s) weights[s] /= z;
    }
    __syncthreads();
    float history = 0;
    for (int k = 0; k < D; ++k) history = fmaf(state[((size_t)k * HV + h) * D + i], q[k], history);
    float result = weights[T] * history * rsqrtf((float)D) * expf(gate[(size_t)t * HV + h]);
    for (int s = 0; s < T; ++s) result = fmaf(weights[s], qkv[(size_t)s * C + (2 * HK + h) * D + i], result);
    out[((size_t)t * HV + h) * D + i] = result;
}

// Research kernel: four warps calculate independent block-key scores in parallel.
// Only the reduction order changes; the committed history is read-only.
__global__ void clover_preview_gdn_warp(const float* qkv, const float* state,
                                        const float* gate, const float* logit_scale,
                                        const float* history_bias, const float* history_c,
                                        float* out, int T, int HK, int HV) {
    constexpr int D = 128;
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int lane = i & 31, warp = i >> 5;
    const int C = (2 * HK + HV) * D, qh = h % HK;
    __shared__ float q[D], part[4], weights[33];
    float qi = qkv[(size_t)t * C + qh * D + i];
    float sum = qi * qi;
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) part[warp] = sum;
    __syncthreads();
    qi *= rsqrtf(part[0] + part[1] + part[2] + part[3] + 1e-6f);
    q[i] = qi;
    __syncthreads();
    for (int s = warp; s < T; s += 4) {
        float dot = 0.f, kn = 0.f;
        for (int j = 0; j < 4; ++j) {
            const int k = lane + j * 32;
            const float v = qkv[(size_t)s * C + (HK + qh) * D + k];
            dot = fmaf(q[k], v, dot);
            kn = fmaf(v, v, kn);
        }
        for (int d = 16; d; d >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, d);
            kn += __shfl_down_sync(0xffffffff, kn, d);
        }
        if (lane == 0) weights[s] = dot * rsqrtf(kn + 1e-6f) * logit_scale[h];
    }
    __syncthreads();
    if (i == 0) {
        weights[T] = history_bias[h] + history_c[h] * logf((float)T / 32.0f);
        float m = weights[0];
        for (int s = 1; s <= T; ++s) m = fmaxf(m, weights[s]);
        float z = 0;
        for (int s = 0; s <= T; ++s) { weights[s] = expf(weights[s] - m); z += weights[s]; }
        for (int s = 0; s <= T; ++s) weights[s] /= z;
    }
    __syncthreads();
    float history = 0;
    for (int k = 0; k < D; ++k) history = fmaf(state[((size_t)k * HV + h) * D + i], q[k], history);
    float result = weights[T] * history * rsqrtf((float)D) * expf(gate[(size_t)t * HV + h]);
    for (int s = 0; s < T; ++s) result = fmaf(weights[s], qkv[(size_t)s * C + (2 * HK + h) * D + i], result);
    out[((size_t)t * HV + h) * D + i] = result;
}
