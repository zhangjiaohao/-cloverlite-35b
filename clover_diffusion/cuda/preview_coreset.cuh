#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Correctness-first Clover coreset for one full, valid block, 256 experts.
// Input and output may alias; stride can include an extra shared-expert logit.
// top_k must be 1..256. Expert ties are resolved by lower expert index.
__global__ void clover_preview_coreset(const float* logits, float* output,
                                       int T, int stride, int top_k, float top_p) {
    constexpr int E = 256;
    const int e = threadIdx.x;
    __shared__ float row[E], reduce[E], sorted[E];
    __shared__ int ids[E], allowed[E];
    float total = 0;
    for (int t = 0; t < T; ++t) {
        float val = logits[(size_t)t * stride + e];
        row[e] = val;
        reduce[e] = val;
        __syncthreads();
        for (int d = E / 2; d; d >>= 1) {
            if (e < d) reduce[e] = fmaxf(reduce[e], reduce[e + d]);
            __syncthreads();
        }
        const float w = expf(val - reduce[0]);
        // All threads must finish reading reduce[0] before it is overwritten.
        __syncthreads();
        reduce[e] = w;
        __syncthreads();
        for (int d = E / 2; d; d >>= 1) {
            if (e < d) reduce[e] += reduce[e + d];
            __syncthreads();
        }
        int rank = 0;
        for (int j = 0; j < E; ++j) rank += row[j] > val || (row[j] == val && j < e);
        if (rank < top_k) total += w / reduce[0];
        __syncthreads();
    }
    sorted[e] = total;
    ids[e] = e;
    __syncthreads();
    // Bitonic sort descending by score, then ascending by ID.
    for (int k = 2; k <= E; k *= 2) {
        for (int j = k / 2; j; j /= 2) {
            int other = e ^ j;
            if (other > e) {
                float a = sorted[e], b = sorted[other];
                int ia = ids[e], ib = ids[other];
                bool b_before_a = b > a || (b == a && ib < ia);
                bool descending = (e & k) == 0;
                if (b_before_a == descending) {
                    sorted[e] = b; sorted[other] = a;
                    ids[e] = ib; ids[other] = ia;
                }
            }
            __syncthreads();
        }
    }
    if (e == 0) {
        float mass = 0, previous = 0;
        for (int j = 0; j < E; ++j) mass += sorted[j];
        for (int j = 0; j < E; ++j) {
            allowed[ids[j]] = (j == 0 || previous < top_p * mass);
            previous += sorted[j];
        }
    }
    __syncthreads();
    for (int t = 0; t < T; ++t)
        output[(size_t)t * stride + e] = allowed[e] ? logits[(size_t)t * stride + e] : -INFINITY;
}

// Equivalent two-stage coreset: independent rows are processed concurrently.
__global__ void clover_preview_coreset_rows(const float* logits, float* scores, int stride, int top_k) {
    constexpr int E = 256;
    const int e = threadIdx.x, t = blockIdx.x;
    __shared__ float row[E], reduce[E];
    const float val = logits[(size_t)t * stride + e];
    row[e] = val; reduce[e] = val;
    __syncthreads();
    for (int d = E / 2; d; d >>= 1) {
        if (e < d) reduce[e] = fmaxf(reduce[e], reduce[e + d]);
        __syncthreads();
    }
    const float w = expf(val - reduce[0]);
    __syncthreads();
    reduce[e] = w;
    __syncthreads();
    for (int d = E / 2; d; d >>= 1) {
        if (e < d) reduce[e] += reduce[e + d];
        __syncthreads();
    }
    int rank = 0;
    for (int j = 0; j < E; ++j) rank += row[j] > val || (row[j] == val && j < e);
    scores[(size_t)t * E + e] = rank < top_k ? w / reduce[0] : 0.f;
}

__global__ void clover_preview_coreset_finish(const float* scores, const float* logits, float* output,
                                             int T, int stride, float top_p) {
    constexpr int E = 256;
    const int e = threadIdx.x;
    __shared__ float sorted[E];
    __shared__ int ids[E], allowed[E];
    float total = 0;
    for (int t = 0; t < T; ++t) total += scores[(size_t)t * E + e];
    sorted[e] = total; ids[e] = e;
    __syncthreads();
    for (int k = 2; k <= E; k *= 2) {
        for (int j = k / 2; j; j /= 2) {
            const int other = e ^ j;
            if (other > e) {
                const float a = sorted[e], b = sorted[other];
                const int ia = ids[e], ib = ids[other];
                const bool b_before_a = b > a || (b == a && ib < ia);
                const bool descending = (e & k) == 0;
                if (b_before_a == descending) {
                    sorted[e] = b; sorted[other] = a;
                    ids[e] = ib; ids[other] = ia;
                }
            }
            __syncthreads();
        }
    }
    if (e == 0) {
        float mass = 0, previous = 0;
        for (int j = 0; j < E; ++j) mass += sorted[j];
        for (int j = 0; j < E; ++j) {
            allowed[ids[j]] = (j == 0 || previous < top_p * mass);
            previous += sorted[j];
        }
    }
    __syncthreads();
    for (int t = 0; t < T; ++t)
        output[(size_t)t * stride + e] = allowed[e] ? logits[(size_t)t * stride + e] : -INFINITY;
}
