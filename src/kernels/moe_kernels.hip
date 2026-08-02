#include "getp_kernels.h"
#include "getp_kernels_device.h"

__global__ void expert_gate_up_swiglu_kernel(
    float *y, const float *x, const char *pool,
    const size_t *gate_offsets, const size_t *up_offsets,
    const int *topk, int slot, int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    int expert = topk[slot];
    const bf16_t *gate = reinterpret_cast<const bf16_t *>(pool + gate_offsets[expert]);
    const bf16_t *up = reinterpret_cast<const bf16_t *>(pool + up_offsets[expert]);
    const bf16_t *gr = gate + (size_t)row * n_in;
    const bf16_t *ur = up + (size_t)row * n_in;
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float xv = x[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0) y[row] = (gs / (1.0f + expf(-gs))) * us;
}

__global__ void expert_down_accum_kernel(
    float *dst, const float *x, const char *pool, const size_t *down_offsets,
    const int *topk, const float *weights, int slot, int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    int expert = topk[slot];
    const bf16_t *w = reinterpret_cast<const bf16_t *>(pool + down_offsets[expert]);
    const bf16_t *wr = w + (size_t)row * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += x[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) dst[row] += weights[slot] * sum;
}

__global__ void router_topk_kernel(float *scores, const float *bias,
                                   int *indices, float *weights,
                                   int experts, int k, int use_sigmoid,
                                   int norm_topk, float routed_scale) {
    __shared__ float scratch[HIP_BLOCK];
    if (use_sigmoid) {
        for (int e = (int)threadIdx.x; e < experts; e += HIP_BLOCK)
            scores[e] = 1.0f / (1.0f + expf(-scores[e]));
        __syncthreads();
    } else {
        float mx = -INFINITY;
        for (int e = (int)threadIdx.x; e < experts; e += HIP_BLOCK)
            mx = fmaxf(mx, scores[e]);
        scratch[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = HIP_BLOCK / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            __syncthreads();
        }
        mx = scratch[0];
        float sum = 0.0f;
        for (int e = (int)threadIdx.x; e < experts; e += HIP_BLOCK) {
            float v = expf(scores[e] - mx);
            scores[e] = v;
            sum += v;
        }
        scratch[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = HIP_BLOCK / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = 1.0f / scratch[0];
        for (int e = (int)threadIdx.x; e < experts; e += HIP_BLOCK) scores[e] *= inv;
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    unsigned char used[1024];
    for (int e = 0; e < experts; ++e) used[e] = 0;
    for (int slot = 0; slot < k; ++slot) {
        int best = -1;
        float best_value = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            if (used[e]) continue;
            float selected = scores[e] + (bias ? bias[e] : 0.0f);
            if (selected > best_value) {
                best_value = selected;
                best = e;
            }
        }
        used[best] = 1;
        indices[slot] = best;
        weights[slot] = scores[best];
    }
    if (norm_topk) {
        float sum = 1e-20f;
        for (int slot = 0; slot < k; ++slot) sum += weights[slot];
        for (int slot = 0; slot < k; ++slot) weights[slot] /= sum;
    }
    for (int slot = 0; slot < k; ++slot) weights[slot] *= routed_scale;
}
