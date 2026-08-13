#include "getp_kernels.h"
#include "getp_kernels_device.h"

namespace moe_kernels {

template <int THREADS>
__global__ void expert_gate_up_swiglu_kernel(
    float *y, const float *x, const char *pool,
    const size_t *gate_offsets, const size_t *up_offsets,
    const int *topk, int slot, int d_out, int n_in) {
    static_assert(THREADS % HIP_WAVE == 0);
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
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

template <int THREADS>
__global__ void expert_down_accum_kernel(
    float *dst, const float *x, const char *pool, const size_t *down_offsets,
    const int *topk, const float *weights, int slot, int d_out, int n_in) {
    static_assert(THREADS % HIP_WAVE == 0);
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
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
template <int THREADS>
__global__ void router_topk_kernel(float *scores, const float *bias,
                                   int *indices, float *weights,
                                   int rows, int experts, int k, int use_sigmoid,
                                   int norm_topk, float routed_scale) {
    __shared__ float scratch[THREADS];
    __shared__ int chosen[16];
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    scores += (size_t)row * experts;
    indices += (size_t)row * k;
    weights += (size_t)row * k;
    if (use_sigmoid) {
        for (int e = (int)threadIdx.x; e < experts; e += THREADS)
            scores[e] = 1.0f / (1.0f + expf(-scores[e]));
        __syncthreads();
    } else {
        float mx = -INFINITY;
        for (int e = (int)threadIdx.x; e < experts; e += THREADS)
            mx = fmaxf(mx, scores[e]);
        scratch[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            __syncthreads();
        }
        mx = scratch[0];
        float sum = 0.0f;
        for (int e = (int)threadIdx.x; e < experts; e += THREADS) {
            float v = expf(scores[e] - mx);
            scores[e] = v;
            sum += v;
        }
        scratch[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        float inv = 1.0f / scratch[0];
        for (int e = (int)threadIdx.x; e < experts; e += THREADS) scores[e] *= inv;
        __syncthreads();
    }

    if (threadIdx.x != 0) return;
    for (int slot = 0; slot < k; ++slot) {
        int best = -1;
        float best_value = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int prev = 0; prev < slot; ++prev)
                if (chosen[prev] == e) used = true;
            if (used) continue;
            float selected = scores[e] + (bias ? bias[e] : 0.0f);
            if (selected > best_value) {
                best_value = selected;
                best = e;
            }
        }
        chosen[slot] = best;
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

template <int THREADS>
__global__ void router_topk_wave64_kernel(
    float *scores, const float *bias, int *indices, float *weights,
    int rows, int experts, int k, int use_sigmoid,
    int norm_topk, float routed_scale) {
    int row = (int)blockIdx.x;
    static_assert(THREADS == HIP_WAVE);
    int lane = (int)threadIdx.x;
    if (row >= rows || lane >= HIP_WAVE) return;
    scores += (size_t)row * experts;
    indices += (size_t)row * k;
    weights += (size_t)row * k;

    float score = lane < experts ? scores[lane] : -INFINITY;
    if (use_sigmoid) {
        if (lane < experts) {
            score = 1.0f / (1.0f + expf(-score));
            scores[lane] = score;
        }
    } else {
        float mx = score;
        for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1)
            mx = fmaxf(mx, __shfl_down(mx, offset, HIP_WAVE));
        mx = __shfl(mx, 0, HIP_WAVE);
        float value = lane < experts ? expf(score - mx) : 0.0f;
        float sum = value;
        for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1)
            sum += __shfl_down(sum, offset, HIP_WAVE);
        float inv = 1.0f / __shfl(sum, 0, HIP_WAVE);
        score = value * inv;
        if (lane < experts) scores[lane] = score;
    }
    __syncthreads();

    bool used = lane >= experts;
    for (int slot = 0; slot < k; ++slot) {
        float best_value = used ? -INFINITY
                                : score + (bias ? bias[lane] : 0.0f);
        int best_index = used ? 0x7fffffff : lane;
        for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1) {
            float other_value = __shfl_down(best_value, offset, HIP_WAVE);
            int other_index = __shfl_down(best_index, offset, HIP_WAVE);
            if (other_value > best_value ||
                (other_value == best_value && other_index < best_index)) {
                best_value = other_value;
                best_index = other_index;
            }
        }
        int best = __shfl(best_index, 0, HIP_WAVE);
        if (lane == 0) {
            indices[slot] = best;
            weights[slot] = scores[best];
        }
        if (lane == best) used = true;
    }
    if (lane == 0) {
        if (norm_topk) {
            float sum = 1e-20f;
            for (int slot = 0; slot < k; ++slot) sum += weights[slot];
            for (int slot = 0; slot < k; ++slot) weights[slot] /= sum;
        }
        for (int slot = 0; slot < k; ++slot) weights[slot] *= routed_scale;
    }
}

template <int THREADS>
__global__ void moe_token_sort_kernel(const int *topk, int n_tokens, int k,
                                      int *expert_counts) {
    int token = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tokens) return;
    
    for (int slot = 0; slot < k; ++slot) {
        int expert = topk[token * k + slot];
        atomicAdd(&expert_counts[expert], 1);
    }
}

template <int THREADS>
__global__ void expert_prefix_sum_kernel(int *offsets, const int *counts, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
    }
}

template <int THREADS>
__global__ void moe_token_scatter_kernel(const int *topk, int n_tokens, int k,
                                         int *expert_offsets, int *sorted_tokens, int *sorted_experts, int *sorted_slots) {
    int token = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tokens) return;
    
    for (int slot = 0; slot < k; ++slot) {
        int expert = topk[token * k + slot];
        int pos = atomicAdd(&expert_offsets[expert], 1);
        sorted_tokens[pos] = token;
        sorted_experts[pos] = expert;
        sorted_slots[pos] = slot;
    }
}

template <int THREADS>
__global__ void batched_sorted_routed_gate_up_kernel(
    float *routed_hidden, const float *x, const char *pool,
    const size_t *gate_offsets, const size_t *up_offsets,
    const int *sorted_tokens, const int *sorted_experts, const int *sorted_slots,
    int total_items, int k, int inter, int input_dim) {
    
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int j = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    int item = (int)blockIdx.y;
    
    if (j >= inter || item >= total_items) return;
    
    int expert = sorted_experts[item];
    int token = sorted_tokens[item];
    int slot = sorted_slots[item];
    
    const bf16_t *gate = reinterpret_cast<const bf16_t *>(pool + gate_offsets[expert]);
    const bf16_t *up = reinterpret_cast<const bf16_t *>(pool + up_offsets[expert]);
    
    float *dst = routed_hidden + ((size_t)token * k + slot) * inter;
    
    const bf16_t *gr = gate + (size_t)j * input_dim;
    const bf16_t *ur = up + (size_t)j * input_dim;
    const float *xb = x + (size_t)token * input_dim;
    
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < input_dim; i += HIP_WAVE) {
        float xv = xb[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0) dst[j] = (gs / (1.0f + expf(-gs))) * us;
}

template <int THREADS>
__global__ void batched_sorted_routed_down_kernel(
    float *out, const float *routed_hidden, const char *pool,
    const size_t *down_offsets, const int *sorted_tokens, const int *sorted_experts, const int *sorted_slots,
    const float *topk_weights, int total_items, int k, int hidden_dim, int inter) {
    
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int d = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    int item = (int)blockIdx.y;
    
    if (d >= hidden_dim || item >= total_items) return;
    
    int expert = sorted_experts[item];
    int token = sorted_tokens[item];
    int slot = sorted_slots[item];
    float weight = topk_weights[token * k + slot];
    
    const bf16_t *down = reinterpret_cast<const bf16_t *>(pool + down_offsets[expert]);
    const float *hp = routed_hidden + ((size_t)token * k + slot) * inter;
    const bf16_t *wr = down + (size_t)d * inter;
    
    float sum = 0.0f;
    for (int j = lane; j < inter; j += HIP_WAVE) {
        sum += hp[j] * gpu_bf16_to_f32(wr[j]);
    }
    sum = wave_sum(sum);
    
    if (lane == 0) {
        float val = sum * weight;
        atomicAdd(&out[token * hidden_dim + d], val);
    }
}

template <int THREADS>
__global__ void batched_shared_gate_up_kernel(
    float *shared_hidden, const float *x,
    const bf16_t *shared_gate, const bf16_t *shared_up,
    int n_tokens, int shared_inter, int input_dim) {
    
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int j = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    int token = (int)blockIdx.y;
    
    if (j >= shared_inter || token >= n_tokens) return;
    
    float *dst = shared_hidden + (size_t)token * shared_inter;
    const bf16_t *gr = shared_gate + (size_t)j * input_dim;
    const bf16_t *ur = shared_up + (size_t)j * input_dim;
    const float *xb = x + (size_t)token * input_dim;
    
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < input_dim; i += HIP_WAVE) {
        float xv = xb[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0) dst[j] = (gs / (1.0f + expf(-gs))) * us;
}

template <int THREADS>
__global__ void batched_shared_down_kernel(
    float *out, const float *shared_hidden,
    const bf16_t *shared_down, int n_tokens, int hidden_dim, int shared_inter) {
    
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int d = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    int token = (int)blockIdx.y;
    
    if (d >= hidden_dim || token >= n_tokens) return;
    
    const float *hp = shared_hidden + (size_t)token * shared_inter;
    const bf16_t *wr = shared_down + (size_t)d * shared_inter;
    
    float sum = 0.0f;
    for (int j = lane; j < shared_inter; j += HIP_WAVE) {
        sum += hp[j] * gpu_bf16_to_f32(wr[j]);
    }
    sum = wave_sum(sum);
    
    if (lane == 0) {
        atomicAdd(&out[token * hidden_dim + d], sum);
    }
}

template <int THREADS>
__global__ void decode_routed_shared_gate_up_kernel(
    float *routed_hidden, float *shared_hidden, const float *x,
    const char *pool, const size_t *gate_offsets, const size_t *up_offsets,
    const int *topk, const bf16_t *shared_gate, const bf16_t *shared_up,
    int top_k, int inter, int shared_inter, int input_dim) {
    static_assert(THREADS % HIP_WAVE == 0);
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    int routed_rows = top_k * inter;
    int total = routed_rows + shared_inter;
    if (row >= total) return;

    const bf16_t *gate;
    const bf16_t *up;
    float *dst;
    int j;
    if (row < routed_rows) {
        int slot = row / inter;
        j = row - slot * inter;
        int expert = topk[slot];
        gate = reinterpret_cast<const bf16_t *>(pool + gate_offsets[expert]);
        up = reinterpret_cast<const bf16_t *>(pool + up_offsets[expert]);
        dst = routed_hidden + (size_t)slot * inter;
    } else {
        j = row - routed_rows;
        gate = shared_gate;
        up = shared_up;
        dst = shared_hidden;
    }
    const bf16_t *gr = gate + (size_t)j * input_dim;
    const bf16_t *ur = up + (size_t)j * input_dim;
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < input_dim; i += HIP_WAVE) {
        float xv = x[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0)
        dst[j] = (gs / (1.0f + expf(-gs))) * us;
}

template <int THREADS>
__global__ void decode_routed_shared_down_kernel(
    float *out, const float *routed_hidden, const float *shared_hidden,
    const char *pool, const size_t *down_offsets, const int *topk,
    const float *topk_weights, const bf16_t *shared_down,
    int top_k, int hidden_dim, int inter, int shared_inter) {
    static_assert(THREADS % HIP_WAVE == 0);
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int d = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
    if (d >= hidden_dim) return;
    float total = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        int expert = topk[slot];
        const bf16_t *down = reinterpret_cast<const bf16_t *>(pool + down_offsets[expert]);
        const bf16_t *wr = down + (size_t)d * inter;
        const float *hp = routed_hidden + (size_t)slot * inter;
        float sum = 0.0f;
        for (int j = lane; j < inter; j += HIP_WAVE)
            sum += hp[j] * gpu_bf16_to_f32(wr[j]);
        sum = wave_sum(sum);
        if (lane == 0) total += topk_weights[slot] * sum;
    }
    const bf16_t *swr = shared_down + (size_t)d * shared_inter;
    float shared = 0.0f;
    for (int j = lane; j < shared_inter; j += HIP_WAVE)
        shared += shared_hidden[j] * gpu_bf16_to_f32(swr[j]);
    shared = wave_sum(shared);
    if (lane == 0) {
        float ffn = total + shared;
        out[d] += ffn;
    }
}

}  // namespace moe_kernels

template __global__ void moe_kernels::expert_gate_up_swiglu_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const char *, const size_t *, const size_t *,
    const int *, int, int, int);
template __global__ void moe_kernels::expert_down_accum_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const char *, const size_t *, const int *,
    const float *, int, int, int);
template __global__ void moe_kernels::router_topk_kernel<kernel_tuning::ROUTER_THREADS>(
    float *, const float *, int *, float *, int, int, int, int, int, float);
template __global__ void moe_kernels::router_topk_wave64_kernel<kernel_tuning::ROUTER_WAVE_THREADS>(
    float *, const float *, int *, float *, int, int, int, int, int, float);
template __global__ void moe_kernels::decode_routed_shared_gate_up_kernel<kernel_tuning::GEMV_THREADS>(
    float *, float *, const float *, const char *, const size_t *,
    const size_t *, const int *, const bf16_t *, const bf16_t *, int, int,
    int, int);
template __global__ void moe_kernels::decode_routed_shared_down_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const float *, const char *, const size_t *,
    const int *, const float *, const bf16_t *, int, int, int, int);

template __global__ void moe_kernels::moe_token_sort_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    const int *, int, int, int *);
template __global__ void moe_kernels::expert_prefix_sum_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    int *, const int *, int);
template __global__ void moe_kernels::moe_token_scatter_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    const int *, int, int, int *, int *, int *, int *);

template __global__ void moe_kernels::batched_sorted_routed_gate_up_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const char *, const size_t *, const size_t *,
    const int *, const int *, const int *, int, int, int, int);
template __global__ void moe_kernels::batched_sorted_routed_down_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const char *, const size_t *, const int *,
    const int *, const int *, const float *, int, int, int, int);
template __global__ void moe_kernels::batched_shared_gate_up_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const bf16_t *, const bf16_t *, int, int, int);
template __global__ void moe_kernels::batched_shared_down_kernel<kernel_tuning::GEMV_THREADS>(
    float *, const float *, const bf16_t *, int, int, int);
