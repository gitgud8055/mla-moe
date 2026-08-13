#ifndef MLA_GETP_KERNELS_H
#define MLA_GETP_KERNELS_H

#include <hip/hip_runtime.h>

#include <cstddef>

#include "getp.h"

#define HIP_BLOCK 256
#define HIP_WAVE 64
#define HIP_ROWS_PER_BLOCK (HIP_BLOCK / HIP_WAVE)

extern __global__ void gemv_kernel(float *y, const float *x, const bf16_t *w,
                                   int d_out, int n_in, int add);

extern __global__ void batched_gemv_kernel(float *y, const float *x, const bf16_t *w,
                                           int d_out, int n_in, int add);

extern __global__ void batched_gate_up_swiglu_kernel(float *y, const float *x,
                                                     const bf16_t *gate, const bf16_t *up,
                                                     int d_out, int n_in);
extern __global__ void gate_up_swiglu_kernel(float *y, const float *x,
                                             const bf16_t *gate,
                                             const bf16_t *up, int d_out,
                                             int n_in);
extern __global__ void gemv_add_kernel(float *dst, const float *x,
                                       const bf16_t *w, int d_out, int n_in);

extern __global__ void expert_gate_up_swiglu_kernel(
    float *y, const float *x, const char *pool, const size_t *gate_offsets,
    const size_t *up_offsets, const int *topk, int slot, int d_out, int n_in);
extern __global__ void expert_down_accum_kernel(
    float *dst, const float *x, const char *pool, const size_t *down_offsets,
    const int *topk, const float *weights, int slot, int d_out, int n_in);
extern __global__ void router_topk_kernel(float *scores, const float *bias,
                                   int *indices, float *weights,
                                   int rows, int experts, int k, int use_sigmoid,
                                   int norm_topk, float routed_scale);

extern __global__ void embedding_kernel(float *x, const bf16_t *table,
                                        int host_token,
                                        const int *device_token, int *generated,
                                        int output_index, int hidden);
extern __global__ void rmsnorm_kernel(float *y, const float *x,
                                      const bf16_t *w, int n, float eps);
extern __global__ void batched_rmsnorm_kernel(float *y, const float *x,
                                              const bf16_t *w, int n, float eps);
extern __global__ void add_kernel(float *dst, const float *src, int n);
extern __global__ void zero_kernel(float *dst, int n);
extern __global__ void argmax_kernel(const float *logits, int n, int *result);

extern __global__ void rope_copy_kernel(const float *src, float *dst, int heads,
                                        int src_stride, int src_offset,
                                        int rope_dim, int pos,
                                        const float *inv_freq, int interleaved);
extern __global__ void softmax_rows_kernel(float *x, int rows, int n,
                                           int stride, int causal_span);
extern __global__ void attention_score_kernel(
    float *scores, const float *qabs, const float *qrope, const float *cache,
    int heads, int kv_len, int kv_dim, int kv_rank, int rope_dim, float scale);
extern __global__ void latent_context_kernel(float *clat, const float *scores,
                                             const float *cache, int heads,
                                             int kv_len, int kv_dim, int kv_rank);
extern __global__ void value_up_kernel(float *ctx, const float *clat,
                                       const bf16_t *wuv, int heads,
                                       int value_dim, int kv_rank,
                                       size_t head_stride);
extern __global__ void batched_value_up_kernel(float *ctx, const float *clat,
                                               const bf16_t *wuv, int heads,
                                               int value_dim, int kv_rank,
                                               size_t head_stride);
extern __global__ void dual_gemv_kernel(
    float *y0, const bf16_t *w0, int d_out0,
    float *y1, const bf16_t *w1, int d_out1,
    const float *x, int n_in);
extern __global__ void q_absorb_rope_kernel(
    float *qabs, float *qrope, const float *q, const bf16_t *wuk_t,
    int heads, int qhd, int qkn, int qkr, int kv_len, int pos,
    const float *inv_freq, int interleaved);
extern __global__ void batched_q_absorb_rope_kernel(
    float *qabs, float *qrope, const float *q, const bf16_t *wuk_t,
    int heads, int qhd, int qkn, int qkr, int kv_len, int pos_base,
    const float *inv_freq, int interleaved);

extern __global__ void router_topk_wave64_kernel(
    float *scores, const float *bias, int *indices, float *weights, int rows,
    int experts, int k, int use_sigmoid, int norm_topk, float routed_scale);

extern __global__ void moe_token_sort_kernel(const int *topk, int n_tokens, int k,
                                             int *expert_counts);

extern __global__ void expert_prefix_sum_kernel(int *offsets, const int *counts, int n);

extern __global__ void moe_token_scatter_kernel(const int *topk, int n_tokens, int k,
                                                int *expert_offsets, int *sorted_tokens, int *sorted_experts, int *sorted_slots);

extern __global__ void batched_sorted_routed_gate_up_kernel(
    float *routed_hidden, const float *x, const char *pool,
    const size_t *gate_offsets, const size_t *up_offsets,
    const int *sorted_tokens, const int *sorted_experts, const int *sorted_slots,
    int total_items, int k, int inter, int input_dim);

extern __global__ void batched_sorted_routed_down_kernel(
    float *out, const float *routed_hidden, const char *pool,
    const size_t *down_offsets, const int *sorted_tokens, const int *sorted_experts, const int *sorted_slots,
    const float *topk_weights, int total_items, int k, int hidden_dim, int inter);

extern __global__ void batched_shared_gate_up_kernel(
    float *shared_hidden, const float *x,
    const bf16_t *shared_gate, const bf16_t *shared_up,
    int n_tokens, int shared_inter, int input_dim);

extern __global__ void batched_shared_down_kernel(
    float *out, const float *shared_hidden,
    const bf16_t *shared_down, int n_tokens, int hidden_dim, int shared_inter);

extern __global__ void decode_routed_shared_gate_up_kernel(
    float *routed_hidden, float *shared_hidden, const float *x,
    const char *pool, const size_t *gate_offsets, const size_t *up_offsets,
    const int *topk, const bf16_t *shared_gate, const bf16_t *shared_up,
    int top_k, int inter, int shared_inter, int input_dim);

extern __global__ void decode_routed_shared_down_kernel(
    float *out, const float *routed_hidden, const float *shared_hidden,
    const char *pool, const size_t *down_offsets, const int *topk,
    const float *topk_weights, const bf16_t *shared_down,
    int top_k, int hidden_dim, int inter, int shared_inter);

extern __global__ void kv_norm_rope_rows_kernel(
    float *cache, const float *comp, const bf16_t *norm,
    int rows, int kv_rank, int kv_dim, int rope_dim, int position_base,
    const float *inv_freq, int interleaved, float eps);
extern __global__ void batched_kv_norm_rope_rows_kernel(
    float *cache, const float *comp, const bf16_t *norm,
    int kv_rank, int kv_dim, int rope_dim, int position_base,
    const float *inv_freq, int interleaved, float eps);

extern __global__ void fused_prefill_mla_kernel(
    float *out_clat, const float *qabs, const float *qrope,
    const float *kv_cache,
    int rows, int heads, int KVL, int QKR, int KVD, float scale);

#endif
