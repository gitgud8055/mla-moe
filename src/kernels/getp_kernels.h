#ifndef MLA_GETP_KERNELS_H
#define MLA_GETP_KERNELS_H

#include <hip/hip_runtime.h>

#include <cstddef>

#include "getp.h"

#define HIP_BLOCK 256
#define HIP_WAVE 64
#define HIP_ROWS_PER_BLOCK (HIP_BLOCK / HIP_WAVE)

extern __global__ void gemv_kernel(float *y, const float *x, const bf16_t *w,
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
extern __global__ void router_topk_kernel(
    float *scores, const float *bias, int *indices, float *weights, int experts,
    int k, int use_sigmoid, int norm_topk, float routed_scale);

extern __global__ void embedding_kernel(float *x, const bf16_t *table,
                                        int host_token,
                                        const int *device_token, int *generated,
                                        int output_index, int hidden);
extern __global__ void rmsnorm_kernel(float *y, const float *x,
                                      const bf16_t *w, int n, float eps);
extern __global__ void add_kernel(float *dst, const float *src, int n);
extern __global__ void zero_kernel(float *dst, int n);
extern __global__ void argmax_kernel(const float *logits, int n, int *result);

extern __global__ void rope_copy_kernel(const float *src, float *dst, int heads,
                                        int src_stride, int src_offset,
                                        int rope_dim, int pos,
                                        const float *inv_freq, int interleaved);
extern __global__ void softmax_rows_kernel(float *x, int rows, int n,
                                           int stride);
extern __global__ void q_absorb_kernel(float *qabs, const float *q,
                                       const bf16_t *wuk, int heads,
                                       int q_head_dim, int qk_nope, int kv_rank,
                                       size_t head_stride);
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

#endif
