#ifndef MLA_GETP_KERNELS_H
#define MLA_GETP_KERNELS_H

#include <hip/hip_runtime.h>

#include <cstddef>

#include "getp.h"

#define HIP_WAVE 64

namespace kernel_tuning
{
    inline constexpr int GEMV_THREADS = 256;
    inline constexpr int ELEMENTWISE_THREADS = 256;
    inline constexpr int ATTENTION_THREADS = 256;
    inline constexpr int ROUTER_THREADS = 256;
    inline constexpr int ROUTER_WAVE_THREADS = 64;
}

namespace projection_kernels
{
    template <int THREADS>
    __global__ void gemv_kernel(
        float *, const float *, const bf16_t *, int, int, int);
    template <int THREADS>
    __global__ void gate_up_swiglu_kernel(
        float *, const float *, const bf16_t *, const bf16_t *, int, int);
    template <int THREADS>
    __global__ void gemv_add_kernel(
        float *, const float *, const bf16_t *, int, int);
    template <int THREADS>
    __global__ void dual_gemv_kernel(
        float *, const bf16_t *, int, float *, const bf16_t *, int,
        const float *, int);
}

namespace elementwise_kernels
{
    template <int THREADS>
    __global__ void embedding_kernel(
        float *, const bf16_t *, int, const int *, int *, int, int);
    template <int THREADS>
    __global__ void rmsnorm_kernel(
        float *, const float *, const bf16_t *, int, float);
    template <int THREADS>
    __global__ void add_kernel(float *, const float *, int);
    template <int THREADS>
    __global__ void zero_kernel(float *, int);
    template <int THREADS>
    __global__ void argmax_kernel(const float *, int, int *);
}

namespace attention_kernels
{
    template <int THREADS>
    __global__ void rope_copy_kernel(
        const float *, float *, int, int, int, int, int, const float *, int);
    template <int THREADS>
    __global__ void softmax_rows_kernel(
        float *, int, int, int, int);
    template <int THREADS>
    __global__ void q_absorb_rope_kernel(
        float *, float *, const float *, const bf16_t *, int, int, int, int, int,
        int, const float *, int);
    template <int THREADS>
    __global__ void attention_score_kernel(
        float *, const float *, const float *, const float *, int, int, int, int,
        int, float);
    template <int THREADS>
    __global__ void latent_context_kernel(
        float *, const float *, const float *, int, int, int, int);
    template <int THREADS>
    __global__ void value_up_kernel(
        float *, const float *, const bf16_t *, int, int, int, size_t);
    template <int THREADS>
    __global__ void kv_norm_rope_rows_kernel(
        float *, const float *, const bf16_t *, int, int, int, int, int,
        const float *, int, float);
}

namespace moe_kernels
{
    template <int THREADS>
    __global__ void expert_gate_up_swiglu_kernel(
        float *, const float *, const char *, const size_t *, const size_t *,
        const int *, int, int, int);
    template <int THREADS>
    __global__ void expert_down_accum_kernel(
        float *, const float *, const char *, const size_t *, const int *,
        const float *, int, int, int);
    template <int THREADS>
    __global__ void router_topk_kernel(
        float *, const float *, int *, float *, int, int, int, int, int, float);
    template <int THREADS>
    __global__ void router_topk_wave64_kernel(
        float *, const float *, int *, float *, int, int, int, int, int, float);
    template <int THREADS>
    __global__ void decode_routed_shared_gate_up_kernel(
        float *, float *, const float *, const char *, const size_t *,
        const size_t *, const int *, const bf16_t *, const bf16_t *, int, int,
        int, int);
    template <int THREADS>
    __global__ void decode_routed_shared_down_kernel(
        float *, const float *, const float *, const char *, const size_t *,
        const int *, const float *, const bf16_t *, int, int, int, int);
}
#endif
