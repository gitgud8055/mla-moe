/* Shared constants, vector types and small device helpers.
 * Everything model- or GPU-specific that more than one translation stage
 * needs lives here; kernel code lives in ops.h / prefill.h / decode.h. */
#ifndef MLA_UTILS_H
#define MLA_UTILS_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "getp.h" /* bf16_t (via model.h -> tensor.h) */

static void hip_fail(hipError_t status, const char *expr, const char *file, int line)
{
    if (status == hipSuccess)
        return;
    std::fprintf(stderr, "HIP failure at %s:%d: %s: %s\n", file, line, expr,
                 hipGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define HIP_CHECK(expr) hip_fail((expr), #expr, __FILE__, __LINE__)
#define HIP_LAUNCH_CHECK() HIP_CHECK(hipGetLastError())

namespace utils {

namespace constants {

/* AMD Instinct MI250 (one GCD, gfx90a) — verified with rocminfo. */
namespace gpu {
    constexpr int WAVE = 64;
    constexpr int CU = 104;
    constexpr int SIMD_PER_CU = 4;
    constexpr size_t LDS = 64 * 1024;
    constexpr int MAX_BLOCK_THREADS = 1024;
    constexpr int CACHELINE = 128;
    constexpr int CLOCK_MHZ = 1700;
}

namespace model {

/* deepseek-ai/DeepSeek-V2-Lite (config.json) */
namespace dsv {
    constexpr int N_LAYERS = 27;
    constexpr int HIDDEN = 2048;
    constexpr int N_HEADS = 16;
    constexpr int Q_LORA = 0; /* q_proj only, no q LoRA */
    constexpr int KV_LORA = 512;
    constexpr int QK_NOPE = 128;
    constexpr int QK_ROPE = 64;
    constexpr int V_HEAD = 128;
    constexpr int DENSE_INTER = 10944;
    constexpr int MOE_INTER = 1408;
    constexpr int N_EXPERTS = 64;
    constexpr int TOP_K = 6;
    constexpr int N_SHARED = 2;
    constexpr int FIRST_DENSE = 1;
    constexpr int VOCAB = 102400;
    constexpr int QK_HEAD = QK_NOPE + QK_ROPE;         /* 192  */
    constexpr int Q_DIM = N_HEADS * QK_HEAD;           /* 3072 */
    constexpr int KV_DIM = KV_LORA + QK_ROPE;          /* 576  */
    constexpr int ATTN_OUT = N_HEADS * V_HEAD;         /* 2048 */
    constexpr int SHARED_INTER = N_SHARED * MOE_INTER; /* 2816 */
}

/* zai-org/GLM-4.7-Flash (config.json) */
namespace glm {
    constexpr int N_LAYERS = 47;
    constexpr int HIDDEN = 2048;
    constexpr int N_HEADS = 20;
    constexpr int Q_LORA = 768;
    constexpr int KV_LORA = 512;
    constexpr int QK_NOPE = 192;
    constexpr int QK_ROPE = 64;
    constexpr int V_HEAD = 256;
    constexpr int DENSE_INTER = 10240;
    constexpr int MOE_INTER = 1536;
    constexpr int N_EXPERTS = 64;
    constexpr int TOP_K = 4;
    constexpr int N_SHARED = 1;
    constexpr int FIRST_DENSE = 1;
    constexpr int VOCAB = 154880;
    constexpr int QK_HEAD = QK_NOPE + QK_ROPE;         /* 256  */
    constexpr int Q_DIM = N_HEADS * QK_HEAD;           /* 5120 */
    constexpr int KV_DIM = KV_LORA + QK_ROPE;          /* 576  */
    constexpr int ATTN_OUT = N_HEADS * V_HEAD;         /* 5120 */
    constexpr int SHARED_INTER = N_SHARED * MOE_INTER; /* 1536 */
}

} // namespace model

/* Default launch/tuning knobs. The per-shape overrides live in the ops.h
 * dispatch branches — tune there, per (model, stage) shape. */
namespace kernel {
    constexpr int GEMV_THREADS = 256;
    constexpr int ELEMENTWISE_THREADS = 256;
    constexpr int ATTENTION_THREADS = 256;
    constexpr int ROUTER_THREADS = 256;
    constexpr int ROUTER_WAVE_THREADS = 64;
    constexpr int MFMA_TILE_N = 64;       /* output columns per MFMA block   */
    constexpr int MFMA_BATCH_TILE = 16;   /* rows per MFMA block (1 tile)    */
    constexpr int PREFILL_ROW_TILES = 2;  /* row tiles in the large-M GEMM   */
    constexpr int PREFILL_ROWS = MFMA_BATCH_TILE * PREFILL_ROW_TILES;
    constexpr int LARGE_M_MIN_ROWS = 1024; /* switch to the large-M GEMM     */
    constexpr int MFMA_MIN_ROWS = 4;       /* below this: one-wave GEMV      */
    constexpr int FLASH_MIN_ROWS = 8;      /* below this: score+softmax path */
    constexpr int FLASH_LARGE_BATCH = 128;
    constexpr int FLASH_SMALL_THREADS = 512, FLASH_SMALL_KV_TILE = 16;
    constexpr int FLASH_LARGE_THREADS = 256, FLASH_LARGE_KV_TILE = 8;
    constexpr int PREFILL_THREADS = 256, PREFILL_KV_TILE = 8;
    constexpr int GROUPED_ROUTE_TILE = 16; /* MoE route alignment tile      */
}

} // namespace constants

namespace types {
    typedef short bf16x4 __attribute__((ext_vector_type(4)));
    typedef short bf16x8 __attribute__((ext_vector_type(8)));
    typedef char i8x8 __attribute__((ext_vector_type(8)));
    typedef int int4v __attribute__((ext_vector_type(4)));
    typedef float fp32v4 __attribute__((ext_vector_type(4)));
}

} // namespace utils

#define HIP_WAVE 64

/* ---- Small device helpers shared by every kernel header ---- */

__device__ __forceinline__ float gpu_bf16_to_f32(bf16_t value) {
    return __uint_as_float((uint32_t)value << 16);
}

__device__ __forceinline__ bf16_t gpu_f32_to_bf16(float value) {
    return (bf16_t)(__float_as_uint(value) >> 16);
}

__device__ __forceinline__ float wave_sum(float value) {
    for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, HIP_WAVE);
    return value;
}

/* Weight-only int8: when `ws` (per-output-row scales) is set, `w` actually
 * holds row-major int8 and each element dequantizes as ws[row] * w8[i].
 * When `ws` is null, `w` is plain bf16. */
__device__ __forceinline__ utils::types::bf16x8 load_w8(
    const bf16_t *w, const float *ws, size_t gc, int n_in, int k0) {
    utils::types::bf16x8 value = {};
    if (!ws) {
        if (k0 + 8 <= n_in)
            value = *reinterpret_cast<const utils::types::bf16x8 *>(
                w + gc * n_in + k0);
        else
#pragma unroll
            for (int q = 0; q < 8; ++q)
                if (k0 + q < n_in)
                    value[q] = reinterpret_cast<const short *>(
                        w)[gc * n_in + k0 + q];
        return value;
    }
    const char *w8 = reinterpret_cast<const char *>(w);
    const float s = ws[gc];
    if (k0 + 8 <= n_in) {
        const utils::types::i8x8 wv =
            *reinterpret_cast<const utils::types::i8x8 *>(w8 + gc * n_in + k0);
#pragma unroll
        for (int q = 0; q < 8; ++q)
            value[q] = (short)gpu_f32_to_bf16(s * (float)wv[q]);
    } else {
#pragma unroll
        for (int q = 0; q < 8; ++q)
            if (k0 + q < n_in)
                value[q] = (short)gpu_f32_to_bf16(
                    s * (float)w8[gc * n_in + k0 + q]);
    }
    return value;
}

/* Stage 8 int8 values of row `gc` starting at column `k0` (column tail-safe). */
__device__ __forceinline__ utils::types::i8x8 load_i8x8(
    const char *w8, size_t gc, int n_in, int k0) {
    utils::types::i8x8 value = {};
    if (k0 + 8 <= n_in)
        value = *reinterpret_cast<const utils::types::i8x8 *>(
            w8 + gc * n_in + k0);
    else
#pragma unroll
        for (int q = 0; q < 8; ++q)
            if (k0 + q < n_in) value[q] = w8[gc * n_in + k0 + q];
    return value;
}

#endif /* MLA_UTILS_H */
