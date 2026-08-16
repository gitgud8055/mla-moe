/* Model-agnostic operators. Each public function is a plain host wrapper that
 * can be reused anywhere: it receives explicit pointers/shapes, dispatches on
 * shape (see the commented if/else ladders — tune the template picks there)
 * and launches the kernel templates that live in this header.
 *
 * Quantization scheme (fixed, no runtime configuration):
 *   - weights passed with a non-null per-row scale array `ws` are row-major
 *     int8; without `ws` they are bf16.
 *   - int8 x int8 runs on V_MFMA_I32_16X16X16I8 with dynamic per-token
 *     activation scales applied once in the epilogue. */
#ifndef MLA_OPS_H
#define MLA_OPS_H

#include "utils.h"
#include "getp_state.h"

namespace ops {

namespace kt = utils::constants::kernel;
using utils::types::bf16x4;
using utils::types::bf16x8;
using utils::types::i8x8;
using utils::types::int4v;
using utils::types::fp32v4;

constexpr int THREADS = kt::GEMV_THREADS;
constexpr int WAVES = THREADS / HIP_WAVE;
constexpr int BATCH_TILE = 2;

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 64;
constexpr int MFMA_K = 64;
constexpr int MFMA_LDS_K = MFMA_K + 4;
constexpr int I8_LDS_K = MFMA_K + 8; /* stride % 8 == 0 keeps i8x8 stores aligned */

/* Worst-case LDS footprints (gate/up kernels stage x + two weight tiles). */
static_assert(((size_t)MFMA_M * kt::PREFILL_ROW_TILES + 2 * MFMA_N)
                  * MFMA_LDS_K * sizeof(bf16_t)
              <= utils::constants::gpu::LDS);
static_assert(((size_t)MFMA_M * kt::PREFILL_ROW_TILES + 2 * MFMA_N) * I8_LDS_K
              <= utils::constants::gpu::LDS);

inline int div_up(int value, int divisor) { return (value + divisor - 1) / divisor; }

/* A wrapper hit a shape with no explicit branch in its dispatch ladder.
 * Every real (model, phase) case must be listed; abort loudly so a missing
 * case is added instead of silently running an untuned default. */
[[noreturn]] inline void unlisted_shape(const char *where,
                                        long d0, long d1, long d2) {
    std::fprintf(stderr,
        "[ops] unlisted shape in %s: (%ld, %ld, %ld) — add an explicit "
        "branch to its dispatch ladder\n", where, d0, d1, d2);
    std::abort();
}

/* ------------------------------------------------------------------ */
/* Quantization kernels                                               */
/* ------------------------------------------------------------------ */

/* Per-output-row symmetric int8 weight quantization, run once at load time. */
__global__ void quantize_rows_kernel(const bf16_t *__restrict__ src,
                                     char *__restrict__ dst,
                                     float *__restrict__ scales,
                                     int rows, int cols) {
    __shared__ float red[THREADS];
    const int r = (int)blockIdx.x;
    if (r >= rows) return;
    const bf16_t *s_row = src + (size_t)r * cols;
    float m = 0.0f;
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x)
        m = fmaxf(m, fabsf(gpu_bf16_to_f32(s_row[i])));
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float amax = red[0];
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) scales[r] = amax > 0.0f ? amax / 127.0f : 1.0f;
    char *d_row = dst + (size_t)r * cols;
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x) {
        int q = (int)nearbyintf(gpu_bf16_to_f32(s_row[i]) * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        d_row[i] = (char)q;
    }
}

/* Dynamic per-token int8 activation quantization (w8a8 path). */
template <typename SRC>
__device__ __forceinline__ void quantize_act_rows_body(
    const SRC *__restrict__ src, char *__restrict__ dst,
    float *__restrict__ scales, int rows, int cols) {
    __shared__ float red[THREADS];
    const int r = (int)blockIdx.x;
    if (r >= rows) return;
    const SRC *s_row = src + (size_t)r * cols;
    float m = 0.0f;
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x) {
        float v;
        if constexpr (sizeof(SRC) == 2) v = gpu_bf16_to_f32(s_row[i]);
        else v = s_row[i];
        m = fmaxf(m, fabsf(v));
    }
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = (int)blockDim.x / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float amax = red[0];
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) scales[r] = amax > 0.0f ? amax / 127.0f : 1.0f;
    char *d_row = dst + (size_t)r * cols;
    for (int i = (int)threadIdx.x; i < cols; i += (int)blockDim.x) {
        float v;
        if constexpr (sizeof(SRC) == 2) v = gpu_bf16_to_f32(s_row[i]);
        else v = s_row[i];
        int q = (int)nearbyintf(v * inv);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        d_row[i] = (char)q;
    }
}

__global__ void quantize_act_rows_f32(const float *src, char *dst,
                                      float *scales, int rows, int cols) {
    quantize_act_rows_body(src, dst, scales, rows, cols);
}

__global__ void quantize_act_rows_bf16(const bf16_t *src, char *dst,
                                       float *scales, int rows, int cols) {
    quantize_act_rows_body(src, dst, scales, rows, cols);
}

/* ------------------------------------------------------------------ */
/* Dense GEMM kernels (bf16 and w8a8)                                 */
/* ------------------------------------------------------------------ */

__global__ void mfma_gemm(float *__restrict__ y,
                          const float *__restrict__ x,
                          const bf16_t *__restrict__ w,
                          int rows, int d_out, int n_in, int add) {
    __shared__ bf16_t sx[MFMA_M][MFMA_LDS_K];
    __shared__ bf16_t sw[MFMA_N][MFMA_LDS_K];

    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * MFMA_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    fp32v4 acc = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < MFMA_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(x[(size_t)gr * n_in + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            bf16x8 value = {};
            const int gc = col0 + c;
            if (gc < d_out)
                value = load_bf16x8(w, (size_t)gc, n_in, bk + k);
            *reinterpret_cast<bf16x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                &sx[lane & 15][k + k4]);
            const bf16x4 bv = *reinterpret_cast<const bf16x4 *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                av, bv, acc, 0, 0, 0);
        }
        __syncthreads();
    }

    if (col < d_out) {
        const int r0 = row0 + 4 * (lane >> 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = r0 + q;
            if (r < rows) {
                float *dst = y + (size_t)r * d_out + col;
                if (add) *dst += acc[q]; else *dst = acc[q];
            }
        }
    }
}

/* Large-M specialization for flattened prefill: multiple 16-row MFMA tiles
 * share one weight tile, cutting repeated global weight traffic. */
template <int M_TILES>
__global__ void mfma_gemm_rows(float *__restrict__ y,
                               const float *__restrict__ x,
                               const bf16_t *__restrict__ w,
                               int rows, int d_out, int n_in, int add) {
    constexpr int BLOCK_M = MFMA_M * M_TILES;
    __shared__ bf16_t sx[BLOCK_M][MFMA_LDS_K];
    __shared__ bf16_t sw[MFMA_N][MFMA_LDS_K];
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * BLOCK_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    fp32v4 acc[M_TILES] = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(x[(size_t)gr * n_in + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            bf16x8 value = {};
            if (gc < d_out)
                value = load_bf16x8(w, (size_t)gc, n_in, bk + k);
            *reinterpret_cast<bf16x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 bv = *reinterpret_cast<const bf16x4 *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                    &sx[mt * MFMA_M + (lane & 15)][k + k4]);
                acc[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bv, acc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < d_out) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = row0 + mt * MFMA_M + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = r0 + q;
                if (r < rows) {
                    float *dst = y + (size_t)r * d_out + col;
                    if (add) *dst += acc[mt][q];
                    else *dst = acc[mt][q];
                }
            }
        }
    }
}

/* w8a8 dense GEMM: pre-quantized int8 activations x int8 weights on the int8
 * MFMA path; int32 accumulation is exact, scales applied in the epilogue. */
template <int M_TILES>
__global__ void mfma_gemm_i8(float *__restrict__ y,
                             const char *__restrict__ x8,
                             const float *__restrict__ x_s,
                             const char *__restrict__ w8,
                             const float *__restrict__ ws,
                             int rows, int d_out, int n_in, int add) {
    constexpr int BLOCK_M = MFMA_M * M_TILES;
    __shared__ char sx[BLOCK_M][I8_LDS_K];
    __shared__ char sw[MFMA_N][I8_LDS_K];
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * BLOCK_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    int4v acc[M_TILES] = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int r = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gr = row0 + r;
            i8x8 value = {};
            if (gr < rows)
                value = load_i8x8(x8, (size_t)gr, n_in, bk + k);
            *reinterpret_cast<i8x8 *>(&sx[r][k]) = value;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            i8x8 value = {};
            if (gc < d_out)
                value = load_i8x8(w8, (size_t)gc, n_in, bk + k);
            *reinterpret_cast<i8x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const int bv = *reinterpret_cast<const int *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int av = *reinterpret_cast<const int *>(
                    &sx[mt * MFMA_M + (lane & 15)][k + k4]);
                acc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bv, acc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < d_out) {
        const float wscale = ws[col];
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = row0 + mt * MFMA_M + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = r0 + q;
                if (r < rows) {
                    const float v = (float)acc[mt][q] * x_s[r] * wscale;
                    float *dst = y + (size_t)r * d_out + col;
                    if (add) *dst += v; else *dst = v;
                }
            }
        }
    }
}

/* Batched per-head projection: activation/result are [rows, head, dim],
 * weights are independently row-major per head. */
__global__ void mfma_head_gemm(float *__restrict__ y,
                               const float *__restrict__ x,
                               const bf16_t *__restrict__ w,
                               int rows, int heads, int d_out, int n_in,
                               int x_row_stride, int x_head_dim,
                               int y_head_dim, size_t weight_head_stride) {
    __shared__ bf16_t sx[MFMA_M][MFMA_LDS_K];
    __shared__ bf16_t sw[MFMA_N][MFMA_LDS_K];

    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int head = (int)blockIdx.z;
    const int row0 = (int)blockIdx.y * MFMA_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    fp32v4 acc = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < MFMA_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(x[(size_t)gr * x_row_stride
                                    + (size_t)head * x_head_dim + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            bf16x8 value = {};
            if (gc < d_out)
                value = load_bf16x8(w + (size_t)head * weight_head_stride,
                                    (size_t)gc, n_in, bk + k);
            *reinterpret_cast<bf16x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                &sx[lane & 15][k + k4]);
            const bf16x4 bv = *reinterpret_cast<const bf16x4 *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                av, bv, acc, 0, 0, 0);
        }
        __syncthreads();
    }

    if (col < d_out) {
        const int r0 = row0 + 4 * (lane >> 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = r0 + q;
            if (r < rows)
                y[((size_t)r * heads + head) * y_head_dim + col] = acc[q];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Fused gate/up + SwiGLU kernels                                     */
/* ------------------------------------------------------------------ */

__global__ void mfma_gate_up_swiglu(float *__restrict__ y,
                                    const float *__restrict__ x,
                                    const bf16_t *__restrict__ gate,
                                    const bf16_t *__restrict__ up,
                                    int rows, int d_out, int n_in) {
    __shared__ bf16_t sx[MFMA_M][MFMA_LDS_K];
    __shared__ bf16_t sg[MFMA_N][MFMA_LDS_K];
    __shared__ bf16_t su[MFMA_N][MFMA_LDS_K];

    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * MFMA_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    fp32v4 gate_acc = {};
    fp32v4 up_acc = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < MFMA_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(x[(size_t)gr * n_in + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            bf16x8 gv = {}, uv = {};
            if (gc < d_out) {
                gv = load_bf16x8(gate, (size_t)gc, n_in, bk + k);
                uv = load_bf16x8(up, (size_t)gc, n_in, bk + k);
            }
            *reinterpret_cast<bf16x8 *>(&sg[c][k]) = gv;
            *reinterpret_cast<bf16x8 *>(&su[c][k]) = uv;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 bg = *reinterpret_cast<const bf16x4 *>(
                &sg[wave * 16 + (lane & 15)][k + k4]);
            const bf16x4 bu = *reinterpret_cast<const bf16x4 *>(
                &su[wave * 16 + (lane & 15)][k + k4]);
            const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                &sx[lane & 15][k + k4]);
            gate_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                av, bg, gate_acc, 0, 0, 0);
            up_acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                av, bu, up_acc, 0, 0, 0);
        }
        __syncthreads();
    }

    if (col < d_out) {
        const int r0 = row0 + 4 * (lane >> 4);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int r = r0 + q;
            if (r < rows) {
                const float gv = gate_acc[q];
                y[(size_t)r * d_out + col] =
                    (gv / (1.0f + expf(-gv))) * up_acc[q];
            }
        }
    }
}

template <int M_TILES>
__global__ void mfma_gate_up_swiglu_rows(
    float *__restrict__ y, const float *__restrict__ x,
    const bf16_t *__restrict__ gate, const bf16_t *__restrict__ up,
    int rows, int d_out, int n_in) {
    constexpr int BLOCK_M = MFMA_M * M_TILES;
    __shared__ bf16_t sx[BLOCK_M][MFMA_LDS_K];
    __shared__ bf16_t sg[MFMA_N][MFMA_LDS_K];
    __shared__ bf16_t su[MFMA_N][MFMA_LDS_K];
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * BLOCK_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    fp32v4 gate_acc[M_TILES] = {};
    fp32v4 up_acc[M_TILES] = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(x[(size_t)gr * n_in + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            bf16x8 gv = {}, uv = {};
            if (gc < d_out) {
                gv = load_bf16x8(gate, (size_t)gc, n_in, bk + k);
                uv = load_bf16x8(up, (size_t)gc, n_in, bk + k);
            }
            *reinterpret_cast<bf16x8 *>(&sg[c][k]) = gv;
            *reinterpret_cast<bf16x8 *>(&su[c][k]) = uv;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 bg = *reinterpret_cast<const bf16x4 *>(
                &sg[wave * 16 + (lane & 15)][k + k4]);
            const bf16x4 bu = *reinterpret_cast<const bf16x4 *>(
                &su[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                    &sx[mt * MFMA_M + (lane & 15)][k + k4]);
                gate_acc[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bg, gate_acc[mt], 0, 0, 0);
                up_acc[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bu, up_acc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < d_out) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = row0 + mt * MFMA_M + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = r0 + q;
                if (r < rows) {
                    const float gv = gate_acc[mt][q];
                    y[(size_t)r * d_out + col] =
                        (gv / (1.0f + expf(-gv))) * up_acc[mt][q];
                }
            }
        }
    }
}

template <int M_TILES>
__global__ void mfma_gate_up_swiglu_i8(
    float *__restrict__ y, const char *__restrict__ x8,
    const float *__restrict__ x_s,
    const char *__restrict__ gate8, const float *__restrict__ gate_ws,
    const char *__restrict__ up8, const float *__restrict__ up_ws,
    int rows, int d_out, int n_in) {
    constexpr int BLOCK_M = MFMA_M * M_TILES;
    __shared__ char sx[BLOCK_M][I8_LDS_K];
    __shared__ char sg[MFMA_N][I8_LDS_K];
    __shared__ char su[MFMA_N][I8_LDS_K];
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int row0 = (int)blockIdx.y * BLOCK_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    int4v gacc[M_TILES] = {};
    int4v uacc[M_TILES] = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int r = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gr = row0 + r;
            i8x8 value = {};
            if (gr < rows)
                value = load_i8x8(x8, (size_t)gr, n_in, bk + k);
            *reinterpret_cast<i8x8 *>(&sx[r][k]) = value;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            i8x8 gv = {}, uv = {};
            if (gc < d_out) {
                gv = load_i8x8(gate8, (size_t)gc, n_in, bk + k);
                uv = load_i8x8(up8, (size_t)gc, n_in, bk + k);
            }
            *reinterpret_cast<i8x8 *>(&sg[c][k]) = gv;
            *reinterpret_cast<i8x8 *>(&su[c][k]) = uv;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const int bg = *reinterpret_cast<const int *>(
                &sg[wave * 16 + (lane & 15)][k + k4]);
            const int bu = *reinterpret_cast<const int *>(
                &su[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int av = *reinterpret_cast<const int *>(
                    &sx[mt * MFMA_M + (lane & 15)][k + k4]);
                gacc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bg, gacc[mt], 0, 0, 0);
                uacc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bu, uacc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < d_out) {
        const float gscale = gate_ws[col];
        const float uscale = up_ws[col];
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = row0 + mt * MFMA_M + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = r0 + q;
                if (r < rows) {
                    const float as = x_s[r];
                    const float gv = (float)gacc[mt][q] * as * gscale;
                    const float uv = (float)uacc[mt][q] * as * uscale;
                    y[(size_t)r * d_out + col] =
                        (gv / (1.0f + expf(-gv))) * uv;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* One-wave-per-row GEMV kernels (tiny-batch fallback)                */
/* ------------------------------------------------------------------ */

__global__ void gemv(float *y, const float *x, const bf16_t *w,
                     const float *ws, int d_out, int n_in, int batch, int add) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * WAVES + wave;
    int b0 = (int)blockIdx.y * BATCH_TILE;
    if (row >= d_out || b0 >= batch) return;
    float sums[BATCH_TILE] = {};
    if (ws) {
        const char *wr8 = reinterpret_cast<const char *>(w)
            + (size_t)row * n_in;
        const float s = ws[row];
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float wi = s * (float)wr8[i];
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j)
                if (b0 + j < batch)
                    sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
        }
    } else {
        const bf16_t *wr = w + (size_t)row * n_in;
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float wi = gpu_bf16_to_f32(wr[i]);
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j)
                if (b0 + j < batch)
                    sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
        }
    }
#pragma unroll
    for (int j = 0; j < BATCH_TILE; ++j) {
        sums[j] = wave_sum(sums[j]);
        if (lane == 0 && b0 + j < batch) {
            float *dst = y + (size_t)(b0 + j) * d_out + row;
            if (add) *dst += sums[j]; else *dst = sums[j];
        }
    }
}

__global__ void dual_gemv(float *y0, const bf16_t *w0, const float *ws0,
                          int d_out0,
                          float *y1, const bf16_t *w1, const float *ws1,
                          int d_out1,
                          const float *x, int n_in, int batch) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * WAVES + wave;
    int total = d_out0 + d_out1;
    int b0 = (int)blockIdx.y * BATCH_TILE;
    if (row >= total || b0 >= batch) return;
    bool first = row < d_out0;
    int local = first ? row : row - d_out0;
    int d_out = first ? d_out0 : d_out1;
    const bf16_t *wbase = first ? w0 : w1;
    const float *ws = first ? ws0 : ws1;
    float sums[BATCH_TILE] = {};
    if (ws) {
        const char *wr8 = reinterpret_cast<const char *>(wbase)
            + (size_t)local * n_in;
        const float s = ws[local];
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float wi = s * (float)wr8[i];
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j)
                if (b0 + j < batch)
                    sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
        }
    } else {
        const bf16_t *wr = wbase + (size_t)local * n_in;
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float wi = gpu_bf16_to_f32(wr[i]);
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j)
                if (b0 + j < batch)
                    sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
        }
    }
#pragma unroll
    for (int j = 0; j < BATCH_TILE; ++j) {
        sums[j] = wave_sum(sums[j]);
        if (lane == 0 && b0 + j < batch) {
            float *base = first ? y0 : y1;
            base[(size_t)(b0 + j) * d_out + local] = sums[j];
        }
    }
}

__global__ void gate_up_swiglu_gemv(float *y, const float *x, const bf16_t *gate,
                                    const float *gate_ws, const bf16_t *up,
                                    const float *up_ws, int d_out, int n_in,
                                    int batch) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * WAVES + wave;
    int b0 = (int)blockIdx.y * BATCH_TILE;
    if (row >= d_out || b0 >= batch) return;
    float gs[BATCH_TILE] = {}, us[BATCH_TILE] = {};
    if (gate_ws) {
        const char *gr8 = reinterpret_cast<const char *>(gate)
            + (size_t)row * n_in;
        const char *ur8 = reinterpret_cast<const char *>(up)
            + (size_t)row * n_in;
        const float sg = gate_ws[row], su = up_ws[row];
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float gw = sg * (float)gr8[i], uw = su * (float)ur8[i];
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j) if (b0 + j < batch) {
                float xv = x[(size_t)(b0 + j) * n_in + i];
                gs[j] += xv * gw; us[j] += xv * uw;
            }
        }
    } else {
        const bf16_t *gr = gate + (size_t)row * n_in;
        const bf16_t *ur = up + (size_t)row * n_in;
        for (int i = lane; i < n_in; i += HIP_WAVE) {
            float gw = gpu_bf16_to_f32(gr[i]), uw = gpu_bf16_to_f32(ur[i]);
#pragma unroll
            for (int j = 0; j < BATCH_TILE; ++j) if (b0 + j < batch) {
                float xv = x[(size_t)(b0 + j) * n_in + i];
                gs[j] += xv * gw; us[j] += xv * uw;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < BATCH_TILE; ++j) {
        gs[j] = wave_sum(gs[j]); us[j] = wave_sum(us[j]);
        if (lane == 0 && b0 + j < batch)
            y[(size_t)(b0 + j) * d_out + row] =
                (gs[j] / (1.0f + expf(-gs[j]))) * us[j];
    }
}

/* ------------------------------------------------------------------ */
/* MoE kernels                                                        */
/* ------------------------------------------------------------------ */

/* vLLM-style expert alignment: routes of the same expert become contiguous,
 * padded to the MFMA row tile. Both target models have 64 routed experts. */
__global__ void align_routes(const int *__restrict__ topk_ids, int routes,
                             int experts, int block_m,
                             int *__restrict__ sorted_ids,
                             int *__restrict__ expert_ids,
                             int *__restrict__ num_post_pad) {
    __shared__ int counts[64];
    __shared__ int starts[64];
    __shared__ int padded_ends[64];
    __shared__ int write_positions[64];
    const int e = (int)threadIdx.x;
    const int max_padded = routes + experts * (block_m - 1);
    const int max_blocks = (max_padded + block_m - 1) / block_m;
    for (int i = e; i < max_blocks; i += (int)blockDim.x)
        expert_ids[i] = -1;
    if (e < experts) counts[e] = 0;
    __syncthreads();

    for (int i = e; i < routes; i += (int)blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();

    if (e == 0) {
        int offset = 0;
        for (int expert = 0; expert < experts; ++expert) {
            starts[expert] = offset;
            const int padded =
                ((counts[expert] + block_m - 1) / block_m) * block_m;
            padded_ends[expert] = offset + padded;
            for (int p = 0; p < padded; p += block_m)
                expert_ids[(offset + p) / block_m] = expert;
            offset += padded;
        }
        num_post_pad[0] = offset;
    }
    __syncthreads();

    if (e < experts) write_positions[e] = starts[e];
    __syncthreads();

    for (int i = e; i < routes; i += (int)blockDim.x) {
        const int expert = topk_ids[i];
        const int dst = atomicAdd(&write_positions[expert], 1);
        sorted_ids[dst] = i;
    }
    __syncthreads();

    if (e < experts)
        for (int dst = starts[e] + counts[e]; dst < padded_ends[e]; ++dst)
            sorted_ids[dst] = routes;
}

/* w8a8 grouped expert gate/up + SwiGLU (int8 MFMA, epilogue scales). */
template <int BLOCK_M, int M_TILES>
__global__ void grouped_expert_gate_up_i8(
    const char *__restrict__ x8, const float *__restrict__ x_s,
    const char *__restrict__ pool,
    const size_t *__restrict__ gate_offsets,
    const size_t *__restrict__ up_offsets,
    const size_t *__restrict__ gate_scale_offsets,
    const size_t *__restrict__ up_scale_offsets,
    const int *__restrict__ sorted_ids,
    const int *__restrict__ expert_ids,
    float *__restrict__ inter, bf16_t *__restrict__ inter_bf16,
    int routes, int hidden, int inter_dim, int top_k, int n_tiles) {
    __shared__ char sa[BLOCK_M * M_TILES][I8_LDS_K];
    __shared__ char sg[MFMA_N][I8_LDS_K];
    __shared__ char su[MFMA_N][I8_LDS_K];
    const int mblock = (int)blockIdx.x / n_tiles;
    const int ntile = (int)blockIdx.x - mblock * n_tiles;
    const int expert = expert_ids[mblock];
    if (expert < 0) return;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int k4 = 4 * (lane >> 4);
    const int col = ntile * MFMA_N + wave * 16 + (lane & 15);
    const char *wg8 = pool + gate_offsets[expert];
    const char *wu8 = pool + up_offsets[expert];
    const float *gws =
        reinterpret_cast<const float *>(pool + gate_scale_offsets[expert]);
    const float *uws =
        reinterpret_cast<const float *>(pool + up_scale_offsets[expert]);
    int4v gacc[M_TILES] = {};
    int4v uacc[M_TILES] = {};

    for (int bk = 0; bk < hidden; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * M_TILES * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int r = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int route = sorted_ids[mblock * BLOCK_M * M_TILES + r];
            i8x8 value = {};
            if (route < routes)
                value = load_i8x8(x8, (size_t)(route / top_k), hidden, bk + k);
            *reinterpret_cast<i8x8 *>(&sa[r][k]) = value;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = ntile * MFMA_N + c;
            i8x8 gv = {}, uv = {};
            if (gc < inter_dim) {
                gv = load_i8x8(wg8, (size_t)gc, hidden, bk + k);
                uv = load_i8x8(wu8, (size_t)gc, hidden, bk + k);
            }
            *reinterpret_cast<i8x8 *>(&sg[c][k]) = gv;
            *reinterpret_cast<i8x8 *>(&su[c][k]) = uv;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const int bg = *reinterpret_cast<const int *>(
                &sg[wave * 16 + (lane & 15)][k + k4]);
            const int bu = *reinterpret_cast<const int *>(
                &su[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int av = *reinterpret_cast<const int *>(
                    &sa[mt * 16 + (lane & 15)][k + k4]);
                gacc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bg, gacc[mt], 0, 0, 0);
                uacc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bu, uacc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < inter_dim) {
        const float gscale = gws[col];
        const float uscale = uws[col];
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = mt * 16 + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int route =
                    sorted_ids[mblock * BLOCK_M * M_TILES + r0 + q];
                if (route < routes) {
                    const float as = x_s[route / top_k];
                    const float gv = (float)gacc[mt][q] * as * gscale;
                    const float uv = (float)uacc[mt][q] * as * uscale;
                    const float value = (gv / (1.0f + expf(-gv))) * uv;
                    if (inter_bf16)
                        inter_bf16[(size_t)route * inter_dim + col] =
                            gpu_f32_to_bf16(value);
                    else
                        inter[(size_t)route * inter_dim + col] = value;
                }
            }
        }
    }
}

/* Grouped expert down on the bf16 MFMA path with in-flight int8 weight
 * dequantization — used when only gate/up runs a8 (GLM). */
template <int BLOCK_M, int M_TILES>
__global__ void grouped_expert_down(
    const float *__restrict__ inter,
    const bf16_t *__restrict__ inter_bf16,
    const char *__restrict__ pool,
    const size_t *__restrict__ down_offsets,
    const size_t *__restrict__ down_scale_offsets,
    const int *__restrict__ sorted_ids,
    const int *__restrict__ expert_ids,
    float *__restrict__ route_out, int routes, int hidden, int inter_dim,
    int n_tiles) {
    __shared__ bf16_t sa[BLOCK_M][MFMA_LDS_K];
    __shared__ bf16_t sw[MFMA_N][MFMA_LDS_K];
    const int mblock = (int)blockIdx.x / n_tiles;
    const int ntile = (int)blockIdx.x - mblock * n_tiles;
    const int expert = expert_ids[mblock];
    if (expert < 0) return;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int k4 = 4 * (lane >> 4);
    const int col = ntile * MFMA_N + wave * 16 + (lane & 15);
    const char *wd8 = pool + down_offsets[expert];
    const float *dws =
        reinterpret_cast<const float *>(pool + down_scale_offsets[expert]);
    fp32v4 acc[M_TILES] = {};

    for (int bk = 0; bk < inter_dim; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int route = sorted_ids[mblock * BLOCK_M + r];
            if (route < routes && bk + k < inter_dim)
                sa[r][k] = inter_bf16
                    ? inter_bf16[(size_t)route * inter_dim + bk + k]
                    : gpu_f32_to_bf16(
                        inter[(size_t)route * inter_dim + bk + k]);
            else
                sa[r][k] = (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = ntile * MFMA_N + c;
            bf16x8 value = {};
            if (gc < hidden)
                value = load_w8_dequant(wd8, dws, (size_t)gc, inter_dim, bk + k);
            *reinterpret_cast<bf16x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const bf16x4 bv = *reinterpret_cast<const bf16x4 *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                    &sa[mt * 16 + (lane & 15)][k + k4]);
                acc[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bv, acc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < hidden) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = mt * 16 + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int route = sorted_ids[mblock * BLOCK_M + r0 + q];
                if (route < routes)
                    route_out[(size_t)route * hidden + col] = acc[mt][q];
            }
        }
    }
}

/* w8a8 grouped expert down (DSV). */
template <int BLOCK_M, int M_TILES>
__global__ void grouped_expert_down_i8(
    const char *__restrict__ inter8, const float *__restrict__ inter_s,
    const char *__restrict__ pool,
    const size_t *__restrict__ down_offsets,
    const size_t *__restrict__ down_scale_offsets,
    const int *__restrict__ sorted_ids,
    const int *__restrict__ expert_ids,
    float *__restrict__ route_out, int routes, int hidden, int inter_dim,
    int n_tiles) {
    __shared__ char sa[BLOCK_M * M_TILES][I8_LDS_K];
    __shared__ char sw[MFMA_N][I8_LDS_K];
    const int mblock = (int)blockIdx.x / n_tiles;
    const int ntile = (int)blockIdx.x - mblock * n_tiles;
    const int expert = expert_ids[mblock];
    if (expert < 0) return;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int k4 = 4 * (lane >> 4);
    const int col = ntile * MFMA_N + wave * 16 + (lane & 15);
    const char *wd8 = pool + down_offsets[expert];
    const float *dws =
        reinterpret_cast<const float *>(pool + down_scale_offsets[expert]);
    int4v acc[M_TILES] = {};

    for (int bk = 0; bk < inter_dim; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * M_TILES * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int r = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int route = sorted_ids[mblock * BLOCK_M * M_TILES + r];
            i8x8 value = {};
            if (route < routes)
                value = load_i8x8(inter8, (size_t)route, inter_dim, bk + k);
            *reinterpret_cast<i8x8 *>(&sa[r][k]) = value;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = ntile * MFMA_N + c;
            i8x8 value = {};
            if (gc < hidden)
                value = load_i8x8(wd8, (size_t)gc, inter_dim, bk + k);
            *reinterpret_cast<i8x8 *>(&sw[c][k]) = value;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < MFMA_K; k += 16) {
            const int bv = *reinterpret_cast<const int *>(
                &sw[wave * 16 + (lane & 15)][k + k4]);
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int av = *reinterpret_cast<const int *>(
                    &sa[mt * 16 + (lane & 15)][k + k4]);
                acc[mt] = __builtin_amdgcn_mfma_i32_16x16x16i8(
                    av, bv, acc[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < hidden) {
        const float wscale = dws[col];
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = mt * 16 + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int route =
                    sorted_ids[mblock * BLOCK_M * M_TILES + r0 + q];
                if (route < routes)
                    route_out[(size_t)route * hidden + col] =
                        (float)acc[mt][q] * inter_s[route] * wscale;
            }
        }
    }
}

__global__ void reduce_routes_add(float *x, const float *route_out,
                                  const float *topk_weights, int batch,
                                  int hidden, int top_k) {
    const int b = (int)blockIdx.x;
    if (b >= batch) return;
    for (int d = (int)threadIdx.x; d < hidden; d += (int)blockDim.x) {
        float sum = 0.0f;
        for (int slot = 0; slot < top_k; ++slot) {
            const int route = b * top_k + slot;
            sum += topk_weights[route] * route_out[(size_t)route * hidden + d];
        }
        x[(size_t)b * hidden + d] += sum;
    }
}

/* Tiny-batch MoE fallback: one wave per output row over routed + shared. */
__global__ void routed_shared_gate_up(float *routed, float *shared, const float *x,
                                      const char *pool, const size_t *gate_offsets,
                                      const size_t *up_offsets,
                                      const size_t *gate_scale_offsets,
                                      const size_t *up_scale_offsets,
                                      const int *topk,
                                      const bf16_t *shared_gate,
                                      const float *shared_gate_ws,
                                      const bf16_t *shared_up,
                                      const float *shared_up_ws,
                                      int batch, int top_k, int inter, int shared_inter,
                                      int input_dim) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave,b=(int)blockIdx.y;
    int routed_rows=top_k*inter,total=routed_rows+shared_inter;
    if(row>=total||b>=batch)return;
    const bf16_t *gate,*up;const float *gws,*uws;float *dst;int j;
    if(row<routed_rows){int slot=row/inter;j=row-slot*inter;int e=topk[(size_t)b*top_k+slot];
        gate=(const bf16_t*)(pool+gate_offsets[e]);up=(const bf16_t*)(pool+up_offsets[e]);
        gws=gate_scale_offsets?(const float*)(pool+gate_scale_offsets[e]):nullptr;
        uws=up_scale_offsets?(const float*)(pool+up_scale_offsets[e]):nullptr;
        dst=routed+((size_t)b*top_k+slot)*inter;}
    else{j=row-routed_rows;gate=shared_gate;up=shared_up;
        gws=shared_gate_ws;uws=shared_up_ws;dst=shared+(size_t)b*shared_inter;}
    const float *xb=x+(size_t)b*input_dim;float gs=0.0f,us=0.0f;
    if(gws){
        const char *gr8=(const char*)gate+(size_t)j*input_dim;
        const char *ur8=(const char*)up+(size_t)j*input_dim;
        const float sg=gws[j],su=uws[j];
        for(int i=lane;i<input_dim;i+=HIP_WAVE){float xv=xb[i];
            gs+=xv*(sg*(float)gr8[i]);us+=xv*(su*(float)ur8[i]);}
    }else{
        const bf16_t *gr=gate+(size_t)j*input_dim,*ur=up+(size_t)j*input_dim;
        for(int i=lane;i<input_dim;i+=HIP_WAVE){float xv=xb[i];
            gs+=xv*gpu_bf16_to_f32(gr[i]);us+=xv*gpu_bf16_to_f32(ur[i]);}
    }
    gs=wave_sum(gs);us=wave_sum(us);if(lane==0)dst[j]=(gs/(1.0f+expf(-gs)))*us;
}

__global__ void routed_shared_down(float *out, const float *routed, const float *shared,
                                   const char *pool, const size_t *down_offsets,
                                   const size_t *down_scale_offsets,
                                   const int *topk, const float *topk_weights,
                                   const bf16_t *shared_down,
                                   const float *shared_down_ws, int batch, int top_k,
                                   int hidden, int inter, int shared_inter) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int d=(int)blockIdx.x*WAVES+wave,b=(int)blockIdx.y;
    if(d>=hidden||b>=batch)return;float total=0.0f;
    for(int slot=0;slot<top_k;++slot){int e=topk[(size_t)b*top_k+slot];
        const float *hp=routed+((size_t)b*top_k+slot)*inter;float sum=0.0f;
        if(down_scale_offsets){
            const char *wr8=(const char*)(pool+down_offsets[e])+(size_t)d*inter;
            const float s=((const float*)(pool+down_scale_offsets[e]))[d];
            for(int j=lane;j<inter;j+=HIP_WAVE)sum+=hp[j]*(s*(float)wr8[j]);
        }else{
            const bf16_t *wr=(const bf16_t*)(pool+down_offsets[e])+(size_t)d*inter;
            for(int j=lane;j<inter;j+=HIP_WAVE)sum+=hp[j]*gpu_bf16_to_f32(wr[j]);
        }
        sum=wave_sum(sum);if(lane==0)total+=topk_weights[(size_t)b*top_k+slot]*sum;}
    float ss=0.0f;
    if(shared_inter>0){
        const float *sh=shared+(size_t)b*shared_inter;
        if(shared_down_ws){
            const char *swr8=(const char*)shared_down+(size_t)d*shared_inter;
            const float s=shared_down_ws[d];
            for(int j=lane;j<shared_inter;j+=HIP_WAVE)ss+=sh[j]*(s*(float)swr8[j]);
        }else{
            const bf16_t *swr=shared_down+(size_t)d*shared_inter;
            for(int j=lane;j<shared_inter;j+=HIP_WAVE)ss+=sh[j]*gpu_bf16_to_f32(swr[j]);
        }
        ss=wave_sum(ss);}
    if(lane==0)out[(size_t)b*hidden+d]+=total+ss;
}

/* Router top-k. The wave64 variant covers both target models (64 experts). */
__global__ void router_topk_wave64(
    float *scores, const float *bias, int *indices, float *weights,
    int rows, int experts, int k, int use_sigmoid,
    int norm_topk, float routed_scale) {
    int row = (int)blockIdx.x;
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

/* ------------------------------------------------------------------ */
/* Elementwise kernels                                                */
/* ------------------------------------------------------------------ */

__global__ void embedding_kernel(float *x, const bf16_t *table, const int *tokens,
                                 int *generated, int generated_stride,
                                 int output_index, int hidden, int batch) {
    int b = (int)blockIdx.y;
    int i = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (b >= batch || i >= hidden) return;
    int token = tokens[b];
    x[(size_t)b * hidden + i] = gpu_bf16_to_f32(table[(size_t)token * hidden + i]);
    if (generated && blockIdx.x == 0 && threadIdx.x == 0)
        generated[(size_t)b * generated_stride + output_index] = token;
}

__global__ void rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                               int n, int batch, float eps) {
    __shared__ float scratch[THREADS];
    int b = (int)blockIdx.x;
    if (b >= batch) return;
    const float *xb = x + (size_t)b * n;
    float *yb = y + (size_t)b * n;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < n; i += THREADS) ss += xb[i] * xb[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int s = THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / (float)n + eps);
    for (int i = (int)threadIdx.x; i < n; i += THREADS)
        yb[i] = xb[i] * inv * gpu_bf16_to_f32(w[i]);
}

__global__ void argmax_kernel(const float *logits, int n, int *result, int batch) {
    __shared__ float bv[THREADS];__shared__ int bi[THREADS];int b=(int)blockIdx.x;
    if(b>=batch)return;const float *row=logits+(size_t)b*n;float vbest=-INFINITY;int ibest=0x7fffffff;
    for(int i=(int)threadIdx.x;i<n;i+=THREADS){float v=row[i];if(v>vbest||(v==vbest&&i<ibest)){vbest=v;ibest=i;}}
    bv[threadIdx.x]=vbest;bi[threadIdx.x]=ibest;__syncthreads();
    for(int s=THREADS/2;s;s>>=1){if((int)threadIdx.x<s){float ov=bv[threadIdx.x+s];int oi=bi[threadIdx.x+s];
        if(ov>bv[threadIdx.x]||(ov==bv[threadIdx.x]&&oi<bi[threadIdx.x])){bv[threadIdx.x]=ov;bi[threadIdx.x]=oi;}}__syncthreads();}
    if(threadIdx.x==0)result[b]=bi[0];
}

/* ================================================================== */
/* Generic op wrappers — the reusable API.                            */
/* Every wrapper receives explicit tensors + shapes; the if/else      */
/* ladders below pick a kernel template per shape. Tune them there.   */
/* ================================================================== */

/* Per-phase dynamic int8 activation scratch (allocated in warm_up). */
struct QuantScratch {
    char *act_i8 = nullptr;   /* [rows x max_in] */
    float *act_s = nullptr;   /* [rows]          */
};

/* Embedding gather: hidden=md::*::HIDDEN(2048) for both models/phases. */
inline void embedding(float *x, const bf16_t *table, const int *tokens,
                      int *generated, int generated_stride, int output_index,
                      int hidden, int rows, hipStream_t stream) {
    if (hidden != utils::constants::model::dsv::HIDDEN)
        unlisted_shape("ops::embedding", hidden, rows, 0);
    dim3 grid((unsigned)div_up(hidden, THREADS), (unsigned)rows);
    embedding_kernel<<<grid, THREADS, 0, stream>>>(
        x, table, tokens, generated, generated_stride, output_index,
        hidden, rows);
    HIP_LAUNCH_CHECK();
}

/* RMSNorm over `rows` rows of width `n`.
 * Shapes: n=md::*::HIDDEN(2048) input/post/final norm (dsv+glm, both phases);
 *         n=md::glm::Q_LORA(768) glm q_a norm (both phases). */
inline void rmsnorm(float *y, const float *x, const bf16_t *w, int n,
                    int rows, float eps, hipStream_t stream) {
    namespace md = utils::constants::model;
    if (n == md::dsv::HIDDEN) {
        /* dsv+glm hidden-width norms: N=md::*::HIDDEN(2048) */
        rmsnorm_kernel<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            y, x, w, n, rows, eps);
    } else if (n == md::glm::Q_LORA) {
        /* glm q_a norm: N=md::glm::Q_LORA(768) */
        rmsnorm_kernel<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            y, x, w, n, rows, eps);
    } else {
        unlisted_shape("ops::rmsnorm", n, rows, 0);
    }
    HIP_LAUNCH_CHECK();
}

/* Greedy sampling over the vocabulary (both phases). */
inline void argmax(const float *logits, int n, int *result, int rows,
                   hipStream_t stream) {
    namespace md = utils::constants::model;
    if (n == md::dsv::VOCAB) {
        /* dsv logits: N=md::dsv::VOCAB(102400), rows=batch */
        argmax_kernel<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            logits, n, result, rows);
    } else if (n == md::glm::VOCAB) {
        /* glm logits: N=md::glm::VOCAB(154880), rows=batch */
        argmax_kernel<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            logits, n, result, rows);
    } else {
        unlisted_shape("ops::argmax", n, rows, 0);
    }
    HIP_LAUNCH_CHECK();
}

inline void quantize_weight_rows(const bf16_t *src, char *dst, float *scales,
                                 int rows, int cols, hipStream_t stream) {
    quantize_rows_kernel<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
        src, dst, scales, rows, cols);
    HIP_LAUNCH_CHECK();
}

inline void quantize_act(const float *x, const QuantScratch &qs, int rows,
                         int cols, hipStream_t stream) {
    quantize_act_rows_f32<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
        x, qs.act_i8, qs.act_s, rows, cols);
    HIP_LAUNCH_CHECK();
}

inline dim3 gemv_grid(int rows, int batch) {
    return dim3((unsigned)div_up(rows, WAVES),
                (unsigned)div_up(batch, BATCH_TILE));
}

/* Launch one MFMA GEMM with the given configuration; `ws` non-null selects
 * the w8a8 int8-MFMA kernel (activations quantized on the fly). The shape
 * ladder in ops::gemm decides which configuration each tensor gets. */
inline void gemm_mfma(float *y, const float *x, const bf16_t *w,
                      const float *ws, int rows, int d_out, int n_in,
                      bool add, bool large_m, const QuantScratch &qs,
                      hipStream_t stream) {
    dim3 grid((unsigned)div_up(d_out, kt::MFMA_TILE_N),
              (unsigned)div_up(rows, large_m ? kt::PREFILL_ROWS
                                             : kt::MFMA_BATCH_TILE));
    if (ws) {
        quantize_act(x, qs, rows, n_in, stream);
        if (large_m)
            mfma_gemm_i8<kt::PREFILL_ROW_TILES><<<grid, 256, 0, stream>>>(
                y, qs.act_i8, qs.act_s, reinterpret_cast<const char *>(w),
                ws, rows, d_out, n_in, add ? 1 : 0);
        else
            mfma_gemm_i8<1><<<grid, 256, 0, stream>>>(
                y, qs.act_i8, qs.act_s, reinterpret_cast<const char *>(w),
                ws, rows, d_out, n_in, add ? 1 : 0);
    } else if (large_m) {
        mfma_gemm_rows<kt::PREFILL_ROW_TILES><<<grid, 256, 0, stream>>>(
            y, x, w, rows, d_out, n_in, add ? 1 : 0);
    } else {
        mfma_gemm<<<grid, 256, 0, stream>>>(
            y, x, w, rows, d_out, n_in, add ? 1 : 0);
    }
    HIP_LAUNCH_CHECK();
}

/* Launch a GEMM with the default configuration for its size regime:
 * rows < kt::MFMA_MIN_ROWS(4) -> one-wave GEMV (dequant int8 in-register);
 * rows < kt::ROW_TILES_MIN_ROWS(512) -> 16-row MFMA blocks (small decode);
 * otherwise -> 32-row MFMA blocks (packed prefill). */
inline void gemm_default(float *y, const float *x, const bf16_t *w,
                         const float *ws, int rows, int d_out, int n_in,
                         bool add, const QuantScratch &qs,
                         hipStream_t stream) {
    if (rows >= kt::MFMA_MIN_ROWS) {
        gemm_mfma(y, x, w, ws, rows, d_out, n_in, add,
                  rows >= kt::ROW_TILES_MIN_ROWS, qs, stream);
        return;
    }
    gemv<<<gemv_grid(d_out, rows), kt::GEMV_THREADS, 0, stream>>>(
        y, x, w, ws, d_out, n_in, rows, add ? 1 : 0);
    HIP_LAUNCH_CHECK();
}

/* Y[rows x d_out] = X[rows x n_in] @ W[d_out x n_in]^T (+= when add).
 *
 * ===================== shape ladder (model, tensor) =====================
 * Keyed on (d_out, n_in) against utils::constants::model — every shape the
 * two models produce has its own branch; an unlisted shape aborts. Every
 * branch launches the same default configuration (gemm_default) today —
 * tune each branch independently later. Within a branch:
 *   rows >= kt::LARGE_M_MIN_ROWS(1024) -> prefill (rows = packed tokens)
 *   rows <  kt::LARGE_M_MIN_ROWS       -> decode (rows = batch,
 *                                         GEMV below kt::MFMA_MIN_ROWS(4)) */
inline void gemm(float *y, const float *x, const bf16_t *w, const float *ws,
                 int rows, int d_out, int n_in, bool add,
                 const QuantScratch &qs, hipStream_t stream) {
    namespace md = utils::constants::model;
    const bool prefill_m = rows >= kt::LARGE_M_MIN_ROWS;
    if (d_out == md::dsv::N_EXPERTS && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL moe_gate router logits (dsv+glm): M=packed tokens (<=md::*::PREFILL_ROWS), N=md::*::N_EXPERTS(64), K=md::*::HIDDEN(2048), bf16 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE moe_gate router logits (dsv+glm): M=batch (<=md::*::DECODE_BATCH(512)), N=md::*::N_EXPERTS(64), K=md::*::HIDDEN(2048), bf16 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::Q_DIM && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL dsv q_proj: M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::Q_DIM(3072), K=md::dsv::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE dsv q_proj: M=batch (<=md::dsv::DECODE_BATCH(512)), N=md::dsv::Q_DIM(3072), K=md::dsv::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::KV_DIM && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL kv_a_proj (dsv+glm): M=packed tokens (<=md::*::PREFILL_ROWS), N=md::*::KV_DIM(576), K=md::*::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE kv_a_proj (dsv+glm): M=batch (<=md::*::DECODE_BATCH(512)), N=md::*::KV_DIM(576), K=md::*::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::Q_LORA && n_in == md::glm::HIDDEN) {
        if (prefill_m) {
            /* PREFILL glm q_a_proj: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::Q_LORA(768), K=md::glm::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm q_a_proj: M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::Q_LORA(768), K=md::glm::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::Q_DIM && n_in == md::glm::Q_LORA) {
        if (prefill_m) {
            /* PREFILL glm q_b_proj: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::Q_DIM(5120), K=md::glm::Q_LORA(768), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm q_b_proj: M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::Q_DIM(5120), K=md::glm::Q_LORA(768), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::HIDDEN && n_in == md::dsv::ATTN_OUT) {
        if (prefill_m) {
            /* PREFILL dsv o_proj: M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::HIDDEN(2048), K=md::dsv::ATTN_OUT(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE dsv o_proj: M=batch (<=md::dsv::DECODE_BATCH(512)), N=md::dsv::HIDDEN(2048), K=md::dsv::ATTN_OUT(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::HIDDEN && n_in == md::glm::ATTN_OUT) {
        if (prefill_m) {
            /* PREFILL glm o_proj: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::HIDDEN(2048), K=md::glm::ATTN_OUT(5120), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm o_proj: M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::HIDDEN(2048), K=md::glm::ATTN_OUT(5120), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::HIDDEN && n_in == md::dsv::SHARED_INTER) {
        if (prefill_m) {
            /* PREFILL dsv shared_down: M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::HIDDEN(2048), K=md::dsv::SHARED_INTER(2816), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE dsv shared_down: M=batch (<=md::dsv::DECODE_BATCH(512)), N=md::dsv::HIDDEN(2048), K=md::dsv::SHARED_INTER(2816), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::HIDDEN && n_in == md::glm::SHARED_INTER) {
        if (prefill_m) {
            /* PREFILL glm shared_down: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::HIDDEN(2048), K=md::glm::SHARED_INTER(1536), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm shared_down: M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::HIDDEN(2048), K=md::glm::SHARED_INTER(1536), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::HIDDEN && n_in == md::dsv::DENSE_INTER) {
        if (prefill_m) {
            /* PREFILL dsv dense_down (layer 0): M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::HIDDEN(2048), K=md::dsv::DENSE_INTER(10944), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE dsv dense_down (layer 0): M=batch (<=md::dsv::DECODE_BATCH(512)), N=md::dsv::HIDDEN(2048), K=md::dsv::DENSE_INTER(10944), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::HIDDEN && n_in == md::glm::DENSE_INTER) {
        if (prefill_m) {
            /* PREFILL glm dense_down (layer 0): M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::HIDDEN(2048), K=md::glm::DENSE_INTER(10240), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm dense_down (layer 0): M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::HIDDEN(2048), K=md::glm::DENSE_INTER(10240), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::dsv::VOCAB && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL dsv lm_head: M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::VOCAB(102400), K=md::dsv::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE dsv lm_head: M=batch (<=md::dsv::DECODE_BATCH(512)), N=md::dsv::VOCAB(102400), K=md::dsv::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else if (d_out == md::glm::VOCAB && n_in == md::glm::HIDDEN) {
        if (prefill_m) {
            /* PREFILL glm lm_head: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::VOCAB(154880), K=md::glm::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        } else {
            /* DECODE glm lm_head: M=batch (<=md::glm::DECODE_BATCH(512)), N=md::glm::VOCAB(154880), K=md::glm::HIDDEN(2048), int8+a8 */
            gemm_default(y, x, w, ws, rows, d_out, n_in, add, qs, stream);
        }
    } else {
        unlisted_shape("ops::gemm", d_out, n_in, rows);
    }
}

/* Two GEMMs sharing one activation. Ladder keyed on (out0, out1, n_in);
 * an unlisted pair aborts. Both phases pass through here. */
inline void gemm_dual(float *y0, const bf16_t *w0, const float *ws0, int out0,
                      float *y1, const bf16_t *w1, const float *ws1, int out1,
                      const float *x, int n_in, int rows,
                      const QuantScratch &qs, hipStream_t stream) {
    namespace md = utils::constants::model;
    const auto launch = [&] {
        if (rows >= kt::MFMA_MIN_ROWS) {
            gemm(y0, x, w0, ws0, rows, out0, n_in, false, qs, stream);
            gemm(y1, x, w1, ws1, rows, out1, n_in, false, qs, stream);
            return;
        }
        dual_gemv<<<gemv_grid(out0 + out1, rows), kt::GEMV_THREADS, 0,
            stream>>>(y0, w0, ws0, out0, y1, w1, ws1, out1, x, n_in, rows);
        HIP_LAUNCH_CHECK();
    };
    if (out0 == md::dsv::Q_DIM && out1 == md::dsv::KV_DIM &&
        n_in == md::dsv::HIDDEN) {
        /* dsv q_proj [M,3072,2048] + kv_a_proj [M,576,2048], int8+a8;
           M=batch (decode) or packed tokens (prefill) */
        launch();
    } else if (out0 == md::glm::Q_LORA && out1 == md::glm::KV_DIM &&
               n_in == md::glm::HIDDEN) {
        /* glm q_a_proj [M,768,2048] + kv_a_proj [M,576,2048], int8+a8;
           M=batch (decode) or packed tokens (prefill) */
        launch();
    } else {
        unlisted_shape("ops::gemm_dual", out0, out1, n_in);
    }
}

/* Fused gate/up + SwiGLU.
 * Shapes routed through here (both models all,w8a8; the bf16 branches stay
 * for one-line QuantPolicy rollbacks):
 *   dense FFN (layer 0): dsv [., 10944, 2048]; glm [., 10240, 2048]
 *   shared experts:      dsv [., 2816, 2048];  glm [., 1536, 2048] */
/* w8a8 gate/up over an activation that is already quantized into `qs`. */
inline void gate_up_swiglu_i8_pre(float *y, const QuantScratch &qs,
                                  const bf16_t *gate, const float *gws,
                                  const bf16_t *up, const float *uws,
                                  int rows, int d_out, int n_in,
                                  hipStream_t stream) {
    const bool large_m = rows >= kt::ROW_TILES_MIN_ROWS;
    dim3 grid((unsigned)div_up(d_out, kt::MFMA_TILE_N),
              (unsigned)div_up(rows, large_m ? kt::PREFILL_ROWS
                                             : kt::MFMA_BATCH_TILE));
    if (large_m)
        mfma_gate_up_swiglu_i8<kt::PREFILL_ROW_TILES><<<grid, 256, 0, stream>>>(
            y, qs.act_i8, qs.act_s,
            reinterpret_cast<const char *>(gate), gws,
            reinterpret_cast<const char *>(up), uws, rows, d_out, n_in);
    else
        mfma_gate_up_swiglu_i8<1><<<grid, 256, 0, stream>>>(
            y, qs.act_i8, qs.act_s,
            reinterpret_cast<const char *>(gate), gws,
            reinterpret_cast<const char *>(up), uws, rows, d_out, n_in);
    HIP_LAUNCH_CHECK();
}

/* Launch one fused gate/up + SwiGLU with the default configuration. */
inline void gate_up_mfma(float *y, const float *x,
                         const bf16_t *gate, const float *gws,
                         const bf16_t *up, const float *uws,
                         int rows, int d_out, int n_in,
                         const QuantScratch &qs, hipStream_t stream) {
    const bool large_m = rows >= kt::ROW_TILES_MIN_ROWS;
    if (gws && uws) { /* w8a8 */
        quantize_act(x, qs, rows, n_in, stream);
        gate_up_swiglu_i8_pre(y, qs, gate, gws, up, uws, rows, d_out, n_in,
                              stream);
        return;
    }
    dim3 grid((unsigned)div_up(d_out, kt::MFMA_TILE_N),
              (unsigned)div_up(rows, large_m ? kt::PREFILL_ROWS
                                             : kt::MFMA_BATCH_TILE));
    if (large_m)
        mfma_gate_up_swiglu_rows<kt::PREFILL_ROW_TILES><<<grid, 256, 0,
            stream>>>(y, x, gate, up, rows, d_out, n_in);
    else
        mfma_gate_up_swiglu<<<grid, 256, 0, stream>>>(
            y, x, gate, up, rows, d_out, n_in);
    HIP_LAUNCH_CHECK();
}

/* Fused gate/up + SwiGLU dispatch. Same convention as ops::gemm: keyed on
 * (d_out, n_in), every real shape has an explicit branch (unlisted shapes
 * abort), every branch launches the default configuration today. dtype is
 * int8+a8 when scales are non-null, bf16 after a QuantPolicy rollback. */
inline void gate_up_default(float *y, const float *x,
                            const bf16_t *gate, const float *gws,
                            const bf16_t *up, const float *uws, int rows,
                            int d_out, int n_in, const QuantScratch &qs,
                            hipStream_t stream) {
    if (rows >= kt::MFMA_MIN_ROWS) {
        gate_up_mfma(y, x, gate, gws, up, uws, rows, d_out, n_in, qs,
                     stream);
        return;
    }
    gate_up_swiglu_gemv<<<gemv_grid(d_out, rows), kt::GEMV_THREADS, 0,
        stream>>>(y, x, gate, gws, up, uws, d_out, n_in, rows);
    HIP_LAUNCH_CHECK();
}

inline void gate_up_swiglu(float *y, const float *x,
                           const bf16_t *gate, const float *gws,
                           const bf16_t *up, const float *uws, int rows,
                           int d_out, int n_in, const QuantScratch &qs,
                           hipStream_t stream) {
    namespace md = utils::constants::model;
    const bool prefill_m = rows >= kt::LARGE_M_MIN_ROWS;
    if (d_out == md::dsv::DENSE_INTER && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL dsv dense FFN gate/up (layer 0): M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::DENSE_INTER(10944), K=md::dsv::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        } else {
            /* DECODE dsv dense FFN gate/up (layer 0): M=batch (<=md::dsv::DECODE_BATCH(512), GEMV below kt::MFMA_MIN_ROWS(4)), N=md::dsv::DENSE_INTER(10944), K=md::dsv::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        }
    } else if (d_out == md::glm::DENSE_INTER && n_in == md::glm::HIDDEN) {
        if (prefill_m) {
            /* PREFILL glm dense FFN gate/up (layer 0): M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::DENSE_INTER(10240), K=md::glm::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        } else {
            /* DECODE glm dense FFN gate/up (layer 0): M=batch (<=md::glm::DECODE_BATCH(512), GEMV below kt::MFMA_MIN_ROWS(4)), N=md::glm::DENSE_INTER(10240), K=md::glm::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        }
    } else if (d_out == md::dsv::SHARED_INTER && n_in == md::dsv::HIDDEN) {
        if (prefill_m) {
            /* PREFILL dsv shared experts gate/up: M=packed tokens (<=md::dsv::PREFILL_ROWS(65520)), N=md::dsv::SHARED_INTER(2816), K=md::dsv::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        } else {
            /* DECODE dsv shared experts gate/up: M=batch (<=md::dsv::DECODE_BATCH(512), GEMV below kt::MFMA_MIN_ROWS(4)), N=md::dsv::SHARED_INTER(2816), K=md::dsv::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        }
    } else if (d_out == md::glm::SHARED_INTER && n_in == md::glm::HIDDEN) {
        if (prefill_m) {
            /* PREFILL glm shared expert gate/up: M=packed tokens (<=md::glm::PREFILL_ROWS(32768)), N=md::glm::SHARED_INTER(1536), K=md::glm::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        } else {
            /* DECODE glm shared expert gate/up: M=batch (<=md::glm::DECODE_BATCH(512), GEMV below kt::MFMA_MIN_ROWS(4)), N=md::glm::SHARED_INTER(1536), K=md::glm::HIDDEN(2048) */
            gate_up_default(y, x, gate, gws, up, uws, rows, d_out, n_in, qs, stream);
        }
    } else {
        unlisted_shape("ops::gate_up_swiglu", d_out, n_in, rows);
    }
}

/* Launch one per-head projection with the default configuration. */
inline void head_gemm_mfma(float *y, const float *x, const bf16_t *w,
                           int rows, int heads, int d_out, int n_in,
                           int x_row_stride, int x_head_dim, int y_head_dim,
                           size_t head_stride, hipStream_t stream) {
    dim3 grid((unsigned)div_up(d_out, kt::MFMA_TILE_N),
              (unsigned)div_up(rows, kt::MFMA_BATCH_TILE),
              (unsigned)heads);
    mfma_head_gemm<<<grid, 256, 0, stream>>>(
        y, x, w, rows, heads, d_out, n_in, x_row_stride, x_head_dim,
        y_head_dim, head_stride);
    HIP_LAUNCH_CHECK();
}

/* Per-head projection, bf16 MFMA: y[row,h,:]{d_out} = w[h] @ x-slice{n_in}.
 * x-slice = x[row*x_row_stride + h*x_head_dim : +n_in] — x_head_dim=0 means
 * every head reads the same per-row vector (kv_b decompression).
 * ============== shape ladder (model, tensor, phase) ============== */
inline void head_gemm(float *y, const float *x, const bf16_t *w,
                      int rows, int heads, int d_out, int n_in,
                      int x_row_stride, int x_head_dim, int y_head_dim,
                      size_t head_stride, hipStream_t stream) {
    namespace md = utils::constants::model;
    if (heads == md::dsv::N_HEADS && d_out == md::dsv::KV_LORA &&
        n_in == md::dsv::QK_NOPE) {
        /* DECODE dsv q-absorb: M=batch (<=md::dsv::DECODE_BATCH(512)),
           H=16, N=md::dsv::KV_LORA(512), K=md::dsv::QK_NOPE(128), bf16 */
        head_gemm_mfma(y, x, w, rows, heads, d_out, n_in, x_row_stride,
                       x_head_dim, y_head_dim, head_stride, stream);
    } else if (heads == md::glm::N_HEADS && d_out == md::glm::KV_LORA &&
               n_in == md::glm::QK_NOPE) {
        /* DECODE glm q-absorb: M=batch (<=md::glm::DECODE_BATCH(512)),
           H=20, N=md::glm::KV_LORA(512), K=md::glm::QK_NOPE(192), bf16 */
        head_gemm_mfma(y, x, w, rows, heads, d_out, n_in, x_row_stride,
                       x_head_dim, y_head_dim, head_stride, stream);
    } else if (heads == md::dsv::N_HEADS && d_out == md::dsv::V_HEAD &&
               n_in == md::dsv::KV_LORA) {
        /* dsv [M,16h,128,512]: DECODE value-up (M=batch) and PREFILL kv_b
           decompression of K_nope AND V (M=packed tokens; QK_NOPE==V_HEAD
           ==128 so both land here), bf16 */
        head_gemm_mfma(y, x, w, rows, heads, d_out, n_in, x_row_stride,
                       x_head_dim, y_head_dim, head_stride, stream);
    } else if (heads == md::glm::N_HEADS && d_out == md::glm::QK_NOPE &&
               n_in == md::glm::KV_LORA) {
        /* PREFILL glm kv_b K_nope decompression: M=packed tokens
           (<=md::glm::PREFILL_ROWS(32768)), H=20,
           N=md::glm::QK_NOPE(192), K=md::glm::KV_LORA(512), bf16 */
        head_gemm_mfma(y, x, w, rows, heads, d_out, n_in, x_row_stride,
                       x_head_dim, y_head_dim, head_stride, stream);
    } else if (heads == md::glm::N_HEADS && d_out == md::glm::V_HEAD &&
               n_in == md::glm::KV_LORA) {
        /* glm [M,20h,256,512]: DECODE value-up (M=batch) and PREFILL kv_b
           V decompression (M=packed tokens), bf16 */
        head_gemm_mfma(y, x, w, rows, heads, d_out, n_in, x_row_stride,
                       x_head_dim, y_head_dim, head_stride, stream);
    } else {
        unlisted_shape("ops::head_gemm", heads, d_out, n_in);
    }
}

/* Router softmax/sigmoid + top-k. Both models have 64 experts (one wave). */
inline void router_topk(float *scores, const float *bias, int *indices,
                        float *weights, int rows, int experts, int k,
                        int use_sigmoid, int norm_topk, float routed_scale,
                        hipStream_t stream) {
    namespace md = utils::constants::model;
    if (experts == md::dsv::N_EXPERTS && k == md::dsv::TOP_K) {
        /* dsv router: E=md::dsv::N_EXPERTS(64), top-k=md::dsv::TOP_K(6),
           softmax; rows=batch (decode) or packed tokens (prefill) */
        router_topk_wave64<<<rows, kt::ROUTER_WAVE_THREADS, 0, stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    } else if (experts == md::glm::N_EXPERTS && k == md::glm::TOP_K) {
        /* glm router: E=md::glm::N_EXPERTS(64), top-k=md::glm::TOP_K(4),
           sigmoid+bias; rows=batch (decode) or packed tokens (prefill) */
        router_topk_wave64<<<rows, kt::ROUTER_WAVE_THREADS, 0, stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    } else {
        unlisted_shape("ops::router_topk", experts, k, rows);
    }
    HIP_LAUNCH_CHECK();
}

/* Routed + shared MoE FFN over `rows` tokens; adds the result into x.
 * Grouped (MFMA) path from MFMA_MIN_ROWS tokens, one-wave fallback below.
 *   dsv: E=64 top-6, inter=md::dsv::MOE_INTER(1408), shared 2816, all int8+a8
 *   glm: E=64 top-4, inter=md::glm::MOE_INTER(1536), shared 1536, all int8+a8
 * (the bf16/w8 branches below stay live for QuantPolicy rollbacks) */
inline void moe_ffn(const Config &c, const DeviceLayer &d, int rows,
                    float *x, const float *xn, float *hb, float *router,
                    int *topk, float *topk_weights, int *sorted_ids,
                    int *expert_ids, int *num_post_pad, float *routed_hidden,
                    bf16_t *routed_hidden_bf16, float *route_out,
                    const QuantScratch &qs, char *routed_i8, float *routed_s,
                    hipStream_t stream) {
    const int H = c.hidden_size;
    const int E = c.n_routed_experts;
    const int K = c.n_experts_per_tok;
    const int inter = c.moe_inter_size;
    const int shared = c.n_shared_experts * inter;
    {
        namespace md = utils::constants::model;
        const bool dsv_shape = E == md::dsv::N_EXPERTS &&
            K == md::dsv::TOP_K && inter == md::dsv::MOE_INTER &&
            shared == md::dsv::SHARED_INTER;
        const bool glm_shape = E == md::glm::N_EXPERTS &&
            K == md::glm::TOP_K && inter == md::glm::MOE_INTER &&
            shared == md::glm::SHARED_INTER;
        if (!dsv_shape && !glm_shape)
            unlisted_shape("ops::moe_ffn", E, K, inter);
    }

    gemm(router, xn, d.moe_gate, nullptr, rows, E, H, false, qs, stream);
    router_topk(router, d.moe_bias, topk, topk_weights, rows, E, K,
                c.router_sigmoid, c.norm_topk, c.routed_scaling, stream);

    if (rows >= kt::MFMA_MIN_ROWS) {
        constexpr int block_m = kt::GROUPED_ROUTE_TILE;
        const int routes = rows * K;
        const int max_padded = routes + E * (block_m - 1);
        const int max_blocks = div_up(max_padded, block_m);
        align_routes<<<1, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            topk, routes, E, block_m, sorted_ids, expert_ids, num_post_pad);
        HIP_LAUNCH_CHECK();

        /* gate/up always runs w8a8 (both models quantize expert gate/up).
         * ---- shape ladder (model, phase) — same default everywhere ----
         *  dsv: routes=M*6, N=md::dsv::MOE_INTER(1408), K=HIDDEN(2048)
         *  glm: routes=M*4, N=md::glm::MOE_INTER(1536), K=HIDDEN(2048)
         *  DECODE M=batch (<=md::*::DECODE_BATCH(512)); PREFILL M=packed tokens. */
        quantize_act(xn, qs, rows, H, stream);
        const int up_tiles = div_up(inter, kt::MFMA_TILE_N);
        grouped_expert_gate_up_i8<block_m, 1><<<
            max_blocks * up_tiles, 256, 0, stream>>>(
                qs.act_i8, qs.act_s, d.pool,
                d.expert_gate_offsets, d.expert_up_offsets,
                d.expert_gate_scale_offsets, d.expert_up_scale_offsets,
                sorted_ids, expert_ids, routed_hidden, routed_hidden_bf16,
                routes, H, inter, K, up_tiles);
        HIP_LAUNCH_CHECK();

        if (d.shared_gate_s) /* dsv: xn is already quantized in `qs` above */
            gate_up_swiglu_i8_pre(hb, qs, d.shared_gate, d.shared_gate_s,
                                  d.shared_up, d.shared_up_s, rows, shared,
                                  H, stream);
        else /* glm: shared experts stay bf16 */
            gate_up_swiglu(hb, xn, d.shared_gate, nullptr,
                           d.shared_up, nullptr, rows, shared, H, qs, stream);

        const int down_tiles = div_up(H, kt::MFMA_TILE_N);
        if (routed_i8) {
            /* dsv: expert down on the int8 MFMA path */
            if (routed_hidden_bf16)
                quantize_act_rows_bf16<<<routes, kt::ELEMENTWISE_THREADS, 0,
                    stream>>>(routed_hidden_bf16, routed_i8, routed_s,
                              routes, inter);
            else
                quantize_act_rows_f32<<<routes, kt::ELEMENTWISE_THREADS, 0,
                    stream>>>(routed_hidden, routed_i8, routed_s,
                              routes, inter);
            HIP_LAUNCH_CHECK();
            grouped_expert_down_i8<block_m, 1><<<
                max_blocks * down_tiles, 256, 0, stream>>>(
                    routed_i8, routed_s, d.pool,
                    d.expert_down_offsets, d.expert_down_scale_offsets,
                    sorted_ids, expert_ids, route_out,
                    routes, H, inter, down_tiles);
        } else {
            /* glm: expert down on bf16 MFMA with in-flight w8 dequant */
            grouped_expert_down<block_m, 1><<<
                max_blocks * down_tiles, 256, 0, stream>>>(
                    routed_hidden, routed_hidden_bf16, d.pool,
                    d.expert_down_offsets, d.expert_down_scale_offsets,
                    sorted_ids, expert_ids, route_out,
                    routes, H, inter, down_tiles);
        }
        HIP_LAUNCH_CHECK();
        reduce_routes_add<<<rows, kt::ELEMENTWISE_THREADS, 0, stream>>>(
            x, route_out, topk_weights, rows, H, K);
        HIP_LAUNCH_CHECK();
    } else {
        dim3 gate_grid((unsigned)div_up(K * inter, WAVES), (unsigned)rows);
        routed_shared_gate_up<<<gate_grid, kt::GEMV_THREADS, 0, stream>>>(
            routed_hidden, hb, xn, d.pool,
            d.expert_gate_offsets, d.expert_up_offsets,
            d.expert_gate_scale_offsets, d.expert_up_scale_offsets, topk,
            d.shared_gate, d.shared_gate_s, d.shared_up, d.shared_up_s,
            rows, K, inter, 0, H);
        HIP_LAUNCH_CHECK();
        gate_up_swiglu(hb, xn, d.shared_gate, d.shared_gate_s,
                       d.shared_up, d.shared_up_s, rows, shared, H, qs,
                       stream);
        dim3 down_grid((unsigned)div_up(H, WAVES), (unsigned)rows);
        routed_shared_down<<<down_grid, kt::GEMV_THREADS, 0, stream>>>(
            x, routed_hidden, hb, d.pool, d.expert_down_offsets,
            d.expert_down_scale_offsets,
            topk, topk_weights, d.shared_down, d.shared_down_s,
            rows, K, H, inter, 0);
        HIP_LAUNCH_CHECK();
    }
    gemm(x, hb, d.shared_down, d.shared_down_s, rows, H, shared, true, qs,
         stream);
}

/* Dense (non-MoE) FFN for the first_k_dense layers; adds into x. */
inline void dense_ffn(const Config &c, const DeviceLayer &d, int rows,
                      float *x, const float *xn, float *hb,
                      const QuantScratch &qs, hipStream_t stream) {
    const int H = c.hidden_size;
    const int inter = c.dense_inter_size;
    gate_up_swiglu(hb, xn, d.dense_gate, d.dense_gate_s,
                   d.dense_up, d.dense_up_s, rows, inter, H, qs, stream);
    gemm(x, hb, d.dense_down, d.dense_down_s, rows, H, inter, true, qs,
         stream);
}

} // namespace ops

#endif /* MLA_OPS_H */
