#include "getp_kernels.h"
#include "getp_kernels_device.h"

namespace batch_decode_kernels {

constexpr int THREADS = kernel_tuning::GEMV_THREADS;
constexpr int WAVES = THREADS / HIP_WAVE;

constexpr int BATCH_TILE = 2;

using bf16x4 = short __attribute__((ext_vector_type(4)));
using bf16x8 = short __attribute__((ext_vector_type(8)));
using float4v = float __attribute__((ext_vector_type(4)));

constexpr int MFMA_M = 16;
constexpr int MFMA_N = 64;
constexpr int MFMA_K = 64;
constexpr int MFMA_LDS_K = MFMA_K + 4;

__device__ __forceinline__ bf16_t gpu_f32_to_bf16(float value) {
    return (bf16_t)(__float_as_uint(value) >> 16);
}

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
    float4v acc = {};

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
            if (gc < d_out && bk + k + 8 <= n_in)
                value = *reinterpret_cast<const bf16x8 *>(
                    w + (size_t)gc * n_in + bk + k);
            else if (gc < d_out) {
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (bk + k + q < n_in)
                        value[q] = reinterpret_cast<const short *>(w)[
                            (size_t)gc * n_in + bk + k + q];
            }
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

/* Large-M specialization for flattened prefill.  Multiple 16-row MFMA tiles
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
    float4v acc[M_TILES] = {};

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
            if (gc < d_out && bk + k + 8 <= n_in)
                value = *reinterpret_cast<const bf16x8 *>(
                    w + (size_t)gc * n_in + bk + k);
            else if (gc < d_out) {
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (bk + k + q < n_in)
                        value[q] = reinterpret_cast<const short *>(w)[
                            (size_t)gc * n_in + bk + k + q];
            }
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

template __global__ void mfma_gemm_rows<2>(
    float *, const float *, const bf16_t *, int, int, int, int);

/* Batched per-head projection.  The activation and result are both
 * [batch, head, dim], while weights are independently row-major per head. */
__global__ void mfma_head_gemm(float *__restrict__ y,
                               const float *__restrict__ x,
                               const bf16_t *__restrict__ w,
                               int rows, int heads, int d_out, int n_in,
                               int x_head_dim, int y_head_dim,
                               size_t weight_head_stride) {
    __shared__ bf16_t sx[MFMA_M][MFMA_LDS_K];
    __shared__ bf16_t sw[MFMA_N][MFMA_LDS_K];

    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int head = (int)blockIdx.z;
    const int row0 = (int)blockIdx.y * MFMA_M;
    const int col0 = (int)blockIdx.x * MFMA_N;
    const int col = col0 + wave * 16 + (lane & 15);
    const int k4 = 4 * (lane >> 4);
    float4v acc = {};

    for (int bk = 0; bk < n_in; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < MFMA_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int gr = row0 + r;
            sx[r][k] = gr < rows && bk + k < n_in
                ? gpu_f32_to_bf16(
                    x[((size_t)gr * heads + head) * x_head_dim + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = col0 + c;
            bf16x8 value = {};
            if (gc < d_out && bk + k + 8 <= n_in)
                value = *reinterpret_cast<const bf16x8 *>(
                    w + (size_t)head * weight_head_stride
                      + (size_t)gc * n_in + bk + k);
            else if (gc < d_out) {
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    if (bk + k + q < n_in)
                        value[q] = reinterpret_cast<const short *>(w)[
                            (size_t)head * weight_head_stride
                          + (size_t)gc * n_in + bk + k + q];
            }
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

/* Gate and up projections share the same activation tile.  Computing both in
 * one block also writes only the final SwiGLU activation, avoiding two fp32
 * intermediates and a separate elementwise kernel. */
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
    float4v gate_acc[MFMA_M / 16] = {};
    float4v up_acc[MFMA_M / 16] = {};

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
            if (gc < d_out && bk + k + 8 <= n_in) {
                gv = *reinterpret_cast<const bf16x8 *>(
                    gate + (size_t)gc * n_in + bk + k);
                uv = *reinterpret_cast<const bf16x8 *>(
                    up + (size_t)gc * n_in + bk + k);
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
            for (int mt = 0; mt < MFMA_M / 16; ++mt) {
                const bf16x4 av = *reinterpret_cast<const bf16x4 *>(
                    &sx[mt * 16 + (lane & 15)][k + k4]);
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
        for (int mt = 0; mt < MFMA_M / 16; ++mt) {
            const int r0 = row0 + mt * 16 + 4 * (lane >> 4);
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
    float4v gate_acc[M_TILES] = {};
    float4v up_acc[M_TILES] = {};

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
            if (gc < d_out && bk + k + 8 <= n_in) {
                gv = *reinterpret_cast<const bf16x8 *>(
                    gate + (size_t)gc * n_in + bk + k);
                uv = *reinterpret_cast<const bf16x8 *>(
                    up + (size_t)gc * n_in + bk + k);
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

template __global__ void mfma_gate_up_swiglu_rows<2>(
    float *, const float *, const bf16_t *, const bf16_t *, int, int, int);

__global__ void swiglu(float *y, const float *gate, const float *up, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        y[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

/* vLLM-style expert alignment: routes for the same expert become contiguous
 * and are padded to the selected MFMA row tile.  Both target models have 64
 * routed experts. */
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

template <int BLOCK_M, int M_TILES>
__global__ void grouped_expert_gate_up(
    const float *__restrict__ x, const char *__restrict__ pool,
    const size_t *__restrict__ gate_offsets,
    const size_t *__restrict__ up_offsets,
    const int *__restrict__ sorted_ids,
    const int *__restrict__ expert_ids,
    float *__restrict__ inter, bf16_t *__restrict__ inter_bf16,
    int routes, int hidden, int inter_dim, int top_k, int n_tiles) {
    __shared__ bf16_t sa[BLOCK_M][MFMA_LDS_K];
    __shared__ bf16_t sg[MFMA_N][MFMA_LDS_K];
    __shared__ bf16_t su[MFMA_N][MFMA_LDS_K];
    const int mblock = (int)blockIdx.x / n_tiles;
    const int ntile = (int)blockIdx.x - mblock * n_tiles;
    const int expert = expert_ids[mblock];
    if (expert < 0) return;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int k4 = 4 * (lane >> 4);
    const int col = ntile * MFMA_N + wave * 16 + (lane & 15);
    const bf16_t *wg = reinterpret_cast<const bf16_t *>(pool + gate_offsets[expert]);
    const bf16_t *wu = reinterpret_cast<const bf16_t *>(pool + up_offsets[expert]);
    float4v gate[M_TILES] = {};
    float4v up[M_TILES] = {};

    for (int bk = 0; bk < hidden; bk += MFMA_K) {
        for (int i = (int)threadIdx.x; i < BLOCK_M * MFMA_K;
             i += (int)blockDim.x) {
            const int r = i / MFMA_K;
            const int k = i - r * MFMA_K;
            const int route = sorted_ids[mblock * BLOCK_M + r];
            sa[r][k] = route < routes && bk + k < hidden
                ? gpu_f32_to_bf16(x[(size_t)(route / top_k) * hidden + bk + k])
                : (bf16_t)0;
        }
        for (int i = (int)threadIdx.x; i < MFMA_N * (MFMA_K / 8);
             i += (int)blockDim.x) {
            const int c = i / (MFMA_K / 8);
            const int k = (i % (MFMA_K / 8)) * 8;
            const int gc = ntile * MFMA_N + c;
            bf16x8 gv = {}, uv = {};
            if (gc < inter_dim && bk + k + 8 <= hidden) {
                gv = *reinterpret_cast<const bf16x8 *>(
                    wg + (size_t)gc * hidden + bk + k);
                uv = *reinterpret_cast<const bf16x8 *>(
                    wu + (size_t)gc * hidden + bk + k);
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
                    &sa[mt * 16 + (lane & 15)][k + k4]);
                gate[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bg, gate[mt], 0, 0, 0);
                up[mt] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    av, bu, up[mt], 0, 0, 0);
            }
        }
        __syncthreads();
    }

    if (col < inter_dim) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int r0 = mt * 16 + 4 * (lane >> 4);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int route = sorted_ids[mblock * BLOCK_M + r0 + q];
                if (route < routes) {
                    const float gv = gate[mt][q];
                    const float value =
                        (gv / (1.0f + expf(-gv))) * up[mt][q];
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

template <int BLOCK_M, int M_TILES>
__global__ void grouped_expert_down(
    const float *__restrict__ inter,
    const bf16_t *__restrict__ inter_bf16,
    const char *__restrict__ pool,
    const size_t *__restrict__ down_offsets,
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
    const bf16_t *wd = reinterpret_cast<const bf16_t *>(pool + down_offsets[expert]);
    float4v acc[M_TILES] = {};

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
            if (gc < hidden && bk + k + 8 <= inter_dim)
                value = *reinterpret_cast<const bf16x8 *>(
                    wd + (size_t)gc * inter_dim + bk + k);
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

template __global__ void grouped_expert_gate_up<16, 1>(
    const float *, const char *, const size_t *, const size_t *, const int *,
    const int *, float *, bf16_t *, int, int, int, int, int);
template __global__ void grouped_expert_down<16, 1>(
    const float *, const bf16_t *, const char *, const size_t *, const int *,
    const int *, float *, int, int, int, int);

__global__ void embedding(float *x, const bf16_t *table, const int *tokens,
                          int *generated, int generated_stride, int output_index,
                          int hidden, int batch) {
    int b = (int)blockIdx.y;
    int i = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (b >= batch || i >= hidden) return;
    int token = tokens[b];
    x[(size_t)b * hidden + i] = gpu_bf16_to_f32(table[(size_t)token * hidden + i]);
    if (generated && blockIdx.x == 0 && threadIdx.x == 0)
        generated[(size_t)b * generated_stride + output_index] = token;
}

__global__ void rmsnorm(float *y, const float *x, const bf16_t *w,
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

__global__ void gemv(float *y, const float *x, const bf16_t *w,
                     int d_out, int n_in, int batch, int add) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * WAVES + wave;
    int b0 = (int)blockIdx.y * BATCH_TILE;
    if (row >= d_out || b0 >= batch) return;
    float sums[BATCH_TILE] = {};
    const bf16_t *wr = w + (size_t)row * n_in;
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float wi = gpu_bf16_to_f32(wr[i]);
#pragma unroll
        for (int j = 0; j < BATCH_TILE; ++j)
            if (b0 + j < batch) sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
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

__global__ void dual_gemv(float *y0, const bf16_t *w0, int d_out0,
                          float *y1, const bf16_t *w1, int d_out1,
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
    const bf16_t *wr = (first ? w0 : w1) + (size_t)local * n_in;
    float sums[BATCH_TILE] = {};
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float wi = gpu_bf16_to_f32(wr[i]);
#pragma unroll
        for (int j = 0; j < BATCH_TILE; ++j)
            if (b0 + j < batch) sums[j] += x[(size_t)(b0 + j) * n_in + i] * wi;
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

__global__ void gate_up_swiglu(float *y, const float *x, const bf16_t *gate,
                               const bf16_t *up, int d_out, int n_in, int batch) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * WAVES + wave;
    int b0 = (int)blockIdx.y * BATCH_TILE;
    if (row >= d_out || b0 >= batch) return;
    const bf16_t *gr = gate + (size_t)row * n_in;
    const bf16_t *ur = up + (size_t)row * n_in;
    float gs[BATCH_TILE] = {}, us[BATCH_TILE] = {};
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float gw = gpu_bf16_to_f32(gr[i]), uw = gpu_bf16_to_f32(ur[i]);
#pragma unroll
        for (int j = 0; j < BATCH_TILE; ++j) if (b0 + j < batch) {
            float xv = x[(size_t)(b0 + j) * n_in + i];
            gs[j] += xv * gw; us[j] += xv * uw;
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

__global__ void kv_norm_rope(bf16_t *cache, const float *comp, const bf16_t *norm,
                             int batch, int capacity, int pos, int kv_rank,
                             int kv_dim, int rope_dim, const float *inv_freq,
                             int interleaved, float eps) {
    __shared__ float scratch[THREADS];
    int b = (int)blockIdx.x;
    if (b >= batch) return;
    const float *src = comp + (size_t)b * kv_dim;
    bf16_t *dst = cache + ((size_t)b * capacity + pos) * kv_dim;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS) ss += src[i] * src[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int s = THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / (float)kv_rank + eps);
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
        dst[i] = gpu_f32_to_bf16(src[i] * inv * gpu_bf16_to_f32(norm[i]));
    int j = (int)threadIdx.x;
    if (j < rope_dim / 2) {
        const float *v = src + kv_rank;
        bf16_t *o = dst + kv_rank;
        float angle = (float)pos * inv_freq[j];
        float co = cosf(angle), si = sinf(angle);
        float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved) {
            o[2*j] = gpu_f32_to_bf16(even*co - odd*si);
            o[2*j+1] = gpu_f32_to_bf16(even*si + odd*co);
        } else {
            int half=rope_dim/2;
            o[j]=gpu_f32_to_bf16(even*co-odd*si);
            o[half+j]=gpu_f32_to_bf16(odd*co+even*si);
        }
    }
}

__global__ void q_absorb_rope(float *qabs, float *qrope, const float *q,
                              const bf16_t *wuk_t, int batch, int heads,
                              int q_head_dim, int qk_nope, int rope_dim,
                              int kv_rank, int pos, const float *inv_freq,
                              int interleaved) {
    int wave=(int)threadIdx.x/HIP_WAVE, lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave, total=heads*kv_rank;
    int b0=(int)blockIdx.y*BATCH_TILE;
    if (row>=total || b0>=batch) return;
    int h=row/kv_rank, r=row-h*kv_rank;
    const bf16_t *wr=wuk_t+((size_t)h*kv_rank+r)*qk_nope;
    float sums[BATCH_TILE]={};
    for (int d=lane; d<qk_nope; d+=HIP_WAVE) {
        float wi=gpu_bf16_to_f32(wr[d]);
#pragma unroll
        for(int j=0;j<BATCH_TILE;++j) if(b0+j<batch)
            sums[j]+=q[((size_t)(b0+j)*heads+h)*q_head_dim+d]*wi;
    }
#pragma unroll
    for(int j=0;j<BATCH_TILE;++j) {
        sums[j]=wave_sum(sums[j]);
        if(lane==0 && b0+j<batch) qabs[((size_t)(b0+j)*heads+h)*kv_rank+r]=sums[j];
    }
    if(r==0 && lane<rope_dim/2) {
#pragma unroll
        for(int j=0;j<BATCH_TILE;++j) if(b0+j<batch) {
            const float *v=q+((size_t)(b0+j)*heads+h)*q_head_dim+qk_nope;
            float *o=qrope+((size_t)(b0+j)*heads+h)*rope_dim;
            float angle=(float)pos*inv_freq[lane], co=cosf(angle), si=sinf(angle);
            float even=v[2*lane], odd=v[2*lane+1];
            if(!interleaved){o[2*lane]=even*co-odd*si;o[2*lane+1]=even*si+odd*co;}
            else{int half=rope_dim/2;o[lane]=even*co-odd*si;o[half+lane]=odd*co+even*si;}
        }
    }
}

__global__ void q_rope(float *qrope, const float *q, int batch, int heads,
                       int q_head_dim, int qk_nope, int rope_dim, int pos,
                       const float *inv_freq, int interleaved) {
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int row = (int)blockIdx.x * WAVES + wave;
    if (row >= batch * heads || lane >= rope_dim / 2) return;
    const float *v = q + (size_t)row * q_head_dim + qk_nope;
    float *o = qrope + (size_t)row * rope_dim;
    const float angle = (float)pos * inv_freq[lane];
    const float co = cosf(angle), si = sinf(angle);
    const float even = v[2 * lane], odd = v[2 * lane + 1];
    if (!interleaved) {
        o[2 * lane] = even * co - odd * si;
        o[2 * lane + 1] = even * si + odd * co;
    } else {
        const int half = rope_dim / 2;
        o[lane] = even * co - odd * si;
        o[half + lane] = odd * co + even * si;
    }
}

/* Prefill rows are flattened in batch-major order: row = b * seq + pos. */
__global__ void prefill_kv_norm_rope(
    bf16_t *cache, const float *comp, const bf16_t *norm, int rows, int seq,
    int capacity, int kv_rank, int kv_dim, int rope_dim,
    const float *inv_freq, int interleaved, float eps) {
    __shared__ float scratch[THREADS];
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    const int b = row / seq;
    const int pos = row - b * seq;
    const float *src = comp + (size_t)row * kv_dim;
    bf16_t *dst = cache + ((size_t)b * capacity + pos) * kv_dim;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
        ss += src[i] * src[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)kv_rank + eps);
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
        dst[i] = gpu_f32_to_bf16(
            src[i] * inv * gpu_bf16_to_f32(norm[i]));

    const int j = (int)threadIdx.x;
    if (j < rope_dim / 2) {
        const float *v = src + kv_rank;
        bf16_t *o = dst + kv_rank;
        const float angle = (float)pos * inv_freq[j];
        const float co = cosf(angle), si = sinf(angle);
        const float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved) {
            o[2 * j] = gpu_f32_to_bf16(even * co - odd * si);
            o[2 * j + 1] = gpu_f32_to_bf16(even * si + odd * co);
        } else {
            const int half = rope_dim / 2;
            o[j] = gpu_f32_to_bf16(even * co - odd * si);
            o[half + j] = gpu_f32_to_bf16(odd * co + even * si);
        }
    }
}

__global__ void prefill_q_rope(
    float *qrope, const float *q, int rows, int seq, int heads,
    int q_head_dim, int qk_nope, const float *inv_freq, int interleaved) {
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int qh = (int)blockIdx.x * WAVES + wave;
    if (qh >= rows * heads) return;
    const int row = qh / heads;
    const int pos = row % seq;
    const int rope_dim = q_head_dim - qk_nope;
    if (lane >= rope_dim / 2) return;
    const float *v = q + (size_t)qh * q_head_dim + qk_nope;
    float *o = qrope + (size_t)qh * rope_dim;
    const float angle = (float)pos * inv_freq[lane];
    const float co = cosf(angle), si = sinf(angle);
    const float even = v[2 * lane], odd = v[2 * lane + 1];
    if (!interleaved) {
        o[2 * lane] = even * co - odd * si;
        o[2 * lane + 1] = even * si + odd * co;
    } else {
        const int half = rope_dim / 2;
        o[lane] = even * co - odd * si;
        o[half + lane] = odd * co + even * si;
    }
}

/* Causal Flash Attention for absorbed MLA prefill.  A wave owns one head;
 * all waves share one (request, query), reusing each staged compressed-KV tile
 * across blockDim/64 heads without divergent causal boundaries. */
template <int KV_TILE>
__global__ void flash_prefill(
    float *__restrict__ clat, const float *__restrict__ qabs,
    const float *__restrict__ qrope, const bf16_t *__restrict__ cache,
    int batch, int seq, int heads, int capacity, int kv_dim, int kv_rank,
    int rope_dim, float scale) {
    extern __shared__ bf16_t shared_cache[];
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int waves = (int)blockDim.x / HIP_WAVE;
    const int qpos = (int)blockIdx.y;
    const int h = (int)blockIdx.x * waves + wave;
    const int b = (int)blockIdx.z;
    const bool active = b < batch && h < heads && qpos < seq;
    const size_t qh = ((size_t)b * seq + qpos) * heads + h;
    const float *qa = active ? qabs + qh * kv_rank : nullptr;
    const float *qp = active ? qrope + qh * rope_dim : nullptr;
    const bf16_t *cache_b = cache + (size_t)b * capacity * kv_dim;
    const int last_query = qpos + 1;
    constexpr int VALUES_PER_LANE = 8;
    float acc[VALUES_PER_LANE] = {};
    float running_max = -INFINITY;
    float denominator = 0.0f;

    for (int k0 = 0; k0 < last_query; k0 += KV_TILE) {
        const int tile = min(KV_TILE, last_query - k0);
        for (int i = (int)threadIdx.x; i < tile * kv_dim;
             i += (int)blockDim.x)
            shared_cache[i] = cache_b[(size_t)k0 * kv_dim + i];
        __syncthreads();

        float scores[KV_TILE];
#pragma unroll
        for (int t = 0; t < KV_TILE; ++t) {
            const int key = k0 + t;
            const bool valid = active && t < tile && key <= qpos;
            float score_nope = 0.0f;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                if (valid && r < kv_rank)
                    score_nope += qa[r] * gpu_bf16_to_f32(ckv[r]);
            }
            float score_rope = valid && lane < rope_dim
                ? qp[lane] * gpu_bf16_to_f32(ckv[kv_rank + lane]) : 0.0f;
            score_nope = wave_sum(score_nope);
            score_rope = wave_sum(score_rope);
            scores[t] = valid
                ? __shfl((score_nope + score_rope) * scale, 0, HIP_WAVE)
                : -INFINITY;
        }

        float tile_max = scores[0];
#pragma unroll
        for (int t = 1; t < KV_TILE; ++t)
            tile_max = fmaxf(tile_max, scores[t]);
        const float next_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - next_max);
        denominator *= rescale;
#pragma unroll
        for (int q = 0; q < VALUES_PER_LANE; ++q)
            acc[q] *= rescale;
#pragma unroll
        for (int t = 0; t < KV_TILE; ++t) {
            const int key = k0 + t;
            if (t >= tile) break;
            const float weight = key <= qpos
                ? expf(scores[t] - next_max) : 0.0f;
            denominator += weight;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                if (active && r < kv_rank)
                    acc[q] += gpu_bf16_to_f32(ckv[r]) * weight;
            }
        }
        running_max = next_max;
        __syncthreads();
    }

    if (!active) return;
    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int q = 0; q < VALUES_PER_LANE; ++q) {
        const int r = lane + q * HIP_WAVE;
        if (r < kv_rank)
            clat[qh * kv_rank + r] = acc[q] * inv_denominator;
    }
}

template __global__ void flash_prefill<8>(
    float *, const float *, const float *, const bf16_t *, int, int, int, int,
    int, int, int, float);

__global__ void gather_last_hidden(float *dst, const float *src, int batch,
                                   int seq, int hidden) {
    const int b = (int)blockIdx.y;
    const int d = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (b < batch && d < hidden)
        dst[(size_t)b * hidden + d] =
            src[((size_t)b * seq + seq - 1) * hidden + d];
}

__global__ void attention_score(float *scores, const float *qabs,
                                const float *qrope, const bf16_t *cache,
                                int batch, int heads, int kv_len, int capacity,
                                int kv_dim, int kv_rank, int rope_dim, float scale) {
    int wave=(int)threadIdx.x/HIP_WAVE, lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave, b=(int)blockIdx.y;
    if(row>=heads*kv_len || b>=batch)return;
    int h=row/kv_len,k=row-h*kv_len;
    const float *qa=qabs+((size_t)b*heads+h)*kv_rank;
    const float *qp=qrope+((size_t)b*heads+h)*rope_dim;
    const bf16_t *ckv=cache+((size_t)b*capacity+k)*kv_dim;
    float sn=0.0f,sp=0.0f;
    for(int r=lane;r<kv_rank;r+=HIP_WAVE)sn+=qa[r]*gpu_bf16_to_f32(ckv[r]);
    for(int d=lane;d<rope_dim;d+=HIP_WAVE)sp+=qp[d]*gpu_bf16_to_f32(ckv[kv_rank+d]);
    sn=wave_sum(sn);sp=wave_sum(sp);
    if(lane==0)scores[((size_t)b*heads+h)*capacity+k]=(sn+sp)*scale;
}

/* Flash decode for absorbed MLA.  One wave owns one (batch, head) row and
 * streams the latent KV cache once.  The online softmax keeps only its running
 * max/denominator and the 512-wide latent context in registers, eliminating
 * the materialized score matrix and the separate softmax/context launches. */
template <int KV_TILE>
__global__ void flash_decode(float *__restrict__ clat,
                             const float *__restrict__ qabs,
                             const float *__restrict__ qrope,
                             const bf16_t *__restrict__ cache,
                             int batch, int heads, int kv_len, int capacity,
                             int kv_dim, int kv_rank, int rope_dim, float scale) {
    extern __shared__ bf16_t shared_cache[];
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int waves = (int)blockDim.x / HIP_WAVE;
    const int b = (int)blockIdx.y;
    const int h = (int)blockIdx.x * waves + wave;
    const bool active = b < batch && h < heads;
    const float *qa = active
        ? qabs + ((size_t)b * heads + h) * kv_rank : nullptr;
    const float *qp = active
        ? qrope + ((size_t)b * heads + h) * rope_dim : nullptr;
    const bf16_t *cache_b = cache + (size_t)b * capacity * kv_dim;
    constexpr int VALUES_PER_LANE = 8;
    float acc[VALUES_PER_LANE] = {};
    float running_max = -INFINITY;
    float denominator = 0.0f;

    for (int k0 = 0; k0 < kv_len; k0 += KV_TILE) {
        const int tile = min(KV_TILE, kv_len - k0);
        for (int i = (int)threadIdx.x; i < tile * kv_dim;
             i += (int)blockDim.x)
            shared_cache[i] = cache_b[(size_t)k0 * kv_dim + i];
        __syncthreads();

        float scores[KV_TILE];
        for (int t = 0; t < KV_TILE; ++t) {
            float score_nope = 0.0f;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                if (active && t < tile && r < kv_rank)
                    score_nope += qa[r] * gpu_bf16_to_f32(ckv[r]);
            }
            float score_rope = active && t < tile && lane < rope_dim
                ? qp[lane] * gpu_bf16_to_f32(ckv[kv_rank + lane]) : 0.0f;
            score_nope = wave_sum(score_nope);
            score_rope = wave_sum(score_rope);
            scores[t] = t < tile
                ? __shfl((score_nope + score_rope) * scale, 0, HIP_WAVE)
                : -INFINITY;
        }

        float tile_max = scores[0];
        for (int t = 1; t < KV_TILE; ++t)
            tile_max = fmaxf(tile_max, scores[t]);
        const float next_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - next_max);
        denominator *= rescale;
#pragma unroll
        for (int q = 0; q < VALUES_PER_LANE; ++q)
            acc[q] *= rescale;
        for (int t = 0; t < KV_TILE; ++t) {
            if (t >= tile) break;
            const float weight = expf(scores[t] - next_max);
            denominator += weight;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                if (active && r < kv_rank)
                    acc[q] += gpu_bf16_to_f32(ckv[r]) * weight;
            }
        }
        running_max = next_max;
        __syncthreads();
    }

    if (!active) return;
    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int q = 0; q < VALUES_PER_LANE; ++q) {
        const int r = lane + q * HIP_WAVE;
        if (r < kv_rank)
            clat[((size_t)b * heads + h) * kv_rank + r] =
                acc[q] * inv_denominator;
    }
}

template __global__ void flash_decode<8>(
    float *, const float *, const float *, const bf16_t *, int, int, int, int,
    int, int, int, float);
template __global__ void flash_decode<16>(
    float *, const float *, const float *, const bf16_t *, int, int, int, int,
    int, int, int, float);

__global__ void latent_context(float *clat, const float *scores, const bf16_t *cache,
                               int batch, int heads, int kv_len, int capacity,
                               int kv_dim, int kv_rank) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave,b=(int)blockIdx.y;
    if(row>=heads*kv_rank||b>=batch)return;
    int h=row/kv_rank,r=row-h*kv_rank;
    const float *a=scores+((size_t)b*heads+h)*capacity;
    const bf16_t *cb=cache+(size_t)b*capacity*kv_dim;
    float sum=0.0f;
    for(int k=lane;k<kv_len;k+=HIP_WAVE)
        sum+=a[k]*gpu_bf16_to_f32(cb[(size_t)k*kv_dim+r]);
    sum=wave_sum(sum);
    if(lane==0)clat[((size_t)b*heads+h)*kv_rank+r]=sum;
}

__global__ void value_up(float *ctx, const float *clat, const bf16_t *wuv,
                         int batch, int heads, int value_dim, int kv_rank,
                         size_t head_stride) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave,b0=(int)blockIdx.y*BATCH_TILE;
    if(row>=heads*value_dim||b0>=batch)return;
    int h=row/value_dim,d=row-h*value_dim;
    const bf16_t *wr=wuv+(size_t)h*head_stride+(size_t)d*kv_rank;
    float sums[BATCH_TILE]={};
    for(int r=lane;r<kv_rank;r+=HIP_WAVE){
        float wi=gpu_bf16_to_f32(wr[r]);
#pragma unroll
        for(int j=0;j<BATCH_TILE;++j)if(b0+j<batch)
            sums[j]+=clat[((size_t)(b0+j)*heads+h)*kv_rank+r]*wi;
    }
#pragma unroll
    for(int j=0;j<BATCH_TILE;++j){sums[j]=wave_sum(sums[j]);if(lane==0&&b0+j<batch)
        ctx[((size_t)(b0+j)*heads+h)*value_dim+d]=sums[j];}
}

__global__ void routed_shared_gate_up(float *routed, float *shared, const float *x,
                                      const char *pool, const size_t *gate_offsets,
                                      const size_t *up_offsets, const int *topk,
                                      const bf16_t *shared_gate, const bf16_t *shared_up,
                                      int batch, int top_k, int inter, int shared_inter,
                                      int input_dim) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*WAVES+wave,b=(int)blockIdx.y;
    int routed_rows=top_k*inter,total=routed_rows+shared_inter;
    if(row>=total||b>=batch)return;
    const bf16_t *gate,*up;float *dst;int j;
    if(row<routed_rows){int slot=row/inter;j=row-slot*inter;int e=topk[(size_t)b*top_k+slot];
        gate=(const bf16_t*)(pool+gate_offsets[e]);up=(const bf16_t*)(pool+up_offsets[e]);
        dst=routed+((size_t)b*top_k+slot)*inter;}
    else{j=row-routed_rows;gate=shared_gate;up=shared_up;dst=shared+(size_t)b*shared_inter;}
    const bf16_t *gr=gate+(size_t)j*input_dim,*ur=up+(size_t)j*input_dim;
    const float *xb=x+(size_t)b*input_dim;float gs=0.0f,us=0.0f;
    for(int i=lane;i<input_dim;i+=HIP_WAVE){float xv=xb[i];gs+=xv*gpu_bf16_to_f32(gr[i]);us+=xv*gpu_bf16_to_f32(ur[i]);}
    gs=wave_sum(gs);us=wave_sum(us);if(lane==0)dst[j]=(gs/(1.0f+expf(-gs)))*us;
}

__global__ void routed_shared_down(float *out, const float *routed, const float *shared,
                                   const char *pool, const size_t *down_offsets,
                                   const int *topk, const float *topk_weights,
                                   const bf16_t *shared_down, int batch, int top_k,
                                   int hidden, int inter, int shared_inter) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int d=(int)blockIdx.x*WAVES+wave,b=(int)blockIdx.y;
    if(d>=hidden||b>=batch)return;float total=0.0f;
    for(int slot=0;slot<top_k;++slot){int e=topk[(size_t)b*top_k+slot];
        const bf16_t *wr=(const bf16_t*)(pool+down_offsets[e])+(size_t)d*inter;
        const float *hp=routed+((size_t)b*top_k+slot)*inter;float sum=0.0f;
        for(int j=lane;j<inter;j+=HIP_WAVE)sum+=hp[j]*gpu_bf16_to_f32(wr[j]);
        sum=wave_sum(sum);if(lane==0)total+=topk_weights[(size_t)b*top_k+slot]*sum;}
    float ss=0.0f;
    if(shared_inter>0){const bf16_t *swr=shared_down+(size_t)d*shared_inter;
        const float *sh=shared+(size_t)b*shared_inter;
        for(int j=lane;j<shared_inter;j+=HIP_WAVE)ss+=sh[j]*gpu_bf16_to_f32(swr[j]);
        ss=wave_sum(ss);}
    if(lane==0)out[(size_t)b*hidden+d]+=total+ss;
}

__global__ void argmax(const float *logits, int n, int *result, int batch) {
    __shared__ float bv[THREADS];__shared__ int bi[THREADS];int b=(int)blockIdx.x;
    if(b>=batch)return;const float *row=logits+(size_t)b*n;float vbest=-INFINITY;int ibest=0x7fffffff;
    for(int i=(int)threadIdx.x;i<n;i+=THREADS){float v=row[i];if(v>vbest||(v==vbest&&i<ibest)){vbest=v;ibest=i;}}
    bv[threadIdx.x]=vbest;bi[threadIdx.x]=ibest;__syncthreads();
    for(int s=THREADS/2;s;s>>=1){if((int)threadIdx.x<s){float ov=bv[threadIdx.x+s];int oi=bi[threadIdx.x+s];
        if(ov>bv[threadIdx.x]||(ov==bv[threadIdx.x]&&oi<bi[threadIdx.x])){bv[threadIdx.x]=ov;bi[threadIdx.x]=oi;}}__syncthreads();}
    if(threadIdx.x==0)result[b]=bi[0];
}

} // namespace batch_decode_kernels
