#ifndef GEMM_H
#define GEMM_H

#include "tensor.h"
#include "utils.h"

namespace mct = utils::constants::model;
namespace gct = utils::constants::gpu;
namespace types = utils::types;

namespace utils {
  namespace gemm {
    // Y[M x N] = X[M x K] @ W[N x K].T
    template<
      size_t MAT_M = 16,
      size_t MAT_N = 16,
      size_t MAT_K = 4,
      size_t WPW_M = 4, // scale workload per wave M
      size_t WPW_N = 4, // scale workload per wave N
      size_t WPW_K = 4, // scale workoad per wave K
      size_t WPB_M = 4, // scale workload per block M
      size_t WPB_N = 2, // sacle workload per block N
      size_t PAD = 0
    >
    static __global__ void gemm_kernel(
      float *__restrict__ Y,
      const float *__restrict__ X,
      const bf16_t *__restrict__ W,
      int M,
      int N,
      int K,
      bool add
    ) {
#define A(x, y) X[(x) * K + (y)]
#define PA(x, y) (X + (x) * K + (y))
#define B(x, y) W[(x) * K + (y)]
#define PB(x, y) (W + (x) * K + (y))
#define C(x, y) Y[(x) * N + (y)]
#define PC(x, y) (Y + (x) * N + (y))
      constexpr size_t BLOCK = gct::WAVE * WPB_M * WPB_N;
      constexpr size_t BLOCK_M = MAT_M * WPW_M * WPB_M;
      constexpr size_t BLOCK_N = MAT_N * WPW_N * WPB_N;
      constexpr size_t BLOCK_K = MAT_K * WPW_K;
            
      int lane = threadIdx.x;
      int wf_ic = threadIdx.y;
      int wf_ir = threadIdx.z;
      int tix = wf_ir * (gct::WAVE * WPB_N) + wf_ic * gct::WAVE + lane;
      int a_i = lane % 16;
      int a_k = lane / 16;
      int b_j = lane % 16;
      int b_k = lane / 16;
      int c_i_base = 4 * (lane / 16);
      int c_j = lane % 16;
      static_assert(gct::LDS >= BLOCK_M * (BLOCK_K + PAD) * sizeof(float) + BLOCK_N * (BLOCK_K + PAD) * sizeof(float));
      __shared__ float shared_a[BLOCK_M][BLOCK_K+PAD];
      __shared__ float shared_b[BLOCK_N][BLOCK_K+PAD];
      int block_c_offset = blockIdx.x * BLOCK_N;
      int block_r_offset = blockIdx.y * BLOCK_M;
      if (block_c_offset >= N || block_r_offset >= M) {
          return;
      }
      int wf_c_offset = wf_ic * (WPW_N * MAT_N);
      int wf_r_offset = wf_ir * (WPW_M * MAT_M);
      types::fp32v4 acc[WPW_M][WPW_N] = {};
      for (int block_k_offset = 0; block_k_offset < K; block_k_offset += BLOCK_K) {
        constexpr size_t VEC_K_FP32 = 4;
        constexpr size_t VEC_K_BF16 = 8;
        static_assert(BLOCK_K % VEC_K_FP32 == 0);
        for (int i = tix; i < BLOCK_M * BLOCK_K / VEC_K_FP32; i += BLOCK) {
            int r = i / (BLOCK_K / VEC_K_FP32);
            int k = i % (BLOCK_K / VEC_K_FP32) * VEC_K_FP32;
            types::fp32v4 vec = {};
            if (block_r_offset + r < M && block_k_offset + k + VEC_K_FP32 <= K) {
                vec = *reinterpret_cast<const types::fp32v4*>(PA(block_r_offset + r, block_k_offset + k));
            } else if (block_r_offset + r < M) {
                for (int v = 0; v < VEC_K_FP32; ++v) {
                    if (block_k_offset + k + v < K) {
                        vec[v] = A(block_r_offset + r, block_k_offset + k + v);
                    }
                }
            }
            *reinterpret_cast<types::fp32v4*>(shared_a[r] + k) = vec;
        }
        static_assert(BLOCK_K % VEC_K_BF16 == 0);
        for (int i = tix; i < BLOCK_N * BLOCK_K / VEC_K_BF16; i += BLOCK) {
            int k = i % (BLOCK_K / VEC_K_BF16) * VEC_K_BF16;
            int c = i / (BLOCK_K / VEC_K_BF16);
            types::bf16v8 vec_bf16v8 = {};
            if (block_c_offset + c < N && block_k_offset + k + VEC_K_BF16 <= K) {
                vec_bf16v8 = *reinterpret_cast<const types::bf16v8*>(PB(block_c_offset + c, block_k_offset + k));
            } else if (block_c_offset + c < N) {
                for (int v = 0; v < VEC_K_BF16; ++v) {
                    if (block_k_offset + k + v < K) {
                        vec_bf16v8[v] = B(block_c_offset + c, block_k_offset + k + v);
                    }
                }
            }
            types::fp32v8 vec_fp32v8;
            for (size_t v = 0; v < VEC_K_BF16; ++v) {
              vec_fp32v8[v] = __bfloat162float(__hip_bfloat16_raw{vec_bf16v8[v]});
            }
            *reinterpret_cast<types::fp32v8*>(shared_b[c] + k) = vec_fp32v8;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_K; k += MAT_K) {
            #pragma unroll
            for (int ws_m = 0; ws_m < WPW_M; ++ws_m) {
                #pragma unroll
                for (int ws_n = 0; ws_n < WPW_N; ++ws_n) {
                    float a = shared_a[wf_r_offset + ws_m * MAT_M + a_i][k + a_k];
                    float b = shared_b[wf_c_offset + ws_n * MAT_N + b_j][k + b_k];
                    acc[ws_m][ws_n] = __builtin_amdgcn_mfma_f32_16x16x4f32(
                        a, b, acc[ws_m][ws_n], 0, 0, 0
                    );
                }
            }
        }
        __syncthreads();
      }
      for (int ws_m = 0; ws_m < WPW_M; ++ws_m) {
        for (int ws_n = 0; ws_n < WPW_N; ++ws_n) {
            int c = block_c_offset + wf_c_offset + ws_n * MAT_N + c_j;
            if (c >= N) {
                break;
            }
            int r_base = block_r_offset + wf_r_offset + ws_m * MAT_M + c_i_base;
            for (int i = r_base; i < r_base + MAT_K && i < M; ++i) {
              if (add) {
                C(i, c) += acc[ws_m][ws_n][i - r_base];
              } else {
                C(i, c) = acc[ws_m][ws_n][i - r_base];
              }
            }
        }
      }
#undef A
#undef PA
#undef B
#undef PB
#undef C
#undef PC
    }
    // Y[M x N] = X[M x K] @ W[N x K].T
    static __forceinline__ void gemm(
      float *__restrict__ Y,
      const float *__restrict__ X,
      const bf16_t *__restrict__ W,
      int M,
      int N,
      int K,
      hipStream_t stream,
      bool add = false
    ) {
      constexpr size_t MAT_M = 16;
      constexpr size_t MAT_N = 16;
      constexpr size_t MAT_K = 4;
      constexpr size_t WPW_M = 4;
      constexpr size_t WPW_N = 4;
      constexpr size_t WPW_K = 4;
      constexpr size_t WPB_M = 4;
      constexpr size_t WPB_N = 2;
      constexpr size_t PAD = 0;
      constexpr size_t BLOCK_M = MAT_M * WPW_M * WPB_M;
      constexpr size_t BLOCK_N = MAT_N * WPW_N * WPB_N;
      dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
      dim3 block(64, WPB_N, WPB_M);
      gemm_kernel
      <<<grid, block, 0, stream>>>(
        Y, X, W, M, N, K, add
      );
    }
  }
}

#endif