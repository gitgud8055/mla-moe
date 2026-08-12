#include "prefill_kernels.h"

namespace prefill_kernels {
  template<
    size_t H,
    size_t BLOCKS,
    size_t THREADS
  >
  __global__ void embed_kernel(
    const int L, // seq len
    float *xs, // [L x H]
    const __hip_bfloat16 *w_embed_tokens, // [VOC x H]
    const int *tokens // [L]
  ) {
    constexpr size_t VEC_H = 8;
    static_assert(H % VEC_H == 0);
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    for (int block_l = bx; block_l < L; block_l += BLOCKS) {
      const __hip_bfloat16 *token = w_embed_tokens + static_cast<size_t>(tokens[block_l]) * H;
      ktp::bf16v8 bf16_reg;
      ktp::fp32v4 fp32_reg;
      for (size_t h_vec = tx; h_vec < H / VEC_H; h_vec += THREADS) {
        size_t h_t = h_vec * VEC_H;
        bf16_reg = *reinterpret_cast<const ktp::bf16v8 *>(token + h_t);
        for (int i = 0; i < 4; ++i) {
          fp32_reg[i] = __bfloat162float(__hip_bfloat16_raw{bf16_reg[i]});
        }
        *reinterpret_cast<ktp::fp32v4 *>(xs + block_l * H + h_t) = fp32_reg;
        for (int i = 4; i < 8; ++i) {
          fp32_reg[i - 4] = __bfloat162float(__hip_bfloat16_raw{bf16_reg[i]});
        }
        *reinterpret_cast<ktp::fp32v4 *>(xs + block_l * H + h_t + 4) = fp32_reg;
      }
    }
  }
  
  template __global__ void embed_kernel(
    const int, 
    float *,
    const __hip_bfloat16 *,
    const int *
  );
}