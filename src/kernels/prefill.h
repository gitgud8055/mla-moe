#ifndef PREFILL_H
#define PREFILL_H

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cassert>
#include <vector>

#include "getp_utils.h"
#include "getp_state.h"   /* GpuContext */

namespace prefill {
  template <
      size_t H,
      size_t BLOCKS,
      size_t THREADS>
  __global__ void embed_kernel(
      const int L,                          // seq len
      float *xs,                            // [L x H]
      const __hip_bfloat16 *w_embed_tokens, // [VOC x H]
      const int *tokens                     // [L]
  )
  {
    using namespace types;
    constexpr size_t VEC_H = 8;
    static_assert(H % VEC_H == 0);
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    for (int block_l = bx; block_l < L; block_l += BLOCKS)
    {
      const __hip_bfloat16 *token = w_embed_tokens + static_cast<size_t>(tokens[block_l]) * H;
      bf16v8 bf16_reg;
      fp32v4 fp32_reg;
      for (size_t h_vec = tx; h_vec < H / VEC_H; h_vec += THREADS)
      {
        size_t h_t = h_vec * VEC_H;
        bf16_reg = *reinterpret_cast<const bf16v8 *>(token + h_t);
        for (int i = 0; i < 4; ++i)
        {
          fp32_reg[i] = __bfloat162float(__hip_bfloat16_raw{bf16_reg[i]});
        }
        *reinterpret_cast<fp32v4 *>(xs + block_l * H + h_t) = fp32_reg;
        for (int i = 4; i < 8; ++i)
        {
          fp32_reg[i - 4] = __bfloat162float(__hip_bfloat16_raw{bf16_reg[i]});
        }
        *reinterpret_cast<fp32v4 *>(xs + block_l * H + h_t + 4) = fp32_reg;
      }
    }
  }

  /* Embed one prompt into g.xs [L x hidden_size]. Enqueues on g.stream and
   * synchronizes before returning (the scratch token buffer is freed here). */
  inline void run(GpuContext &g, const std::vector<int> &tokens) {
    size_t L = tokens.size();
    const int *h_tokens = tokens.data();
    assert((size_t)g.owner->config.hidden_size == constants::models::HIDDENS_SIZE);

    int *d_tokens = nullptr;
    HIP_CHECK(hipMalloc(&d_tokens, L * sizeof(int)));
    HIP_CHECK(hipMemcpyAsync(d_tokens, h_tokens, L * sizeof(int),
                             hipMemcpyHostToDevice, g.stream));

    constexpr size_t EMBED_BLOCKS = 64;
    constexpr size_t EMBED_THREADS = constants::kernel::WAVE * 4;
    embed_kernel<constants::models::HIDDENS_SIZE, EMBED_BLOCKS, EMBED_THREADS><<<
        EMBED_BLOCKS, EMBED_THREADS, 0, g.stream>>>(
        (int)L,
        g.xs,
        reinterpret_cast<const __hip_bfloat16 *>(g.embed),
        d_tokens
    );
    HIP_LAUNCH_CHECK();

    HIP_CHECK(hipStreamSynchronize(g.stream));
    HIP_CHECK(hipFree(d_tokens));
  }

}

#endif
