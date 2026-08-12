#ifndef PREFILL_H
#define PREFILL_H

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cassert>
#include <vector>

#include "getp_utils.h"
#include "getp_state.h"
#include "embed.h"

namespace prefill {
  /* Embed one prompt into g.xs [L x hidden_size]. Enqueues on g.stream and
   * synchronizes before returning (the scratch token buffer is freed here). */
  inline void run(GpuContext &g, const std::vector<int> &tokens) {
    size_t L = tokens.size();
    const int *h_tokens = tokens.data();
    assert((size_t)g.owner->config.hidden_size == constants::models::HIDDEN_SIZE);

    int *d_tokens = nullptr;
    HIP_CHECK(hipMalloc(&d_tokens, L * sizeof(int)));
    HIP_CHECK(hipMemcpyAsync(d_tokens, h_tokens, L * sizeof(int),
                             hipMemcpyHostToDevice, g.stream));

    constexpr size_t EMBED_BLOCKS = 64;
    constexpr size_t EMBED_THREADS = constants::kernels::WAVE * 4;
    embed::embed_kernel<constants::models::HIDDEN_SIZE, EMBED_BLOCKS, EMBED_THREADS><<<
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
