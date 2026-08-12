#ifndef PREFILL_KERNELS_H
#define PREFILL_KERNELS_H

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "kernels_utils.h"

namespace prefill_kernels {

  template<
    size_t H = kct::HIDDEN_SIZE,
    size_t BLOCKS = kct::EMBED_BLOCKS,
    size_t THREADS = kct::EMBED_THREADS
  >
  __global__ void embed_kernel(
    const int L, // seq len
    float *xs, // [L x H]
    const __hip_bfloat16 *w_embed_tokens, // [VOC x H]
    const int *tokens // [L]
  );
}

#endif