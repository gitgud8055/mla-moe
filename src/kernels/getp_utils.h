#ifndef GETP_UTILS_H
#define GETP_UTILS_H

#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace constants {
  namespace models {
    constexpr size_t HIDDEN_SIZE = 2048; 
    constexpr size_t SEQ_LENS[] = {64, 128, 256, 384, 512};
  }
  namespace kernels {
    constexpr size_t WAVE = 64;
    constexpr size_t LDS = 64 * 1024;
  }
  namespace datasets {
    /* requests / max prompt len / max new tokens:
     *   data/       512 reqs   dsv2lite 512 / 64   glm47 512 / 64
     *   tests/eval/   5 reqs   dsv2lite 942 / 64   glm47 930 / 64
     * model max_position_embeddings: dsv2lite 163840, glm47 202752 */
    constexpr size_t MAX_REQS     = 512;
    constexpr size_t MAX_N_PROMPT = 1024;
    constexpr size_t MAX_NEW      = 64;
    constexpr size_t MAX_SEQ      = MAX_N_PROMPT + MAX_NEW;
  }
}

namespace types {
  typedef unsigned short bf16v8 __attribute__((ext_vector_type(8)));
  typedef float fp32v4 __attribute__((ext_vector_type(4)));
}

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

namespace rmsnorm_kernel {
  template<
    size_t BLOCKS,
    size_t THREADS,
    size_t BLOCK_TILE_R
  >
  __global__ void rmsnorm_kernel(
    float *y, // [r x c]
    const float *x, // [r x c]
    const __hip_bfloat16 *w, // [c]
    int r,
    int c,
    float eps
  ) {
    // c is oneof
    // q_lora_rank = null(DSV) || 768(GLM)
    // kv_lora_rank = 512
    // hidden_size = 2048
    static_assert((THREADS & (THREADS - 1)) == 0);

    constexpr size_t SM_SIZ_0 = constants::models::HIDDEN_SIZE * sizeof(float);
    static_assert(SM_SIZ_0 <= constants::kernels::LDS);
    __shared__ float s_x[constants::models::HIDDEN_SIZE];

    constexpr size_t SM_SIZ_1 = SM_SIZ_0 + (THREADS + 1) * sizeof(float);
    static_assert(SM_SIZ_1 <= constants::kernels::LDS);
    __shared__ float s_sum[THREADS];
    __shared__ float inv;

    size_t tix = threadIdx.x;
    size_t bix = blockIdx.x;
    for (size_t block_r_offset = bix * BLOCK_TILE_R; block_r_offset < static_cast<size_t>(r); block_r_offset += BLOCKS * BLOCK_TILE_R) {
      for (size_t block_r = block_r_offset; block_r < block_r_offset + BLOCK_TILE_R && block_r < r; ++block_r) {
        const float *block_x = x + block_r * c;
        constexpr size_t VEC_C = 4;
        static_assert(constants::models::HIDDEN_SIZE % VEC_C == 0);
        types::fp32v4 fp32_reg;
        float thread_sum = 0.f;
        for (size_t thread_c_vec = tix; thread_c_vec < c / VEC_C; thread_c_vec += THREADS) {
          size_t thread_c_offset = thread_c_vec * VEC_C;
          fp32_reg = *reinterpret_cast<const types::fp32v4 *>(block_x + thread_c_offset);
          *reinterpret_cast<types::fp32v4 *>(s_x + thread_c_offset) = fp32_reg;
          fp32_reg *= fp32_reg;
          for (size_t v = 0; v < VEC_C; ++v) {
            thread_sum += fp32_reg[v];
          }
        }
        s_sum[tix] = thread_sum;
        __syncthreads();

        for (size_t stride = THREADS / 2; stride > 0; stride /= 2) {
          if (tix < stride) {
            s_sum[tix] += s_sum[tix + stride];
          }
          __syncthreads();
        }

        if (tix == 0) {
          inv = 1.f / sqrtf(s_sum[0] / static_cast<float>(c) + eps);
        }
        __syncthreads();
        
        constexpr size_t VEC_C_BF16 = 8;
        types::bf16v8 bf16_reg;
        types::fp32v4 fp32_reg_y;
        float *block_y = y + block_r * c;
        for (size_t thread_c_vec = tix; thread_c_vec < c / VEC_C_BF16; thread_c_vec += THREADS) {
          size_t thread_c_offset = thread_c_vec * VEC_C_BF16;
          bf16_reg = *reinterpret_cast<const types::bf16v8 *>(w + thread_c_offset);
          
          for (size_t i = 0; i < VEC_C_BF16 / VEC_C; ++i) {
            fp32_reg = *reinterpret_cast<types::fp32v4 *>(s_x + thread_c_offset + i * VEC_C);
            for (size_t j = 0; j < VEC_C; ++j) {
              fp32_reg_y[j] = fp32_reg[j] * inv * __bfloat162float(__hip_bfloat16_raw{bf16_reg[j + i * VEC_C]});
            }
            *reinterpret_cast<types::fp32v4 *>(block_y + thread_c_offset + i * VEC_C) = fp32_reg_y;
          }
        }
        __syncthreads();
      }
    }
  }
}

#endif