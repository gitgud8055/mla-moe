#ifndef UTILS_H
#define UTILS_H

#include <cstdio>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

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
    namespace gpu {
      constexpr size_t LDS = 64 * 1024;
      constexpr size_t WAVE = 64;
    }
    namespace model {
      constexpr size_t HIDDEN_SIZE = 2048;
      constexpr size_t SEQ_LENS[] = {64, 128, 256, 384, 512};
    }
    namespace kernel {

    }
  }

  namespace types {
    typedef unsigned short bf16v8 __attribute__((ext_vector_type(8)));
    typedef float fp32v4 __attribute__((ext_vector_type(4)));
    typedef float fp32v8 __attribute__((ext_vector_type(8)));
  }
}

#endif