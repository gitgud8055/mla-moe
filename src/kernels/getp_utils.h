#ifndef GETP_UTILS_H
#define GETP_UTILS_H

#include <cstdio>

namespace constants {
  namespace models {
    constexpr size_t HIDDENS_SIZE = 2048; 
  }
  namespace kernel {
    constexpr size_t WAVE = 64;
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

#endif