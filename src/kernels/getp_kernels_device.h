#ifndef MLA_GETP_KERNELS_DEVICE_H
#define MLA_GETP_KERNELS_DEVICE_H

#include <cstdint>

__device__ __forceinline__ float gpu_bf16_to_f32(bf16_t value) {
    return __uint_as_float((uint32_t)value << 16);
}

__device__ __forceinline__ float wave_sum(float value) {
    for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1)
        value += __shfl_down(value, offset, HIP_WAVE);
    return value;
}

__device__ __forceinline__ float wave_max(float value) {
    for (int offset = HIP_WAVE / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_down(value, offset, HIP_WAVE));
    return value;
}

#endif
