#ifndef MLA_GETP_KERNELS_DEVICE_H
#define MLA_GETP_KERNELS_DEVICE_H

#include <cstdint>
#include <cstddef>

using getp_int8x8 = signed char __attribute__((ext_vector_type(8)));
using getp_bf16x8 = short __attribute__((ext_vector_type(8)));

constexpr size_t GETP_W8_ALIGN = 256;

__device__ __forceinline__ float gpu_bf16_to_f32(bf16_t value) {
    return __uint_as_float((uint32_t)value << 16);
}

__device__ __forceinline__ bf16_t gpu_f32_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000U) != 0x7f800000U) {
        /* IEEE round-to-nearest-even, matching HIP/PyTorch BF16 casts. */
        bits += 0x7fffU + ((bits >> 16) & 1U);
    } else if (bits & 0xffffU) {
        /* Keep a NaN payload instead of rounding it to infinity. */
        bits |= 0x10000U;
    }
    return (bf16_t)(bits >> 16);
}

__device__ __forceinline__ const float *w8_scales(const void *packed,
                                                 int rows, int cols) {
    const size_t qbytes = (size_t)rows * (size_t)cols;
    const size_t aligned =
        (qbytes + (GETP_W8_ALIGN - 1)) & ~(GETP_W8_ALIGN - 1);
    return reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(packed) + aligned);
}

__device__ __forceinline__ float w8_scale(const void *packed, int rows,
                                         int cols, int row, int w8) {
    return w8 ? w8_scales(packed, rows, cols)[row] : 1.0f;
}

__device__ __forceinline__ float load_weight_f32(const bf16_t *w, size_t index,
                                                int w8) {
    if (w8)
        return (float)reinterpret_cast<const int8_t *>(w)[index];
    return gpu_bf16_to_f32(w[index]);
}

__device__ __forceinline__ getp_bf16x8 load_weight_x8(
    const bf16_t *w, int gc, int d_out, int bk, int k, int n_in, int w8) {
    getp_bf16x8 value = {};
    if (gc >= d_out)
        return value;
    if (w8) {
        const int8_t *q = reinterpret_cast<const int8_t *>(w);
        if (bk + k + 8 <= n_in) {
            getp_int8x8 qv = *reinterpret_cast<const getp_int8x8 *>(
                q + (size_t)gc * n_in + bk + k);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                value[i] = gpu_f32_to_bf16((float)(int)qv[i]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (bk + k + i < n_in)
                    value[i] = gpu_f32_to_bf16(
                        (float)q[(size_t)gc * n_in + bk + k + i]);
        }
        return value;
    }
    if (bk + k + 8 <= n_in) {
        value = *reinterpret_cast<const getp_bf16x8 *>(
            w + (size_t)gc * n_in + bk + k);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (bk + k + i < n_in)
                value[i] = reinterpret_cast<const short *>(w)[
                    (size_t)gc * n_in + bk + k + i];
    }
    return value;
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
