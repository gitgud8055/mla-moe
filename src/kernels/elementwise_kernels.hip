#include "getp_kernels.h"
#include "getp_kernels_device.h"

namespace elementwise_kernels
{

    template <int THREADS>
    __global__ void embedding_kernel(float *x, const bf16_t *table, int host_token,
                                     const int *device_token, int *generated,
                                     int output_index, int hidden)
    {
        __shared__ int token;
        if (threadIdx.x == 0)
        {
            token = device_token ? *device_token : host_token;
            if (blockIdx.x == 0 && generated)
                generated[output_index] = token;
        }
        __syncthreads();
        int i = (int)blockIdx.x * THREADS + (int)threadIdx.x;
        if (i < hidden)
            x[i] = gpu_bf16_to_f32(table[(size_t)token * hidden + i]);
    }

    template <int THREADS>
    __global__ void rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                                   int n, float eps)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        constexpr int ROWS_PER_BLOCK = THREADS / HIP_WAVE;
        __shared__ float scratch[ROWS_PER_BLOCK];
        float ss = 0.0f;
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            ss += x[i] * x[i];

        ss = wave_sum(ss);
        if (threadIdx.x % HIP_WAVE == 0)
            scratch[threadIdx.x / HIP_WAVE] = ss;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            float total_ss = 0.0f;
#pragma unroll
            for (int i = 0; i < ROWS_PER_BLOCK; ++i)
            {
                total_ss += scratch[i];
            }
            scratch[0] = 1.0f / sqrtf(total_ss / (float)n + eps);
        }
        __syncthreads();
        float inv = scratch[0];
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            y[i] = x[i] * inv * gpu_bf16_to_f32(w[i]);
    }

    template <int THREADS>
    __global__ void batched_rmsnorm_kernel(float *y, const float *x, const bf16_t *w,
                                           int n, float eps)
    {
        int b = (int)blockIdx.y;
        const float *xb = x + (size_t)b * n;
        float *yb = y + (size_t)b * n;

        static_assert(THREADS % HIP_WAVE == 0);
        constexpr int ROWS_PER_BLOCK = THREADS / HIP_WAVE;
        __shared__ float scratch[ROWS_PER_BLOCK];
        float ss = 0.0f;
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            ss += xb[i] * xb[i];

        ss = wave_sum(ss);
        if (threadIdx.x % HIP_WAVE == 0)
            scratch[threadIdx.x / HIP_WAVE] = ss;
        __syncthreads();

        if (threadIdx.x == 0)
        {
            float total_ss = 0.0f;
#pragma unroll
            for (int i = 0; i < ROWS_PER_BLOCK; ++i)
            {
                total_ss += scratch[i];
            }
            scratch[0] = 1.0f / sqrtf(total_ss / (float)n + eps);
        }
        __syncthreads();
        float inv = scratch[0];
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            yb[i] = xb[i] * inv * gpu_bf16_to_f32(w[i]);
    }

    template <int THREADS>
    __global__ void add_kernel(float *dst, const float *src, int n)
    {
        int i = (int)blockIdx.x * THREADS + (int)threadIdx.x;
        if (i < n)
            dst[i] += src[i];
    }

    template <int THREADS>
    __global__ void zero_kernel(float *dst, int n)
    {
        int i = (int)blockIdx.x * THREADS + (int)threadIdx.x;
        if (i < n)
            dst[i] = 0.0f;
    }

    template <int THREADS>
    __global__ void argmax_kernel(const float *logits, int n, int *result)
    {
        __shared__ float best_value[THREADS];
        __shared__ int best_index[THREADS];
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
        {
            float v = logits[i];
            if (v > bv || (v == bv && i < bi))
            {
                bv = v;
                bi = i;
            }
        }
        best_value[threadIdx.x] = bv;
        best_index[threadIdx.x] = bi;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if ((int)threadIdx.x < stride)
            {
                float ov = best_value[threadIdx.x + stride];
                int oi = best_index[threadIdx.x + stride];
                if (ov > best_value[threadIdx.x] ||
                    (ov == best_value[threadIdx.x] && oi < best_index[threadIdx.x]))
                {
                    best_value[threadIdx.x] = ov;
                    best_index[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            *result = best_index[0];
    }

} // namespace elementwise_kernels

template __global__ void elementwise_kernels::embedding_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    float *, const bf16_t *, int, const int *, int *, int, int);
template __global__ void elementwise_kernels::rmsnorm_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    float *, const float *, const bf16_t *, int, float);
template __global__ void elementwise_kernels::batched_rmsnorm_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    float *, const float *, const bf16_t *, int, float);
template __global__ void elementwise_kernels::add_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    float *, const float *, int);
template __global__ void elementwise_kernels::zero_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    float *, int);
template __global__ void elementwise_kernels::argmax_kernel<kernel_tuning::ELEMENTWISE_THREADS>(
    const float *, int, int *);
