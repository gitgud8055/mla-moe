#include "getp_kernels.h"
#include "getp_kernels_device.h"

__global__ void gemv_kernel(float *y, const float *x, const bf16_t *w,
                            int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    const bf16_t *wr = w + (size_t)row * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += x[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) y[row] = sum;
}

__global__ void gate_up_swiglu_kernel(float *y, const float *x,
                                      const bf16_t *gate, const bf16_t *up,
                                      int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    const bf16_t *gr = gate + (size_t)row * n_in;
    const bf16_t *ur = up + (size_t)row * n_in;
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float xv = x[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0) y[row] = (gs / (1.0f + expf(-gs))) * us;
}

__global__ void gemv_add_kernel(float *dst, const float *x, const bf16_t *w,
                                int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    const bf16_t *wr = w + (size_t)row * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += x[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) dst[row] += sum;
}
