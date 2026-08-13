#include "getp_kernels.h"
#include "getp_kernels_device.h"

__global__ void gemv_kernel(float *y, const float *x, const bf16_t *w,
                            int d_out, int n_in, int add) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= d_out) return;
    const bf16_t *wr = w + (size_t)row * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += x[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) {
        if (add) y[row] += sum;
        else y[row] = sum;
    }
}

__global__ void batched_gemv_kernel(float *y, const float *x, const bf16_t *w,
                                    int d_out, int n_in, int add) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int b = (int)blockIdx.y; // batch (token) index
    if (row >= d_out) return;
    const bf16_t *wr = w + (size_t)row * n_in;
    const float *xb = x + (size_t)b * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += xb[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) {
        float *yb = y + (size_t)b * d_out;
        if (add) yb[row] += sum;
        else yb[row] = sum;
    }
}

__global__ void dual_gemv_kernel(
    float *y0, const bf16_t *w0, int d_out0,
    float *y1, const bf16_t *w1, int d_out1,
    const float *x, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int total = d_out0 + d_out1;
    if (row >= total) return;
    float *y = row < d_out0 ? y0 : y1;
    const bf16_t *w = row < d_out0 ? w0 : w1;
    int local_row = row < d_out0 ? row : row - d_out0;
    const bf16_t *wr = w + (size_t)local_row * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE)
        sum += x[i] * gpu_bf16_to_f32(wr[i]);
    sum = wave_sum(sum);
    if (lane == 0) y[local_row] = sum;
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

__global__ void batched_gate_up_swiglu_kernel(float *y, const float *x,
                                              const bf16_t *gate, const bf16_t *up,
                                              int d_out, int n_in) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int b = (int)blockIdx.y;
    if (row >= d_out) return;
    const bf16_t *gr = gate + (size_t)row * n_in;
    const bf16_t *ur = up + (size_t)row * n_in;
    const float *xb = x + (size_t)b * n_in;
    float gs = 0.0f, us = 0.0f;
    for (int i = lane; i < n_in; i += HIP_WAVE) {
        float xv = xb[i];
        gs += xv * gpu_bf16_to_f32(gr[i]);
        us += xv * gpu_bf16_to_f32(ur[i]);
    }
    gs = wave_sum(gs);
    us = wave_sum(us);
    if (lane == 0) {
        float *yb = y + (size_t)b * d_out;
        yb[row] = (gs / (1.0f + expf(-gs))) * us;
    }
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
