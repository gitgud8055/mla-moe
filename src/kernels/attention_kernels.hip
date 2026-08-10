#include "getp_kernels.h"
#include "getp_kernels_device.h"

__global__ void rope_copy_kernel(const float *src, float *dst, int heads,
                                 int src_stride, int src_offset, int rope_dim,
                                 int pos, const float *inv_freq, int interleaved) {
    int h = (int)blockIdx.x;
    int j = (int)threadIdx.x;
    int half = rope_dim / 2;
    if (h >= heads || j >= half) return;
    const float *v = src + (size_t)h * src_stride + src_offset;
    float *o = dst + (size_t)h * rope_dim;
    float angle = (float)pos * inv_freq[j];
    float co = cosf(angle), si = sinf(angle);
    float even = v[2 * j], odd = v[2 * j + 1];
    if (!interleaved) {
        o[2 * j] = even * co - odd * si;
        o[2 * j + 1] = even * si + odd * co;
    } else {
        o[j] = even * co - odd * si;
        o[half + j] = odd * co + even * si;
    }
}

__global__ void softmax_rows_kernel(float *x, int rows, int n, int stride) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    if (row >= rows) return;

    float *xr = x + (size_t)row * stride;

    float mx = -INFINITY;
    for (int i = lane; i < n; i += HIP_WAVE) {
        mx = fmaxf(mx, xr[i]);
    }
    mx = wave_max(mx);
    mx = __shfl(mx, 0, HIP_WAVE);

    float sum = 0.0f;
    for (int i = lane; i < n; i += HIP_WAVE) {
        float v = xr[i];
        v = (v == -INFINITY) ? 0.0f : expf(v - mx);
        xr[i] = v;
        sum += v;
    }
    sum = wave_sum(sum);
    float inv = 1.0f / __shfl(sum, 0, HIP_WAVE);

    for (int i = lane; i < n; i += HIP_WAVE) {
        xr[i] *= inv;
    }
}

/* q_absorbed[h,r] = sum_d q_nope[h,d] * W_UK[h,d,r]. */
__global__ void q_absorb_kernel(float *qabs, const float *q, const bf16_t *wuk,
                                int heads, int q_head_dim, int qk_nope,
                                int kv_rank, size_t head_stride) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int total = heads * kv_rank;
    if (row >= total) return;
    int h = row / kv_rank, r = row - h * kv_rank;
    const float *qh = q + (size_t)h * q_head_dim;
    const bf16_t *wh = wuk + (size_t)h * head_stride;
    float sum = 0.0f;
    for (int d = lane; d < qk_nope; d += HIP_WAVE)
        sum += qh[d] * gpu_bf16_to_f32(wh[(size_t)d * kv_rank + r]);
    sum = wave_sum(sum);
    if (lane == 0) qabs[row] = sum;
}

__global__ void attention_score_kernel(float *scores, const float *qabs,
                                       const float *qrope, const float *cache,
                                       int heads, int kv_len, int kv_dim,
                                       int kv_rank, int rope_dim, float scale) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int total = heads * kv_len;
    if (row >= total) return;
    int h = row / kv_len, k = row - h * kv_len;
    const float *qa = qabs + (size_t)h * kv_rank;
    const float *qp = qrope + (size_t)h * rope_dim;
    const float *ckv = cache + (size_t)k * kv_dim;
    float sn = 0.0f, sp = 0.0f;
    for (int r = lane; r < kv_rank; r += HIP_WAVE) sn += qa[r] * ckv[r];
    for (int d = lane; d < rope_dim; d += HIP_WAVE) sp += qp[d] * ckv[kv_rank + d];
    sn = wave_sum(sn);
    sp = wave_sum(sp);
    if (lane == 0) scores[row] = (sn + sp) * scale;
}

__global__ void latent_context_kernel(float *clat, const float *scores,
                                      const float *cache, int heads, int kv_len,
                                      int kv_dim, int kv_rank) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int total = heads * kv_rank;
    if (row >= total) return;
    int h = row / kv_rank, r = row - h * kv_rank;
    const float *a = scores + (size_t)h * kv_len;
    float sum = 0.0f;
    for (int k = lane; k < kv_len; k += HIP_WAVE)
        sum += a[k] * cache[(size_t)k * kv_dim + r];
    sum = wave_sum(sum);
    if (lane == 0) clat[row] = sum;
}

__global__ void value_up_kernel(float *ctx, const float *clat, const bf16_t *wuv,
                                int heads, int value_dim, int kv_rank,
                                size_t head_stride) {
    int wave = (int)threadIdx.x / HIP_WAVE;
    int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    int row = (int)blockIdx.x * HIP_ROWS_PER_BLOCK + wave;
    int total = heads * value_dim;
    if (row >= total) return;
    int h = row / value_dim, d = row - h * value_dim;
    const float *ch = clat + (size_t)h * kv_rank;
    const bf16_t *wh = wuv + (size_t)h * head_stride + (size_t)d * kv_rank;
    float sum = 0.0f;
    for (int r = lane; r < kv_rank; r += HIP_WAVE)
        sum += ch[r] * gpu_bf16_to_f32(wh[r]);
    sum = wave_sum(sum);
    if (lane == 0) ctx[row] = sum;
}
