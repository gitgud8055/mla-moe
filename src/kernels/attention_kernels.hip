#include "getp_kernels.h"
#include "getp_kernels_device.h"

namespace attention_kernels
{

    template <int THREADS>
    __global__ void rope_copy_kernel(const float *src, float *dst, int heads,
                                     int src_stride, int src_offset, int rope_dim,
                                     int pos, const float *inv_freq, int interleaved)
    {
        int h = (int)blockIdx.x;
        int j = (int)threadIdx.x;
        int half = rope_dim / 2;
        if (h >= heads || j >= half)
            return;
        const float *v = src + (size_t)h * src_stride + src_offset;
        float *o = dst + (size_t)h * rope_dim;
        float angle = (float)pos * inv_freq[j];
        float co = cosf(angle), si = sinf(angle);
        float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved)
        {
            o[2 * j] = even * co - odd * si;
            o[2 * j + 1] = even * si + odd * co;
        }
        else
        {
            o[j] = even * co - odd * si;
            o[half + j] = odd * co + even * si;
        }
    }

    template <int THREADS>
    __global__ void softmax_rows_kernel(float *x, int rows, int n, int stride,
                                        int causal_span)
    {
        __shared__ float scratch[THREADS];
        int row = (int)blockIdx.x;
        if (row >= rows)
            return;
        if (causal_span > 0)
            n = row % causal_span + 1;
        float *xr = x + (size_t)row * stride;
        float mx = -INFINITY;
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            mx = fmaxf(mx, xr[i]);
        scratch[threadIdx.x] = mx;
        __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1)
        {
            if ((int)threadIdx.x < s)
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
            __syncthreads();
        }
        mx = scratch[0];
        float sum = 0.0f;
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
        {
            float v = expf(xr[i] - mx);
            xr[i] = v;
            sum += v;
        }
        scratch[threadIdx.x] = sum;
        __syncthreads();
        for (int s = THREADS / 2; s > 0; s >>= 1)
        {
            if ((int)threadIdx.x < s)
                scratch[threadIdx.x] += scratch[threadIdx.x + s];
            __syncthreads();
        }
        float inv = 1.0f / scratch[0];
        for (int i = (int)threadIdx.x; i < n; i += THREADS)
            xr[i] *= inv;
    }

    /* q_absorbed[h,r] = sum_d q_nope[h,d] * W_UK[h,d,r]. */
    template <int THREADS>
    __global__ void q_absorb_rope_kernel(
        float *qabs, float *qrope, const float *q, const bf16_t *wuk_t,
        int heads, int q_head_dim, int qk_nope, int rope_dim,
        int kv_rank, int pos,
        const float *inv_freq, int interleaved)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int total = heads * kv_rank;
        if (row >= total)
            return;
        int h = row / kv_rank, r = row - h * kv_rank;
        const float *qh = q + (size_t)h * q_head_dim;
        const bf16_t *wr = wuk_t + ((size_t)h * kv_rank + r) * qk_nope;
        float sum = 0.0f;
        for (int d = lane; d < qk_nope; d += HIP_WAVE)
            sum += qh[d] * gpu_bf16_to_f32(wr[d]);
        sum = wave_sum(sum);
        if (lane == 0)
            qabs[row] = sum;

        if (r == 0 && lane < rope_dim / 2)
        {
            const float *v = qh + qk_nope;
            float *o = qrope + (size_t)h * rope_dim;
            float angle = (float)pos * inv_freq[lane];
            float co = cosf(angle), si = sinf(angle);
            float even = v[2 * lane], odd = v[2 * lane + 1];
            if (!interleaved)
            {
                o[2 * lane] = even * co - odd * si;
                o[2 * lane + 1] = even * si + odd * co;
            }
            else
            {
                int half = rope_dim / 2;
                o[lane] = even * co - odd * si;
                o[half + lane] = odd * co + even * si;
            }
        }
    }

    template <int THREADS>
    __global__ void batched_q_absorb_rope_kernel(
        float *qabs, float *qrope, const float *q, const bf16_t *wuk_t,
        int heads, int q_head_dim, int qk_nope, int rope_dim,
        int kv_rank, int pos_base,
        const float *inv_freq, int interleaved)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int b = (int)blockIdx.y;
        int total = heads * kv_rank;
        if (row >= total)
            return;

        float *qabs_b = qabs + (size_t)b * heads * kv_rank;
        float *qrope_b = qrope + (size_t)b * heads * rope_dim;
        const float *q_b = q + (size_t)b * heads * q_head_dim;
        int pos = pos_base + b;

        int h = row / kv_rank, r = row - h * kv_rank;
        const float *qh = q_b + (size_t)h * q_head_dim;
        const bf16_t *wr = wuk_t + ((size_t)h * kv_rank + r) * qk_nope;
        float sum = 0.0f;
        for (int d = lane; d < qk_nope; d += HIP_WAVE)
            sum += qh[d] * gpu_bf16_to_f32(wr[d]);
        sum = wave_sum(sum);
        if (lane == 0)
            qabs_b[row] = sum;

        if (r == 0 && lane < rope_dim / 2)
        {
            const float *v = qh + qk_nope;
            float *o = qrope_b + (size_t)h * rope_dim;
            float angle = (float)pos * inv_freq[lane];
            float co = cosf(angle), si = sinf(angle);
            float even = v[2 * lane], odd = v[2 * lane + 1];
            if (!interleaved)
            {
                o[2 * lane] = even * co - odd * si;
                o[2 * lane + 1] = even * si + odd * co;
            }
            else
            {
                int half = rope_dim / 2;
                o[lane] = even * co - odd * si;
                o[half + lane] = odd * co + even * si;
            }
        }
    }

    template <int THREADS>
    __global__ void attention_score_kernel(float *scores, const float *qabs,
                                           const float *qrope, const float *cache,
                                           int heads, int kv_len, int kv_dim,
                                           int kv_rank, int rope_dim, float scale)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int total = heads * kv_len;
        if (row >= total)
            return;
        int h = row / kv_len, k = row - h * kv_len;
        const float *qa = qabs + (size_t)h * kv_rank;
        const float *qp = qrope + (size_t)h * rope_dim;
        const float *ckv = cache + (size_t)k * kv_dim;
        float sn = 0.0f, sp = 0.0f;
        for (int r = lane; r < kv_rank; r += HIP_WAVE)
            sn += qa[r] * ckv[r];
        for (int d = lane; d < rope_dim; d += HIP_WAVE)
            sp += qp[d] * ckv[kv_rank + d];
        sn = wave_sum(sn);
        sp = wave_sum(sp);
        if (lane == 0)
            scores[row] = (sn + sp) * scale;
    }

    template <int THREADS>
    __global__ void latent_context_kernel(float *clat, const float *scores,
                                          const float *cache, int heads, int kv_len,
                                          int kv_dim, int kv_rank)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int total = heads * kv_rank;
        if (row >= total)
            return;
        int h = row / kv_rank, r = row - h * kv_rank;
        const float *a = scores + (size_t)h * kv_len;
        float sum = 0.0f;
        for (int k = lane; k < kv_len; k += HIP_WAVE)
            sum += a[k] * cache[(size_t)k * kv_dim + r];
        sum = wave_sum(sum);
        if (lane == 0)
            clat[row] = sum;
    }

    template <int THREADS>
    __global__ void value_up_kernel(float *ctx, const float *clat, const bf16_t *wuv,
                                    int heads, int value_dim, int kv_rank,
                                    size_t head_stride)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int total = heads * value_dim;
        if (row >= total)
            return;
        int h = row / value_dim, d = row - h * value_dim;
        const float *ch = clat + (size_t)h * kv_rank;
        const bf16_t *wh = wuv + (size_t)h * head_stride + (size_t)d * kv_rank;
        float sum = 0.0f;
        for (int r = lane; r < kv_rank; r += HIP_WAVE)
            sum += ch[r] * gpu_bf16_to_f32(wh[r]);
        sum = wave_sum(sum);
        if (lane == 0)
            ctx[row] = sum;
    }

    template <int THREADS>
    __global__ void batched_value_up_kernel(float *ctx, const float *clat, const bf16_t *wuv,
                                            int heads, int value_dim, int kv_rank,
                                            size_t head_stride)
    {
        static_assert(THREADS % HIP_WAVE == 0);
        int wave = (int)threadIdx.x / HIP_WAVE;
        int lane = (int)threadIdx.x & (HIP_WAVE - 1);
        int row = (int)blockIdx.x * (THREADS / HIP_WAVE) + wave;
        int b = (int)blockIdx.y;
        int total = heads * value_dim;
        if (row >= total)
            return;

        float *ctx_b = ctx + (size_t)b * heads * value_dim;
        const float *clat_b = clat + (size_t)b * heads * kv_rank;

        int h = row / value_dim, d = row - h * value_dim;
        const float *ch = clat_b + (size_t)h * kv_rank;
        const bf16_t *wh = wuv + (size_t)h * head_stride + (size_t)d * kv_rank;
        float sum = 0.0f;
        for (int r = lane; r < kv_rank; r += HIP_WAVE)
            sum += ch[r] * gpu_bf16_to_f32(wh[r]);
        sum = wave_sum(sum);
        if (lane == 0)
            ctx_b[row] = sum;
    }

    template <int THREADS>
    __global__ void kv_norm_rope_rows_kernel(
        float *cache, const float *comp, const bf16_t *norm,
        int rows, int kv_rank, int kv_dim, int rope_dim, int position_base,
        const float *inv_freq, int interleaved, float eps)
    {
        __shared__ float scratch[THREADS];
        int row = (int)blockIdx.x;
        if (row >= rows)
            return;
        const float *src = comp + (size_t)row * kv_dim;
        float *dst = cache + (size_t)row * kv_dim;
        float ss = 0.0f;
        for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
            ss += src[i] * src[i];
        scratch[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            scratch[0] = 1.0f / sqrtf(scratch[0] / (float)kv_rank + eps);
        __syncthreads();
        float inv = scratch[0];
        for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
            dst[i] = src[i] * inv * gpu_bf16_to_f32(norm[i]);

        int j = (int)threadIdx.x;
        int half = rope_dim / 2;
        if (j < half)
        {
            const float *v = src + kv_rank;
            float *o = dst + kv_rank;
            float angle = (float)(position_base + row) * inv_freq[j];
            float co = cosf(angle), si = sinf(angle);
            float even = v[2 * j], odd = v[2 * j + 1];
            if (!interleaved)
            {
                o[2 * j] = even * co - odd * si;
                o[2 * j + 1] = even * si + odd * co;
            }
            else
            {
                o[j] = even * co - odd * si;
                o[half + j] = odd * co + even * si;
            }
        }
    }

    template <int THREADS>
    __global__ void batched_kv_norm_rope_rows_kernel(
        float *cache, const float *comp, const bf16_t *norm,
        int kv_rank, int kv_dim, int rope_dim, int position_base,
        const float *inv_freq, int interleaved, float eps)
    {
        __shared__ float scratch[THREADS];
        int b = (int)blockIdx.y;
        const float *src = comp + (size_t)b * kv_dim;
        float *dst = cache + (size_t)b * kv_dim;

        float ss = 0.0f;
        for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
            ss += src[i] * src[i];
        scratch[threadIdx.x] = ss;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1)
        {
            if ((int)threadIdx.x < stride)
                scratch[threadIdx.x] += scratch[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            scratch[0] = 1.0f / sqrtf(scratch[0] / (float)kv_rank + eps);
        __syncthreads();
        float inv = scratch[0];
        for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS)
            dst[i] = src[i] * inv * gpu_bf16_to_f32(norm[i]);

        int j = (int)threadIdx.x;
        int half = rope_dim / 2;
        if (j < half)
        {
            const float *v = src + kv_rank;
            float *o = dst + kv_rank;
            float angle = (float)(position_base + b) * inv_freq[j];
            float co = cosf(angle), si = sinf(angle);
            float even = v[2 * j], odd = v[2 * j + 1];
            if (!interleaved)
            {
                o[2 * j] = even * co - odd * si;
                o[2 * j + 1] = even * si + odd * co;
            }
            else
            {
                o[j] = even * co - odd * si;
                o[half + j] = odd * co + even * si;
            }
        }
    }

    template <int THREADS>
    __global__ void fused_prefill_mla_kernel(
        float *out_clat,
        const float *qabs,
        const float *qrope,
        const float *kv_cache,
        int rows, int heads, int KVL, int QKR, int KVD, float scale)
    {
        int row = blockIdx.x;
        int h = blockIdx.y;
        int lane = threadIdx.x;

        if (row >= rows)
            return;

        const float *qabs_h = qabs + ((size_t)row * heads + h) * KVL;
        const float *qrope_h = qrope + ((size_t)row * heads + h) * QKR;

        float m = -INFINITY;
        float l = 0.0f;

        float clat_acc[8] = {0};

        for (int k = 0; k <= row; ++k)
        {
            const float *ckv = kv_cache + (size_t)k * KVD;
            const float *kpe = ckv + KVL;

            float sum = 0.0f;
            for (int i = lane; i < KVL; i += HIP_WAVE)
            {
                sum += qabs_h[i] * ckv[i];
            }
            for (int i = lane; i < QKR; i += HIP_WAVE)
            {
                sum += qrope_h[i] * kpe[i];
            }

            sum = wave_sum(sum);
            sum = __shfl(sum, 0, HIP_WAVE);

            float s = sum * scale;
            float m_new = fmaxf(m, s);
            float exp_old = expf(m - m_new);
            float exp_new = expf(s - m_new);

            l = l * exp_old + exp_new;
            m = m_new;

            for (int i = 0; i < 8; ++i)
            {
                int idx = lane + i * HIP_WAVE;
                if (idx < KVL)
                {
                    clat_acc[i] = clat_acc[i] * exp_old + exp_new * ckv[idx];
                }
            }
        }

        float *out_h = out_clat + ((size_t)row * heads + h) * KVL;
        for (int i = 0; i < 8; ++i)
        {
            int idx = lane + i * HIP_WAVE;
            if (idx < KVL)
            {
                out_h[idx] = clat_acc[i] / l;
            }
        }
    }

} // namespace attention_kernels

template __global__ void attention_kernels::rope_copy_kernel<kernel_tuning::ATTENTION_THREADS>(
    const float *, float *, int, int, int, int, int, const float *, int);
template __global__ void attention_kernels::softmax_rows_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, int, int, int, int);
template __global__ void attention_kernels::q_absorb_rope_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, float *, const float *, const bf16_t *, int, int, int, int, int,
    int, const float *, int);
template __global__ void attention_kernels::batched_q_absorb_rope_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, float *, const float *, const bf16_t *, int, int, int, int, int,
    int, const float *, int);
template __global__ void attention_kernels::attention_score_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const float *, const float *, int, int, int, int,
    int, float);
template __global__ void attention_kernels::latent_context_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const float *, int, int, int, int);
template __global__ void attention_kernels::value_up_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const bf16_t *, int, int, int, size_t);
template __global__ void attention_kernels::batched_value_up_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const bf16_t *, int, int, int, size_t);
template __global__ void attention_kernels::kv_norm_rope_rows_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const bf16_t *, int, int, int, int, int,
    const float *, int, float);
template __global__ void attention_kernels::batched_kv_norm_rope_rows_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const bf16_t *, int, int, int, int,
    const float *, int, float);
template __global__ void attention_kernels::fused_prefill_mla_kernel<kernel_tuning::ATTENTION_THREADS>(
    float *, const float *, const float *, const float *, int, int, int, int,
    int, float);
