/* Batched token decode: kernels that only run during decode plus
 * decode::run(), one decode step over every active sequence. */
#ifndef MLA_DECODE_H
#define MLA_DECODE_H

#include "ops.h"

namespace decode {

namespace kt = utils::constants::kernel;

__global__ void kv_norm_rope(bf16_t *cache, const float *comp, const bf16_t *norm,
                             const int *positions, int batch, int capacity, int kv_rank,
                             int kv_dim, int rope_dim, const float *inv_freq,
                             int interleaved, float eps) {
    __shared__ float scratch[ops::THREADS];
    int b = (int)blockIdx.x;
    if (b >= batch) return;
    const int pos = positions[b];
    const float *src = comp + (size_t)b * kv_dim;
    bf16_t *dst = cache + ((size_t)b * capacity + pos) * kv_dim;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_rank; i += ops::THREADS)
        ss += src[i] * src[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int s = ops::THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(scratch[0] / (float)kv_rank + eps);
    for (int i = (int)threadIdx.x; i < kv_rank; i += ops::THREADS)
        dst[i] = gpu_f32_to_bf16(src[i] * inv * gpu_bf16_to_f32(norm[i]));
    int j = (int)threadIdx.x;
    if (j < rope_dim / 2) {
        const float *v = src + kv_rank;
        bf16_t *o = dst + kv_rank;
        float angle = (float)pos * inv_freq[j];
        float co = cosf(angle), si = sinf(angle);
        float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved) {
            o[2*j] = gpu_f32_to_bf16(even*co - odd*si);
            o[2*j+1] = gpu_f32_to_bf16(even*si + odd*co);
        } else {
            int half = rope_dim / 2;
            o[j] = gpu_f32_to_bf16(even*co - odd*si);
            o[half+j] = gpu_f32_to_bf16(odd*co + even*si);
        }
    }
}

/* Fused absorbed-q projection + q RoPE for tiny batches (one wave per row). */
__global__ void q_absorb_rope(float *qabs, float *qrope, const float *q,
                              const bf16_t *wuk_t, const int *positions,
                              int batch, int heads,
                              int q_head_dim, int qk_nope, int rope_dim,
                              int kv_rank, const float *inv_freq,
                              int interleaved) {
    int wave=(int)threadIdx.x/HIP_WAVE, lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*ops::WAVES+wave, total=heads*kv_rank;
    int b0=(int)blockIdx.y*ops::BATCH_TILE;
    if (row>=total || b0>=batch) return;
    int h=row/kv_rank, r=row-h*kv_rank;
    const bf16_t *wr=wuk_t+((size_t)h*kv_rank+r)*qk_nope;
    float sums[ops::BATCH_TILE]={};
    for (int d=lane; d<qk_nope; d+=HIP_WAVE) {
        float wi=gpu_bf16_to_f32(wr[d]);
#pragma unroll
        for(int j=0;j<ops::BATCH_TILE;++j) if(b0+j<batch)
            sums[j]+=q[((size_t)(b0+j)*heads+h)*q_head_dim+d]*wi;
    }
#pragma unroll
    for(int j=0;j<ops::BATCH_TILE;++j) {
        sums[j]=wave_sum(sums[j]);
        if(lane==0 && b0+j<batch) qabs[((size_t)(b0+j)*heads+h)*kv_rank+r]=sums[j];
    }
    if(r==0 && lane<rope_dim/2) {
#pragma unroll
        for(int j=0;j<ops::BATCH_TILE;++j) if(b0+j<batch) {
            const float *v=q+((size_t)(b0+j)*heads+h)*q_head_dim+qk_nope;
            float *o=qrope+((size_t)(b0+j)*heads+h)*rope_dim;
            float angle=(float)positions[b0+j]*inv_freq[lane], co=cosf(angle), si=sinf(angle);
            float even=v[2*lane], odd=v[2*lane+1];
            if(!interleaved){o[2*lane]=even*co-odd*si;o[2*lane+1]=even*si+odd*co;}
            else{int half=rope_dim/2;o[lane]=even*co-odd*si;o[half+lane]=odd*co+even*si;}
        }
    }
}

__global__ void q_rope(float *qrope, const float *q, const int *positions,
                       int batch, int heads, int q_head_dim, int qk_nope, int rope_dim,
                       const float *inv_freq, int interleaved) {
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int row = (int)blockIdx.x * ops::WAVES + wave;
    if (row >= batch * heads || lane >= rope_dim / 2) return;
    const int b = row / heads;
    const float *v = q + (size_t)row * q_head_dim + qk_nope;
    float *o = qrope + (size_t)row * rope_dim;
    const float angle = (float)positions[b] * inv_freq[lane];
    const float co = cosf(angle), si = sinf(angle);
    const float even = v[2 * lane], odd = v[2 * lane + 1];
    if (!interleaved) {
        o[2 * lane] = even * co - odd * si;
        o[2 * lane + 1] = even * si + odd * co;
    } else {
        const int half = rope_dim / 2;
        o[lane] = even * co - odd * si;
        o[half + lane] = odd * co + even * si;
    }
}

__global__ void attention_score(float *scores, const float *qabs,
                                const float *qrope, const bf16_t *cache,
                                const int *positions, int batch, int heads,
                                int max_kv_len, int capacity,
                                int kv_dim, int kv_rank, int rope_dim, float scale) {
    int wave=(int)threadIdx.x/HIP_WAVE, lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*ops::WAVES+wave, b=(int)blockIdx.y;
    if(row>=heads*max_kv_len || b>=batch)return;
    int h=row/max_kv_len,k=row-h*max_kv_len;
    const int kv_len=positions[b]+1;
    if(k>=kv_len){if(lane==0)scores[((size_t)b*heads+h)*capacity+k]=-INFINITY;return;}
    const float *qa=qabs+((size_t)b*heads+h)*kv_rank;
    const float *qp=qrope+((size_t)b*heads+h)*rope_dim;
    const bf16_t *ckv=cache+((size_t)b*capacity+k)*kv_dim;
    float sn=0.0f,sp=0.0f;
    for(int r=lane;r<kv_rank;r+=HIP_WAVE)sn+=qa[r]*gpu_bf16_to_f32(ckv[r]);
    for(int d=lane;d<rope_dim;d+=HIP_WAVE)sp+=qp[d]*gpu_bf16_to_f32(ckv[kv_rank+d]);
    sn=wave_sum(sn);sp=wave_sum(sp);
    if(lane==0)scores[((size_t)b*heads+h)*capacity+k]=(sn+sp)*scale;
}

__global__ void softmax_rows(float *x, int rows, int n, int stride) {
    __shared__ float scratch[kt::ATTENTION_THREADS];
    constexpr int T = kt::ATTENTION_THREADS;
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    float *xr = x + (size_t)row * stride;
    float mx = -INFINITY;
    for (int i = (int)threadIdx.x; i < n; i += T)
        mx = fmaxf(mx, xr[i]);
    scratch[threadIdx.x] = mx;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    mx = scratch[0];
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < n; i += T) {
        float v = expf(xr[i] - mx);
        xr[i] = v;
        sum += v;
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int s = T / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / scratch[0];
    for (int i = (int)threadIdx.x; i < n; i += T)
        xr[i] *= inv;
}

/* Flash decode for absorbed MLA. One wave owns one (sequence, head) row and
 * streams the latent KV cache once; online softmax keeps only running
 * max/denominator plus the 512-wide latent context in registers. The loop
 * bound depends only on blockIdx.y (uniform), keeping the in-loop barrier
 * convergent for head counts that do not divide the wave count. */
template <int KV_TILE>
__global__ void flash_attention(float *__restrict__ clat,
                             const float *__restrict__ qabs,
                             const float *__restrict__ qrope,
                             const bf16_t *__restrict__ cache,
                             const int *__restrict__ positions,
                             int batch, int heads, int capacity,
                             int kv_dim, int kv_rank, int rope_dim, float scale) {
    extern __shared__ bf16_t shared_cache[];
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int waves = (int)blockDim.x / HIP_WAVE;
    const int b = (int)blockIdx.y;
    const int h = (int)blockIdx.x * waves + wave;
    const bool active = b < batch && h < heads;
    const int kv_len = b < batch ? positions[b] + 1 : 0;
    const float *qa = active
        ? qabs + ((size_t)b * heads + h) * kv_rank : nullptr;
    const float *qp = active
        ? qrope + ((size_t)b * heads + h) * rope_dim : nullptr;
    const bf16_t *cache_b = cache + (size_t)b * capacity * kv_dim;
    constexpr int VALUES_PER_LANE = 8;
    float acc[VALUES_PER_LANE] = {};
    float running_max = -INFINITY;
    float denominator = 0.0f;
    float qreg[VALUES_PER_LANE];
#pragma unroll
    for (int q = 0; q < VALUES_PER_LANE; ++q) {
        const int r = lane + q * HIP_WAVE;
        qreg[q] = (active && r < kv_rank) ? qa[r] : 0.0f;
    }
    const float qpr = (active && lane < rope_dim) ? qp[lane] : 0.0f;
    const int rope_lane = lane < rope_dim ? kv_rank + lane : kv_rank;

    for (int k0 = 0; k0 < kv_len; k0 += KV_TILE) {
        const int tile = min(KV_TILE, kv_len - k0);
        for (int i = (int)threadIdx.x; i < tile * kv_dim;
             i += (int)blockDim.x)
            shared_cache[i] = cache_b[(size_t)k0 * kv_dim + i];
        __syncthreads();

        float scores[KV_TILE];
        for (int t = 0; t < KV_TILE; ++t) {
            float s = 0.0f;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                s += qreg[q] * gpu_bf16_to_f32(ckv[r]);
            }
            s += qpr * gpu_bf16_to_f32(ckv[rope_lane]);
            s = wave_sum(s);
            scores[t] = t < tile
                ? __shfl(s * scale, 0, HIP_WAVE)
                : -INFINITY;
        }

        float tile_max = scores[0];
        for (int t = 1; t < KV_TILE; ++t)
            tile_max = fmaxf(tile_max, scores[t]);
        const float next_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - next_max);
        denominator *= rescale;
#pragma unroll
        for (int q = 0; q < VALUES_PER_LANE; ++q)
            acc[q] *= rescale;
        for (int t = 0; t < KV_TILE; ++t) {
            if (t >= tile) break;
            const float weight = expf(scores[t] - next_max);
            denominator += weight;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                acc[q] += gpu_bf16_to_f32(ckv[r]) * weight;
            }
        }
        running_max = next_max;
        __syncthreads();
    }

    if (!active) return;
    const float inv_denominator = 1.0f / denominator;
#pragma unroll
    for (int q = 0; q < VALUES_PER_LANE; ++q) {
        const int r = lane + q * HIP_WAVE;
        if (r < kv_rank)
            clat[((size_t)b * heads + h) * kv_rank + r] =
                acc[q] * inv_denominator;
    }
}

__global__ void latent_context(float *clat, const float *scores, const bf16_t *cache,
                               const int *positions, int batch, int heads,
                               int capacity, int kv_dim, int kv_rank) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*ops::WAVES+wave,b=(int)blockIdx.y;
    if(row>=heads*kv_rank||b>=batch)return;
    int h=row/kv_rank,r=row-h*kv_rank;
    const int kv_len=positions[b]+1;
    const float *a=scores+((size_t)b*heads+h)*capacity;
    const bf16_t *cb=cache+(size_t)b*capacity*kv_dim;
    float sum=0.0f;
    for(int k=lane;k<kv_len;k+=HIP_WAVE)
        sum+=a[k]*gpu_bf16_to_f32(cb[(size_t)k*kv_dim+r]);
    sum=wave_sum(sum);
    if(lane==0)clat[((size_t)b*heads+h)*kv_rank+r]=sum;
}

__global__ void value_up(float *ctx, const float *clat, const bf16_t *wuv,
                         int batch, int heads, int value_dim, int kv_rank,
                         size_t head_stride) {
    int wave=(int)threadIdx.x/HIP_WAVE,lane=(int)threadIdx.x&(HIP_WAVE-1);
    int row=(int)blockIdx.x*ops::WAVES+wave,b0=(int)blockIdx.y*ops::BATCH_TILE;
    if(row>=heads*value_dim||b0>=batch)return;
    int h=row/value_dim,d=row-h*value_dim;
    const bf16_t *wr=wuv+(size_t)h*head_stride+(size_t)d*kv_rank;
    float sums[ops::BATCH_TILE]={};
    for(int r=lane;r<kv_rank;r+=HIP_WAVE){
        float wi=gpu_bf16_to_f32(wr[r]);
#pragma unroll
        for(int j=0;j<ops::BATCH_TILE;++j)if(b0+j<batch)
            sums[j]+=clat[((size_t)(b0+j)*heads+h)*kv_rank+r]*wi;
    }
#pragma unroll
    for(int j=0;j<ops::BATCH_TILE;++j){sums[j]=wave_sum(sums[j]);if(lane==0&&b0+j<batch)
        ctx[((size_t)(b0+j)*heads+h)*value_dim+d]=sums[j];}
}

/* One decode step: one token per active sequence. `positions[b]` is
 * independent per sequence, so prompts of different lengths share a batch. */
inline void run(GpuContext &g, const Config &c, int max_batch,
                const int *device_tokens, int *generated,
                int generated_stride, int output_index,
                const int *positions, int max_kv_len, int batch) {
    const int H = c.hidden_size;
    const int NH = c.n_heads;
    const int QKN = c.qk_nope_head_dim;
    const int QKR = c.qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int QD = NH * QHD;
    const int KVL = c.kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VHD = c.v_head_dim;
    const size_t head_stride = (size_t)(QKN + VHD) * KVL;
    if (batch < 1 || batch > max_batch || max_kv_len < 1 ||
        max_kv_len > g.kv_capacity) {
        std::fprintf(stderr, "HIP token decode: batch=%d/%d max_kv=%d/%d\n",
                     batch, max_batch, max_kv_len, g.kv_capacity);
        std::exit(EXIT_FAILURE);
    }
    {
        namespace md = utils::constants::model;
        const bool dsv_shape = NH == md::dsv::N_HEADS &&
            H == md::dsv::HIDDEN && QKN == md::dsv::QK_NOPE &&
            QKR == md::dsv::QK_ROPE && VHD == md::dsv::V_HEAD &&
            KVL == md::dsv::KV_LORA && c.q_lora_rank == md::dsv::Q_LORA;
        const bool glm_shape = NH == md::glm::N_HEADS &&
            H == md::glm::HIDDEN && QKN == md::glm::QK_NOPE &&
            QKR == md::glm::QK_ROPE && VHD == md::glm::V_HEAD &&
            KVL == md::glm::KV_LORA && c.q_lora_rank == md::glm::Q_LORA;
        if (!dsv_shape && !glm_shape)
            ops::unlisted_shape("decode::run", NH, QKN, VHD);
    }
    const ops::QuantScratch qs{g.act_i8, g.act_s};

    ops::embedding(g.x, g.embed, device_tokens, generated, generated_stride,
                   output_index, H, batch, g.stream);

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        ops::rmsnorm(g.xn, g.x, d.input_norm, H, batch, c.rms_eps, g.stream);
        if (c.q_lora_rank > 0) {
            /* glm: q_a [B,768,2048] + kv_a [B,576,2048]; q_b [B,5120,768] */
            ops::gemm_dual(g.q_a, d.q_a_proj, d.q_a_proj_s, c.q_lora_rank,
                           g.comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                           g.xn, H, batch, qs, g.stream);
            ops::rmsnorm(g.q_a, g.q_a, d.q_a_norm, c.q_lora_rank, batch,
                         c.mla_norm_eps, g.stream);
            ops::gemm(g.q, g.q_a, d.q_b_proj, d.q_b_proj_s, batch, QD,
                      c.q_lora_rank, false, qs, g.stream);
        } else {
            /* dsv: q_proj [B,3072,2048] + kv_a [B,576,2048] */
            ops::gemm_dual(g.q, d.q_proj, d.q_proj_s, QD,
                           g.comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                           g.xn, H, batch, qs, g.stream);
        }
        bf16_t *layer_cache = g.kv_cache +
            (size_t)l * max_batch * g.kv_capacity * KVD;
        kv_norm_rope<<<batch, kt::ATTENTION_THREADS, 0, g.stream>>>(
            layer_cache, g.comp, d.kv_a_norm, positions, batch,
            g.kv_capacity, KVL, KVD, QKR, g.rope_inv_freq,
            c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        if (batch >= kt::MFMA_MIN_ROWS) {
            /* q-absorb: dsv [B,16h,512,192]; glm [B,20h,512,192] */
            ops::head_gemm(g.qabs, g.q, d.W_UK_T, batch, NH, KVL, QKN,
                           QHD, KVL, (size_t)KVL * QKN, g.stream);
            q_rope<<<ops::div_up(batch * NH,
                                 kt::ATTENTION_THREADS / HIP_WAVE),
                kt::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qrope, g.q, positions, batch, NH, QHD, QKN, QKR,
                    g.rope_inv_freq, c.rope_interleaved);
        } else {
            q_absorb_rope<<<ops::gemv_grid(NH * KVL, batch),
                kt::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qabs, g.qrope, g.q, d.W_UK_T, positions, batch, NH,
                    QHD, QKN, QKR, KVL, g.rope_inv_freq, c.rope_interleaved);
        }
        HIP_LAUNCH_CHECK();

        if (batch >= kt::FLASH_MIN_ROWS) {
            namespace md = utils::constants::model;
            const auto launch_small = [&] { /* 512 threads, KV tile 16 */
                dim3 flash_grid((unsigned)ops::div_up(
                                    NH, kt::FLASH_SMALL_THREADS / HIP_WAVE),
                                (unsigned)batch);
                flash_attention<kt::FLASH_SMALL_KV_TILE><<<
                    flash_grid, kt::FLASH_SMALL_THREADS,
                    (size_t)kt::FLASH_SMALL_KV_TILE * KVD * sizeof(bf16_t),
                    g.stream>>>(
                        g.clat, g.qabs, g.qrope, layer_cache, positions,
                        batch, NH, g.kv_capacity, KVD, KVL, QKR,
                        c.softmax_scale);
            };
            const auto launch_large = [&] { /* 256 threads, KV tile 8 */
                dim3 flash_grid((unsigned)ops::div_up(
                                    NH, kt::FLASH_LARGE_THREADS / HIP_WAVE),
                                (unsigned)batch);
                flash_attention<kt::FLASH_LARGE_KV_TILE><<<
                    flash_grid, kt::FLASH_LARGE_THREADS,
                    (size_t)kt::FLASH_LARGE_KV_TILE * KVD * sizeof(bf16_t),
                    g.stream>>>(
                        g.clat, g.qabs, g.qrope, layer_cache, positions,
                        batch, NH, g.kv_capacity, KVD, KVL, QKR,
                        c.softmax_scale);
            };
            if (NH == md::dsv::N_HEADS) {
                /* DECODE dsv flash, every batch >= kt::FLASH_MIN_ROWS(8):
                   H=md::dsv::N_HEADS(16), KV=md::dsv::KV_DIM(576),
                   tile=kt::FLASH_SMALL_KV_TILE(16),
                   threads=kt::FLASH_SMALL_THREADS(512) — measured 1205 vs
                   1187 TPS against the 256thr/tile8 config at batch 512 */
                launch_small();
            } else if (NH == md::glm::N_HEADS) {
                if (batch >= kt::FLASH_LARGE_BATCH)
                    /* DECODE glm flash, B>=kt::FLASH_LARGE_BATCH(128):
                       H=md::glm::N_HEADS(20), KV=md::glm::KV_DIM(576),
                       tile=kt::FLASH_LARGE_KV_TILE(8),
                       threads=kt::FLASH_LARGE_THREADS(256) — measured 666
                       vs 662 TPS against 512thr/tile16 at batch 512 */
                    launch_large();
                else
                    /* DECODE glm flash, kt::FLASH_MIN_ROWS(8)<=B<128:
                       H=20, KV=576, tile=kt::FLASH_SMALL_KV_TILE(16),
                       threads=kt::FLASH_SMALL_THREADS(512) */
                    launch_small();
            } else {
                ops::unlisted_shape("decode flash_attention", NH, KVD, batch);
            }
            HIP_LAUNCH_CHECK();
        } else {
            int score_rows = NH * max_kv_len;
            dim3 score_grid((unsigned)ops::div_up(score_rows, ops::WAVES),
                            (unsigned)batch);
            attention_score<<<score_grid, kt::ATTENTION_THREADS, 0,
                g.stream>>>(
                    g.scores, g.qabs, g.qrope, layer_cache, positions, batch,
                    NH, max_kv_len, g.kv_capacity, KVD, KVL, QKR,
                    c.softmax_scale);
            HIP_LAUNCH_CHECK();
            softmax_rows<<<batch * NH, kt::ATTENTION_THREADS, 0, g.stream>>>(
                g.scores, batch * NH, max_kv_len, g.kv_capacity);
            HIP_LAUNCH_CHECK();
            dim3 latent_grid((unsigned)ops::div_up(NH * KVL, ops::WAVES),
                             (unsigned)batch);
            latent_context<<<latent_grid, kt::ATTENTION_THREADS, 0,
                g.stream>>>(
                    g.clat, g.scores, layer_cache, positions, batch, NH,
                    g.kv_capacity, KVD, KVL);
            HIP_LAUNCH_CHECK();
        }

        if (batch >= kt::MFMA_MIN_ROWS) {
            /* value-up: dsv [B,16h,128,512]; glm [B,20h,256,512] */
            ops::head_gemm(g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL,
                           KVL, VHD, head_stride, g.stream);
        } else {
            value_up<<<ops::gemv_grid(NH * VHD, batch),
                kt::ATTENTION_THREADS, 0, g.stream>>>(
                    g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL, head_stride);
            HIP_LAUNCH_CHECK();
        }
        ops::gemm(g.x, g.ctx, d.o_proj, d.o_proj_s, batch, H, NH * VHD, true,
                  qs, g.stream);

        ops::rmsnorm(g.xn, g.x, d.post_norm, H, batch, c.rms_eps, g.stream);
        if (l < c.first_k_dense)
            ops::dense_ffn(c, d, batch, g.x, g.xn, g.hb, qs, g.stream);
        else
            ops::moe_ffn(c, d, batch, g.x, g.xn, g.hb, g.router, g.topk,
                         g.topk_weights, g.moe_sorted_ids,
                         g.moe_expert_ids, g.moe_num_padded,
                         g.routed_hidden, nullptr, g.moe_route_out,
                         qs, g.routed_i8, g.routed_s, g.stream);
    }

    ops::rmsnorm(g.xn, g.x, g.final_norm, H, batch, c.rms_eps, g.stream);
    ops::gemm(g.logits, g.xn, g.lm_head, g.lm_head_s, batch, c.vocab_size, H,
              false, qs, g.stream);
    ops::argmax(g.logits, c.vocab_size, g.next_token, batch, g.stream);
}

} // namespace decode

#endif /* MLA_DECODE_H */
