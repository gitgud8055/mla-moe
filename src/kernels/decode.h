/* Batched token decode: kernels that only run during decode plus
 * decode::run(), one decode step over every active sequence. */
#ifndef MLA_DECODE_H
#define MLA_DECODE_H

#include "ops.h"

namespace decode {

namespace kt = utils::constants::kernel;
using utils::types::bf16x4;
using utils::types::bf16x8;
using utils::types::fp32v4;

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

/* MFMA absorbed-MLA decode attention (FlashAttention-2 on matrix cores).
 * One block per sequence handles ALL heads: the latent cache C is read once
 * and shared, S = Q@C^T and Ctx = P@C[:,:rank] run on v_mfma_f32_16x16x16bf16
 * instead of the scalar kernel's per-head dot-product + per-key wave_sum.
 * WV(8) waves; heads split into MT 16-row M-tiles, each tile's lead wave
 * computes S + online-softmax and publishes P + rescale/denominator to LDS,
 * then every wave accumulates a disjoint CTW-wide slab of the rank-wide
 * context (o_acc stays small). Removes the scalar kernel's cache re-read AND
 * its VALU/latency bottleneck. Block-uniform kv_len keeps barriers
 * convergent. Caller guarantees batch >= 1. */
template <int HEADS, int KV_RANK, int ROPE_DIM>
__global__ void flash_attention_mfma(
    float *__restrict__ clat, const float *__restrict__ qabs,
    const float *__restrict__ qrope, const bf16_t *__restrict__ cache,
    const int *__restrict__ positions, int batch, int capacity, float scale) {
    constexpr int KV_DIM = KV_RANK + ROPE_DIM;   /* 576 */
    constexpr int BK = 16;                        /* keys per MFMA step */
    constexpr int KP = 4;                         /* LDS bank pad       */
    constexpr int KT = KV_DIM / 16;               /* 36 score k-tiles   */
    constexpr int CT = KV_RANK / 16;              /* 32 context col-tiles */
    constexpr int MT = (HEADS + 15) / 16;         /* M-tiles: dsv 1, glm 2 */
    constexpr int WV = 8;                          /* waves per block     */
    constexpr int GRP = WV / MT;                   /* waves per M-tile     */
    constexpr int CTW = CT / GRP;                  /* context c-tiles/wave */
    /* Score k-range split across SPL waves per M-tile. Holding all KT=36 Q
     * fragments live across the kv loop costs 72 VGPRs, which blew the 128-VGPR
     * budget of 2 blocks/CU and spilled 44 (dsv) / 56 (glm) of them to scratch
     * *inside* the loop; it also left 1 of 8 waves doing every score MFMA while
     * the rest idled at the barrier. Splitting halves q_reg and the score
     * critical path for one extra partial-sum barrier. */
    constexpr int SPL = 2;                         /* waves per score tile */
    constexpr int KTW = KT / SPL;                  /* score k-tiles per wave */
    static_assert(KV_DIM % 16 == 0 && KV_RANK % 16 == 0, "MFMA tiles");
    static_assert(WV % MT == 0 && CT % GRP == 0, "even split");
    static_assert(SPL >= 2 && GRP >= SPL && KT % SPL == 0, "score split");

    const int b = (int)blockIdx.x;
    if (b >= batch) return;
    const int kv_len = positions[b] + 1;
    const bf16_t *cache_b = cache + (size_t)b * capacity * KV_DIM;

    /* Single staging buffer sk [key][k]. The context MFMA's B operand wants
     * V^T [col][key], which used to be re-read from global into a second LDS
     * layout — that re-read was a strict SUBSET of sk (c < KV_RANK <= KV_DIM),
     * i.e. 8192 of the 17408 staged elements per tile were pure duplicate HBM
     * traffic plus 8192 transposing ds_write_b16. Instead the B fragment is
     * gathered straight out of sk with 4 strided ds_read_u16 (2-way bank
     * conflict, ~32 cycles/wave/tile). Drops one barrier and 2KB of LDS. */
    constexpr int SKW = KV_DIM + KP;            /* sk row stride  */
    static_assert(KV_DIM % 8 == 0 && SKW % 4 == 0, "vectorized staging");
    __shared__ bf16_t sbuf[BK * SKW];           /* sk [key][k] */
    __shared__ bf16_t sp[MT][16][BK + KP];      /* P [mtile][head][key] */
    __shared__ float sresc[MT][16];             /* per-head rescale     */
    __shared__ float sl[MT][16];                /* per-head running sum */
    __shared__ float sm[MT][16];                /* per-head running max */
    __shared__ float spart[MT][SPL - 1][16][16]; /* score partials -> lead   */

    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int a_row = lane & 15;                 /* MFMA A-row / C-col   */
    const int k4 = 4 * (lane >> 4);              /* MFMA A/B k offset    */
    const int mtile = wave / GRP;                /* this wave's 16-head tile */
    const int local = wave % GRP;                /* wave within the tile */
    const int c0 = local * CTW;                  /* first context c-tile */
    const bool lead = (local == 0);              /* computes score for mtile */
    const int h_base = mtile * 16;

    /* Q registers: wave `local` (< SPL) owns k-tiles [local*KTW, +KTW) of
     * Q[h_base+a_row][k] in A-layout. */
    const bool scorer = (local < SPL);
    bf16x4 q_reg[KTW];
    if (scorer) {
#pragma unroll
        for (int t = 0; t < KTW; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = (local * KTW + t) * 16 + k4 + j;
                const int h = h_base + a_row;
                float f = 0.0f;
                if (h < HEADS)
                    f = k < KV_RANK
                        ? qabs[((size_t)b * HEADS + h) * KV_RANK + k]
                        : qrope[((size_t)b * HEADS + h) * ROPE_DIM +
                                (k - KV_RANK)];
                q_reg[t][j] = (short)gpu_f32_to_bf16(f);
            }
    }
    if (lead && lane < 16) { sm[mtile][lane] = -INFINITY; sl[mtile][lane] = 0.0f; }
    fp32v4 o_acc[CTW] = {};
    __syncthreads();

    for (int k0 = 0; k0 < kv_len; k0 += BK) {
        /* stage sk [key][k]: move 8 bf16 per step (one global dwordx4 + two
         * ds_write_b64) instead of one ushort — 18+18 memory ops per thread
         * become ~2.3+4.6. cache rows are 1152B apart so the source is 16B
         * aligned; SKW*2=1160 leaves the LDS side 8B aligned, hence b64 pairs. */
        constexpr int KCH = KV_DIM / 8;         /* 8-wide chunks per key */
        for (int i = (int)threadIdx.x; i < BK * KCH; i += (int)blockDim.x) {
            const int key = i / KCH, k = (i - key * KCH) * 8;
            short v[8] = {};
            if (k0 + key < kv_len)
                *reinterpret_cast<bf16x8 *>(v) =
                    *reinterpret_cast<const bf16x8 *>(
                        &cache_b[(size_t)(k0 + key) * KV_DIM + k]);
            bf16_t *dst = &sbuf[key * SKW + k];
            *reinterpret_cast<bf16x4 *>(dst) =
                *reinterpret_cast<const bf16x4 *>(v);
            *reinterpret_cast<bf16x4 *>(dst + 4) =
                *reinterpret_cast<const bf16x4 *>(v + 4);
        }
        __syncthreads();

        /* every scorer wave: partial S over its own k-tiles. C-layout:
         * sacc[i] = S[row=4*(lane>>4)+i, col=a_row(=key)] */
        fp32v4 sacc = {};
        if (scorer) {
#pragma unroll
            for (int t = 0; t < KTW; ++t)
                sacc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    q_reg[t],
                    *reinterpret_cast<const bf16x4 *>(
                        &sbuf[a_row * SKW + (local * KTW + t) * 16 + k4]),
                    sacc, 0, 0, 0);
            if (!lead)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    spart[mtile][local - 1][4 * (lane >> 4) + i][a_row] =
                        sacc[i];
        }
        __syncthreads();   /* partials visible to the lead wave */

        if (lead) {
            const int key_pos = k0 + a_row;
#pragma unroll
            for (int s = 0; s < SPL - 1; ++s)
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    sacc[i] += spart[mtile][s][4 * (lane >> 4) + i][a_row];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = 4 * (lane >> 4) + i;   /* head within tile */
                float sv = sacc[i] * scale;
                if (key_pos >= kv_len) sv = -INFINITY;
                float mx = sv;
                for (int d = 1; d < 16; d <<= 1)
                    mx = fmaxf(mx, __shfl_xor(mx, d, HIP_WAVE));
                const float m_old = sm[mtile][row];
                const float m_new = fmaxf(m_old, mx);
                const float resc = m_new == -INFINITY ? 1.0f
                                                      : expf(m_old - m_new);
                const float p = m_new == -INFINITY ? 0.0f : expf(sv - m_new);
                float ps = p;
                for (int d = 1; d < 16; d <<= 1)
                    ps += __shfl_xor(ps, d, HIP_WAVE);
                sp[mtile][row][a_row] = gpu_f32_to_bf16(p);
                if (a_row == 0) {
                    sm[mtile][row] = m_new;
                    sl[mtile][row] = sl[mtile][row] * resc + ps;
                    sresc[mtile][row] = resc;
                }
            }
        }
        __syncthreads();   /* P + stats visible; sk stays live for the B gather */

        /* every wave: rescale its o_acc slab, then O += P @ V for its c-tiles.
         * B[k=key][col=c] is gathered column-wise out of sk: keys k4..k4+3 of
         * latent column c live at sk[k4+j][c], stride SKW. */
#pragma unroll
        for (int ct = 0; ct < CTW; ++ct) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                o_acc[ct][i] *= sresc[mtile][4 * (lane >> 4) + i];
            const int c = (c0 + ct) * 16 + a_row;
            bf16x4 bv;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                bv[j] = (short)sbuf[(k4 + j) * SKW + c];
            o_acc[ct] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                *reinterpret_cast<const bf16x4 *>(&sp[mtile][a_row][k4]),
                bv, o_acc[ct], 0, 0, 0);
        }
        __syncthreads();   /* context done reading sk before it is restaged */
    }

    /* epilogue: divide by running sum, store (C-layout) */
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = 4 * (lane >> 4) + i;
        const int h = h_base + row;
        if (h >= HEADS) continue;
        const float inv = sl[mtile][row] > 0.0f ? 1.0f / sl[mtile][row] : 0.0f;
#pragma unroll
        for (int ct = 0; ct < CTW; ++ct)
            clat[((size_t)b * HEADS + h) * KV_RANK + (c0 + ct) * 16 + a_row] =
                o_acc[ct][i] * inv;
    }
}

/* Wrapper for the decode flash kernel — same convention as the ops.h
 * ladders: keyed on the FULL shape tuple (heads, kv_dim, kv_rank, rope) x
 * batch regime, one branch per real case, unlisted shapes abort. dsv and
 * glm share every attention dim except the head count (absorbed MLA runs
 * in the 512-dim latent + 64 rope for both); each branch spells out all
 * dims so every case can be tuned independently. Caller guarantees
 * batch >= kt::FLASH_MIN_ROWS. */
inline void flash_attention_dispatch(
    float *clat, const float *qabs, const float *qrope, const bf16_t *cache,
    const int *positions, int batch, int heads, int capacity, int kv_dim,
    int kv_rank, int rope_dim, float scale, hipStream_t stream) {
    namespace md = utils::constants::model;
    const auto launch_small = [&] { /* 512 threads, KV tile 16 */
        dim3 grid((unsigned)ops::div_up(
                      heads, kt::FLASH_SMALL_THREADS / HIP_WAVE),
                  (unsigned)batch);
        flash_attention<kt::FLASH_SMALL_KV_TILE><<<
            grid, kt::FLASH_SMALL_THREADS,
            (size_t)kt::FLASH_SMALL_KV_TILE * kv_dim * sizeof(bf16_t),
            stream>>>(clat, qabs, qrope, cache, positions, batch, heads,
                      capacity, kv_dim, kv_rank, rope_dim, scale);
        HIP_LAUNCH_CHECK();
    };
    if (heads == md::dsv::N_HEADS && kv_dim == md::dsv::KV_DIM &&
        kv_rank == md::dsv::KV_LORA && rope_dim == md::dsv::QK_ROPE) {
        if (batch >= kt::FLASH_LARGE_BATCH) {
            /* DECODE dsv flash, B>=kt::FLASH_LARGE_BATCH(128): MFMA FA2, one
               block/seq, 8 waves, all 16 heads share one latent-cache read.
               +9.3% TPS; bf16-Q rounding costs METEOR 0.4245->0.4068 /
               BERTScore 0.8867->0.8839 (both still > 0.30/0.83 gates). */
            flash_attention_mfma<md::dsv::N_HEADS, md::dsv::KV_LORA,
                                 md::dsv::QK_ROPE><<<(unsigned)batch,
                8 * HIP_WAVE, 0, stream>>>(clat, qabs, qrope, cache, positions,
                                           batch, capacity, scale);
            HIP_LAUNCH_CHECK();
        } else {
            /* DECODE dsv flash, kt::FLASH_MIN_ROWS(8)<=B<128: H=16, KV=576,
               latent=512, rope=64, tile=kt::FLASH_SMALL_KV_TILE(16),
               threads=kt::FLASH_SMALL_THREADS(512). */
            launch_small();
        }
    } else if (heads == md::glm::N_HEADS && kv_dim == md::glm::KV_DIM &&
               kv_rank == md::glm::KV_LORA && rope_dim == md::glm::QK_ROPE) {
        if (batch >= kt::FLASH_LARGE_BATCH) {
            /* DECODE glm flash, B>=kt::FLASH_LARGE_BATCH(128): MFMA FA2, one
               block/seq, 8 waves, all 20 heads share one latent-cache read. */
            flash_attention_mfma<md::glm::N_HEADS, md::glm::KV_LORA,
                                 md::glm::QK_ROPE><<<(unsigned)batch,
                8 * HIP_WAVE, 0, stream>>>(clat, qabs, qrope, cache, positions,
                                           batch, capacity, scale);
            HIP_LAUNCH_CHECK();
        } else {
            /* DECODE glm flash, kt::FLASH_MIN_ROWS(8)<=B<128: H=20, KV=576,
               latent=512, rope=64, tile=kt::FLASH_SMALL_KV_TILE(16),
               threads=kt::FLASH_SMALL_THREADS(512) */
            launch_small();
        }
    } else {
        ops::unlisted_shape("decode::flash_attention", heads, kv_dim, batch);
    }
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
        bf16_t *layer_cache = g.kv_cache + (size_t)l * g.kv_layer_stride;
        kv_norm_rope<<<batch, kt::ATTENTION_THREADS, 0, g.stream>>>(
            layer_cache, g.comp, d.kv_a_norm, positions, batch,
            g.kv_capacity, KVL, KVD, QKR, g.rope_inv_freq,
            c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        if (batch >= kt::MFMA_MIN_ROWS) {
            /* q-absorb: dsv [B,16h,512,192]; glm [B,20h,512,192] */
            ops::head_gemm(g.qabs, g.q, d.W_UK_T, batch, NH, KVL, QKN,
                           NH * QHD, QHD, KVL, (size_t)KVL * QKN, g.stream);
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
            flash_attention_dispatch(
                g.clat, g.qabs, g.qrope, layer_cache, positions, batch, NH,
                g.kv_capacity, KVD, KVL, QKR, c.softmax_scale, g.stream);
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
                           NH * KVL, KVL, VHD, head_stride, g.stream);
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
                         qs, g.routed_i8, g.routed_s, false, g.stream);
    }

    ops::rmsnorm(g.xn, g.x, g.final_norm, H, batch, c.rms_eps, g.stream);
    ops::gemm(g.logits, g.xn, g.lm_head, g.lm_head_s, batch, c.vocab_size, H,
              false, qs, g.stream);
    ops::argmax(g.logits, c.vocab_size, g.next_token, batch, g.stream);
}

} // namespace decode

#endif /* MLA_DECODE_H */
