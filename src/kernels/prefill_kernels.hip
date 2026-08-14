// FlashAttention-v2 style batched prefill kernels for MLA (unabsorbed form).
//
// The decode path feeds one prompt position at a time through the absorbed MLA
// kernels (batch = number of requests). That makes prefill of an L-token prompt
// L sequential batch-1 passes -- every weight is re-read once per token, so the
// whole regime is HBM-bandwidth bound. These kernels instead process all L
// prompt tokens at once (rows = L) so the projections/FFN become MFMA GEMMs and
// attention becomes a single causal flash pass, mirroring the frozen CPU
// reference forward_unabsorbed() in src/run.c.
//
// Numerics track the reference: c_kv is RMS-normed, K_nope/V are decompressed
// from the (float) latent via W_UK/W_UV, q's rope half is roped in place, and
// the score is qnope.K_nope + qpe.k_pe, causal-masked, softmaxed, then .V.

#include "getp_kernels.h"
#include "getp_kernels_device.h"

namespace prefill_kernels {

constexpr int THREADS = kernel_tuning::ATTENTION_THREADS;   // 256 (4 waves)
constexpr int WAVES = THREADS / HIP_WAVE;

__device__ __forceinline__ bf16_t gpu_f32_to_bf16(float value) {
    return (bf16_t)(__float_as_uint(value) >> 16);
}

// Apply RoPE in place to the rope half of q for every (position, head).
// Row index encodes the position: row = p * heads + h, so the rotation angle
// uses pos = p. Non-interleaved layout only (the split-half GLM interleaving
// would alias reads/writes in place); rope_interleaved prompts are unsupported
// here for now.
__global__ void q_rope_rows(float *q, int seq, int heads, int q_head_dim,
                            int qk_nope, int rope_dim, const float *inv_freq,
                            int interleaved) {
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int row = (int)blockIdx.x * WAVES + wave;
    const int half = rope_dim / 2;
    if (row >= seq * heads || lane >= half) return;
    const int p = row / heads;
    float *o = q + (size_t)row * q_head_dim + qk_nope;
    const float angle = (float)p * inv_freq[lane];
    const float co = cosf(angle), si = sinf(angle);
    if (!interleaved) {
        const float even = o[2 * lane], odd = o[2 * lane + 1];
        o[2 * lane] = even * co - odd * si;
        o[2 * lane + 1] = even * si + odd * co;
    } else {
        const float even = o[lane], odd = o[half + lane];
        o[lane] = even * co - odd * si;
        o[half + lane] = odd * co + even * si;
    }
}

// One block per prompt position. Produces the float latent c_kv (for the K/V
// decompression GEMM), the float roped k_pe (for the attention rope dot), and
// the bf16 KV-cache row [c_kv | k_pe] that the decode path reads afterward.
// Cache layout matches batch_decode_kernels::kv_norm_rope: [slot, pos, kv_dim].
__global__ void kv_norm_rope_prefill(float *ckv_out, float *kpe_out,
                                     bf16_t *cache, const float *comp,
                                     const bf16_t *norm, int seq, int capacity,
                                     int slot, int kv_rank, int kv_dim,
                                     int rope_dim, const float *inv_freq,
                                     int interleaved, float eps) {
    __shared__ float scratch[THREADS];
    const int p = (int)blockIdx.x;
    if (p >= seq) return;
    const float *src = comp + (size_t)p * kv_dim;
    float *co = ckv_out + (size_t)p * kv_rank;
    bf16_t *dst = cache + ((size_t)slot * capacity + p) * kv_dim;

    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS) ss += src[i] * src[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int s = THREADS / 2; s; s >>= 1) {
        if ((int)threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)kv_rank + eps);
    for (int i = (int)threadIdx.x; i < kv_rank; i += THREADS) {
        const float v = src[i] * inv * gpu_bf16_to_f32(norm[i]);
        co[i] = v;
        dst[i] = gpu_f32_to_bf16(v);
    }

    const int j = (int)threadIdx.x;
    const int halfr = rope_dim / 2;
    if (j < halfr) {
        const float *v = src + kv_rank;
        float *ko = kpe_out + (size_t)p * rope_dim;
        bf16_t *o = dst + kv_rank;
        const float angle = (float)p * inv_freq[j];
        const float cs = cosf(angle), sn = sinf(angle);
        const float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved) {
            const float a = even * cs - odd * sn, b = even * sn + odd * cs;
            ko[2 * j] = a;      ko[2 * j + 1] = b;
            o[2 * j] = gpu_f32_to_bf16(a);
            o[2 * j + 1] = gpu_f32_to_bf16(b);
        } else {
            const float a = even * cs - odd * sn, b = odd * cs + even * sn;
            ko[j] = a;          ko[halfr + j] = b;
            o[j] = gpu_f32_to_bf16(a);
            o[halfr + j] = gpu_f32_to_bf16(b);
        }
    }
}

// Causal FlashAttention-v2 for one prompt. One wave owns one (query, head) and
// streams keys 0..query with an online softmax, holding the running max, the
// denominator, and its slice of the value accumulator in registers -- no LxL
// score matrix is ever materialized.
//   q      : [seq, heads, q_head_dim]  (nope | roped rope half)
//   knope  : [heads, seq, qk_nope]     (decompressed K_nope, head-major)
//   kpe    : [seq, rope_dim]           (roped k_pe, shared across heads)
//   value  : [heads, seq, v_dim]       (decompressed V, head-major)
//   ctx    : [seq, heads, v_dim]       (output context)
__global__ void flash_prefill_causal(float *__restrict__ ctx,
                                     const float *__restrict__ q,
                                     const float *__restrict__ knope,
                                     const float *__restrict__ kpe,
                                     const float *__restrict__ value,
                                     int seq, int heads, int qk_nope,
                                     int rope_dim, int v_dim, int q_head_dim,
                                     float scale) {
    constexpr int MAXV = 4;   // v_dim <= MAXV * HIP_WAVE (256)
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int gw = (int)blockIdx.x * WAVES + wave;
    if (gw >= heads * seq) return;
    const int h = gw / seq;
    const int qi = gw - h * seq;

    const float *qh = q + ((size_t)qi * heads + h) * q_head_dim;
    const float *qnope = qh;
    const float *qpe = qh + qk_nope;
    const float *knope_h = knope + (size_t)h * seq * qk_nope;
    const float *value_h = value + (size_t)h * seq * v_dim;

    float acc[MAXV] = {};
    float running_max = -INFINITY;
    float denom = 0.0f;

    for (int k = 0; k <= qi; ++k) {
        const float *kn = knope_h + (size_t)k * qk_nope;
        const float *kp = kpe + (size_t)k * rope_dim;
        float partial = 0.0f;
        for (int d = lane; d < qk_nope; d += HIP_WAVE) partial += qnope[d] * kn[d];
        for (int d = lane; d < rope_dim; d += HIP_WAVE) partial += qpe[d] * kp[d];
        float score = wave_sum(partial) * scale;
        score = __shfl(score, 0, HIP_WAVE);

        const float next_max = fmaxf(running_max, score);
        const float rescale = expf(running_max - next_max);
        const float w = expf(score - next_max);
        denom = denom * rescale + w;
        const float *vk = value_h + (size_t)k * v_dim;
#pragma unroll
        for (int q = 0; q < MAXV; ++q) {
            const int r = lane + q * HIP_WAVE;
            acc[q] = acc[q] * rescale + (r < v_dim ? w * vk[r] : 0.0f);
        }
        running_max = next_max;
    }

    const float inv = 1.0f / denom;
    float *out = ctx + ((size_t)qi * heads + h) * v_dim;
#pragma unroll
    for (int q = 0; q < MAXV; ++q) {
        const int r = lane + q * HIP_WAVE;
        if (r < v_dim) out[r] = acc[q] * inv;
    }
}

// LDS-tiled causal FlashAttention-v2 (fp32). The per-(query,head) kernel above
// re-reads every key's K_nope|k_pe|V from HBM once per query >= k, i.e. O(L^2)
// global traffic -- HBM-bandwidth bound, not compute bound. This version tiles
// BQ queries per block and stages each BK-key block into LDS exactly once, so
// all BQ queries share it: K/V HBM traffic drops by ~BQ. Math stays fp32 (no
// MFMA / bf16 rounding), so the greedy cascade tracks the fp32 reference.
//
// One block owns (head = blockIdx.y, query tile [r0, r0+BQ) = blockIdx.x). The
// 256 threads map as tid = qr*BK + kc. In the QK^T phase tid computes the score
// for (query r0+qr, key kb0+kc); in the P.V phase kc re-roles as a v-dim lane so
// each thread owns the strided output columns { kc + j*BK }. LDS is sized for
// the MLA dims of both families (KC = qk_nope+rope_dim <= 192, v_dim <= 128).
constexpr int TQ = 16;   // query rows per tile  (BQ)
constexpr int TK = 16;   // key cols per tile    (BK)
constexpr int KC_MAX = 192;   // qk_nope(128) + rope_dim(64)
constexpr int VD_MAX = 128;   // v_head_dim
constexpr int OW = VD_MAX / TK;   // v-dims per thread in the P.V phase (8)

__global__ void flash_prefill_causal_tiled(float *__restrict__ ctx,
                                           const float *__restrict__ q,
                                           const float *__restrict__ knope,
                                           const float *__restrict__ kpe,
                                           const float *__restrict__ value,
                                           int seq, int heads, int qk_nope,
                                           int rope_dim, int v_dim, int q_head_dim,
                                           float scale) {
    __shared__ float sQ[TQ][KC_MAX];
    __shared__ float sK[TK][KC_MAX];
    __shared__ float sV[TK][VD_MAX];
    __shared__ float sS[TQ][TK];   // scores, then softmax weights P
    __shared__ float sM[TQ];       // running row max
    __shared__ float sL[TQ];       // running denom
    __shared__ float sCorr[TQ];    // per-block rescale factor exp(m_prev - m_new)

    const int tid = (int)threadIdx.x;
    const int qr = tid / TK;
    const int kc = tid - qr * TK;
    const int h = (int)blockIdx.y;
    const int r0 = (int)blockIdx.x * TQ;
    const int KC = qk_nope + rope_dim;
    const int gq = r0 + qr;   // this thread's global query row

    // Q tile -> LDS once (rows beyond seq are zero-padded but never written out).
    for (int i = tid; i < TQ * KC; i += (int)blockDim.x) {
        const int r = i / KC, d = i - r * KC;
        const int grow = r0 + r;
        sQ[r][d] = grow < seq
            ? q[((size_t)grow * heads + h) * q_head_dim + d] : 0.0f;
    }
    if (tid < TQ) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }

    float acc[OW] = {};   // this thread's output accumulators (v-dims kc+j*TK)

    // Only key blocks that can be causally visible to some query in the tile.
    const int kmax = (r0 + TQ - 1 < seq ? r0 + TQ - 1 : seq - 1);
    for (int kb0 = 0; kb0 <= kmax; kb0 += TK) {
        __syncthreads();
        // K tile: combined [K_nope(qk_nope) | k_pe(rope_dim)] per key.
        for (int i = tid; i < TK * KC; i += (int)blockDim.x) {
            const int k = i / KC, d = i - k * KC;
            const int gk = kb0 + k;
            float val = 0.0f;
            if (gk < seq) {
                val = d < qk_nope
                    ? knope[((size_t)h * seq + gk) * qk_nope + d]
                    : kpe[(size_t)gk * rope_dim + (d - qk_nope)];
            }
            sK[k][d] = val;
        }
        for (int i = tid; i < TK * v_dim; i += (int)blockDim.x) {
            const int k = i / v_dim, d = i - k * v_dim;
            const int gk = kb0 + k;
            sV[k][d] = gk < seq
                ? value[((size_t)h * seq + gk) * v_dim + d] : 0.0f;
        }
        __syncthreads();

        // QK^T: one score cell per thread, causal-masked.
        const int gk = kb0 + kc;
        float s = -INFINITY;
        if (gq < seq && gk <= gq) {
            float dot = 0.0f;
            for (int d = 0; d < KC; ++d) dot += sQ[qr][d] * sK[kc][d];
            s = dot * scale;
        }
        sS[qr][kc] = s;
        __syncthreads();

        // Online softmax: one thread per query row folds this block into the
        // running (max, denom) and overwrites sS with the block's weights P.
        if (tid < TQ && r0 + tid < seq) {
            float bmax = -INFINITY;
            for (int k = 0; k < TK; ++k) bmax = fmaxf(bmax, sS[tid][k]);
            const float mprev = sM[tid];
            const float mnew = fmaxf(mprev, bmax);
            const float corr = mprev == -INFINITY ? 0.0f : expf(mprev - mnew);
            float bsum = 0.0f;
            for (int k = 0; k < TK; ++k) {
                const float p = sS[tid][k] == -INFINITY
                    ? 0.0f : expf(sS[tid][k] - mnew);
                sS[tid][k] = p;
                bsum += p;
            }
            sM[tid] = mnew;
            sL[tid] = sL[tid] * corr + bsum;
            sCorr[tid] = corr;
        }
        __syncthreads();

        // P.V: rescale the running accumulator, then add this block. Zero-padded
        // sV keeps out-of-range keys harmless (their P is 0 anyway).
        if (gq < seq) {
            const float corr = sCorr[qr];
#pragma unroll
            for (int j = 0; j < OW; ++j) {
                const int vd = kc + j * TK;
                if (vd < v_dim) {
                    float o = acc[j] * corr;
                    for (int k = 0; k < TK; ++k) o += sS[qr][k] * sV[k][vd];
                    acc[j] = o;
                }
            }
        }
    }

    if (gq >= seq) return;
    const float inv = 1.0f / sL[qr];
    float *out = ctx + ((size_t)gq * heads + h) * v_dim;
#pragma unroll
    for (int j = 0; j < OW; ++j) {
        const int vd = kc + j * TK;
        if (vd < v_dim) out[vd] = acc[j] * inv;
    }
}

} // namespace prefill_kernels
