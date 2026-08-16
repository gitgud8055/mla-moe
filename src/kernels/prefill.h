/* Packed-prompt prefill: kernels that only run during prefill plus
 * prefill::run(), the layer-major pass over one packed prompt chunk. */
#ifndef MLA_PREFILL_H
#define MLA_PREFILL_H

#include "ops.h"

namespace prefill {

namespace kt = utils::constants::kernel;
using utils::types::bf16x4;
using utils::types::fp32v4;

/* Packed prefill rows contain only real tokens. Row metadata maps each row
 * to its local sequence and absolute position, so heterogeneous prompts
 * share one projection/MoE batch without padding or length sorting. */
/* Normalize c_kv + rope k_pe for one packed row. Writes BOTH the bf16 KV
 * cache entry (for decode) and, in place over `comp`, the f32 result the
 * unabsorbed prefill attention reads (k_pe at comp[row][kv_rank:kv_dim]). */
__global__ void kv_norm_rope(
    bf16_t *cache, float *comp, const bf16_t *norm,
    const int *row_batches, const int *row_positions, int slot_base, int rows,
    int capacity, int kv_rank, int kv_dim, int rope_dim,
    const float *inv_freq, int interleaved, float eps) {
    __shared__ float scratch[ops::THREADS];
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    const int b = slot_base + row_batches[row];
    const int pos = row_positions[row];
    const float *src = comp + (size_t)row * kv_dim;
    bf16_t *dst = cache + ((size_t)b * capacity + pos) * kv_dim;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_rank; i += ops::THREADS)
        ss += src[i] * src[i];
    scratch[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = ops::THREADS / 2; stride; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / (float)kv_rank + eps);
    float *keep = comp + (size_t)row * kv_dim; /* f32 copy for prefill attn */
    for (int i = (int)threadIdx.x; i < kv_rank; i += ops::THREADS) {
        const float c_kv = src[i] * inv * gpu_bf16_to_f32(norm[i]);
        dst[i] = gpu_f32_to_bf16(c_kv);
        keep[i] = c_kv;
    }

    const int j = (int)threadIdx.x;
    if (j < rope_dim / 2) {
        const float *v = src + kv_rank;
        bf16_t *o = dst + kv_rank;
        float *ko = keep + kv_rank;
        const float angle = (float)pos * inv_freq[j];
        const float co = cosf(angle), si = sinf(angle);
        const float even = v[2 * j], odd = v[2 * j + 1];
        float r0, r1; int i0, i1;
        if (!interleaved) {
            i0 = 2 * j; i1 = 2 * j + 1;
            r0 = even * co - odd * si;
            r1 = even * si + odd * co;
        } else {
            const int half = rope_dim / 2;
            i0 = j; i1 = half + j;
            r0 = even * co - odd * si;
            r1 = odd * co + even * si;
        }
        o[i0] = gpu_f32_to_bf16(r0);
        o[i1] = gpu_f32_to_bf16(r1);
        ko[i0] = r0;
        ko[i1] = r1;
    }
}

__global__ void q_rope(
    float *qrope, const float *q, const int *row_positions, int rows, int heads,
    int q_head_dim, int qk_nope, const float *inv_freq, int interleaved) {
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int qh = (int)blockIdx.x * ops::WAVES + wave;
    if (qh >= rows * heads) return;
    const int row = qh / heads;
    const int pos = row_positions[row];
    const int rope_dim = q_head_dim - qk_nope;
    if (lane >= rope_dim / 2) return;
    const float *v = q + (size_t)qh * q_head_dim + qk_nope;
    float *o = qrope + (size_t)qh * rope_dim;
    const float angle = (float)pos * inv_freq[lane];
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

/* Query-tile map for the FA2 prefill flash kernel: one tile = BQ
 * consecutive rows of ONE sequence (tiles start where pos % BQ == 0).
 * Appended in arbitrary order via an atomic counter; unused slots keep
 * the -1 the caller memsets, and the kernel exits on them. */
__global__ void build_q_tiles(int *tile_q0, int *counter,
                              const int *positions, int rows, int q_tile) {
    const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= rows || positions[row] % q_tile != 0) return;
    tile_q0[atomicAdd(counter, 1)] = row;
}

/* FlashAttention-2 for the DECOMPRESSED (unabsorbed) prefill — the same
 * math as the attention loop of run.c forward_unabsorbed(). For each
 * row in [0, rows) and head h in [0, HEADS), with pos = positions[row] and
 * key row r_t = row - pos + t (a sequence's rows are packed contiguously):
 *   S[pos+1]: S[t] = scale * ( q[row,h,0:QK_NOPE]      @ k_nope[r_t,h,:]{QK_NOPE}
 *                            + q_pe[row,h,:]{ROPE_DIM} @ k_pe[r_t]{ROPE_DIM} )
 *                    for t = 0..pos (causal)
 *   P[pos+1] = softmax(S)  — FA2 online: running max m / running sum l,
 *                            unnormalized O rescaled per tile
 *   out[row,h,:]{V_HEAD} = ( sum_{t=0..pos} P[t] * value[r_t,h,:]{V_HEAD} ) / l
 * Result: out [rows x HEADS x V_HEAD] f32 — per-head context (o_proj input).
 * Block = (one q-tile of BQ rows of one sequence) x (one head); 4 waves of
 * 16 q-rows; BK keys staged per step as K [key][k] and V^T [col][key] in
 * LDS shared by all waves; S = Q@K^T and O += P@V on
 * v_mfma_f32_16x16x16bf16 (Q/K/P/V rounded to bf16 at the MFMA inputs). */
template <int BQ,       // query rows per block: kt::PREFILL_Q_TILE(64)
          int BK,       // keys per MFMA step: kt::PREFILL_KEY_TILE(16)
          int HEADS,    // query heads: dsv 16, glm 20
          int QK_NOPE,  // decompressed K width: dsv 128, glm 192
          int V_HEAD,   // decompressed V width: dsv 128, glm 256
          int ROPE_DIM> // rope width: 64 (both models)
__global__ void flash_attention(
    float *__restrict__ out,           // [rows x HEADS x V_HEAD] f32 (result)
    const float *__restrict__ q,       // [rows x HEADS x (QK_NOPE+ROPE_DIM)] f32; rope tail unused
    const float *__restrict__ q_pe,    // [rows x HEADS x ROPE_DIM] f32, roped
    const float *__restrict__ k_nope,  // [rows x HEADS x QK_NOPE] f32, decompressed
    const float *__restrict__ value,   // [rows x HEADS x V_HEAD] f32, decompressed
    const float *__restrict__ k_pe,    // [rows x kpe_stride] f32; roped k_pe at [kpe_offset:...]
    const int *__restrict__ positions, // [rows] position within its sequence
    const int *__restrict__ tile_q0,   // [grid.y] first packed row per q-tile, -1 = unused slot
    int rows,        // packed tokens: dsv <=65520, glm <=32768 (md::*::PREFILL_ROWS)
    int kpe_offset,  // k_pe column offset in its row: KV_LORA(512), both models
    int kpe_stride,  // k_pe row stride: KV_DIM(576), both models
    float scale) {   // dsv 1/sqrt(192), glm 1/sqrt(256)
    constexpr int K_DIM = QK_NOPE + ROPE_DIM; // 192 dsv / 256 glm
    constexpr int KP = 4;
    static_assert(K_DIM % 16 == 0 && V_HEAD % 16 == 0, "MFMA tiles");
    static_assert(BQ == 64 && BK == 16, "4 waves x 16 rows, 16-key steps");
    const int head = (int)blockIdx.x;
    const int row0 = tile_q0[(int)blockIdx.y];
    if (row0 < 0) return;                 /* uniform for the whole block */
    const int pos0 = positions[row0];
    const int seq0 = row0 - pos0;         /* packed row of this seq's t=0 */

    __shared__ int s_nact;
    __shared__ bf16_t sk[BK][K_DIM + KP];     /* K tile   [key][k]       */
    __shared__ bf16_t svT[V_HEAD][BK + KP];   /* V^T tile [col][key]     */
    __shared__ bf16_t sp[4][16][BK + KP];     /* P tiles  [wave][q][key] */
    if (threadIdx.x == 0) {
        int n = 1;
        while (n < BQ && row0 + n < rows && positions[row0 + n] == pos0 + n)
            ++n;
        s_nact = n;
    }
    __syncthreads();
    const int nact = s_nact;              /* active q rows in this tile   */
    const int kv_len = pos0 + nact;       /* keys visible to the last row */

    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int a_row = lane & 15;          /* A-layout row, C-layout col   */
    const int k4 = 4 * (lane >> 4);       /* A/B-layout k offset          */
    const int qlocal = wave * 16 + a_row; /* this lane's q row (A layout) */
    const int grow = row0 + qlocal;
    const bool q_ok = qlocal < nact;

    /* Q tile, A layout, registers: q_reg[kt] = Q[a_row][kt*16+k4 .. +3] */
    bf16x4 q_reg[K_DIM / 16];
#pragma unroll
    for (int kt = 0; kt < K_DIM / 16; ++kt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = kt * 16 + k4 + j;
            float f = 0.0f;
            if (q_ok) {
                const size_t qh = (size_t)grow * HEADS + head;
                f = k < QK_NOPE ? q[qh * K_DIM + k]
                                : q_pe[qh * ROPE_DIM + (k - QK_NOPE)];
            }
            q_reg[kt][j] = (short)gpu_f32_to_bf16(f);
        }
    }

    fp32v4 o_acc[V_HEAD / 16] = {};
    float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float l[4] = {};
    /* highest key this wave's rows can see (block-uniform per wave) */
    const int wave_last_pos = pos0 + min(wave * 16 + 15, nact - 1);

    for (int k0 = 0; k0 < kv_len; k0 += BK) {
        /* ---- stage K [key][k] and V^T [col][key] (f32 -> bf16) ---- */
        for (int i = (int)threadIdx.x; i < BK * K_DIM; i += (int)blockDim.x) {
            const int key = i / K_DIM, k = i - (i / K_DIM) * K_DIM;
            float f = 0.0f;
            if (k0 + key < kv_len) {
                const size_t kr = (size_t)(seq0 + k0 + key);
                f = k < QK_NOPE
                    ? k_nope[(kr * HEADS + head) * QK_NOPE + k]
                    : k_pe[kr * kpe_stride + kpe_offset + (k - QK_NOPE)];
            }
            sk[key][k] = gpu_f32_to_bf16(f);
        }
        for (int i = (int)threadIdx.x; i < BK * V_HEAD;
             i += (int)blockDim.x) {
            const int key = i / V_HEAD, c = i - (i / V_HEAD) * V_HEAD;
            float f = 0.0f;
            if (k0 + key < kv_len) {
                const size_t kr = (size_t)(seq0 + k0 + key);
                f = value[(kr * HEADS + head) * V_HEAD + c];
            }
            svT[c][key] = gpu_f32_to_bf16(f);
        }
        __syncthreads();

        const bool wave_active = k0 <= wave_last_pos;
        if (wave_active) {
            /* ---- S = Q @ K^T (C layout: row 4*(lane>>4)+i, col a_row) */
            fp32v4 sacc = {};
#pragma unroll
            for (int kt = 0; kt < K_DIM / 16; ++kt)
                sacc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    q_reg[kt],
                    *reinterpret_cast<const bf16x4 *>(&sk[a_row][kt * 16 + k4]),
                    sacc, 0, 0, 0);

            /* ---- mask + FA2 online softmax, then P -> LDS (bf16) ---- */
            const int key_pos = k0 + a_row;   /* this lane's key (C col) */
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rowl = 4 * (lane >> 4) + i;   /* C-layout row  */
                const int p_q = pos0 + wave * 16 + rowl;
                const bool row_live = wave * 16 + rowl < nact;
                float sv = sacc[i] * scale;
                if (!row_live || key_pos > p_q || key_pos >= kv_len)
                    sv = -INFINITY;
                /* row max/sum: one row's 16 cols live in the 16 lanes that
                 * share lane>>4 — reduce with xor shuffles over bits 0-3 */
                float mx = sv;
                for (int d = 1; d < 16; d <<= 1)
                    mx = fmaxf(mx, __shfl_xor(mx, d, HIP_WAVE));
                const float m_new = fmaxf(m[i], mx);
                const float resc =
                    m_new == -INFINITY ? 1.0f : expf(m[i] - m_new);
                const float p =
                    m_new == -INFINITY ? 0.0f : expf(sv - m_new);
                float ps = p;
                for (int d = 1; d < 16; d <<= 1)
                    ps += __shfl_xor(ps, d, HIP_WAVE);
                l[i] = l[i] * resc + ps;
                m[i] = m_new;
#pragma unroll
                for (int ct = 0; ct < V_HEAD / 16; ++ct)
                    o_acc[ct][i] *= resc;
                sp[wave][rowl][a_row] = gpu_f32_to_bf16(p);
            }
        }
        __syncthreads();   /* sp visible; uniform (outside wave_active) */

        if (wave_active) {
            /* ---- O += P @ V: av = P[a_row][k4..], bv = V^T[col][k4..] */
            const bf16x4 pv =
                *reinterpret_cast<const bf16x4 *>(&sp[wave][a_row][k4]);
#pragma unroll
            for (int ct = 0; ct < V_HEAD / 16; ++ct)
                o_acc[ct] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    pv,
                    *reinterpret_cast<const bf16x4 *>(
                        &svT[ct * 16 + a_row][k4]),
                    o_acc[ct], 0, 0, 0);
        }
        __syncthreads();   /* PV done before restaging sk/svT           */
    }

    /* ---- epilogue: divide by l, store (C layout) ---- */
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int rowl = 4 * (lane >> 4) + i;
        if (wave * 16 + rowl >= nact) continue;
        const size_t orow = (size_t)(row0 + wave * 16 + rowl) * HEADS + head;
        const float inv_l = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
#pragma unroll
        for (int ct = 0; ct < V_HEAD / 16; ++ct)
            out[orow * V_HEAD + ct * 16 + a_row] = o_acc[ct][i] * inv_l;
    }
}

/* Rebuild per-row metadata on device: row r belongs to the sequence b whose
 * [offsets[b], offsets[b+1]) range contains r; its position is the offset
 * into that range. Lets the host upload only the token slices + offsets. */
__global__ void build_row_meta(int *row_batches, int *row_positions,
                               const int *offsets, int batch, int rows) {
    const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= rows) return;
    int lo = 0, hi = batch; /* invariant: offsets[lo] <= row < offsets[hi] */
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= row) lo = mid; else hi = mid;
    }
    row_batches[row] = lo;
    row_positions[row] = row - offsets[lo];
}

/* Shape dispatch for flash_attention: full-tuple ladder, one branch per
 * model, each instantiates the kernel with compile-time constants;
 * unlisted shapes abort. `tile_q0`/`n_tiles` come from build_q_tiles.
 * Pointer shapes documented on the kernel above. */
inline void flash_attention_dispatch(
    float *out,              // [rows x heads x v_head]
    const float *q,          // [rows x heads x (qk_nope+rope_dim)]
    const float *q_pe,       // [rows x heads x rope_dim]
    const float *k_nope,     // [rows x heads x qk_nope]
    const float *value,      // [rows x heads x v_head]
    const float *k_pe,       // [rows x kpe_stride], k_pe at [kpe_offset:...]
    const int *positions,    // [rows]
    const int *tile_q0,      // [n_tiles] first packed row per q-tile (-1 pad)
    int n_tiles,             // upper bound: rows/kt::PREFILL_Q_TILE + batch
    int rows,                // packed tokens in the chunk
    int heads,               // branch key: dsv 16 / glm 20
    int qk_nope,             // branch key: dsv 128 / glm 192
    int v_head,              // branch key: dsv 128 / glm 256
    int rope_dim,            // branch key: 64 (both models)
    int kpe_offset,          // KV_LORA(512), both models
    int kpe_stride,          // KV_DIM(576), both models
    float scale,             // dsv 1/sqrt(192), glm 1/sqrt(256)
    hipStream_t stream) {
    namespace md = utils::constants::model;
    if (heads == md::dsv::N_HEADS && qk_nope == md::dsv::QK_NOPE &&
        v_head == md::dsv::V_HEAD && rope_dim == md::dsv::QK_ROPE) {
        /* PREFILL dsv FA2 flash: rows<=md::dsv::PREFILL_ROWS(65520), H=16,
           K=md::dsv::QK_NOPE(128)+rope 64, V=md::dsv::V_HEAD(128),
           q_tile=kt::PREFILL_Q_TILE(64), key_tile=kt::PREFILL_KEY_TILE(16),
           threads=kt::PREFILL_THREADS(256) */
        dim3 grid((unsigned)md::dsv::N_HEADS, (unsigned)n_tiles);
        flash_attention<kt::PREFILL_Q_TILE, kt::PREFILL_KEY_TILE,
                        md::dsv::N_HEADS, md::dsv::QK_NOPE, md::dsv::V_HEAD,
                        md::dsv::QK_ROPE>
            <<<grid, kt::PREFILL_THREADS, 0, stream>>>(
                out, q, q_pe, k_nope, value, k_pe, positions, tile_q0,
                rows, kpe_offset, kpe_stride, scale);
        HIP_LAUNCH_CHECK();
    } else if (heads == md::glm::N_HEADS && qk_nope == md::glm::QK_NOPE &&
               v_head == md::glm::V_HEAD && rope_dim == md::glm::QK_ROPE) {
        /* PREFILL glm FA2 flash: rows<=md::glm::PREFILL_ROWS(32768), H=20,
           K=md::glm::QK_NOPE(192)+rope 64, V=md::glm::V_HEAD(256),
           q_tile=kt::PREFILL_Q_TILE(64), key_tile=kt::PREFILL_KEY_TILE(16),
           threads=kt::PREFILL_THREADS(256) */
        dim3 grid((unsigned)md::glm::N_HEADS, (unsigned)n_tiles);
        flash_attention<kt::PREFILL_Q_TILE, kt::PREFILL_KEY_TILE,
                        md::glm::N_HEADS, md::glm::QK_NOPE, md::glm::V_HEAD,
                        md::glm::QK_ROPE>
            <<<grid, kt::PREFILL_THREADS, 0, stream>>>(
                out, q, q_pe, k_nope, value, k_pe, positions, tile_q0,
                rows, kpe_offset, kpe_stride, scale);
        HIP_LAUNCH_CHECK();
    } else {
        ops::unlisted_shape("prefill::flash_attention", heads, qk_nope, rows);
    }
}

__global__ void gather_last_hidden(float *dst, const float *src,
                                   const int *offsets, int batch,
                                   int slot_base, int hidden) {
    const int b = (int)blockIdx.y;
    const int d = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (b < batch && d < hidden)
        dst[(size_t)(slot_base + b) * hidden + d] =
            src[(size_t)(offsets[b + 1] - 1) * hidden + d];
}

/* Process one packed prompt chunk layer-major, mirroring the staging of
 * run.c forward_unabsorbed():
 *   1) per-position projections: q (roped), c_kv (norm) + k_pe (roped),
 *      written to the bf16 KV cache (for decode) and kept in f32
 *   2) decompress per-head K_nope and V for every key (W_UK / W_UV)
 *   3) causal attention per (query, head) in the decompressed space
 *   4) o_proj + residual, then FFN; final norm + lm_head at the last
 *      position of each sequence
 * `rows` counts real input tokens; row metadata gives each token's sequence
 * and position. Emits the first generated token per sequence into
 * g.next_token[slot]. */
inline void run(GpuContext &g, const Config &c, int max_batch,
                const int *device_tokens, const int *row_batches,
                const int *row_positions, const int *offsets,
                int rows, int batch, int slot_base) {
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
    if (batch < 1 || slot_base < 0 || slot_base + batch > max_batch ||
        rows < batch || rows > g.prefill_capacity) {
        std::fprintf(stderr,
            "HIP packed prefill: slots=%d+%d/%d tokens=%d/%d\n",
            slot_base, batch, max_batch, rows, g.prefill_capacity);
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
            ops::unlisted_shape("prefill::run", NH, QKN, VHD);
    }
    const ops::QuantScratch qs{g.prefill_act_i8, g.prefill_act_s};

    /* Query-tile map for the FA2 flash kernel — layer-invariant, built
     * once per chunk. n_tiles is an upper bound; unused slots stay -1. */
    const int n_tiles = ops::div_up(rows, kt::PREFILL_Q_TILE) + batch;
    HIP_CHECK(hipMemsetAsync(g.prefill_tile_q0, 0xff,
                             (size_t)n_tiles * sizeof(int), g.stream));
    HIP_CHECK(hipMemsetAsync(g.prefill_tile_count, 0, sizeof(int), g.stream));
    build_q_tiles<<<ops::div_up(rows, kt::ELEMENTWISE_THREADS),
        kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.prefill_tile_q0, g.prefill_tile_count, row_positions, rows,
            kt::PREFILL_Q_TILE);
    HIP_LAUNCH_CHECK();

    ops::embedding(g.prefill_x, g.embed, device_tokens, nullptr, 0, 0,
                   H, rows, g.stream);

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        ops::rmsnorm(g.prefill_xn, g.prefill_x, d.input_norm, H, rows,
                     c.rms_eps, g.stream);
        if (c.q_lora_rank > 0) {
            /* glm: q_a [rows,768,2048] + kv_a [rows,576,2048] */
            ops::gemm_dual(g.prefill_q_a, d.q_a_proj, d.q_a_proj_s, c.q_lora_rank,
                           g.prefill_comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                           g.prefill_xn, H, rows, qs, g.stream);
            ops::rmsnorm(g.prefill_q_a, g.prefill_q_a, d.q_a_norm,
                         c.q_lora_rank, rows, c.mla_norm_eps, g.stream);
            /* glm: q_b [rows,5120,768] */
            ops::gemm(g.prefill_q, g.prefill_q_a, d.q_b_proj, d.q_b_proj_s,
                      rows, QD, c.q_lora_rank, false, qs, g.stream);
        } else {
            /* dsv: q_proj [rows,3072,2048] + kv_a [rows,576,2048] */
            ops::gemm_dual(g.prefill_q, d.q_proj, d.q_proj_s, QD,
                           g.prefill_comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                           g.prefill_xn, H, rows, qs, g.stream);
        }

        /* c_kv norm + k_pe rope -> bf16 cache (decode) + f32 comp (here) */
        bf16_t *layer_cache = g.kv_cache
            + (size_t)l * max_batch * g.kv_capacity * KVD;
        kv_norm_rope<<<rows, kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
            layer_cache, g.prefill_comp, d.kv_a_norm, row_batches,
            row_positions, slot_base, rows, g.kv_capacity, KVL, KVD,
            QKR, g.rope_inv_freq, c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();
        q_rope<<<ops::div_up(rows * NH, kt::ATTENTION_THREADS / HIP_WAVE),
            kt::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_qrope, g.prefill_q, row_positions, rows, NH, QHD,
                QKN, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        /* decompress every key: K_nope = W_UK @ c_kv, V = W_UV @ c_kv
         * (x_head_dim = 0: all heads read the same per-row c_kv) */
        ops::head_gemm(g.prefill_knope, g.prefill_comp, d.W_UK, rows, NH,
                       QKN, KVL, KVD, 0, QKN, head_stride, g.stream);
        ops::head_gemm(g.prefill_value, g.prefill_comp, d.W_UV, rows, NH,
                       VHD, KVL, KVD, 0, VHD, head_stride, g.stream);

        /* causal attention in the decompressed space -> per-head context */
        flash_attention_dispatch(
            g.prefill_ctx, g.prefill_q, g.prefill_qrope, g.prefill_knope,
            g.prefill_value, g.prefill_comp, row_positions,
            g.prefill_tile_q0, n_tiles, rows, NH, QKN, VHD, QKR, KVL, KVD,
            c.softmax_scale, g.stream);

        /* o_proj: dsv [rows,2048,2048]; glm [rows,2048,5120] */
        ops::gemm(g.prefill_x, g.prefill_ctx, d.o_proj, d.o_proj_s,
                  rows, H, NH * VHD, true, qs, g.stream);

        ops::rmsnorm(g.prefill_xn, g.prefill_x, d.post_norm, H, rows,
                     c.rms_eps, g.stream);
        if (l < c.first_k_dense)
            ops::dense_ffn(c, d, rows, g.prefill_x, g.prefill_xn,
                           g.prefill_hb, qs, g.stream);
        else
            ops::moe_ffn(c, d, rows, g.prefill_x, g.prefill_xn, g.prefill_hb,
                         g.prefill_router, g.prefill_route_ids,
                         g.prefill_topk_weights, g.moe_sorted_ids,
                         g.moe_expert_ids, g.moe_num_padded,
                         nullptr, g.prefill_routed_hidden, g.moe_route_out,
                         qs, g.prefill_routed_i8, g.prefill_routed_s,
                         g.stream);
    }

    dim3 gather_grid((unsigned)ops::div_up(H, kt::ELEMENTWISE_THREADS),
                     (unsigned)batch);
    gather_last_hidden<<<gather_grid, kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
        g.x, g.prefill_x, offsets, batch, slot_base, H);
    HIP_LAUNCH_CHECK();
    ops::rmsnorm(g.xn + (size_t)slot_base * H, g.x + (size_t)slot_base * H,
                 g.final_norm, H, batch, c.rms_eps, g.stream);
    const ops::QuantScratch decode_qs{g.act_i8, g.act_s};
    ops::gemm(g.logits + (size_t)slot_base * c.vocab_size,
              g.xn + (size_t)slot_base * H, g.lm_head, g.lm_head_s,
              batch, c.vocab_size, H, false, decode_qs, g.stream);
    ops::argmax(g.logits + (size_t)slot_base * c.vocab_size, c.vocab_size,
                g.next_token + slot_base, batch, g.stream);
}

} // namespace prefill

#endif /* MLA_PREFILL_H */
