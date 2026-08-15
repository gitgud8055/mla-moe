/* Packed-prompt prefill: kernels that only run during prefill plus
 * prefill::run(), the layer-major pass over one packed prompt chunk. */
#ifndef MLA_PREFILL_H
#define MLA_PREFILL_H

#include "ops.h"

namespace prefill {

namespace kt = utils::constants::kernel;

/* Packed prefill rows contain only real tokens. Row metadata maps each row
 * to its local sequence and absolute position, so heterogeneous prompts
 * share one projection/MoE batch without padding or length sorting. */
__global__ void kv_norm_rope(
    bf16_t *cache, const float *comp, const bf16_t *norm,
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
    for (int i = (int)threadIdx.x; i < kv_rank; i += ops::THREADS)
        dst[i] = gpu_f32_to_bf16(src[i] * inv * gpu_bf16_to_f32(norm[i]));

    const int j = (int)threadIdx.x;
    if (j < rope_dim / 2) {
        const float *v = src + kv_rank;
        bf16_t *o = dst + kv_rank;
        const float angle = (float)pos * inv_freq[j];
        const float co = cosf(angle), si = sinf(angle);
        const float even = v[2 * j], odd = v[2 * j + 1];
        if (!interleaved) {
            o[2 * j] = gpu_f32_to_bf16(even * co - odd * si);
            o[2 * j + 1] = gpu_f32_to_bf16(even * si + odd * co);
        } else {
            const int half = rope_dim / 2;
            o[j] = gpu_f32_to_bf16(even * co - odd * si);
            o[half + j] = gpu_f32_to_bf16(odd * co + even * si);
        }
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

/* Causal Flash Attention for absorbed MLA prefill. A wave owns one head; all
 * waves of a block share one (request, query) and each staged KV tile.
 * The query is preloaded into registers and both dot products share a single
 * wave reduction. The loop bound depends only on blockIdx (uniform), so the
 * in-loop barrier stays convergent for head counts that do not divide the
 * wave count. */
template <int KV_TILE>
__global__ void flash_attention(
    float *__restrict__ clat, const float *__restrict__ qabs,
    const float *__restrict__ qrope, const bf16_t *__restrict__ cache,
    const int *__restrict__ row_batches,
    const int *__restrict__ row_positions, int slot_base, int rows, int heads,
    int capacity, int kv_dim, int kv_rank, int rope_dim, float scale) {
    extern __shared__ bf16_t shared_cache[];
    const int wave = (int)threadIdx.x / HIP_WAVE;
    const int lane = (int)threadIdx.x & (HIP_WAVE - 1);
    const int waves = (int)blockDim.x / HIP_WAVE;
    const int row = (int)blockIdx.y;
    const int h = (int)blockIdx.x * waves + wave;
    const bool active = row < rows && h < heads;
    const int b = row < rows ? slot_base + row_batches[row] : slot_base;
    const int qpos = row < rows ? row_positions[row] : 0;
    const size_t qh = (size_t)row * heads + h;
    const float *qa = active ? qabs + qh * kv_rank : nullptr;
    const float *qp = active ? qrope + qh * rope_dim : nullptr;
    const bf16_t *cache_b = cache + (size_t)b * capacity * kv_dim;
    const int last_query = qpos + 1;
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

    for (int k0 = 0; k0 < last_query; k0 += KV_TILE) {
        const int tile = min(KV_TILE, last_query - k0);
        for (int i = (int)threadIdx.x; i < tile * kv_dim;
             i += (int)blockDim.x)
            shared_cache[i] = cache_b[(size_t)k0 * kv_dim + i];
        __syncthreads();

        float scores[KV_TILE];
#pragma unroll
        for (int t = 0; t < KV_TILE; ++t) {
            const int key = k0 + t;
            const bool valid = active && t < tile && key <= qpos;
            float s = 0.0f;
            const bf16_t *ckv = shared_cache + t * kv_dim;
#pragma unroll
            for (int q = 0; q < VALUES_PER_LANE; ++q) {
                const int r = lane + q * HIP_WAVE;
                s += qreg[q] * gpu_bf16_to_f32(ckv[r]);
            }
            s += qpr * gpu_bf16_to_f32(ckv[rope_lane]);
            s = wave_sum(s);
            scores[t] = valid ? __shfl(s * scale, 0, HIP_WAVE) : -INFINITY;
        }

        float tile_max = scores[0];
#pragma unroll
        for (int t = 1; t < KV_TILE; ++t)
            tile_max = fmaxf(tile_max, scores[t]);
        const float next_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - next_max);
        denominator *= rescale;
#pragma unroll
        for (int q = 0; q < VALUES_PER_LANE; ++q)
            acc[q] *= rescale;
#pragma unroll
        for (int t = 0; t < KV_TILE; ++t) {
            const int key = k0 + t;
            if (t >= tile) break;
            const float weight = key <= qpos
                ? expf(scores[t] - next_max) : 0.0f;
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
            clat[qh * kv_rank + r] = acc[q] * inv_denominator;
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

/* Process one packed prompt chunk layer-major. `rows` counts real input
 * tokens; row metadata gives each token's sequence and position. Emits the
 * first generated token per sequence into g.next_token[slot]. */
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
    const ops::QuantScratch qs{g.prefill_act_i8, g.prefill_act_s};

    ops::embedding(g.prefill_x, g.embed, device_tokens, nullptr, 0, 0,
                   H, rows, g.stream);

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        ops::rmsnorm(g.prefill_xn, g.prefill_x, d.input_norm, H, rows,
                     c.rms_eps, g.stream);
        if (c.q_lora_rank > 0) {
            /* glm: q_a [rows,768,2048] + kv_a [rows,576,2048] */
            ops::gemm_dual(g.prefill_q_a, d.q_a_proj, nullptr, c.q_lora_rank,
                           g.prefill_comp, d.kv_a_proj, nullptr, KVD,
                           g.prefill_xn, H, rows, qs, g.stream);
            ops::rmsnorm(g.prefill_q_a, g.prefill_q_a, d.q_a_norm,
                         c.q_lora_rank, rows, c.mla_norm_eps, g.stream);
            /* glm: q_b [rows,5120,768] */
            ops::gemm(g.prefill_q, g.prefill_q_a, d.q_b_proj, nullptr,
                      rows, QD, c.q_lora_rank, false, qs, g.stream);
        } else {
            /* dsv: q_proj [rows,3072,2048] + kv_a [rows,576,2048] */
            ops::gemm_dual(g.prefill_q, d.q_proj, nullptr, QD,
                           g.prefill_comp, d.kv_a_proj, nullptr, KVD,
                           g.prefill_xn, H, rows, qs, g.stream);
        }

        bf16_t *layer_cache = g.kv_cache
            + (size_t)l * max_batch * g.kv_capacity * KVD;
        kv_norm_rope<<<rows, kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
            layer_cache, g.prefill_comp, d.kv_a_norm, row_batches,
            row_positions, slot_base, rows, g.kv_capacity, KVL, KVD,
            QKR, g.rope_inv_freq, c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        ops::head_gemm(g.prefill_qabs, g.prefill_q, d.W_UK_T, rows, NH,
                       KVL, QKN, QHD, KVL, (size_t)KVL * QKN, g.stream);
        q_rope<<<ops::div_up(rows * NH, kt::ATTENTION_THREADS / HIP_WAVE),
            kt::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_qrope, g.prefill_q, row_positions, rows, NH, QHD,
                QKN, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        constexpr int head_tile = kt::PREFILL_THREADS / HIP_WAVE;
        dim3 flash_grid((unsigned)ops::div_up(NH, head_tile), (unsigned)rows);
        flash_attention<kt::PREFILL_KV_TILE><<<
            flash_grid, kt::PREFILL_THREADS,
            (size_t)kt::PREFILL_KV_TILE * KVD * sizeof(bf16_t), g.stream>>>(
                g.prefill_clat, g.prefill_qabs, g.prefill_qrope,
                layer_cache, row_batches, row_positions, slot_base, rows, NH,
                g.kv_capacity, KVD, KVL, QKR, c.softmax_scale);
        HIP_LAUNCH_CHECK();

        ops::head_gemm(g.prefill_ctx, g.prefill_clat, d.W_UV, rows, NH,
                       VHD, KVL, KVL, VHD, head_stride, g.stream);
        /* o_proj: dsv [rows,2048,2048]; glm [rows,2048,5120] */
        ops::gemm(g.prefill_x, g.prefill_ctx, d.o_proj, nullptr,
                  rows, H, NH * VHD, true, qs, g.stream);

        ops::rmsnorm(g.prefill_xn, g.prefill_x, d.post_norm, H, rows,
                     c.rms_eps, g.stream);
        if (l < c.first_k_dense)
            ops::dense_ffn(c, d, rows, g.prefill_x, g.prefill_xn,
                           g.prefill_hb, qs, g.stream);
        else
            ops::moe_ffn(c, d, rows, g.prefill_x, g.prefill_xn, g.prefill_hb,
                         g.prefill_router, g.prefill_route_ids,
                         g.prefill_topk_weights, g.prefill_topk,
                         g.prefill_expert_counts, g.prefill_expert_buckets,
                         nullptr, g.prefill_routed_hidden, g.prefill_ffn,
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
              g.xn + (size_t)slot_base * H, g.lm_head, nullptr,
              batch, c.vocab_size, H, false, decode_qs, g.stream);
    ops::argmax(g.logits + (size_t)slot_base * c.vocab_size, c.vocab_size,
                g.next_token + slot_base, batch, g.stream);
}

} // namespace prefill

#endif /* MLA_PREFILL_H */
