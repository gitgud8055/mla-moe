// Baseline HIP inference for the getp grading surface.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <limits>
#include <vector>

#include "getp.h"
#include "tokenizer.h"
#include "kernels/getp_kernels.h"
#include "kernels/getp_utils.h"
#include "kernels/getp_state.h"
#include "kernels/prefill.h"


static GpuContext g;

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t add_chunk(LayerLayout &layout, const void *host, size_t bytes) {
    layout.bytes = align_up(layout.bytes, 256);
    size_t offset = layout.bytes;
    layout.chunks.push_back({host, offset, bytes});
    layout.bytes += bytes;
    return offset;
}

static LayerLayout make_layer_layout(const Transformer *t, int l,
                                     const bf16_t *wuk_t) {
    const Config &c = t->config;
    const ModelWeights &w = t->weights;
    LayerLayout a;
    const int H = c.hidden_size;
    const int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    const int QD = c.n_heads * QHD;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    const int AOUT = c.n_heads * c.v_head_dim;
    auto bfbytes = [](size_t n) { return n * sizeof(bf16_t); };

    if (c.q_lora_rank > 0) {
        a.q_a_proj = add_chunk(a, w.q_a_proj[l], bfbytes((size_t)c.q_lora_rank * H));
        a.q_b_proj = add_chunk(a, w.q_b_proj[l], bfbytes((size_t)QD * c.q_lora_rank));
        a.q_a_norm = add_chunk(a, w.q_a_layernorm[l], bfbytes(c.q_lora_rank));
    } else {
        a.q_proj = add_chunk(a, w.q_proj[l], bfbytes((size_t)QD * H));
    }
    a.kv_a_proj = add_chunk(a, w.kv_a_proj[l], bfbytes((size_t)KVD * H));
    size_t kv_b_elems = (size_t)c.n_heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank;
    a.kv_b_proj = add_chunk(a, w.W_UK[l], bfbytes(kv_b_elems));
    size_t wuk_elems = (size_t)c.n_heads * c.kv_lora_rank * c.qk_nope_head_dim;
    a.wuk_t = add_chunk(a, wuk_t, bfbytes(wuk_elems));
    a.o_proj = add_chunk(a, w.o_proj[l], bfbytes((size_t)H * AOUT));
    a.input_norm = add_chunk(a, w.input_layernorm[l], bfbytes(H));
    a.post_norm = add_chunk(a, w.post_attn_norm[l], bfbytes(H));
    a.kv_a_norm = add_chunk(a, w.kv_a_layernorm[l], bfbytes(c.kv_lora_rank));

    if (l < c.first_k_dense) {
        size_t ih = (size_t)c.dense_inter_size * H;
        a.dense_gate = add_chunk(a, w.dense_gate[l], bfbytes(ih));
        a.dense_up = add_chunk(a, w.dense_up[l], bfbytes(ih));
        a.dense_down = add_chunk(a, w.dense_down[l], bfbytes(ih));
    } else {
        a.moe_gate = add_chunk(a, w.moe_gate[l], bfbytes((size_t)c.n_routed_experts * H));
        if (w.moe_gate_bias[l])
            a.moe_bias = add_chunk(a, w.moe_gate_bias[l], (size_t)c.n_routed_experts * sizeof(float));
        int shared = c.n_shared_experts * c.moe_inter_size;
        size_t sh = (size_t)shared * H;
        a.shared_gate = add_chunk(a, w.shared_gate[l], bfbytes(sh));
        a.shared_up = add_chunk(a, w.shared_up[l], bfbytes(sh));
        a.shared_down = add_chunk(a, w.shared_down[l], bfbytes(sh));
        a.expert_gate.resize(c.n_routed_experts);
        a.expert_up.resize(c.n_routed_experts);
        a.expert_down.resize(c.n_routed_experts);
        size_t eh = (size_t)c.moe_inter_size * H;
        for (int e = 0; e < c.n_routed_experts; ++e) {
            a.expert_gate[e] = add_chunk(a, w.expert_gate[l][e], bfbytes(eh));
            a.expert_up[e] = add_chunk(a, w.expert_up[l][e], bfbytes(eh));
            a.expert_down[e] = add_chunk(a, w.expert_down[l][e], bfbytes(eh));
        }
    }
    a.bytes = align_up(a.bytes, 256);
    return a;
}

template <typename T>
static void device_alloc(T **ptr, size_t count) {
    if (count == 0) count = 1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)));
}


template <typename T>
static void device_free(T *&ptr) {
    if (ptr) HIP_CHECK(hipFree(reinterpret_cast<void *>(ptr)));
    ptr = nullptr;
}

static const void *offset_ptr(const char *base, size_t offset) {
    return offset == NO_OFFSET ? nullptr : base + offset;
}

static DeviceLayer materialize_layer(const LayerLayout &a, void *pool, int l) {
    const Config &c = g.owner->config;
    const char *base = reinterpret_cast<const char *>(pool);
    DeviceLayer d;
    d.pool = base;
    d.q_proj = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.q_proj));
    d.q_a_proj = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.q_a_proj));
    d.q_b_proj = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.q_b_proj));
    d.kv_a_proj = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.kv_a_proj));
    d.W_UK = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.kv_b_proj));
    d.W_UV = d.W_UK ? d.W_UK + (size_t)c.qk_nope_head_dim * c.kv_lora_rank : nullptr;
    d.W_UK_T = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.wuk_t));
    d.o_proj = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.o_proj));
    d.input_norm = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.input_norm));
    d.post_norm = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.post_norm));
    d.q_a_norm = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.q_a_norm));
    d.kv_a_norm = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.kv_a_norm));
    d.dense_gate = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.dense_gate));
    d.dense_up = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.dense_up));
    d.dense_down = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.dense_down));
    d.moe_gate = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.moe_gate));
    d.moe_bias = reinterpret_cast<const float *>(offset_ptr(base, a.moe_bias));
    d.shared_gate = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.shared_gate));
    d.shared_up = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.shared_up));
    d.shared_down = reinterpret_cast<const bf16_t *>(offset_ptr(base, a.shared_down));
    if (c.n_routed_experts > 0) {
        size_t layer_base = (size_t)l * 3 * c.n_routed_experts;
        d.expert_gate_offsets = g.expert_offsets + layer_base;
        d.expert_up_offsets = g.expert_offsets + layer_base + c.n_routed_experts;
        d.expert_down_offsets = g.expert_offsets + layer_base + 2 * c.n_routed_experts;
    }
    return d;
}

static void enqueue_layer_copy(const LayerLayout &a, void *pool) {
    char *base = reinterpret_cast<char *>(pool);
    for (const CopyChunk &chunk : a.chunks)
        HIP_CHECK(hipMemcpyAsync(base + chunk.offset, chunk.host, chunk.bytes,
                                 hipMemcpyHostToDevice, g.stream));
}

static void launch_gemv(float *y, const float *x, const bf16_t *w,
                        int out, int in, bool add = false) {
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    int blocks = (out + rows_per_block - 1) / rows_per_block;
    projection_kernels::gemv_kernel<kernel_tuning::GEMV_THREADS><<<
        blocks, kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
        y, x, w, out, in, add ? 1 : 0);
    HIP_LAUNCH_CHECK();
}

static void launch_gate_up_swiglu(float *y, const float *x,
                                  const bf16_t *gate, const bf16_t *up,
                                  int out, int in)
{
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    int blocks = (out + rows_per_block - 1) / rows_per_block;
    projection_kernels::gate_up_swiglu_kernel<kernel_tuning::GEMV_THREADS><<<
        blocks, kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
        y, x, gate, up, out, in);
    HIP_LAUNCH_CHECK();
}


static void launch_rmsnorm(float *y, const float *x, const bf16_t *w, int n, float eps)
{
    elementwise_kernels::rmsnorm_kernel<kernel_tuning::ELEMENTWISE_THREADS><<<
        1, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(y, x, w, n, eps);
    HIP_LAUNCH_CHECK();
}

static void launch_kv_norm_rope(float *cache, const float *comp,
                                const bf16_t *norm, int rows,
                                int kv_rank, int kv_dim, int rope_dim,
                                int position_base, float eps) {
    const Config &c = g.owner->config;
    attention_kernels::kv_norm_rope_rows_kernel<kernel_tuning::ATTENTION_THREADS><<<
        rows, kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
        cache, comp, norm, rows, kv_rank, kv_dim, rope_dim, position_base,
        g.rope_inv_freq, c.rope_interleaved, eps);
    HIP_LAUNCH_CHECK();
}


static void launch_dual_gemv(
    float *y0, const bf16_t *w0, int out0,
    float *y1, const bf16_t *w1, int out1,
    const float *x, int in) {
    int total = out0 + out1;
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    int blocks = (total + rows_per_block - 1) / rows_per_block;
    projection_kernels::dual_gemv_kernel<kernel_tuning::GEMV_THREADS><<<
        blocks, kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
        y0, w0, out0, y1, w1, out1, x, in);
    HIP_LAUNCH_CHECK();
}

static void launch_router_topk(float *scores, const float *bias,
                               int *indices, float *weights, int rows,
                               int experts, int k, int use_sigmoid,
                               int norm_topk, float routed_scale) {
    if (experts <= HIP_WAVE) {
        moe_kernels::router_topk_wave64_kernel<kernel_tuning::ROUTER_WAVE_THREADS><<<
            rows, kernel_tuning::ROUTER_WAVE_THREADS, 0, g.stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    } else {
        moe_kernels::router_topk_kernel<kernel_tuning::ROUTER_THREADS><<<
            rows, kernel_tuning::ROUTER_THREADS, 0, g.stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    }
    HIP_LAUNCH_CHECK();
}

static void gpu_ffn(const Config &c, const DeviceLayer &d, int l) {
    const int H = c.hidden_size;
    if (l < c.first_k_dense) {
        int inter = c.dense_inter_size;
        launch_gate_up_swiglu(g.hb, g.xn, d.dense_gate, d.dense_up, inter, H);
        launch_gemv(g.x, g.hb, d.dense_down, H, inter, true);
        return;
    }

    int E = c.n_routed_experts;
    int K = c.n_experts_per_tok;
    int inter = c.moe_inter_size;
    launch_gemv(g.router, g.xn, d.moe_gate, E, H);
    launch_router_topk(g.router, d.moe_bias, g.topk, g.topk_weights,
                       1, E, K, c.router_sigmoid, c.norm_topk,
                       c.routed_scaling);
    int shared = c.n_shared_experts * inter;
    int gate_up_rows = K * inter + shared;
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    moe_kernels::decode_routed_shared_gate_up_kernel<kernel_tuning::GEMV_THREADS><<<
        (gate_up_rows + rows_per_block - 1) / rows_per_block,
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            g.routed_hidden, g.hb, g.xn, d.pool,
            d.expert_gate_offsets, d.expert_up_offsets,
            g.topk, d.shared_gate, d.shared_up, K, inter, shared, H);
    HIP_LAUNCH_CHECK();
    moe_kernels::decode_routed_shared_down_kernel<kernel_tuning::GEMV_THREADS><<<
        (H + rows_per_block - 1) / rows_per_block,
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            g.x, g.routed_hidden, g.hb, d.pool,
            d.expert_down_offsets, g.topk, g.topk_weights, d.shared_down,
            K, H, inter, shared);
    HIP_LAUNCH_CHECK();
}

static void gpu_forward_absorbed(int host_token, const int *device_token,
                              int *generated, int output_index,
                              int pos, bool produce_token)
{
    const Config &c = g.owner->config;
    const int H = c.hidden_size;
    const int NH = c.n_heads;
    const int QKN = c.qk_nope_head_dim;
    const int QKR = c.qk_rope_head_dim;
    const int QHD = QKN + QKR;
    const int QD = NH * QHD;
    const int KVL = c.kv_lora_rank;
    const int KVD = KVL + QKR;
    const int VHD = c.v_head_dim;
    const int kv_len = pos + 1;
    const size_t head_stride = (size_t)(QKN + VHD) * KVL;

    if (pos < 0 || pos >= g.kv_capacity)
    {
        std::fprintf(stderr, "HIP inference: position %d exceeds KV capacity %d\n", pos, g.kv_capacity);
        std::exit(EXIT_FAILURE);
    }
    elementwise_kernels::embedding_kernel<kernel_tuning::ELEMENTWISE_THREADS><<<
        (H + kernel_tuning::ELEMENTWISE_THREADS - 1) /
            kernel_tuning::ELEMENTWISE_THREADS,
        kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
        g.x, g.embed, host_token, device_token, generated, output_index, H);
    HIP_LAUNCH_CHECK();

    for (int l = 0; l < c.n_layers; ++l)
    {
        const DeviceLayer &d = g.layers[l];
        launch_rmsnorm(g.xn, g.x, d.input_norm, H, c.rms_eps);
        if (c.q_lora_rank > 0)
        {
            launch_dual_gemv(
                g.q_a, d.q_a_proj, c.q_lora_rank,
                g.comp, d.kv_a_proj, KVD, g.xn, H);
            launch_rmsnorm(g.q_a, g.q_a, d.q_a_norm, c.q_lora_rank, c.mla_norm_eps);
            launch_gemv(g.q, g.q_a, d.q_b_proj, QD, c.q_lora_rank);
        }
        else
        {
            launch_dual_gemv(
                g.q, d.q_proj, QD,
                g.comp, d.kv_a_proj, KVD, g.xn, H);
        }
        float *layer_cache = g.kv_cache + (size_t)l * g.kv_capacity * KVD;
        float *cache_pos = layer_cache + (size_t)pos * KVD;
        launch_kv_norm_rope(cache_pos, g.comp, d.kv_a_norm,
                            1, KVL, KVD, QKR, pos, c.mla_norm_eps);

        int qabs_rows = NH * KVL;
        attention_kernels::q_absorb_rope_kernel<kernel_tuning::ATTENTION_THREADS><<<
            (qabs_rows + kernel_tuning::ATTENTION_THREADS / HIP_WAVE - 1) /
                (kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.qabs, g.qrope, g.q, d.W_UK_T, NH, QHD, QKN, QKR,
                KVL, pos, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();
        int score_rows = NH * kv_len;
        attention_kernels::attention_score_kernel<kernel_tuning::ATTENTION_THREADS><<<
            (score_rows + kernel_tuning::ATTENTION_THREADS / HIP_WAVE - 1) /
                (kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.scores, g.qabs, g.qrope, layer_cache,
                NH, kv_len, KVD, KVL, QKR, c.softmax_scale);
        HIP_LAUNCH_CHECK();
        attention_kernels::softmax_rows_kernel<kernel_tuning::ATTENTION_THREADS><<<
            NH, kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
            g.scores, NH, kv_len, kv_len, 0);
        HIP_LAUNCH_CHECK();
        attention_kernels::latent_context_kernel<kernel_tuning::ATTENTION_THREADS><<<
            (qabs_rows + kernel_tuning::ATTENTION_THREADS / HIP_WAVE - 1) /
                (kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.clat, g.scores, layer_cache, NH, kv_len, KVD, KVL);
        HIP_LAUNCH_CHECK();
        int value_rows = NH * VHD;
        attention_kernels::value_up_kernel<kernel_tuning::ATTENTION_THREADS><<<
            (value_rows + kernel_tuning::ATTENTION_THREADS / HIP_WAVE - 1) /
                (kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
            g.ctx, g.clat, d.W_UV, NH, VHD, KVL, head_stride);
        HIP_LAUNCH_CHECK();
        launch_gemv(g.x, g.ctx, d.o_proj, H, NH * VHD, true);

        launch_rmsnorm(g.xn, g.x, d.post_norm, H, c.rms_eps);
        gpu_ffn(c, d, l);
    }

    if (!produce_token)
        return;
    launch_rmsnorm(g.xn, g.x, g.final_norm, H, c.rms_eps);
    launch_gemv(g.logits, g.xn, g.lm_head, c.vocab_size, H);
    elementwise_kernels::argmax_kernel<kernel_tuning::ELEMENTWISE_THREADS><<<
        1, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
        g.logits, c.vocab_size, g.next_token);
    HIP_LAUNCH_CHECK();
}

extern "C" void warm_up(Transformer *t) {
    if (g.owner) {
        std::fprintf(stderr, "warm_up: HIP context is already initialized\n");
        std::exit(EXIT_FAILURE);
    }
    const Config &c = t->config;
    if (c.n_routed_experts < 1 || c.n_routed_experts > 1024 ||
        c.n_experts_per_tok < 1 || c.n_experts_per_tok > 16 ||
        c.n_experts_per_tok > c.n_routed_experts) {
        std::fprintf(stderr, "HIP baseline requires 1..1024 experts and valid top-k <= 16\n");
        std::exit(EXIT_FAILURE);
    }
    g.owner = t;
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (prop.warpSize != HIP_WAVE) {
        std::fprintf(stderr, "HIP baseline requires wavefront-64, device reports %d\n", prop.warpSize);
        std::exit(EXIT_FAILURE);
    }
    if (kernel_tuning::GEMV_THREADS > prop.maxThreadsPerBlock ||
        kernel_tuning::ELEMENTWISE_THREADS > prop.maxThreadsPerBlock ||
        kernel_tuning::ATTENTION_THREADS > prop.maxThreadsPerBlock ||
        kernel_tuning::ROUTER_THREADS > prop.maxThreadsPerBlock ||
        c.v_head_dim <= 0 ||
        c.v_head_dim > prop.maxThreadsPerBlock ||
        c.qk_rope_head_dim <= 0 || (c.qk_rope_head_dim & 1) ||
        c.qk_rope_head_dim / 2 > prop.maxThreadsPerBlock) {
        std::fprintf(stderr, "HIP inference dimensions exceed the device block limit\n");
        std::exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipStreamCreateWithFlags(&g.stream, hipStreamNonBlocking));

    g.layouts.reserve(c.n_layers);
    g.host_wuk_t.resize(c.n_layers);
    size_t layer_total = 0;
    for (int l = 0; l < c.n_layers; ++l) {
        size_t elems = (size_t)c.n_heads * c.kv_lora_rank * c.qk_nope_head_dim;
        g.host_wuk_t[l].resize(elems);
        const bf16_t *src = t->weights.W_UK[l];
        size_t src_head_stride =
            (size_t)(c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank;
        for (int h = 0; h < c.n_heads; ++h) {
            const bf16_t *src_h = src + (size_t)h * src_head_stride;
            bf16_t *dst_h = g.host_wuk_t[l].data()
                          + (size_t)h * c.kv_lora_rank * c.qk_nope_head_dim;
            for (int r = 0; r < c.kv_lora_rank; ++r)
                for (int d = 0; d < c.qk_nope_head_dim; ++d)
                    dst_h[(size_t)r * c.qk_nope_head_dim + d] =
                        src_h[(size_t)d * c.kv_lora_rank + r];
        }
        g.layouts.push_back(make_layer_layout(t, l, g.host_wuk_t[l].data()));
        layer_total += g.layouts.back().bytes;
    }
    size_t global_bytes = ((size_t)2 * c.vocab_size * c.hidden_size + c.hidden_size) * sizeof(bf16_t)
                        + (size_t)(c.qk_rope_head_dim / 2) * sizeof(float);
    size_t free_bytes = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));

    device_alloc(&g.embed, (size_t)c.vocab_size * c.hidden_size);
    device_alloc(&g.final_norm, c.hidden_size);
    device_alloc(&g.lm_head, (size_t)c.vocab_size * c.hidden_size);
    device_alloc(&g.rope_inv_freq, c.qk_rope_head_dim / 2);
    HIP_CHECK(hipMemcpyAsync(g.embed, t->weights.embed_tokens,
                             (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    HIP_CHECK(hipMemcpyAsync(g.final_norm, t->weights.norm,
                             (size_t)c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    HIP_CHECK(hipMemcpyAsync(g.lm_head, t->weights.lm_head,
                             (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    HIP_CHECK(hipMemcpyAsync(g.rope_inv_freq, t->rope_inv_freq,
                             (size_t)(c.qk_rope_head_dim / 2) * sizeof(float),
                             hipMemcpyHostToDevice, g.stream));

    size_t offset_count = (size_t)c.n_layers * 3 * std::max(1, c.n_routed_experts);
    std::vector<size_t> host_offsets(offset_count, 0);
    if (c.n_routed_experts > 0) {
        for (int l = c.first_k_dense; l < c.n_layers; ++l) {
            size_t base = (size_t)l * 3 * c.n_routed_experts;
            for (int e = 0; e < c.n_routed_experts; ++e) {
                host_offsets[base + e] = g.layouts[l].expert_gate[e];
                host_offsets[base + c.n_routed_experts + e] = g.layouts[l].expert_up[e];
                host_offsets[base + 2 * c.n_routed_experts + e] = g.layouts[l].expert_down[e];
            }
        }
    }
    device_alloc(&g.expert_offsets, offset_count);
    HIP_CHECK(hipMemcpy(g.expert_offsets, host_offsets.data(), offset_count * sizeof(size_t),
                        hipMemcpyHostToDevice));

    /* Keep all layer weights resident in VRAM for the entire inference run.
     * Each layer owns one packed pool; forward only dereferences g.layers[l]
     * and performs no per-layer host-to-device weight copies. */
    g.layers.resize(c.n_layers);
    g.layer_pools.resize(c.n_layers, nullptr);
    for (int l = 0; l < c.n_layers; ++l) {
        void *pool = nullptr;
        HIP_CHECK(hipMalloc(&pool, g.layouts[l].bytes));
        g.layer_pools[l] = pool;
        enqueue_layer_copy(g.layouts[l], pool);
        g.layers[l] = materialize_layer(g.layouts[l], pool, l);
    }
    int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    int shared = c.n_shared_experts * c.moe_inter_size;
    int inter = std::max(c.dense_inter_size, std::max(shared, c.moe_inter_size));

    // prefill
    device_alloc(&g.xs, constants::datasets::MAX_SEQ * c.hidden_size);

    device_alloc(&g.x, c.hidden_size);
    device_alloc(&g.xn, c.hidden_size);
    device_alloc(&g.q, (size_t)c.n_heads * QHD);
    device_alloc(&g.q_a, std::max(1, c.q_lora_rank));
    device_alloc(&g.comp, KVD);
    device_alloc(&g.qrope, (size_t)c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.qabs, (size_t)c.n_heads * c.kv_lora_rank);
    device_alloc(&g.clat, (size_t)c.n_heads * c.kv_lora_rank);
    device_alloc(&g.ctx, (size_t)c.n_heads * c.v_head_dim);
    device_alloc(&g.hb, inter);
    device_alloc(&g.routed_hidden, (size_t)c.n_experts_per_tok * c.moe_inter_size);
    device_alloc(&g.router, std::max(1, c.n_routed_experts));
    device_alloc(&g.topk, std::max(1, c.n_experts_per_tok));
    device_alloc(&g.topk_weights, std::max(1, c.n_experts_per_tok));
    device_alloc(&g.logits, c.vocab_size);
    device_alloc(&g.next_token, 1);
    device_alloc(&g.generated, constants::datasets::MAX_SEQ);

    /* KV + attention scores for the full dataset bound, allocated once here.
     * inference() never resizes; a position past this is caught by the
     * pos >= g.kv_capacity check in gpu_forward_absorbed(). */
    g.kv_capacity = std::max(1, std::min(c.max_seq_len,
                                         (int)constants::datasets::MAX_SEQ));
    device_alloc(&g.kv_cache, (size_t)c.n_layers * g.kv_capacity * KVD);
    device_alloc(&g.scores, (size_t)c.n_heads * g.kv_capacity);

    g.prompts.resize(constants::datasets::MAX_REQS);
    for (std::vector<int> &p : g.prompts)
        p.reserve(constants::datasets::MAX_N_PROMPT);
    g.h_generated.resize(constants::datasets::MAX_SEQ);

    HIP_CHECK(hipStreamSynchronize(g.stream));

    std::fprintf(stderr, "[getp HIP] device=%s vram=%.2f GiB weights=%.2f GiB mode=resident kv_cap=%d\n",
                 prop.name, (double)total_bytes / (double)(1ULL << 30),
                 (double)(global_bytes + layer_total) / (double)(1ULL << 30),
                 g.kv_capacity);
}

extern "C" void finish(Transformer *t) {
    (void)t;
    if (!g.owner) return;
    HIP_CHECK(hipStreamSynchronize(g.stream));

    // prefill
    device_free(g.xs);

    device_free(g.embed);
    device_free(g.final_norm);
    device_free(g.lm_head);
    device_free(g.rope_inv_freq);
    for (void *pool : g.layer_pools) if (pool) HIP_CHECK(hipFree(pool));
    g.layer_pools.clear();
    device_free(g.expert_offsets);
    device_free(g.x); device_free(g.xn);
    device_free(g.q); device_free(g.q_a); device_free(g.comp);
    device_free(g.qrope); device_free(g.qabs); device_free(g.scores); device_free(g.clat);
    device_free(g.ctx); device_free(g.hb); device_free(g.router);
    device_free(g.topk); device_free(g.topk_weights); device_free(g.logits);
    device_free(g.kv_cache); device_free(g.next_token); device_free(g.generated);
    device_free(g.prompt_tokens); device_free(g.prefill_topk);
    device_free(g.prefill_expert_counts); device_free(g.prefill_expert_buckets);
    device_free(g.prefill_x); device_free(g.prefill_xn);
    device_free(g.prefill_q); device_free(g.prefill_q_a); device_free(g.prefill_comp);
    device_free(g.prefill_qrope); device_free(g.prefill_knope); device_free(g.prefill_value);
    device_free(g.prefill_scores); device_free(g.prefill_ctx);
    device_free(g.prefill_ffn);
    device_free(g.prefill_router); device_free(g.prefill_topk_weights);
    device_free(g.prefill_hb); device_free(g.routed_hidden);
    HIP_CHECK(hipStreamDestroy(g.stream));
    g = GpuContext();
}

extern "C" long long inference(Transformer *t, Requests *reqs)
{
    if (g.owner != t)
    {
        std::fprintf(stderr, "inference: warm_up() was not called for this Transformer\n");
        std::exit(EXIT_FAILURE);
    }
    if (!t->tokenizer)
    {
        std::fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        std::exit(EXIT_FAILURE);
    }
    if ((size_t)reqs->num_reqs > constants::datasets::MAX_REQS)
    {
        std::fprintf(stderr, "inference: %d requests exceeds MAX_REQS %zu\n",
                     reqs->num_reqs, constants::datasets::MAX_REQS);
        std::exit(EXIT_FAILURE);
    }
    /* Buffers were all sized in warm_up() from constants::datasets, so nothing
     * below allocates: prompts[] is MAX_REQS x MAX_N_PROMPT, h_generated is
     * MAX_SEQ, and the KV cache is fixed at g.kv_capacity. */
    const int prompt_cap = (int)constants::datasets::MAX_N_PROMPT;
    const int eos = tokenizer_eos_id(t->tokenizer);
    std::vector<std::vector<int>> &prompts = g.prompts;
    std::vector<int> &generated = g.h_generated;
    for (int r = 0; r < reqs->num_reqs; ++r)
    {
        prompts[r].resize((size_t)prompt_cap);
        int n = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                 prompts[r].data(), prompt_cap);
        if (n < 0)
            n = 0;
        prompts[r].resize((size_t)n);
    }

    long long total = 0;
    for (int r = 0; r < reqs->num_reqs; ++r)
    {
        reqs->out_lens[r] = 0;
        int n_prompt = (int)prompts[r].size();
        if (n_prompt < 1)
            continue;
        /* Cap generation so prompt + decode stays within the KV cache. */
        int budget = g.kv_capacity - n_prompt;
        int steps = std::min(reqs->max_steps, budget);
        if (steps < 0)
            steps = 0;

        // prefill::run(g, prompts[r]);

        for (int p = 0; p < n_prompt; ++p)
            gpu_forward_absorbed(prompts[r][p], nullptr, nullptr, 0,
                              p, p == n_prompt - 1);
        int pos = n_prompt;
        for (int step = 0; step < steps; ++step)
        {
            gpu_forward_absorbed(0, g.next_token, g.generated, step,
                              pos, true);
            ++pos;
        }
        if (steps > 0)
        {
            HIP_CHECK(hipMemcpyAsync(generated.data(), g.generated,
                                     (size_t)steps * sizeof(int),
                                     hipMemcpyDeviceToHost, g.stream));
            HIP_CHECK(hipStreamSynchronize(g.stream));
        }
        int n_out = 0;
        while (n_out < steps && generated[n_out] != eos)
        {
            reqs->out_tokens[r][n_out] = generated[n_out];
            ++n_out;
        }
        reqs->out_lens[r] = n_out;
        total += n_out;
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));
    return total;
}
