// Baseline HIP inference for the getp grading surface.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "getp.h"
#include "tokenizer.h"
#include "kernels/getp_kernels.h"
#include "kernels/utils.h"
#include "kernels/prefill.h"
#include "kernels/rmsnom.h"

struct CopyChunk
{
    const void *host;
    size_t offset;
    size_t bytes;
};

static const size_t NO_OFFSET = std::numeric_limits<size_t>::max();

struct LayerLayout {
    size_t bytes = 0;
    std::vector<CopyChunk> chunks;
    size_t q_proj = NO_OFFSET, q_a_proj = NO_OFFSET, q_b_proj = NO_OFFSET;
    size_t kv_a_proj = NO_OFFSET, kv_b_proj = NO_OFFSET, wuk_t = NO_OFFSET;
    size_t o_proj = NO_OFFSET;
    size_t input_norm = NO_OFFSET, post_norm = NO_OFFSET;
    size_t q_a_norm = NO_OFFSET, kv_a_norm = NO_OFFSET;
    size_t dense_gate = NO_OFFSET, dense_up = NO_OFFSET, dense_down = NO_OFFSET;
    size_t moe_gate = NO_OFFSET, moe_bias = NO_OFFSET;
    size_t shared_gate = NO_OFFSET, shared_up = NO_OFFSET, shared_down = NO_OFFSET;
    std::vector<size_t> expert_gate, expert_up, expert_down;
};

struct DeviceLayer
{
    const char *pool = nullptr;
    const bf16_t *q_proj = nullptr, *q_a_proj = nullptr, *q_b_proj = nullptr;
    const bf16_t *kv_a_proj = nullptr, *W_UK = nullptr, *W_UV = nullptr;
    const bf16_t *W_UK_T = nullptr;
    const bf16_t *o_proj = nullptr;
    const bf16_t *input_norm = nullptr, *post_norm = nullptr;
    const bf16_t *q_a_norm = nullptr, *kv_a_norm = nullptr;
    const bf16_t *dense_gate = nullptr, *dense_up = nullptr, *dense_down = nullptr;
    const bf16_t *moe_gate = nullptr;
    const float *moe_bias = nullptr;
    const bf16_t *shared_gate = nullptr, *shared_up = nullptr, *shared_down = nullptr;
    const size_t *expert_gate_offsets = nullptr;
    const size_t *expert_up_offsets = nullptr;
    const size_t *expert_down_offsets = nullptr;
};

struct GpuContext {
    Transformer *owner = nullptr;
    hipStream_t stream = nullptr;
    bf16_t *embed = nullptr, *final_norm = nullptr, *lm_head = nullptr;
    float *rope_inv_freq = nullptr;
    std::vector<LayerLayout> layouts;
    std::vector<std::vector<bf16_t>> host_wuk_t;
    std::vector<DeviceLayer> layers;
    std::vector<void *> layer_pools;
    size_t *expert_offsets = nullptr;

    float *x = nullptr, *xn = nullptr;
    float *q = nullptr, *q_a = nullptr, *comp = nullptr;
    float *qrope = nullptr, *qabs = nullptr, *scores = nullptr, *clat = nullptr;
    float *ctx = nullptr, *hb = nullptr, *router = nullptr;
    float *logits = nullptr, *kv_cache = nullptr;
    int *topk = nullptr, *next_token = nullptr, *generated = nullptr;
    float *topk_weights = nullptr;
    int kv_capacity = 0;

    /* Layer-major unabsorbed prefill workspace. */
    int *prompt_tokens = nullptr, *prefill_topk = nullptr;
    int *prefill_expert_counts = nullptr, *prefill_expert_buckets = nullptr;
    float *prefill_x = nullptr, *prefill_xn = nullptr;
    float *prefill_q = nullptr, *prefill_q_a = nullptr, *prefill_comp = nullptr;
    float *prefill_qrope = nullptr, *prefill_knope = nullptr, *prefill_value = nullptr;
    float *prefill_qabs = nullptr, *prefill_clat = nullptr;
    float *prefill_scores = nullptr, *prefill_ctx = nullptr;
    float *prefill_ffn = nullptr;
    float *prefill_router = nullptr, *prefill_topk_weights = nullptr;
    float *prefill_hb = nullptr, *routed_hidden = nullptr;
    
    int *prefill_expert_offsets = nullptr;
    int *prefill_sorted_tokens = nullptr;
    int *prefill_sorted_experts = nullptr;
    int *prefill_sorted_slots = nullptr;
    int prefill_capacity = 0;
};

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
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    gemv_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(
        y, x, w, out, in, add ? 1 : 0);
    HIP_LAUNCH_CHECK();
}

static void launch_gate_up_swiglu(float *y, const float *x,
                                  const bf16_t *gate, const bf16_t *up,
                                  int out, int in)
{
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    gate_up_swiglu_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(
        y, x, gate, up, out, in);
    HIP_LAUNCH_CHECK();
}


static void launch_rmsnorm(float *y, const float *x, const bf16_t *w, int n, float eps)
{
    rmsnorm_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(y, x, w, n, eps);
    HIP_LAUNCH_CHECK();
}

static void launch_kv_norm_rope(float *cache, const float *comp,
                                const bf16_t *norm, int rows,
                                int kv_rank, int kv_dim, int rope_dim,
                                int position_base, float eps) {
    const Config &c = g.owner->config;
    kv_norm_rope_rows_kernel<<<rows, HIP_BLOCK, 0, g.stream>>>(
        cache, comp, norm, rows, kv_rank, kv_dim, rope_dim, position_base,
        g.rope_inv_freq, c.rope_interleaved, eps);
    HIP_LAUNCH_CHECK();
}


static void launch_dual_gemv(
    float *y0, const bf16_t *w0, int out0,
    float *y1, const bf16_t *w1, int out1,
    const float *x, int in) {
    int total = out0 + out1;
    int blocks = (total + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    dual_gemv_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(
        y0, w0, out0, y1, w1, out1, x, in);
    HIP_LAUNCH_CHECK();
}

static void launch_router_topk(float *scores, const float *bias,
                               int *indices, float *weights, int rows,
                               int experts, int k, int use_sigmoid,
                               int norm_topk, float routed_scale) {
    if (experts <= HIP_WAVE) {
        router_topk_wave64_kernel<<<rows, HIP_WAVE, 0, g.stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    } else {
        router_topk_kernel<<<rows, HIP_BLOCK, 0, g.stream>>>(
            scores, bias, indices, weights, rows, experts, k, use_sigmoid,
            norm_topk, routed_scale);
    }
    HIP_LAUNCH_CHECK();
}

static hipError_t try_allocate_sequence_buffers(const Config &c, int cap)
{
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    hipError_t status = hipMalloc(reinterpret_cast<void **>(&g.kv_cache),
                                  (size_t)c.n_layers * cap * KVD * sizeof(float));
    if (status != hipSuccess)
        return status;

    status = hipMalloc(reinterpret_cast<void **>(&g.scores),
                       (size_t)c.n_heads * cap * sizeof(float));
    if (status != hipSuccess)
    {
        hipError_t cleanup = hipFree(g.kv_cache);
        g.kv_cache = nullptr;
        if (cleanup != hipSuccess)
            return cleanup;
    }
    return status;
}

static void ensure_sequence_capacity(int required)
{
    const Config &c = g.owner->config;
    if (required <= g.kv_capacity)
        return;
    int cap = std::max(1, g.kv_capacity);
    while (cap < required && cap < c.max_seq_len)
    {
        long long doubled = (long long)cap * 2;
        cap = (int)std::min<long long>(doubled, c.max_seq_len);
    }
    if (cap < required)
        cap = required;
    HIP_CHECK(hipStreamSynchronize(g.stream));
    device_free(g.kv_cache);
    device_free(g.scores);
    g.kv_capacity = 0;

    hipError_t status = try_allocate_sequence_buffers(c, cap);
    HIP_CHECK(status);
    g.kv_capacity = cap;
}

static void ensure_prefill_capacity(int required) {
    const Config &c = g.owner->config;
    if (required <= g.prefill_capacity) return;
    int cap = std::max(1, g.prefill_capacity);
    while (cap < required && cap < c.max_seq_len) {
        long long doubled = (long long)cap * 2;
        cap = (int)std::min<long long>(doubled, c.max_seq_len);
    }
    if (cap < required) cap = required;
    HIP_CHECK(hipStreamSynchronize(g.stream));

    device_free(g.prompt_tokens); device_free(g.prefill_topk);
    device_free(g.prefill_expert_counts); device_free(g.prefill_expert_buckets);
    device_free(g.prefill_x); device_free(g.prefill_xn);
    device_free(g.prefill_q); device_free(g.prefill_q_a); device_free(g.prefill_comp);
    device_free(g.prefill_qrope); device_free(g.prefill_knope); device_free(g.prefill_value);
    device_free(g.prefill_qabs); device_free(g.prefill_clat);
    device_free(g.prefill_scores); device_free(g.prefill_ctx);
    device_free(g.prefill_ffn);
    device_free(g.prefill_router); device_free(g.prefill_topk_weights);
    device_free(g.prefill_hb); device_free(g.routed_hidden);
    
    device_free(g.prefill_expert_offsets);
    device_free(g.prefill_sorted_tokens);
    device_free(g.prefill_sorted_experts);
    device_free(g.prefill_sorted_slots);

    int H = c.hidden_size;
    int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    int QD = c.n_heads * QHD;
    int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    int shared_inter = c.n_shared_experts * c.moe_inter_size;
    int inter = std::max(c.dense_inter_size, std::max(shared_inter, c.moe_inter_size));
    int experts = std::max(1, c.n_routed_experts);
    int top_k = std::max(1, c.n_experts_per_tok);
    device_alloc(&g.prompt_tokens, cap);
    device_alloc(&g.prefill_x, (size_t)cap * H);
    device_alloc(&g.prefill_xn, (size_t)cap * H);
    device_alloc(&g.prefill_q, (size_t)cap * QD);
    device_alloc(&g.prefill_q_a, (size_t)cap * std::max(1, c.q_lora_rank));
    device_alloc(&g.prefill_comp, (size_t)cap * KVD);
    device_alloc(&g.prefill_qrope, (size_t)cap * c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.prefill_qabs, (size_t)cap * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.prefill_clat, (size_t)cap * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.prefill_knope, (size_t)c.n_heads * cap * c.qk_nope_head_dim);
    device_alloc(&g.prefill_value, (size_t)c.n_heads * cap * c.v_head_dim);
    size_t attention_elems = (size_t)c.n_heads * cap * cap;
    size_t slot_down_elems = (size_t)cap * top_k * H;
    device_alloc(&g.prefill_scores, std::max(attention_elems, slot_down_elems));
    device_alloc(&g.prefill_ctx, (size_t)cap * c.n_heads * c.v_head_dim);
    device_alloc(&g.prefill_ffn, (size_t)cap * H);
    device_alloc(&g.prefill_router, (size_t)cap * std::max(1, c.n_routed_experts));
    device_alloc(&g.prefill_topk, (size_t)cap * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.prefill_topk_weights, (size_t)cap * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.prefill_expert_counts, experts);
    device_alloc(&g.prefill_expert_buckets, (size_t)experts * cap);
    
    device_alloc(&g.prefill_expert_offsets, experts);
    device_alloc(&g.prefill_sorted_tokens, (size_t)cap * top_k);
    device_alloc(&g.prefill_sorted_experts, (size_t)cap * top_k);
    device_alloc(&g.prefill_sorted_slots, (size_t)cap * top_k);
    
    device_alloc(&g.prefill_hb, (size_t)cap * inter);
    device_alloc(&g.routed_hidden,
                 (size_t)cap * std::max(1, c.n_experts_per_tok) * std::max(1, c.moe_inter_size));
    g.prefill_capacity = cap;
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
    decode_routed_shared_gate_up_kernel<<<
        (gate_up_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
        HIP_BLOCK, 0, g.stream>>>(
            g.routed_hidden, g.hb, g.xn, d.pool,
            d.expert_gate_offsets, d.expert_up_offsets,
            g.topk, d.shared_gate, d.shared_up, K, inter, shared, H);
    HIP_LAUNCH_CHECK();
    decode_routed_shared_down_kernel<<<
        (H + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
        HIP_BLOCK, 0, g.stream>>>(
            g.x, g.routed_hidden, g.hb, d.pool,
            d.expert_down_offsets, g.topk, g.topk_weights, d.shared_down,
            K, H, inter, shared);
    HIP_LAUNCH_CHECK();
}

static void batched_gpu_ffn(const Config &c, const DeviceLayer &d, int l, int n_prompt) {
    const int H = c.hidden_size;
    
    // Clear expert counts to 0
    HIP_CHECK(hipMemsetAsync(g.prefill_expert_counts, 0, c.n_routed_experts * sizeof(int), g.stream));
    HIP_CHECK(hipMemsetAsync(g.prefill_expert_offsets, 0, c.n_routed_experts * sizeof(int), g.stream));
    
    int E = c.n_routed_experts;
    int K = c.n_experts_per_tok;
    int inter = c.moe_inter_size;
    int shared = c.n_shared_experts * inter;
    
    dim3 grid_router((E + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
    batched_gemv_kernel<<<grid_router, HIP_BLOCK, 0, g.stream>>>(
        g.prefill_router, g.prefill_xn, d.moe_gate, E, H, 0);
    HIP_LAUNCH_CHECK();
    
    router_topk_wave64_kernel<<<n_prompt, HIP_BLOCK, 0, g.stream>>>(
        g.prefill_router, d.moe_bias, g.prefill_topk, g.prefill_topk_weights,
        n_prompt, E, K, c.router_sigmoid, c.norm_topk, c.routed_scaling);
    HIP_LAUNCH_CHECK();
    
    int total_items = n_prompt * K;
    int blocks_items = (n_prompt + HIP_BLOCK - 1) / HIP_BLOCK;
    
    moe_token_sort_kernel<<<blocks_items, HIP_BLOCK, 0, g.stream>>>(
        g.prefill_topk, n_prompt, K, g.prefill_expert_counts);
    HIP_LAUNCH_CHECK();
    
    expert_prefix_sum_kernel<<<1, 1, 0, g.stream>>>(
        g.prefill_expert_offsets, g.prefill_expert_counts, E);
    HIP_LAUNCH_CHECK();
    
    moe_token_scatter_kernel<<<blocks_items, HIP_BLOCK, 0, g.stream>>>(
        g.prefill_topk, n_prompt, K, g.prefill_expert_offsets,
        g.prefill_sorted_tokens, g.prefill_sorted_experts, g.prefill_sorted_slots);
    HIP_LAUNCH_CHECK();
    
    dim3 grid_gate_up((inter + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, total_items);
    batched_sorted_routed_gate_up_kernel<<<grid_gate_up, HIP_BLOCK, 0, g.stream>>>(
        g.routed_hidden, g.prefill_xn, d.pool,
        d.expert_gate_offsets, d.expert_up_offsets,
        g.prefill_sorted_tokens, g.prefill_sorted_experts, g.prefill_sorted_slots,
        total_items, K, inter, H);
    HIP_LAUNCH_CHECK();
    
    dim3 grid_down((H + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, total_items);
    batched_sorted_routed_down_kernel<<<grid_down, HIP_BLOCK, 0, g.stream>>>(
        g.prefill_x, g.routed_hidden, d.pool,
        d.expert_down_offsets, g.prefill_sorted_tokens, g.prefill_sorted_experts, g.prefill_sorted_slots,
        g.prefill_topk_weights, total_items, K, H, inter);
    HIP_LAUNCH_CHECK();
    
    if (shared > 0) {
        dim3 shared_gate_up((shared + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        batched_shared_gate_up_kernel<<<shared_gate_up, HIP_BLOCK, 0, g.stream>>>(
            g.prefill_hb, g.prefill_xn, d.shared_gate, d.shared_up,
            n_prompt, shared, H);
        HIP_LAUNCH_CHECK();
        
        dim3 shared_down((H + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        batched_shared_down_kernel<<<shared_down, HIP_BLOCK, 0, g.stream>>>(
            g.prefill_x, g.prefill_hb, d.shared_down,
            n_prompt, H, shared);
        HIP_LAUNCH_CHECK();
    }
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
    embedding_kernel<<<(H + HIP_BLOCK - 1) / HIP_BLOCK, HIP_BLOCK, 0, g.stream>>>(
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
        q_absorb_rope_kernel<<<
            (qabs_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
            HIP_BLOCK, 0, g.stream>>>(
                g.qabs, g.qrope, g.q, d.W_UK_T, NH, QHD, QKN, QKR,
                KVL, pos, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();
        int score_rows = NH * kv_len;
        attention_score_kernel<<<
            (score_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
            HIP_BLOCK, 0, g.stream>>>(
                g.scores, g.qabs, g.qrope, layer_cache,
                NH, kv_len, KVD, KVL, QKR, c.softmax_scale);
        HIP_LAUNCH_CHECK();
        softmax_rows_kernel<<<NH, HIP_BLOCK, 0, g.stream>>>(
            g.scores, NH, kv_len, kv_len, 0);
        HIP_LAUNCH_CHECK();
        latent_context_kernel<<<
            (qabs_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
            HIP_BLOCK, 0, g.stream>>>(
                g.clat, g.scores, layer_cache, NH, kv_len, KVD, KVL);
        HIP_LAUNCH_CHECK();
        int value_rows = NH * VHD;
        value_up_kernel<<<(value_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                            HIP_BLOCK, 0, g.stream>>>(
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
    argmax_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(g.logits, c.vocab_size, g.next_token);
    HIP_LAUNCH_CHECK();
}

extern "C" float* gpu_forward_unabsorbed(Transformer *t, const int *tokens, int n_prompt,
                                          double *nll_out, void *tf)
{
    namespace gct = utils::constants::gpu;
    namespace mct = utils::constants::model;
    // C-compatible wrapper for prefill logic using FlashAttention-2
    if (!g.owner) return nullptr;
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
    const size_t head_stride = (size_t)(QKN + VHD) * KVL;

    ensure_prefill_capacity(n_prompt);

    HIP_CHECK(hipMemcpyAsync(g.prompt_tokens, tokens, (size_t)n_prompt * sizeof(int),
                             hipMemcpyHostToDevice, g.stream));
    constexpr bool BATCHING = true;
    printf("BATCH MODE: %b\n", BATCHING);
    // Embed all tokens one by one (since embedding_kernel is hardcoded to 1 token)
    if constexpr (BATCHING) {
        constexpr size_t EMBED_BLOCKS = 512;
        constexpr size_t EMBED_THREADS = gct::WAVE * 4;
        prefill::embed::embed<mct::HIDDEN_SIZE, EMBED_BLOCKS, EMBED_THREADS><<<EMBED_BLOCKS, EMBED_THREADS>>>(
            n_prompt, 
            g.prefill_x,
            g.embed,
            g.prompt_tokens
        );
    } else {
        for (int r = 0; r < n_prompt; ++r) {
            embedding_kernel<<<(H + HIP_BLOCK - 1) / HIP_BLOCK,
                            HIP_BLOCK, 0, g.stream>>>(
                g.prefill_x + (size_t)r * H, g.embed, 0, g.prompt_tokens + r, nullptr, 0, H);
        }
    }
    HIP_LAUNCH_CHECK();
    
    // Note: Since batched MoE kernels are missing in mla-moe, 
    // we use a loop over rows for projections (fallback for incomplete mla-moe branch)
    // Update: Loops refactored into batched kernels for Q/K/V/O projections!
    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        
        dim3 grid_b(1, n_prompt);
        batched_rmsnorm_kernel<<<grid_b, HIP_BLOCK, 0, g.stream>>>(g.prefill_xn, g.prefill_x, d.input_norm, H, c.rms_eps);
        
        dim3 grid_q_lora((c.q_lora_rank + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        dim3 grid_q((QD + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        dim3 grid_kv((KVD + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);

        if (c.q_lora_rank > 0) {
            batched_gemv_kernel<<<grid_q_lora, HIP_BLOCK, 0, g.stream>>>(g.prefill_q_a, g.prefill_xn, d.q_a_proj, c.q_lora_rank, H, 0);
            batched_rmsnorm_kernel<<<grid_b, HIP_BLOCK, 0, g.stream>>>(g.prefill_q_a, g.prefill_q_a, d.q_a_norm, c.q_lora_rank, c.mla_norm_eps);
            batched_gemv_kernel<<<grid_q, HIP_BLOCK, 0, g.stream>>>(g.prefill_q, g.prefill_q_a, d.q_b_proj, QD, c.q_lora_rank, 0);
        } else {
            batched_gemv_kernel<<<grid_q, HIP_BLOCK, 0, g.stream>>>(g.prefill_q, g.prefill_xn, d.q_proj, QD, H, 0);
        }
        batched_gemv_kernel<<<grid_kv, HIP_BLOCK, 0, g.stream>>>(g.prefill_comp, g.prefill_xn, d.kv_a_proj, KVD, H, 0);
        
        // rope + cache
        float *layer_cache_base = g.kv_cache + (size_t)l * g.kv_capacity * KVD;
        dim3 grid_kv_rope(1, n_prompt);
        batched_kv_norm_rope_rows_kernel<<<grid_kv_rope, HIP_BLOCK, 0, g.stream>>>(
            layer_cache_base, g.prefill_comp, d.kv_a_norm, KVL, KVD, QKR, 0, g.rope_inv_freq, c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        // =====================================================================
        // Compute Q_proj (qabs) and qrope for all tokens in prompt
        // =====================================================================
        int qabs_rows = NH * KVL;
        dim3 grid_qabs((qabs_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        batched_q_absorb_rope_kernel<<<grid_qabs, HIP_BLOCK, 0, g.stream>>>(
            g.prefill_qabs, g.prefill_qrope, g.prefill_q, d.W_UK_T,
            NH, QHD, QKN, QKR, KVL, 0, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        // =====================================================================
        // Wave-level Fused FA2 MLA Kernel
        // =====================================================================
        dim3 grid(n_prompt, NH);
        dim3 block(64); // One HIP_WAVE
        
        fused_prefill_mla_kernel<<<grid, block, 0, g.stream>>>(
            g.prefill_clat, 
            g.prefill_qabs, 
            g.prefill_qrope, 
            layer_cache_base,
            n_prompt, NH, KVL, QKR, KVD, c.softmax_scale);
        HIP_LAUNCH_CHECK();
        
        // =====================================================================
        // Post-compute V-projection (Ctx = O_latent * W_UV)
        // =====================================================================
        int value_rows = NH * VHD;
        dim3 grid_vup((value_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        batched_value_up_kernel<<<grid_vup, HIP_BLOCK, 0, g.stream>>>(
            g.prefill_ctx, g.prefill_clat, d.W_UV, NH, VHD, KVL, head_stride);
        HIP_LAUNCH_CHECK();
        // =====================================================================

        // O-Projection and RMSNorm
        dim3 grid_o((H + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
        batched_gemv_kernel<<<grid_o, HIP_BLOCK, 0, g.stream>>>(g.prefill_x, g.prefill_ctx, d.o_proj, H, NH * VHD, 1);
        batched_rmsnorm_kernel<<<grid_b, HIP_BLOCK, 0, g.stream>>>(g.prefill_xn, g.prefill_x, d.post_norm, H, c.rms_eps);
        HIP_LAUNCH_CHECK();
        
        // FFN / Batched MoE
        if (l < c.first_k_dense) {
            int inter = c.dense_inter_size;
            dim3 grid_dense_gate_up((inter + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
            batched_gate_up_swiglu_kernel<<<grid_dense_gate_up, HIP_BLOCK, 0, g.stream>>>(
                g.prefill_hb, g.prefill_xn, d.dense_gate, d.dense_up, inter, H);
            HIP_LAUNCH_CHECK();
            
            dim3 grid_dense_down((H + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK, n_prompt);
            batched_gemv_kernel<<<grid_dense_down, HIP_BLOCK, 0, g.stream>>>(
                g.prefill_x, g.prefill_hb, d.dense_down, H, inter, 1);
            HIP_LAUNCH_CHECK();
        } else {
            batched_gpu_ffn(c, d, l, n_prompt);
        }
    }
    
    // Final logits
    float *host_logits = (float*)malloc((size_t)c.vocab_size * sizeof(float));
    launch_rmsnorm(g.xn, g.prefill_x + (size_t)(n_prompt - 1) * H, g.final_norm, H, c.rms_eps);
    launch_gemv(g.logits, g.xn, g.lm_head, c.vocab_size, H);
    
    // Sample the next token!
    argmax_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(g.logits, c.vocab_size, g.next_token);
    HIP_LAUNCH_CHECK();
    
    HIP_CHECK(hipMemcpyAsync(host_logits, g.logits, c.vocab_size * sizeof(float),
                             hipMemcpyDeviceToHost, g.stream));
    HIP_CHECK(hipStreamSynchronize(g.stream));
    return host_logits;
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
    if (HIP_BLOCK > prop.maxThreadsPerBlock || c.v_head_dim <= 0 ||
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
    int initial_cap = std::max(1, std::min(c.max_seq_len, 2048));
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
    device_alloc(&g.generated, c.max_seq_len);
    g.kv_capacity = 0;
    ensure_sequence_capacity(initial_cap);
    HIP_CHECK(hipStreamSynchronize(g.stream));

    std::fprintf(stderr, "[getp HIP] device=%s vram=%.2f GiB weights=%.2f GiB mode=resident initial_kv=%d\n",
                 prop.name, (double)total_bytes / (double)(1ULL << 30),
                 (double)(global_bytes + layer_total) / (double)(1ULL << 30),
                 g.kv_capacity);
}

extern "C" void finish(Transformer *t) {
    (void)t;
    if (!g.owner) return;
    HIP_CHECK(hipStreamSynchronize(g.stream));
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
    device_free(g.prefill_expert_offsets); device_free(g.prefill_sorted_tokens);
    device_free(g.prefill_sorted_experts); device_free(g.prefill_sorted_slots);
    device_free(g.prefill_x); device_free(g.prefill_xn);
    device_free(g.prefill_q); device_free(g.prefill_q_a); device_free(g.prefill_comp);
    device_free(g.prefill_qrope); device_free(g.prefill_knope); device_free(g.prefill_value);
    device_free(g.prefill_qabs); device_free(g.prefill_clat);
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
    const int max_seq = t->config.max_seq_len;
    const int eos = tokenizer_eos_id(t->tokenizer);
    std::vector<std::vector<int>> prompts((size_t)reqs->num_reqs);
    int required_capacity = 1;
    for (int r = 0; r < reqs->num_reqs; ++r)
    {
        prompts[r].resize(max_seq);
        int n = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                 prompts[r].data(), max_seq);
        if (n < 0)
            n = 0;
        prompts[r].resize((size_t)n);
        int budget = max_seq - n;
        int steps = std::min(reqs->max_steps, budget);
        if (steps < 0)
            steps = 0;
        required_capacity = std::max(required_capacity, n + steps);
    }
    ensure_sequence_capacity(required_capacity);

    long long total = 0;
    std::vector<int> generated((size_t)std::max(0, reqs->max_steps));
    for (int r = 0; r < reqs->num_reqs; ++r)
    {
        reqs->out_lens[r] = 0;
        int n_prompt = (int)prompts[r].size();
        if (n_prompt < 1)
            continue;
        int budget = max_seq - n_prompt;
        int steps = std::min(reqs->max_steps, budget);
        if (steps < 0)
            steps = 0;

        // Use FlashAttention-2 prefill instead of naive loop
        gpu_forward_unabsorbed(t, prompts[r].data(), n_prompt, nullptr, nullptr);
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
