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


static void hip_fail(hipError_t status, const char *expr, const char *file, int line)
{
    if (status == hipSuccess)
        return;
    std::fprintf(stderr, "HIP failure at %s:%d: %s: %s\n", file, line, expr,
                 hipGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define HIP_CHECK(expr) hip_fail((expr), #expr, __FILE__, __LINE__)
#define HIP_LAUNCH_CHECK() HIP_CHECK(hipGetLastError())

struct CopyChunk
{
    const void *host;
    size_t offset;
    size_t bytes;
};

static const size_t NO_OFFSET = std::numeric_limits<size_t>::max();

struct LayerLayout
{
    size_t bytes = 0;
    std::vector<CopyChunk> chunks;
    size_t q_proj = NO_OFFSET, q_a_proj = NO_OFFSET, q_b_proj = NO_OFFSET;
    size_t kv_a_proj = NO_OFFSET, kv_b_proj = NO_OFFSET, o_proj = NO_OFFSET;
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

struct GpuContext
{
    Transformer *owner = nullptr;
    hipStream_t stream = nullptr;

    bf16_t *embed = nullptr, *final_norm = nullptr, *lm_head = nullptr;
    float *rope_inv_freq = nullptr;
    std::vector<LayerLayout> layouts;
    std::vector<DeviceLayer> layers;
    std::vector<void *> layer_pools;
    void *staging_pool = nullptr;
    size_t *expert_offsets = nullptr;

    float *x = nullptr, *xn = nullptr, *attn_out = nullptr, *ffn_out = nullptr;
    float *q = nullptr, *q_a = nullptr, *comp = nullptr;
    float *qrope = nullptr, *qabs = nullptr, *scores = nullptr, *clat = nullptr;
    float *ctx = nullptr, *hb = nullptr, *router = nullptr;
    float *logits = nullptr, *kv_cache = nullptr;
    int *topk = nullptr, *next_token = nullptr, *generated = nullptr;
    float *topk_weights = nullptr;
    int kv_capacity = 0;
};

static GpuContext g;

// align 256-byte
static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t add_chunk(LayerLayout &layout, const void *host, size_t bytes)
{
    layout.bytes = align_up(layout.bytes, 256);
    size_t offset = layout.bytes;
    layout.chunks.push_back({host, offset, bytes});
    layout.bytes += bytes;
    return offset;
}

static LayerLayout make_layer_layout(const Transformer *t, int l)
{
    const Config &c = t->config;
    const ModelWeights &w = t->weights;
    LayerLayout a;
    const int H = c.hidden_size;
    const int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    const int QD = c.n_heads * QHD;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    const int AOUT = c.n_heads * c.v_head_dim;
    auto bfbytes = [](size_t n)
    { return n * sizeof(bf16_t); };

    if (c.q_lora_rank > 0)
    {
        a.q_a_proj = add_chunk(a, w.q_a_proj[l], bfbytes((size_t)c.q_lora_rank * H));
        a.q_b_proj = add_chunk(a, w.q_b_proj[l], bfbytes((size_t)QD * c.q_lora_rank));
        a.q_a_norm = add_chunk(a, w.q_a_layernorm[l], bfbytes(c.q_lora_rank));
    }
    else
    {
        a.q_proj = add_chunk(a, w.q_proj[l], bfbytes((size_t)QD * H));
    }
    a.kv_a_proj = add_chunk(a, w.kv_a_proj[l], bfbytes((size_t)KVD * H));
    size_t kv_b_elems = (size_t)c.n_heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank;
    a.kv_b_proj = add_chunk(a, w.W_UK[l], bfbytes(kv_b_elems));
    a.o_proj = add_chunk(a, w.o_proj[l], bfbytes((size_t)H * AOUT));
    a.input_norm = add_chunk(a, w.input_layernorm[l], bfbytes(H));
    a.post_norm = add_chunk(a, w.post_attn_norm[l], bfbytes(H));
    a.kv_a_norm = add_chunk(a, w.kv_a_layernorm[l], bfbytes(c.kv_lora_rank));

    if (l < c.first_k_dense)
    {
        size_t ih = (size_t)c.dense_inter_size * H;
        a.dense_gate = add_chunk(a, w.dense_gate[l], bfbytes(ih));
        a.dense_up = add_chunk(a, w.dense_up[l], bfbytes(ih));
        a.dense_down = add_chunk(a, w.dense_down[l], bfbytes(ih));
    }
    else
    {
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
        for (int e = 0; e < c.n_routed_experts; ++e)
        {
            a.expert_gate[e] = add_chunk(a, w.expert_gate[l][e], bfbytes(eh));
            a.expert_up[e] = add_chunk(a, w.expert_up[l][e], bfbytes(eh));
            a.expert_down[e] = add_chunk(a, w.expert_down[l][e], bfbytes(eh));
        }
    }
    a.bytes = align_up(a.bytes, 256);
    return a;
}

template <typename T>
static void device_alloc(T **ptr, size_t count)
{
    if (count == 0)
        count = 1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)));
}

static void device_alloc_bytes(void **ptr, size_t bytes)
{
    if (bytes == 0)
        bytes = 1;
    HIP_CHECK(hipMalloc(ptr, bytes));
}

template <typename T>
static void device_free(T *&ptr)
{
    if (ptr)
        HIP_CHECK(hipFree(reinterpret_cast<void *>(ptr)));
    ptr = nullptr;
}

static const void *offset_ptr(const char *base, size_t offset)
{
    return offset == NO_OFFSET ? nullptr : base + offset;
}

static DeviceLayer materialize_layer(const LayerLayout &a, void *pool, int l)
{
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
    if (c.n_routed_experts > 0)
    {
        size_t layer_base = (size_t)l * 3 * c.n_routed_experts;
        d.expert_gate_offsets = g.expert_offsets + layer_base;
        d.expert_up_offsets = g.expert_offsets + layer_base + c.n_routed_experts;
        d.expert_down_offsets = g.expert_offsets + layer_base + 2 * c.n_routed_experts;
    }
    return d;
}

static void enqueue_layer_copy(const LayerLayout &a, void *pool)
{
    char *base = reinterpret_cast<char *>(pool);
    for (const CopyChunk &chunk : a.chunks)
        HIP_CHECK(hipMemcpyAsync(base + chunk.offset, chunk.host, chunk.bytes,
                                 hipMemcpyHostToDevice, g.stream));
}

static DeviceLayer acquire_layer(int l)
{
    return g.layers[l];
}

static void launch_gemv(float *y, const float *x, const bf16_t *w, int out, int in)
{
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    gemv_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(y, x, w, out, in);
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

static void launch_expert_gate_up_swiglu(float *y, const float *x,
                                         const DeviceLayer &d, int slot,
                                         int out, int in)
{
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    expert_gate_up_swiglu_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(
        y, x, d.pool, d.expert_gate_offsets, d.expert_up_offsets,
        g.topk, slot, out, in);
    HIP_LAUNCH_CHECK();
}

static void launch_gemv_add(float *dst, const float *x, const bf16_t *w,
                            int out, int in)
{
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    gemv_add_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(dst, x, w, out, in);
    HIP_LAUNCH_CHECK();
}

static void launch_expert_down_accum(float *dst, const float *x,
                                     const DeviceLayer &d, int slot,
                                     int out, int in)
{
    int blocks = (out + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK;
    expert_down_accum_kernel<<<blocks, HIP_BLOCK, 0, g.stream>>>(
        dst, x, d.pool, d.expert_down_offsets, g.topk, g.topk_weights,
        slot, out, in);
    HIP_LAUNCH_CHECK();
}

static void launch_rmsnorm(float *y, const float *x, const bf16_t *w, int n, float eps)
{
    rmsnorm_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(y, x, w, n, eps);
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

static void gpu_ffn(const Config &c, const DeviceLayer &d, int l)
{
    const int H = c.hidden_size;
    if (l < c.first_k_dense)
    {
        int inter = c.dense_inter_size;
        launch_gate_up_swiglu(g.hb, g.xn, d.dense_gate, d.dense_up, inter, H);
        launch_gemv(g.ffn_out, g.hb, d.dense_down, H, inter);
        return;
    }

    int E = c.n_routed_experts;
    int K = c.n_experts_per_tok;
    int inter = c.moe_inter_size;
    launch_gemv(g.router, g.xn, d.moe_gate, E, H);
    router_topk_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(
        g.router, d.moe_bias, g.topk, g.topk_weights, E, K,
        c.router_sigmoid, c.norm_topk, c.routed_scaling);
    HIP_LAUNCH_CHECK();
    zero_kernel<<<(H + HIP_BLOCK - 1) / HIP_BLOCK, HIP_BLOCK, 0, g.stream>>>(g.ffn_out, H);
    HIP_LAUNCH_CHECK();
    for (int slot = 0; slot < K; ++slot)
    {
        launch_expert_gate_up_swiglu(g.hb, g.xn, d, slot, inter, H);
        launch_expert_down_accum(g.ffn_out, g.hb, d, slot, H, inter);
    }

    int shared = c.n_shared_experts * inter;
    launch_gate_up_swiglu(g.hb, g.xn, d.shared_gate, d.shared_up, shared, H);
    launch_gemv_add(g.ffn_out, g.hb, d.shared_down, H, shared);
}

static void gpu_forward_token(int host_token, const int *device_token,
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
        DeviceLayer d = acquire_layer(l);
        launch_rmsnorm(g.xn, g.x, d.input_norm, H, c.rms_eps);
        if (c.q_lora_rank > 0)
        {
            launch_gemv(g.q_a, g.xn, d.q_a_proj, c.q_lora_rank, H);
            launch_rmsnorm(g.q_a, g.q_a, d.q_a_norm, c.q_lora_rank, c.mla_norm_eps);
            launch_gemv(g.q, g.q_a, d.q_b_proj, QD, c.q_lora_rank);
        }
        else
        {
            launch_gemv(g.q, g.xn, d.q_proj, QD, H);
        }
        rope_copy_kernel<<<NH, std::max(1, QKR / 2), 0, g.stream>>>(
            g.q, g.qrope, NH, QHD, QKN, QKR, pos, g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        launch_gemv(g.comp, g.xn, d.kv_a_proj, KVD, H);
        float *layer_cache = g.kv_cache + (size_t)l * g.kv_capacity * KVD;
        float *cache_pos = layer_cache + (size_t)pos * KVD;
        launch_rmsnorm(cache_pos, g.comp, d.kv_a_norm, KVL, c.mla_norm_eps);
        rope_copy_kernel<<<1, std::max(1, QKR / 2), 0, g.stream>>>(
            g.comp + KVL, cache_pos + KVL, 1, QKR, 0, QKR, pos,
            g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        int qabs_rows = NH * KVL;
        q_absorb_kernel<<<(qabs_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                          HIP_BLOCK, 0, g.stream>>>(
            g.qabs, g.q, d.W_UK, NH, QHD, QKN, KVL, head_stride);
        HIP_LAUNCH_CHECK();
        int score_rows = NH * kv_len;
        attention_score_kernel<<<(score_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                                 HIP_BLOCK, 0, g.stream>>>(
            g.scores, g.qabs, g.qrope, layer_cache, NH, kv_len, KVD, KVL, QKR,
            c.softmax_scale);
        HIP_LAUNCH_CHECK();
        softmax_rows_kernel<<<(NH + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                               HIP_BLOCK, 0, g.stream>>>(g.scores, NH, kv_len, kv_len);
        HIP_LAUNCH_CHECK();
        latent_context_kernel<<<(qabs_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                                HIP_BLOCK, 0, g.stream>>>(
            g.clat, g.scores, layer_cache, NH, kv_len, KVD, KVL);
        HIP_LAUNCH_CHECK();
        int value_rows = NH * VHD;
        value_up_kernel<<<(value_rows + HIP_ROWS_PER_BLOCK - 1) / HIP_ROWS_PER_BLOCK,
                          HIP_BLOCK, 0, g.stream>>>(
            g.ctx, g.clat, d.W_UV, NH, VHD, KVL, head_stride);
        HIP_LAUNCH_CHECK();
        launch_gemv(g.attn_out, g.ctx, d.o_proj, H, NH * VHD);
        add_kernel<<<(H + HIP_BLOCK - 1) / HIP_BLOCK, HIP_BLOCK, 0, g.stream>>>(g.x, g.attn_out, H);
        HIP_LAUNCH_CHECK();

        launch_rmsnorm(g.xn, g.x, d.post_norm, H, c.rms_eps);
        gpu_ffn(c, d, l);
        add_kernel<<<(H + HIP_BLOCK - 1) / HIP_BLOCK, HIP_BLOCK, 0, g.stream>>>(g.x, g.ffn_out, H);
        HIP_LAUNCH_CHECK();
    }

    if (!produce_token)
        return;
    launch_rmsnorm(g.xn, g.x, g.final_norm, H, c.rms_eps);
    launch_gemv(g.logits, g.xn, g.lm_head, c.vocab_size, H);
    argmax_kernel<<<1, HIP_BLOCK, 0, g.stream>>>(g.logits, c.vocab_size, g.next_token);
    HIP_LAUNCH_CHECK();
}

extern "C" void warm_up(Transformer *t)
{
    if (g.owner)
    {
        std::fprintf(stderr, "warm_up: HIP context is already initialized\n");
        std::exit(EXIT_FAILURE);
    }
    const Config &c = t->config;
    if (c.n_routed_experts < 1 || c.n_routed_experts > 1024 ||
        c.n_experts_per_tok < 1 || c.n_experts_per_tok > 16 ||
        c.n_experts_per_tok > c.n_routed_experts)
    {
        std::fprintf(stderr, "HIP baseline requires 1..1024 experts and valid top-k <= 16\n");
        std::exit(EXIT_FAILURE);
    }
    g.owner = t;
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (prop.warpSize != HIP_WAVE)
    {
        std::fprintf(stderr, "HIP baseline requires wavefront-64, device reports %d\n", prop.warpSize);
        std::exit(EXIT_FAILURE);
    }
    if (c.qk_rope_head_dim <= 0 || (c.qk_rope_head_dim & 1) ||
        c.qk_rope_head_dim / 2 > prop.maxThreadsPerBlock)
    {
        std::fprintf(stderr, "HIP baseline needs an even RoPE dimension supported by one block\n");
        std::exit(EXIT_FAILURE);
    }
    HIP_CHECK(hipStreamCreateWithFlags(&g.stream, hipStreamNonBlocking));

    g.layouts.reserve(c.n_layers);
    size_t layer_total = 0, max_layer = 0;
    for (int l = 0; l < c.n_layers; ++l)
    {
        g.layouts.push_back(make_layer_layout(t, l));
        layer_total += g.layouts.back().bytes;
        max_layer = std::max(max_layer, g.layouts.back().bytes);
    }
    size_t global_bytes = ((size_t)2 * c.vocab_size * c.hidden_size + c.hidden_size) * sizeof(bf16_t) + (size_t)(c.qk_rope_head_dim / 2) * sizeof(float);
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
    if (c.n_routed_experts > 0)
    {
        for (int l = c.first_k_dense; l < c.n_layers; ++l)
        {
            size_t base = (size_t)l * 3 * c.n_routed_experts;
            for (int e = 0; e < c.n_routed_experts; ++e)
            {
                host_offsets[base + e] = g.layouts[l].expert_gate[e];
                host_offsets[base + c.n_routed_experts + e] = g.layouts[l].expert_up[e];
                host_offsets[base + 2 * c.n_routed_experts + e] = g.layouts[l].expert_down[e];
            }
        }
    }
    device_alloc(&g.expert_offsets, offset_count);
    HIP_CHECK(hipMemcpy(g.expert_offsets, host_offsets.data(), offset_count * sizeof(size_t),
                        hipMemcpyHostToDevice));

    g.layers.resize(c.n_layers);
    g.layer_pools.resize(c.n_layers, nullptr);
    for (int l = 0; l < c.n_layers; ++l)
    {
        device_alloc_bytes(&g.layer_pools[l], g.layouts[l].bytes);
        enqueue_layer_copy(g.layouts[l], g.layer_pools[l]);
        g.layers[l] = materialize_layer(g.layouts[l], g.layer_pools[l], l);
    }
    
    int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    int shared = c.n_shared_experts * c.moe_inter_size;
    int inter = std::max(c.dense_inter_size, std::max(shared, c.moe_inter_size));
    device_alloc(&g.x, c.hidden_size);
    device_alloc(&g.xn, c.hidden_size);
    device_alloc(&g.attn_out, c.hidden_size);
    device_alloc(&g.ffn_out, c.hidden_size);
    device_alloc(&g.q, (size_t)c.n_heads * QHD);
    device_alloc(&g.q_a, std::max(1, c.q_lora_rank));
    device_alloc(&g.comp, KVD);
    device_alloc(&g.qrope, (size_t)c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.qabs, (size_t)c.n_heads * c.kv_lora_rank);
    device_alloc(&g.clat, (size_t)c.n_heads * c.kv_lora_rank);
    device_alloc(&g.ctx, (size_t)c.n_heads * c.v_head_dim);
    device_alloc(&g.hb, inter);
    device_alloc(&g.router, std::max(1, c.n_routed_experts));
    device_alloc(&g.topk, std::max(1, c.n_experts_per_tok));
    device_alloc(&g.topk_weights, std::max(1, c.n_experts_per_tok));
    device_alloc(&g.logits, c.vocab_size);
    device_alloc(&g.next_token, 1);
    device_alloc(&g.generated, c.max_seq_len);
    g.kv_capacity = 0;
    ensure_sequence_capacity(initial_cap);
    HIP_CHECK(hipStreamSynchronize(g.stream));

    std::fprintf(stderr, "[getp HIP] device=%s vram=%.2f GiB weights=%.2f initial_kv=%d\n",
                 prop.name, (double)total_bytes / (double)(1ULL << 30),
                 (double)(global_bytes + layer_total) / (double)(1ULL << 30),
                 g.kv_capacity);
}

extern "C" void finish(Transformer *t)
{
    (void)t;
    if (!g.owner)
        return;
    HIP_CHECK(hipStreamSynchronize(g.stream));
    device_free(g.embed);
    device_free(g.final_norm);
    device_free(g.lm_head);
    device_free(g.rope_inv_freq);
    for (void *pool : g.layer_pools)
        if (pool)
            HIP_CHECK(hipFree(pool));
    g.layer_pools.clear();
    if (g.staging_pool)
        HIP_CHECK(hipFree(g.staging_pool));
    g.staging_pool = nullptr;
    device_free(g.expert_offsets);
    device_free(g.x);
    device_free(g.xn);
    device_free(g.attn_out);
    device_free(g.ffn_out);
    device_free(g.q);
    device_free(g.q_a);
    device_free(g.comp);
    device_free(g.qrope);
    device_free(g.qabs);
    device_free(g.scores);
    device_free(g.clat);
    device_free(g.ctx);
    device_free(g.hb);
    device_free(g.router);
    device_free(g.topk);
    device_free(g.topk_weights);
    device_free(g.logits);
    device_free(g.kv_cache);
    device_free(g.next_token);
    device_free(g.generated);
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

        for (int p = 0; p < n_prompt; ++p)
            gpu_forward_token(prompts[r][p], nullptr, nullptr, 0,
                              p, p == n_prompt - 1);
        int pos = n_prompt;
        for (int step = 0; step < steps; ++step)
        {
            gpu_forward_token(0, g.next_token, g.generated, step,
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
