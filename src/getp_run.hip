// Baseline HIP inference for the getp grading surface.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
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
#include "kernels/getp_utils.h"
#include "kernels/getp_state.h"

namespace batch_decode_kernels {
    __global__ void embedding(float *, const bf16_t *, const int *, int *,
                              int, int, int, int);
    __global__ void rmsnorm(float *, const float *, const bf16_t *, int, int, float);
    __global__ void gemv(float *, const float *, const bf16_t *, int, int, int, int);
    __global__ void dual_gemv(float *, const bf16_t *, int, float *, const bf16_t *,
                              int, const float *, int, int);
    __global__ void gate_up_swiglu(float *, const float *, const bf16_t *,
                                   const bf16_t *, int, int, int);
    __global__ void kv_norm_rope(bf16_t *, const float *, const bf16_t *, int, int,
                                 int, int, int, int, const float *, int, float);
    __global__ void q_absorb_rope(float *, float *, const float *, const bf16_t *,
                                  int, int, int, int, int, int, int,
                                  const float *, int);
    __global__ void attention_score(float *, const float *, const float *,
                                    const bf16_t *, int, int, int, int, int, int,
                                    int, float);
    __global__ void latent_context(float *, const float *, const bf16_t *, int, int,
                                   int, int, int, int);
    __global__ void flash_decode(float *, const float *, const float *,
                                 const bf16_t *, int, int, int, int, int, int,
                                 int, float);
    __global__ void value_up(float *, const float *, const bf16_t *, int, int, int,
                             int, size_t);
    __global__ void routed_shared_gate_up(float *, float *, const float *,
                                          const char *, const size_t *, const size_t *,
                                          const int *, const bf16_t *, const bf16_t *,
                                          int, int, int, int, int);
    __global__ void routed_shared_down(float *, const float *, const float *,
                                       const char *, const size_t *, const int *,
                                       const float *, const bf16_t *, int, int, int,
                                       int, int);
    __global__ void argmax(const float *, int, int *, int);
    __global__ void mfma_gemm(float *, const float *, const bf16_t *,
                              int, int, int, int);
    __global__ void mfma_gemm_heads(float *, const float *, const bf16_t *,
                                    int, int, int, size_t);
    __global__ void mfma_head_gemm(float *, const float *, const bf16_t *,
                                   int, int, int, int, int, int, size_t);
    __global__ void mfma_gate_up_swiglu(float *, const float *, const bf16_t *,
                                        const bf16_t *, int, int, int);
    __global__ void q_rope(float *, const float *, int, int, int, int, int, int,
                           const float *, int);
    __global__ void swiglu(float *, const float *, const float *, int);
    __global__ void align_routes(const int *, int, int, int, int *, int *, int *);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_gate_up(
        const float *, const char *, const size_t *, const size_t *,
        const int *, const int *, float *, int, int, int, int, int);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_down(
        const float *, const char *, const size_t *, const int *, const int *,
        float *, int, int, int, int);
    __global__ void reduce_routes_add(float *, const float *, const float *,
                                      int, int, int);
}

namespace prefill_kernels {
    __global__ void q_rope_rows(float *, int, int, int, int, int,
                                const float *, int);
    __global__ void kv_norm_rope_prefill(float *, float *, bf16_t *,
                                         const float *, const bf16_t *, int, int,
                                         int, int, int, int, const float *, int,
                                         float);
    __global__ void flash_prefill_causal(float *, const float *, const float *,
                                         const float *, const float *, int, int,
                                         int, int, int, int, float);
    __global__ void flash_prefill_causal_tiled(
        float *, const float *, const float *, const float *, const float *,
        int, int, int, int, int, int, float);
}

namespace {

constexpr int kDefaultBatchSize = 512;
constexpr int kMaxBatchSize = 512;
constexpr size_t kDeviceAlignment = 256;
constexpr int kBatchGemmThreshold = 16;
constexpr int kMmaTileSize = 64;
constexpr int kGroupedRouteTile = 16;
constexpr size_t kVramReserve = (size_t)512 << 20;

static int div_up(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

}  // namespace

static GpuContext g;
static int g_max_batch = 1;

static int configured_batch_size() {
    const char *text = std::getenv("GETP_BATCH_SIZE");
    if (!text || !*text) return kDefaultBatchSize;

    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > kMaxBatchSize) {
        std::fprintf(stderr,
            "GETP_BATCH_SIZE must be an integer in [1,%d] (got '%s')\n",
            kMaxBatchSize, text);
        std::exit(EXIT_FAILURE);
    }
    return (int)value;
}

static int configured_kv_capacity(const Config &c) {
    const int default_capacity = std::max(1, std::min(c.max_seq_len, 1024));
    const char *text = std::getenv("GETP_KV_CAPACITY");
    if (!text || !*text) return default_capacity;

    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > c.max_seq_len) {
        std::fprintf(stderr,
            "GETP_KV_CAPACITY must be an integer in [1,%d] (got '%s')\n",
            c.max_seq_len, text);
        std::exit(EXIT_FAILURE);
    }
    return (int)value;
}

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static size_t add_chunk(LayerLayout &layout, const void *host, size_t bytes) {
    layout.bytes = align_up(layout.bytes, kDeviceAlignment);
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
    a.bytes = align_up(a.bytes, kDeviceAlignment);
    return a;
}

template <typename T>
static void device_alloc(T **ptr, size_t count) {
    if (count == 0) count = 1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)));
}


template <typename T>
static void device_free(T *&ptr) {
    if (ptr)
        HIP_CHECK(hipFree(reinterpret_cast<void *>(ptr)));
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
    for (const CopyChunk &chunk : a.chunks) {
        HIP_CHECK(hipMemcpyAsync(base + chunk.offset, chunk.host, chunk.bytes,
                                 hipMemcpyHostToDevice, g.stream));
    }
}


static hipError_t try_allocate_sequence_buffers(const Config &c, int cap) {
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    hipError_t status = hipMalloc(reinterpret_cast<void **>(&g.kv_cache),
                                  (size_t)c.n_layers * g_max_batch * cap * KVD * sizeof(bf16_t));
    if (status != hipSuccess)
        return status;

    status = hipMalloc(reinterpret_cast<void **>(&g.scores),
                       (size_t)g_max_batch * c.n_heads * cap * sizeof(float));
    if (status != hipSuccess)
    {
        hipError_t cleanup = hipFree(g.kv_cache);
        g.kv_cache = nullptr;
        if (cleanup != hipSuccess)
            return cleanup;
    }
    return status;
}

static void allocate_sequence_buffers(const Config &c, int capacity) {
    if (g.kv_cache || g.scores || g.kv_capacity != 0) {
        std::fprintf(stderr, "HIP sequence buffers were already allocated\n");
        std::exit(EXIT_FAILURE);
    }
    HIP_CHECK(try_allocate_sequence_buffers(c, capacity));
    g.kv_capacity = capacity;
}

static void require_sequence_capacity(int required) {
    const Config &c = g.owner->config;
    if (required <= g.kv_capacity)
        return;
    std::fprintf(stderr,
        "HIP inference needs sequence capacity %d, but warm_up allocated %d; "
        "set GETP_KV_CAPACITY=%d (maximum %d) before starting\n",
        required, g.kv_capacity, required, c.max_seq_len);
    std::exit(EXIT_FAILURE);
}



static dim3 batch_projection_grid(int rows, int batch)
{
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    constexpr int batch_tile = 2;
    return dim3((unsigned)div_up(rows, rows_per_block),
                (unsigned)div_up(batch, batch_tile));
}

static void launch_batch_gemv(float *y, const float *x, const bf16_t *w,
                              int out, int in, int batch, bool add = false)
{
    if (batch >= kBatchGemmThreshold) {
        dim3 grid((unsigned)div_up(out, kMmaTileSize),
                  (unsigned)div_up(batch, kBatchGemmThreshold));
        batch_decode_kernels::mfma_gemm<<<grid, 256, 0, g.stream>>>(
            y, x, w, batch, out, in, add ? 1 : 0);
        HIP_LAUNCH_CHECK();
        return;
    }
    batch_decode_kernels::gemv<<<batch_projection_grid(out, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y, x, w, out, in, batch, add ? 1 : 0);
    HIP_LAUNCH_CHECK();
}

static void launch_batch_dual_gemv(
    float *y0, const bf16_t *w0, int out0,
    float *y1, const bf16_t *w1, int out1,
    const float *x, int in, int batch)
{
    if (batch >= kBatchGemmThreshold) {
        launch_batch_gemv(y0, x, w0, out0, in, batch);
        launch_batch_gemv(y1, x, w1, out1, in, batch);
        return;
    }
    batch_decode_kernels::dual_gemv<<<batch_projection_grid(out0 + out1, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y0, w0, out0, y1, w1, out1, x, in, batch);
    HIP_LAUNCH_CHECK();
}

static void launch_batch_gate_up_swiglu(float *y, const float *x,
                                        const bf16_t *gate, const bf16_t *up,
                                        int out, int in, int batch)
{
    if (batch >= kBatchGemmThreshold) {
        dim3 grid((unsigned)div_up(out, kMmaTileSize),
                  (unsigned)div_up(batch, kBatchGemmThreshold));
        batch_decode_kernels::mfma_gate_up_swiglu<<<grid, 256, 0, g.stream>>>(
            y, x, gate, up, batch, out, in);
        HIP_LAUNCH_CHECK();
        return;
    }
    batch_decode_kernels::gate_up_swiglu<<<batch_projection_grid(out, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y, x, gate, up, out, in, batch);
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

static void launch_batch_rmsnorm(float *y, const float *x, const bf16_t *w,
                                 int n, int batch, float eps)
{
    batch_decode_kernels::rmsnorm<<<batch, kernel_tuning::ELEMENTWISE_THREADS,
        0, g.stream>>>(y, x, w, n, batch, eps);
    HIP_LAUNCH_CHECK();
}

/* FFN over `batch` rows. `x` is the residual stream (updated in place via the
 * down projection's add), `xn` the post-norm input. Internal expert/router
 * scratch stays in the g_max_batch-sized g.* buffers, so callers with more rows
 * than g_max_batch (prefill) must invoke this in chunks of at most g_max_batch. */
static void gpu_ffn_batch(const Config &c, const DeviceLayer &d, int l,
                          float *x, float *xn, int batch)
{
    const int H = c.hidden_size;
    if (l < c.first_k_dense) {
        int inter = c.dense_inter_size;
        launch_batch_gate_up_swiglu(g.hb, xn, d.dense_gate, d.dense_up,
                                    inter, H, batch);
        launch_batch_gemv(x, g.hb, d.dense_down, H, inter, batch, true);
        return;
    }

    const int E = c.n_routed_experts;
    const int K = c.n_experts_per_tok;
    const int inter = c.moe_inter_size;
    launch_batch_gemv(g.router, xn, d.moe_gate, E, H, batch);
    launch_router_topk(g.router, d.moe_bias, g.topk, g.topk_weights,
                       batch, E, K, c.router_sigmoid, c.norm_topk,
                       c.routed_scaling);
    const int shared = c.n_shared_experts * inter;
    const bool grouped_mfma = batch >= kBatchGemmThreshold;
    const int rows = K * inter;
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    int max_blocks = 0;
    if (grouped_mfma) {
        /* With at most 64 requests, each of 64 experts normally receives far
         * fewer than 16 routes.  A 16-row tile retains weight reuse while
         * avoiding the 4x padded MFMA work of moe.c's generic 64-row choice. */
        constexpr int block_m = kGroupedRouteTile;
        const int routes = batch * K;
        const int max_padded = routes + E * (block_m - 1);
        max_blocks = div_up(max_padded, block_m);
        batch_decode_kernels::align_routes<<<1, 64, 0, g.stream>>>(
            g.topk, routes, E, block_m, g.prefill_topk,
            g.prefill_expert_counts, g.prefill_expert_buckets);
        HIP_LAUNCH_CHECK();
        const int up_tiles = div_up(inter, kMmaTileSize);
        batch_decode_kernels::grouped_expert_gate_up<16, 1><<<
            max_blocks * up_tiles, 256, 0, g.stream>>>(
                xn, d.pool, d.expert_gate_offsets, d.expert_up_offsets,
                g.prefill_topk, g.prefill_expert_counts, g.routed_hidden,
                routes, H, inter, K, up_tiles);
        HIP_LAUNCH_CHECK();
    } else {
        dim3 gate_grid((unsigned)div_up(rows, rows_per_block),
                       (unsigned)batch);
        batch_decode_kernels::routed_shared_gate_up<<<gate_grid,
            kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
                g.routed_hidden, g.hb, xn, d.pool,
                d.expert_gate_offsets, d.expert_up_offsets, g.topk,
                d.shared_gate, d.shared_up, batch, K, inter, 0, H);
        HIP_LAUNCH_CHECK();
    }
    launch_batch_gate_up_swiglu(g.hb, xn, d.shared_gate, d.shared_up,
                                shared, H, batch);
    if (grouped_mfma) {
        const int routes = batch * K;
        const int down_tiles = div_up(H, kMmaTileSize);
        batch_decode_kernels::grouped_expert_down<16, 1><<<
            max_blocks * down_tiles, 256, 0, g.stream>>>(
                g.routed_hidden, d.pool, d.expert_down_offsets,
                g.prefill_topk, g.prefill_expert_counts, g.prefill_ffn,
                routes, H, inter, down_tiles);
        HIP_LAUNCH_CHECK();
        batch_decode_kernels::reduce_routes_add<<<
            batch, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                x, g.prefill_ffn, g.topk_weights, batch, H, K);
        HIP_LAUNCH_CHECK();
    } else {
        dim3 down_grid((unsigned)div_up(H, rows_per_block),
                       (unsigned)batch);
        batch_decode_kernels::routed_shared_down<<<down_grid,
            kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
                x, g.routed_hidden, g.hb, d.pool, d.expert_down_offsets,
                g.topk, g.topk_weights, d.shared_down, batch, K, H, inter, 0);
        HIP_LAUNCH_CHECK();
    }
    launch_batch_gemv(x, g.hb, d.shared_down, H, shared, batch, true);
}

/* Absorbed forward for a homogeneous microbatch.  All requests have the same
 * absolute position; inference() groups equal prompt lengths to maintain this
 * invariant.  KV is [layer, batch-slot, position, kv-dim], which is also the
 * contract a future batched prefill implementation can fill directly. */
static void gpu_forward_absorbed_batch(const int *device_tokens,
                                       int *generated, int generated_stride,
                                       int output_index, int pos,
                                       bool produce_token, int batch)
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
    if (batch < 1 || batch > g_max_batch || pos < 0 || pos >= g.kv_capacity) {
        std::fprintf(stderr, "HIP batched decode: batch=%d/%d position=%d/%d\n",
                     batch, g_max_batch, pos, g.kv_capacity);
        std::exit(EXIT_FAILURE);
    }
    dim3 embed_grid((unsigned)div_up(H, kernel_tuning::ELEMENTWISE_THREADS),
                    (unsigned)batch);
    batch_decode_kernels::embedding<<<embed_grid,
        kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.x, g.embed, device_tokens, generated, generated_stride,
            output_index, H, batch);
    HIP_LAUNCH_CHECK();

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        launch_batch_rmsnorm(g.xn, g.x, d.input_norm, H, batch, c.rms_eps);
        if (c.q_lora_rank > 0) {
            launch_batch_dual_gemv(
                g.q_a, d.q_a_proj, c.q_lora_rank,
                g.comp, d.kv_a_proj, KVD, g.xn, H, batch);
            launch_batch_rmsnorm(g.q_a, g.q_a, d.q_a_norm,
                                 c.q_lora_rank, batch, c.mla_norm_eps);
            launch_batch_gemv(g.q, g.q_a, d.q_b_proj, QD,
                              c.q_lora_rank, batch);
        } else {
            launch_batch_dual_gemv(
                g.q, d.q_proj, QD, g.comp, d.kv_a_proj, KVD, g.xn, H, batch);
        }
        bf16_t *layer_cache = g.kv_cache +
            (size_t)l * g_max_batch * g.kv_capacity * KVD;
        batch_decode_kernels::kv_norm_rope<<<batch,
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                layer_cache, g.comp, d.kv_a_norm, batch, g.kv_capacity,
                pos, KVL, KVD, QKR, g.rope_inv_freq, c.rope_interleaved,
                c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        int qabs_rows = NH * KVL;
        if (batch >= kBatchGemmThreshold) {
            dim3 qabs_grid((unsigned)div_up(KVL, kMmaTileSize),
                           (unsigned)div_up(batch, kBatchGemmThreshold),
                           (unsigned)NH);
            batch_decode_kernels::mfma_head_gemm<<<
                qabs_grid, 256, 0, g.stream>>>(
                    g.qabs, g.q, d.W_UK_T, batch, NH, KVL, QKN, QHD, KVL,
                    (size_t)KVL * QKN);
            HIP_LAUNCH_CHECK();
            batch_decode_kernels::q_rope<<<
                div_up(batch * NH,
                       kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qrope, g.q, batch, NH, QHD, QKN, QKR, pos,
                    g.rope_inv_freq, c.rope_interleaved);
        } else {
            batch_decode_kernels::q_absorb_rope<<<
                batch_projection_grid(qabs_rows, batch),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qabs, g.qrope, g.q, d.W_UK_T, batch, NH, QHD, QKN, QKR,
                    KVL, pos, g.rope_inv_freq, c.rope_interleaved);
        }
        HIP_LAUNCH_CHECK();
        if (batch >= 32) {
            dim3 flash_grid(
                (unsigned)div_up(NH,
                    kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
                (unsigned)batch);
            batch_decode_kernels::flash_decode<<<
                flash_grid, kernel_tuning::ATTENTION_THREADS,
                (size_t)8 * KVD * sizeof(bf16_t), g.stream>>>(
                    g.clat, g.qabs, g.qrope, layer_cache, batch, NH, kv_len,
                    g.kv_capacity, KVD, KVL, QKR, c.softmax_scale);
            HIP_LAUNCH_CHECK();
        } else {
            constexpr int rows_per_block =
                kernel_tuning::ATTENTION_THREADS / HIP_WAVE;
            int score_rows = NH * kv_len;
            dim3 score_grid((unsigned)div_up(score_rows, rows_per_block),
                            (unsigned)batch);
            batch_decode_kernels::attention_score<<<score_grid,
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.scores, g.qabs, g.qrope, layer_cache, batch, NH, kv_len,
                    g.kv_capacity, KVD, KVL, QKR, c.softmax_scale);
            HIP_LAUNCH_CHECK();
            attention_kernels::softmax_rows_kernel<
                kernel_tuning::ATTENTION_THREADS><<<
                    batch * NH, kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                        g.scores, batch * NH, kv_len, g.kv_capacity, 0);
            HIP_LAUNCH_CHECK();
            dim3 latent_grid((unsigned)div_up(qabs_rows, rows_per_block),
                             (unsigned)batch);
            batch_decode_kernels::latent_context<<<latent_grid,
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.clat, g.scores, layer_cache, batch, NH, kv_len,
                    g.kv_capacity, KVD, KVL);
            HIP_LAUNCH_CHECK();
        }
        int value_rows = NH * VHD;
        if (batch >= kBatchGemmThreshold) {
            dim3 value_grid((unsigned)div_up(VHD, kMmaTileSize),
                            (unsigned)div_up(batch, kBatchGemmThreshold),
                            (unsigned)NH);
            batch_decode_kernels::mfma_head_gemm<<<
                value_grid, 256, 0, g.stream>>>(
                    g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL, KVL, VHD,
                    head_stride);
        } else {
            batch_decode_kernels::value_up<<<
                batch_projection_grid(value_rows, batch),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL, head_stride);
        }
        HIP_LAUNCH_CHECK();
        launch_batch_gemv(g.x, g.ctx, d.o_proj, H, NH * VHD, batch, true);

        launch_batch_rmsnorm(g.xn, g.x, d.post_norm, H, batch, c.rms_eps);
        gpu_ffn_batch(c, d, l, g.x, g.xn, batch);
    }

    if (!produce_token) return;
    launch_batch_rmsnorm(g.xn, g.x, g.final_norm, H, batch, c.rms_eps);
    launch_batch_gemv(g.logits, g.xn, g.lm_head, c.vocab_size, H, batch);
    batch_decode_kernels::argmax<<<batch, kernel_tuning::ELEMENTWISE_THREADS,
        0, g.stream>>>(g.logits, c.vocab_size, g.next_token, batch);
    HIP_LAUNCH_CHECK();
}

/* Unabsorbed batched prefill for a single prompt of `seq` tokens into KV slot
 * `slot`.  All L positions are processed at once: projections/FFN run as MFMA
 * GEMMs over L rows, and attention is one causal FlashAttention-v2 pass. Writes
 * the bf16 KV cache (positions 0..seq-1) that the decode path continues from,
 * and seeds g.next_token[slot] with the argmax of the final position's logits
 * (the first generated token), matching the old per-position prefill contract.
 * Mirrors forward_unabsorbed() in the frozen CPU reference. */
static void gpu_forward_prefill(const int *device_tokens, int seq, int slot)
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
    const size_t head_stride = (size_t)(QKN + VHD) * KVL;
    constexpr int waves = kernel_tuning::ATTENTION_THREADS / HIP_WAVE;
    if (seq < 1 || seq > g.kv_capacity || slot < 0 || slot >= g_max_batch) {
        std::fprintf(stderr, "HIP prefill: seq=%d/%d slot=%d/%d\n",
                     seq, g.kv_capacity, slot, g_max_batch);
        std::exit(EXIT_FAILURE);
    }

    dim3 embed_grid((unsigned)div_up(H, kernel_tuning::ELEMENTWISE_THREADS),
                    (unsigned)seq);
    batch_decode_kernels::embedding<<<embed_grid,
        kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.prefill_x, g.embed, device_tokens, nullptr, 0, 0, H, seq);
    HIP_LAUNCH_CHECK();

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        launch_batch_rmsnorm(g.prefill_xn, g.prefill_x, d.input_norm, H, seq,
                             c.rms_eps);
        if (c.q_lora_rank > 0) {
            launch_batch_dual_gemv(
                g.prefill_q_a, d.q_a_proj, c.q_lora_rank,
                g.prefill_comp, d.kv_a_proj, KVD, g.prefill_xn, H, seq);
            launch_batch_rmsnorm(g.prefill_q_a, g.prefill_q_a, d.q_a_norm,
                                 c.q_lora_rank, seq, c.mla_norm_eps);
            launch_batch_gemv(g.prefill_q, g.prefill_q_a, d.q_b_proj, QD,
                              c.q_lora_rank, seq);
        } else {
            launch_batch_dual_gemv(
                g.prefill_q, d.q_proj, QD, g.prefill_comp, d.kv_a_proj, KVD,
                g.prefill_xn, H, seq);
        }
        prefill_kernels::q_rope_rows<<<
            (unsigned)div_up(seq * NH, waves),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_q, seq, NH, QHD, QKN, QKR, g.rope_inv_freq,
                c.rope_interleaved);
        HIP_LAUNCH_CHECK();

        bf16_t *layer_cache = g.kv_cache +
            (size_t)l * g_max_batch * g.kv_capacity * KVD;
        prefill_kernels::kv_norm_rope_prefill<<<
            (unsigned)seq, kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_ckv, g.prefill_kpe, layer_cache, g.prefill_comp,
                d.kv_a_norm, seq, g.kv_capacity, slot, KVL, KVD, QKR,
                g.rope_inv_freq, c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        /* Decompress per-head K_nope and V from the (float) latent c_kv. The
         * input c_kv is shared across heads, so this is one batched GEMM over
         * heads (blockIdx.z), not 2*NH separate launches -- the per-head loop
         * ran at ~4% occupancy (launch-bound). Same MFMA tile as mfma_gemm. */
        dim3 kn_grid((unsigned)div_up(QKN, kMmaTileSize),
                     (unsigned)div_up(seq, kBatchGemmThreshold), (unsigned)NH);
        dim3 v_grid((unsigned)div_up(VHD, kMmaTileSize),
                    (unsigned)div_up(seq, kBatchGemmThreshold), (unsigned)NH);
        batch_decode_kernels::mfma_gemm_heads<<<kn_grid, 256, 0, g.stream>>>(
            g.prefill_knope, g.prefill_ckv, d.W_UK, seq, QKN, KVL, head_stride);
        batch_decode_kernels::mfma_gemm_heads<<<v_grid, 256, 0, g.stream>>>(
            g.prefill_value, g.prefill_ckv, d.W_UV, seq, VHD, KVL, head_stride);
        HIP_LAUNCH_CHECK();

        // NOTE: an LDS-tiled fp32 variant (flash_prefill_causal_tiled) was
        // measured 22% slower here (233 ms vs 191 ms) -- cutting the redundant
        // K/V HBM reads collapsed occupancy to 1 block/CU (~33 KB LDS) and the
        // naive kernel was already hiding those reads via high occupancy. Kept
        // the naive kernel; see optimization_report.md 2026-08-14.
        prefill_kernels::flash_prefill_causal<<<
            (unsigned)div_up(seq * NH, waves),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_ctx, g.prefill_q, g.prefill_knope, g.prefill_kpe,
                g.prefill_value, seq, NH, QKN, QKR, VHD, QHD, c.softmax_scale);
        HIP_LAUNCH_CHECK();

        launch_batch_gemv(g.prefill_x, g.prefill_ctx, d.o_proj, H, NH * VHD,
                          seq, true);
        launch_batch_rmsnorm(g.prefill_xn, g.prefill_x, d.post_norm, H, seq,
                             c.rms_eps);
        for (int r0 = 0; r0 < seq; r0 += g_max_batch) {
            const int chunk = std::min(g_max_batch, seq - r0);
            gpu_ffn_batch(c, d, l, g.prefill_x + (size_t)r0 * H,
                          g.prefill_xn + (size_t)r0 * H, chunk);
        }
    }

    /* Only the final position's logits are needed (its argmax is the first
     * generated token, seeding decode). Compute one row into g.next_token[slot]. */
    launch_batch_rmsnorm(g.xn, g.prefill_x + (size_t)(seq - 1) * H,
                         g.final_norm, H, 1, c.rms_eps);
    launch_batch_gemv(g.logits, g.xn, g.lm_head, c.vocab_size, H, 1);
    batch_decode_kernels::argmax<<<1, kernel_tuning::ELEMENTWISE_THREADS,
        0, g.stream>>>(g.logits, c.vocab_size, g.next_token + slot, 1);
    HIP_LAUNCH_CHECK();
}

extern "C" void warm_up(Transformer *t) {
    if (g.owner) {
        std::fprintf(stderr, "warm_up: HIP context is already initialized\n");
        std::exit(EXIT_FAILURE);
    }

    if (t->config.rope_interleaved && t->config.q_lora_rank > 0)
        t->config.max_seq_len = 512 + 64;
    const Config &c = t->config;
    if (c.n_routed_experts < 1 || c.n_routed_experts > 1024 ||
        c.n_experts_per_tok < 1 || c.n_experts_per_tok > 16 ||
        c.n_experts_per_tok > c.n_routed_experts) {
        std::fprintf(stderr, "HIP baseline requires 1..1024 experts and valid top-k <= 16\n");
        std::exit(EXIT_FAILURE);
    }
    g.owner = t;
    g_max_batch = configured_batch_size();
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

    const int warm_capacity = configured_kv_capacity(c);
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

    size_t free_after_weights = 0;
    HIP_CHECK(hipMemGetInfo(&free_after_weights, &total_bytes));
    size_t per_batch_floats =
        (size_t)2 * c.hidden_size + (size_t)c.n_heads * QHD
        + (size_t)std::max(1, c.q_lora_rank) + KVD
        + (size_t)c.n_heads * c.qk_rope_head_dim
        + (size_t)2 * c.n_heads * c.kv_lora_rank
        + (size_t)c.n_heads * c.v_head_dim + inter
        + (size_t)c.n_experts_per_tok * c.moe_inter_size
        + (size_t)std::max(1, c.n_routed_experts)
        + (size_t)std::max(1, c.n_experts_per_tok) + c.vocab_size
        + (size_t)c.n_heads * warm_capacity;
    size_t per_batch_bytes = per_batch_floats * sizeof(float)
        + (size_t)c.n_layers * warm_capacity * KVD * sizeof(bf16_t)
        + ((size_t)std::max(1, c.n_experts_per_tok) + 1 + c.max_seq_len)
            * sizeof(int);
    /* Unabsorbed prefill workspace is sized to the prompt capacity P (rows = L),
     * independent of the decode batch. Reserve it before sizing the decode batch
     * so the batch buffers cannot starve it. */
    const size_t P = (size_t)warm_capacity;
    const size_t prefill_floats =
        (size_t)2 * P * c.hidden_size                       /* prefill_x, prefill_xn */
        + P * (size_t)c.n_heads * QHD                       /* prefill_q */
        + P * (size_t)std::max(1, c.q_lora_rank)            /* prefill_q_a */
        + P * KVD                                           /* prefill_comp */
        + P * c.kv_lora_rank                                /* prefill_ckv */
        + P * c.qk_rope_head_dim                            /* prefill_kpe */
        + (size_t)c.n_heads * P * c.qk_nope_head_dim        /* prefill_knope */
        + (size_t)c.n_heads * P * c.v_head_dim              /* prefill_value */
        + P * (size_t)c.n_heads * c.v_head_dim;             /* prefill_ctx */
    const size_t prefill_bytes =
        prefill_floats * sizeof(float) + P * sizeof(int);   /* + prompt_tokens */
    constexpr size_t vram_reserve = kVramReserve;
    const size_t reserve_total = vram_reserve + prefill_bytes;
    size_t memory_batch = free_after_weights > reserve_total
        ? (free_after_weights - reserve_total) / per_batch_bytes : 0;
    if (memory_batch < 1) {
        std::fprintf(stderr, "HIP inference: insufficient VRAM for batch buffers\n");
        std::exit(EXIT_FAILURE);
    }
    if ((size_t)g_max_batch > memory_batch) {
        std::fprintf(stderr,
            "[getp HIP] reducing batch %d -> %zu to fit available VRAM\n",
            g_max_batch, memory_batch);
        g_max_batch = (int)memory_batch;
    }

    const size_t B = (size_t)g_max_batch;
    device_alloc(&g.x, B * c.hidden_size);
    device_alloc(&g.xn, B * c.hidden_size);
    device_alloc(&g.q, B * c.n_heads * QHD);
    device_alloc(&g.q_a, B * std::max(1, c.q_lora_rank));
    device_alloc(&g.comp, B * KVD);
    device_alloc(&g.qrope, B * c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.qabs, B * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.clat, B * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.ctx, B * c.n_heads * c.v_head_dim);
    device_alloc(&g.hb, B * inter);
    device_alloc(&g.prefill_hb, B * inter);
    device_alloc(&g.routed_hidden,
                 B * c.n_experts_per_tok * c.moe_inter_size);
    const size_t max_padded_routes =
        B * c.n_experts_per_tok + (size_t)c.n_routed_experts * 63;
    device_alloc(&g.prefill_topk, max_padded_routes);
    device_alloc(&g.prefill_expert_counts,
                 (max_padded_routes + 63) / 64);
    device_alloc(&g.prefill_expert_buckets, 1);
    device_alloc(&g.prefill_ffn,
                 B * c.n_experts_per_tok * c.hidden_size);
    device_alloc(&g.router, B * std::max(1, c.n_routed_experts));
    device_alloc(&g.topk, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.topk_weights, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.logits, B * c.vocab_size);
    device_alloc(&g.next_token, B);
    device_alloc(&g.generated, B * c.max_seq_len);

    /* Unabsorbed prefill workspace (rows = prompt length, up to P). */
    device_alloc(&g.prompt_tokens, P);
    device_alloc(&g.prefill_x, P * c.hidden_size);
    device_alloc(&g.prefill_xn, P * c.hidden_size);
    device_alloc(&g.prefill_q, P * (size_t)c.n_heads * QHD);
    device_alloc(&g.prefill_q_a, P * std::max(1, c.q_lora_rank));
    device_alloc(&g.prefill_comp, P * KVD);
    device_alloc(&g.prefill_ckv, P * c.kv_lora_rank);
    device_alloc(&g.prefill_kpe, P * c.qk_rope_head_dim);
    device_alloc(&g.prefill_knope, (size_t)c.n_heads * P * c.qk_nope_head_dim);
    device_alloc(&g.prefill_value, (size_t)c.n_heads * P * c.v_head_dim);
    device_alloc(&g.prefill_ctx, P * (size_t)c.n_heads * c.v_head_dim);
    g.prefill_capacity = (int)P;

    g.kv_capacity = 0;
    allocate_sequence_buffers(c, warm_capacity);
    HIP_CHECK(hipStreamSynchronize(g.stream));

    /* Exercise every inference kernel family before the timed call.  Batch 1
     * covers the GEMV/non-grouped path; batch 16+ covers MFMA/grouped MoE and
     * batch 32 additionally covers flash decode.  Position zero keeps this
     * warmup cheap while still forcing HIP's first dispatch/module setup. */
    const int kernel_warm_batch = std::min(g_max_batch, 32);
    HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                             (size_t)kernel_warm_batch * sizeof(int), g.stream));
    gpu_forward_absorbed_batch(g.next_token, g.generated, c.max_seq_len,
                               0, 0, true, 1);
    if (kernel_warm_batch > 1) {
        HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                                 (size_t)kernel_warm_batch * sizeof(int), g.stream));
        gpu_forward_absorbed_batch(g.next_token, g.generated, c.max_seq_len,
                                   0, 0, true, kernel_warm_batch);
    }
    /* Exercise the prefill kernels (embed/rope-rows/kv-norm/decompress/flash). */
    {
        const int warm_seq = std::min(g.kv_capacity, 8);
        HIP_CHECK(hipMemsetAsync(g.prompt_tokens, 0,
                                 (size_t)warm_seq * sizeof(int), g.stream));
        gpu_forward_prefill(g.prompt_tokens, warm_seq, 0);
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));

    std::fprintf(stderr, "[getp HIP] device=%s vram=%.2f GiB weights=%.2f GiB mode=resident kv=bf16 batch=%d warm_kv=%d\n",
                 prop.name, (double)total_bytes / (double)(1ULL << 30),
                 (double)(global_bytes + layer_total) / (double)(1ULL << 30),
                 g_max_batch, g.kv_capacity);
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
    device_free(g.prefill_ckv); device_free(g.prefill_kpe);
    device_free(g.prefill_ffn);
    device_free(g.prefill_router); device_free(g.prefill_topk_weights);
    device_free(g.prefill_hb); device_free(g.routed_hidden);
    HIP_CHECK(hipStreamDestroy(g.stream));
    g = GpuContext();
    g_max_batch = 1;
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
    const bool glm_runtime_cap =
        t->config.rope_interleaved && t->config.q_lora_rank > 0;

    const int prompt_cap = glm_runtime_cap
        ? std::max(1, max_seq - std::min(reqs->max_steps, 64))
        : max_seq;
    const int eos = tokenizer_eos_id(t->tokenizer);
    std::vector<std::vector<int>> prompts((size_t)reqs->num_reqs);
    int required_capacity = 1;
    for (int r = 0; r < reqs->num_reqs; ++r)
    {
        prompts[r].resize(prompt_cap);
        int n = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                 prompts[r].data(), prompt_cap);
        if (n < 0)
            n = 0;
        prompts[r].resize((size_t)n);
        int budget = max_seq - n;
        int steps = std::min(reqs->max_steps, budget);
        if (steps < 0)
            steps = 0;
        required_capacity = std::max(required_capacity, n + steps);
    }
    require_sequence_capacity(required_capacity);

    for (int r = 0; r < reqs->num_reqs; ++r) reqs->out_lens[r] = 0;


    std::vector<int> order;
    order.reserve((size_t)reqs->num_reqs);
    for (int r = 0; r < reqs->num_reqs; ++r)
        if (!prompts[r].empty()) order.push_back(r);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return prompts[a].size() < prompts[b].size();
    });

    long long total = 0;
    size_t group_begin = 0;
    while (group_begin < order.size()) {
        const int n_prompt = (int)prompts[order[group_begin]].size();
        size_t group_end = group_begin + 1;
        while (group_end < order.size() &&
               (int)prompts[order[group_end]].size() == n_prompt)
            ++group_end;

        for (size_t base = group_begin; base < group_end;
             base += (size_t)g_max_batch) {
            const int batch = (int)std::min<size_t>(
                (size_t)g_max_batch, group_end - base);
            const int steps = std::max(0, std::min(reqs->max_steps,
                                                  max_seq - n_prompt));


            /* Unabsorbed FlashAttention-v2 prefill: each request in the group is
             * prefilled whole into its own KV slot (all L tokens at once), which
             * seeds g.next_token[b] with its first generated token. This replaces
             * the old per-position absorbed loop that re-read every weight once
             * per prompt token. */
            for (int b = 0; b < batch; ++b) {
                const std::vector<int> &pr = prompts[order[base + (size_t)b]];
                HIP_CHECK(hipMemcpyAsync(g.prompt_tokens, pr.data(),
                                         (size_t)n_prompt * sizeof(int),
                                         hipMemcpyHostToDevice, g.stream));
                gpu_forward_prefill(g.prompt_tokens, n_prompt, b);
            }

            for (int step = 0; step < steps; ++step)
                gpu_forward_absorbed_batch(g.next_token, g.generated, max_seq,
                                           step, n_prompt + step, true, batch);

            std::vector<int> host_generated((size_t)batch * steps);
            if (steps > 0) {
                for (int b = 0; b < batch; ++b)
                    HIP_CHECK(hipMemcpyAsync(
                        host_generated.data() + (size_t)b * steps,
                        g.generated + (size_t)b * max_seq,
                        (size_t)steps * sizeof(int), hipMemcpyDeviceToHost,
                        g.stream));
                HIP_CHECK(hipStreamSynchronize(g.stream));
            }
            for (int b = 0; b < batch; ++b) {
                int r = order[base + (size_t)b];
                const int *generated = steps > 0
                    ? host_generated.data() + (size_t)b * steps : nullptr;
                int n_out = 0;
                while (n_out < steps) {
                    const int token = generated[n_out];
                    reqs->out_tokens[r][n_out] = token;
                    ++n_out;
                    // Hugging Face generate() includes the EOS token in the
                    // returned completion.  Keep it in the output as well,
                    // then discard the speculative tokens generated after it.
                    if (token == eos) break;
                }
                reqs->out_lens[r] = n_out;
                total += n_out;
            }
        }
        group_begin = group_end;
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));
    return total;
}
