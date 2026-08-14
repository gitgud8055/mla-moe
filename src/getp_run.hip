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
#include <string>
#include <vector>

#include "getp.h"
#include "tokenizer.h"
#include "kernels/getp_kernels.h"
#include "kernels/getp_utils.h"
#include "kernels/getp_state.h"

using namespace std;
namespace batch_decode_kernels {
    __global__ void embedding(float *, const bf16_t *, const int *, int *,
                              int, int, int, int);
    __global__ void rmsnorm(float *, const float *, const bf16_t *, int, int, float);
    __global__ void gemv(float *, const float *, const bf16_t *, const float *,
                         int, int, int, int);
    __global__ void dual_gemv(float *, const bf16_t *, const float *, int,
                              float *, const bf16_t *, const float *, int,
                              const float *, int, int);
    __global__ void gate_up_swiglu(float *, const float *, const bf16_t *,
                                   const float *, const bf16_t *, const float *,
                                   int, int, int);
    __global__ void quantize_rows(const bf16_t *, char *, float *, int, int);
    namespace decode {
        __global__ void kv_norm_rope(
            bf16_t *, const float *, const bf16_t *, const int *, int, int,
            int, int, int, const float *, int, float);
        __global__ void q_absorb_rope(
            float *, float *, const float *, const bf16_t *, const int *, int,
            int, int, int, int, int, const float *, int);
        __global__ void attention_score(
            float *, const float *, const float *, const bf16_t *, const int *,
            int, int, int, int, int, int, int, float);
        __global__ void latent_context(
            float *, const float *, const bf16_t *, const int *, int, int, int,
            int, int, int);
        template <int KV_TILE>
        __global__ void flash_attention(
            float *, const float *, const float *, const bf16_t *, const int *,
            int, int, int, int, int, int, float);
        __global__ void q_rope(
            float *, const float *, const int *, int, int, int, int, int,
            const float *, int);
        __global__ void value_up(
            float *, const float *, const bf16_t *, int, int, int, int, size_t);
    }
    namespace prefill {
        __global__ void kv_norm_rope(
            bf16_t *, const float *, const bf16_t *, const int *, const int *,
            int, int, int, int, int, int, const float *, int, float);
        __global__ void q_rope(
            float *, const float *, const int *, int, int, int, int,
            const float *, int);
        template <int KV_TILE>
        __global__ void flash_attention(
            float *, const float *, const float *, const bf16_t *, const int *,
            const int *, int, int, int, int, int, int, int, float);
        __global__ void gather_last_hidden(
            float *, const float *, const int *, int, int, int);
    }
    __global__ void routed_shared_gate_up(float *, float *, const float *,
                                          const char *, const size_t *, const size_t *,
                                          const size_t *, const size_t *,
                                          const int *, const bf16_t *, const float *,
                                          const bf16_t *, const float *,
                                          int, int, int, int, int);
    __global__ void routed_shared_down(float *, const float *, const float *,
                                       const char *, const size_t *, const size_t *,
                                       const int *,
                                       const float *, const bf16_t *, const float *,
                                       int, int, int,
                                       int, int);
    __global__ void argmax(const float *, int, int *, int);
    __global__ void mfma_gemm(float *, const float *, const bf16_t *,
                              const float *, int, int, int, int);
    template <int M_TILES>
    __global__ void mfma_gemm_rows(float *, const float *, const bf16_t *,
                                   const float *, int, int, int, int);
    __global__ void mfma_head_gemm(float *, const float *, const bf16_t *,
                                   int, int, int, int, int, int, size_t);
    __global__ void mfma_gate_up_swiglu(float *, const float *, const bf16_t *,
                                        const float *, const bf16_t *,
                                        const float *, int, int, int);
    template <int M_TILES>
    __global__ void mfma_gate_up_swiglu_rows(
        float *, const float *, const bf16_t *, const float *, const bf16_t *,
        const float *, int, int, int);
    __global__ void swiglu(float *, const float *, const float *, int);
    __global__ void align_routes(const int *, int, int, int, int *, int *, int *);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_gate_up(
        const float *, const char *, const size_t *, const size_t *,
        const size_t *, const size_t *,
        const int *, const int *, float *, bf16_t *, int, int, int, int, int);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_down(
        const float *, const bf16_t *, const char *, const size_t *,
        const size_t *,
        const int *, const int *, float *, int, int, int, int);
    __global__ void quantize_act_rows_f32(const float *, char *, float *,
                                          int, int);
    __global__ void quantize_act_rows_bf16(const bf16_t *, char *, float *,
                                           int, int);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_gate_up_i8(
        const char *, const float *, const char *, const size_t *,
        const size_t *, const size_t *, const size_t *, const int *,
        const int *, float *, bf16_t *, int, int, int, int, int);
    template <int BLOCK_M, int M_TILES>
    __global__ void grouped_expert_down_i8(
        const char *, const float *, const char *, const size_t *,
        const size_t *, const int *, const int *, float *, int, int, int,
        int);
    template <int M_TILES>
    __global__ void mfma_gemm_i8(float *, const char *, const float *,
                                 const char *, const float *, int, int, int,
                                 int);
    template <int M_TILES>
    __global__ void mfma_gate_up_swiglu_i8(
        float *, const char *, const float *, const char *, const float *,
        const char *, const float *, int, int, int);
    __global__ void reduce_routes_add(float *, const float *, const float *,
                                      int, int, int);
}

namespace {

constexpr int kDefaultBatchSize = 512;
constexpr int kMaxBatchSize = 512;
constexpr int kMaxPrefillRows = 65520;
constexpr int kDefaultPrefillBatchTokens = kMaxPrefillRows;
constexpr size_t kDeviceAlignment = 256;
constexpr int kDefaultMfmaThreshold = 4;
constexpr int kDefaultFlashThreshold = 8;
constexpr int kMmaBatchTile = 16;
constexpr int kMmaTileSize = 64;
constexpr int kPrefillMmaRowTiles = 2;
constexpr int kPrefillMmaRows = kMmaBatchTile * kPrefillMmaRowTiles;
constexpr int kPrefillMmaThreshold = 1024;
constexpr int kGroupedRouteTile = 16;
constexpr int kFlashLargeBatchThreshold = 128;
constexpr int kFlashSmallBatchThreads = 512;
constexpr int kFlashSmallBatchKvTile = 16;
constexpr int kFlashLargeBatchThreads = 256;
constexpr int kFlashLargeBatchKvTile = 8;
constexpr int kPrefillThreads = 256;
constexpr int kPrefillKvTile = 8;
constexpr size_t kVramReserve = (size_t)512 << 20;

static int div_up(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

}  // namespace

static GpuContext g;

/* Weight-only int8 (per-output-row scales), selected per tensor group via
 * GETP_QUANT (comma list of: experts, shared, proj, lmhead; or "all"). */
struct QuantGroups {
    bool experts = false;
    bool shared_experts = false;
    bool proj = false;
    bool lmhead = false;
    /* int8 MFMA compute for the grouped expert path (implies experts). */
    bool w8a8 = false;
    bool any() const { return experts || shared_experts || proj || lmhead; }
};
static QuantGroups g_quant;

static QuantGroups configured_quant_groups() {
    QuantGroups q;
    const char *text = std::getenv("GETP_QUANT");
    if (!text || !*text) {
        text = "all,w8a8";
    }
    std::string s(text);
    size_t begin = 0;
    while (begin <= s.size()) {
        size_t end = s.find(',', begin);
        if (end == std::string::npos) end = s.size();
        std::string token = s.substr(begin, end - begin);
        if (token == "experts") q.experts = true;
        else if (token == "shared") q.shared_experts = true;
        else if (token == "proj") q.proj = true;
        else if (token == "lmhead") q.lmhead = true;
        else if (token == "w8a8") q.w8a8 = q.experts = true;
        else if (token == "all") q.experts = q.shared_experts = q.proj = q.lmhead = true;
        else if (token == "none" || token.empty()) { /* no-op */ }
        else {
            std::fprintf(stderr,
                "GETP_QUANT: unknown group '%s' (use experts,shared,proj,lmhead,all)\n",
                token.c_str());
            std::exit(EXIT_FAILURE);
        }
        begin = end + 1;
    }
    return q;
}

static int g_max_batch = 1;
static int g_prefill_token_budget = kDefaultPrefillBatchTokens;
static int g_decode_token_budget = kDefaultBatchSize;
static int g_mfma_threshold = kDefaultMfmaThreshold;
static int g_flash_threshold = kDefaultFlashThreshold;
static bool g_prefill_large_m = true;

static int configured_threshold(const char *name, int default_value) {
    const char *text = std::getenv(name);
    if (!text || !*text) return default_value;
    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > kMaxBatchSize + 1) {
        std::fprintf(stderr,
            "%s must be an integer in [1,%d] (got '%s')\n",
            name, kMaxBatchSize + 1, text);
        std::exit(EXIT_FAILURE);
    }
    return (int)value;
}

static int configured_positive_int(const char *name, int default_value,
                                   int maximum) {
    const char *text = std::getenv(name);
    if (!text || !*text) return default_value;

    char *end = nullptr;
    long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > maximum) {
        std::fprintf(stderr,
            "%s must be an integer in [1,%d] (got '%s')\n",
            name, maximum, text);
        std::exit(EXIT_FAILURE);
    }
    return (int)value;
}

static int configured_decode_token_budget() {
    if (std::getenv("GETP_DECODE_BATCH_TOKENS"))
        return configured_positive_int(
            "GETP_DECODE_BATCH_TOKENS", kDefaultBatchSize, kMaxBatchSize);
    /* Compatibility with the old knob: one decode step consumes exactly one
     * token for every active sequence. */
    return configured_positive_int(
        "GETP_BATCH_SIZE", kDefaultBatchSize, kMaxBatchSize);
}

static int configured_prefill_token_budget() {
    return configured_positive_int(
        "GETP_PREFILL_BATCH_TOKENS", kDefaultPrefillBatchTokens,
        kMaxPrefillRows);
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

/* Row-major [rows x cols] weight: plain bf16 copy, or (when `quant`) an int8
 * data block plus per-row fp32 scales written by quantize_rows at load time. */
static size_t add_weight(LayerLayout &layout, const void *host, size_t rows,
                         size_t cols, bool quant, size_t *scale_offset) {
    if (!quant)
        return add_chunk(layout, host, rows * cols * sizeof(bf16_t));
    layout.bytes = align_up(layout.bytes, kDeviceAlignment);
    const size_t data_offset = layout.bytes;
    layout.bytes += rows * cols;
    layout.bytes = align_up(layout.bytes, kDeviceAlignment);
    const size_t s_offset = layout.bytes;
    layout.bytes += rows * sizeof(float);
    layout.qchunks.push_back({host, data_offset, s_offset,
                              (int)rows, (int)cols});
    if (scale_offset) *scale_offset = s_offset;
    return data_offset;
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
        a.q_a_proj = add_weight(a, w.q_a_proj[l], c.q_lora_rank, H,
                                g_quant.proj, &a.q_a_proj_s);
        a.q_b_proj = add_weight(a, w.q_b_proj[l], QD, c.q_lora_rank,
                                g_quant.proj, &a.q_b_proj_s);
        a.q_a_norm = add_chunk(a, w.q_a_layernorm[l], bfbytes(c.q_lora_rank));
    } else {
        a.q_proj = add_weight(a, w.q_proj[l], QD, H, g_quant.proj, &a.q_proj_s);
    }
    a.kv_a_proj = add_weight(a, w.kv_a_proj[l], KVD, H, g_quant.proj,
                             &a.kv_a_proj_s);
    size_t kv_b_elems = (size_t)c.n_heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank;
    a.kv_b_proj = add_chunk(a, w.W_UK[l], bfbytes(kv_b_elems));
    size_t wuk_elems = (size_t)c.n_heads * c.kv_lora_rank * c.qk_nope_head_dim;
    a.wuk_t = add_chunk(a, wuk_t, bfbytes(wuk_elems));
    a.o_proj = add_weight(a, w.o_proj[l], H, AOUT, g_quant.proj, &a.o_proj_s);
    a.input_norm = add_chunk(a, w.input_layernorm[l], bfbytes(H));
    a.post_norm = add_chunk(a, w.post_attn_norm[l], bfbytes(H));
    a.kv_a_norm = add_chunk(a, w.kv_a_layernorm[l], bfbytes(c.kv_lora_rank));

    if (l < c.first_k_dense) {
        a.dense_gate = add_weight(a, w.dense_gate[l], c.dense_inter_size, H,
                                  g_quant.proj, &a.dense_gate_s);
        a.dense_up = add_weight(a, w.dense_up[l], c.dense_inter_size, H,
                                g_quant.proj, &a.dense_up_s);
        a.dense_down = add_weight(a, w.dense_down[l], H, c.dense_inter_size,
                                  g_quant.proj, &a.dense_down_s);
    } else {
        a.moe_gate = add_chunk(a, w.moe_gate[l], bfbytes((size_t)c.n_routed_experts * H));
        if (w.moe_gate_bias[l])
            a.moe_bias = add_chunk(a, w.moe_gate_bias[l], (size_t)c.n_routed_experts * sizeof(float));
        int shared = c.n_shared_experts * c.moe_inter_size;
        a.shared_gate = add_weight(a, w.shared_gate[l], shared, H,
                                   g_quant.shared_experts, &a.shared_gate_s);
        a.shared_up = add_weight(a, w.shared_up[l], shared, H,
                                 g_quant.shared_experts, &a.shared_up_s);
        a.shared_down = add_weight(a, w.shared_down[l], H, shared,
                                   g_quant.shared_experts, &a.shared_down_s);
        a.expert_gate.resize(c.n_routed_experts);
        a.expert_up.resize(c.n_routed_experts);
        a.expert_down.resize(c.n_routed_experts);
        a.expert_gate_s.resize(c.n_routed_experts, NO_OFFSET);
        a.expert_up_s.resize(c.n_routed_experts, NO_OFFSET);
        a.expert_down_s.resize(c.n_routed_experts, NO_OFFSET);
        for (int e = 0; e < c.n_routed_experts; ++e) {
            a.expert_gate[e] = add_weight(a, w.expert_gate[l][e],
                                          c.moe_inter_size, H,
                                          g_quant.experts, &a.expert_gate_s[e]);
            a.expert_up[e] = add_weight(a, w.expert_up[l][e],
                                        c.moe_inter_size, H,
                                        g_quant.experts, &a.expert_up_s[e]);
            a.expert_down[e] = add_weight(a, w.expert_down[l][e],
                                          H, c.moe_inter_size,
                                          g_quant.experts, &a.expert_down_s[e]);
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
        if (g_quant.experts && g.expert_scale_offsets) {
            d.expert_gate_scale_offsets = g.expert_scale_offsets + layer_base;
            d.expert_up_scale_offsets =
                g.expert_scale_offsets + layer_base + c.n_routed_experts;
            d.expert_down_scale_offsets =
                g.expert_scale_offsets + layer_base + 2 * c.n_routed_experts;
        }
    }
    auto scale_ptr = [&](size_t offset) {
        return reinterpret_cast<const float *>(offset_ptr(base, offset));
    };
    d.q_proj_s = scale_ptr(a.q_proj_s);
    d.q_a_proj_s = scale_ptr(a.q_a_proj_s);
    d.q_b_proj_s = scale_ptr(a.q_b_proj_s);
    d.kv_a_proj_s = scale_ptr(a.kv_a_proj_s);
    d.o_proj_s = scale_ptr(a.o_proj_s);
    d.dense_gate_s = scale_ptr(a.dense_gate_s);
    d.dense_up_s = scale_ptr(a.dense_up_s);
    d.dense_down_s = scale_ptr(a.dense_down_s);
    d.shared_gate_s = scale_ptr(a.shared_gate_s);
    d.shared_up_s = scale_ptr(a.shared_up_s);
    d.shared_down_s = scale_ptr(a.shared_down_s);
    return d;
}

static void enqueue_layer_copy(const LayerLayout &a, void *pool,
                               bf16_t *quant_stage) {
    char *base = reinterpret_cast<char *>(pool);
    for (const CopyChunk &chunk : a.chunks) {
        HIP_CHECK(hipMemcpyAsync(base + chunk.offset, chunk.host, chunk.bytes,
                                 hipMemcpyHostToDevice, g.stream));
    }
    /* Stream order serializes the staging copy against the previous chunk's
     * quantize_rows launch, so one staging buffer is enough. */
    for (const QuantChunk &chunk : a.qchunks) {
        HIP_CHECK(hipMemcpyAsync(
            quant_stage, chunk.host,
            (size_t)chunk.rows * chunk.cols * sizeof(bf16_t),
            hipMemcpyHostToDevice, g.stream));
        batch_decode_kernels::quantize_rows<<<
            chunk.rows, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                quant_stage, base + chunk.data_offset,
                reinterpret_cast<float *>(base + chunk.scale_offset),
                chunk.rows, chunk.cols);
        HIP_LAUNCH_CHECK();
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

static size_t prefill_workspace_bytes(const Config &c, size_t rows) {
    const size_t qhd = (size_t)c.qk_nope_head_dim + c.qk_rope_head_dim;
    const size_t kvd = (size_t)c.kv_lora_rank + c.qk_rope_head_dim;
    const size_t shared = (size_t)c.n_shared_experts * c.moe_inter_size;
    const size_t inter = std::max<size_t>(
        c.dense_inter_size, std::max<size_t>(shared, c.moe_inter_size));
    const size_t k = std::max(1, c.n_experts_per_tok);
    size_t floats_per_row =
        (size_t)2 * c.hidden_size
        + (size_t)c.n_heads * qhd
        + (size_t)std::max(1, c.q_lora_rank)
        + kvd
        + (size_t)c.n_heads * c.qk_rope_head_dim
        + (size_t)2 * c.n_heads * c.kv_lora_rank
        + (size_t)c.n_heads * c.v_head_dim
        + inter
        + (size_t)std::max(1, c.n_routed_experts)
        + k
        + k * c.hidden_size;
    const size_t routes = rows * k;
    const size_t max_padded = routes
        + (size_t)c.n_routed_experts * (kGroupedRouteTile - 1);
    const size_t w8a8_act_cols = std::max(
        inter, std::max((size_t)c.hidden_size,
                        (size_t)c.n_heads * c.v_head_dim));
    const size_t w8a8_bytes = g_quant.w8a8
        ? rows * w8a8_act_cols + routes * (size_t)c.moe_inter_size
          + (rows + routes) * sizeof(float)
        : 0;
    return rows * floats_per_row * sizeof(float)
        + rows * k * c.moe_inter_size * sizeof(bf16_t)
        + w8a8_bytes
        + rows * sizeof(int)                 // flattened prompt tokens
        + rows * 2 * sizeof(int)             // packed row batch + position
        + ((size_t)g_max_batch + 1) * sizeof(int) // packed prompt offsets
        + rows * k * sizeof(int)             // top-k expert ids
        + max_padded * sizeof(int)            // aligned route ids
        + (max_padded + kGroupedRouteTile - 1) / kGroupedRouteTile
            * sizeof(int)
        + sizeof(int);
}

static void allocate_prefill_workspace(const Config &c, int target_rows) {
    size_t free_bytes = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    (void)total_bytes;
    const size_t usable = free_bytes > kVramReserve
        ? free_bytes - kVramReserve : 0;
    int lo = 0, hi = std::max(1, target_rows);
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (prefill_workspace_bytes(c, (size_t)mid) <= usable)
            lo = mid;
        else
            hi = mid - 1;
    }
    if (lo < 1) {
        std::fprintf(stderr,
            "HIP inference: insufficient VRAM for batched prefill workspace\n");
        std::exit(EXIT_FAILURE);
    }
    const size_t rows = (size_t)lo;
    const size_t qhd = (size_t)c.qk_nope_head_dim + c.qk_rope_head_dim;
    const size_t kvd = (size_t)c.kv_lora_rank + c.qk_rope_head_dim;
    const size_t shared = (size_t)c.n_shared_experts * c.moe_inter_size;
    const size_t inter = std::max<size_t>(
        c.dense_inter_size, std::max<size_t>(shared, c.moe_inter_size));
    const size_t k = std::max(1, c.n_experts_per_tok);
    const size_t routes = rows * k;
    const size_t max_padded = routes
        + (size_t)c.n_routed_experts * (kGroupedRouteTile - 1);

    device_alloc(&g.prompt_tokens, rows);
    device_alloc(&g.prefill_row_batches, rows);
    device_alloc(&g.prefill_row_positions, rows);
    device_alloc(&g.prefill_offsets, (size_t)g_max_batch + 1);
    device_alloc(&g.prefill_x, rows * c.hidden_size);
    device_alloc(&g.prefill_xn, rows * c.hidden_size);
    device_alloc(&g.prefill_q, rows * c.n_heads * qhd);
    device_alloc(&g.prefill_q_a, rows * std::max(1, c.q_lora_rank));
    device_alloc(&g.prefill_comp, rows * kvd);
    device_alloc(&g.prefill_qrope,
                 rows * c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.prefill_qabs, rows * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.prefill_clat, rows * c.n_heads * c.kv_lora_rank);
    device_alloc(&g.prefill_ctx, rows * c.n_heads * c.v_head_dim);
    device_alloc(&g.prefill_hb, rows * inter);
    device_alloc(&g.prefill_router,
                 rows * std::max(1, c.n_routed_experts));
    device_alloc(&g.prefill_route_ids, rows * k);
    device_alloc(&g.prefill_topk_weights, rows * k);
    device_alloc(&g.prefill_topk, max_padded);
    device_alloc(&g.prefill_expert_counts,
                 div_up((int)max_padded, kGroupedRouteTile));
    device_alloc(&g.prefill_expert_buckets, 1);
    device_alloc(&g.prefill_routed_hidden, routes * c.moe_inter_size);
    device_alloc(&g.prefill_ffn, routes * c.hidden_size);
    if (g_quant.w8a8) {
        const size_t act_cols = std::max(
            inter,
            std::max((size_t)c.hidden_size,
                     (size_t)c.n_heads * c.v_head_dim));
        device_alloc(&g.prefill_act_i8, rows * act_cols);
        device_alloc(&g.prefill_act_s, rows);
        device_alloc(&g.prefill_routed_i8, routes * c.moe_inter_size);
        device_alloc(&g.prefill_routed_s, routes);
    }
    g.prefill_capacity = lo;
}



static dim3 batch_projection_grid(int rows, int batch)
{
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    constexpr int batch_tile = 2;
    return dim3((unsigned)div_up(rows, rows_per_block),
                (unsigned)div_up(batch, batch_tile));
}

static void launch_batch_gemv(float *y, const float *x, const bf16_t *w,
                              const float *ws, int out, int in, int batch,
                              bool add = false)
{
    if (batch >= g_mfma_threshold) {
        const bool large_m =
            g_prefill_large_m && batch >= kPrefillMmaThreshold;
        dim3 grid((unsigned)div_up(out, kMmaTileSize),
                  (unsigned)div_up(
                      batch, large_m ? kPrefillMmaRows : kMmaBatchTile));
        char *xq = batch <= g_max_batch ? g.act_i8 : g.prefill_act_i8;
        float *xqs = batch <= g_max_batch ? g.act_s : g.prefill_act_s;
        if (g_quant.w8a8 && ws && xq) {
            batch_decode_kernels::quantize_act_rows_f32<<<
                batch, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                    x, xq, xqs, batch, in);
            HIP_LAUNCH_CHECK();
            if (large_m)
                batch_decode_kernels::mfma_gemm_i8<kPrefillMmaRowTiles><<<
                    grid, 256, 0, g.stream>>>(
                        y, xq, xqs, reinterpret_cast<const char *>(w), ws,
                        batch, out, in, add ? 1 : 0);
            else
                batch_decode_kernels::mfma_gemm_i8<1><<<
                    grid, 256, 0, g.stream>>>(
                        y, xq, xqs, reinterpret_cast<const char *>(w), ws,
                        batch, out, in, add ? 1 : 0);
            HIP_LAUNCH_CHECK();
            return;
        }
        if (large_m)
            batch_decode_kernels::mfma_gemm_rows<kPrefillMmaRowTiles><<<
                grid, 256, 0, g.stream>>>(
                    y, x, w, ws, batch, out, in, add ? 1 : 0);
        else
            batch_decode_kernels::mfma_gemm<<<grid, 256, 0, g.stream>>>(
                y, x, w, ws, batch, out, in, add ? 1 : 0);
        HIP_LAUNCH_CHECK();
        return;
    }
    batch_decode_kernels::gemv<<<batch_projection_grid(out, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y, x, w, ws, out, in, batch, add ? 1 : 0);
    HIP_LAUNCH_CHECK();
}

static void launch_batch_dual_gemv(
    float *y0, const bf16_t *w0, const float *ws0, int out0,
    float *y1, const bf16_t *w1, const float *ws1, int out1,
    const float *x, int in, int batch)
{
    if (batch >= g_mfma_threshold) {
        launch_batch_gemv(y0, x, w0, ws0, out0, in, batch);
        launch_batch_gemv(y1, x, w1, ws1, out1, in, batch);
        return;
    }
    batch_decode_kernels::dual_gemv<<<batch_projection_grid(out0 + out1, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y0, w0, ws0, out0, y1, w1, ws1, out1, x, in, batch);
    HIP_LAUNCH_CHECK();
}

static void launch_batch_gate_up_swiglu(float *y, const float *x,
                                        const bf16_t *gate, const float *gws,
                                        const bf16_t *up, const float *uws,
                                        int out, int in, int batch)
{
    if (batch >= g_mfma_threshold) {
        const bool large_m =
            g_prefill_large_m && batch >= kPrefillMmaThreshold;
        dim3 grid((unsigned)div_up(out, kMmaTileSize),
                  (unsigned)div_up(
                      batch, large_m ? kPrefillMmaRows : kMmaBatchTile));
        char *xq = batch <= g_max_batch ? g.act_i8 : g.prefill_act_i8;
        float *xqs = batch <= g_max_batch ? g.act_s : g.prefill_act_s;
        if (g_quant.w8a8 && gws && uws && xq) {
            batch_decode_kernels::quantize_act_rows_f32<<<
                batch, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                    x, xq, xqs, batch, in);
            HIP_LAUNCH_CHECK();
            if (large_m)
                batch_decode_kernels::mfma_gate_up_swiglu_i8<
                    kPrefillMmaRowTiles><<<grid, 256, 0, g.stream>>>(
                        y, xq, xqs, reinterpret_cast<const char *>(gate), gws,
                        reinterpret_cast<const char *>(up), uws,
                        batch, out, in);
            else
                batch_decode_kernels::mfma_gate_up_swiglu_i8<1><<<
                    grid, 256, 0, g.stream>>>(
                        y, xq, xqs, reinterpret_cast<const char *>(gate), gws,
                        reinterpret_cast<const char *>(up), uws,
                        batch, out, in);
            HIP_LAUNCH_CHECK();
            return;
        }
        if (large_m)
            batch_decode_kernels::mfma_gate_up_swiglu_rows<
                kPrefillMmaRowTiles><<<grid, 256, 0, g.stream>>>(
                    y, x, gate, gws, up, uws, batch, out, in);
        else
            batch_decode_kernels::mfma_gate_up_swiglu<<<
                grid, 256, 0, g.stream>>>(
                    y, x, gate, gws, up, uws, batch, out, in);
        HIP_LAUNCH_CHECK();
        return;
    }
    batch_decode_kernels::gate_up_swiglu<<<batch_projection_grid(out, batch),
        kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
            y, x, gate, gws, up, uws, out, in, batch);
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

static void gpu_ffn_rows(
    const Config &c, const DeviceLayer &d, int l, int rows,
    float *x, float *xn, float *hb, float *router, int *topk,
    float *topk_weights, int *sorted_ids, int *expert_ids,
    int *num_post_pad, float *routed_hidden,
    bf16_t *routed_hidden_bf16, float *route_out,
    char *act_i8, float *act_s, char *routed_i8, float *routed_s)
{
    const int H = c.hidden_size;
    if (l < c.first_k_dense) {
        int inter = c.dense_inter_size;
        launch_batch_gate_up_swiglu(hb, xn, d.dense_gate, d.dense_gate_s,
                                    d.dense_up, d.dense_up_s,
                                    inter, H, rows);
        launch_batch_gemv(x, hb, d.dense_down, d.dense_down_s, H, inter,
                          rows, true);
        return;
    }

    const int E = c.n_routed_experts;
    const int K = c.n_experts_per_tok;
    const int inter = c.moe_inter_size;
    launch_batch_gemv(router, xn, d.moe_gate, nullptr, E, H, rows);
    launch_router_topk(router, d.moe_bias, topk, topk_weights,
                       rows, E, K, c.router_sigmoid, c.norm_topk,
                       c.routed_scaling);
    const int shared = c.n_shared_experts * inter;
    const bool grouped_mfma = rows >= g_mfma_threshold;
    const int output_rows = K * inter;
    constexpr int rows_per_block = kernel_tuning::GEMV_THREADS / HIP_WAVE;
    int max_blocks = 0;
    if (grouped_mfma) {
        constexpr int block_m = kGroupedRouteTile;
        const int routes = rows * K;
        const int max_padded = routes + E * (block_m - 1);
        max_blocks = div_up(max_padded, block_m);
        batch_decode_kernels::align_routes<<<
            1, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            topk, routes, E, block_m, sorted_ids, expert_ids,
            num_post_pad);
        HIP_LAUNCH_CHECK();
        const int up_tiles = div_up(inter, kMmaTileSize);
        const bool w8a8 = g_quant.w8a8 && act_i8 != nullptr;
        if (w8a8) {
            batch_decode_kernels::quantize_act_rows_f32<<<
                rows, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                    xn, act_i8, act_s, rows, H);
            HIP_LAUNCH_CHECK();
            batch_decode_kernels::grouped_expert_gate_up_i8<16, 1><<<
                max_blocks * up_tiles, 256, 0, g.stream>>>(
                    act_i8, act_s, d.pool,
                    d.expert_gate_offsets, d.expert_up_offsets,
                    d.expert_gate_scale_offsets, d.expert_up_scale_offsets,
                    sorted_ids, expert_ids, routed_hidden, routed_hidden_bf16,
                    routes, H, inter, K, up_tiles);
        } else {
            batch_decode_kernels::grouped_expert_gate_up<16, 1><<<
                max_blocks * up_tiles, 256, 0, g.stream>>>(
                    xn, d.pool, d.expert_gate_offsets, d.expert_up_offsets,
                    d.expert_gate_scale_offsets, d.expert_up_scale_offsets,
                    sorted_ids, expert_ids, routed_hidden, routed_hidden_bf16,
                    routes, H, inter, K, up_tiles);
        }
        HIP_LAUNCH_CHECK();
    } else {
        dim3 gate_grid((unsigned)div_up(output_rows, rows_per_block),
                       (unsigned)rows);
        batch_decode_kernels::routed_shared_gate_up<<<gate_grid,
            kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
                routed_hidden, hb, xn, d.pool,
                d.expert_gate_offsets, d.expert_up_offsets,
                d.expert_gate_scale_offsets, d.expert_up_scale_offsets, topk,
                d.shared_gate, d.shared_gate_s, d.shared_up, d.shared_up_s,
                rows, K, inter, 0, H);
        HIP_LAUNCH_CHECK();
    }
    launch_batch_gate_up_swiglu(hb, xn, d.shared_gate, d.shared_gate_s,
                                d.shared_up, d.shared_up_s,
                                shared, H, rows);
    if (grouped_mfma) {
        const int routes = rows * K;
        const int down_tiles = div_up(H, kMmaTileSize);
        const bool w8a8 = g_quant.w8a8 && routed_i8 != nullptr;
        if (w8a8) {
            if (routed_hidden_bf16)
                batch_decode_kernels::quantize_act_rows_bf16<<<
                    routes, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                        routed_hidden_bf16, routed_i8, routed_s, routes, inter);
            else
                batch_decode_kernels::quantize_act_rows_f32<<<
                    routes, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                        routed_hidden, routed_i8, routed_s, routes, inter);
            HIP_LAUNCH_CHECK();
            batch_decode_kernels::grouped_expert_down_i8<16, 1><<<
                max_blocks * down_tiles, 256, 0, g.stream>>>(
                    routed_i8, routed_s, d.pool,
                    d.expert_down_offsets, d.expert_down_scale_offsets,
                    sorted_ids, expert_ids, route_out,
                    routes, H, inter, down_tiles);
        } else {
            batch_decode_kernels::grouped_expert_down<16, 1><<<
                max_blocks * down_tiles, 256, 0, g.stream>>>(
                    routed_hidden, routed_hidden_bf16, d.pool,
                    d.expert_down_offsets, d.expert_down_scale_offsets,
                    sorted_ids, expert_ids, route_out,
                    routes, H, inter, down_tiles);
        }
        HIP_LAUNCH_CHECK();
        batch_decode_kernels::reduce_routes_add<<<
            rows, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                x, route_out, topk_weights, rows, H, K);
        HIP_LAUNCH_CHECK();
    } else {
        dim3 down_grid((unsigned)div_up(H, rows_per_block),
                       (unsigned)rows);
        batch_decode_kernels::routed_shared_down<<<down_grid,
            kernel_tuning::GEMV_THREADS, 0, g.stream>>>(
                x, routed_hidden, hb, d.pool, d.expert_down_offsets,
                d.expert_down_scale_offsets,
                topk, topk_weights, d.shared_down, d.shared_down_s,
                rows, K, H, inter, 0);
        HIP_LAUNCH_CHECK();
    }
    launch_batch_gemv(x, hb, d.shared_down, d.shared_down_s, H, shared,
                      rows, true);
}

static void gpu_ffn_batch(const Config &c, const DeviceLayer &d, int l,
                          int batch)
{
    gpu_ffn_rows(c, d, l, batch, g.x, g.xn, g.hb, g.router, g.topk,
                 g.topk_weights, g.prefill_topk, g.prefill_expert_counts,
                 g.prefill_expert_buckets, g.routed_hidden, nullptr,
                 g.prefill_ffn,
                 g.act_i8, g.act_s, g.routed_i8, g.routed_s);
}

namespace prefill {

/* Process packed prompts layer-major.  `rows` is the number of real input
 * tokens, not batch*max_length; row metadata gives each token's sequence and
 * position.  This keeps request order and avoids both padding and length
 * sorting while all dense/MoE operations see one token batch. */
static void run_chunk(const int *device_tokens, const int *row_batches,
                      const int *row_positions, const int *offsets,
                      int rows, int batch, int slot_base)
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
    if (batch < 1 || slot_base < 0 || slot_base + batch > g_max_batch ||
        rows < batch || rows > g.prefill_capacity) {
        std::fprintf(stderr,
            "HIP packed prefill: slots=%d+%d/%d tokens=%d/%d\n",
            slot_base, batch, g_max_batch, rows, g.prefill_capacity);
        std::exit(EXIT_FAILURE);
    }

    dim3 embed_grid((unsigned)div_up(H, kernel_tuning::ELEMENTWISE_THREADS),
                    (unsigned)rows);
    batch_decode_kernels::embedding<<<embed_grid,
        kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.prefill_x, g.embed, device_tokens, nullptr, 0, 0, H, rows);
    HIP_LAUNCH_CHECK();

    const bool profile = std::getenv("GETP_PROFILE_PREFILL") != nullptr;
    hipEvent_t phase_start = nullptr, phase_stop = nullptr;
    float qkv_ms = 0.0f, flash_ms = 0.0f, output_ms = 0.0f, ffn_ms = 0.0f;
    if (profile) {
        HIP_CHECK(hipEventCreate(&phase_start));
        HIP_CHECK(hipEventCreate(&phase_stop));
    }
    auto phase_begin = [&]() {
        if (profile) HIP_CHECK(hipEventRecord(phase_start, g.stream));
    };
    auto phase_end = [&](float &total_ms) {
        if (!profile) return;
        HIP_CHECK(hipEventRecord(phase_stop, g.stream));
        HIP_CHECK(hipEventSynchronize(phase_stop));
        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, phase_start, phase_stop));
        total_ms += elapsed_ms;
    };

    for (int l = 0; l < c.n_layers; ++l) {
        const DeviceLayer &d = g.layers[l];
        phase_begin();
        launch_batch_rmsnorm(g.prefill_xn, g.prefill_x, d.input_norm,
                             H, rows, c.rms_eps);
        if (c.q_lora_rank > 0) {
            launch_batch_dual_gemv(
                g.prefill_q_a, d.q_a_proj, d.q_a_proj_s, c.q_lora_rank,
                g.prefill_comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                g.prefill_xn, H, rows);
            launch_batch_rmsnorm(g.prefill_q_a, g.prefill_q_a, d.q_a_norm,
                                 c.q_lora_rank, rows, c.mla_norm_eps);
            launch_batch_gemv(g.prefill_q, g.prefill_q_a, d.q_b_proj,
                              d.q_b_proj_s, QD, c.q_lora_rank, rows);
        } else {
            launch_batch_dual_gemv(
                g.prefill_q, d.q_proj, d.q_proj_s, QD,
                g.prefill_comp, d.kv_a_proj, d.kv_a_proj_s, KVD,
                g.prefill_xn, H, rows);
        }

        bf16_t *layer_cache = g.kv_cache
            + (size_t)l * g_max_batch * g.kv_capacity * KVD;
        batch_decode_kernels::prefill::kv_norm_rope<<<
            rows, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                layer_cache, g.prefill_comp, d.kv_a_norm, row_batches,
                row_positions, slot_base, rows, g.kv_capacity, KVL, KVD,
                QKR, g.rope_inv_freq,
                c.rope_interleaved, c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        dim3 qabs_grid((unsigned)div_up(KVL, kMmaTileSize),
                       (unsigned)div_up(rows, kMmaBatchTile),
                       (unsigned)NH);
        batch_decode_kernels::mfma_head_gemm<<<
            qabs_grid, 256, 0, g.stream>>>(
                g.prefill_qabs, g.prefill_q, d.W_UK_T, rows, NH, KVL,
                QKN, QHD, KVL, (size_t)KVL * QKN);
        HIP_LAUNCH_CHECK();
        batch_decode_kernels::prefill::q_rope<<<
            div_up(rows * NH, kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                g.prefill_qrope, g.prefill_q, row_positions, rows, NH, QHD, QKN,
                g.rope_inv_freq, c.rope_interleaved);
        HIP_LAUNCH_CHECK();
        phase_end(qkv_ms);

        phase_begin();
        constexpr int head_tile = kPrefillThreads / HIP_WAVE;
        dim3 flash_grid((unsigned)div_up(NH, head_tile), (unsigned)rows);
        batch_decode_kernels::prefill::flash_attention<kPrefillKvTile><<<
            flash_grid, kPrefillThreads,
            (size_t)kPrefillKvTile * KVD * sizeof(bf16_t), g.stream>>>(
                g.prefill_clat, g.prefill_qabs, g.prefill_qrope,
                layer_cache, row_batches, row_positions, slot_base, rows, NH,
                g.kv_capacity, KVD, KVL, QKR, c.softmax_scale);
        HIP_LAUNCH_CHECK();
        phase_end(flash_ms);

        phase_begin();
        dim3 value_grid((unsigned)div_up(VHD, kMmaTileSize),
                        (unsigned)div_up(rows, kMmaBatchTile),
                        (unsigned)NH);
        batch_decode_kernels::mfma_head_gemm<<<
            value_grid, 256, 0, g.stream>>>(
                g.prefill_ctx, g.prefill_clat, d.W_UV, rows, NH, VHD,
                KVL, KVL, VHD, head_stride);
        HIP_LAUNCH_CHECK();
        launch_batch_gemv(g.prefill_x, g.prefill_ctx, d.o_proj, d.o_proj_s,
                          H, NH * VHD, rows, true);
        phase_end(output_ms);

        phase_begin();
        launch_batch_rmsnorm(g.prefill_xn, g.prefill_x, d.post_norm,
                             H, rows, c.rms_eps);
        gpu_ffn_rows(
            c, d, l, rows, g.prefill_x, g.prefill_xn, g.prefill_hb,
            g.prefill_router, g.prefill_route_ids, g.prefill_topk_weights,
            g.prefill_topk, g.prefill_expert_counts,
            g.prefill_expert_buckets, nullptr, g.prefill_routed_hidden,
            g.prefill_ffn,
            g.prefill_act_i8, g.prefill_act_s,
            g.prefill_routed_i8, g.prefill_routed_s);
        phase_end(ffn_ms);
    }

    dim3 gather_grid((unsigned)div_up(H, kernel_tuning::ELEMENTWISE_THREADS),
                     (unsigned)batch);
    batch_decode_kernels::prefill::gather_last_hidden<<<
        gather_grid, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.x, g.prefill_x, offsets, batch, slot_base, H);
    HIP_LAUNCH_CHECK();
    launch_batch_rmsnorm(g.xn + (size_t)slot_base * H,
                         g.x + (size_t)slot_base * H,
                         g.final_norm, H, batch, c.rms_eps);
    launch_batch_gemv(g.logits + (size_t)slot_base * c.vocab_size,
                      g.xn + (size_t)slot_base * H, g.lm_head, g.lm_head_s,
                      c.vocab_size, H, batch);
    batch_decode_kernels::argmax<<<
        batch, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.logits + (size_t)slot_base * c.vocab_size, c.vocab_size,
            g.next_token + slot_base, batch);
    HIP_LAUNCH_CHECK();
    if (profile) {
        HIP_CHECK(hipEventDestroy(phase_start));
        HIP_CHECK(hipEventDestroy(phase_stop));
        std::fprintf(stderr,
            "[getp prefill profile] tokens=%d batch=%d qkv=%.3f "
            "flash=%.3f output=%.3f ffn=%.3f ms\n",
            rows, batch, qkv_ms, flash_ms, output_ms, ffn_ms);
    }
}

} // namespace prefill

namespace decode {

/* One token per active sequence.  `positions[b]` is independent for every
 * sequence, so prompts of different lengths remain in the same decode batch. */
static void run_step(const int *device_tokens, int *generated,
                     int generated_stride, int output_index,
                     const int *positions, int max_kv_len,
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
    const size_t head_stride = (size_t)(QKN + VHD) * KVL;
    if (batch < 1 || batch > g_max_batch || max_kv_len < 1 ||
        max_kv_len > g.kv_capacity) {
        std::fprintf(stderr, "HIP token decode: batch=%d/%d max_kv=%d/%d\n",
                     batch, g_max_batch, max_kv_len, g.kv_capacity);
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
                g.q_a, d.q_a_proj, d.q_a_proj_s, c.q_lora_rank,
                g.comp, d.kv_a_proj, d.kv_a_proj_s, KVD, g.xn, H, batch);
            launch_batch_rmsnorm(g.q_a, g.q_a, d.q_a_norm,
                                 c.q_lora_rank, batch, c.mla_norm_eps);
            launch_batch_gemv(g.q, g.q_a, d.q_b_proj, d.q_b_proj_s, QD,
                              c.q_lora_rank, batch);
        } else {
            launch_batch_dual_gemv(
                g.q, d.q_proj, d.q_proj_s, QD,
                g.comp, d.kv_a_proj, d.kv_a_proj_s, KVD, g.xn, H, batch);
        }
        bf16_t *layer_cache = g.kv_cache +
            (size_t)l * g_max_batch * g.kv_capacity * KVD;
        batch_decode_kernels::decode::kv_norm_rope<<<batch,
            kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                layer_cache, g.comp, d.kv_a_norm, positions, batch,
                g.kv_capacity, KVL, KVD, QKR, g.rope_inv_freq, c.rope_interleaved,
                c.mla_norm_eps);
        HIP_LAUNCH_CHECK();

        int qabs_rows = NH * KVL;
        if (batch >= g_mfma_threshold) {
            dim3 qabs_grid((unsigned)div_up(KVL, kMmaTileSize),
                           (unsigned)div_up(batch, kMmaBatchTile),
                           (unsigned)NH);
            batch_decode_kernels::mfma_head_gemm<<<
                qabs_grid, 256, 0, g.stream>>>(
                    g.qabs, g.q, d.W_UK_T, batch, NH, KVL, QKN, QHD, KVL,
                    (size_t)KVL * QKN);
            HIP_LAUNCH_CHECK();
            batch_decode_kernels::decode::q_rope<<<
                div_up(batch * NH,
                       kernel_tuning::ATTENTION_THREADS / HIP_WAVE),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qrope, g.q, positions, batch, NH, QHD, QKN, QKR,
                    g.rope_inv_freq, c.rope_interleaved);
        } else {
            batch_decode_kernels::decode::q_absorb_rope<<<
                batch_projection_grid(qabs_rows, batch),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.qabs, g.qrope, g.q, d.W_UK_T, positions, batch, NH,
                    QHD, QKN, QKR, KVL, g.rope_inv_freq, c.rope_interleaved);
        }
        HIP_LAUNCH_CHECK();
        if (batch >= g_flash_threshold) {
            if (batch >= kFlashLargeBatchThreshold) {
                dim3 flash_grid(
                    (unsigned)div_up(
                        NH, kFlashLargeBatchThreads / HIP_WAVE),
                    (unsigned)batch);
                batch_decode_kernels::decode::flash_attention<
                    kFlashLargeBatchKvTile><<<
                        flash_grid, kFlashLargeBatchThreads,
                        (size_t)kFlashLargeBatchKvTile * KVD * sizeof(bf16_t),
                        g.stream>>>(
                            g.clat, g.qabs, g.qrope, layer_cache, positions,
                            batch, NH, g.kv_capacity, KVD, KVL, QKR,
                            c.softmax_scale);
            } else {
                dim3 flash_grid(
                    (unsigned)div_up(
                        NH, kFlashSmallBatchThreads / HIP_WAVE),
                    (unsigned)batch);
                batch_decode_kernels::decode::flash_attention<
                    kFlashSmallBatchKvTile><<<
                        flash_grid, kFlashSmallBatchThreads,
                        (size_t)kFlashSmallBatchKvTile * KVD * sizeof(bf16_t),
                        g.stream>>>(
                            g.clat, g.qabs, g.qrope, layer_cache, positions,
                            batch, NH, g.kv_capacity, KVD, KVL, QKR,
                            c.softmax_scale);
            }
            HIP_LAUNCH_CHECK();
        } else {
            constexpr int rows_per_block =
                kernel_tuning::ATTENTION_THREADS / HIP_WAVE;
            int score_rows = NH * max_kv_len;
            dim3 score_grid((unsigned)div_up(score_rows, rows_per_block),
                            (unsigned)batch);
            batch_decode_kernels::decode::attention_score<<<score_grid,
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.scores, g.qabs, g.qrope, layer_cache, positions, batch,
                    NH, max_kv_len, g.kv_capacity, KVD, KVL, QKR,
                    c.softmax_scale);
            HIP_LAUNCH_CHECK();
            attention_kernels::softmax_rows_kernel<
                kernel_tuning::ATTENTION_THREADS><<<
                    batch * NH, kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                        g.scores, batch * NH, max_kv_len, g.kv_capacity, 0);
            HIP_LAUNCH_CHECK();
            dim3 latent_grid((unsigned)div_up(qabs_rows, rows_per_block),
                             (unsigned)batch);
            batch_decode_kernels::decode::latent_context<<<latent_grid,
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.clat, g.scores, layer_cache, positions, batch, NH,
                    max_kv_len, g.kv_capacity, KVD, KVL);
            HIP_LAUNCH_CHECK();
        }
        int value_rows = NH * VHD;
        if (batch >= g_mfma_threshold) {
            dim3 value_grid((unsigned)div_up(VHD, kMmaTileSize),
                            (unsigned)div_up(batch, kMmaBatchTile),
                            (unsigned)NH);
            batch_decode_kernels::mfma_head_gemm<<<
                value_grid, 256, 0, g.stream>>>(
                    g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL, KVL, VHD,
                    head_stride);
        } else {
            batch_decode_kernels::decode::value_up<<<
                batch_projection_grid(value_rows, batch),
                kernel_tuning::ATTENTION_THREADS, 0, g.stream>>>(
                    g.ctx, g.clat, d.W_UV, batch, NH, VHD, KVL, head_stride);
        }
        HIP_LAUNCH_CHECK();
        launch_batch_gemv(g.x, g.ctx, d.o_proj, d.o_proj_s, H, NH * VHD,
                          batch, true);

        launch_batch_rmsnorm(g.xn, g.x, d.post_norm, H, batch, c.rms_eps);
        gpu_ffn_batch(c, d, l, batch);
    }

    if (!produce_token) return;
    launch_batch_rmsnorm(g.xn, g.x, g.final_norm, H, batch, c.rms_eps);
    launch_batch_gemv(g.logits, g.xn, g.lm_head, g.lm_head_s,
                      c.vocab_size, H, batch);
    batch_decode_kernels::argmax<<<batch, kernel_tuning::ELEMENTWISE_THREADS,
        0, g.stream>>>(g.logits, c.vocab_size, g.next_token, batch);
    HIP_LAUNCH_CHECK();
}

} // namespace decode

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
    g_decode_token_budget = configured_decode_token_budget();
    g_prefill_token_budget = configured_prefill_token_budget();
    g_max_batch = g_decode_token_budget;
    g_mfma_threshold = configured_threshold(
        "GETP_MFMA_THRESHOLD", kDefaultMfmaThreshold);
    g_flash_threshold = configured_threshold(
        "GETP_FLASH_THRESHOLD", kDefaultFlashThreshold);
    g_quant = configured_quant_groups();
    g_prefill_large_m =
        std::getenv("GETP_DISABLE_PREFILL_LARGE_M") == nullptr;
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
        kFlashSmallBatchThreads > prop.maxThreadsPerBlock ||
        kFlashLargeBatchThreads > prop.maxThreadsPerBlock ||
        kPrefillThreads > prop.maxThreadsPerBlock ||
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

    /* One staging buffer covers every tensor quantized at load time. */
    size_t quant_stage_elems = 0;
    for (const LayerLayout &a : g.layouts)
        for (const QuantChunk &chunk : a.qchunks)
            quant_stage_elems = std::max(
                quant_stage_elems, (size_t)chunk.rows * chunk.cols);
    if (g_quant.lmhead)
        quant_stage_elems = std::max(
            quant_stage_elems, (size_t)c.vocab_size * c.hidden_size);
    bf16_t *quant_stage = nullptr;
    if (quant_stage_elems > 0)
        device_alloc(&quant_stage, quant_stage_elems);

    device_alloc(&g.embed, (size_t)c.vocab_size * c.hidden_size);
    device_alloc(&g.final_norm, c.hidden_size);
    device_alloc(&g.rope_inv_freq, c.qk_rope_head_dim / 2);
    HIP_CHECK(hipMemcpyAsync(g.embed, t->weights.embed_tokens,
                             (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    HIP_CHECK(hipMemcpyAsync(g.final_norm, t->weights.norm,
                             (size_t)c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    if (g_quant.lmhead) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&g.lm_head),
                            (size_t)c.vocab_size * c.hidden_size));
        device_alloc(&g.lm_head_s, (size_t)c.vocab_size);
        HIP_CHECK(hipMemcpyAsync(
            quant_stage, t->weights.lm_head,
            (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
            hipMemcpyHostToDevice, g.stream));
        batch_decode_kernels::quantize_rows<<<
            c.vocab_size, kernel_tuning::ELEMENTWISE_THREADS, 0, g.stream>>>(
                quant_stage, reinterpret_cast<char *>(g.lm_head), g.lm_head_s,
                c.vocab_size, c.hidden_size);
        HIP_LAUNCH_CHECK();
    } else {
        device_alloc(&g.lm_head, (size_t)c.vocab_size * c.hidden_size);
        HIP_CHECK(hipMemcpyAsync(g.lm_head, t->weights.lm_head,
                                 (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
                                 hipMemcpyHostToDevice, g.stream));
    }
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
    if (g_quant.experts && c.n_routed_experts > 0) {
        std::vector<size_t> host_scale_offsets(offset_count, 0);
        for (int l = c.first_k_dense; l < c.n_layers; ++l) {
            size_t base = (size_t)l * 3 * c.n_routed_experts;
            for (int e = 0; e < c.n_routed_experts; ++e) {
                host_scale_offsets[base + e] = g.layouts[l].expert_gate_s[e];
                host_scale_offsets[base + c.n_routed_experts + e] =
                    g.layouts[l].expert_up_s[e];
                host_scale_offsets[base + 2 * c.n_routed_experts + e] =
                    g.layouts[l].expert_down_s[e];
            }
        }
        device_alloc(&g.expert_scale_offsets, offset_count);
        HIP_CHECK(hipMemcpy(g.expert_scale_offsets, host_scale_offsets.data(),
                            offset_count * sizeof(size_t),
                            hipMemcpyHostToDevice));
    }

    g.layers.resize(c.n_layers);
    g.layer_pools.resize(c.n_layers, nullptr);
    for (int l = 0; l < c.n_layers; ++l) {
        void *pool = nullptr;
        HIP_CHECK(hipMalloc(&pool, g.layouts[l].bytes));
        g.layer_pools[l] = pool;
        enqueue_layer_copy(g.layouts[l], pool, quant_stage);
        g.layers[l] = materialize_layer(g.layouts[l], pool, l);
    }
    if (quant_stage) {
        HIP_CHECK(hipStreamSynchronize(g.stream));
        device_free(quant_stage);
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
    constexpr size_t vram_reserve = kVramReserve;
    size_t memory_batch = free_after_weights > vram_reserve
        ? (free_after_weights - vram_reserve) / per_batch_bytes : 0;
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
    g_decode_token_budget = g_max_batch;

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
    device_alloc(&g.routed_hidden,
                 B * c.n_experts_per_tok * c.moe_inter_size);
    if (g_quant.w8a8) {
        const size_t act_cols = std::max(
            (size_t)inter,
            std::max((size_t)c.hidden_size,
                     (size_t)c.n_heads * c.v_head_dim));
        device_alloc(&g.act_i8, B * act_cols);
        device_alloc(&g.act_s, B);
        device_alloc(&g.routed_i8, B * c.n_experts_per_tok * c.moe_inter_size);
        device_alloc(&g.routed_s, B * c.n_experts_per_tok);
    }
    device_alloc(&g.router, B * std::max(1, c.n_routed_experts));
    device_alloc(&g.topk, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.topk_weights, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.logits, B * c.vocab_size);
    device_alloc(&g.next_token, B);
    device_alloc(&g.generated, B * c.max_seq_len);
    device_alloc(&g.decode_positions, B);
    g.kv_capacity = 0;
    allocate_sequence_buffers(c, warm_capacity);
    /* Prefill workspace follows its token budget, independently of how many
     * decode tokens are scheduled per step. */
    const int target_prefill_rows = std::min(
        kMaxPrefillRows,
        std::max(g_prefill_token_budget, warm_capacity));
    allocate_prefill_workspace(c, target_prefill_rows);
    g_prefill_token_budget = std::min(
        g_prefill_token_budget, g.prefill_capacity);
    HIP_CHECK(hipStreamSynchronize(g.stream));

    /* Exercise every inference kernel family before the timed call.  Batch 1
     * covers the GEMV/non-grouped path; batch 16+ covers MFMA/grouped MoE and
     * batch 32 additionally covers flash decode.  Position zero keeps this
     * warmup cheap while still forcing HIP's first dispatch/module setup. */
    const int kernel_warm_batch = std::min(g_max_batch, 32);
    HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                             (size_t)kernel_warm_batch * sizeof(int), g.stream));
    HIP_CHECK(hipMemsetAsync(g.decode_positions, 0,
                             (size_t)kernel_warm_batch * sizeof(int), g.stream));
    decode::run_step(g.next_token, g.generated, c.max_seq_len,
                     0, g.decode_positions, 1, true, 1);
    if (kernel_warm_batch > 1) {
        HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                                 (size_t)kernel_warm_batch * sizeof(int), g.stream));
        decode::run_step(g.next_token, g.generated, c.max_seq_len,
                         0, g.decode_positions, 1, true, kernel_warm_batch);
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));

    std::fprintf(stderr, "[getp HIP] device=%s vram=%.2f GiB weights=%.2f GiB mode=resident kv=bf16 quant=%s%s%s%s%s%s prefill_tokens=%d decode_tokens=%d warm_kv=%d prefill_rows=%d prefill_m=%d mfma_at=%d flash_at=%d\n",
                 prop.name, (double)total_bytes / (double)(1ULL << 30),
                 (double)(global_bytes + layer_total) / (double)(1ULL << 30),
                 g_quant.any() ? "" : "none",
                 g_quant.w8a8 ? "w8a8," : "",
                 g_quant.experts ? "experts," : "",
                 g_quant.shared_experts ? "shared," : "",
                 g_quant.proj ? "proj," : "",
                 g_quant.lmhead ? "lmhead," : "",
                 g_prefill_token_budget, g_decode_token_budget,
                 g.kv_capacity, g.prefill_capacity,
                 g_prefill_large_m ? kPrefillMmaRows : kMmaBatchTile,
                 g_mfma_threshold,
                 g_flash_threshold);
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
    device_free(g.expert_scale_offsets);
    device_free(g.lm_head_s);
    device_free(g.x); device_free(g.xn);
    device_free(g.q); device_free(g.q_a); device_free(g.comp);
    device_free(g.qrope); device_free(g.qabs); device_free(g.scores); device_free(g.clat);
    device_free(g.ctx); device_free(g.hb); device_free(g.router);
    device_free(g.topk); device_free(g.topk_weights); device_free(g.logits);
    device_free(g.kv_cache); device_free(g.next_token); device_free(g.generated);
    device_free(g.decode_positions);
    device_free(g.prompt_tokens); device_free(g.prefill_route_ids);
    device_free(g.prefill_row_batches); device_free(g.prefill_row_positions);
    device_free(g.prefill_offsets);
    device_free(g.prefill_topk);
    device_free(g.prefill_expert_counts); device_free(g.prefill_expert_buckets);
    device_free(g.prefill_x); device_free(g.prefill_xn);
    device_free(g.prefill_q); device_free(g.prefill_q_a); device_free(g.prefill_comp);
    device_free(g.prefill_qrope); device_free(g.prefill_qabs);
    device_free(g.prefill_clat);
    device_free(g.prefill_scores); device_free(g.prefill_ctx);
    device_free(g.prefill_ffn);
    device_free(g.prefill_router); device_free(g.prefill_topk_weights);
    device_free(g.prefill_hb); device_free(g.routed_hidden);
    device_free(g.prefill_routed_hidden);
    device_free(g.act_i8); device_free(g.act_s);
    device_free(g.routed_i8); device_free(g.routed_s);
    device_free(g.prefill_act_i8); device_free(g.prefill_act_s);
    device_free(g.prefill_routed_i8); device_free(g.prefill_routed_s);
    HIP_CHECK(hipStreamDestroy(g.stream));
    g = GpuContext();
    g_quant = QuantGroups();
    g_max_batch = 1;
    g_prefill_token_budget = kDefaultPrefillBatchTokens;
    g_decode_token_budget = kDefaultBatchSize;
    g_mfma_threshold = kDefaultMfmaThreshold;
    g_flash_threshold = kDefaultFlashThreshold;
    g_prefill_large_m = true;
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


    /* Preserve request order.  Token budgets decide batch boundaries; prompt
     * length is metadata, never a sorting/grouping key. */
    std::vector<int> active_requests;
    active_requests.reserve((size_t)reqs->num_reqs);
    for (int r = 0; r < reqs->num_reqs; ++r)
        if (!prompts[r].empty()) active_requests.push_back(r);

    long long total = 0;
    const bool report_prefill_timing =
        std::getenv("GETP_REPORT_PREFILL_TIMING") != nullptr;
    const bool report_decode_timing =
        std::getenv("GETP_REPORT_DECODE_TIMING") != nullptr;
    hipEvent_t prefill_start = nullptr, prefill_stop = nullptr;
    if (report_prefill_timing) {
        HIP_CHECK(hipEventCreate(&prefill_start));
        HIP_CHECK(hipEventCreate(&prefill_stop));
    }
    hipEvent_t decode_start = nullptr, decode_stop = nullptr;
    if (report_decode_timing) {
        HIP_CHECK(hipEventCreate(&decode_start));
        HIP_CHECK(hipEventCreate(&decode_stop));
    }
    size_t group_begin = 0;
    while (group_begin < active_requests.size()) {
        const int first_request = active_requests[group_begin];
        const int group_steps = std::max(
            0, std::min(reqs->max_steps,
                        max_seq - (int)prompts[first_request].size()));
        size_t group_end = group_begin;
        while (group_end < active_requests.size() &&
               group_end - group_begin < (size_t)g_decode_token_budget) {
            const int r = active_requests[group_end];
            const int steps = std::max(
                0, std::min(reqs->max_steps,
                            max_seq - (int)prompts[r].size()));
            if (steps != group_steps) break;
            ++group_end;
        }
        const int batch = (int)(group_end - group_begin);
        std::vector<int> prompt_lengths((size_t)batch);
        int group_tokens = 0;
        for (int b = 0; b < batch; ++b) {
            const int r = active_requests[group_begin + (size_t)b];
            prompt_lengths[b] = (int)prompts[r].size();
            group_tokens += prompt_lengths[b];
            if (prompt_lengths[b] > g.prefill_capacity) {
                std::fprintf(stderr,
                    "HIP packed prefill needs %d tokens for request %d, but "
                    "workspace holds %d\n", prompt_lengths[b], r,
                    g.prefill_capacity);
                std::exit(EXIT_FAILURE);
            }
        }

        if (report_prefill_timing)
            HIP_CHECK(hipEventRecord(prefill_start, g.stream));
        int slot_base = 0;
        int prefill_chunks = 0;
        while (slot_base < batch) {
            int chunk_batch = 0;
            int rows = 0;
            while (slot_base + chunk_batch < batch) {
                const int n = prompt_lengths[slot_base + chunk_batch];
                if (chunk_batch > 0 &&
                    (rows + n > g_prefill_token_budget ||
                     rows + n > g.prefill_capacity))
                    break;
                rows += n;
                ++chunk_batch;
            }

            std::vector<int> flat_tokens;
            std::vector<int> row_batches;
            std::vector<int> row_positions;
            std::vector<int> offsets((size_t)chunk_batch + 1, 0);
            flat_tokens.reserve((size_t)rows);
            row_batches.reserve((size_t)rows);
            row_positions.reserve((size_t)rows);
            for (int b = 0; b < chunk_batch; ++b) {
                const int r = active_requests[
                    group_begin + (size_t)slot_base + (size_t)b];
                const int n = (int)prompts[r].size();
                flat_tokens.insert(flat_tokens.end(), prompts[r].begin(),
                                   prompts[r].end());
                for (int p = 0; p < n; ++p) {
                    row_batches.push_back(b);
                    row_positions.push_back(p);
                }
                offsets[(size_t)b + 1] = offsets[(size_t)b] + n;
            }
            HIP_CHECK(hipMemcpyAsync(
                g.prompt_tokens, flat_tokens.data(),
                (size_t)rows * sizeof(int), hipMemcpyHostToDevice, g.stream));
            HIP_CHECK(hipMemcpyAsync(
                g.prefill_row_batches, row_batches.data(),
                (size_t)rows * sizeof(int), hipMemcpyHostToDevice, g.stream));
            HIP_CHECK(hipMemcpyAsync(
                g.prefill_row_positions, row_positions.data(),
                (size_t)rows * sizeof(int), hipMemcpyHostToDevice, g.stream));
            HIP_CHECK(hipMemcpyAsync(
                g.prefill_offsets, offsets.data(),
                ((size_t)chunk_batch + 1) * sizeof(int),
                hipMemcpyHostToDevice, g.stream));
            prefill::run_chunk(
                g.prompt_tokens, g.prefill_row_batches,
                g.prefill_row_positions, g.prefill_offsets,
                rows, chunk_batch, slot_base);
            slot_base += chunk_batch;
            ++prefill_chunks;
        }
        if (report_prefill_timing) {
            HIP_CHECK(hipEventRecord(prefill_stop, g.stream));
            HIP_CHECK(hipEventSynchronize(prefill_stop));
            float prefill_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(
                &prefill_ms, prefill_start, prefill_stop));
            std::fprintf(stderr,
                "[getp prefill] tokens=%d batch=%d chunks=%d ms=%.3f "
                "input_tps=%.3f\n", group_tokens, batch, prefill_chunks,
                prefill_ms, (double)group_tokens * 1e3 / prefill_ms);
        }

        std::vector<int> host_positions((size_t)batch);
        if (report_decode_timing && group_steps > 0)
            HIP_CHECK(hipEventRecord(decode_start, g.stream));
        for (int step = 0; step < group_steps; ++step) {
            int max_kv_len = 1;
            for (int b = 0; b < batch; ++b) {
                host_positions[b] = prompt_lengths[b] + step;
                max_kv_len = std::max(max_kv_len, host_positions[b] + 1);
            }
            HIP_CHECK(hipMemcpyAsync(
                g.decode_positions, host_positions.data(),
                (size_t)batch * sizeof(int), hipMemcpyHostToDevice, g.stream));
            decode::run_step(g.next_token, g.generated, max_seq, step,
                             g.decode_positions, max_kv_len, true, batch);
        }
        if (report_decode_timing && group_steps > 0) {
            HIP_CHECK(hipEventRecord(decode_stop, g.stream));
            HIP_CHECK(hipEventSynchronize(decode_stop));
            float decode_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(
                &decode_ms, decode_start, decode_stop));
            std::fprintf(stderr,
                "[getp decode] tokens_per_step=%d steps=%d ms=%.3f "
                "compute_tps=%.3f\n", batch, group_steps, decode_ms,
                (double)batch * group_steps * 1e3 / decode_ms);
        }

        std::vector<int> host_generated((size_t)batch * group_steps);
        if (group_steps > 0) {
            for (int b = 0; b < batch; ++b)
                HIP_CHECK(hipMemcpyAsync(
                    host_generated.data() + (size_t)b * group_steps,
                    g.generated + (size_t)b * max_seq,
                    (size_t)group_steps * sizeof(int), hipMemcpyDeviceToHost,
                    g.stream));
            HIP_CHECK(hipStreamSynchronize(g.stream));
        }
        for (int b = 0; b < batch; ++b) {
            const int r = active_requests[group_begin + (size_t)b];
            const int *generated = group_steps > 0
                ? host_generated.data() + (size_t)b * group_steps : nullptr;
            int n_out = 0;
            while (n_out < group_steps) {
                const int token = generated[n_out];
                reqs->out_tokens[r][n_out] = token;
                ++n_out;
                if (token == eos) break;
            }
            reqs->out_lens[r] = n_out;
            total += n_out;
        }
        group_begin = group_end;
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));
    if (report_prefill_timing) {
        HIP_CHECK(hipEventDestroy(prefill_start));
        HIP_CHECK(hipEventDestroy(prefill_stop));
    }
    if (report_decode_timing) {
        HIP_CHECK(hipEventDestroy(decode_start));
        HIP_CHECK(hipEventDestroy(decode_stop));
    }
    return total;
}
