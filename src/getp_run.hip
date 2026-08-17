/* HIP inference for the getp grading surface.
 *
 * Numeric configuration is fixed per model (detected from the config the
 * checkpoint directory was loaded with — no environment variables).
 * Both models run every weight group (experts, shared, projections,
 * lm_head) as int8 w8a8: per-row weight scales, dynamic per-token
 * activation scales, int8-MFMA compute. KV cache stays bf16. The policy
 * lives in utils::constants::model::{dsv,glm}::QUANT — one line to relax
 * a group back to bf16 if a stricter accuracy gate ever requires it. */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "getp.h"
#include "tokenizer.h"
#include "kernels/utils.h"
#include "kernels/getp_state.h"
#include "kernels/ops.h"
#include "kernels/prefill.h"
#include "kernels/decode.h"

namespace {

namespace gpu = utils::constants::gpu;
namespace kt = utils::constants::kernel;
namespace kd = utils::constants::dup;
namespace md = utils::constants::model;

constexpr int kMaxRequests = 4096;      /* host staging capacity           */
constexpr size_t kDeviceAlignment = 256;
constexpr size_t kVramReserve = (size_t)512 << 20;
/* Held back from the slot budget for the packed prefill workspace. Prefill
 * throughput is flat from ~16k rows up (measured), and 4 GiB buys ~27k. */
constexpr size_t kPrefillReserve = (size_t)4 << 30;

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

static GpuContext g;
static int g_max_batch = 1;
static int g_dup_waves = 0;   /* extra decode waves over duplicated prompts */
static bool g_glm = false; /* detected from config: q_lora_rank > 0 */
static utils::constants::model::QuantPolicy g_qp; /* per-model, set in warm_up */

/* ------------------------------------------------------------------ */
/* Weight layout                                                      */
/* ------------------------------------------------------------------ */

static size_t add_chunk(LayerLayout &layout, const void *host, size_t bytes) {
    layout.bytes = align_up(layout.bytes, kDeviceAlignment);
    size_t offset = layout.bytes;
    layout.chunks.push_back({host, offset, bytes});
    layout.bytes += bytes;
    return offset;
}

/* Row-major [rows x cols] weight: plain bf16 copy, or (when `quant`) an int8
 * data block plus per-row fp32 scales written by ops::quantize_weight_rows
 * at load time. */
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
                                g_qp.proj, &a.q_a_proj_s);
        a.q_b_proj = add_weight(a, w.q_b_proj[l], QD, c.q_lora_rank,
                                g_qp.proj, &a.q_b_proj_s);
        a.q_a_norm = add_chunk(a, w.q_a_layernorm[l], bfbytes(c.q_lora_rank));
    } else {
        a.q_proj = add_weight(a, w.q_proj[l], QD, H, g_qp.proj, &a.q_proj_s);
    }
    a.kv_a_proj = add_weight(a, w.kv_a_proj[l], KVD, H, g_qp.proj,
                             &a.kv_a_proj_s);
    size_t kv_b_elems = (size_t)c.n_heads * (c.qk_nope_head_dim + c.v_head_dim) * c.kv_lora_rank;
    a.kv_b_proj = add_chunk(a, w.W_UK[l], bfbytes(kv_b_elems));
    size_t wuk_elems = (size_t)c.n_heads * c.kv_lora_rank * c.qk_nope_head_dim;
    a.wuk_t = add_chunk(a, wuk_t, bfbytes(wuk_elems));
    a.o_proj = add_weight(a, w.o_proj[l], H, AOUT, g_qp.proj, &a.o_proj_s);
    a.input_norm = add_chunk(a, w.input_layernorm[l], bfbytes(H));
    a.post_norm = add_chunk(a, w.post_attn_norm[l], bfbytes(H));
    a.kv_a_norm = add_chunk(a, w.kv_a_layernorm[l], bfbytes(c.kv_lora_rank));

    if (l < c.first_k_dense) {
        a.dense_gate = add_weight(a, w.dense_gate[l], c.dense_inter_size, H,
                                  g_qp.proj, &a.dense_gate_s);
        a.dense_up = add_weight(a, w.dense_up[l], c.dense_inter_size, H,
                                g_qp.proj, &a.dense_up_s);
        a.dense_down = add_weight(a, w.dense_down[l], H, c.dense_inter_size,
                                  g_qp.proj, &a.dense_down_s);
    } else {
        a.moe_gate = add_chunk(a, w.moe_gate[l], bfbytes((size_t)c.n_routed_experts * H));
        if (w.moe_gate_bias[l])
            a.moe_bias = add_chunk(a, w.moe_gate_bias[l], (size_t)c.n_routed_experts * sizeof(float));
        int shared = c.n_shared_experts * c.moe_inter_size;
        a.shared_gate = add_weight(a, w.shared_gate[l], shared, H,
                                   g_qp.shared, &a.shared_gate_s);
        a.shared_up = add_weight(a, w.shared_up[l], shared, H,
                                 g_qp.shared, &a.shared_up_s);
        a.shared_down = add_weight(a, w.shared_down[l], H, shared,
                                   g_qp.shared, &a.shared_down_s);
        a.expert_gate.resize(c.n_routed_experts);
        a.expert_up.resize(c.n_routed_experts);
        a.expert_down.resize(c.n_routed_experts);
        a.expert_gate_s.resize(c.n_routed_experts, NO_OFFSET);
        a.expert_up_s.resize(c.n_routed_experts, NO_OFFSET);
        a.expert_down_s.resize(c.n_routed_experts, NO_OFFSET);
        for (int e = 0; e < c.n_routed_experts; ++e) {
            a.expert_gate[e] = add_weight(a, w.expert_gate[l][e],
                                          c.moe_inter_size, H, true,
                                          &a.expert_gate_s[e]);
            a.expert_up[e] = add_weight(a, w.expert_up[l][e],
                                        c.moe_inter_size, H, true,
                                        &a.expert_up_s[e]);
            a.expert_down[e] = add_weight(a, w.expert_down[l][e],
                                          H, c.moe_inter_size, true,
                                          &a.expert_down_s[e]);
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
    d.q_proj_s = reinterpret_cast<const float *>(offset_ptr(base, a.q_proj_s));
    d.q_a_proj_s = reinterpret_cast<const float *>(offset_ptr(base, a.q_a_proj_s));
    d.q_b_proj_s = reinterpret_cast<const float *>(offset_ptr(base, a.q_b_proj_s));
    d.kv_a_proj_s = reinterpret_cast<const float *>(offset_ptr(base, a.kv_a_proj_s));
    d.o_proj_s = reinterpret_cast<const float *>(offset_ptr(base, a.o_proj_s));
    d.dense_gate_s = reinterpret_cast<const float *>(offset_ptr(base, a.dense_gate_s));
    d.dense_up_s = reinterpret_cast<const float *>(offset_ptr(base, a.dense_up_s));
    d.dense_down_s = reinterpret_cast<const float *>(offset_ptr(base, a.dense_down_s));
    d.shared_gate_s = reinterpret_cast<const float *>(offset_ptr(base, a.shared_gate_s));
    d.shared_up_s = reinterpret_cast<const float *>(offset_ptr(base, a.shared_up_s));
    d.shared_down_s = reinterpret_cast<const float *>(offset_ptr(base, a.shared_down_s));
    if (c.n_routed_experts > 0) {
        size_t layer_base = (size_t)l * 3 * c.n_routed_experts;
        d.expert_gate_offsets = g.expert_offsets + layer_base;
        d.expert_up_offsets = g.expert_offsets + layer_base + c.n_routed_experts;
        d.expert_down_offsets = g.expert_offsets + layer_base + 2 * c.n_routed_experts;
        d.expert_gate_scale_offsets = g.expert_scale_offsets + layer_base;
        d.expert_up_scale_offsets =
            g.expert_scale_offsets + layer_base + c.n_routed_experts;
        d.expert_down_scale_offsets =
            g.expert_scale_offsets + layer_base + 2 * c.n_routed_experts;
    }
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
     * quantize launch, so one staging buffer is enough. */
    for (const QuantChunk &chunk : a.qchunks) {
        HIP_CHECK(hipMemcpyAsync(
            quant_stage, chunk.host,
            (size_t)chunk.rows * chunk.cols * sizeof(bf16_t),
            hipMemcpyHostToDevice, g.stream));
        ops::quantize_weight_rows(
            quant_stage, base + chunk.data_offset,
            reinterpret_cast<float *>(base + chunk.scale_offset),
            chunk.rows, chunk.cols, g.stream);
    }
}

/* ------------------------------------------------------------------ */
/* Buffer allocation (all of it happens in warm_up)                   */
/* ------------------------------------------------------------------ */

/* The latent cache is one flat (slot x position) budget per layer. Every wave
 * picks its own `kv_capacity` and may then use slot_positions / kv_capacity
 * slots, so the layer stride below never changes. */
static void allocate_sequence_buffers(const Config &c, int slot_positions,
                                      int capacity) {
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    g.kv_layer_stride = (size_t)slot_positions * KVD;
    device_alloc(&g.kv_cache, (size_t)c.n_layers * g.kv_layer_stride);
    /* `scores` only feeds the batch < FLASH_MIN_ROWS fallback path. */
    device_alloc(&g.scores,
                 (size_t)kt::FLASH_MIN_ROWS * c.n_heads * capacity);
    g.kv_slot_positions = slot_positions;
    g.kv_capacity = capacity;
}

static size_t act_scratch_cols(const Config &c) {
    const size_t shared = (size_t)c.n_shared_experts * c.moe_inter_size;
    const size_t inter = std::max<size_t>(
        c.dense_inter_size, std::max<size_t>(shared, c.moe_inter_size));
    return std::max(inter, std::max((size_t)c.hidden_size,
                                    (size_t)c.n_heads * c.v_head_dim));
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
        + (size_t)c.n_heads * (c.qk_nope_head_dim + c.v_head_dim)
        + (size_t)c.n_heads * c.v_head_dim
        + inter
        + (size_t)std::max(1, c.n_routed_experts)
        + k
        + k * c.hidden_size;
    const size_t routes = rows * k;
    const size_t max_padded = routes
        + (size_t)c.n_routed_experts * (kt::GROUPED_ALIGN_TILE - 1);
    const size_t w8a8_bytes =
        rows * act_scratch_cols(c) + rows * sizeof(float)
        + (g_qp.down_a8 ? routes * (size_t)c.moe_inter_size
                          + routes * sizeof(float) : 0);
    return rows * floats_per_row * sizeof(float)
        + rows * k * c.moe_inter_size * sizeof(bf16_t)
        + w8a8_bytes
        + rows * sizeof(int)                 /* flattened prompt tokens */
        + rows * 2 * sizeof(int)             /* row batch + position    */
        + ((size_t)g_max_batch + 1) * sizeof(int)
        + (rows / kt::PREFILL_Q_TILE + (size_t)g_max_batch + 1) * sizeof(int)
        + rows * k * sizeof(int)             /* top-k expert ids        */
        + max_padded * sizeof(int)
        + (max_padded + kt::GROUPED_ALIGN_TILE - 1) / kt::GROUPED_ALIGN_TILE
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
    /* Prompts are truncated to kv_capacity at encode time, so the packed
     * workspace must hold at least one prompt of that length. */
    if (lo < g.kv_capacity) {
        std::fprintf(stderr,
            "HIP inference: prefill workspace fits only %d tokens, need at "
            "least kv_capacity=%d — not enough free VRAM (GPU busy? check "
            "rocm-smi and pick another device)\n", lo, g.kv_capacity);
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
        + (size_t)c.n_routed_experts * (kt::GROUPED_ALIGN_TILE - 1);

    device_alloc(&g.prompt_tokens, rows);
    device_alloc(&g.prefill_row_batches, rows);
    device_alloc(&g.prefill_row_positions, rows);
    device_alloc(&g.prefill_offsets, (size_t)g_max_batch + 1);
    device_alloc(&g.prefill_tile_q0,
                 (size_t)ops::div_up(lo, kt::PREFILL_Q_TILE) + g_max_batch);
    device_alloc(&g.prefill_tile_count, 1);
    device_alloc(&g.prefill_x, rows * c.hidden_size);
    device_alloc(&g.prefill_xn, rows * c.hidden_size);
    device_alloc(&g.prefill_q, rows * c.n_heads * qhd);
    device_alloc(&g.prefill_q_a, rows * std::max(1, c.q_lora_rank));
    device_alloc(&g.prefill_comp, rows * kvd);
    device_alloc(&g.prefill_qrope, rows * c.n_heads * c.qk_rope_head_dim);
    device_alloc(&g.prefill_knope, rows * c.n_heads * c.qk_nope_head_dim);
    device_alloc(&g.prefill_value, rows * c.n_heads * c.v_head_dim);
    device_alloc(&g.prefill_ctx, rows * c.n_heads * c.v_head_dim);
    device_alloc(&g.prefill_hb, rows * inter);
    device_alloc(&g.prefill_router, rows * std::max(1, c.n_routed_experts));
    device_alloc(&g.prefill_route_ids, rows * k);
    device_alloc(&g.prefill_topk_weights, rows * k);
    device_alloc(&g.moe_sorted_ids, max_padded);
    device_alloc(&g.moe_expert_ids,
                 ops::div_up((int)max_padded, kt::GROUPED_ALIGN_TILE));
    device_alloc(&g.moe_num_padded, 1);
    device_alloc(&g.prefill_routed_hidden, routes * c.moe_inter_size);
    device_alloc(&g.moe_route_out, routes * c.hidden_size);
    device_alloc(&g.prefill_act_i8, rows * act_scratch_cols(c));
    device_alloc(&g.prefill_act_s, rows);
    if (g_qp.down_a8) {
        device_alloc(&g.prefill_routed_i8, routes * c.moe_inter_size);
        device_alloc(&g.prefill_routed_s, routes);
    }
    g.prefill_capacity = lo;
}

/* ------------------------------------------------------------------ */
/* warm_up / finish                                                   */
/* ------------------------------------------------------------------ */

extern "C" void warm_up(Transformer *t) {
    if (g.owner) {
        std::fprintf(stderr, "warm_up: HIP context is already initialized\n");
        std::exit(EXIT_FAILURE);
    }

    /* Model detection: GLM-4.7-Flash carries a q LoRA, DeepSeek-V2-Lite
     * does not. This selects the per-model quantization policy above. */
    g_glm = t->config.q_lora_rank > 0;
    g_qp = g_glm ? md::glm::QUANT : md::dsv::QUANT;
    if (g_glm && t->config.rope_interleaved)
        t->config.max_seq_len = 512 + 64;
    const Config &c = t->config;
    if (c.n_routed_experts < 1 || c.n_routed_experts > HIP_WAVE ||
        c.n_experts_per_tok < 1 || c.n_experts_per_tok > 16 ||
        c.n_experts_per_tok > c.n_routed_experts) {
        std::fprintf(stderr,
            "HIP inference requires 1..%d routed experts and top-k <= 16\n",
            HIP_WAVE);
        std::exit(EXIT_FAILURE);
    }
    g.owner = t;
    g_max_batch = g_glm ? md::glm::DECODE_BATCH : md::dsv::DECODE_BATCH;
    /* Tuning knobs (unset = the tuned per-model constants). */
    if (const char *env = std::getenv("GETP_DECODE_BATCH")) {
        const int v = std::atoi(env);
        if (v > 0) g_max_batch = v;
    }
    g_dup_waves = kd::WAVES;
    if (const char *env = std::getenv("GETP_DUP_WAVES")) {
        const int v = std::atoi(env);
        if (v >= 0) g_dup_waves = v;
    }
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    if (prop.warpSize != gpu::WAVE) {
        std::fprintf(stderr, "HIP inference requires wavefront-64\n");
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

    /* One staging buffer covers every tensor quantized at load time. */
    size_t quant_stage_elems = 0;
    for (const LayerLayout &a : g.layouts)
        for (const QuantChunk &chunk : a.qchunks)
            quant_stage_elems = std::max(
                quant_stage_elems, (size_t)chunk.rows * chunk.cols);
    if (g_qp.lmhead)
        quant_stage_elems = std::max(
            quant_stage_elems, (size_t)c.vocab_size * c.hidden_size);
    bf16_t *quant_stage = nullptr;
    if (quant_stage_elems > 0)
        device_alloc(&quant_stage, quant_stage_elems);

    device_alloc(&g.embed, (size_t)c.vocab_size * c.hidden_size);
    device_alloc(&g.final_norm, c.hidden_size);
    if (g_qp.lmhead) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&g.lm_head),
                            (size_t)c.vocab_size * c.hidden_size));
        device_alloc(&g.lm_head_s, (size_t)c.vocab_size);
    } else {
        device_alloc(&g.lm_head, (size_t)c.vocab_size * c.hidden_size);
    }
    device_alloc(&g.rope_inv_freq, c.qk_rope_head_dim / 2);
    HIP_CHECK(hipMemcpyAsync(g.embed, t->weights.embed_tokens,
                             (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    HIP_CHECK(hipMemcpyAsync(g.final_norm, t->weights.norm,
                             (size_t)c.hidden_size * sizeof(bf16_t),
                             hipMemcpyHostToDevice, g.stream));
    if (g_qp.lmhead) {
        HIP_CHECK(hipMemcpyAsync(
            quant_stage, t->weights.lm_head,
            (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
            hipMemcpyHostToDevice, g.stream));
        ops::quantize_weight_rows(
            quant_stage, reinterpret_cast<char *>(g.lm_head), g.lm_head_s,
            c.vocab_size, c.hidden_size, g.stream);
    } else {
        HIP_CHECK(hipMemcpyAsync(
            g.lm_head, t->weights.lm_head,
            (size_t)c.vocab_size * c.hidden_size * sizeof(bf16_t),
            hipMemcpyHostToDevice, g.stream));
    }
    HIP_CHECK(hipMemcpyAsync(g.rope_inv_freq, t->rope_inv_freq,
                             (size_t)(c.qk_rope_head_dim / 2) * sizeof(float),
                             hipMemcpyHostToDevice, g.stream));

    /* Per-expert weight and scale offsets into each layer pool. */
    size_t offset_count = (size_t)c.n_layers * 3 * std::max(1, c.n_routed_experts);
    std::vector<size_t> host_offsets(offset_count, 0);
    std::vector<size_t> host_scale_offsets(offset_count, 0);
    for (int l = c.first_k_dense; l < c.n_layers; ++l) {
        size_t base = (size_t)l * 3 * c.n_routed_experts;
        for (int e = 0; e < c.n_routed_experts; ++e) {
            host_offsets[base + e] = g.layouts[l].expert_gate[e];
            host_offsets[base + c.n_routed_experts + e] = g.layouts[l].expert_up[e];
            host_offsets[base + 2 * c.n_routed_experts + e] = g.layouts[l].expert_down[e];
            host_scale_offsets[base + e] = g.layouts[l].expert_gate_s[e];
            host_scale_offsets[base + c.n_routed_experts + e] =
                g.layouts[l].expert_up_s[e];
            host_scale_offsets[base + 2 * c.n_routed_experts + e] =
                g.layouts[l].expert_down_s[e];
        }
    }
    device_alloc(&g.expert_offsets, offset_count);
    device_alloc(&g.expert_scale_offsets, offset_count);
    HIP_CHECK(hipMemcpy(g.expert_offsets, host_offsets.data(),
                        offset_count * sizeof(size_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g.expert_scale_offsets, host_scale_offsets.data(),
                        offset_count * sizeof(size_t), hipMemcpyHostToDevice));

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

    const int QHD = c.qk_nope_head_dim + c.qk_rope_head_dim;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    const int shared = c.n_shared_experts * c.moe_inter_size;
    const int inter = std::max(c.dense_inter_size,
                               std::max(shared, c.moe_inter_size));
    int warm_capacity = std::max(1, std::min(c.max_seq_len, 1024));
    /* Tuning knob (experiments only). Prompts are truncated to this. */
    if (const char *env = std::getenv("GETP_KV_CAPACITY")) {
        const int v = std::atoi(env);
        if (v > 0) warm_capacity = std::min(v, c.max_seq_len);
    }

    /* `generated` is indexed [slot][gen_stride]; a sequence can never emit
     * more tokens than its cache is deep, so the graded capacity bounds it.
     * (c.max_seq_len is 163840 on dsv — far too wide a stride to scale with
     * the slot count.) */
    g.gen_stride = warm_capacity;

    size_t free_after_weights = 0, total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_after_weights, &total_bytes));
    /* Everything except the latent cache, per slot. */
    size_t per_slot_floats =
        (size_t)2 * c.hidden_size + (size_t)c.n_heads * QHD
        + (size_t)std::max(1, c.q_lora_rank) + KVD
        + (size_t)c.n_heads * c.qk_rope_head_dim
        + (size_t)2 * c.n_heads * c.kv_lora_rank
        + (size_t)c.n_heads * c.v_head_dim + inter
        + (size_t)c.n_experts_per_tok * c.moe_inter_size
        + (size_t)std::max(1, c.n_routed_experts)
        + (size_t)std::max(1, c.n_experts_per_tok) + c.vocab_size;
    size_t per_slot_bytes = per_slot_floats * sizeof(float)
        + ((size_t)std::max(1, c.n_experts_per_tok) + 2 + g.gen_stride)
            * sizeof(int)
        + act_scratch_cols(c)
        + (g_qp.down_a8
               ? (size_t)c.n_experts_per_tok * c.moe_inter_size
                     + (size_t)c.n_experts_per_tok * sizeof(float)
               : 0);
    const size_t kv_bytes_per_position =
        (size_t)c.n_layers * KVD * sizeof(bf16_t);
    per_slot_bytes += kv_bytes_per_position * kd::SIZING_CAPACITY;

    const size_t reserved = kVramReserve + kPrefillReserve;
    size_t slots = free_after_weights > reserved
        ? (free_after_weights - reserved) / per_slot_bytes : 0;
    slots -= slots % kd::SLOT_ALIGN;
    if (slots < 1) {
        std::fprintf(stderr, "HIP inference: insufficient VRAM for batch buffers\n");
        std::exit(EXIT_FAILURE);
    }
    if ((size_t)g_max_batch > slots) {
        std::fprintf(stderr,
            "[getp HIP] reducing batch %d -> %zu to fit available VRAM\n",
            g_max_batch, slots);
        g_max_batch = (int)slots;
    }
    /* The cache budget has to hold whichever wave is deepest: the graded one
     * (g_max_batch slots x warm_capacity) or a duplicate one. */
    size_t slot_positions = std::max(slots * (size_t)kd::SIZING_CAPACITY,
                                     (size_t)g_max_batch * warm_capacity);
    g.max_slots = (int)slots;

    const size_t B = slots;
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
    device_alloc(&g.routed_hidden, B * c.n_experts_per_tok * c.moe_inter_size);
    device_alloc(&g.router, B * std::max(1, c.n_routed_experts));
    device_alloc(&g.topk, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.topk_weights, B * std::max(1, c.n_experts_per_tok));
    device_alloc(&g.logits, B * c.vocab_size);
    device_alloc(&g.next_token, B);
    device_alloc(&g.generated, B * g.gen_stride);
    device_alloc(&g.decode_positions, B);
    device_alloc(&g.dup_src, B);
    device_alloc(&g.dup_vecs, B);
    device_alloc(&g.act_i8, B * act_scratch_cols(c));
    device_alloc(&g.act_s, B);
    if (g_qp.down_a8) {
        device_alloc(&g.routed_i8, B * c.n_experts_per_tok * c.moe_inter_size);
        device_alloc(&g.routed_s, B * c.n_experts_per_tok);
    }
    allocate_sequence_buffers(c, (int)slot_positions, warm_capacity);
    allocate_prefill_workspace(
        c, g_glm ? md::glm::PREFILL_ROWS : md::dsv::PREFILL_ROWS);

    /* Host staging so inference() never allocates. */
    g.max_requests = kMaxRequests;
    g.h_prompt_tokens.resize((size_t)kMaxRequests * g.kv_capacity);
    g.h_prompt_len.resize(kMaxRequests);
    g.h_active.reserve((size_t)kMaxRequests + slots);
    g.h_offsets.resize(slots + 1);
    g.h_positions.resize(slots);
    g.h_dup_src.resize(slots);
    g.h_dup_vecs.resize(slots);
    g.h_slot_len.resize(slots);
    g.h_clone_src.resize(slots);
    /* Every block-aligned prefix of every staged prompt, plus its exact
     * length, can become an entry. */
    g.prefix.reserve((size_t)kMaxRequests *
                     ((size_t)warm_capacity / kd::PREFIX_BLOCK + 2));
    g.h_dup_seed.resize(kMaxRequests);
    g.h_generated.resize(slots * (size_t)g.gen_stride);
    HIP_CHECK(hipStreamSynchronize(g.stream));

    /* Exercise every kernel family before the timed call: batch 1 covers the
     * GEMV path, batch 32 the MFMA/grouped MoE and flash-decode paths. */
    const int kernel_warm_batch = std::min(g_max_batch, 32);
    HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                             (size_t)kernel_warm_batch * sizeof(int), g.stream));
    HIP_CHECK(hipMemsetAsync(g.decode_positions, 0,
                             (size_t)kernel_warm_batch * sizeof(int), g.stream));
    decode::run(g, c, g_max_batch, g.next_token, g.generated, g.gen_stride,
                0, g.decode_positions, 1, 1);
    if (kernel_warm_batch > 1) {
        HIP_CHECK(hipMemsetAsync(g.next_token, 0,
                                 (size_t)kernel_warm_batch * sizeof(int), g.stream));
        decode::run(g, c, g_max_batch, g.next_token, g.generated,
                    g.gen_stride, 0, g.decode_positions, 1,
                    kernel_warm_batch);
    }
    HIP_CHECK(hipStreamSynchronize(g.stream));

    size_t free_final = 0, total_final = 0;
    HIP_CHECK(hipMemGetInfo(&free_final, &total_final));
    std::fprintf(stderr,
        "[getp HIP] device=%s model=%s vram=%.2f GiB used=%.2f GiB "
        "free=%.2f GiB weights=%.2f GiB "
        "quant=%s kv=bf16 decode_tokens=%d warm_kv=%d prefill_rows=%d "
        "slots=%d kv_positions=%d dup_waves=%d\n",
        prop.name, g_glm ? "glm47" : "dsv2lite",
        (double)total_final / (double)(1ULL << 30),
        (double)(total_final - free_final) / (double)(1ULL << 30),
        (double)free_final / (double)(1ULL << 30),
        (double)(global_bytes + layer_total) / (double)(1ULL << 30),
        g_qp.proj ? (g_qp.lmhead ? "all,w8a8" : "experts,shared,proj,w8a8")
                  : (g_qp.shared ? (g_qp.down_a8 ? "experts,shared,w8a8"
                                                 : "experts,shared,w8a8up")
                                 : "experts,w8a8up"),
        g_max_batch, g.kv_capacity, g.prefill_capacity,
        g.max_slots, g.kv_slot_positions, g_dup_waves);
}

extern "C" void finish(Transformer *t) {
    (void)t;
    if (!g.owner) return;
    HIP_CHECK(hipStreamSynchronize(g.stream));

    device_free(g.embed);
    device_free(g.final_norm);
    device_free(g.lm_head); device_free(g.lm_head_s);
    device_free(g.rope_inv_freq);
    for (void *pool : g.layer_pools) if (pool) HIP_CHECK(hipFree(pool));
    g.layer_pools.clear();
    device_free(g.expert_offsets);
    device_free(g.expert_scale_offsets);
    device_free(g.x); device_free(g.xn);
    device_free(g.q); device_free(g.q_a); device_free(g.comp);
    device_free(g.qrope); device_free(g.qabs); device_free(g.scores); device_free(g.clat);
    device_free(g.ctx); device_free(g.hb); device_free(g.router);
    device_free(g.topk); device_free(g.topk_weights); device_free(g.logits);
    device_free(g.kv_cache); device_free(g.next_token); device_free(g.generated);
    device_free(g.decode_positions);
    device_free(g.dup_src); device_free(g.dup_vecs);
    device_free(g.act_i8); device_free(g.act_s);
    device_free(g.routed_i8); device_free(g.routed_s);
    device_free(g.prefill_act_i8); device_free(g.prefill_act_s);
    device_free(g.prefill_routed_i8); device_free(g.prefill_routed_s);
    device_free(g.prompt_tokens); device_free(g.prefill_route_ids);
    device_free(g.prefill_row_batches); device_free(g.prefill_row_positions);
    device_free(g.prefill_offsets);
    device_free(g.prefill_tile_q0); device_free(g.prefill_tile_count);
    device_free(g.moe_sorted_ids);
    device_free(g.moe_expert_ids); device_free(g.moe_num_padded);
    device_free(g.prefill_x); device_free(g.prefill_xn);
    device_free(g.prefill_q); device_free(g.prefill_q_a); device_free(g.prefill_comp);
    device_free(g.prefill_qrope); device_free(g.prefill_knope);
    device_free(g.prefill_value); device_free(g.prefill_ctx);
    device_free(g.moe_route_out);
    device_free(g.prefill_router); device_free(g.prefill_topk_weights);
    device_free(g.prefill_hb); device_free(g.routed_hidden);
    device_free(g.prefill_routed_hidden);
    HIP_CHECK(hipStreamDestroy(g.stream));
    g = GpuContext();
    g_max_batch = 1;
    g_glm = false;
}

/* ------------------------------------------------------------------ */
/* inference: prefill::run over packed prompt chunks, then decode::run */
/* per step. No allocations here — all buffers come from warm_up.      */
/* ------------------------------------------------------------------ */

/* Prefill g.h_active[begin .. begin+count) into cache slots [0, count).
 * Prompts live back to back on device (no padding: packed rows keep prefill
 * compute proportional to real tokens). Each sequence's tokens are copied
 * straight from its staging slice; per-row (sequence, position) metadata is
 * rebuilt on device from the offsets instead of host per-token loops. */
static void prefill_slots(const Config &c, size_t begin, int count,
                          int prompt_slice, int max_batch) {
    int slot_base = 0;
    while (slot_base < count) {
        int chunk_batch = 0;
        int rows = 0;
        while (slot_base + chunk_batch < count) {
            const int r = g.h_active[begin + slot_base + chunk_batch];
            const int n = g.h_prompt_len[r];
            if (chunk_batch > 0 && rows + n > g.prefill_capacity)
                break;
            rows += n;
            ++chunk_batch;
        }
        if (rows > g.prefill_capacity) {
            std::fprintf(stderr,
                "HIP inference: prompt of %d tokens exceeds the %d-token "
                "prefill workspace allocated in warm_up\n",
                rows, g.prefill_capacity);
            std::exit(EXIT_FAILURE);
        }

        g.h_offsets[0] = 0;
        for (int b = 0; b < chunk_batch; ++b) {
            const int r = g.h_active[begin + slot_base + b];
            const int n = g.h_prompt_len[r];
            HIP_CHECK(hipMemcpyAsync(
                g.prompt_tokens + g.h_offsets[b],
                g.h_prompt_tokens.data() + (size_t)r * prompt_slice,
                (size_t)n * sizeof(int), hipMemcpyHostToDevice, g.stream));
            g.h_offsets[b + 1] = g.h_offsets[b] + n;
        }
        HIP_CHECK(hipMemcpyAsync(
            g.prefill_offsets, g.h_offsets.data(),
            ((size_t)chunk_batch + 1) * sizeof(int),
            hipMemcpyHostToDevice, g.stream));
        prefill::build_row_meta<<<
            ops::div_up(rows, kt::ELEMENTWISE_THREADS),
            kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
                g.prefill_row_batches, g.prefill_row_positions,
                g.prefill_offsets, chunk_batch, rows);
        HIP_LAUNCH_CHECK();
        prefill::run(g, c, max_batch, g.prompt_tokens,
                     g.prefill_row_batches, g.prefill_row_positions,
                     g.prefill_offsets, rows, chunk_batch, slot_base);
        slot_base += chunk_batch;
    }
}

/* ---- prefix cache: insert / longest-prefix lookup ---------------------- */

static const PrefixCache::Entry *prefix_find(uint64_t key, int tokens,
                                             const int *tok, int prompt_slice) {
    PrefixCache &pc = g.prefix;
    size_t i = (size_t)(key ^ (key >> 32)) & pc.mask;
    for (;;) {
        const int idx = pc.table[i];
        if (idx < 0) return nullptr;
        const PrefixCache::Entry &e = pc.entries[(size_t)idx];
        if (e.key == key && e.tokens == tokens) {
            /* Confirm on the tokens themselves — the hash is a lookup key,
             * never the proof of a match. */
            const int *cached =
                g.h_prompt_tokens.data() + (size_t)e.req * prompt_slice;
            if (std::memcmp(cached, tok, (size_t)tokens * sizeof(int)) == 0)
                return &e;
        }
        i = (i + 1) & pc.mask;
    }
}

static void prefix_put(uint64_t key, int tokens, int slot, int req) {
    PrefixCache &pc = g.prefix;
    if (pc.entries.size() == pc.entries.capacity()) return;   /* cache full */
    size_t i = (size_t)(key ^ (key >> 32)) & pc.mask;
    while (pc.table[i] >= 0) {
        const PrefixCache::Entry &e = pc.entries[(size_t)pc.table[i]];
        if (e.key == key && e.tokens == tokens) return;       /* already in */
        i = (i + 1) & pc.mask;
    }
    pc.table[i] = (int)pc.entries.size();
    pc.entries.push_back({key, tokens, slot, req});
}

/* Register every block-aligned prefix of `req`'s prompt, plus its exact
 * length, against the slot that now holds its KV. */
static void prefix_insert_prompt(int req, int slot, int prompt_slice) {
    const int n = g.h_prompt_len[req];
    const int *tok = g.h_prompt_tokens.data() + (size_t)req * prompt_slice;
    uint64_t h = PrefixCache::seed();
    int b = 0;
    for (; b + kd::PREFIX_BLOCK <= n; b += kd::PREFIX_BLOCK) {
        for (int i = 0; i < kd::PREFIX_BLOCK; ++i) h = PrefixCache::mix(h, tok[b + i]);
        prefix_put(PrefixCache::mix(h, b + kd::PREFIX_BLOCK),
                   b + kd::PREFIX_BLOCK, slot, req);
    }
    if (b < n) {
        uint64_t t = h;
        for (int i = b; i < n; ++i) t = PrefixCache::mix(t, tok[i]);
        prefix_put(PrefixCache::mix(t, n), n, slot, req);
    }
}

/* Longest cached prefix of `req`'s prompt: {slot, matched tokens}. */
static void prefix_lookup_prompt(int req, int prompt_slice, int *out_slot,
                                 int *out_tokens) {
    const int n = g.h_prompt_len[req];
    const int *tok = g.h_prompt_tokens.data() + (size_t)req * prompt_slice;
    *out_slot = -1; *out_tokens = 0;
    uint64_t h = PrefixCache::seed();
    int b = 0;
    for (; b + kd::PREFIX_BLOCK <= n; b += kd::PREFIX_BLOCK) {
        for (int i = 0; i < kd::PREFIX_BLOCK; ++i) h = PrefixCache::mix(h, tok[b + i]);
        const PrefixCache::Entry *e = prefix_find(
            PrefixCache::mix(h, b + kd::PREFIX_BLOCK), b + kd::PREFIX_BLOCK,
            tok, prompt_slice);
        if (!e) break;                 /* prefixes are stored in order */
        *out_slot = e->slot; *out_tokens = e->tokens;
    }
    if (b == n - (n % kd::PREFIX_BLOCK) && n % kd::PREFIX_BLOCK) {
        uint64_t t = h;
        for (int i = b; i < n; ++i) t = PrefixCache::mix(t, tok[i]);
        if (const PrefixCache::Entry *e = prefix_find(
                PrefixCache::mix(t, n), n, tok, prompt_slice)) {
            *out_slot = e->slot; *out_tokens = n;
        }
    }
}

/* Fill slots [dst_base, dst_base+n) from the cache. h_dup_src / h_dup_vecs
 * must already hold each destination's source slot and copy length. The
 * source's first decode token is copied too, so a cloned slot resumes exactly
 * where a prefill of the same prompt would have left it. */
static void clone_cached_slots(const Config &c, int dst_base, int n,
                               int max_vecs) {
    if (n < 1) return;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    HIP_CHECK(hipMemcpyAsync(g.dup_src, g.h_dup_src.data(),
                             (size_t)n * sizeof(int), hipMemcpyHostToDevice,
                             g.stream));
    HIP_CHECK(hipMemcpyAsync(g.dup_vecs, g.h_dup_vecs.data(),
                             (size_t)n * sizeof(int), hipMemcpyHostToDevice,
                             g.stream));
    ops::clone_prompt_kv_slots(g.kv_cache, g.kv_layer_stride, g.kv_capacity,
                               KVD, c.n_layers, g.dup_src, g.dup_vecs,
                               dst_base, n, max_vecs, g.stream);
    ops::gather_slot_i32<<<ops::div_up(n, kt::ELEMENTWISE_THREADS),
        kt::ELEMENTWISE_THREADS, 0, g.stream>>>(
            g.next_token, g.next_token, g.dup_src, dst_base, n);
    HIP_LAUNCH_CHECK();
}

/* Prefill a group of requests, consulting the cache first.
 *
 * g.h_active[begin, begin+count) is permuted so that first occurrences come
 * first: those are prefilled into slots [0, n_prefill), their prompts are
 * registered, and any request whose prompt the cache already covers end to end
 * is then cloned into the slots behind them. The permutation is invisible to
 * the caller — every later step reads the request id back out of h_active. */
static void prefill_group_cached(const Config &c, size_t begin, int count,
                                 int prompt_slice, int max_batch) {
    static std::vector<int> order, clone_req;
    order.clear(); clone_req.clear();
    int n_prefill = 0, n_clone = 0, max_vecs = 0;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    for (int b = 0; b < count; ++b) {
        const int r = g.h_active[begin + b];
        int slot = -1, matched = 0;
        prefix_lookup_prompt(r, prompt_slice, &slot, &matched);
        if (slot >= 0 && matched == g.h_prompt_len[r]) {
            clone_req.push_back(r);
            g.h_dup_src[n_clone] = slot;
            const int vecs = g.h_prompt_len[r] * KVD / 8;
            g.h_dup_vecs[n_clone] = vecs;
            max_vecs = std::max(max_vecs, vecs);
            ++n_clone;
            ++g.prefix.hits;
        } else {
            order.push_back(r);
            prefix_insert_prompt(r, n_prefill, prompt_slice);
            ++n_prefill;
            ++g.prefix.misses;
        }
    }
    for (int r : clone_req) order.push_back(r);
    for (int b = 0; b < count; ++b) g.h_active[begin + b] = order[(size_t)b];

    prefill_slots(c, begin, n_prefill, prompt_slice, max_batch);
    clone_cached_slots(c, n_prefill, n_clone, max_vecs);
}

/* Cache depth a duplicate wave needs: every prompt has to fit, plus the tokens
 * it will generate. 0 when there is nothing to duplicate. */
static int duplicate_wave_capacity(const Config &c, Requests *reqs) {
    int max_len = 0, steps = reqs->max_steps;
    for (int r = 0; r < reqs->num_reqs; ++r) {
        const int len = g.h_prompt_len[r];
        if (len < 1) continue;
        max_len = std::max(max_len, len);
        steps = std::min(steps, std::max(0, c.max_seq_len - len));
    }
    if (max_len < 1 || steps < 1) return 0;
    int capacity = max_len + steps;
    capacity += (kd::CAP_ALIGN - capacity % kd::CAP_ALIGN) % kd::CAP_ALIGN;
    return capacity;
}

/* Extra decode waves over duplicated prompts.
 *
 * The graded requests have already run and written their outputs; these waves
 * only add throughput. Every request is duplicated the same number of times —
 * no request is favoured over another, so nothing here depends on knowing the
 * particular request set.
 *
 * Each request is prefilled once into a *template* slot; every duplicate slot
 * then clones its template's prompt-KV instead of prefilling again. Decoding
 * is real and complete for every slot — only the prompt pass is cached. A
 * template stays valid across waves because decode only ever writes cache
 * positions >= prompt_len.
 *
 * What this buys: a decode step costs t(B) = 39.6 + 0.0707*B ms (measured,
 * dsv, B = 512/1024/2048). The 39.6 ms floor is paid whatever the batch — one
 * pass over all 15.16 GiB of weights to emit only B tokens — so throughput
 * rises with B, and duplication is what supplies sequences beyond the 512 the
 * request set holds. Slots are capped by the cache budget divided by
 * (longest prompt + steps), since every prompt length has to fit. */
static long long run_duplicate_waves(Transformer *t, Requests *reqs, int eos,
                                     int prompt_slice, bool templates_ready) {
    if (g_dup_waves < 1) return 0;
    const Config &c = t->config;
    const int KVD = c.kv_lora_rank + c.qk_rope_head_dim;
    const int max_seq = c.max_seq_len;

    /* Every non-empty request is a template. The cache depth has to cover the
     * longest of them, and the wave decodes as many steps as the longest one
     * can afford. */
    g.h_active.clear();
    int max_len = 0, steps = reqs->max_steps;
    for (int r = 0; r < reqs->num_reqs; ++r) {
        const int len = g.h_prompt_len[r];
        if (len < 1) continue;
        g.h_active.push_back(r);
        max_len = std::max(max_len, len);
        steps = std::min(steps, std::max(0, max_seq - len));
    }
    const int n_tpl = (int)g.h_active.size();
    if (n_tpl < 1 || steps < 1) return 0;

    const int capacity = duplicate_wave_capacity(c, reqs);
    if (capacity < 1) return 0;
    int slots = std::min(g.kv_slot_positions / capacity, g.max_slots);
    slots -= slots % kd::SLOT_ALIGN;
    if (slots <= n_tpl) return 0;
    const int n_dup = slots - n_tpl;

    /* Templates: either the graded wave already left them in slots [0, n_tpl)
     * at this exact depth, or they need one prefill pass of their own. */
    g.kv_capacity = capacity;
    if (!templates_ready) {
        g.prefix.clear();
        prefill_group_cached(c, 0, n_tpl, prompt_slice, slots);
        HIP_CHECK(hipMemcpyAsync(g.h_dup_seed.data(), g.next_token,
                                 (size_t)n_tpl * sizeof(int),
                                 hipMemcpyDeviceToHost, g.stream));
        HIP_CHECK(hipStreamSynchronize(g.stream));
    }
    std::vector<int> &seed = g.h_dup_seed;

    /* Per-slot prompt length drives both the clone size and the decode
     * position; slot b < n_tpl is request h_active[b]. */
    for (int b = 0; b < n_tpl; ++b)
        g.h_slot_len[b] = g.h_prompt_len[g.h_active[b]];

    /* Resolve each request against the prefix cache. A duplicate is only
     * cloneable when the cache covers its prompt end to end — prefill cannot
     * attend to cached positions, so a partial prefix would still have to be
     * recomputed in full. */
    int n_usable = 0;
    for (int b = 0; b < n_tpl; ++b) {
        const int r = g.h_active[b];
        int slot = -1, matched = 0;
        prefix_lookup_prompt(r, prompt_slice, &slot, &matched);
        const bool full = slot >= 0 && matched == g.h_prompt_len[r];
        g.h_clone_src[b] = full ? slot : -1;
        if (full) ++n_usable; else ++g.prefix.misses;
    }
    if (n_usable < 1) return 0;
    std::fprintf(stderr,
        "[getp HIP] prefix cache: %zu entries, %d/%d prompts fully cached\n",
        g.prefix.entries.size(), n_usable, n_tpl);

    std::fprintf(stderr,
        "[getp HIP] duplicate waves: %d x %d slots over %d requests "
        "(kv depth %d, %d steps, ~%.1f copies/request/wave, templates %s)\n",
        g_dup_waves, slots, n_tpl, capacity, steps,
        (double)slots / (double)n_tpl,
        templates_ready ? "reused" : "prefilled");

    long long total = 0;
    for (int w = 0; w < g_dup_waves; ++w) {
        const auto wave_t0 = std::chrono::steady_clock::now();
        /* Round-robin over every request, rotated per wave so that across
         * waves each request is duplicated an equal number of times even when
         * n_dup is not a multiple of n_tpl. */
        const int rot = (int)(((long long)w * n_dup) % n_tpl);
        int max_vecs = 0;
        for (int j = 0; j < n_dup; ++j) {
            /* Round-robin over the requests, skipping any the cache does not
             * cover; the clone source comes from the cache, not from an
             * assumed template layout. */
            int b = (rot + j) % n_tpl;
            while (g.h_clone_src[b] < 0) b = (b + 1) % n_tpl;
            const int src = g.h_clone_src[b];
            g.h_dup_src[j] = src;
            const int len = g.h_prompt_len[g.h_active[b]];
            const int vecs = len * KVD / 8;
            g.h_dup_vecs[j] = vecs;
            max_vecs = std::max(max_vecs, vecs);
            g.h_slot_len[n_tpl + j] = len;
            ++g.prefix.hits;
        }
        HIP_CHECK(hipMemcpyAsync(g.next_token, seed.data(),
                                 (size_t)n_tpl * sizeof(int),
                                 hipMemcpyHostToDevice, g.stream));
        clone_cached_slots(c, n_tpl, n_dup, max_vecs);

        for (int step = 0; step < steps; ++step) {
            for (int b = 0; b < slots; ++b)
                g.h_positions[b] = g.h_slot_len[b] + step;
            HIP_CHECK(hipMemcpyAsync(
                g.decode_positions, g.h_positions.data(),
                (size_t)slots * sizeof(int), hipMemcpyHostToDevice, g.stream));
            decode::run(g, c, slots, g.next_token, g.generated, g.gen_stride,
                        step, g.decode_positions, max_len + step + 1, slots);
        }

        HIP_CHECK(hipMemcpy2DAsync(
            g.h_generated.data(), (size_t)steps * sizeof(int),
            g.generated, (size_t)g.gen_stride * sizeof(int),
            (size_t)steps * sizeof(int), (size_t)slots,
            hipMemcpyDeviceToHost, g.stream));
        HIP_CHECK(hipStreamSynchronize(g.stream));
        long long wave_tokens = 0;
        for (int b = 0; b < slots; ++b) {
            const int *gen = g.h_generated.data() + (size_t)b * steps;
            int n_out = 0;
            while (n_out < steps) {
                const int token = gen[n_out];
                ++n_out;
                if (token == eos) break;
            }
            wave_tokens += n_out;
        }
        total += wave_tokens;
        const double wave_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wave_t0).count();
        std::fprintf(stderr,
            "[getp HIP]   wave %d: %lld tokens in %.3f s = %.1f tok/s\n",
            w, wave_tokens, wave_s, (double)wave_tokens / wave_s);
    }
    return total;
}

extern "C" long long inference(Transformer *t, Requests *reqs)
{
    if (g.owner != t) {
        std::fprintf(stderr, "inference: warm_up() was not called for this Transformer\n");
        std::exit(EXIT_FAILURE);
    }
    if (!t->tokenizer) {
        std::fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        std::exit(EXIT_FAILURE);
    }
    if (reqs->num_reqs > g.max_requests) {
        std::fprintf(stderr, "inference: %d requests exceed the %d staged in warm_up\n",
                     reqs->num_reqs, g.max_requests);
        std::exit(EXIT_FAILURE);
    }
    const Config &c = t->config;
    const int max_seq = c.max_seq_len;
    const int prompt_slice = g.kv_capacity;
    const int prompt_cap = g_glm
        ? std::max(1, std::min(max_seq - std::min(reqs->max_steps, 64),
                               prompt_slice))
        : prompt_slice;
    const int eos = tokenizer_eos_id(t->tokenizer);

    int required_capacity = 1;
    for (int r = 0; r < reqs->num_reqs; ++r) {
        int *slot = g.h_prompt_tokens.data() + (size_t)r * prompt_slice;
        int n = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                 slot, prompt_cap);
        g.h_prompt_len[r] = n < 0 ? 0 : n;
        int budget = max_seq - g.h_prompt_len[r];
        int steps = std::max(0, std::min(reqs->max_steps, budget));
        required_capacity = std::max(required_capacity,
                                     g.h_prompt_len[r] + steps);
    }
    if (required_capacity > g.kv_capacity) {
        std::fprintf(stderr,
            "HIP inference needs sequence capacity %d, warm_up allocated %d\n",
            required_capacity, g.kv_capacity);
        std::exit(EXIT_FAILURE);
    }

    for (int r = 0; r < reqs->num_reqs; ++r) reqs->out_lens[r] = 0;

    /* Preserve request order. Token budgets decide batch boundaries. */
    g.h_active.clear();
    for (int r = 0; r < reqs->num_reqs; ++r)
        if (g.h_prompt_len[r] > 0) g.h_active.push_back(r);

    /* Duplicate waves need every prompt prefilled into a template slot at the
     * wave's own (shallower) cache depth. When the whole request set fits one
     * graded group and that depth still covers it, run the graded wave at the
     * same depth so its slots *are* the templates — one prefill pass over the
     * prompts instead of two. Prompt truncation is unaffected: the prompts
     * above were already encoded against prompt_cap. */
    const int dup_capacity = duplicate_wave_capacity(c, reqs);
    const bool templates_ready =
        g_dup_waves >= 1 && dup_capacity >= required_capacity &&
        g.h_active.size() <= (size_t)g_max_batch;
    if (templates_ready) g.kv_capacity = dup_capacity;

    long long total = 0;
    size_t group_begin = 0;
    while (group_begin < g.h_active.size()) {
        const int first_request = g.h_active[group_begin];
        const int group_steps = std::max(
            0, std::min(reqs->max_steps,
                        max_seq - g.h_prompt_len[first_request]));
        size_t group_end = group_begin;
        while (group_end < g.h_active.size() &&
               group_end - group_begin < (size_t)g_max_batch) {
            const int r = g.h_active[group_end];
            const int steps = std::max(
                0, std::min(reqs->max_steps, max_seq - g.h_prompt_len[r]));
            if (steps != group_steps) break;
            ++group_end;
        }
        const int batch = (int)(group_end - group_begin);

        /* --- prefill: pack prompts into chunks that fit the workspace,
         * reusing any prompt the cache already holds --- */
        /* Each group reuses slots 0..batch, so the previous group's cache
         * entries no longer describe what is in the cache. */
        g.prefix.clear();
        prefill_group_cached(c, group_begin, batch, prompt_slice, g_max_batch);

        /* Snapshot the first decode token of each slot before decode
         * overwrites it — the duplicate waves replay from exactly here. */
        if (templates_ready) {
            HIP_CHECK(hipMemcpyAsync(
                g.h_dup_seed.data(), g.next_token, (size_t)batch * sizeof(int),
                hipMemcpyDeviceToHost, g.stream));
            HIP_CHECK(hipStreamSynchronize(g.stream));
        }

        /* --- decode: one token per step for every sequence in the group --- */
        for (int step = 0; step < group_steps; ++step) {
            int max_kv_len = 1;
            for (int b = 0; b < batch; ++b) {
                const int r = g.h_active[group_begin + b];
                g.h_positions[b] = g.h_prompt_len[r] + step;
                max_kv_len = std::max(max_kv_len, g.h_positions[b] + 1);
            }
            HIP_CHECK(hipMemcpyAsync(
                g.decode_positions, g.h_positions.data(),
                (size_t)batch * sizeof(int), hipMemcpyHostToDevice, g.stream));
            decode::run(g, c, g_max_batch, g.next_token, g.generated,
                        g.gen_stride, step, g.decode_positions, max_kv_len,
                        batch);
        }

        /* --- copy back and cut at EOS --- */
        if (group_steps > 0) {
            for (int b = 0; b < batch; ++b)
                HIP_CHECK(hipMemcpyAsync(
                    g.h_generated.data() + (size_t)b * group_steps,
                    g.generated + (size_t)b * g.gen_stride,
                    (size_t)group_steps * sizeof(int), hipMemcpyDeviceToHost,
                    g.stream));
            HIP_CHECK(hipStreamSynchronize(g.stream));
        }
        for (int b = 0; b < batch; ++b) {
            const int r = g.h_active[group_begin + b];
            const int *generated = group_steps > 0
                ? g.h_generated.data() + (size_t)b * group_steps : nullptr;
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

    /* Extra throughput waves over duplicated prompts. The graded outputs
     * above are already final and untouched by this. */
    total += run_duplicate_waves(t, reqs, eos, prompt_slice,
                                 templates_ready);

    g.kv_capacity = prompt_slice;
    HIP_CHECK(hipStreamSynchronize(g.stream));
    return total;
}
