/* Device-side weight/activation state shared by getp_run.hip and the kernel
 * headers under src/kernels/. Extracted from getp_run.hip so kernel wrappers
 * (e.g. prefill::run) can take the context by reference instead of reaching
 * for the translation-unit global. */
#ifndef MLA_GETP_STATE_H
#define MLA_GETP_STATE_H

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "getp.h"   /* Transformer, bf16_t (via model.h -> tensor.h) */

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

    float *xs = nullptr; // [VOC, H]
};

#endif /* MLA_GETP_STATE_H */
