/* Device weight/activation state shared by getp_run.hip, prefill.h and
 * decode.h. Every buffer (device AND host staging) is allocated in warm_up;
 * inference() only reuses it. */
#ifndef MLA_GETP_STATE_H
#define MLA_GETP_STATE_H

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "getp.h" /* Transformer, bf16_t (via model.h -> tensor.h) */

struct CopyChunk
{
    const void *host;
    size_t offset;
    size_t bytes;
};

/* Weight tensor to quantize to per-row int8 during warm_up: the bf16 source
 * is staged to the device, then ops::quantize_rows writes int8 data + fp32
 * scales into the layer pool at the recorded offsets. */
struct QuantChunk
{
    const void *host;
    size_t data_offset;
    size_t scale_offset;
    int rows;
    int cols;
};

static const size_t NO_OFFSET = std::numeric_limits<size_t>::max();

struct LayerLayout {
    size_t bytes = 0;
    std::vector<CopyChunk> chunks;
    std::vector<QuantChunk> qchunks;
    size_t q_proj = NO_OFFSET, q_a_proj = NO_OFFSET, q_b_proj = NO_OFFSET;
    size_t kv_a_proj = NO_OFFSET, kv_b_proj = NO_OFFSET, wuk_t = NO_OFFSET;
    size_t o_proj = NO_OFFSET;
    size_t input_norm = NO_OFFSET, post_norm = NO_OFFSET;
    size_t q_a_norm = NO_OFFSET, kv_a_norm = NO_OFFSET;
    size_t dense_gate = NO_OFFSET, dense_up = NO_OFFSET, dense_down = NO_OFFSET;
    size_t moe_gate = NO_OFFSET, moe_bias = NO_OFFSET;
    size_t shared_gate = NO_OFFSET, shared_up = NO_OFFSET, shared_down = NO_OFFSET;
    /* Per-row int8 scale offsets; NO_OFFSET when that tensor stays bf16. */
    size_t q_proj_s = NO_OFFSET, q_a_proj_s = NO_OFFSET, q_b_proj_s = NO_OFFSET;
    size_t kv_a_proj_s = NO_OFFSET, o_proj_s = NO_OFFSET;
    size_t dense_gate_s = NO_OFFSET, dense_up_s = NO_OFFSET, dense_down_s = NO_OFFSET;
    size_t shared_gate_s = NO_OFFSET, shared_up_s = NO_OFFSET, shared_down_s = NO_OFFSET;
    std::vector<size_t> expert_gate, expert_up, expert_down;
    std::vector<size_t> expert_gate_s, expert_up_s, expert_down_s;
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
    /* Non-null only for tensors quantized to int8 (then the matching weight
     * pointer above actually holds int8 data). */
    const float *q_proj_s = nullptr, *q_a_proj_s = nullptr, *q_b_proj_s = nullptr;
    const float *kv_a_proj_s = nullptr, *o_proj_s = nullptr;
    const float *dense_gate_s = nullptr, *dense_up_s = nullptr, *dense_down_s = nullptr;
    const float *shared_gate_s = nullptr, *shared_up_s = nullptr, *shared_down_s = nullptr;
    const size_t *expert_gate_offsets = nullptr;
    const size_t *expert_up_offsets = nullptr;
    const size_t *expert_down_offsets = nullptr;
    const size_t *expert_gate_scale_offsets = nullptr;
    const size_t *expert_up_scale_offsets = nullptr;
    const size_t *expert_down_scale_offsets = nullptr;
};

struct GpuContext {
    Transformer *owner = nullptr;
    hipStream_t stream = nullptr;
    bf16_t *embed = nullptr, *final_norm = nullptr, *lm_head = nullptr;
    float *lm_head_s = nullptr; /* non-null when lm_head is int8 */
    float *rope_inv_freq = nullptr;
    std::vector<LayerLayout> layouts;
    std::vector<std::vector<bf16_t>> host_wuk_t;
    std::vector<DeviceLayer> layers;
    std::vector<void *> layer_pools;
    size_t *expert_offsets = nullptr;
    size_t *expert_scale_offsets = nullptr;

    float *x = nullptr, *xn = nullptr;
    float *q = nullptr, *q_a = nullptr, *comp = nullptr;
    float *qrope = nullptr, *qabs = nullptr, *scores = nullptr, *clat = nullptr;
    float *ctx = nullptr, *hb = nullptr, *router = nullptr;
    float *logits = nullptr;
    bf16_t *kv_cache = nullptr;
    int *topk = nullptr, *next_token = nullptr, *generated = nullptr;
    int *decode_positions = nullptr;
    float *topk_weights = nullptr;
    int kv_capacity = 0;

    /* w8a8: dynamic per-token int8 activations. `routed_*` only exist when
     * the expert down projection also runs on the int8 MFMA path (DSV). */
    char *act_i8 = nullptr;
    float *act_s = nullptr;
    char *routed_i8 = nullptr;
    float *routed_s = nullptr;
    char *prefill_act_i8 = nullptr;
    float *prefill_act_s = nullptr;
    char *prefill_routed_i8 = nullptr;
    float *prefill_routed_s = nullptr;

    /* MoE grouped-path scratch, shared by prefill and decode. */
    int *moe_sorted_ids = nullptr;   /* expert-aligned route ids            */
    int *moe_expert_ids = nullptr;   /* expert per aligned route block      */
    int *moe_num_padded = nullptr;   /* single int: total padded routes     */
    float *moe_route_out = nullptr;  /* per-route down-projection output    */

    /* Layer-major packed prefill workspace. */
    int *prompt_tokens = nullptr, *prefill_route_ids = nullptr;
    int *prefill_row_batches = nullptr, *prefill_row_positions = nullptr;
    int *prefill_offsets = nullptr;
    float *prefill_x = nullptr, *prefill_xn = nullptr;
    float *prefill_q = nullptr, *prefill_q_a = nullptr, *prefill_comp = nullptr;
    float *prefill_qrope = nullptr, *prefill_qabs = nullptr, *prefill_clat = nullptr;
    float *prefill_ctx = nullptr;
    float *prefill_router = nullptr, *prefill_topk_weights = nullptr;
    float *prefill_hb = nullptr, *routed_hidden = nullptr;
    bf16_t *prefill_routed_hidden = nullptr;
    int prefill_capacity = 0;

    /* Host staging, reserved once in warm_up so inference() never allocates. */
    std::vector<int> h_prompt_tokens;   /* [request][kv_capacity], flat      */
    std::vector<int> h_prompt_len;      /* tokens per request                */
    std::vector<int> h_active;          /* indices of non-empty requests     */
    std::vector<int> h_flat_tokens;     /* packed prefill chunk tokens       */
    std::vector<int> h_row_batches, h_row_positions;
    std::vector<int> h_offsets;         /* per-chunk prompt offsets          */
    std::vector<int> h_positions;       /* decode positions per sequence     */
    std::vector<int> h_generated;       /* copied-back generated tokens      */
    int max_requests = 0;
};

#endif /* MLA_GETP_STATE_H */
