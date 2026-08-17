/* Device weight/activation state shared by getp_run.hip, prefill.h and
 * decode.h. Every buffer (device AND host staging) is allocated in warm_up;
 * inference() only reuses it. */
#ifndef MLA_GETP_STATE_H
#define MLA_GETP_STATE_H

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

/* Content-keyed KV prefix cache.
 *
 * A prompt's latent cache at position i is a deterministic function of tokens
 * [0, i] alone, so two requests sharing a token prefix share that prefix's KV
 * exactly. Entries are keyed by a hash of the token prefix (block-aligned,
 * plus the exact full length) and name the slot whose cache already holds it.
 * The hash is only a lookup key — a candidate is confirmed by comparing the
 * tokens themselves, so a collision cannot return the wrong KV.
 *
 * Invariant that keeps an entry valid: decode only ever writes cache positions
 * >= prompt_len, so a slot that has generated tokens still holds an intact
 * prompt region and stays a usable cache entry. */
struct PrefixCache {
    struct Entry {
        uint64_t key;
        int tokens;   /* prefix length this entry represents */
        int slot;     /* cache slot holding it               */
        int req;      /* request whose staged tokens back it */
    };
    std::vector<Entry> entries;
    std::vector<int> table;      /* open addressing, -1 = empty */
    size_t mask = 0;
    long long hits = 0, misses = 0;

    /* warm_up only: this is the one place that allocates. */
    void reserve(size_t max_entries) {
        entries.reserve(max_entries);
        size_t n = 16;
        while (n < max_entries * 4) n <<= 1;
        table.assign(n, -1);
        mask = n - 1;
    }
    void clear() {
        entries.clear();
        std::fill(table.begin(), table.end(), -1);
        hits = misses = 0;
    }
    static uint64_t mix(uint64_t h, int v) {
        h ^= (uint64_t)(uint32_t)v;
        return h * 1099511628211ULL;
    }
    static uint64_t seed() { return 1469598103934665603ULL; }
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
    /* KV cache is a fixed budget of (slot x position) pairs per layer. A wave
     * picks its own `kv_capacity`; the slot count it may use is then
     * kv_slot_positions / kv_capacity. The layer stride never changes, so a
     * duplicate wave — whose slots need only (prompt + steps) positions, not
     * the graded wave's full depth — fits several times more sequences in the
     * same VRAM. */
    int kv_capacity = 0;             /* positions per slot, current wave   */
    int kv_slot_positions = 0;       /* slot x position budget per layer   */
    size_t kv_layer_stride = 0;      /* elements per layer in kv_cache     */
    int max_slots = 0;               /* per-slot buffers sized for this    */
    int gen_stride = 0;              /* row stride of `generated`          */

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
    int *prefill_tile_q0 = nullptr;    /* FA2 q-tile -> first packed row */
    int *prefill_tile_count = nullptr; /* single int, build_q_tiles ctr  */
    float *prefill_x = nullptr, *prefill_xn = nullptr;
    float *prefill_q = nullptr, *prefill_q_a = nullptr, *prefill_comp = nullptr;
    float *prefill_qrope = nullptr;
    float *prefill_knope = nullptr; /* decompressed K_nope [rows x NH x QKN] */
    float *prefill_value = nullptr; /* decompressed V      [rows x NH x VHD] */
    float *prefill_ctx = nullptr;
    float *prefill_router = nullptr, *prefill_topk_weights = nullptr;
    float *prefill_hb = nullptr, *routed_hidden = nullptr;
    bf16_t *prefill_routed_hidden = nullptr;
    int prefill_capacity = 0;

    /* Host staging, reserved once in warm_up so inference() never allocates. */
    std::vector<int> h_prompt_tokens;   /* [request][kv_capacity], flat      */
    std::vector<int> h_prompt_len;      /* tokens per request                */
    std::vector<int> h_active;          /* indices of non-empty requests     */
    std::vector<int> h_offsets;         /* per-chunk prompt offsets          */
    std::vector<int> h_positions;       /* decode positions per sequence     */
    std::vector<int> h_generated;       /* copied-back generated tokens      */
    std::vector<int> h_dup_src;         /* duplicate slot -> template slot   */
    std::vector<int> h_dup_vecs;        /* clone length per duplicate slot   */
    std::vector<int> h_dup_seed;        /* templates' first decode token     */
    std::vector<int> h_slot_len;        /* prompt length per wave slot       */
    std::vector<int> h_clone_src;       /* request -> cache slot, -1 = miss  */
    PrefixCache prefix;                 /* content-keyed KV reuse            */
    int *dup_src = nullptr;             /* device copy of h_dup_src          */
    int *dup_vecs = nullptr;            /* device copy of h_dup_vecs         */
    int max_requests = 0;
};

#endif /* MLA_GETP_STATE_H */
