/* ============================================================================
 * THIS IS THE FILE YOU MODIFY (a HIP/C++ translation unit).
 *
 * Your task: maximize end-to-end throughput (tok/s) of inference() over the
 * request batch, WITHOUT breaking correctness (`make eval` must still pass its
 * thresholds). The intended path is to port the compute to the GPU: start from
 * this correct CPU reference, then replace the forward_* calls with your own
 * kernels — allocate/upload in warm_up(), free in finish().
 *
 * Write __global__ kernels here (note the trailing double underscores) and launch
 * them with <<<>>> directly; expose inference()/warm_up()/finish() as usual — they
 * link to the C harness via the extern "C" guards in getp.h. To split kernels
 * across files, add more .hip files under src/kernels/ (picked up automatically,
 * no Makefile edit). Everything else is frozen.
 *
 * The frozen harness (src/getp_eval.c) times inference() and prints your score.
 * The frozen reference kernels are declared in include/engine.h.
 * ==========================================================================*/
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "getp.h"       /* Requests + the warm_up/finish/inference contract */
#include "engine.h"     /* forward_unabsorbed / forward_absorbed / sample */
#include "tokenizer.h"  /* tokenizer_encode / tokenizer_eos_id */
#include "kernels/kernels_utils.h"
#include "kernels/prefill_kernels.h"

namespace opt {

    struct DeviceWeights {
        __hip_bfloat16 *embed_tokens;     /* [vocab_size, hidden_size] */
        __hip_bfloat16 *norm;             /* [hidden_size] */
        __hip_bfloat16 *lm_head;          /* [vocab_size, hidden_size] */
        float          *rope_inv_freq;    /* [qk_rope_head_dim/2] F32                      dùng chung mọi layer */

        __hip_bfloat16 *q_proj;           /* [n_layers, n_heads*qk_head_dim, hidden_size]  q_lora_rank == 0, else NULL */
        __hip_bfloat16 *q_a_proj;         /* [n_layers, q_lora_rank, hidden_size]          q_lora_rank  > 0, else NULL */
        __hip_bfloat16 *q_b_proj;         /* [n_layers, n_heads*qk_head_dim, q_lora_rank]  q_lora_rank  > 0, else NULL */
        __hip_bfloat16 *kv_a_proj;        /* [n_layers, kv_lora_rank+qk_rope_head_dim, hidden_size] */
        __hip_bfloat16 *W_UK;             /* [n_layers, n_heads, qk_nope_head_dim, kv_lora_rank]  head-interleaved, stride (QKN+VHD)*KVL */
        __hip_bfloat16 *W_UV;             /* [n_layers, n_heads, v_head_dim,       kv_lora_rank]  = W_UK + QKN*KVL, cùng stride */
        __hip_bfloat16 *o_proj;           /* [n_layers, hidden_size, n_heads*v_head_dim] */

        __hip_bfloat16 *input_layernorm;  /* [n_layers, hidden_size] */
        __hip_bfloat16 *post_attn_norm;   /* [n_layers, hidden_size] */
        __hip_bfloat16 *q_a_layernorm;    /* [n_layers, q_lora_rank]   q_lora_rank > 0, else NULL */
        __hip_bfloat16 *kv_a_layernorm;   /* [n_layers, kv_lora_rank] */

        __hip_bfloat16 *dense_gate;       /* [first_k_dense, dense_inter_size, hidden_size]  index l */
        __hip_bfloat16 *dense_up;         /* [first_k_dense, dense_inter_size, hidden_size]  index l */
        __hip_bfloat16 *dense_down;       /* [first_k_dense, hidden_size, dense_inter_size]  index l */

        __hip_bfloat16 *moe_gate;         /* [n_moe_layers, n_routed_experts, hidden_size]   index m */
        float          *moe_gate_bias;    /* [n_moe_layers, n_routed_experts] F32            index m; NULL nếu ckpt không có */

        __hip_bfloat16 *expert_gate;      /* [n_moe_layers, n_routed_experts, moe_inter_size, hidden_size]  index m*NRE+e */
        __hip_bfloat16 *expert_up;        /* [n_moe_layers, n_routed_experts, moe_inter_size, hidden_size]  index m*NRE+e */
        __hip_bfloat16 *expert_down;      /* [n_moe_layers, n_routed_experts, hidden_size, moe_inter_size]  index m*NRE+e */

        __hip_bfloat16 *shared_gate;      /* [n_moe_layers, n_shared*moe_inter_size, hidden_size]  index m */
        __hip_bfloat16 *shared_up;        /* [n_moe_layers, n_shared*moe_inter_size, hidden_size]  index m */
        __hip_bfloat16 *shared_down;      /* [n_moe_layers, hidden_size, n_shared*moe_inter_size]  index m */
    };

    static DeviceWeights dw;

    struct DeviceRunState {
        float *x;          /* [hidden_size] F32 — residual 1 token, dùng ở decode */
        float *xs;         /* [max_seq_len, hidden_size] F32 — residual mọi vị trí, dùng ở prefill */
        float *xb;         /* [max_seq_len, hidden_size] F32 — residual mọi vị trí, dùng ở prefill */
        float *kv_cache;   /* [n_layers, max_seq_len, kv_lora_rank+qk_rope_head_dim] F32 */
        int   *tokens;     /* [max_seq_len] I32 — token id của request hiện tại */
    };

    static DeviceRunState drs;

    struct DeviceConfig {
        /* --- kích thước cơ bản (config.json) --- */
        int H;              /* hidden_size */
        int NH;             /* n_heads — MLA không GQA nên n_kv_heads == NH */
        int QKN;            /* qk_nope_head_dim — phần q/k KHÔNG rope */
        int QKR;            /* qk_rope_head_dim — phần q/k CÓ rope */
        int VHD;            /* v_head_dim */
        int KVL;            /* kv_lora_rank — bề rộng latent c_kv nằm trong cache */
        int QLR;            /* q_lora_rank — 0 = không LoRA, dùng q_proj thẳng (dsv2) */
        int VOC;            /* vocab_size */
        int NL;             /* n_layers */
        int MAXPOS;         /* max_seq_len */
        int FKD;            /* first_k_dense — layer < FKD là dense, còn lại là MoE */

        /* --- MoE --- */
        int NRE;            /* n_routed_experts */
        int TOPK;           /* n_experts_per_tok */
        int NSHARED;        /* n_shared_experts */
        int DENSE_INTER;    /* intermediate_size — bề rộng FFN layer dense */
        int MOE_INTER;      /* moe_intermediate_size — bề rộng mỗi expert */

        /* --- cờ & hằng số float --- */
        int router_sigmoid;   /* 1 = sigmoid + e_score bias (glm), 0 = softmax (dsv2) */
        int norm_topk;        /* 1 = chuẩn hoá trọng số top-k trước khi scale */
        int rope_interleaved; /* 0 = dsv2 xoay cặp kề, 1 = glm split-half */
        float routed_scaling; /* nhân vào trọng số routed (dsv2 1.0, glm 1.8) */
        float rms_eps;        /* input_layernorm / post_attn_norm / norm cuối */
        float mla_norm_eps;   /* q_a_layernorm & kv_a_layernorm */
        float softmax_scale;  /* 1/sqrt(QHD) */

        /* --- dẫn xuất --- */
        int NMOE;           /* NL - FKD — số layer MoE */
        int QHD;            /* QKN + QKR — head_dim của q/k */
        int KVD;            /* KVL + QKR — 1 hàng kv_cache / vị trí / layer */
        int QDIM;           /* NH * QHD — bề rộng q sau projection */

        /* --- stride buffer đã duỗi phẳng, đơn vị PHẦN TỬ --- */
        size_t KV_STRIDE;     /* (QKN+VHD)*KVL — bước theo head trong kv_b_proj */
        size_t CACHE_ROW;     /* MAXPOS*KVD — bước theo layer trong kv_cache */
        size_t KVB_LAYER;     /* NH*(QKN+VHD)*KVL — trọn kv_b_proj 1 layer */
        size_t OPROJ_LAYER;   /* H*NH*VHD — o_proj 1 layer */
        size_t DENSE_TENSOR;  /* DENSE_INTER*H — mỗi tensor FFN dense 1 layer */
        size_t EXPERT_TENSOR; /* MOE_INTER*H — mỗi tensor 1 routed expert */
        size_t SHARED_TENSOR; /* NSHARED*MOE_INTER*H — mỗi tensor shared expert */
    };

    static DeviceConfig dc;

    void init_config(const Config *c) {
        dc.H = c->hidden_size;          dc.NH  = c->n_heads;
        dc.QKN = c->qk_nope_head_dim;   dc.QKR = c->qk_rope_head_dim;
        dc.VHD = c->v_head_dim;         dc.KVL = c->kv_lora_rank;
        dc.QLR = c->q_lora_rank;        dc.VOC = c->vocab_size;
        dc.NL  = c->n_layers;           dc.MAXPOS = c->max_seq_len;
        dc.FKD = c->first_k_dense;

        dc.NRE     = c->n_routed_experts;
        dc.TOPK    = c->n_experts_per_tok;
        dc.NSHARED = c->n_shared_experts;
        dc.DENSE_INTER = c->dense_inter_size;
        dc.MOE_INTER   = c->moe_inter_size;

        dc.router_sigmoid   = c->router_sigmoid;
        dc.norm_topk        = c->norm_topk;
        dc.rope_interleaved = c->rope_interleaved;
        dc.routed_scaling   = c->routed_scaling;
        dc.rms_eps          = c->rms_eps;
        dc.mla_norm_eps     = c->mla_norm_eps;
        dc.softmax_scale    = c->softmax_scale;

        dc.NMOE = dc.NL - dc.FKD;
        dc.QHD  = dc.QKN + dc.QKR;
        dc.KVD  = dc.KVL + dc.QKR;
        dc.QDIM = dc.NH * dc.QHD;

        dc.KV_STRIDE     = (size_t)(dc.QKN + dc.VHD) * dc.KVL;
        dc.CACHE_ROW     = (size_t)dc.MAXPOS * dc.KVD;
        dc.KVB_LAYER     = (size_t)dc.NH * (dc.QKN + dc.VHD) * dc.KVL;
        dc.OPROJ_LAYER   = (size_t)dc.H * dc.NH * dc.VHD;
        dc.DENSE_TENSOR  = (size_t)dc.DENSE_INTER * dc.H;
        dc.EXPERT_TENSOR = (size_t)dc.MOE_INTER * dc.H;
        dc.SHARED_TENSOR = (size_t)dc.NSHARED * dc.MOE_INTER * dc.H;
    }

    namespace prefill {

        // template<
        //     size_t BLOCK_TILE_L = 4,
        //     size_t H = 2048
        // >
        // __device__ __forceinline__ void rmsnorm(
        //     float *y,
        //     const float *x,
        //     const __hip_bfloat16 w,
        //     float buf[THREADS],
        //     int n,
        //     float eps
        // ) {
        //     int tid = threadIdx.x;
        //     float t_sum = 0.f;
        //     for (int i = tid; i < n; i += THREADS) {
        //         t_sum += x[i] * x[i];
        //     }
        //     buf[tid] = t_sum;
        //     __syncthreads();
        //     for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        //         if (tid < stride) {
        //             buf[tid] += buf[tid + stride];
        //         }
        //         __syncthreads();
        //     }
        //     float inv;
        //     if (tid == 0) {
        //         inv = 1.f / sqrtf(buf[0] / static_cast<float>(n) + eps);
        //     }
        //     // q_lora_rank = null(DSV) || 768(GLM)
        //     // kv_lora_rank = 512
        //     // hidden_size = 2048
        //     // all length always devided by 8
        //     assert(n % 8 == 0);
        //     for (int i = tid; i < n / 8; i += THREADS) {
        //         bf16v8 reg = {};
        //         reg = 
        //     }
        // }

        // template<
        //     size_t BLOCK_TILE_L = 4,
        //     size_t H = 2048
        // >
        // __global__ void kernel(
        //     const int L, // n_prompt (sequence length)
        //     const float eps,
        //     const DeviceWeights w,
        //     DeviceRunState s,
        //     const DeviceConfig c,
        //     const int *tokens
        // ) {
        //     static_assert(LDS >= (BLOCK_TILE_L * H + H + THREADS) * sizeof(float));
        //     __shared__ float xs[BLOCK_TILE_L][H];
        //     __shared__ float xb[H];
        //     __shared__ float buf[THREADS];

        //     int tix = threadIdx.x;
        //     int bix = blockIdx.x;
        //     // int bdx = blockDim.x;
        //     // int gdx = gridDim.x;
        //     int block_l = bix * BLOCK_TILE_L;
        //     int block_l_lim = min(static_cast<int>(BLOCK_TILE_L), L - block_l);
        //     for (int i = tix; i < block_l_lim * static_cast<int>(H); i += THREADS) {
        //         int l = i / H;
        //         int j = i % H;
        //         xs[l][j] = __bfloat162float(w.embed_tokens[tokens[block_l + l] * H + j]);
        //     }
        //     __syncthreads();
        //     for (int l = 0; i < block_l_lim; l++) {
        //         rmsnorm(xb, xs[l], w->input_layernorm, buf, H, eps);
        //     }
        // }

        float *run(
            Transformer *tfm,
            const int *tokens,   /* con trỏ device */
            const int L
        ) {
            assert(dc.H == kct::HIDDEN_SIZE);
            prefill_kernels::embed_kernel<<<kct::EMBED_BLOCKS, kct::EMBED_THREADS>>>(
                L,
                drs.xs,
                dw.embed_tokens,
                tokens
            );
            for (size_t nl = 0; nl < dc.NL; ++nl) {
                float *kv_l = drs.kv_cache + nl * dc.CACHE_ROW;
                constexpr size_t RMSNORM_BLOCKS = 64;
                constexpr size_t RMSNORM_THREADS = kct::WAVE * 4;
                constexpr size_t RMSNORM_BLOCK_TILE_R = 8;
                rmsnorm_kernel::rmsnorm_kernel<RMSNORM_BLOCKS, RMSNORM_THREADS, RMSNORM_BLOCK_TILE_R><<<RMSNORM_BLOCKS, RMSNORM_THREADS>>>(
                    drs.xb,
                    drs.xs,
                    dw.input_layernorm + nl * dc.H,
                    L,
                    dc.H,
                    dc.rms_eps
                );
            }
            exit(0);
        }
    }

    namespace decode {
        template<
            size_t N
        >
        __global__ void kernel() {
            
        }

        float *run(
            Transformer *tfm,
            const int token,
            const int pos
        ) {
            printf("hidden_size = %d\n", tfm->config.hidden_size);
        }
    }

    static inline const __hip_bfloat16 *as_bf16(const bf16_t *p) {
        static_assert(sizeof(__hip_bfloat16) == sizeof(bf16_t),
                      "__hip_bfloat16 phải cùng kích thước với bf16_t");
        return reinterpret_cast<const __hip_bfloat16 *>(p);
    }

    template <typename T>
    static void device_malloc(T *&dst, size_t bytes) {
        HIP_CHECK(hipMalloc(&dst, bytes));
    }

    template <typename T>
    static void device_memcpy(T *dst, const T *src, size_t bytes) {
        HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice));
    }

    template <typename T>
    static void malloc_and_memcpy(T *&dst, const T *src, size_t bytes) {
        device_malloc(dst, bytes);
        device_memcpy(dst, src, bytes);
    }

    template <typename T>
    static void device_free(T *&ptr) {
        if (ptr)
            HIP_CHECK(hipFree(ptr));
        ptr = NULL;
    }

    void init_model_weights(Transformer *tfm) {
        ModelWeights *hw = &tfm->weights;

        const size_t B = sizeof(__hip_bfloat16);

        malloc_and_memcpy(dw.embed_tokens,  as_bf16(hw->embed_tokens), (size_t)dc.VOC * dc.H * B);
        malloc_and_memcpy(dw.norm,          as_bf16(hw->norm),         (size_t)dc.H * B);
        malloc_and_memcpy(dw.lm_head,       as_bf16(hw->lm_head),      (size_t)dc.VOC * dc.H * B);
        malloc_and_memcpy(dw.rope_inv_freq, tfm->rope_inv_freq,        (size_t)(dc.QKR / 2) * sizeof(float));

        device_malloc(dw.input_layernorm, (size_t)dc.NL * dc.H * B);
        if (dc.QLR > 0) {
            device_malloc(dw.q_a_proj,      (size_t)dc.NL * dc.QLR * dc.H * B);
            device_malloc(dw.q_a_layernorm, (size_t)dc.NL * dc.QLR * B);
            device_malloc(dw.q_b_proj,      (size_t)dc.NL * dc.QDIM * dc.QLR * B);
            dw.q_proj = NULL;
        } else {
            device_malloc(dw.q_proj, (size_t)dc.NL * dc.QDIM * dc.H * B);
            dw.q_a_proj = dw.q_b_proj = dw.q_a_layernorm = NULL;
        }
        device_malloc(dw.kv_a_proj,      (size_t)dc.NL * dc.KVD * dc.H * B);
        device_malloc(dw.kv_a_layernorm, (size_t)dc.NL * dc.KVL * B);
        device_malloc(dw.W_UK,           (size_t)dc.NL * dc.KVB_LAYER * B);
        dw.W_UV = dw.W_UK + (size_t)dc.QKN * dc.KVL;
        device_malloc(dw.o_proj,         (size_t)dc.NL * dc.OPROJ_LAYER * B);
        device_malloc(dw.post_attn_norm, (size_t)dc.NL * dc.H * B);

        if (dc.FKD > 0) {
            device_malloc(dw.dense_gate, (size_t)dc.FKD * dc.DENSE_TENSOR * B);
            device_malloc(dw.dense_up,   (size_t)dc.FKD * dc.DENSE_TENSOR * B);
            device_malloc(dw.dense_down, (size_t)dc.FKD * dc.DENSE_TENSOR * B);
        } else {
            dw.dense_gate = dw.dense_up = dw.dense_down = NULL;
        }

        if (dc.NMOE > 0) {
            device_malloc(dw.moe_gate, (size_t)dc.NMOE * dc.NRE * dc.H * B);
            dw.moe_gate_bias = NULL;
            for (int l = dc.FKD; l < dc.NL; l++) {
                if (!hw->moe_gate_bias[l])
                    continue;
                const size_t bias_bytes = (size_t)dc.NMOE * dc.NRE * sizeof(float);
                device_malloc(dw.moe_gate_bias, bias_bytes);
                HIP_CHECK(hipMemsetAsync(dw.moe_gate_bias, 0, bias_bytes));
                break;
            }
            device_malloc(dw.expert_gate, (size_t)dc.NMOE * dc.NRE * dc.EXPERT_TENSOR * B);
            device_malloc(dw.expert_up,   (size_t)dc.NMOE * dc.NRE * dc.EXPERT_TENSOR * B);
            device_malloc(dw.expert_down, (size_t)dc.NMOE * dc.NRE * dc.EXPERT_TENSOR * B);
            device_malloc(dw.shared_gate, (size_t)dc.NMOE * dc.SHARED_TENSOR * B);
            device_malloc(dw.shared_up,   (size_t)dc.NMOE * dc.SHARED_TENSOR * B);
            device_malloc(dw.shared_down, (size_t)dc.NMOE * dc.SHARED_TENSOR * B);
        } else {
            dw.moe_gate = dw.expert_gate = dw.expert_up = dw.expert_down = NULL;
            dw.shared_gate = dw.shared_up = dw.shared_down = NULL;
            dw.moe_gate_bias = NULL;
        }

        for (int l = 0; l < dc.NL; l++) {
            device_memcpy(dw.input_layernorm + (size_t)l * dc.H,
                          as_bf16(hw->input_layernorm[l]), (size_t)dc.H * B);

            if (dc.QLR > 0) {
                device_memcpy(dw.q_a_proj      + (size_t)l * dc.QLR * dc.H,
                              as_bf16(hw->q_a_proj[l]),      (size_t)dc.QLR * dc.H * B);
                device_memcpy(dw.q_a_layernorm + (size_t)l * dc.QLR,
                              as_bf16(hw->q_a_layernorm[l]), (size_t)dc.QLR * B);
                device_memcpy(dw.q_b_proj      + (size_t)l * dc.QDIM * dc.QLR,
                              as_bf16(hw->q_b_proj[l]),      (size_t)dc.QDIM * dc.QLR * B);
            } else {
                device_memcpy(dw.q_proj + (size_t)l * dc.QDIM * dc.H,
                              as_bf16(hw->q_proj[l]), (size_t)dc.QDIM * dc.H * B);
            }

            device_memcpy(dw.kv_a_proj      + (size_t)l * dc.KVD * dc.H,
                          as_bf16(hw->kv_a_proj[l]),      (size_t)dc.KVD * dc.H * B);
            device_memcpy(dw.kv_a_layernorm + (size_t)l * dc.KVL,
                          as_bf16(hw->kv_a_layernorm[l]), (size_t)dc.KVL * B);
            device_memcpy(dw.W_UK           + (size_t)l * dc.KVB_LAYER,
                          as_bf16(hw->W_UK[l]),           dc.KVB_LAYER * B);
            device_memcpy(dw.o_proj         + (size_t)l * dc.OPROJ_LAYER,
                          as_bf16(hw->o_proj[l]),         dc.OPROJ_LAYER * B);
            device_memcpy(dw.post_attn_norm + (size_t)l * dc.H,
                          as_bf16(hw->post_attn_norm[l]), (size_t)dc.H * B);

            if (l < dc.FKD) {
                device_memcpy(dw.dense_gate + (size_t)l * dc.DENSE_TENSOR,
                              as_bf16(hw->dense_gate[l]), dc.DENSE_TENSOR * B);
                device_memcpy(dw.dense_up   + (size_t)l * dc.DENSE_TENSOR,
                              as_bf16(hw->dense_up[l]),   dc.DENSE_TENSOR * B);
                device_memcpy(dw.dense_down + (size_t)l * dc.DENSE_TENSOR,
                              as_bf16(hw->dense_down[l]), dc.DENSE_TENSOR * B);
                continue;
            }

            const int m = l - dc.FKD;   /* chỉ số layer trong các buffer MoE */
            device_memcpy(dw.moe_gate + (size_t)m * dc.NRE * dc.H,
                          as_bf16(hw->moe_gate[l]), (size_t)dc.NRE * dc.H * B);
            if (dw.moe_gate_bias && hw->moe_gate_bias[l])
                device_memcpy(dw.moe_gate_bias + (size_t)m * dc.NRE,
                              hw->moe_gate_bias[l], (size_t)dc.NRE * sizeof(float));

            for (int e = 0; e < dc.NRE; e++) {
                const size_t off = ((size_t)m * dc.NRE + e) * dc.EXPERT_TENSOR;
                device_memcpy(dw.expert_gate + off, as_bf16(hw->expert_gate[l][e]), dc.EXPERT_TENSOR * B);
                device_memcpy(dw.expert_up   + off, as_bf16(hw->expert_up[l][e]),   dc.EXPERT_TENSOR * B);
                device_memcpy(dw.expert_down + off, as_bf16(hw->expert_down[l][e]), dc.EXPERT_TENSOR * B);
            }

            device_memcpy(dw.shared_gate + (size_t)m * dc.SHARED_TENSOR,
                          as_bf16(hw->shared_gate[l]), dc.SHARED_TENSOR * B);
            device_memcpy(dw.shared_up   + (size_t)m * dc.SHARED_TENSOR,
                          as_bf16(hw->shared_up[l]),   dc.SHARED_TENSOR * B);
            device_memcpy(dw.shared_down + (size_t)m * dc.SHARED_TENSOR,
                          as_bf16(hw->shared_down[l]), dc.SHARED_TENSOR * B);
        }

        HIP_CHECK(hipDeviceSynchronize());
    }

    void deinit_model_weights() {
        HIP_CHECK(hipDeviceSynchronize());

        device_free(dw.embed_tokens);
        device_free(dw.norm);
        device_free(dw.lm_head);
        device_free(dw.rope_inv_freq);

        device_free(dw.input_layernorm);
        device_free(dw.q_proj);
        device_free(dw.q_a_proj);
        device_free(dw.q_a_layernorm);
        device_free(dw.q_b_proj);
        device_free(dw.kv_a_proj);
        device_free(dw.kv_a_layernorm);
        device_free(dw.W_UK);
        dw.W_UV = NULL;   /* trỏ vào giữa W_UK */
        device_free(dw.o_proj);
        device_free(dw.post_attn_norm);

        device_free(dw.dense_gate);
        device_free(dw.dense_up);
        device_free(dw.dense_down);

        device_free(dw.moe_gate);
        device_free(dw.moe_gate_bias);
        device_free(dw.expert_gate);
        device_free(dw.expert_up);
        device_free(dw.expert_down);
        device_free(dw.shared_gate);
        device_free(dw.shared_up);
        device_free(dw.shared_down);
    }

    void init_run_state() {
        const size_t kv_n = (size_t)dc.NL * dc.MAXPOS * dc.KVD;

        device_malloc(drs.x,  (size_t)dc.H * sizeof(float));
        device_malloc(drs.xs, (size_t)dc.MAXPOS * dc.H * sizeof(float));
        device_malloc(drs.xb, (size_t)dc.MAXPOS * dc.H * sizeof(float));
        device_malloc(drs.tokens, (size_t)dc.MAXPOS * sizeof(int));

        device_malloc(drs.kv_cache, kv_n * sizeof(float));
        HIP_CHECK(hipMemsetAsync(drs.kv_cache, 0, kv_n * sizeof(float)));
        HIP_CHECK(hipDeviceSynchronize());
    }

    void deinit_run_state() {
        HIP_CHECK(hipDeviceSynchronize());
        device_free(drs.x);
        device_free(drs.xs);
        device_free(drs.xb);
        device_free(drs.kv_cache);
        device_free(drs.tokens);
    }

    void upload_tokens(const int *host_tokens, int n) {
        HIP_CHECK(hipMemcpy(drs.tokens, host_tokens, (size_t)n * sizeof(int),
                            hipMemcpyHostToDevice));
    }
}

void warm_up(Transformer *t) {
    constexpr bool FOLLOW_DATASET = true;
    if constexpr (FOLLOW_DATASET) {
        constexpr size_t MAX_SEQ_LEN = 512;
        constexpr size_t MAX_NEW = 64;
        t->config.max_seq_len = MAX_SEQ_LEN + MAX_NEW;
    }
    /* Do not run inference here. Do one-time setup:
     *   - allocate scratch / device buffers
     *   - upload weights to the GPU
     * TODO(candidate). */
    opt::init_config(&t->config);
    opt::init_model_weights(t);
    opt::init_run_state();
}

void finish(Transformer *t) {
    /* One-time teardown mirroring warm_up(). TODO(candidate). */
    opt::deinit_run_state();
    opt::deinit_model_weights();
}

/* Reference implementation: greedy generate each request independently.
 * Per request: prefill the whole prompt (unabsorbed), then absorbed decode
 * until EOS or max_steps. Returns the total number of tokens generated. */
long long inference(Transformer *t, Requests *reqs) {
    const int   vocab       = t->config.vocab_size;
    const int   max_seq_len = t->config.max_seq_len;
    const int   eos         = t->tokenizer ? tokenizer_eos_id(t->tokenizer) : -1;

    if (!t->tokenizer) {
        fprintf(stderr, "inference: model_dir has no tokenizer.json\n");
        exit(EXIT_FAILURE);
    }

    int *prompt = (int *)malloc((size_t)max_seq_len * sizeof(int));
    long long total = 0;

    for (int r = 0; r < reqs->num_reqs; r++) {
        /* add_bos=0 matches the oracle/reference tokenization path. */
        int n_prompt = tokenizer_encode(t->tokenizer, reqs->prompts[r], 0,
                                        prompt, max_seq_len);
        if (n_prompt < 1) continue;

        /* Cap generation so prompt + decode stays within the KV cache. */
        int budget = max_seq_len - n_prompt;
        int steps  = reqs->max_steps < budget ? reqs->max_steps : budget;
        if (steps < 0) steps = 0;
        
        opt::upload_tokens(prompt, n_prompt);
        float *logits = opt::prefill::run(t, opt::drs.tokens, n_prompt);
        // float *logits = forward_unabsorbed(t, prompt, n_prompt, NULL, NULL);
        int    tok    = sample(logits, vocab);
        int    pos    = n_prompt;
        int    n_out  = 0;

        while (n_out < steps && tok != eos) {
            reqs->out_tokens[r][n_out++] = tok;
            logits = opt::decode::run(t, tok, pos);
            // logits = forward_absorbed(t, tok, pos);
            tok    = sample(logits, vocab);
            pos++;
        }
        reqs->out_lens[r] = n_out;
        total += n_out;
    }

    free(prompt);
    return total;
}