# MLA-MoE — spec port, đối chiếu từng dòng với `src/run.c` (+ `src/model_load.c`, `include/model.h`)

```
# Params

## Đọc thẳng từ config.json
- L: n_prompt (số token prompt, không phải config)
- H = hidden_size ; NH = num_attention_heads
- QKN = qk_nope_head_dim ; QKR = qk_rope_head_dim ; QHD = QKN + QKR
- KVL = kv_lora_rank ; VHD = v_head_dim ; VOC = vocab_size
- q_lora_rank            # JSON null -> 0 (dsv2: dùng q_proj thẳng)
- n_layers = num_hidden_layers ; first_k_dense = first_k_dense_replace
- E = n_routed_experts ; K = num_experts_per_tok ; n_shared_experts
- dense_inter_size = intermediate_size ; moe_inter_size = moe_intermediate_size
- rms_eps = rms_norm_eps : input_layernorm / post_attn_norm / norm cuối
- norm_topk = norm_topk_prob ; routed_scaling = routed_scaling_factor (dsv2: 1.0, glm: 1.8)
- rope_theta ; rope_scaling{factor, beta_fast, beta_slow, original_max_position_embeddings}
  # rope_scaling null -> factor=1, beta_fast=32, beta_slow=1, orig_max=4096

## KHÔNG có trong config.json — phải tự dẫn xuất
- q_dim = NH * QHD
- kvd = KVL + QKR                        # 1 hàng s_kv_cache / vị trí / layer
- scale = softmax_scale = 1/sqrt(QHD)    # QHD^-0.5, KHÔNG nhân mscale cho cả 2 model
- mla_norm_eps = 1e-6                    # HARDCODE (class default), KHÔNG phải rms_norm_eps
- max_seq_len = min(max_position_embeddings, 163840)
- rope_interleaved = (model_type == "glm4_moe_lite") ? 1 : 0     # 1 = glm, 0 = dsv2
- router_sigmoid   = (topk_method == "noaux_tc")    ? 1 : 0      # 1 = glm, 0 = dsv2 softmax
- inv_freq[QKR/2]  = build_rope_inv_freq()  # YaRN, xem cuối file; dùng chung mọi layer
- s_kv_cache[n_layers, max_seq_len, kvd]   # fp32, dùng chung Prefill & Decode
- n_group / topk_group: KHÔNG cài — code luôn chạy top-k phẳng. run.c chỉ biện minh
  cho glm47 (cả hai = 1 nên group-limited routing suy biến thành top-k phẳng);
  dsv2lite đi nhánh softmax nên cũng không đụng tới. Model khác thì phải cài thêm.

## Quy ước
- Tên mảng: w_* = field của ModelWeights (run.c: w->*) ; s_* = field của RunState
  (run.c: s->*) ; không tiền tố = buffer cục bộ, đặt trùng tên biến trong run.c.
- a..b inclusive 2 đầu; W[m x n] row-major; y = x @ W.T nghĩa là y[d] = Σ_i x[i]*W[d,i]
- MỌI weight là bf16, mọi compute là fp32: đọc weight -> bf16_to_f32 rồi mới nhân.
  Ngoại lệ duy nhất: moe_gate_bias (e_score_correction_bias) lưu F32 trong checkpoint.
- lm_head là tensor riêng, KHÔNG tie với embed_tokens.

# Generate

- s_logits[VOC] = Prefill(tokens[0..L-1])
- tok = argmax(s_logits) ; pos = L
- i = 0..max_new-1:
  - nếu tok == eos: break
  - emit tok                             # tok nằm ở vị trí pos
  - s_logits[VOC] = Decode(tok, pos)
  - tok = argmax(s_logits) ; pos += 1

# Prefill  (forward_unabsorbed)
# Mỗi layer = 4 phase, mỗi phase ~ 1 kernel HIP.

- p = 0..L-1:
  xs[p, H] = bf16_to_f32(w_embed_tokens[tokens[p], H])

- layer l = 0..n_layers-1:

  ## Phase 1 — projections + ghi KV cache.  p = 0..L-1:
  - xb[H] = rmsnorm(xs[p, H], w_input_layernorm[l, H], rms_eps)
  - qall[p, q_dim] = project_q(xb):
    + case q_lora_rank > 0:
      s_q_a[q_lora_rank] = xb[H] @ w_q_a_proj[l, q_lora_rank x H].T
      s_q_a = rmsnorm(s_q_a, w_q_a_layernorm[l, q_lora_rank], mla_norm_eps)
      qall[p] = s_q_a @ w_q_b_proj[l, q_dim x q_lora_rank].T
    + case q_lora_rank == 0:
      qall[p] = xb @ w_q_proj[l, q_dim x H].T
  - h = 0..NH-1:
    rope_apply(qall[p, h, QKN:QKN+QKR], pos=p, inv_freq, QKR, rope_interleaved)
  - comp[kvd] = xb[H] @ w_kv_a_proj[l, kvd x H].T
  - s_kv_cache[l, p, 0:KVL] = rmsnorm(comp[0:KVL], w_kv_a_layernorm[l, KVL], mla_norm_eps)
  - s_kv_cache[l, p, KVL:kvd] = comp[KVL:kvd]
  - rope_apply(s_kv_cache[l, p, KVL:kvd], pos=p, inv_freq, QKR, rope_interleaved)   # k_pe PHẢI rope

  ## Phase 2 — giải nén K_nope / V.  h = 0..NH-1, k = 0..L-1:
  - c_kv[KVL] = s_kv_cache[l, k, 0:KVL]
  - knope[h, k, QKN] = c_kv @ w_W_UK[l, h, QKN x KVL].T
  - value[h, k, VHD] = c_kv @ w_W_UV[l, h, VHD x KVL].T
  # layout: tách zero-copy từ kv_b_proj [NH*(QKN+VHD), KVL], head-INTERLEAVED
  #   (mỗi head là QKN hàng K rồi VHD hàng V):
  #   w_W_UK[l] = kv_b_proj[l]                      # head 0
  #   w_W_UV[l] = kv_b_proj[l] + QKN*KVL            # sau W_UK của HEAD 0, không phải của mọi head
  #   head h   -> w_W_UK[l] + h*stride, w_W_UV[l] + h*stride  với stride = (QKN+VHD)*KVL

  ## Phase 3 — attention.  q = 0..L-1, h = 0..NH-1:
  - qnope[QKN] = qall[q, h, 0:QKN] ; qpe[QKR] = qall[q, h, QKN:QKN+QKR]
  - k = 0..q:                            # causal: k > q -> score = -inf (softmax coi = 0)
    kpe[QKR]  = s_kv_cache[l, k, KVL:kvd]
    score[k]  = (qnope · knope[h, k, 0:QKN] + qpe · kpe) * scale
  - wgt[0:q+1] = softmax(score[0:q+1])
  - ctx[h, VHD] = Σ_{k=0..q} wgt[k] * value[h, k, 0:VHD]
  # sau khi đủ NH head:
  - ao[q, H] = ctx[NH*VHD] @ w_o_proj[l, H x (NH*VHD)].T
  # residual cộng thành SWEEP RIÊNG sau khi xong hết q (ao không phụ thuộc xs mới,
  # nên in-loop cũng ra cùng kết quả — nhưng đừng gộp chung kernel với vòng q ở trên):
  - p = 0..L-1: xs[p, H] += ao[p, H]

  ## Phase 4 — FFN.  p = 0..L-1:
  - xb[H] = rmsnorm(xs[p, H], w_post_attn_norm[l, H], rms_eps)
  - mo[p, H] = ffn_compute(xb, l)
  - p = 0..L-1: xs[p, H] += mo[p, H]      # lại là sweep riêng, sau khi xong hết p

- LM head (chỉ vị trí cuối):
  - xb[H] = rmsnorm(xs[L-1, H], w_norm[H], rms_eps)
  - s_logits[VOC] = xb @ w_lm_head[VOC x H].T

# Decode  (forward_absorbed) — 1 token tại pos ; kv_len = pos + 1

- x[H] = bf16_to_f32(w_embed_tokens[token, H])
- layer l = 0..n_layers-1:

  ## 1. Projections + append KV cache (y hệt Phase 1, chỉ 1 vị trí pos):
  - xb[H] = rmsnorm(x[H], w_input_layernorm[l, H], rms_eps)
  - q[q_dim] = project_q(xb)             # 2 case như trên
  - h = 0..NH-1: rope_apply(q[h, QKN:QKN+QKR], pos, inv_freq, QKR, rope_interleaved)
  - comp[kvd] = xb @ w_kv_a_proj[l, kvd x H].T
  - s_kv_cache[l, pos, 0:KVL] = rmsnorm(comp[0:KVL], w_kv_a_layernorm[l, KVL], mla_norm_eps)
  - s_kv_cache[l, pos, KVL:kvd] = comp[KVL:kvd]
  - rope_apply(s_kv_cache[l, pos, KVL:kvd], pos, inv_freq, QKR, rope_interleaved)

  ## 2. Absorbed attention.  h = 0..NH-1:
  - qnope[QKN] = q[h, 0:QKN] ; qpe[QKR] = q[h, QKN:QKN+QKR]
  - qabs[KVL] = qnope[1 x QKN] @ w_W_UK[l, h, QKN x KVL]      # KHÔNG .T (fold W_UK vào q)
    # qabs[r] = Σ_d qnope[d] * W_UK[d, r]
  - k = 0..pos:
    score[k] = (qabs · s_kv_cache[l, k, 0:KVL] + qpe · s_kv_cache[l, k, KVL:kvd]) * scale
  - score[0:kv_len] = softmax(score[0:kv_len])
  - clat[KVL] = Σ_{k=0..pos} score[k] * s_kv_cache[l, k, 0:KVL]
  - ctx[h, VHD] = clat @ w_W_UV[l, h, VHD x KVL].T            # CÓ .T

  ## 3. Output + residual:
  - ao[H] = ctx[NH*VHD] @ w_o_proj[l, H x (NH*VHD)].T
  - x[H] += ao[H]

  ## 4. FFN:
  - xb[H] = rmsnorm(x[H], w_post_attn_norm[l, H], rms_eps)
  - x[H] += ffn_compute(xb, l)

- LM head:
  - xb[H] = rmsnorm(x[H], w_norm[H], rms_eps)
  - s_logits[VOC] = xb @ w_lm_head[VOC x H].T

# ffn_compute(x[H], l)

- case l < first_k_dense:
  out[H] = swiglu(x, w_dense_gate[l], w_dense_up[l], w_dense_down[l], dense_inter_size)

- case l >= first_k_dense (MoE):
  s_moe_logits[E] = x @ w_moe_gate[l, E x H].T
  + router_sigmoid == 1 (glm):  s_moe_logits[e] = sigmoid(s_moe_logits[e])
  + router_sigmoid == 0 (dsv2): s_moe_logits[E] = softmax(s_moe_logits[E])
  bias[E] = w_moe_gate_bias[l]           # fp32; dsv2: NULL -> coi = 0
  idx[K]  = ArgTopK(s_moe_logits[E] + bias[E])      # greedy K vòng, tie -> index nhỏ thắng (so sánh >)
  wts[r]  = s_moe_logits[idx[r]]                    # raw score, KHÔNG cộng bias
  + norm_topk: wts[r] /= (Σ_r wts[r] + 1e-20)
  wts[r] *= routed_scaling               # luôn nhân (dsv2: routed_scaling = 1)
  out[H] = Σ_{r=0..K-1} wts[r] * swiglu(x, w_expert_gate[l, idx[r]],
                                           w_expert_up[l, idx[r]],
                                           w_expert_down[l, idx[r]], moe_inter_size)
  # shared expert: trong layer MoE thì LUÔN chạy (không qua router), 1 MLP duy nhất
  out[H] += swiglu(x, w_shared_gate[l], w_shared_up[l], w_shared_down[l],
                   n_shared_experts * moe_inter_size)

# rmsnorm(x[n], w[n], eps) -> y[n]
- ss  = Σ_{i=0..n-1} x[i]*x[i]
- inv = 1 / sqrt(ss/n + eps)             # eps NẰM TRONG sqrt và SAU khi chia n
- y[i] = x[i] * inv * bf16_to_f32(w[i])
# gọi được in-place (y == x); code dùng đúng kiểu đó cho s_q_a và cho c_kv trong cache

# swiglu(x[H], gate, up, down, inter)
- s_hb[inter]  = x @ gate[inter x H].T
- s_hb2[inter] = x @ up[inter x H].T
- s_hb[i] = silu(s_hb[i]) * s_hb2[i]           # silu(z) = z / (1 + exp(-z))
- ret s_hb @ down[H x inter].T

# rope_apply(v[QKR], pos, inv_freq[QKR/2], interleaved)   # in-place
- half = QKR/2 ; j = 0..half-1: ang = pos * inv_freq[j] ; c = cos(ang), s = sin(ang)
- case interleaved == 0 (dsv2): cặp kề, in-place
  (v[2j], v[2j+1]) <- (v[2j]*c - v[2j+1]*s,  v[2j]*s + v[2j+1]*c)
- case interleaved == 1 (glm): even/odd -> split-half
  even = v[2j] ; odd = v[2j+1]
  out[j] = even*c - odd*s ; out[half+j] = odd*c + even*s
  v <- out[0:QKR]

# softmax(x[n]) — ổn định số
- m = max(x) ; x[i] = (x[i] == -inf ? 0 : exp(x[i] - m)) ; x[i] /= Σx

# build_rope_inv_freq() -> inv_freq[QKR/2]   # chạy 1 lần lúc load, YaRN kiểu DeepSeek-V2
# dim = QKR ; half = QKR/2 ; base = rope_theta ; factor = rope_factor
- lo_d = dim * log(orig_max / (beta_fast * 2π)) / (2 * log(base))
- hi_d = dim * log(orig_max / (beta_slow * 2π)) / (2 * log(base))
- low  = max(floor(lo_d), 0) ; high = min(ceil(hi_d), dim-1)      # clamp theo dim-1, KHÔNG phải half-1
- denom = (high == low) ? 0.001 : (high - low)
- j = 0..half-1:
    f_extra = base^(-(2j)/dim)            # ngoại suy: tần số gốc
    f_inter = f_extra / factor            # nội suy: chia factor
    ramp    = clamp((j - low) / denom, 0, 1) ; mask = 1 - ramp
    inv_freq[j] = f_inter * ramp + f_extra * mask
# factor == 1 (rope_scaling null, vd glm) -> f_inter == f_extra -> rope thường.
# mscale == mscale_all_dim nên biên độ quay = 1: CHỈ nội suy tần số, không scale q/k.
```

Các chỗ dễ dính bug nhất khi port, theo thứ tự nguy hiểm:

1. **`@ W_UK` (không .T, decode) vs `@ W_UK.T` (có .T, prefill Phase 2)** — cùng một tensor, hai chiều thu gọn khác nhau.
2. **Stride head của `W_UV`**: `kv_b_proj` là head-INTERLEAVED `[NH*(QKN+VHD), KVL]`, mỗi head là QKN hàng K rồi VHD hàng V. Nên `W_UV` bắt đầu ở `kv_b + QKN*KVL` (sau W_UK của **head 0**), *không* phải sau W_UK của tất cả head; và cả hai đều bước `(QKN+VHD)*KVL` mỗi head.
3. **Thứ tự rmsnorm → copy raw → rope** khi ghi `k_pe` vào cache.
4. **`mla_norm_eps = 1e-6` hardcode**, không lấy `rms_norm_eps` (1e-5) — phương sai của `q_a` rất nhỏ nên nhạy với eps.
5. **`inv_freq` phải build bằng YaRN**, không đọc từ config; sai chỗ này thì chỉ lệch ở context dài nên rất khó phát hiện bằng test ngắn.
6. **`scale = 1/sqrt(QKN+QKR)`**, tính từ QHD chứ không phải VHD và không có trong config.

Còn lại là matmul thẳng, port sang kernel khá cơ học.