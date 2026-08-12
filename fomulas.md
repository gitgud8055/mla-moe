Đoạn mô tả ban đầu của bạn đã **rất chính xác** về mặt logic và kích thước chiều cho phần chiếu Query (Q) và nén KV.

Để bạn dễ dàng port toàn bộ sang HIP (C++ kernels), tôi sẽ viết lại hoàn chỉnh và chuẩn hóa **toàn bộ quy trình (Generate, Prefill, Decode, FFN)** theo đúng format bạn yêu cầu. Các hằng số sẽ được viết tắt để công thức gọn gàng (khớp với code C).

### # Params (Các hằng số kích thước)

* `L`: Độ dài chuỗi đầu vào prompt (`n_prompt`).
* `H`: Kích thước vector ẩn (`hidden_size`).
* `NH`: Số lượng attention heads (`n_heads`).
* `QKN`: Số chiều Query/Key không chứa vị trí (`qk_nope_head_dim`).
* `QKR`: Số chiều Query/Key chứa vị trí - RoPE (`qk_rope_head_dim`).
* `QHD`: Tổng số chiều 1 head của Query (`QKN + QKR`).
* `QD`: Tổng số chiều Query của tất cả heads (`NH * QHD`).
* `KVL`: Kích thước nén của KV (`kv_lora_rank`).
* `KVD`: Tổng số chiều cache mỗi vị trí (`KVL + QKR`).
* `VHD`: Số chiều Value của 1 head (`v_head_dim`).
* `VOC`: Số lượng từ vựng (`vocab_size`).
* `scale`: Hệ số scale của Softmax (`softmax_scale`).

---

### # Generate Loop (Tổng quan)

* `logits[VOC] = Prefill(tokens[0..L-1], L)`
* `tok = argmax(logits)`
* `pos = L`
* `Loop max_new lần:`
* Nếu `tok == EOS` -> Break
* `logits[VOC] = Decode(tok, pos)`
* `tok = argmax(logits)`
* `pos = pos + 1`



---

### # Prefill (`forward_unabsorbed`)

* **p = 0..L-1:**
`X[p, H] = bf16_to_fp32(embed(tokens[p]))[H]`
* **layer = 0..n_layers-1:**
**1. Projections (Tính Q, c_kv, k_pe):**
* **p = 0..L-1:**
`xb[p, H] = rmsnorm(X[p, H], w_input_layernorm[layer, H], eps)`
*Tính Query (qall):*
* Case `q_lora_rank > 0`:
`qa[q_lora_rank] = xb[p, H] @ w_q_a_proj[layer, q_lora_rank x H].T`
`qa[q_lora_rank] = rmsnorm(qa[q_lora_rank], w_q_a_layernorm[layer, q_lora_rank], eps)`
`qall[p, QD] = qa[q_lora_rank] @ w_q_b_proj[layer, QD x q_lora_rank].T`
* Case `q_lora_rank == 0`:
`qall[p, QD] = xb[p, H] @ w_q_proj[layer, QD x H].T`


*Áp dụng RoPE cho Q:*
* **h = 0..NH-1:**
`qall[p, h, QKN:QKN+QKR] = rope_apply(qall[p, h, QKN:QKN+QKR], pos=p, inv_freq)`


*Tính nén KV (comp) và ghi vào KV Cache:*
`comp[KVD] = xb[p, H] @ w_kv_a_proj[layer, KVD x H].T`
`kv_cache[layer, p, 0:KVL] = rmsnorm(comp[0:KVL], w_kv_a_layernorm[layer, KVL], eps)`
`kv_cache[layer, p, KVL:KVD] = rope_apply(comp[KVL:KVD], pos=p, inv_freq)`


**2. Decompress K & V (Giải nén):**
* **h = 0..NH-1:**
* **k = 0..L-1:**
`c_kv[KVL] = kv_cache[layer, k, 0:KVL]`
`knope[h, k, QKN] = c_kv[KVL] @ w_WUK[layer, h, QKN x KVL].T`
`value[h, k, VHD] = c_kv[KVL] @ w_WUV[layer, h, VHD x KVL].T`




**3. Attention (Tính toán Dot Product):**
* **q = 0..L-1:**
* **h = 0..NH-1:**
* **k = 0..q (causal mask):**
`qnope[QKN] = qall[q, h, 0:QKN]`
`qpe[QKR]   = qall[q, h, QKN:QKN+QKR]`
`kpe[QKR]   = kv_cache[layer, k, KVL:KVD]`
`dot_nope = qnope[QKN] @ knope[h, k, QKN].T`
`dot_rope = qpe[QKR] @ kpe[QKR].T`
`score[h, q, k] = (dot_nope + dot_rope) * scale`


`wrow[h, q, 0:q+1] = softmax(score[h, q, 0:q+1])`
`ctx[h, VHD] = wrow[h, q, 0:q+1] @ value[h, 0:q+1, VHD]`


**4. Output Projection & Residual:**
`ao[q, H] = ctx[NH x VHD] @ w_o_proj[layer, H x (NH * VHD)].T`
`X[q, H] = X[q, H] + ao[q, H]`


**5. FFN (MoE / Dense):**
* **p = 0..L-1:**
`xb[p, H] = rmsnorm(X[p, H], w_post_attn_norm[layer, H], eps)`
`mo[p, H] = ffn_compute(xb[p, H], layer)`  *(Chi tiết FFN ở phần cuối)*
`X[p, H] = X[p, H] + mo[p, H]`


* **LM Head (Chỉ tính cho vị trí cuối cùng p = L-1):**
`xb[H] = rmsnorm(X[L-1, H], w_norm[H], eps)`
`logits[VOC] = xb[H] @ w_lm_head[VOC x H].T`

---

### # Decode (`forward_absorbed`)

*Lưu ý: Chỉ xử lý 1 `token` tại vị trí `pos`. `L_kv = pos + 1*`

* `X[H] = bf16_to_fp32(embed(token))[H]`
* **layer = 0..n_layers-1:**
**1. Projections & KV Cache Update:**
`xb[H] = rmsnorm(X[H], w_input_layernorm[layer, H], eps)`
*Tính Query (q): (Giống hệt Prefill)*
* Case `q_lora_rank > 0`:
`qa[q_lora_rank] = xb[H] @ w_q_a_proj[layer, q_lora_rank x H].T`
`qa[q_lora_rank] = rmsnorm(qa[q_lora_rank], w_q_a_layernorm[layer, q_lora_rank], eps)`
`q[QD] = qa[q_lora_rank] @ w_q_b_proj[layer, QD x q_lora_rank].T`
* Case `q_lora_rank == 0`:
`q[QD] = xb[H] @ w_q_proj[layer, QD x H].T`
* **h = 0..NH-1:**
`q[h, QKN:QKN+QKR] = rope_apply(q[h, QKN:QKN+QKR], pos, inv_freq)`


*Ghi KV Cache:*
`comp[KVD] = xb[H] @ w_kv_a_proj[layer, KVD x H].T`
`kv_cache[layer, pos, 0:KVL] = rmsnorm(comp[0:KVL], w_kv_a_layernorm[layer, KVL], eps)`
`kv_cache[layer, pos, KVL:KVD] = rope_apply(comp[KVL:KVD], pos, inv_freq)`
**2. Absorbed Attention (Ma thuật của MLA):**
* **h = 0..NH-1:**
*Gộp trọng số vào Query (qabs):*
`qnope[QKN] = q[h, 0:QKN]`
`qpe[QKR]   = q[h, QKN:QKN+QKR]`
`qabs[KVL]  = qnope[1 x QKN] @ w_WUK[layer, h, QKN x KVL]` *(Lưu ý: Không transpose, vì fold W_UK vào Q)*
*Tính Score trực tiếp trong không gian nén (Latent Space):*
* **k = 0..pos:**
`c_kv[KVL] = kv_cache[layer, k, 0:KVL]`
`kpe[QKR]  = kv_cache[layer, k, KVL:KVD]`
`score[k]  = (qabs[KVL] @ c_kv[KVL].T + qpe[QKR] @ kpe[QKR].T) * scale`


`score[0:pos+1] = softmax(score[0:pos+1])`
*Tính Context Nén (clat) & Bung Value (ctx):*
`clat[KVL] = score[1 x pos+1] @ kv_cache[layer, 0:pos+1, 0:KVL]`
`ctx[h, VHD] = clat[KVL] @ w_WUV[layer, h, VHD x KVL].T`


**3. Output Projection & Residual:**
`ao[H] = ctx[NH x VHD] @ w_o_proj[layer, H x (NH * VHD)].T`
`X[H] = X[H] + ao[H]`
**4. FFN:**
`xb[H] = rmsnorm(X[H], w_post_attn_norm[layer, H], eps)`
`mo[H] = ffn_compute(xb[H], layer)`
`X[H] = X[H] + mo[H]`
* **LM Head:**
`xb[H] = rmsnorm(X[H], w_norm[H], eps)`
`logits[VOC] = xb[H] @ w_lm_head[VOC x H].T`

---

### # Chi tiết `ffn_compute(x[H], layer)`

*Hàm SwiGLU: `swiglu(x, gate, up, down) = (silu(x @ gate.T) * (x @ up.T)) @ down.T*`

* **Nếu layer < first_k_dense (Dense FFN):**
`out[H] = swiglu(x[H], w_dense_gate, w_dense_up, w_dense_down)`
* **Nếu layer >= first_k_dense (MoE FFN):**
`logits_moe[E] = x[H] @ w_moe_gate[E x H].T`  *(E = n_routed_experts)*
* Case `router_sigmoid == true` (GLM): `scores[E] = sigmoid(logits_moe[E])`
* Case `router_sigmoid == false` (DSV2): `scores[E] = softmax(logits_moe[E])`


*Top-K Routing:*
`topk_idx[K], topk_w[K] = TopK(scores[E] + w_moe_bias[E], K)`
* Nếu `norm_topk`: `topk_w[K] = normalize(topk_w[K])`
`topk_w[K] = topk_w[K] * routed_scaling`


*Tính Experts được chọn:*
`out[H] = 0`
* **r = 0..K-1:**
`idx = topk_idx[r]`
`exp_out[H] = swiglu(x[H], w_expert_gate[idx], w_expert_up[idx], w_expert_down[idx])`
`out[H] += topk_w[r] * exp_out[H]`


*Tính Shared Expert (Luôn chạy):*
`shared_out[H] = swiglu(x[H], w_shared_gate, w_shared_up, w_shared_down)`
`out[H] += shared_out[H]`