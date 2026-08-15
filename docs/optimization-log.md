# Optimization log

GPU: AMD Instinct MI250X / MI250 (`gfx90a`), HIP device isolated via `HIP_VISIBLE_DEVICES`.
Model: DeepSeek-V2-Lite (`/remote/vast0/share-mv/deepseek-ai/DeepSeek-V2-Lite`).
Workload: `data/dsv2lite/requests.txt`, 512 requests, 64 decode steps.
Command:

```sh
HIP_VISIBLE_DEVICES=<free> GETP_REPORT_DECODE_TIMING=1 GETP_REPORT_PREFILL_TIMING=1 \
  ./run "$DSV" getp data/dsv2lite/requests.txt /tmp/getp_dsv2lite.txt 64
```

Binary at kickoff: commit `b2c2d66` (`graph cap 120 + fix bf16 quantize error`).
Prefill is ~17.78 s / 3313 input tok/s and is left unchanged unless noted.

## Baseline (2026-08-15)

| Run | Prefill ms | Decode ms | Decode tok/s | E2E TPS |
|---:|---:|---:|---:|---:|
| 1 | 17792.646 | 14906.633 | 2198.216 | 993.366 |
| 2 | 17777.895 | 14931.175 | 2194.603 | 993.035 |
| **median** | **17785** | **14919** | **2196** | **993.2** |

Profile prior (`rocprof_kernel_report.md`, short 5-request trace): `mfma_gemm` 40% GPU time, VALU busy 4.8%, MFMA util 1.2%, mem busy 6.9% — latency-bound GEMM, not CU-full.

## 2026-08-15 — decode GEMM M-tile 32/64

**Hypothesis:** decode batch 512 uses `mfma_gemm` M=16, so the 400 MiB LM-head is reloaded 32 times per step. `mfma_gemm_rows` (M=32/64) should cut that traffic.

**Change:** dispatch `mfma_gemm_rows<2|4>` for decode rows >= 256 (all GEMMs, then wide `d_out>=2048` only). Prefill path unchanged.

**Result:** REJECT. Decode slowed in every variant.

| Variant | Decode ms | E2E TPS |
|---|---:|---:|
| M=64 all GEMMs | 16129 | 958.3 |
| M=64 wide only | 15560 | 974.0 |
| M=32 wide only | 15393 | 978.8 |

Skinny projections (router N=64, kv N=576) lose CTA count; even wide-only M=32/64 lost occupancy to extra LDS/VGPR. Reverted.

## 2026-08-15 — decode grouped-expert tile 32

**Hypothesis:** at B=512, K=6, E=64 each expert has ~48 routes, so tile 16 reloads expert weights 3 times. Tile 32 should cut that to 2. Prefill stays at 16.

**Change:** `grouped_expert_{gate_up,down}<32,2>` when decode batch >= 256.

**Result:** REJECT. Decode 15648 ms / 971.4 TPS. Extra padding and lower occupancy outweighed reuse. Reverted.

## 2026-08-15 — `mfma_gemm` vector load + K-tile prefetch — KEEP

**Hypothesis:** `mfma_gemm` is latency-bound (MFMA util 1.2%). Issuing the next K-tile's global loads into registers before the current LDS MFMA, plus coalesced `float4` activation loads, should hide HBM latency on the decode GEMM path (LM head, q/o/shared down). Prefill large-M uses `mfma_gemm_rows` and should be unchanged.

**Change:** `src/kernels/batch_decode_kernels.hip` `mfma_gemm` only. Double-buffer LDS, prefetch next A (`float4` fp32→bf16) and B (`bf16x8`) into registers, then MFMA the resident tile.

**Result:**

| Run | Prefill ms | Decode ms | Decode tok/s | E2E TPS |
|---:|---:|---:|---:|---:|
| 1 | 17776.395 | 14704.576 | 2228.422 | 999.925 |
| 2 | 17733.793 | 14710.676 | 2227.498 | 1000.830 |
| **median** | **17755** | **14708** | **2228** | **1000.4** |

Decode 14919 → 14708 ms (**+1.4%**). E2E 993.2 → 1000.4 TPS (**+0.7%**). Prefill flat.

Accuracy (`score_completions.py` on `data/dsv2lite`, thresholds METEOR>=0.30 / BERTScore-F1>=0.91):

| Metric | Value | Gate |
|---|---:|---|
| METEOR | 0.5472 | PASS |
| BERTScore-F1 | 0.9119 | PASS |
| Sequence acc | 0.2793 | diagnostic |
| Token acc | 0.4172 | diagnostic |

**KEEP.**

## 2026-08-15 — same prefetch on `mfma_gate_up_swiglu` — REJECT

**Hypothesis:** shared-expert SwiGLU is the same latency-bound MFMA family.

**Result:** decode 15016 ms / 990.1 TPS — slower (double LDS for two weight tiles). Reverted. `mfma_gemm` prefetch kept.

## 2026-08-15 — decode MoE bf16 intermediate — KEEP

**Hypothesis:** prefill already writes routed SwiGLU as bf16 into `g.prefill_routed_hidden`. Decode passed `nullptr` and stored fp32, then converted back to bf16 in `grouped_expert_down`. That doubled intermediate traffic on the decode MoE path.

**Change:** `gpu_ffn_batch()` reuses `g.prefill_routed_hidden` (capacity is prefill-sized, decode routes fit). Prefill call site unchanged.

**Result:**

| Run | Prefill ms | Decode ms | Decode tok/s | E2E TPS |
|---:|---:|---:|---:|---:|
| 1 | 17781.463 | 14517.232 | 2257.180 | 1005.472 |
| 2 | 17778.875 | 14534.344 | 2254.522 | 1003.623 |
| **median** | **17780** | **14526** | **2256** | **1004.5** |

Vs original baseline decode 14919 → 14526 ms (**+2.7%**). Vs `mfma_gemm` prefetch-only 14708 → 14526 ms (**+1.2%**). Combined E2E 993.2 → 1004.5 TPS (**+1.1%**).

Accuracy (same command / data as above): METEOR 0.5472, BERTScore-F1 0.9119, token acc 13649/32714 — identical to the prefetch-only run, PASS.

**KEEP** together with the `mfma_gemm` prefetch.

## Combined decode status

| | Decode ms | Decode tok/s | E2E TPS |
|---|---:|---:|---:|
| Baseline `b2c2d66` | 14919 | 2196 | 993.2 |
| + `mfma_gemm` prefetch + decode bf16 MoE | 14526 | 2256 | 1004.5 |

Accuracy DSV2Lite: METEOR 0.5472, BERTScore-F1 0.9119, both over gate.

Failed this session: decode M-tile 32/64, grouped-expert tile 32, `mfma_gate_up_swiglu` prefetch.

## 2026-08-15 — fully split prefill vs decode kernels — KEEP

**Hypothesis:** shared kernels/launchers let decode edits leak into prefill. Copying HEAD prefill kernels into their own TU, and keeping decode KEEPs only under `batch_decode_kernels::decode`, should restore HEAD prefill time while leaving decode at the prefetch + bf16-MoE numbers.

**Change:**
- New `src/kernels/prefill_kernels.hip` (`namespace prefill_kernels`) from HEAD `batch_decode_kernels.hip` (original `mfma_gemm`, `mfma_gemm_rows`, prefill flash/q_rope/kv_norm_rope).
- `src/kernels/batch_decode_kernels.hip` is decode-only (`namespace batch_decode_kernels::decode`), including a private `softmax_rows` copy so decode does not call `attention_kernels`.
- `src/getp_run.hip`: `namespace prefill` launchers/FFN/`run_chunk` call only `prefill_kernels::*`; `namespace decode` launchers/FFN/`run_step` call only `batch_decode_kernels::decode::*`. No shared GEMM/FFN/attention/router launchers.

**Result** (HIP device 0, same command as header):

| | Prefill ms | Decode ms | Decode tok/s | E2E TPS |
|---|---:|---:|---:|---:|
| HEAD baseline median | 17785 | 14919 | 2196 | 993.2 |
| Decode KEEPs before split | 17780 | 14526 | 2256 | 1004.5 |
| After split | **17732** | **14505** | **2259** | **1007.4** |

Prefill matches HEAD (~17.78 s, within noise). Decode stays on the two KEEPs.

Accuracy (`score_completions.py` on `data/dsv2lite`): METEOR 0.5472, BERTScore-F1 0.9119, sequence acc 0.2793, token acc 13649/32714 — PASS, identical to the KEEP run.

**KEEP.** Further decode work must stay in `decode::` / `batch_decode_kernels::decode`. Prefill stays at HEAD kernels unless a dedicated prefill experiment says otherwise.

## Next decode experiments

1. Vectorized-A + K-prefetch on `grouped_expert_gate_up/down` (still the largest decode bytes).
2. Profile a real 512×64 decode slice with `rocprof-compute`.
3. GLM-4.7-Flash smoke + accuracy for these shared kernel/buffer changes.
