# ROCprof Compute profile — code hiện tại

Ngày chạy: 2026-08-14 (Asia/Ho_Chi_Minh)

## Kết luận nhanh

- Đã rebuild và profile đúng working tree hiện tại bằng `rocprof-compute 3.4.0`.
- Thu được **2.624 GPU dispatch** trong mỗi counter pass.
- Bốn kernel đứng đầu chiếm khoảng **73,36%** tổng thời gian GPU có instrumentation:
  `mfma_gemm`, `grouped_expert_gate_up`, `grouped_expert_down`, và
  `mfma_gate_up_swiglu`.
- `mfma_gemm` là hotspot lớn nhất: **540 calls, 51,817 ms, 40,75%**.
- Toàn workload chỉ phát sinh MFMA BF16; rocprof-compute báo trung bình
  **1.402,445 GFLOP/s**, tương đương **0,775% peak** khi lấy trung bình trên toàn bộ
  dispatch (bao gồm rất nhiều kernel không dùng MFMA).
- Average vL1D utilization là **28,50%**, L2 utilization là **31,10%**.
- LDS bank conflict rất thấp: **0,002 conflict/access**. LDS không phải bottleneck rõ
  ràng trong workload ngắn này.

## Code được profile

```text
git commit: 8823051
src/getp_run.hip SHA-256:
3b4d5ca31e3425d4d2c2b7874e16e6f0929ba8362a908ba93f16211482030da8

run SHA-256:
9b1aa679fd2b3843bf72bbfb0fba11a809e15359109043099d8b2b73e13e91fb
```

`src/getp_run.hip` có thay đổi chưa commit và được rebuild ngay trước khi profile.

## Workload

```text
Model: DeepSeek-V2-Lite
GPU vật lý: HIP_VISIBLE_DEVICES=1 (MI250X/MI250, gfx90a)
Request: 5 request giống nhau, "The capital of France is"
Batch: 5
Decode steps: 2
KV capacity: 64
Generated tokens: 10
```

Lệnh workload:

```bash
HIP_VISIBLE_DEVICES=1 GETP_BATCH_SIZE=5 GETP_KV_CAPACITY=64 \
./run /remote/vast0/share-mv/deepseek-ai/DeepSeek-V2-Lite \
  getp /tmp/mla-moe-rocprof-requests.txt /tmp/mla-moe-current-profile.out 2
```

Workload ngắn chạy cả batched prefill và decode batch 5, nhưng kết quả attention
không đại diện cho context dài.

## Timing không profiler

| Chỉ số | Kết quả |
|---|---:|
| Warm-up / upload weights | 15,665 s |
| Inference | 79,479 ms |
| Throughput | 125,819 tok/s |

Counter instrumentation làm inference tăng lên 256–422 ms tùy pass. Vì vậy TPS
trong log profiler không phải số benchmark; dùng run không profiler ở bảng trên.

## Top kernel theo thời gian

Số liệu timing dưới đây lấy từ pass `compute_thruput_util`. `GPU %` được
rocprof-compute chuẩn hóa theo tổng duration của 2.624 dispatch.

| # | Kernel | Calls | Total ms | Avg us | Median us | GPU % |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `batch_decode_kernels::mfma_gemm` | 540 | 51,817 | 95,957 | 85,681 | 40,748 |
| 2 | `grouped_expert_gate_up<16,1>` | 104 | 19,742 | 189,822 | 136,161 | 15,525 |
| 3 | `grouped_expert_down<16,1>` | 104 | 11,570 | 111,247 | 80,081 | 9,098 |
| 4 | `batch_decode_kernels::mfma_gate_up_swiglu` | 108 | 10,154 | 94,020 | 88,880 | 7,985 |
| 5 | `batch_decode_kernels::mfma_head_gemm` | 216 | 4,319 | 19,993 | 20,080 | 3,396 |
| 6 | `batch_decode_kernels::rmsnorm` | 275 | 3,473 | 12,630 | 12,640 | 2,731 |
| 7 | `batch_decode_kernels::reduce_routes_add` | 104 | 3,071 | 29,529 | 29,840 | 2,415 |
| 8 | `batch_decode_kernels::latent_context` | 108 | 3,053 | 28,265 | 34,000 | 2,401 |
| 9 | `batch_decode_kernels::gemv` | 81 | 2,558 | 31,581 | 23,041 | 2,012 |
| 10 | `batch_decode_kernels::routed_shared_gate_up` | 26 | 2,398 | 92,216 | 92,320 | 1,885 |
| 11 | `router_topk_wave64_kernel<64>` | 130 | 2,096 | 16,122 | 16,160 | 1,648 |
| 12 | `batch_decode_kernels::align_routes` | 104 | 1,826 | 17,562 | 16,960 | 1,436 |
| 13 | `batch_decode_kernels::routed_shared_down` | 26 | 1,687 | 64,886 | 64,880 | 1,327 |
| 14 | `batch_decode_kernels::kv_norm_rope` | 108 | 1,124 | 10,409 | 10,400 | 0,884 |
| 15 | `batch_decode_kernels::attention_score` | 108 | 1,088 | 10,077 | 10,080 | 0,856 |

## Compute pipeline

Pass: `--set compute_thruput_util`

| Metric | Avg | Min | Max |
|---|---:|---:|---:|
| VALU utilization | 7,261% | 0,002% | 70,656% |
| VMEM utilization | 0,454% | 0,000% | 7,195% |
| Branch utilization | 0,800% | 0,002% | 7,970% |
| VALU active threads | 56,363 | 12,916 | 64,000 |

Các average này là trung bình trên toàn bộ dispatch, không time-weighted. Giá trị
VALU/VMEM thấp cùng với số dispatch lớn cho thấy còn nhiều kernel nhỏ và launch
fragmentation; không nên diễn giải chúng như utilization riêng của `mfma_gemm`.

## MFMA throughput

Pass: `--set compute_thruput_flops`

| Metric | Average | Peak | % peak |
|---|---:|---:|---:|
| MFMA BF16 | 1.402,445 GFLOP/s | 181.043,200 GFLOP/s | 0,775% |
| MFMA F16 | 0 | 181.043,200 GFLOP/s | 0% |
| MFMA F32 | 0 | 45.260,800 GFLOP/s | 0% |
| MFMA F64 | 0 | 45.260,800 GFLOP/s | 0% |
| MFMA Int8 | 0 | 181.043,200 GIOP/s | 0% |

BF16 MFMA là đường compute duy nhất đang hoạt động. Con số 0,775% là average trên
tất cả kernel, nên chủ yếu dùng để xác nhận datatype/path; muốn đánh giá hiệu suất
MFMA chính xác hơn cần filter riêng các dispatch `mfma_*`.

## Memory hierarchy

Pass: `--set mem_thruput`

| Metric | Average | Peak / unit | % peak |
|---|---:|---:|---:|
| Theoretical LDS bandwidth | 524,049 GB/s | 22.630,400 GB/s | 2,316% |
| LDS bank conflicts/access | 0,002 | 32 conflict/access | 0,005% |
| Vector L1 data-cache utilization | 28,502% | — | — |
| L2 utilization | 31,102% | — | — |

## Hướng tối ưu ưu tiên

1. Tập trung trước vào `mfma_gemm` và hai grouped-expert kernel; bốn kernel đầu
   chiếm hơn 73% GPU time dưới cùng instrumentation.
2. Giảm số launch/ghép các kernel nhỏ quanh MoE route, RMSNorm và reduce. Toàn run
   có 2.624 dispatch cho chỉ 10 output token.
3. Filter riêng kernel `mfma_gemm`, `grouped_expert_gate_up/down` trong lần profile
   tiếp theo để tránh metric average bị pha loãng bởi kernel nhỏ.
4. Profile thêm request context dài trước khi thay đổi attention; workload hiện tại
   chỉ có prompt 5 token/request.

## Raw data và output chính thức từ rocprof-compute

- Compute utilization raw: `rocprof_compute_current/pmc_perf.csv`
- MFMA FLOPs raw: `rocprof_compute_current_flops/pmc_perf.csv`
- Memory throughput raw: `rocprof_compute_current_mem/pmc_perf.csv`
- Full analyzer output: `rocprof_compute_current_analysis.txt`
- MFMA analyzer output: `rocprof_compute_current_flops_analysis.txt`
- Memory analyzer output: `rocprof_compute_current_mem_analysis.txt`

Các warning về roofline/PC sampling trong bước analyze là expected vì cả ba pass
dùng `--no-roof` và không bật PC sampling. Cả ba lệnh profile và analyze đều hoàn
tất thành công.
