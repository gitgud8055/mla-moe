# Báo cáo profile kernel MLA-MoE

Ngày chạy: 2026-08-14  
Máy: `mv-mi250-08`, AMD Instinct MI250X/MI250 (`gfx90a`)  
Model: DeepSeek-V2-Lite, 27 layer, hidden size 2048, vocab 102400

## Kết luận nhanh

- Workload có **2.624 dispatch** thuộc **29 kernel unique**.
- Bốn kernel tốn nhiều thời gian nhất chiếm **76,10% tổng thời lượng dispatch được ghi nhận**:
  `mfma_gemm` 40,03%, `grouped_expert_gate_up` 17,23%,
  `grouped_expert_down` 10,29%, và `mfma_gate_up_swiglu` 8,55%.
- Với prompt ngắn này, attention không phải bottleneck chính. `flash_prefill` chỉ chiếm
  0,41%; phần FFN của prefill chiếm 24,778/35,760 ms, tương đương 69,3% prefill.
- Các kernel đứng đầu chưa lấp đầy GPU: `mfma_gemm` chỉ có VALU busy 4,8% và MFMA
  utilization 1,2%; hai grouped-expert kernel có VALU busy 17,1–20,6% và MFMA
  utilization 4,6–5,5%. Đây là dấu hiệu batch/M còn nhỏ và workload bị chia thành
  nhiều launch hơn là đạt giới hạn compute của MI250.
- Lưu lượng đọc cộng dồn của các kernel ứng dụng khoảng **30,0 GiB** cho workload này,
  trong khi L2 hit tổng hợp chỉ khoảng 14,3%. Weight streaming và khả năng tái sử dụng
  weight của MoE là hướng tối ưu quan trọng.

## Workload

```text
5 request giống nhau: "The capital of France is"
prompt tokens: 5/request
batch: 5
max_steps: 2
KV capacity: 64
GPU vật lý: 1 (rocprof agent/gpu-id 3)
```

Lệnh workload tương đương:

```bash
HIP_VISIBLE_DEVICES=1 GETP_BATCH_SIZE=5 GETP_KV_CAPACITY=64 \
./run /remote/vast0/share-mv/deepseek-ai/DeepSeek-V2-Lite \
  getp <5-short-prompts> /tmp/out.txt 2
```

Workload ngắn có chủ ý: nó chạy cả batched prefill và decode batch 5, kích hoạt nhánh
MFMA hiện tại, nhưng tránh trace quá lớn từ request dài. Kết luận attention sẽ thay đổi
với sequence dài hơn.

## Wall-clock và phase timing

Run timing dưới profiler legacy:

| Chỉ số | Kết quả |
|---|---:|
| Warm-up/weight upload | 14,095 s |
| Inference | 81,051 ms |
| Throughput | 123,379 tok/s |
| Generated tokens | 10 |

Run riêng bằng HIP events để chia phase:

| Phase | Thời gian | Tỷ lệ inference |
|---|---:|---:|
| Prefill | 35,760 ms | 43,3% |
| Decode, 2 step | 45,030 ms | 54,5% |
| Host/khác | khoảng 1,81 ms | 2,2% |

Chi tiết prefill:

| Thành phần | Thời gian | Tỷ lệ prefill |
|---|---:|---:|
| QKV + RoPE + absorb | 5,505 ms | 15,4% |
| Flash attention | 0,794 ms | 2,2% |
| Output projection | 2,944 ms | 8,2% |
| FFN/MoE | 24,778 ms | 69,3% |
| Launch/sync/khác | khoảng 1,739 ms | 4,9% |

## Tất cả kernel theo thời gian

`GPU %` được chuẩn hóa theo tổng `DurationNs` của 2.624 dispatch. Tổng duration cộng
dồn là 112,173 ms; không dùng con số này làm wall-clock vì profiler legacy đo từng
dispatch và counter/timestamp instrumentation khác với timer end-to-end. Dùng bảng
phase phía trên cho wall-clock thực tế.

| Kernel | Calls | Total ms | GPU % | Avg us | P50 us | P95 us | Max us | Grid min–max | WG | VGPR | SGPR | LDS B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `batch_decode_kernels::mfma_gemm` | 540 | 44,901 | 40,03 | 83,1 | 69,2 | 125,6 | 500,8 | 256–409600 | 256 | 24 | 48 | 11264 |
| `batch_decode_kernels::grouped_expert_gate_up<16, 1>` | 104 | 19,324 | 17,23 | 185,8 | 130,8 | 367,9 | 387,2 | 349184–394240 | 256 | 36 | 48 | 19968 |
| `batch_decode_kernels::grouped_expert_down<16, 1>` | 104 | 11,548 | 10,29 | 111,0 | 78,9 | 214,5 | 227,2 | 507904–573440 | 256 | 24 | 48 | 11264 |
| `batch_decode_kernels::mfma_gate_up_swiglu` | 108 | 9,592 | 8,55 | 88,8 | 82,9 | 101,8 | 212,0 | 11264–87552 | 256 | 32 | 48 | 19968 |
| `batch_decode_kernels::mfma_head_gemm` | 216 | 3,332 | 2,97 | 15,4 | 16,6 | 22,3 | 25,8 | 8192–65536 | 256 | 24 | 48 | 11264 |
| `batch_decode_kernels::latent_context` | 108 | 2,785 | 2,48 | 25,8 | 31,5 | 31,8 | 35,2 | 524288–2621440 | 256 | 16 | 32 | 0 |
| `batch_decode_kernels::gemv` | 81 | 2,361 | 2,10 | 29,1 | 20,5 | 28,5 | 575,8 | 4096–6553600 | 256 | 16 | 32 | 0 |
| `batch_decode_kernels::routed_shared_gate_up` | 26 | 2,342 | 2,09 | 90,1 | 90,2 | 91,7 | 92,3 | 540672 | 256 | 20 | 48 | 0 |
| `batch_decode_kernels::rmsnorm` | 275 | 2,219 | 1,98 | 8,1 | 7,4 | 8,8 | 134,1 | 256–6400 | 256 | 12 | 32 | 1024 |
| `batch_decode_kernels::reduce_routes_add` | 104 | 1,838 | 1,64 | 17,7 | 15,5 | 25,3 | 25,8 | 1280–6400 | 256 | 12 | 32 | 0 |
| `batch_decode_kernels::routed_shared_down` | 26 | 1,676 | 1,49 | 64,4 | 64,6 | 65,9 | 66,4 | 131072 | 256 | 24 | 48 | 0 |
| `batch_decode_kernels::align_routes` | 104 | 1,612 | 1,44 | 15,5 | 15,8 | 18,9 | 19,4 | 256 | 256 | 12 | 64 | 1024 |
| `moe_kernels::router_topk_wave64_kernel<64>` | 130 | 1,607 | 1,43 | 12,4 | 12,4 | 14,6 | 16,5 | 64–1600 | 64 | 16 | 48 | 0 |
| `batch_decode_kernels::gate_up_swiglu` | 27 | 0,991 | 0,88 | 36,7 | 33,9 | 36,0 | 112,8 | 180224–700416 | 256 | 24 | 32 | 0 |
| `batch_decode_kernels::dual_gemv` | 27 | 0,984 | 0,88 | 36,5 | 36,3 | 37,7 | 38,2 | 233472 | 256 | 16 | 32 | 0 |
| `batch_decode_kernels::argmax` | 5 | 0,916 | 0,82 | 183,2 | 184,3 | 186,7 | 186,9 | 256–1280 | 256 | 8 | 32 | 2048 |
| `batch_decode_kernels::kv_norm_rope` | 108 | 0,688 | 0,61 | 6,4 | 6,2 | 7,8 | 9,0 | 256–1280 | 256 | 24 | 32 | 1024 |
| `batch_decode_kernels::attention_score` | 108 | 0,663 | 0,59 | 6,1 | 6,2 | 6,9 | 13,9 | 1024–35840 | 256 | 16 | 32 | 0 |
| `attention_kernels::softmax_rows_kernel<256>` | 108 | 0,572 | 0,51 | 5,3 | 5,0 | 6,7 | 8,2 | 4096–20480 | 256 | 20 | 48 | 1024 |
| `batch_decode_kernels::flash_prefill<8>` | 27 | 0,465 | 0,41 | 17,2 | 16,5 | 22,5 | 26,1 | 25600 | 256 | 64 | 96 | 9216 |
| `__amd_rocclr_copyBuffer.kd` | 89 | 0,451 | 0,40 | 5,1 | 4,8 | 7,3 | 8,3 | 512 | 512 | 20 | 32 | 0 |
| `batch_decode_kernels::q_rope` | 81 | 0,407 | 0,36 | 5,0 | 5,0 | 5,3 | 8,0 | 5120 | 256 | 24 | 32 | 0 |
| `batch_decode_kernels::q_absorb_rope` | 27 | 0,309 | 0,28 | 11,4 | 11,2 | 12,9 | 13,4 | 524288 | 256 | 36 | 32 | 0 |
| `batch_decode_kernels::value_up` | 27 | 0,234 | 0,21 | 8,7 | 8,6 | 9,4 | 10,1 | 131072 | 256 | 20 | 32 | 0 |
| `batch_decode_kernels::prefill_kv_norm_rope` | 27 | 0,171 | 0,15 | 6,3 | 5,9 | 8,0 | 8,2 | 6400 | 256 | 24 | 32 | 1024 |
| `batch_decode_kernels::prefill_q_rope` | 27 | 0,135 | 0,12 | 5,0 | 5,0 | 5,1 | 5,3 | 25600 | 256 | 24 | 32 | 0 |
| `batch_decode_kernels::embedding` | 5 | 0,024 | 0,02 | 4,9 | 4,8 | 5,3 | 5,4 | 2048–51200 | 256 | 8 | 32 | 0 |
| `__amd_rocclr_fillBufferAligned.kd` | 4 | 0,020 | 0,02 | 5,1 | 4,8 | 6,3 | 6,6 | 256 | 256 | 12 | 32 | 0 |
| `batch_decode_kernels::gather_last_hidden` | 1 | 0,005 | 0,00 | 5,1 | 5,1 | 5,1 | 5,1 | 10240 | 256 | 4 | 16 | 0 |

## Hardware counters của 15 kernel lớn nhất

Các tỷ lệ utilization/busy được lấy trung bình theo thời lượng dispatch. `VALU inst/wave`
được weighted theo số wave. L2 hit được weighted gần đúng theo read+write traffic. Counter
collection serialize kernel và làm chương trình chậm hơn nhiều, nên không dùng thời gian của
counter run để tính performance.

| Kernel | Calls | Time % | Waves | VALU util % | VALU busy % | MFMA util % | VALU inst/wave | Read MiB | Write MiB | Mem busy % | Mem stall % | L2 hit % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `batch_decode_kernels::mfma_gemm` | 540 | 40,03 | 91460 | 99,8 | 4,8 | 1,2 | 4710,2 | 5408,9 | 7,6 | 6,9 | 0,1 | 23,0 |
| `batch_decode_kernels::grouped_expert_gate_up<16, 1>` | 104 | 17,23 | 585728 | 98,4 | 17,1 | 5,5 | 992,6 | 11571,4 | 11,6 | 24,3 | 0,2 | 10,3 |
| `batch_decode_kernels::grouped_expert_down<16, 1>` | 104 | 10,29 | 851968 | 99,5 | 20,6 | 4,6 | 498,6 | 5801,0 | 23,9 | 26,5 | 0,3 | 13,0 |
| `batch_decode_kernels::mfma_gate_up_swiglu` | 108 | 8,55 | 26300 | 98,3 | 7,3 | 2,9 | 5120,0 | 2653,8 | 1,1 | 11,0 | 0,1 | 20,5 |
| `batch_decode_kernels::mfma_head_gemm` | 216 | 2,97 | 86400 | 96,8 | 4,4 | 0,9 | 506,4 | 474,6 | 0,2 | 6,8 | 0,5 | 26,9 |
| `batch_decode_kernels::latent_context` | 108 | 2,48 | 3538944 | 60,2 | 65,0 | 0,0 | 111,0 | 2,4 | 0,1 | 30,9 | 4,8 | 98,0 |
| `batch_decode_kernels::gemv` | 81 | 2,10 | 214656 | 99,1 | 22,6 | 0,0 | 468,8 | 952,6 | 0,4 | 36,9 | 0,6 | 17,9 |
| `batch_decode_kernels::routed_shared_gate_up` | 26 | 2,09 | 219648 | 93,3 | 27,9 | 0,0 | 536,0 | 1716,7 | 0,7 | 42,4 | 0,7 | 5,7 |
| `batch_decode_kernels::rmsnorm` | 275 | 1,98 | 8940 | 99,4 | 0,2 | 0,0 | 187,2 | 18,9 | 0,0 | 2,5 | 0,3 | 47,8 |
| `batch_decode_kernels::reduce_routes_add` | 104 | 1,64 | 4160 | 100,0 | 0,2 | 0,0 | 290,0 | 57,0 | 0,4 | 4,2 | 0,3 | 23,4 |
| `batch_decode_kernels::routed_shared_down` | 26 | 1,49 | 53248 | 98,2 | 22,6 | 0,0 | 1340,0 | 859,1 | 0,0 | 44,6 | 0,1 | 10,4 |
| `batch_decode_kernels::align_routes` | 104 | 1,44 | 416 | 22,8 | 0,0 | 0,0 | 108,9 | 0,2 | 0,0 | 0,4 | 0,1 | 60,7 |
| `moe_kernels::router_topk_wave64_kernel<64>` | 130 | 1,43 | 1066 | 74,3 | 0,1 | 0,0 | 395,0 | 0,6 | 0,0 | 0,8 | 0,4 | 83,7 |
| `batch_decode_kernels::gate_up_swiglu` | 27 | 0,88 | 84160 | 94,2 | 26,9 | 0,0 | 620,0 | 657,8 | 0,0 | 41,9 | 0,7 | 5,8 |
| `batch_decode_kernels::dual_gemv` | 27 | 0,88 | 98496 | 98,9 | 21,3 | 0,0 | 439,0 | 385,1 | 0,0 | 35,0 | 0,8 | 11,5 |

Tổng hợp kernel ứng dụng, không tính hai kernel runtime `__amd_rocclr_*`:

| Counter | Giá trị |
|---|---:|
| Wavefronts | 6.234.454 |
| VALU utilization | 95,9% |
| VALU busy | 11,4% |
| MFMA utilization | 2,17% |
| Read traffic | 30.727,9 MiB |
| Write traffic | 46,0 MiB |
| Memory busy | 15,0% |
| Memory stalled | 0,32% |
| L2 hit, traffic-weighted | 14,3% |

`VALU utilization` cao nhưng `VALU busy` thấp không mâu thuẫn: các wave đang chạy có
lane utilization tốt, nhưng GPU tổng thể chỉ bận một phần nhỏ thời gian/CU vì shape nhỏ và
nhiều launch tuần tự.

## Thứ tự tối ưu đề xuất

1. **Tối ưu `mfma_gemm` trước.** Nó chiếm 40,03%, có 540 launch và MFMA utilization chỉ
   1,2%. Gom row/batch lớn hơn, giảm số launch, và chọn tile riêng theo shape có khả năng
   đem lại lợi ích lớn nhất.
2. **Tập trung grouped MoE gate/up/down.** Hai kernel grouped chiếm 27,52%, đọc khoảng
   17,0 GiB và chỉ đạt L2 hit 10–13%. Tăng reuse weight trong tile, gom route/expert tốt hơn,
   và tránh expert micro-batch quá nhỏ.
3. **Giảm launch nhỏ/fuse kernel.** Có 2.624 dispatch cho chỉ 10 output token. Các ứng viên
   fuse: RMSNorm + projection, router + align/count, và route reduction + residual khi layout
   cho phép.
4. **Xử lý outlier theo shape.** `mfma_gemm` max 500,8 us so với median 69,2 us; `gemv`
   max 575,8 us so với median 20,5 us. Tách thống kê theo `(M,N,K)`/grid để tìm đúng shape
   gây tail latency.
5. **`argmax` là mục tiêu phụ.** Chỉ 5 call nhưng mỗi call khoảng 183 us trên vocab 102400;
   có thể tối ưu reduction/fuse logits sau khi xử lý MFMA và MoE.
6. **Chưa ưu tiên attention cho workload này.** Với sequence dài, cần profile lại vì tỷ trọng
   attention/flash sẽ tăng theo KV length.

## Trạng thái rocprof-compute trên node

Đã thử `rocprof-compute 3.4.0` bằng cả backend mặc định `rocprofiler-sdk` và backend CLI
`rocprofv3`, định dạng CSV lẫn ROCPD. Workload chạy xong nhưng profiler abort khi finalize:

```text
ring_buffer: munmap failed: Invalid argument
mmap failed with errno 22 :: Invalid argument
```

Một smoke kernel chỉ có 1 dispatch cũng gặp cùng lỗi, nên đây là lỗi cài đặt/backend profiler
trên node, không phải do kernel MLA-MoE. Để có báo cáo usable, số liệu ở trên được thu bằng
`rocprof` ROCm 7.2.2 legacy backend: một timing trace và ba pass counter tương ứng nhóm
compute, memory và L2. Source đang sửa `src/getp_run.hip` không bị thay đổi.
