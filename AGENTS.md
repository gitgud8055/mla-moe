# AGENTS.md

## Mục tiêu

Mục tiêu cuối cùng là tăng throughput `getp` trên AMD MI250/gfx90a cho
`dsv2lite` và `glm47` nhưng vẫn đạt toàn bộ accuracy gate trên dữ liệu chuẩn.
Throughput tăng nhưng accuracy không đạt thì không được coi là thành công.

## Phạm vi thay đổi

- Code tối ưu chỉ nằm trong `src/getp_run.hip` và `src/kernels/*.hip`.
- Có thể cập nhật `docs/` để lưu bằng chứng và kết quả tối ưu.
- Không sửa `Makefile`, `include/`, `tests/`, dữ liệu golden, CPU reference,
  scoring hay timing harness.
- Không hard-code prompt/output, bỏ bớt công việc, hoặc chuyển công việc được
  tính giờ ra ngoài `inference()`. `warm_up()` chỉ dùng cho chuẩn bị hợp lệ như
  cấp phát và upload weight.
- Giữ nguyên thay đổi không liên quan của người dùng; không tự ý dọn worktree.

## Accuracy gate bắt buộc

Nguồn chuẩn duy nhất cho input, prompt, completion và reference của mọi test,
eval, benchmark, profile và scoring là `data/dsv2lite/` và `data/glm47/`.
Không dùng các bản sao trong `tests/eval/*`. Thư mục `tests/` chỉ cung cấp mã
harness; không dùng README hay numerical gate cũ làm tiêu chí accuracy.

Accuracy gate chính thức duy nhất là chạy `score_completions.py` trên dữ liệu
trong `data/` và đạt đồng thời:

- METEOR `>= 0.30`;
- BERTScore-F1 `>= 0.91`.

Sequence accuracy và token accuracy chỉ là diagnostic bổ sung. Thay đổi dùng
chung phải qua gate trên cả hai model; thay đổi đặc thù model phải qua model đó
và smoke-test model còn lại. Không báo speedup hoặc giữ optimization nếu gate
accuracy chưa đạt.

Các lệnh chuẩn:

```sh
make
./run "$DSV" getp data/dsv2lite/requests.txt /tmp/getp_dsv2lite.txt 64
uv run python score_completions.py /tmp/getp_dsv2lite.txt dsv2lite \
  --data-dir data/dsv2lite --model-dir "$DSV" \
  --meteor-threshold 0.30 --bertscore-f1-threshold 0.91
./run "$GLM" getp data/glm47/requests.txt /tmp/getp_glm47.txt 64
uv run python score_completions.py /tmp/getp_glm47.txt glm47 \
  --data-dir data/glm47 --model-dir "$GLM" \
  --meteor-threshold 0.30 --bertscore-f1-threshold 0.91
```

Dùng lệnh tương ứng với `GLM`/`glm47`, luôn truyền `-d/--data-dir` trỏ vào
`data/`. Phải chạy scorer đầy đủ trước khi chốt; kiểm tra token nhanh hơn chỉ là
chẩn đoán trong vòng lặp và không thay thế accuracy gate.
Mọi benchmark/profile cũng phải đọc request/prompt từ `data/`; không dùng
`make bench` hoặc `tests/bench/bench.py` với mặc định đang trỏ vào `tests/eval/`.

## Workflow tối ưu bắt buộc

Lặp đúng chu trình: **thử nghiệm/benchmark → profile → xác định nguyên nhân
gốc của bottleneck → khắc phục triệt để → kiểm tra accuracy →
benchmark/profile lại → lặp tiếp**.

1. Đọc `docs/optimization-log.md` và profile gần nhất; tái lập baseline trên
   cùng model, input lấy từ `data/`, steps, binary, GPU và số lần lặp.
2. Từ profile cũ, nêu một giả thuyết có thể bác bỏ: kernel/cơ chế nào đang giới
   hạn, bằng counter/timing nào, và thay đổi dự kiến tác động metric nào.
3. Sửa đến nguyên nhân gốc của bottleneck, không dùng workaround hoặc tối ưu
   che triệu chứng. Build và chạy test/accuracy liên quan.
4. Benchmark không gắn profiler để lấy median; sau đó profile đúng workload để
   xác nhận nguyên nhân. Không dùng thời gian bị profiler làm chậm làm số perf.
5. So sánh trước/sau cùng cấu hình. Chỉ giữ thay đổi khi bottleneck được xử lý
   triệt để, throughput tăng lặp lại được và accuracy đạt gate; nếu không, ghi
   nhận và bỏ phần thay đổi của chính thử nghiệm đó.

## Công cụ profile ROCm Compute

Dùng `docs/install_rocprof_compute.sh` để cài rocprof-compute; profile request trong `data/` bằng `rocprof-compute profile`, phân tích bằng `rocprof-compute analyze`, và chỉ lấy TPS từ lần chạy không profiler.
Profile phải ghi vào thư mục artifact riêng bằng `--output-directory`; dùng `rocprof-compute analyze -p <workload_dir> > analysis.txt 2>&1`, không commit raw profile dung lượng lớn.

## Nhật ký bắt buộc

Tạo/duy trì `docs/optimization-log.md`. Sau **mỗi** thử nghiệm, ghi ngắn gọn:

- ngày, commit/worktree, GPU, model, workload và command tái lập;
- dữ kiện từ profile trước → suy luận → giả thuyết của thử nghiệm;
- thay đổi đã làm;
- kết quả trước/sau: median TPS, prefill/decode/TPOT khi liên quan, mọi accuracy
  metric và độ biến thiên;
- kernel/counter/profile cần thiết để chứng minh bottleneck đã đổi hay chưa;
- kết luận `KEEP`/`REJECT` và thử nghiệm kế tiếp.

Không đưa raw profile dung lượng lớn vào git. Ghi command, đường dẫn artifact và
trích các số liệu đủ để người khác kiểm chứng. Khi bàn giao, nêu speedup so với
baseline, accuracy của cả hai model, bottleneck còn lại và các thử nghiệm thất bại
đáng chú ý.
