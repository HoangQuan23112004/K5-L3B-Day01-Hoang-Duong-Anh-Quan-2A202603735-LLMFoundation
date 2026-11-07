# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> - Ở mức `temperature = 0.0`, câu trả lời mang tính tất định (deterministic), tập trung vào các sự thật phổ biến nhất (như hang Sơn Đoòng hoặc sản lượng xuất khẩu cà phê Robusta) với cấu trúc câu chuẩn mực, lặp lại giống nhau qua các lần chạy.
> - Khi tăng lên `0.5` và `1.0`, ngôn từ trở nên tự nhiên, đa dạng hơn, mô hình bắt đầu chọn các khía cạnh văn hoá hoặc địa danh ít phổ biến hơn.
> - Ở mức `temperature = 1.5`, độ ngẫu nhiên quá cao khiến câu văn bắt đầu lan man, dùng từ ngữ bất thường và dễ xuất hiện hiện tượng "ảo giác" (hallucination) / sai lệch sự thật.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đặt temperature trong khoảng **0.0 đến 0.2** (tối ưu là 0.0). Lý do vì chatbot hỗ trợ khách hàng ưu tiên hàng đầu tính chính xác và nhất quán về thông tin chính sách, giá cả và giải pháp kỹ thuật; mức nhiệt độ thấp giúp triệt tiêu tính ngẫu nhiên và giảm thiểu tối đa ảo giác, đảm bảo câu trả lời luôn đồng nhất và đáng tin cậy.

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> `gpt-6.1-terra` đắt hơn `gpt-6-luna` **đúng 10 lần (10x)**. Cả đơn giá token input ($1.00 so với $0.10) và token output ($5.00 so với $0.50) của terra đều gấp 10 lần luna, nên với bất kỳ tỷ lệ input/output nào trong 350 token, tổng chi phí luôn chênh lệch 10 lần. Ví dụ với 30.000 lượt gọi/ngày (giả sử 150 token input + 200 token output), chi phí của luna chỉ khoảng $3.45/ngày, trong khi terra tốn tới $34.50/ngày — chênh lệch hơn $900 mỗi tháng.

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> - Xứng đáng dùng `gpt-6.1-terra`: Các tác vụ yêu cầu suy luận đa bước chuyên sâu, phân tích hợp đồng pháp lý / y tế, lập trình sinh mã nguồn phức tạp hoặc trích xuất dữ liệu có cấu trúc từ văn bản nhập nhằng, nơi mà một sai sót nhỏ có thể gây thiệt hại nghiêm trọng.
> - Nên dùng `gpt-6-luna`: Các tác vụ lặp đi lặp lại với số lượng lớn (high-volume) nhưng độ phức tạp thấp như phân loại tin nhắn (spam/ham), tóm tắt văn bản ngắn, trích xuất thực thể đơn giản, sửa lỗi chính tả hoặc làm bộ lọc định tuyến (router) ban đầu trước khi quyết định gọi model lớn.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các giao diện tương tác trực tiếp theo thời gian thực với con người (như chatbot, giao diện soạn thảo văn bản hoặc trợ lý lập trình). Đối với các câu trả lời dài, streaming giúp giảm chỉ số Time-To-First-Token (TTFT) xuống dưới 1 giây, cho phép người dùng đọc nội dung ngay khi nó đang được sinh ra, loại bỏ cảm giác chờ đợi gây khó chịu (5–15 giây) như khi dùng non-streaming. Ngược lại, non-streaming lại phù hợp hơn trong các tác vụ ngầm (background jobs, batch pipeline), giao tiếp giữa các hệ thống backend-to-backend cần nhận trọn vẹn payload JSON để parse dữ liệu vào database, hoặc khi cần kiểm duyệt an toàn (content safety moderation) toàn bộ câu trả lời trước khi trả về cho client.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
