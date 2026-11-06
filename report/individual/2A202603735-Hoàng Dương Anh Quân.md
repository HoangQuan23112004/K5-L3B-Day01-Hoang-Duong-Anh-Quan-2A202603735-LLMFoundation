# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603735
- **Họ và tên:**Hoàng Dương Anh Quân
- **SEED** (in ở ô đầu notebook):9207

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

Tôi so sánh các phương pháp phân loại SMS trên cùng dữ liệu sau khi loại bỏ tin nhắn trùng lặp. Dữ liệu gồm 5.169 tin: 4.516 ham và 653 spam; chia thành 3.619 tin train, 775 tin validation và 775 tin test. SEED của bài là 9207.

| Phương pháp | F1 test | F1 SUB | p95 ms/tin | USD/1.000 tin |
|---|---:|---:|---:|---:|
| BoW + Logistic Regression | 0,826 | 0,750 | 0,067 | 0,000 |
| BoW + MLP | 0,931 | 0,889 | 0,519 | 0,000 |
| Embedding + kNN | 0,920 | 0,947 | 117,640 | 0,000 |
| Embedding + Logistic Regression | 0,933 | 0,950 | 29,665 | 0,000 |
| Embedding + MLP head | 0,933 | 0,923 | 28,868 | 0,000 |
| LLM nhỏ zero-shot | Không đo toàn bộ test | 1,000 | 806,274 | 0,022 |


Lưu ý về phép đo: bảng gốc ghi “SUB 40”, nhưng ô thí nghiệm LLM đã thực hiện SUB = SUB[:10]. Vì vậy, kết quả LLM được đo trên 10 tin, còn các kết quả SUB của phương pháp trước đó được đo trên 40 tin. Chưa thể coi đây là so sánh chất lượng hoàn toàn công bằng.

Ở lần chạy zero-shot khác trong cùng notebook, LLM vẫn đạt F1 = 1,000 nhưng p95 tăng lên 7.722,32 ms. Điều này cho thấy độ trễ API biến động đáng kể. Few-shot đạt F1 = 1,000 trên 10 tin, với p95 = 1.046,10 ms và tổng chi phí ước lượng khoảng 0,00055 USD, cao hơn mức khoảng 0,00022 USD của zero-shot.

Các giá trị chi phí bằng 0 của model chạy tại máy chỉ biểu thị không tính phí API trong notebook, chưa bao gồm chi phí máy chủ và điện.

## 3. Dự đoán và thực tế

Tôi chọn ba dự đoán có khác biệt rõ với kết quả.

Thứ nhất, dự đoán về bỏ ReLU. Tôi dự đoán bỏ ReLU sẽ làm F1 giảm. Thực tế, BoW + MLP có ReLU và mô hình không ReLU đều đạt F1 test = 0,931. Điều này cho thấy trên lần chia dữ liệu và cách huấn luyện này, lớp phi tuyến chưa tạo ra cải thiện quan sát được ở F1. Không thể từ đó kết luận ReLU không có tác dụng với mọi bài toán.

Thứ hai, dự đoán về few-shot. Tôi dự đoán thêm ví dụ sẽ làm F1 giảm. Thực tế, zero-shot và few-shot đều đạt F1 = 1,000 trên 10 tin. Few-shot không làm giảm chất lượng nhưng cũng chưa chứng minh được cải thiện, trong khi chi phí ước lượng tăng khoảng 2,5 lần theo các số đã in. Tập đánh giá nhỏ và zero-shot đã đúng hết nên khó quan sát lợi ích thêm của few-shot.

Thứ ba, dự đoán về stress test. Tôi dự đoán accuracy của BoW + MLP khoảng 68%, thực tế chỉ đạt 52%, thấp hơn 16 điểm phần trăm. Ngược lại, LLM với prompt phòng thủ đạt 100%, cao hơn dự đoán 90%. Tôi đã đánh giá quá cao khả năng tổng quát hóa của BoW + MLP: model đạt điểm cao trên test thường nhưng gặp khó khi spam được viết tự nhiên hoặc cố tình biến dạng từ khóa.

## 4. Phân tích 50 case khó

| Phương pháp | F1 SUB đã lưu | F1 stress test | Accuracy stress test |
|---|---:|---:|---:|
| BoW + Logistic Regression | 0,750 | 0,333 | 52% |
| BoW + MLP | 0,889 | 0,333 | 52% |
| Embedding + kNN | 0,947 | 0,750 | 72% |
| Embedding + Logistic Regression | 0,950 | 0,612 | 62% |
| Embedding + MLP head | 0,923 | 0,612 | 62% |
| LLM nhỏ zero-shot | 1,000 | 0,967 | 96% |
| LLM nhỏ + prompt phòng thủ | Không có phép đo SUB riêng | 1,000 | 100% |

BoW + MLP giảm nhiều nhất: F1 từ 0,889 xuống 0,333, giảm 0,556, tương đương 55,6 điểm phần trăm. Model này đúng 0% ở nhóm spam giọng tự nhiên và 0% ở nhóm né từ khóa. Điều đó cho thấy kết quả tốt trên test thường chưa bảo đảm model xử lý được các cách diễn đạt khác dữ liệu train.

Hai case cụ thể mà BoW + MLP phân loại sai là:

Case 11 — spam giọng tự nhiên:
Hey it's Jess from the gym we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it

Nhãn đúng là spam. Tin nhắn quảng cáo được viết thân mật như người quen. Biểu diễn BoW không giữ đầy đủ quan hệ và mục đích của cả câu, nên model có thể bị các dấu hiệu hội thoại bình thường che khuất nội dung khuyến mãi.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu 10 triệu SMS/ngày, tương đương khoảng 116 tin/giây trung bình, và p95 < 50 ms/tin, tôi chọn BoW + MLP làm phương án xử lý chính ban đầu.

Phương pháp này đạt:
- F1 test 0,931.
- p95 0,519 ms/tin.
- Throughput cả lô khoảng 16.682 tin/giây trong môi trường thử nghiệm.
Embedding + Logistic Regression có F1 test nhỉnh hơn, 0,933, và p95 29,665 ms, nhưng throughput đo được chỉ khoảng 52 tin/giây trên một môi trường chạy. Cần thêm tài nguyên, chạy song song hoặc tối ưu trước khi đáp ứng lưu lượng mục tiêu.

Đánh đổi lớn nhất của BoW + MLP là chất lượng trên stress test thấp: accuracy chỉ 52%. Do đó, tôi đề xuất dùng thêm nhánh kiểm tra cho tin không chắc chắn, chẳng hạn chuyển sang mô hình embedding hoặc đưa vào hàng đợi LLM để phân tích bổ sung. Nhánh LLM cần xử lý bất đồng bộ vì p95 API đã đo vượt xa 50 ms.

Theo mức 0,022 USD/1.000 tin trong bảng zero-shot, nếu áp dụng cùng chi phí trung bình cho 10 triệu SMS/ngày thì phí token ước lượng là:

220 USD/ngày × 30 ngày = 6.600 USD/tháng.

Đây là phép ngoại suy từ mẫu nhỏ theo bảng giá trong notebook, chưa tính thay đổi độ dài tin, retry, hạ tầng hoặc giới hạn quota. Trước khi triển khai cần kiểm thử tải và đánh giá lại trên dữ liệu thực tế.

## 7. AI log và điều đã học

| Nội dung hỏi AI | AI hỗ trợ | Việc sửa và kiểm chứng |
|---|---|---|
| Cách sử dụng Gemini API key và Colab Secrets | Giải thích nơi lưu key và phân biệt key với base_url | Kiểm tra cấu hình và chạy thử API |
| Hoàn thành TODO 4.1–4.4 | Viết hàm gọi API, tạo prompt, đọc nhãn và phân loại hàng loạt | Chạy bộ kiểm tra giả lập; sửa lỗi thụt lề và thêm kết quả trùng |
| Lỗi _run_check chưa được định nghĩa | Tìm hàm trong ô công cụ của Phần 0 | Chạy lại các ô khởi tạo cần thiết |
| Lỗi 429 và tham số minimal | Giải thích giới hạn API và tham số suy luận không tương thích | Thêm khoảng nghỉ giữa lượt gọi và kiểm tra lại lời gọi Gemini |
| Giải thích câu hỏi và viết báo cáo | Hỗ trợ phân tích, tổng hợp kết quả | Đối chiếu số liệu với output notebook; thay các nhận xét giả định trước đó bằng kết quả thực tế |
Tôi hiểu rằng model đạt F1 cao trên test thường vẫn có thể thất bại khi cách diễn đạt thay đổi. Tôi cũng học được rằng dùng LLM cần quản lý quota, tham số và độ trễ, không chỉ có API key. Khi so sánh các phương pháp, cần kiểm tra chúng có được đo trên cùng tập dữ liệu và cùng điều kiện hay không.
