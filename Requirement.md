DATAFLOW 2026: THE ALCHEMY OF MINDS

**CÂU LẠC BỘ TOÁN TIN HAMIC**

* **Website:** [https://dataflow.hamictoantin.com/vi](https://dataflow.hamictoantin.com/vi)
* **Fanpage:** [https://www.facebook.com/toantinhamic](https://www.facebook.com/toantinhamic)
* **Email:** hamic@hus.edu.vn

---

AUTOSCALING ANALYSIS

# PHẦN 1: GIỚI THIỆU BÀI TOÁN

Trong quản trị hệ thống đám mây, việc cấp phát tài nguyên cố định thường dẫn đến hai vấn đề: lãng phí tài nguyên khi ít người truy cập hoặc sập hệ thống khi lượng truy cập tăng đột biến.

**Nhiệm vụ:** Thí sinh đóng vai trò là kỹ sư dữ liệu xây dựng hệ thống phân tích nhật ký truy cập (log).

**Trọng tâm cốt lõi là giải quyết Bài toán Hồi quy và Bài toán tối ưu chi phí:** Xây dựng mô hình học máy để dự báo giá trị số thực của lưu lượng truy cập (Số request và số byte) trong tương lai, kết quả hồi quy là đầu vào để thuật toán tự động điều chỉnh số lượng máy chủ (Autoscaling) nhằm tối ưu hóa chi phí vận hành.

# PHẦN 2: BỘ DỮ LIỆU

Bộ dữ liệu chứa thông tin hai tháng về tất cả các yêu cầu HTTP gửi đến máy chủ WWW của 1 công ty.

* Nhật ký đầu tiên được thu thập từ 00:00:00 ngày 1 tháng 7 năm 1995 đến 23:59:59 ngày 31 tháng 7 năm 1995, tổng cộng là 31 ngày.
* Nhật ký thứ hai được thu thập từ 00:00:00 ngày 1 tháng 8 năm 1995 đến 23:59:59 ngày 31 tháng 8 năm 1995, tổng cộng là 31 ngày.
* Các dấu thời gian có độ phân giải 1 giây.

**Lưu ý:** Từ 14:52:01 ngày 01/08/1995 đến 04:36:13 ngày 03/08/1995 không có truy cập nào được ghi lại, do máy chủ Web đã bị tắt vì Bão.

**Thành phần dữ liệu:** Dữ liệu gốc ở định dạng ASCII, với mỗi dòng tương ứng một yêu cầu. Thí sinh cần xử lý và trích xuất các trường thông tin:

* **Host (Nguồn):** Địa chỉ IP hoặc tên miền của máy khách gửi yêu cầu (Ví dụ: 199.72.81.55).
* **Timestamp (Thời gian):** Thời điểm yêu cầu được ghi nhận (Ví dụ: [01/Jul/1995:00:00:01 -0400]). Đây là trường quan trọng nhất để tạo chuỗi thời gian.
* **Request (Yêu cầu):** Chứa phương thức (GET/POST), đường dẫn tài nguyên (URL) và giao thức (Ví dụ: "GET /history/apollo/ HTTP/1.0").
* **HTTP Reply Code (Trạng thái):** Mã phản hồi từ máy chủ.
* **Bytes in the reply (Kích thước):** Dung lượng dữ liệu trả về.

**Quy định chia tập dữ liệu (Train/Test Split):**

* **Tập Huấn luyện (Train Set):** Dữ liệu của tháng 7 và 22 ngày đầu tiên của tháng 8.
* **Tập Kiểm thử (Test Set):** Dữ liệu của các ngày còn lại trong tháng 8.

**Yêu cầu tiền xử lý (Ingest + EDA):**

* Pipeline đọc log, chuẩn hóa timestamp, parse fields (IP, URL, status).
* Khai thác time series: hits/sec, error rate, spike detection.

# PHẦN 3: YÊU CẦU VỀ BÀI TOÁN HỒI QUY (DỰ ĐOÁN TẢI)

* **Mô hình hóa (Modeling):** Thí sinh thử nghiệm và lựa chọn tối thiểu 02 mô hình trong các nhóm sau:
    * Thống kê truyền thống: ARIMA, SARIMA.
    * Mô hình hiện đại: Prophet (Facebook).
    * Deep Learning: LSTM, RNN.
    * Machine Learning: GBDT (XGBoost/LightGBM).

* **Khung thời gian (Benchmarking):** Thực hiện tổng hợp dữ liệu và dự báo trên nhiều khung thời gian khác nhau để tìm ra phương án tối ưu:
    * Khung 1 phút (1m).
    * Khung 5 phút (5m).
    * Khung 15 phút (15m).

* **Các chỉ số đánh giá:** Hiệu quả của mô hình phải được đo lường bằng các chỉ số sau: RMSE, MSE, MAE, MAPE.

# PHẦN 4: BÀI TOÁN TỐI ƯU

* Thiết kế chính sách scaling (CPU/requests-based, predictive scaling).
* Mô phỏng/logic rules: Ví dụ scale-out khi dự báo > ngưỡng trong 5 phút liên tiếp. Cooldown để tránh flapping.
* Phân tích chi phí vs hiệu năng (định tính hoặc định lượng).

# PHẦN 5: TRIỂN KHAI (Demo)

* Dashboard (Streamlit/Dash): biểu đồ tải, dự báo, đề xuất scale events.
* API: `/forecast` & `/recommend-scaling`.
* (Tuỳ chọn) Simulator giả lập dàn máy với queue/latency.

# PHẦN 6: ĐIỂM CỘNG

* Phát hiện DDoS/spike bất thường (anomaly detection).
* Tích hợp hysteresis/cooldown thông minh, chống dao động.
* Report chi phí với giả định unit cost (đơn vị/server/giờ).

# PHẦN 7: TIÊU CHÍ ĐÁNH GIÁ

| Hạng mục | Tiêu chí chi tiết |
| --- | --- |
| **Tính đúng đắn & Hiệu quả** | Mô hình và logic giải quyết vấn đề hợp lý. Sử dụng các metric đánh giá chuẩn xác. Có quy trình kiểm thử chặt chẽ và độ tin cậy cao. |
| **Trình bày & Demo** | Slide thiết kế rõ ràng, thẩm mỹ, dẫn dắt vấn đề tốt. Demo sản phẩm mượt mà, trực quan. |
| **Giải pháp kỹ thuật** | Mô hình/thuật toán + ứng dụng/API/dashboard. Đánh giá bằng metric rõ ràng. Tài liệu: README.md, kiến trúc giải pháp, hướng dẫn chạy, phân tích dữ liệu. |
| **Tính sáng tạo & Ứng dụng** | Ý tưởng tiếp cận độc đáo, mới lạ. Giải pháp có giá trị thực tiễn cao và khả năng mở rộng (scalability) tốt. |
| **Tính hoàn thiện** | Mã nguồn sạch (Clean code), kiến trúc hệ thống rõ ràng. Tài liệu (README) đầy đủ hướng dẫn cài đặt. Kết quả có thể tái lập (reproducible). |

# PHẦN 8: YÊU CẦU NỘP BÀI

* **Bản mềm các tài liệu:** báo cáo (cần giới thiệu bài toán, tóm tắt và phân tích bài toán được giao) tối đa 30 trang, slide thuyết trình.
* **Mã nguồn:** Các tệp mã nguồn dự án (Jupyter Notebook / Python Script) chứa toàn bộ quá trình phân tích, xử lý dữ liệu và huấn luyện mô hình, README.md, script hướng dẫn chạy code (bắt buộc).
* **Notebook online:** Các notebook trên mạng (ví dụ: Kaggle notebook, Google Collab,...) các đội thi cần tải xuống và đồng thời gắn thêm link trực tiếp vào trong mã nguồn để nộp.
* **Github:** Link Github cần phải được gắn vào báo cáo, cần đảm bảo quyền truy cập vào các đường link này. Sau thời gian nộp bài, không được có bất kỳ commit nào lên Github.
* Link GitHub repo (public) gồm:
    * README.md (mẫu ở trong folder data của tài liệu)
    * Code huấn luyện/inference
    * Notebook phân tích dữ liệu (EDA)
    * Mô hình đã huấn luyện (nếu dung lượng cho phép) hoặc script tái tạo
    * Demo: API (FastAPI/Flask) hoặc UI (Streamlit/Dash)
    * Slide, báo cáo.
* **Video demo hệ thống:** (3 - 5 phút).
* **Thời gian báo cáo:** 20 phút, trong đó có 5 phút demo sản phẩm, 10 phút thuyết trình và 5 phút vấn đáp với ban giám khảo.

**LƯU Ý:**

* Đội thi nộp bản mềm dưới định dạng file pdf hoặc pptx, không nộp dưới dạng ảnh, ảnh scan từ báo cáo bản cứng.
* Bản nộp cần là nộp bản gốc cho BTC trước thời gian quy định.
* **Ngôn ngữ sử dụng:** Sử dụng ngôn ngữ tiếng Việt, một số thuật ngữ chuyên ngành khó dịch có thể sử dụng tiếng Anh nhưng cần có chú thích trong phụ lục.
* Các từ ngữ không sử dụng từ địa phương, không sử dụng teencode.
* Các từ ngữ viết tắt cần được chú thích rõ ràng trong phụ lục.