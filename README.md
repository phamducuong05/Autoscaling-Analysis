
# 📈 Autoscaling Analysis - Dataflow 2026

Dự án tập trung vào việc xây dựng hệ thống phân tích nhật ký truy cập (log) để dự báo lưu lượng và tối ưu hóa chi phí vận hành thông qua cơ chế tự động điều chỉnh số lượng máy chủ (Autoscaling).

---

## 📂 Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo cấu trúc tiêu chuẩn để đảm bảo tính tái lập và dễ dàng mở rộng:

```text
Autoscaling-Analysis/
├── api/                       # 🔌Backend API
│   ├── app.py                 # Entry point của API
│   └── schema.py              # Định nghĩa các schema dữ liệu vào ra
│
├── app/                       # 📊 Dashboard & API
│   ├── dashboard.py           # Trực quan hóa các kết quả
│
├── config/                    # ⚙️ Cấu hình toàn cục
│   ├── settings.py            # Cấu hình toàn cục: path, random seed, hằng số, environment variables
│   ├── train_config.yaml      # Cấu hình huấn luyện model: feature, hyperparameters, strategy
│   ├── autoscaling_config.yaml # ⚙️ Cấu hình Chiến lược Scaling (Quan trọng)
│
├── data/                      # 🔒 Quản lý dữ liệu
│   ├── raw/                   # Nhật ký gốc (ASCII) từ tháng 7 & 8/1995
│   ├── cleaned/               # Dữ liệu đã được làm sạch và chuẩn hóa
│
├── models/                    # 💾 Lưu trữ weight models đã huấn luyện
│
├── notebooks/                 # 📓 Jupyter Notebooks
│   ├── experimental/          # Code thử nghiệm
│   └── final/                 # Notebook sạch dùng cho báo cáo
│
├── output/                    # 📤 Kết quả đầu ra (Figures, Logs)
│
├── src/                       # 🧠 MÃ NGUỒN CHÍNH (Pipeline)
│   ├── __init__.py
│   ├── autoscaler.py          # Logic điều phối Server (Gen 2 AI)
│   ├── data_loader.py         # Pipeline đọc log & parse fields
│   ├── evaluation.py          # Metrics: RMSE, MSE, MAE, MAPE
│   ├── features.py            # Feature Engineering cho chuỗi thời gian
│   ├── optimization.py        # Thuật toán tối ưu hóa
│   └── utils.py               # Tiện ích: Logger, Save/Load Model
│
├── main.py                    # 🚀 ENTRY POINT: Chạy toàn bộ quy trình từ A-Z
├── run_benchmark.py           # 🏆 Script chạy Benchmark so sánh hiệu năng
├── requirements.txt           # Danh sách thư viện cần thiết
└── README.md                  # Hướng dẫn sử dụng dự án
```


# 🛠 Hướng dẫn Cài đặt Môi trường

### Bước 1: Tạo môi trường ảo với phiên bản python 3.10
Hãy sử dụng python 3.10 để đảm bảo tính tương thích tốt nhất
Tại thư mục gốc của dự án (`Learning-process-prediction/`), chạy lệnh:
```bash
python3.10 -m venv venv
```

### Bước 2: Kích hoạt môi trường (Activate)
*Mỗi lần bắt đầu làm việc, bạn phải chạy lệnh này.*

*   **Đối với Windows (Command Prompt/PowerShell):**
    ```bash
    .\venv\Scripts\activate
    ```
    *(Nếu thấy dấu `(venv)` hiện ở đầu dòng lệnh là thành công)*

*   **Đối với macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

### Bước 3: Cài đặt thư viện dự án
Sau khi kích hoạt môi trường, hãy cài đặt các thư viện cần thiết từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 5: Thêm Kernel vào Jupyter Notebook (QUAN TRỌNG)
Để chạy được Notebooks trong thư mục `notebooks/` với môi trường ảo vừa tạo:

1.  Cài đặt ipykernel:
    ```bash
    pip install ipykernel
    ```
2.  Gắn môi trường vào Jupyter:
    ```bash
    python -m ipykernel install --user --name=venv_learning_prediction --display-name "Python (Learning Prediction)"
    ```
3.  Khi mở Jupyter Notebook, chọn Kernel: **Kernel** -> **Change kernel** -> **Python (Learning Prediction)**.

---
### 🛑 Cách thoát môi trường
Khi làm xong việc, chạy lệnh:
```bash
deactivate
```

---

# 🚀 Hướng Dẫn Chạy Demo & Benchmark

## 1. Chạy Benchmark Hiệu Năng
Để so sánh hiệu năng giữa AI Autoscaler (Gen 2), Reactive Scaling và Static Capacity, chạy lệnh:

```bash
python run_benchmark.py
```
Kết quả báo cáo sẽ được lưu tại `evaluation_results/report.md` và biểu đồ tại `evaluation_results/benchmark_plot.png`.

## 2. Khởi chạy Dashboard Demo (Real-time)

Hệ thống Demo gồm 2 thành phần chính: **API Server** (Backend) và **Dashboard** (Frontend). Bạn cần mở 2 terminal riêng biệt để chạy chúng.

### Terminal 1: Khởi động API Server
```bash
# Kích hoạt môi trường ảo trước
source venv/bin/activate

# Chạy server (Port 8000)
uvicorn api.api:app --reload
```
*Server sẽ lắng nghe tại: http://localhost:8000*

### Terminal 2: Khởi động Dashboard
```bash
# Kích hoạt môi trường ảo trước
source venv/bin/activate

# Chạy Streamlit App
streamlit run app/dashboard.py
```
*Giao diện sẽ tự động mở tại: http://localhost:8501*

### 💡 Lưu ý
- Đảm bảo bạn đã huấn luyện model hoặc có sẵn model trong thư mục `models/` (đã có sẵn file `lstm_model.pth`).
- API và Dashboard hoạt động trên cơ chế **Stateful**: Dữ liệu lịch sử 12 bước (60 phút) được quản lý trong RAM của API Server. Reset server sẽ làm mất trạng thái (Cold Start).
