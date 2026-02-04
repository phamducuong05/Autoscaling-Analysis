
# 📈 Autoscaling Analysis - Dataflow 2026

Dự án tập trung vào việc xây dựng hệ thống phân tích nhật ký truy cập (log) để dự báo lưu lượng và tối ưu hóa chi phí vận hành thông qua cơ chế tự động điều chỉnh số lượng máy chủ (Autoscaling).

---

## 📂 Cấu trúc Dự án (Project Structure)

Dự án được tổ chức theo cấu trúc tiêu chuẩn để đảm bảo tính tái lập và dễ dàng mở rộng:

```text
Autoscaling-Analysis/
├── api/                           # 🔌 Backend API Server
│   ├── api.py                     # FastAPI entry point - Khởi tạo endpoints REST
│   ├── schema.py                  # Định nghĩa request/response schemas
│   ├── constants.py               # Hằng số của API (model paths, dimensions, buffers)
│   ├── feature_engineering.py     # Chuẩn bị features cho model prediction
│   ├── model_loader.py            # Load LSTM model weights vào memory
│   └── __pycache__/               # Compiled Python cache
│
├── app/                           # 📊 Frontend Dashboard (Streamlit)
│   ├── dashboard.py               # Giao diện user - Real-time monitoring & simulation
│   ├── api_client.py              # HTTP client để gọi API backend
│   ├── data_loader.py             # Tải dữ liệu test từ file CSV/TXT
│   ├── visualization.py           # Vẽ biểu đồ real-time (Plotly)
│   ├── constants.py               # Hằng số của Dashboard (window sizes, paths)
│   └── __pycache__/
│
├── config/                        # ⚙️ Cấu hình toàn cục
│   ├── settings.py                # Cấu hình: paths, random seed, regex patterns, env vars
│   ├── train_config.yaml          # Config huấn luyện: features, hyperparameters, epochs
│   ├── autoscaling_config.yaml    # ⚙️ Config chiến lược scaling (thresholds, server limits)
│   └── __pycache__/
│
├── data/                          # 🔒 Quản lý dữ liệu
│   ├── raw/                       # Nhật ký gốc (ASCII format) - HTTP access logs từ NASA
│   │   ├── train.txt              # Dữ liệu huấn luyện (tháng 7/1995)
│   │   └── test.txt               # Dữ liệu kiểm thử (tháng 8/1995)
│   └── cleaned/                   # Dữ liệu đã xử lý & chuẩn hóa
│       ├── data_1m.csv            # Đếm request theo từng phút
│       ├── data_5m.csv            # Đếm request theo từng 5 phút
│       └── data_15m.csv           # Đếm request theo từng 15 phút
│
├── models/                        # 💾 Lưu trữ model weights đã huấn luyện
│   └── lstm_5m_best_model.pth     # PyTorch LSTM model cho dự báo 5-phút
│
├── models_export/                 # 📦 Export models cho production
│   ├── model_weights.pth          # Model weights
│   ├── model_metadata.json        # Metadata: input_size, hidden_size, etc.
│   └── README.md
│
├── notebooks/                     # 📓 Jupyter Notebooks (EDA & Experiments)
│   ├── eda.ipynb                  # Phân tích dữ liệu khám phá (trends, patterns)
│   ├── process_data.ipynb         # Tiền xử lý dữ liệu từ log gốc
│   ├── ml_model.ipynb             # Huấn luyện & kiểm thử LSTM
│   ├── arima_model.ipynb          # Thử nghiệm ARIMA (baseline)
│   ├── duccuong_lstm.ipynb        # LSTM với feature engineering nâng cao
│   └── ducer_system1.ipynb        # Hệ thống autoscaling version 1
│
├── output/                        # 📤 Kết quả đầu ra (Figures, Metrics)
│   ├── lstm_5m_results.json       # Metrics: RMSE, MAE, MAPE của model
│   └── model_metadata.json        # Thông tin model (version, timestamp, parameters)
│
├── evaluation_results/            # 📊 Kết quả Benchmark & Report
│   └── report.md                  # Báo cáo so sánh AI vs Reactive vs Static strategies
│
├── src/                           # 🧠 MÃ NGUỒN CHÍNH (Core Pipeline)
│   ├── __init__.py
│   ├── autoscaler.py              # 🚀 Logic Autoscaling (decide_scale function)
│   ├── data_loader.py             # Parse log files & extract features
│   ├── evaluation.py              # Tính toán metrics (RMSE, MAE, MAPE, MASE)
│   ├── features.py                # Feature engineering cho time-series
│   ├── utils.py                   # Utilities (logging, load/save config)
│   ├── __pycache__/
│   │
│   └── lstm/                      # 📦 LSTM Module (Deep Learning)
│       ├── data/
│       │   └── data_preparation.py    # Chuẩn bị data loaders cho training/validation
│       │
│       ├── models/
│       │   ├── lstm_model.py          # Class LSTMModel (architecture)
│       │   ├── model_utils.py         # Save/load model, convert to production
│       │   └── __pycache__/
│       │
│       ├── training/
│       │   ├── training.py            # train_epoch & validation functions
│       │   ├── evaluation.py          # Calculate metrics từ predictions
│       │   ├── tuning.py              # Hyperparameter tuning
│       │   └── __init__.py
│       │
│       └── inference/
│           ├── predictor.py           # Hàm predict() sử dụng trained model
│           └── __init__.py
│
├── main.py                        # 🚀 ENTRY POINT: Chạy toàn bộ pipeline từ A-Z
├── run_benchmark.py               # 🏆 Script Benchmark: so sánh 3 chiến lược scaling
├── requirements.txt               # Danh sách thư viện (torch, fastapi, streamlit, etc.)
├── LICENSE                        # Giấy phép dự án
└── README.md                      # Hướng dẫn sử dụng (tài liệu này)
```

---

## 🔍 Mô tả Chi tiết các Thư mục

### 📌 **api/** - Backend API Server
- **Chức năng**: Cung cấp REST API endpoints để nhận traffic hiện tại và dự báo tải
- **Các file chính**:
  - `api.py`: Khởi tạo FastAPI app, định nghĩa endpoints (predict, forecast, scale decision)
  - `schema.py`: Pydantic models cho request/response validation
  - `feature_engineering.py`: Transform raw data thành features cho LSTM
  - `model_loader.py`: Load trained LSTM weights từ disk
- **Trạng thái**: Stateful - giữ lịch sử 12 bước (60 phút) trong RAM

### 📌 **app/** - Frontend Dashboard
- **Chức năng**: Giao diện web real-time để giám sát & simulate autoscaling
- **Công nghệ**: Streamlit + Plotly
- **Các file chính**:
  - `dashboard.py`: Main UI - hiển thị biểu đồ traffic, predictions, server decisions
  - `api_client.py`: HTTP client gọi backend API
  - `data_loader.py`: Tải test data từ file CSV/TXT
  - `visualization.py`: Vẽ biểu đồ real-time interactiv
- **Chạy trên**: Port 8501 (mặc định Streamlit)

### 📌 **config/** - Cấu hình Toàn cục
- **Chức năng**: Lưu trữ tất cả parameters & configurations
- **Các file**:
  - `settings.py`: Paths, regex patterns, random seed (Python code)
  - `train_config.yaml`: Features, batch size, epochs, learning rate (YAML)
  - `autoscaling_config.yaml`: Server capacity, thresholds, safety factors (YAML)
- **Lợi ích**: Dễ thay đổi parameters mà không sửa code

### 📌 **data/** - Quản lý Dữ liệu
- **Chức năng**: Lưu trữ raw logs & processed datasets
- **Cấu trúc**:
  - `raw/`: NASA HTTP access logs (binary format) từ 1995
  - `cleaned/`: CSV files đã parse & aggregated
    - `data_1m.csv`: 1440 rows (1 ngày × 1440 phút)
    - `data_5m.csv`: Aggregated per 5-minute window
    - `data_15m.csv`: Aggregated per 15-minute window

### 📌 **models/** - Model Weights
- **Chức năng**: Lưu trữ trained PyTorch models
- **Nội dung**: 
  - `lstm_5m_best_model.pth`: Best LSTM checkpoint (cho 5-phút aggregation)

### 📌 **models_export/** - Production Export
- **Chức năng**: Models được export cho deployment
- **Nội dung**:
  - `model_weights.pth`: Model weights (PyTorch)
  - `model_metadata.json`: Schema (input_size, hidden_size, layers)

### 📌 **notebooks/** - Jupyter Notebooks
- **Chức năng**: EDA, experiments, model prototyping
- **Các notebook**:
  - `eda.ipynb`: Khám phá dữ liệu (trends, seasonality, anomalies)
  - `process_data.ipynb`: Parse logs → create cleaned datasets
  - `ml_model.ipynb`: LSTM training & evaluation
  - `arima_model.ipynb`: Baseline model (ARIMA comparison)
  - `duccuong_lstm.ipynb`: Advanced feature engineering
  - `ducer_system1.ipynb`: Autoscaling logic prototype

### 📌 **output/** - Kết quả Đầu ra
- **Chức năng**: Lưu kết quả từ training & inference
- **Nội dung**:
  - `lstm_5m_results.json`: Metrics (RMSE, MAE, MAPE)
  - `model_metadata.json`: Model info (version, training timestamp)

### 📌 **evaluation_results/** - Benchmark & Reports
- **Chức năng**: Kết quả so sánh giữa các chiến lược scaling
- **Nội dung**:
  - `report.md`: Báo cáo chi tiết (AI Autoscaler vs Reactive vs Static)
  - `benchmark_plot.png`: Biểu đồ so sánh

### 📌 **src/** - Core Source Code
- **Chức năng**: Logic chính của hệ thống
- **Các module**:
  - `autoscaler.py` 🚀: Decide scale-up/down based on traffic & forecast
  - `data_loader.py`: Parse NASA logs, extract features
  - `evaluation.py`: Tính metrics (RMSE, MAE, MAPE, MASE)
  - `features.py`: Feature engineering (normalization, time-based features)
  - `utils.py`: Utilities (config loader, logger, model I/O)

#### **src/lstm/** - Deep Learning Components
- **models/**: LSTM architecture & utilities
  - `lstm_model.py`: LSTMModel class (PyTorch nn.Module)
  - `model_utils.py`: Save/load/convert functions
- **training/**: Model training pipeline
  - `training.py`: train_epoch(), validation loops
  - `evaluation.py`: Calculate metrics
  - `tuning.py`: Hyperparameter search
- **inference/**: Production prediction
  - `predictor.py`: predict() function cho real-time forecasting
- **data/**: Data preparation
  - `data_preparation.py`: DataLoader creation, batch preparation

### 📌 **main.py** - Entry Point
- **Chức năng**: Run toàn bộ pipeline từ A-Z
- **Workflow**: Load config → Load data → Train model → Evaluate → Save results

### 📌 **run_benchmark.py** - Benchmark Script
- **Chức năng**: So sánh 3 chiến lược scaling
- **Output**: Báo cáo + biểu đồ tại `evaluation_results/`
- **Chiến lược**:
  1. **AI Autoscaler**: Dự báo LSTM + Logic thông minh
  2. **Reactive Scaling**: Scale khi traffic vượt threshold
  3. **Static Capacity**: Dung lượng cố định

---


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

**⚠️ Lưu ý Quan trọng trước khi chạy:**
1.  Đảm bảo file dữ liệu test tồn tại tại đường dẫn: `data/raw/test.txt`. Nếu chưa có, hãy chạy script chuẩn bị dữ liệu hoặc copy file log vào thư mục này.
2.  Bạn phải chạy **API Server trước** hoặc **song song** với Dashboard. Nếu API chưa bật, Dashboard sẽ báo lỗi kết nối.

Hệ thống Demo gồm 2 thành phần chính: **API Server** (Backend) và **Dashboard** (Frontend). Bạn cần mở 2 cửa sổ Terminal (hoặc CMD) riêng biệt để chạy chúng cùng một lúc.


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
