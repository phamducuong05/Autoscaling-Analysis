#!/bin/bash

# Kích hoạt virtual environment và chạy benchmark
echo "🚀 Đang chuẩn bị chạy Benchmark..."

# Kiểm tra nếu venv tồn tại
if [ -d "venv" ]; then
    PYTHON_CMD="./venv/bin/python"
else
    echo "⚠️  Không tìm thấy thư mục 'venv'. Đang thử dùng python3 hệ thống..."
    PYTHON_CMD="python3"
fi

# Cài đặt dependencies nếu cần (chỉ chạy 1 lần, bỏ comment nếu cần)
# $PYTHON_CMD -m pip install pandas matplotlib seaborn tqdm joblib scikit-learn

echo "📊 Đang chạy script run_benchmark.py..."
$PYTHON_CMD run_benchmark.py

echo "✅ Hoàn tất! Vui lòng kiểm tra báo cáo tại:"
echo "   - Báo cáo chi tiết: evaluation_results/report.md"
echo "   - Biểu đồ: evaluation_results/benchmark_plot.png"
