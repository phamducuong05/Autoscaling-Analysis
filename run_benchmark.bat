@echo off
chcp 65001 > nul
echo 🚀 Dang chuan bi chay Benchmark...

:: Kiem tra xem venv co ton tai khong
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
) else (
    echo ⚠️  Khong tim thay thu muc 'venv'. Dang thu dung python he thong...
    set "PYTHON_CMD=python"
)

echo 📊 Dang chay script run_benchmark.py...
"%PYTHON_CMD%" run_benchmark.py

echo ✅ Hoan tat! Vui long kiem tra bao cao tai:
echo    - Bao cao chi tiet: evaluation_results/report.md
echo    - Bieu do: evaluation_results/benchmark_plot.png
pause
