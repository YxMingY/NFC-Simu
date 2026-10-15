@echo off
chcp 65001 >nul
echo ========================================
echo  6G 近场 XL-MIMO 仿真系统
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查依赖是否安装
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo [提示] 首次运行，正在安装依赖...
    pip install -r requirements.txt
    echo.
)

echo [启动] 正在启动仿真系统...
echo [提示] 浏览器将自动打开，如未打开请访问 http://localhost:8501
echo.

streamlit run app.py --server.port 8501

pause
