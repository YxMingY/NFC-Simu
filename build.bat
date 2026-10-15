@echo off
chcp 65001 >nul
echo ========================================
echo  6G 近场 XL-MIMO 仿真系统 - 打包工具
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查/安装 PyInstaller
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装 PyInstaller...
    pip install pyinstaller
    echo.
)

echo [开始] 正在打包应用...
echo [提示] 这可能需要几分钟，请耐心等待
echo.

python build.py

echo.
pause
