"""
Streamlit 应用启动器
用于 PyInstaller 打包后的入口点
"""
import sys
import os
import webbrowser
import threading
import time
import streamlit.web.cli as stcli


def open_browser():
    """延迟打开浏览器，等待 Streamlit 启动"""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")


def main():
    """启动 Streamlit 应用"""
    # 获取应用脚本路径
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后的路径
        base_path = sys._MEIPASS
    else:
        # 开发环境路径
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    app_path = os.path.join(base_path, "app.py")
    
    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动 Streamlit
    sys.argv = ["streamlit", "run", app_path, 
                "--server.port=8501", 
                "--server.headless=false",
                "--browser.gatherUsageStats=false",
                "--global.developmentMode=false"]
    
    stcli.main()


if __name__ == "__main__":
    main()
