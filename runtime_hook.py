"""
Runtime hook: 将 PyInstaller 临时目录加入 DLL 搜索路径
解决 scipy/pyarrow 的 DLL load failed 问题
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    try:
        os.add_dll_directory(base)
    except Exception:
        pass
    os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
