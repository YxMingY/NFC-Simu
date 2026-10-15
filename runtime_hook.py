"""
Runtime hook: 将 PyInstaller 临时目录及所有子目录加入 DLL 搜索路径
解决 pyarrow 等包的 DLL load failed 问题
"""
import os
import sys

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    dirs_added = set()

    def _add_dll_dirs(root):
        for dirpath, dirnames, filenames in os.walk(root):
            if any(f.endswith(('.dll', '.pyd')) for f in filenames):
                real = os.path.realpath(dirpath)
                if real not in dirs_added:
                    dirs_added.add(real)
                    try:
                        os.add_dll_directory(real)
                    except Exception:
                        pass

    _add_dll_dirs(base)
    os.environ['PATH'] = base + os.pathsep + os.environ.get('PATH', '')
