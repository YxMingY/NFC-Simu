# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件
用于打包 6G 近场 XL-MIMO 仿真系统
"""
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# 收集 pyarrow DLL 和数据文件 (streamlit 内部依赖)
extra_binaries = []
extra_datas = []
try:
    extra_binaries += collect_dynamic_libs('pyarrow')
    extra_datas += collect_data_files('pyarrow')
except Exception:
    pass

# 收集项目文件
project_datas = []
project_files = [
    ("app.py", "."),
    ("core.py", "."),
    ("modules/__init__.py", "modules"),
    ("modules/nusw.py", "modules"),
    ("modules/region.py", "modules"),
    ("modules/snr.py", "modules"),
    ("modules/edof.py", "modules"),
    ("modules/beam.py", "modules"),
    ("modules/resolution.py", "modules"),
]

for src, dest in project_files:
    if os.path.exists(src):
        project_datas.append((src, dest))

# 添加 Streamlit 静态文件和元数据
try:
    import streamlit
    import importlib.metadata

    streamlit_path = Path(streamlit.__file__).parent
    static_path = streamlit_path / "static"
    if static_path.exists():
        project_datas.append((str(static_path), "streamlit/static"))

    dist = importlib.metadata.distribution('streamlit')
    if dist._path:
        project_datas.append((str(dist._path), "streamlit-" + str(dist.version) + ".dist-info"))
except Exception as e:
    print(f"Warning: Could not collect streamlit metadata: {e}")

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=extra_binaries,
    datas=project_datas + extra_datas,
    hiddenimports=[
        'streamlit',
        'streamlit.web',
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.scriptrunner.magic_funcs',
        'streamlit.runtime.caching',
        'streamlit.runtime.state',
        'streamlit.runtime.state.safe_session_state',
        'streamlit.runtime.state.session_state',
        'streamlit.components',
        'streamlit.components.v1',
        'modules',
        'modules.nusw',
        'modules.region',
        'modules.snr',
        'modules.edof',
        'modules.beam',
        'modules.resolution',
        'core',
        'numpy',
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_agg',
        'pyarrow',
        'pyarrow.lib',
        'pyarrow._compute',
        'pyarrow._csv',
        'pyarrow._dataset',
        'pyarrow._feather',
        'pyarrow._json',
        'pyarrow._parquet',
        'pyarrow.fs',
        'pyarrow.ipc',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'torch',
        'tensorflow',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
        'PyQt5',
        'PyQt6',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='6G_NearField_XL-MIMO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
