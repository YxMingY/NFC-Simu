# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec 文件
用于打包 6G 近场 XL-MIMO 仿真系统
"""
import sys
# 增加递归深度限制
sys.setrecursionlimit(sys.getrecursionlimit() * 5)

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# 收集 scipy 和 pyarrow 的 DLL (解决 DLL load failed)
extra_binaries = []

# pyarrow DLLs (arrow.dll 等在包目录内)
try:
    extra_binaries += collect_dynamic_libs('pyarrow')
except Exception:
    pass

# scipy 依赖的 BLAS/LAPACK DLL (conda 环境下在 Library/bin 中)
try:
    import sysconfig
    conda_prefix = Path(sysconfig.get_path('data'))
    conda_bin = conda_prefix / "Library" / "bin"
    if conda_bin.exists():
        for dll in conda_bin.glob("*.dll"):
            name = dll.name.lower()
            if any(k in name for k in ['blas', 'lapack', 'mkl', 'openblas', 'libblas', 'liblapack', 'libcblas', 'libgcc', 'libwinpthread', 'libgfortran']):
                extra_binaries.append((str(dll), '.'))
                print(f"  Adding BLAS DLL: {dll.name}")
except Exception as e:
    print(f"Warning: Could not collect BLAS DLLs: {e}")

# 收集 pyarrow 数据文件
extra_datas = []
try:
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
    
    # 添加 streamlit 的 dist-info (解决 PackageNotFoundError)
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
        'scipy',
        'scipy.linalg',
        'scipy.linalg.blas',
        'scipy.linalg.lapack',
        'scipy.linalg._fblas',
        'scipy.linalg._flapack',
        'scipy._lib',
        'scipy._lib.messagestream',
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
        'pyarrow.table',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[
        'torch',
        'tensorflow',
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
    console=True,  # 显示控制台以便调试
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
