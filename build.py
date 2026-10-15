"""
PyInstaller 打包脚本
将 Streamlit 应用打包为可分发的 Windows 应用程序
"""
import os
import sys
import shutil
import subprocess


def clean_build():
    """清理旧的构建文件"""
    dirs_to_remove = ["build", "dist", "__pycache__"]
    
    for d in dirs_to_remove:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"已删除: {d}")


def build():
    """执行打包"""
    print("=" * 50)
    print("6G 近场 XL-MIMO 仿真系统 - 打包工具")
    print("=" * 50)
    
    # 检查 PyInstaller
    try:
        import PyInstaller
        print(f"PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("错误: 未安装 PyInstaller")
        print("请运行: pip install pyinstaller")
        sys.exit(1)
    
    # 清理旧文件
    print("\n[1/3] 清理旧构建文件...")
    clean_build()
    
    # 使用 spec 文件打包
    print("\n[2/3] 开始打包 (这可能需要几分钟)...")
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "app.spec"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n打包失败!")
        sys.exit(1)
    
    print("\n[3/3] 打包完成!")
    print(f"\n输出目录: dist/6G_NearField_XL-MIMO/")
    print(f"可执行文件: dist/6G_NearField_XL-MIMO/6G_NearField_XL-MIMO.exe")
    print("\n你可以将整个 dist/6G_NearField_XL-MIMO/ 文件夹分发给其他用户。")
    print("用户双击 exe 即可运行，无需安装 Python。")


if __name__ == "__main__":
    build()
