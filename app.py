"""
6G 近场 XL-MIMO 仿真系统
基于 Streamlit 的交互式仿真平台
"""
import streamlit as st
import matplotlib.pyplot as plt
from modules import MODULE_MAP

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="6G 近场 XL-MIMO 仿真系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📡 6G 近场 XL-MIMO 仿真系统")
st.sidebar.markdown("---")

module = st.sidebar.radio("📋 功能模块", list(MODULE_MAP.keys()), index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 全局参数")
fc_global = st.sidebar.number_input(
    "载波频率 (GHz)", min_value=0.1, max_value=100.0, value=2.4, step=0.1
)

MODULE_MAP[module](fc_global)
