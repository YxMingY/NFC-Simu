import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import get_nusw_vector


def render(fc_global):
    st.header("NUSW 阵列响应分析")
    st.markdown("验证近场球面波的**相位非线性**和**幅度非均匀性**特性。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**NUSW (Non-Uniform Spherical Wave) 模型**

在远场 (夫琅禾费区)，电磁波可近似为平面波，阵列响应向量仅包含线性相位。但在近场 (菲涅尔区)，
电磁波以球面波形式传播，需同时考虑**幅度衰减**和**非线性相位**两个效应。

对于均匀线性阵列 (ULA)，第 $m$ 个天线单元到用户的精确距离为：

$$r_m = \sqrt{r^2 + (m \cdot d)^2 - 2r \cdot m \cdot d \cdot \cos\theta}$$

其中 $r$ 是用户到阵列参考点的距离，$d$ 是天线间距，$\theta$ 是入射角。

**阵列响应向量**的每个元素包含两部分：

$$a_m = \underbrace{\sqrt{\frac{r}{r_m}}}_{\text{幅度项}} \cdot \underbrace{\exp\!\left(-j\frac{2\pi}{\lambda}(r_m - r)\right)}_{\text{相位项}}$$

- **幅度项** $\sqrt{r/r_m}$：反映球面波扩散导致各天线接收功率不均匀。远场时 $r_m \approx r$，幅度趋于均匀。
- **相位项**：反映球面波相位的非线性变化。远场时相位关于天线索引 $m$ 近似线性（即平面波假设）。

**关键观察**：
- 距离越近 → 相位非线性越明显，幅度差异越大
- 距离越远 → 趋近远场平面波，相位线性、幅度均匀
""")

    c = 3e8
    lam = c / (fc_global * 1e9)
    d = lam / 2

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        M = st.slider("天线数量 M", 8, 512, 128, step=8)
        theta_deg = st.slider("角度 θ (度)", 0, 180, 90)
        theta = np.radians(theta_deg)
        test_distances = st.multiselect(
            "测试距离 (m)",
            [1.25, 2.5, 5, 10, 25, 50, 100, 200, 500, 1000],
            default=[1.25, 5, 10, 50, 200, 1000],
        )
        st.markdown("---")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")
        st.metric("天线间距 d", f"{d * 1000:.2f} mm")
        st.metric("阵列孔径 D", f"{(M - 1) * d:.2f} m")

    with main_col:
        if not test_distances:
            st.warning("请至少选择一个测试距离")
            return

        fig1, ax1 = plt.subplots(figsize=(7, 3.5))
        for r_test in test_distances:
            _, _, phase = get_nusw_vector(r_test, theta, M, d, lam)
            ax1.plot(np.arange(M), phase, linewidth=2, label=f"r = {r_test} m")
        ax1.set_xlabel("天线单元索引")
        ax1.set_ylabel("相位 (rad)")
        ax1.set_title("阵列响应相位线性度对比")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        for r_test in test_distances:
            _, amp, _ = get_nusw_vector(r_test, theta, M, d, lam)
            power = np.abs(amp) ** 2
            ax2.plot(np.arange(M), power, linewidth=2, label=f"r = {r_test} m")
        ax2.set_xlabel("天线单元索引")
        ax2.set_ylabel("归一化接收功率")
        ax2.set_title("阵列响应幅度均匀性对比")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.7)
        st.pyplot(fig2)
        plt.close(fig2)
