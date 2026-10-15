import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import compute_distance_resolution


def render(fc_global):
    st.header("距离分辨率分析")
    st.markdown("分析近场波束聚焦的距离分辨率极限：聚焦在 r 处时，多近的两个目标无法分辨。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**距离分辨率的物理含义**

近场波束聚焦类似于一个"距离滤波器"——将波束聚焦在距离 $r$ 处时，
只有在 $[r_1, r_2]$ 范围内的目标都能获得较高增益，超出此范围增益急剧下降。
区间宽度 $\delta r = r_2 - r_1$ 即为**距离分辨率**。

**半功率有效距离**

$$r_{\text{hp}} = 0.1 \sin^2\theta' \cdot \frac{2D^2}{\lambda}$$

这是距离分辨率的特征标尺，决定了分辨率的"粒度"。

**可分辨边界**

对于聚焦在 $r$ 处的波束，能分辨的最近和最远点分别为：

$$r_1 = \frac{1}{\frac{1}{r} + \omega}, \quad r_2 = \frac{1}{\frac{1}{r} - \omega}$$

其中 $\omega = 1/r_{\text{hp}}$。不可分辨区间宽度为：

$$\delta r = r_2 - r_1 = \frac{2\omega r^2}{1 - \omega^2 r^2}$$

**近似公式** (当 $r \ll r_{\text{hp}}$ 时)：

$$\delta r \approx \frac{2r^2}{r_{\text{hp}}}$$

**关键规律**：
- 分辨率 $\delta r \propto r^2$：聚焦越远，分辨率越差（不可分辨区间越宽）
- 分辨率 $\delta r \propto 1/D^2$：阵列孔径越大，分辨率越好
- 这是近场通信的基本物理约束：无法无限精细地区分不同距离上的用户
""")

    c = 3e8
    lam = c / (fc_global * 1e9)
    d = lam / 2

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        M = st.slider("天线数量 M", 32, 1024, 512, step=32, key="res_M")
        theta_prime_deg = st.slider("角度 θ (度)", 10, 170, 90, key="res_theta")
        theta_prime = np.radians(theta_prime_deg)
        st.markdown("---")
        D = M * d
        r_hp = 0.1 * np.sin(theta_prime) ** 2 * (2 * D**2 / lam)
        st.metric("阵列孔径 D", f"{D:.2f} m")
        st.metric("半功率有效距离 r_hp", f"{r_hp:.1f} m")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    with main_col:
        r_max = r_hp / 5.0
        r_array = np.linspace(max(1, r_max * 0.01), r_max, 2000)

        r1, r2, delta_r, delta_r_approx, _ = compute_distance_resolution(
            r_array, M, d, lam, theta_prime
        )

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))

        ax1 = axes[0, 0]
        ax1.plot(r_array, delta_r, "b-", linewidth=2.5, label="精确值")
        ax1.plot(r_array, delta_r_approx, "g--", linewidth=2, label="近似 $2r^2/r_{hp}$")
        ax1.set_xlabel("聚焦距离 r (m)")
        ax1.set_ylabel("距离分辨率 δr (m)")
        ax1.set_title("距离分辨率 vs 聚焦距离 (线性)")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        ax2 = axes[0, 1]
        ax2.loglog(r_array, delta_r, "b-", linewidth=2.5, label="精确值")
        ax2.loglog(r_array, delta_r_approx, "g--", linewidth=2, label="近似 (斜率=2)")
        ax2.set_xlabel("聚焦距离 r (m)")
        ax2.set_ylabel("距离分辨率 δr (m)")
        ax2.set_title("距离分辨率 vs 聚焦距离 (双对数)")
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5, which="both")

        ax3 = axes[1, 0]
        ax3.plot(r_array, r1, "g-", linewidth=2, label="$r_1$ (最近可分辨)")
        ax3.plot(r_array, r_array, "k--", linewidth=1.5, label="$r$ (聚焦距离)")
        ax3.plot(r_array, r2, "r-", linewidth=2, label="$r_2$ (最远可分辨)")
        ax3.fill_between(r_array, r1, r2, alpha=0.15, color="orange", label="不可分辨区域")
        ax3.set_xlabel("聚焦距离 r (m)")
        ax3.set_ylabel("距离 (m)")
        ax3.set_title("不可分辨区域 [$r_1$, $r_2$]")
        ax3.legend(fontsize=8)
        ax3.grid(True, linestyle=":", alpha=0.5)

        ax4 = axes[1, 1]
        n_examples = min(10, int(len(r_array) / 200))
        example_indices = np.linspace(int(len(r_array) * 0.05), int(len(r_array) * 0.95), n_examples).astype(int)
        cmap = plt.cm.turbo(np.linspace(0, 1, len(example_indices)))
        for idx, ei in enumerate(example_indices):
            r_ex = r_array[ei]
            r1_ex, r2_ex = r1[ei], r2[ei]
            dr_ex = r2_ex - r1_ex
            ax4.plot([r1_ex, r2_ex], [r_ex, r_ex], "-o", color=cmap[idx], linewidth=3,
                     markersize=7, markerfacecolor="none",
                     label=f"r={r_ex:.0f}m: δr={dr_ex:.1f}m")
        ax4.plot(r_array, r_array, "k--", linewidth=1, alpha=0.3)
        ax4.set_xlabel("距离 (m)")
        ax4.set_ylabel("聚焦距离 r (m)")
        ax4.set_title("各聚焦距离处的不可分辨区间")
        ax4.legend(fontsize=7, loc="upper left")
        ax4.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("典型距离处的分辨率数值")
        n_table = min(10, int(len(r_array) / 200))
        table_indices = np.linspace(int(len(r_array) * 0.1), int(len(r_array) * 0.9), n_table).astype(int)
        table_data = {
            "聚焦距离 r (m)": [f"{r_array[i]:.1f}" for i in table_indices],
            "r₁ (m)": [f"{r1[i]:.2f}" for i in table_indices],
            "r₂ (m)": [f"{r2[i]:.2f}" for i in table_indices],
            "δr (m)": [f"{(r2[i] - r1[i]):.2f}" for i in table_indices],
        }
        st.table(table_data)
