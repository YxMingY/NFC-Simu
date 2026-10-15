import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import compute_dd_rayleigh, compute_upd


def render(fc_global):
    st.header("近场通信区域划分边界图")
    st.markdown("展示经典瑞利距离、方向相关瑞利距离和均匀功率距离的边界。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
近场通信中，不同物理效应在不同距离尺度上变得显著。以下三条边界线定义了三个关键距离尺度：

**1. 经典瑞利距离 (Rayleigh Distance)**

$$r_{\text{Rayl}} = \frac{2D^2}{\lambda}$$

其中 $D = (M-1)d$ 为阵列物理孔径，$\lambda$ 为波长。这是传统的近场/远场分界距离（标量，与方向无关）。
当 $r < r_{\text{Rayl}}$ 时，阵列处于近场区域。

**2. 方向相关瑞利距离 (Direction-Dependent Rayleigh)**

$$r_{\text{DDRayl}}(\theta) = \frac{2D^2 \sin^2\theta}{\lambda}$$

物理意义：近场范围随入射角 $\theta$ 变化。有效孔径 $D_{\text{eff}} = D\sin\theta$：
- 当 $\theta \to 0$（端射方向）时，有效孔径趋近于零，近场区收缩
- 当 $\theta = \pi/2$（正入射）时，近场区最远

**3. 均匀功率距离 (Uniform Power Distance, UPD)**

$$r_{\text{UPD}}(\theta) = \min\left\{r : \frac{\min_m P_m}{\max_m P_m} \geq \Upsilon_{\text{th}}\right\}$$

物理定义：在该距离之外，阵列各单元接收功率的最大波动不超过阈值 $\Upsilon_{\text{th}}$（默认 0.9）。
此时可以安全地忽略近场幅度差异，简化为仅考虑相位变化。

**区域划分**：
- **NUSW 区域**（红色填充）：$r < \min(r_{\text{UPD}}, r_{\text{DDRayl}})$，需同时考虑球面波相位和幅度不均匀性
- **相位非均匀区域**：$r < r_{\text{DDRayl}}$ 但 $r > r_{\text{UPD}}$，幅度可近似均匀但相位仍需精确建模
- **远场区域**：$r > r_{\text{Rayl}}$，平面波假设完全成立
""")

    c = 3e8
    lam = c / (fc_global * 1e9)
    d = lam / 2

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        M = st.slider("天线数量 M", 16, 512, 128, step=16, key="region_M")
        Upsilon_th = st.slider("功率均匀度阈值 Υ_th", 0.5, 0.99, 0.9, step=0.01)
        st.markdown("---")
        D = (M - 1) * d
        r_Rayl = 2 * D**2 / lam
        st.metric("阵列孔径 D", f"{D:.2f} m")
        st.metric("经典瑞利距离", f"{r_Rayl:.1f} m")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    with main_col:
        thetas = np.linspace(0.01, np.pi / 2, 300)

        with st.spinner("计算区域边界..."):
            r_DDRayl = compute_dd_rayleigh(M, d, lam, thetas)
            r_UPD_vals = compute_upd(M, d, lam, thetas, Upsilon_th)

        def polar_to_cart(r, th):
            th2 = np.pi / 2 - th
            return r * np.cos(th2), r * np.sin(th2)

        fig, ax = plt.subplots(figsize=(6, 6))

        x_rayl, y_rayl = polar_to_cart(r_Rayl, thetas)
        ax.plot(x_rayl, y_rayl, "k--", linewidth=2, label=f"经典瑞利距离 = {r_Rayl:.1f} m")

        x_dd, y_dd = polar_to_cart(r_DDRayl, thetas)
        ax.plot(x_dd, y_dd, "b-", linewidth=2, label="方向相关瑞利距离")

        x_upd, y_upd = polar_to_cart(r_UPD_vals, thetas)
        ax.plot(x_upd, y_upd, "r-", linewidth=2, label=f"均匀功率距离 (Υ_th={Upsilon_th})")

        r_inner = np.minimum(r_UPD_vals, r_DDRayl)
        x_fill = np.concatenate([[0], polar_to_cart(r_inner, thetas)[0], [0]])
        y_fill = np.concatenate([[0], polar_to_cart(r_inner, thetas)[1], [0]])
        ax.fill(x_fill, y_fill, color="red", alpha=0.1, label="NUSW 区域")

        ax.set_xlabel("x 距离 (m)")
        ax.set_ylabel("y 距离 (m)")
        ax.set_title("近场通信区域划分边界图 (第一象限)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(0, max(x_rayl) * 1.1)
        ax.set_ylim(0, max(y_rayl) * 1.1)
        st.pyplot(fig)
        plt.close(fig)
