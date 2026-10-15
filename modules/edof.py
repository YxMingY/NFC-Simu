import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import compute_edof_sweep


def render(fc_global):
    st.header("EDoF 有效自由度分析")
    st.markdown("分析近场 LoS MIMO 信道的有效自由度、能量集中度和孔径利用率。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**1. 有效自由度 (EDoF, Effective Degrees of Freedom)**

对近场 LoS 信道矩阵 $\mathbf{H}$ 进行奇异值分解 (SVD)：

$$\mathbf{H} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^H$$

EDoF 定义为归一化奇异值中超过阈值 $\epsilon$ 的个数：

$$\text{EDoF} = \left|\left\{k : \frac{\sigma_k}{\sigma_1} > \epsilon\right\}\right|$$

物理含义：近场信道矩阵的"有效秩"，即可独立传输的数据流数量。
与远场 LoS 信道 (秩为 1) 不同，近场球面波相位差异使信道矩阵具有更高的秩。

**理论近似** (连续孔径公式)：

$$\text{EDoF}_{\text{theory}} \approx \frac{D_r \cdot D_t}{\lambda \cdot r}$$

其中 $D_r, D_t$ 分别为接收端和发射端的有效孔径。实际值受限于 $\min(M_r, M_t)$。

**2. 能量集中度 (Energy Concentration)**

$$\eta = \frac{\sum_{k=1}^{\text{EDoF}} \sigma_k^2}{\sum_{k} \sigma_k^2}$$

衡量前 EDoF 个主导奇异值携带的能量占总能量的比例。
- $\eta > 90\%$：信道能量高度集中，低秩近似效果极佳
- $\eta < 80\%$：信道能量分散，存在显著的旁瓣泄漏

**3. 有效孔径利用率 (Effective Aperture Ratio)**

$$\rho = \frac{D_{t,\text{eff}}}{D_{t,\text{phys}}} = \frac{\text{EDoF} \cdot \lambda \cdot r / D_r}{(M_t - 1) \cdot d}$$

衡量实际 EDoF 相对于理论最大值的比例，反映发射天线孔径被有效利用的程度。
- $\rho \approx 1$：天线利用率高
- $\rho \ll 1$：部分天线因近场距离效应（边缘天线距离过远）而未能有效贡献

**工程启示**：当 $M_t \gg M_r$ 时，系统进入"接收受限"区域，EDoF 饱和于 $M_r$，
此时应优先优化接收端算法或增加接收天线，而非继续增加发射天线。
""")

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        Mr = st.slider("接收天线数 Mr", 4, 64, 16, key="edof_mr")
        r_user = st.number_input("用户距离 r (m)", 0.1, 50.0, 1.0, step=0.1, key="edof_r")
        st.markdown("---")
        st.markdown("##### Mt 扫描范围")
        Mt_start = st.number_input("Mt 起始值", 10, 500, 10, key="edof_mt_start")
        Mt_end = st.number_input("Mt 结束值", 50, 2000, 500, key="edof_mt_end")
        Mt_step = st.number_input("Mt 步长", 5, 100, 20, key="edof_mt_step")
        st.markdown("---")
        threshold = st.slider("EDoF 阈值", 0.01, 0.9, 0.3, step=0.01, key="edof_thr")

    with main_col:
        Mt_list = list(range(int(Mt_start), int(Mt_end) + 1, int(Mt_step)))

        with st.spinner(f"扫描 Mt: {Mt_list[0]} → {Mt_list[-1]} (SVD 计算中)..."):
            edof_sim, edof_theory, energy_conc, ap_ratio = compute_edof_sweep(
                Mt_list, Mr, r_user, fc_global * 1e9, threshold
            )

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        fig.suptitle(f"近场 XL-MIMO 信道分析 (Mr={Mr}, r={r_user}m)")

        ax1 = axes[0, 0]
        ax1.plot(Mt_list, edof_sim, "b-o", markersize=3, label=f"仿真 EDoF (thr={threshold})")
        ax1.plot(Mt_list, edof_theory, "r-", linewidth=2, label="理论上限 (bounded)")
        ax1.axhline(y=Mr, color="gray", linestyle=":", label=f"Mr 上限 ({Mr})")
        ax1.set_xlabel("发射天线数 Mt")
        ax1.set_ylabel("EDoF")
        ax1.set_title("有效自由度")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(Mt_list, energy_conc, "m-s", markersize=3, linewidth=2)
        ax2.fill_between(Mt_list, energy_conc, alpha=0.2, color="magenta")
        ax2.axhline(y=0.9, color="gray", linestyle="--", label="90% 参考线")
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel("发射天线数 Mt")
        ax2.set_ylabel("能量集中度")
        ax2.set_title("前 EDoF 个模式携带的能量占比")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(Mt_list, ap_ratio, "g-^", markersize=3)
        ax3.axhline(y=1.0, color="k", linestyle=":", label="理想 100%")
        ax3.set_xlabel("发射天线数 Mt")
        ax3.set_ylabel("有效孔径利用率")
        ax3.set_title("物理孔径 vs 有效孔径")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        errors = [abs(s - t) / max(t, 1) * 100 for s, t in zip(edof_sim, edof_theory)]
        ax4.plot(Mt_list, errors, "k-x", linewidth=1.5)
        ax4.set_xlabel("发射天线数 Mt")
        ax4.set_ylabel("相对误差 (%)")
        ax4.set_title("仿真 vs 理论误差")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        peak_idx = int(np.argmax(edof_sim))
        st.markdown(f"**EDoF 峰值**: Mt={Mt_list[peak_idx]}, EDoF={edof_sim[peak_idx]}  |  "
                    f"**最终能量集中度**: {energy_conc[-1] * 100:.1f}%  |  "
                    f"**最终孔径利用率**: {ap_ratio[-1] * 100:.1f}%")
