import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import (compute_beam_gain_grid, compute_beam_gain_curve,
                  compute_array_beams, compute_energy_spread)


def render(fc_global):
    st.header("波束聚焦图案分析")
    st.markdown("对比近场 NUSW 波束（聚光灯效应）与远场 DFT 波束（能量扩散）的增益分布。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**远场 vs 近场波束成形**

传统远场波束成形 (DFT 波束) 仅补偿**角度**维度的相位差：

$$w_m^{\text{DFT}} = \exp\!\left(j\frac{2\pi}{\lambda} m d \cos\theta'\right)$$

这只在特定角度方向形成波束峰值，但在**距离**维度上无聚焦能力——能量沿传播方向持续扩散（类似手电筒）。

近场 NUSW 波束成形同时补偿**角度和距离**两个维度的相位和幅度：

$$w_m^{\text{NUSW}} = \sqrt{\frac{r'}{r_m'}} \cdot \exp\!\left(-j\frac{2\pi}{\lambda}(r_m' - r')\right)$$

其中 $r_m' = \sqrt{r'^2 + (md)^2 - 2r'(md)\cos\theta'}$。这能在特定的 $(r', \theta')$ 位置形成精准的聚焦斑点（类似聚光灯）。

**归一化增益定义**

$$G(\mathbf{s}; \mathbf{s}') = \frac{|\mathbf{w}^H(\mathbf{s}') \cdot \mathbf{a}(\mathbf{s})|}{\|\mathbf{w}(\mathbf{s}')\| \cdot \|\mathbf{a}(\mathbf{s})\|}$$

其中 $\mathbf{s} = (r, \theta)$ 为观测点，$\mathbf{s}' = (r', \theta')$ 为波束焦点。
$G = 1$ (即 0 dB) 表示完美匹配，$G \ll 1$ 表示失配。

**物理对比**：
| 特性 | 远场 DFT 波束 | 近场 NUSW 波束 |
|------|-------------|---------------|
| 聚焦维度 | 仅角度 | 角度 + 距离 |
| 波束形状 | 扇形（能量扩散） | 斑点（能量集中） |
| 类比 | 手电筒 | 聚光灯 |
| 适用场景 | $r \gg r_{\text{Rayl}}$ | $r \lesssim r_{\text{Rayl}}$ |
""")

    c = 3e8
    lam = c / (fc_global * 1e9)
    d = lam / 2

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        M = st.slider("天线数量 M", 32, 1024, 512, step=32, key="beam_M")
        r_target = st.number_input("目标距离 r (m)", 5.0, 500.0, 100.0, step=5.0, key="beam_r")
        theta_target_deg = st.slider("目标角度 θ (度)", 10, 170, 90, key="beam_theta")
        theta_target = np.radians(theta_target_deg)
        st.markdown("---")
        st.markdown("##### 计算精度")
        grid_size = st.slider("网格分辨率", 50, 300, 150, step=10, key="beam_grid")
        st.markdown("---")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    with main_col:
        r_grid = np.linspace(max(5, r_target * 0.3), r_target * 2.0, grid_size)
        theta_grid = np.linspace(np.pi / 6, 5 * np.pi / 6, grid_size)

        with st.spinner("计算波束聚焦增益网格 (可能需要数秒)..."):
            gain_nf, gain_ff = compute_beam_gain_grid(
                r_grid, theta_grid, r_target, theta_target, M, d, lam
            )

        fig1, axes = plt.subplots(1, 2, figsize=(9, 4))

        im1 = axes[0].pcolormesh(
            r_grid, np.degrees(theta_grid), 20 * np.log10(gain_nf + 1e-10),
            shading="auto", cmap="hot",
        )
        axes[0].plot(r_target, theta_target_deg, "o", markersize=10,
                     markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
        axes[0].set_xlabel("距离 r (m)")
        axes[0].set_ylabel("角度 θ (度)")
        axes[0].set_title("近场 NUSW 波束 (聚光灯)")
        plt.colorbar(im1, ax=axes[0], label="归一化增益 (dB)")
        axes[0].grid(True, linestyle=":", alpha=0.5)

        im2 = axes[1].pcolormesh(
            r_grid, np.degrees(theta_grid), 20 * np.log10(gain_ff + 1e-10),
            shading="auto", cmap="hot",
        )
        axes[1].plot(r_target, theta_target_deg, "o", markersize=10,
                     markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
        axes[1].set_xlabel("距离 r (m)")
        axes[1].set_ylabel("角度 θ (度)")
        axes[1].set_title("远场 DFT 波束 (能量扩散)")
        plt.colorbar(im2, ax=axes[1], label="归一化增益 (dB)")
        axes[1].grid(True, linestyle=":", alpha=0.5)

        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("目标方向增益随距离变化")
        r_curve = np.linspace(max(5, r_target * 0.3), r_target * 2.0, 500)
        gain_nf_curve, gain_ff_curve = compute_beam_gain_curve(
            r_curve, theta_target, r_target, M, d, lam
        )

        gain_nf_dB = 20 * np.log10(gain_nf_curve + 1e-10)
        gain_ff_dB = 20 * np.log10(gain_ff_curve + 1e-10)

        fig2, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(r_curve, gain_nf_dB, "r-", linewidth=2.5, label="近场 NUSW 波束")
        ax.plot(r_curve, gain_ff_dB, "b--", linewidth=2.5, label="远场 DFT 波束")
        target_idx = np.argmin(np.abs(r_curve - r_target))
        ax.axvline(x=r_target, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.plot(r_target, gain_nf_dB[target_idx], "o", markersize=12,
                markerfacecolor="none", markeredgecolor="red", markeredgewidth=2.5)

        D_ap = M * d
        r_hp = 0.1 * np.sin(theta_target)**2 * (2 * D_ap**2 / lam)
        omega_res = 1.0 / r_hp
        r1_res = 1.0 / (1.0 / r_target + omega_res)
        r2_res = 1.0 / (1.0 / r_target - omega_res)
        delta_r_res = r2_res - r1_res

        ax.axvline(x=r1_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(x=r2_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvspan(r1_res, r2_res, alpha=0.12, color="green")

        ax.set_xlabel("距离 r (m)")
        ax.set_ylabel("归一化增益 (dB)")
        ax.set_title(f"波束增益随距离变化 (θ = {theta_target_deg}°, $\\delta r$ = {delta_r_res:.2f} m)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim([-40, 5])

        peak_gain = gain_nf_dB[target_idx]
        arrow_y = min(peak_gain + 1.5, 4.0)
        text_y = min(peak_gain + 2.5, 4.6)
        ax.annotate("",
                    xy=(r1_res, arrow_y),
                    xytext=(r2_res, arrow_y),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=2))
        ax.text((r1_res + r2_res) / 2, text_y,
                f"$\\delta r$ = {delta_r_res:.2f} m",
                ha="center", va="bottom", fontsize=11, color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.85))
        ax.text(r1_res, -36, f"$r_1$={r1_res:.1f}m",
                ha="center", fontsize=9, color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.85))
        ax.text(r2_res, -36, f"$r_2$={r2_res:.1f}m",
                ha="center", fontsize=9, color="green", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.85))

        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")
    st.subheader("三种阵列架构波束方向图对比")
    st.markdown("对比**共址 ULA (Collocated)**、**模块化 ULA (Modular)** 和**稀疏 ULA (Sparse)** 的远场波束模式。")

    with st.expander("📖 理论基础", expanded=False):
        st.markdown(r"""
**三种阵列架构**

| 架构 | 描述 | 波束特征 |
|------|------|----------|
| Collocated ULA | 所有天线等间距 $d=\lambda/2$ 排列 | 标准 sinc 型波束，无栅瓣 |
| Modular ULA | $N$ 个模块，每模块 $M$ 个天线，模块间距 $\Gamma d$ | 主瓣 + 周期性栅瓣 |
| Sparse ULA | 总天线数相同但间距 $I d \gg \lambda/2$ | 主瓣变窄但栅瓣严重 |

模块化阵列的方向图可分解为**模块级因子** × **子阵因子**：

$$G_{\text{mod}}(\theta) = \frac{1}{N}\left|\sum_{n=0}^{N-1} e^{j\frac{2\pi}{\lambda} n \Gamma d (\cos\theta - \cos\theta')}\right| \cdot \frac{1}{M}\left|\sum_{m=0}^{M-1} e^{j\frac{2\pi}{\lambda} m d (\cos\theta - \cos\theta')}\right|$$
""")

    arch_col, arch_ctrl = st.columns([3, 1])
    with arch_ctrl:
        st.markdown("##### 阵列参数")
        N_mod = st.slider("模块数 N", 2, 16, 4, key="arch_N")
        M_elem = st.slider("每模块天线数 M", 2, 16, 4, key="arch_M")
        Gamma = st.slider("模块间隔 Γ", 2, 32, N_mod + M_elem, key="arch_Gamma")
        I_sparse = st.slider("稀疏间隔 I", 2, 32, Gamma, key="arch_I")
        M_total = N_mod * M_elem
        st.metric("总天线数", f"{M_total}")

    with arch_col:
        theta_arr = np.linspace(0, np.pi, 2000)
        target_arr = np.pi / 2
        G_co, G_mo, G_sp = compute_array_beams(
            theta_arr, target_arr, M_total, d, lam,
            N_mod, M_elem, Gamma, I_sparse
        )

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        deg = np.degrees(theta_arr)
        ax3.plot(deg, 20 * np.log10(G_co + 1e-10), "b-", label="Collocated ULA", linewidth=2)
        ax3.plot(deg, 20 * np.log10(G_mo + 1e-10), "r--",
                 label=f"Modular ULA ($\\Gamma={Gamma}$)", linewidth=2)
        ax3.plot(deg, 20 * np.log10(G_sp + 1e-10), "g-.",
                 label=f"Sparse ULA ($I={I_sparse}$)", linewidth=2)
        ax3.axvline(90, color="k", linestyle=":", alpha=0.5, label="目标方向 (90°)")
        ax3.set_xlabel("物理角度 (度)")
        ax3.set_ylabel("归一化增益 (dB)")
        ax3.set_title("远场波束方向图: Collocated vs Modular vs Sparse")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([-40, 5])
        ax3.set_xlim([0, 180])
        st.pyplot(fig3)
        plt.close(fig3)

    st.markdown("---")
    st.subheader("能量扩散效应")
    st.markdown("远场波束赋形在**近场观测**时出现能量扩散——波束增益大幅下降且主瓣展宽。")

    with st.expander("📖 理论基础", expanded=False):
        st.markdown(r"""
**能量扩散 (Energy Spread)**

远场波束赋形向量仅补偿角度相位：

$$\mathbf{w}_{\text{FF}} = \left[ e^{j\frac{2\pi}{\lambda} m d \cos\theta'} \right]_{m}$$

当在**近场**距离 $r$ 观测时，信道响应为球面波：

$$\mathbf{a}_{\text{NF}}(r, \theta) = \left[ \sqrt{\frac{r}{r_m}} \cdot e^{-j\frac{2\pi}{\lambda} r_m} \right]_{m}$$

其中 $r_m = \sqrt{r^2 + (md)^2 - 2r(md)\cos\theta}$。

由于 $r_m$ 关于 $m$ 的**非线性**相位，与远场波束的**线性**相位不匹配，
导致内积 $\mathbf{w}_{\text{FF}}^H \mathbf{a}_{\text{NF}}$ 大幅衰减——能量在距离维度上扩散。

**物理直觉**：远场波束像手电筒，能量沿一个方向均匀扩散；
近场需要聚光灯，能量在特定点集中。距离越近，差异越大。
""")

    es_col, es_ctrl = st.columns([3, 1])
    with es_ctrl:
        st.markdown("##### 观测参数")
        M_es = st.slider("天线数量 M", 32, 1024, 256, step=32, key="es_M")
        r_near = st.number_input("近场观测距离 (m)", 5.0, 500.0, 50.0, step=5.0, key="es_r")
        theta_es_deg = st.slider("目标角度 (度)", 10, 170, 90, key="es_theta")
        theta_es = np.radians(theta_es_deg)
        delta_range = st.slider("角度扫描范围", 0.01, 0.5, 0.1, step=0.01, key="es_delta")

    with es_col:
        delta_theta = np.linspace(-delta_range, delta_range, 1000)
        G_far, G_near = compute_energy_spread(delta_theta, theta_es, r_near, M_es, d, lam)

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        ax4.plot(delta_theta, 20 * np.log10(G_far + 1e-10), "b-",
                 label=f"远场观测 ($r\\to\\infty$)", linewidth=2)
        ax4.plot(delta_theta, 20 * np.log10(G_near + 1e-10), "r--",
                 label=f"近场观测 ($r={r_near}$m)", linewidth=2)
        ax4.set_xlabel(r"$\Delta_\theta = \cos\theta' - \cos\theta$", fontsize=12)
        ax4.set_ylabel("波束聚焦增益 (dB)")
        ax4.set_title(f"能量扩散效应: 远场波束赋形在近场观测 (M={M_es}, θ={theta_es_deg}°)")
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close(fig4)

        far_peak = 20 * np.log10(np.max(G_far) + 1e-10)
        near_peak = 20 * np.log10(np.max(G_near) + 1e-10)
        c1, c2, c3 = st.columns(3)
        c1.metric("远场峰值增益", f"{far_peak:.1f} dB")
        c2.metric("近场峰值增益", f"{near_peak:.1f} dB")
        c3.metric("增益损失", f"{far_peak - near_peak:.1f} dB")
