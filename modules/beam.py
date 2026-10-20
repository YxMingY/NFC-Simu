import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import (compute_beam_gain_grid, compute_beam_gain_curve,
                  compute_array_beams, compute_energy_spread)


def render(fc_global):
    st.header("波束聚焦图案分析")
    st.markdown("从**观测域**（近场/远场）与**波束成形域**（近场/远场）两个维度，分析增益分布与能量聚焦特性。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**远场 vs 近场波束成形**

传统远场波束成形 (DFT 波束) 仅补偿**角度**维度的相位差：

$$w_m^{\text{DFT}} = \exp\!\left(j\frac{2\pi}{\lambda} m d \cos\theta'\right)$$

近场 NUSW 波束成形同时补偿**角度和距离**两个维度的相位和幅度：

$$w_m^{\text{NUSW}} = \sqrt{\frac{r'}{r_m'}} \cdot \exp\!\left(-j\frac{2\pi}{\lambda}(r_m' - r')\right)$$

其中 $r_m' = \sqrt{r'^2 + (md)^2 - 2r'(md)\cos\theta'}$。

**归一化增益定义**

$$G(\mathbf{s}; \mathbf{s}') = \frac{|\mathbf{w}^H(\mathbf{s}') \cdot \mathbf{a}(\mathbf{s})|}{\|\mathbf{w}(\mathbf{s}')\| \cdot \|\mathbf{a}(\mathbf{s})\|}$$

**三种场景对比**：
| 场景 | 观测域 | 波束成形 | 效果 |
|------|--------|----------|------|
| 近场观测 + 近场波束 | 球面波 | NUSW | 聚光灯，完美聚焦 |
| 近场观测 + 远场波束 | 球面波 | DFT | 能量扩散，增益损失 |
| 远场观测 + 远场波束 | 平面波 | DFT | 标准波束方向图 |
""")

    c = 3e8
    lam = c / (fc_global * 1e9)
    d = lam / 2

    st.markdown("##### 全局参数")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        M = st.slider("天线数量 M", 32, 1024, 512, step=32, key="beam_M")
    with pc2:
        r_target = st.number_input("目标距离 r (m)", 5.0, 500.0, 100.0, step=5.0, key="beam_r")
    with pc3:
        theta_target_deg = st.slider("目标角度 θ (度)", 10, 170, 90, key="beam_theta")
    with pc4:
        grid_size = st.slider("网格分辨率", 50, 300, 150, step=10, key="beam_grid")
    theta_target = np.radians(theta_target_deg)
    st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    r_grid = np.linspace(max(5, r_target * 0.3), r_target * 2.0, grid_size)
    theta_grid = np.linspace(np.pi / 6, 5 * np.pi / 6, grid_size)

    with st.spinner("计算波束增益网格..."):
        gain_nf, gain_ff = compute_beam_gain_grid(
            r_grid, theta_grid, r_target, theta_target, M, d, lam
        )

    r_curve = np.linspace(max(5, r_target * 0.3), r_target * 2.0, 500)
    gain_nf_curve, gain_ff_curve = compute_beam_gain_curve(
        r_curve, theta_target, r_target, M, d, lam
    )
    gain_nf_dB = 20 * np.log10(gain_nf_curve + 1e-10)
    gain_ff_dB = 20 * np.log10(gain_ff_curve + 1e-10)

    D_ap = M * d
    r_hp = 0.1 * np.sin(theta_target)**2 * (2 * D_ap**2 / lam)
    omega_res = 1.0 / r_hp
    r1_res = 1.0 / (1.0 / r_target + omega_res)
    r2_res = 1.0 / (1.0 / r_target - omega_res)
    delta_r_res = r2_res - r1_res

    # ================================================================
    # 场景一：近场观测与近场波束成形
    # ================================================================
    st.markdown("---")
    st.subheader("近场观测与近场波束成形")
    st.markdown("近场 NUSW 波束在近场观测下实现**聚光灯效应**——能量在目标点 $(r', \\theta')$ 精准聚焦。")

    s1_heat, s1_curve = st.columns(2)
    with s1_heat:
        fig1a, ax1a = plt.subplots(figsize=(5, 4))
        im1 = ax1a.pcolormesh(
            r_grid, np.degrees(theta_grid), 20 * np.log10(gain_nf + 1e-10),
            shading="auto", cmap="hot",
        )
        ax1a.plot(r_target, theta_target_deg, "o", markersize=10,
                  markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
        ax1a.set_xlabel("距离 r (m)")
        ax1a.set_ylabel("角度 θ (度)")
        ax1a.set_title("NUSW 波束聚焦 (聚光灯)")
        plt.colorbar(im1, ax=ax1a, label="增益 (dB)")
        ax1a.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig1a)
        plt.close(fig1a)

    with s1_curve:
        fig1b, ax1b = plt.subplots(figsize=(5, 4))
        ax1b.plot(r_curve, gain_nf_dB, "r-", linewidth=2.5, label="近场 NUSW 波束")
        target_idx = np.argmin(np.abs(r_curve - r_target))
        ax1b.axvline(x=r_target, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1b.plot(r_target, gain_nf_dB[target_idx], "o", markersize=10,
                  markerfacecolor="none", markeredgecolor="red", markeredgewidth=2.5)
        ax1b.axvline(x=r1_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1b.axvline(x=r2_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1b.axvspan(r1_res, r2_res, alpha=0.12, color="green")
        ax1b.set_xlabel("距离 r (m)")
        ax1b.set_ylabel("归一化增益 (dB)")
        ax1b.set_title(f"目标方向增益 ($\\delta r$ = {delta_r_res:.2f} m)")
        ax1b.legend(fontsize=9)
        ax1b.grid(True, linestyle=":", alpha=0.5)
        ax1b.set_ylim([-40, 5])
        peak_gain = gain_nf_dB[target_idx]
        arrow_y = min(peak_gain + 1.5, 4.0)
        text_y = min(peak_gain + 2.5, 4.6)
        ax1b.annotate("", xy=(r1_res, arrow_y), xytext=(r2_res, arrow_y),
                      arrowprops=dict(arrowstyle="<->", color="green", lw=2))
        ax1b.text((r1_res + r2_res) / 2, text_y,
                  f"$\\delta r$ = {delta_r_res:.2f} m",
                  ha="center", va="bottom", fontsize=10, color="green", fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.85))
        st.pyplot(fig1b)
        plt.close(fig1b)

    # ================================================================
    # 场景二：近场观测与远场波束赋形
    # ================================================================
    st.markdown("---")
    st.subheader("近场观测与远场波束赋形")
    st.markdown("远场 DFT 波束在近场观测时出现**能量扩散**——波束增益大幅下降，主瓣展宽。")

    s2_heat, s2_curve = st.columns(2)
    with s2_heat:
        fig2a, ax2a = plt.subplots(figsize=(5, 4))
        im2 = ax2a.pcolormesh(
            r_grid, np.degrees(theta_grid), 20 * np.log10(gain_ff + 1e-10),
            shading="auto", cmap="hot",
        )
        ax2a.plot(r_target, theta_target_deg, "o", markersize=10,
                  markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
        ax2a.set_xlabel("距离 r (m)")
        ax2a.set_ylabel("角度 θ (度)")
        ax2a.set_title("DFT 波束在近场 (能量扩散)")
        plt.colorbar(im2, ax=ax2a, label="增益 (dB)")
        ax2a.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig2a)
        plt.close(fig2a)

    with s2_curve:
        fig2b, ax2b = plt.subplots(figsize=(5, 4))
        ax2b.plot(r_curve, gain_ff_dB, "b--", linewidth=2.5, label="远场 DFT 波束")
        ax2b.axvline(x=r_target, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2b.set_xlabel("距离 r (m)")
        ax2b.set_ylabel("归一化增益 (dB)")
        ax2b.set_title("目标方向增益 (远场波束)")
        ax2b.legend(fontsize=9)
        ax2b.grid(True, linestyle=":", alpha=0.5)
        ax2b.set_ylim([-40, 5])
        st.pyplot(fig2b)
        plt.close(fig2b)

    st.markdown("**能量扩散量化分析**")
    es_col, es_ctrl = st.columns([3, 1])
    with es_ctrl:
        st.markdown("##### 观测参数")
        M_es = st.slider("天线数量 M", 32, 1024, 256, step=32, key="es_M")
        r_far = st.number_input("远场观测距离 (m)", 100.0, 10000.0, 2000.0, step=100.0, key="es_r_far")
        r_near = st.number_input("近场观测距离 (m)", 5.0, 500.0, 50.0, step=5.0, key="es_r")
        theta_es_deg = st.slider("目标角度 (度)", 10, 170, 90, key="es_theta")
        theta_es = np.radians(theta_es_deg)
        delta_range = st.slider("角度扫描范围", 0.01, 0.5, 0.1, step=0.01, key="es_delta")

    with es_col:
        delta_theta = np.linspace(-delta_range, delta_range, 1000)
        G_far, G_near_es = compute_energy_spread(delta_theta, theta_es, r_near, r_far, M_es, d, lam)

        fig2c, ax2c = plt.subplots(figsize=(9, 4))
        ax2c.plot(delta_theta, 20 * np.log10(G_far + 1e-10), "b-",
                  label=f"远场观测 ($r={r_far:.0f}$m)", linewidth=2)
        ax2c.plot(delta_theta, 20 * np.log10(G_near_es + 1e-10), "r--",
                  label=f"近场观测 ($r={r_near:.0f}$m)", linewidth=2)
        ax2c.set_xlabel(r"$\Delta_\theta = \cos\theta' - \cos\theta$", fontsize=12)
        ax2c.set_ylabel("波束聚焦增益 (dB)")
        ax2c.set_title(f"能量扩散效应: 远场波束赋形在近场观测 (M={M_es}, θ={theta_es_deg}°)")
        ax2c.legend(fontsize=11)
        ax2c.grid(True, alpha=0.3)
        st.pyplot(fig2c)
        plt.close(fig2c)

        far_peak = 20 * np.log10(np.max(G_far) + 1e-10)
        near_peak = 20 * np.log10(np.max(G_near_es) + 1e-10)
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("远场峰值增益", f"{far_peak:.1f} dB")
        ec2.metric("近场峰值增益", f"{near_peak:.1f} dB")
        ec3.metric("增益损失", f"{far_peak - near_peak:.1f} dB")

    # ================================================================
    # 场景三：远场观测与远场波束赋形
    # ================================================================
    st.markdown("---")
    st.subheader("远场观测与远场波束赋形")
    st.markdown("对比**共址 ULA**、**模块化 ULA** 和**稀疏 ULA** 三种阵列架构的远场波束方向图。")

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
