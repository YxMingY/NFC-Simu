"""
6G 近场 XL-MIMO 仿真系统
基于 Streamlit 的交互式仿真平台
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import (
    get_nusw_vector,
    compute_dd_rayleigh,
    compute_upd,
    compute_snr_scaling,
    compute_edof_sweep,
    compute_beam_gain_grid,
    compute_beam_gain_curve,
    compute_distance_resolution,
)

# 全局 matplotlib 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="6G 近场 XL-MIMO 仿真系统",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 侧边栏 ====================
st.sidebar.title("📡 6G 近场 XL-MIMO 仿真系统")
st.sidebar.markdown("---")

module = st.sidebar.radio(
    "📋 功能模块",
    [
        "NUSW 阵列响应",
        "近场区域划分",
        "SNR 缩放律",
        "EDoF 分析",
        "波束聚焦",
        "距离分辨率",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌐 全局参数")
fc_global = st.sidebar.number_input(
    "载波频率 (GHz)", min_value=0.1, max_value=100.0, value=2.4, step=0.1
)


# ==================== 模块: NUSW 阵列响应 ====================
def page_nusw():
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

    col1, col2, col3 = st.columns(3)
    with col1:
        M = st.slider("天线数量 M", 8, 512, 128, step=8)
        r = st.number_input("用户距离 r (m)", min_value=0.1, value=10.0, step=1.0)
    with col2:
        theta_deg = st.slider("角度 θ (度)", 0, 180, 90)
        theta = np.radians(theta_deg)
        test_distances = st.multiselect(
            "测试距离 (m)",
            [1.25, 2.5, 5, 10, 25, 50, 100, 200, 500, 1000],
            default=[1.25, 5, 10, 50, 200, 1000],
        )
    with col3:
        d = lam / 2
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")
        st.metric("天线间距 d", f"{d * 1000:.2f} mm")
        st.metric("阵列孔径 D", f"{(M - 1) * d:.2f} m")

    if not test_distances:
        st.warning("请至少选择一个测试距离")
        return

    # 图1: 相位对比
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for r_test in test_distances:
        _, _, phase = get_nusw_vector(r_test, theta, M, d, lam)
        ax1.plot(np.arange(M), phase, linewidth=2, label=f"r = {r_test} m")
    ax1.set_xlabel("天线单元索引")
    ax1.set_ylabel("相位 (rad)")
    ax1.set_title("近场 vs 远场：阵列响应相位线性度对比")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig1)
    plt.close(fig1)

    # 图2: 幅度对比
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for r_test in test_distances:
        _, amp, _ = get_nusw_vector(r_test, theta, M, d, lam)
        power = np.abs(amp) ** 2
        ax2.plot(np.arange(M), power, linewidth=2, label=f"r = {r_test} m")
    ax2.set_xlabel("天线单元索引")
    ax2.set_ylabel("归一化接收功率")
    ax2.set_title("近场 vs 远场：阵列响应幅度均匀性对比")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.7)
    st.pyplot(fig2)
    plt.close(fig2)


# ==================== 模块: 近场区域划分 ====================
def page_region():
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

    col1, col2 = st.columns(2)
    with col1:
        M = st.slider("天线数量 M", 16, 512, 128, step=16, key="region_M")
        Upsilon_th = st.slider("功率均匀度阈值 Υ_th", 0.5, 0.99, 0.9, step=0.01)
    with col2:
        d = lam / 2
        D = (M - 1) * d
        r_Rayl = 2 * D**2 / lam
        st.metric("阵列孔径 D", f"{D:.2f} m")
        st.metric("经典瑞利距离", f"{r_Rayl:.1f} m")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    thetas = np.linspace(0.01, np.pi / 2, 300)

    with st.spinner("计算区域边界..."):
        r_DDRayl = compute_dd_rayleigh(M, d, lam, thetas)
        r_UPD_vals = compute_upd(M, d, lam, thetas, Upsilon_th)

    # 极坐标转笛卡尔
    def polar_to_cart(r, th):
        th2 = np.pi / 2 - th
        return r * np.cos(th2), r * np.sin(th2)

    fig, ax = plt.subplots(figsize=(9, 9))

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
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlim(0, max(x_rayl) * 1.1)
    ax.set_ylim(0, max(y_rayl) * 1.1)
    st.pyplot(fig)
    plt.close(fig)


# ==================== 模块: SNR 缩放律 ====================
def page_snr():
    st.header("SNR 缩放律分析")
    st.markdown("对比 UPW (远场)、NUSW (近场向量法) 和 NUSW (解析公式) 的 SNR 随天线数量变化。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**1. UPW 模型 (Uniform Plane Wave, 远场平面波)**

假设所有天线到用户的距离相等，阵列响应仅包含线性相位：

$$\text{SNR}_{\text{UPW}} = \bar{P} \cdot |\alpha|^2 \cdot \sum_{m=1}^{M} |a_m|^2 = \bar{P} \cdot \frac{\lambda^2}{(4\pi r)^2} \cdot M$$

SNR 随天线数 $M$ **线性增长**（dB 域中每翻倍增加 ~3dB）。这在远场成立，但在近场会**突破物理能量上限**。

**2. NUSW 模型 (Non-Uniform Spherical Wave, 近场球面波)**

精确计算每个天线到用户的球面波增益，包含幅度衰减和相位变化：

$$\text{SNR}_{\text{NUSW}} = \bar{P} \cdot |\alpha|^2 \cdot \sum_{m=1}^{M} \frac{r}{r_m}$$

其中 $r_m = \sqrt{r^2 + (md)^2 - 2r(md)\cos\theta}$。由于边缘天线的 $r_m$ 更大、增益更低，
SNR 随 $M$ 增长逐渐**饱和**，始终遵守能量守恒。

**3. 解析公式 (连续孔径近似)**

将离散阵列等效为连续积分，得到闭式解：

$$\text{SNR}_{\text{formula}} = \frac{\bar{P}\lambda^2}{(4\pi)^2 d \cdot r \sin\theta} \cdot \Delta_{\text{span}}$$

$$\Delta_{\text{span}} = \arctan\!\left(\frac{Md}{2r\sin\theta} + \cot\theta\right) + \arctan\!\left(\frac{Md}{2r\sin\theta} - \cot\theta\right)$$

当 $M \to \infty$ 时，$\Delta_{\text{span}} \to \pi$，SNR 趋于上限。该公式是**保守估计**（略低于向量法），
因为连续近似低估了离散阵列中心天线的高增益贡献。

**4. 能量守恒上限**

$$\text{SNR}_{\max} = \frac{\bar{P} \lambda^2 \pi}{(4\pi)^2 d \cdot r \sin\theta}$$

无论天线数如何增加，接收 SNR 都不可能超过此上限——这是物理定律的约束。

**核心结论**：远场模型 (UPW) 在近场会预测出违反物理定律的结果；NUSW 模型才能正确捕捉近场 SNR 饱和效应。
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        r_user = st.number_input("用户距离 r (m)", 1.0, 500.0, 15.0, step=1.0, key="snr_r")
        theta_deg = st.slider("角度 θ (度)", 0, 180, 90, key="snr_theta")
    with col2:
        P_tx_dB = st.slider("发射功率 (dBm)", 0, 50, 30, key="snr_ptx")
        sigma2_dB = st.slider("噪声功率 (dBm)", -100, 0, -60, key="snr_noise")
    with col3:
        M_min = st.number_input("M 最小值 (10^x)", 1, 7, 1, key="snr_mmin")
        M_max = st.number_input("M 最大值 (10^x)", 1, 7, 7, key="snr_mmax")

    theta = np.radians(theta_deg)
    M_list = np.array([10**x for x in range(M_min, M_max + 1)])

    with st.spinner("计算 SNR..."):
        snr_upw, snr_nusw, snr_formula, snr_limit = compute_snr_scaling(
            M_list, r_user, theta, fc_global * 1e9, P_tx_dB, sigma2_dB
        )

    eps = 1e-30
    snr_upw_dB = 10 * np.log10(np.array(snr_upw) + eps)
    snr_nusw_dB = 10 * np.log10(np.array(snr_nusw) + eps)
    snr_formula_dB = 10 * np.log10(np.array(snr_formula) + eps)
    snr_limit_dB = 10 * np.log10(snr_limit + eps)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(M_list, snr_upw_dB, "b-o", linewidth=2, markersize=6, label="UPW (远场)")
    ax.plot(M_list, snr_nusw_dB, "r-s", linewidth=2, markersize=6, label="NUSW 向量法")
    ax.plot(M_list, snr_formula_dB, "g-^", linewidth=2, markersize=6, label="NUSW 解析公式")
    ax.axhline(y=snr_limit_dB, color="k", linestyle="--", linewidth=2,
               label=f"能量守恒上限 ({snr_limit_dB:.1f} dB)")
    ax.set_xscale("log")
    ax.set_xlabel("天线数量 M")
    ax.set_ylabel("接收 SNR (dB)")
    ax.set_title(f"SNR 缩放律 (r={r_user}m, θ={theta_deg}°, f={fc_global}GHz)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.7)
    st.pyplot(fig)
    plt.close(fig)

    # 结论
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("UPW 最终 SNR", f"{snr_upw_dB[-1]:.1f} dB",
              delta="超出上限" if snr_upw_dB[-1] > snr_limit_dB else "符合")
    c2.metric("NUSW 向量法 SNR", f"{snr_nusw_dB[-1]:.1f} dB")
    c3.metric("NUSW 解析公式 SNR", f"{snr_formula_dB[-1]:.1f} dB")


# ==================== 模块: EDoF 分析 ====================
def page_edof():
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

    col1, col2, col3 = st.columns(3)
    with col1:
        Mr = st.slider("接收天线数 Mr", 4, 64, 16, key="edof_mr")
        r_user = st.number_input("用户距离 r (m)", 0.1, 50.0, 1.0, step=0.1, key="edof_r")
    with col2:
        Mt_start = st.number_input("Mt 起始值", 10, 500, 10, key="edof_mt_start")
        Mt_end = st.number_input("Mt 结束值", 50, 2000, 500, key="edof_mt_end")
        Mt_step = st.number_input("Mt 步长", 5, 100, 20, key="edof_mt_step")
    with col3:
        threshold = st.slider("EDoF 阈值", 0.01, 0.9, 0.3, step=0.01, key="edof_thr")

    Mt_list = list(range(int(Mt_start), int(Mt_end) + 1, int(Mt_step)))

    with st.spinner(f"扫描 Mt: {Mt_list[0]} → {Mt_list[-1]} (SVD 计算中)..."):
        edof_sim, edof_theory, energy_conc, ap_ratio = compute_edof_sweep(
            Mt_list, Mr, r_user, fc_global * 1e9, threshold
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"近场 XL-MIMO 信道分析 (Mr={Mr}, r={r_user}m)")

    # 图1: EDoF
    ax1 = axes[0, 0]
    ax1.plot(Mt_list, edof_sim, "b-o", markersize=3, label=f"仿真 EDoF (thr={threshold})")
    ax1.plot(Mt_list, edof_theory, "r-", linewidth=2, label="理论上限 (bounded)")
    ax1.axhline(y=Mr, color="gray", linestyle=":", label=f"Mr 上限 ({Mr})")
    ax1.set_xlabel("发射天线数 Mt")
    ax1.set_ylabel("EDoF")
    ax1.set_title("有效自由度")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 能量集中度
    ax2 = axes[0, 1]
    ax2.plot(Mt_list, energy_conc, "m-s", markersize=3, linewidth=2)
    ax2.fill_between(Mt_list, energy_conc, alpha=0.2, color="magenta")
    ax2.axhline(y=0.9, color="gray", linestyle="--", label="90% 参考线")
    ax2.set_ylim([0, 1.05])
    ax2.set_xlabel("发射天线数 Mt")
    ax2.set_ylabel("能量集中度")
    ax2.set_title("前 EDoF 个模式携带的能量占比")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 孔径利用率
    ax3 = axes[1, 0]
    ax3.plot(Mt_list, ap_ratio, "g-^", markersize=3)
    ax3.axhline(y=1.0, color="k", linestyle=":", label="理想 100%")
    ax3.set_xlabel("发射天线数 Mt")
    ax3.set_ylabel("有效孔径利用率")
    ax3.set_title("物理孔径 vs 有效孔径")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 图4: 相对误差
    ax4 = axes[1, 1]
    errors = [abs(s - t) / max(t, 1) * 100 for s, t in zip(edof_sim, edof_theory)]
    ax4.plot(Mt_list, errors, "k-x", linewidth=1.5)
    ax4.set_xlabel("发射天线数 Mt")
    ax4.set_ylabel("相对误差 (%)")
    ax4.set_title("仿真 vs 理论误差")
    ax4.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # 结论
    peak_idx = int(np.argmax(edof_sim))
    st.markdown(f"**EDoF 峰值**: Mt={Mt_list[peak_idx]}, EDoF={edof_sim[peak_idx]}  |  "
                f"**最终能量集中度**: {energy_conc[-1] * 100:.1f}%  |  "
                f"**最终孔径利用率**: {ap_ratio[-1] * 100:.1f}%")


# ==================== 模块: 波束聚焦 ====================
def page_beam():
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

    col1, col2, col3 = st.columns(3)
    with col1:
        M = st.slider("天线数量 M", 32, 1024, 512, step=32, key="beam_M")
        r_target = st.number_input("目标距离 r (m)", 5.0, 500.0, 100.0, step=5.0, key="beam_r")
    with col2:
        theta_target_deg = st.slider("目标角度 θ (度)", 10, 170, 90, key="beam_theta")
        theta_target = np.radians(theta_target_deg)
    with col3:
        grid_size = st.slider("网格分辨率", 50, 300, 150, step=10, key="beam_grid")
        d = lam / 2
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    r_grid = np.linspace(max(5, r_target * 0.3), r_target * 2.0, grid_size)
    theta_grid = np.linspace(np.pi / 6, 5 * np.pi / 6, grid_size)

    with st.spinner("计算波束聚焦增益网格 (可能需要数秒)..."):
        gain_nf, gain_ff = compute_beam_gain_grid(
            r_grid, theta_grid, r_target, theta_target, M, d, lam
        )

    # 图1: 2D 增益热力图
    fig1, axes = plt.subplots(1, 2, figsize=(16, 6))

    im1 = axes[0].pcolormesh(
        r_grid, np.degrees(theta_grid), 20 * np.log10(gain_nf + 1e-10),
        shading="auto", cmap="hot",
    )
    axes[0].plot(r_target, theta_target_deg, "o", markersize=12,
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
    axes[1].plot(r_target, theta_target_deg, "o", markersize=12,
                 markerfacecolor="none", markeredgecolor="cyan", markeredgewidth=2)
    axes[1].set_xlabel("距离 r (m)")
    axes[1].set_ylabel("角度 θ (度)")
    axes[1].set_title("远场 DFT 波束 (能量扩散)")
    plt.colorbar(im2, ax=axes[1], label="归一化增益 (dB)")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    st.pyplot(fig1)
    plt.close(fig1)

    # 图2: 距离方向切面
    st.subheader("目标方向增益随距离变化")
    r_curve = np.linspace(max(5, r_target * 0.3), r_target * 2.0, 500)
    gain_nf_curve, gain_ff_curve = compute_beam_gain_curve(
        r_curve, theta_target, r_target, M, d, lam
    )

    gain_nf_dB = 20 * np.log10(gain_nf_curve + 1e-10)
    gain_ff_dB = 20 * np.log10(gain_ff_curve + 1e-10)

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_curve, gain_nf_dB, "r-", linewidth=2.5, label="近场 NUSW 波束")
    ax.plot(r_curve, gain_ff_dB, "b--", linewidth=2.5, label="远场 DFT 波束")
    target_idx = np.argmin(np.abs(r_curve - r_target))
    ax.axvline(x=r_target, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.plot(r_target, gain_nf_dB[target_idx], "o", markersize=12,
            markerfacecolor="none", markeredgecolor="red", markeredgewidth=2.5)

    # 距离分辨率标注
    D_ap = M * d
    r_hp = 0.1 * np.sin(theta_target)**2 * (2 * D_ap**2 / lam)
    omega_res = 1.0 / r_hp
    r1_res = 1.0 / (1.0 / r_target + omega_res)
    r2_res = 1.0 / (1.0 / r_target - omega_res)
    delta_r_res = r2_res - r1_res

    ax.axvline(x=r1_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(x=r2_res, color="green", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvspan(r1_res, r2_res, alpha=0.12, color="green")

    peak_gain = gain_nf_dB[target_idx]
    ax.annotate("",
                xy=(r1_res, peak_gain + 1.5),
                xytext=(r2_res, peak_gain + 1.5),
                arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    ax.text((r1_res + r2_res) / 2, peak_gain + 2.2,
            f"$\\delta r$ = {delta_r_res:.2f} m",
            ha="center", va="bottom", fontsize=11, color="green", fontweight="bold")
    ax.text(r1_res, ax.get_ylim()[0] + 1, f"$r_1$={r1_res:.1f}m",
            ha="center", fontsize=9, color="green")
    ax.text(r2_res, ax.get_ylim()[0] + 1, f"$r_2$={r2_res:.1f}m",
            ha="center", fontsize=9, color="green")

    ax.set_xlabel("距离 r (m)")
    ax.set_ylabel("归一化增益 (dB)")
    ax.set_title(f"波束增益随距离变化 (θ = {theta_target_deg}°, $\\delta r$ = {delta_r_res:.2f} m)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim([-40, 2])
    st.pyplot(fig2)
    plt.close(fig2)


# ==================== 模块: 距离分辨率 ====================
def page_resolution():
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

    col1, col2 = st.columns(2)
    with col1:
        M = st.slider("天线数量 M", 32, 1024, 512, step=32, key="res_M")
        theta_prime_deg = st.slider("角度 θ (度)", 10, 170, 90, key="res_theta")
    with col2:
        d = lam / 2
        D = M * d
        theta_prime = np.radians(theta_prime_deg)
        r_hp = 0.1 * np.sin(theta_prime) ** 2 * (2 * D**2 / lam)
        st.metric("阵列孔径 D", f"{D:.2f} m")
        st.metric("半功率有效距离 r_hp", f"{r_hp:.1f} m")
        st.metric("波长 λ", f"{lam * 1000:.2f} mm")

    r_max = r_hp / 5.0
    r_array = np.linspace(max(1, r_max * 0.01), r_max, 2000)

    r1, r2, delta_r, delta_r_approx, _ = compute_distance_resolution(
        r_array, M, d, lam, theta_prime
    )

    # 图1: delta_r vs r (线性)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(r_array, delta_r, "b-", linewidth=2.5, label="精确值")
    ax1.plot(r_array, delta_r_approx, "g--", linewidth=2, label="近似 $2r^2/r_{hp}$")
    ax1.set_xlabel("聚焦距离 r (m)")
    ax1.set_ylabel("距离分辨率 δr (m)")
    ax1.set_title("距离分辨率 vs 聚焦距离 (线性)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 图2: 双对数
    ax2 = axes[0, 1]
    ax2.loglog(r_array, delta_r, "b-", linewidth=2.5, label="精确值")
    ax2.loglog(r_array, delta_r_approx, "g--", linewidth=2, label="近似 (斜率=2)")
    ax2.set_xlabel("聚焦距离 r (m)")
    ax2.set_ylabel("距离分辨率 δr (m)")
    ax2.set_title("距离分辨率 vs 聚焦距离 (双对数)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5, which="both")

    # 图3: r1, r, r2
    ax3 = axes[1, 0]
    ax3.plot(r_array, r1, "g-", linewidth=2, label="$r_1$ (最近可分辨)")
    ax3.plot(r_array, r_array, "k--", linewidth=1.5, label="$r$ (聚焦距离)")
    ax3.plot(r_array, r2, "r-", linewidth=2, label="$r_2$ (最远可分辨)")
    ax3.fill_between(r_array, r1, r2, alpha=0.15, color="orange", label="不可分辨区域")
    ax3.set_xlabel("聚焦距离 r (m)")
    ax3.set_ylabel("距离 (m)")
    ax3.set_title("不可分辨区域 [$r_1$, $r_2$]")
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # 图4: 典型值
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
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(True, linestyle=":", alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

    # 典型值表格
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


# ==================== 路由 ====================
module_map = {
    "NUSW 阵列响应": page_nusw,
    "近场区域划分": page_region,
    "SNR 缩放律": page_snr,
    "EDoF 分析": page_edof,
    "波束聚焦": page_beam,
    "距离分辨率": page_resolution,
}

module_map[module]()
