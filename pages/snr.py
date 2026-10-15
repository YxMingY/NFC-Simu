import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from core import compute_snr_scaling


def render(fc_global):
    st.header("SNR 缩放律分析")
    st.markdown("对比 UPW (远场)、NUSW (近场向量法) 和 NUSW (解析公式) 的 SNR 随天线数量变化。")

    with st.expander("📖 理论基础与公式定义", expanded=False):
        st.markdown(r"""
**1. UPW 模型 (Uniform Plane Wave, 远场平面波)**

假设所有天线到用户的距离相等，阵列响应仅包含线性相位：

$$\text{SNR}_{\text{UPW}} = \bar{P} \cdot |\alpha|^2 \cdot \sum_{m=1}^{M} |a_m|^2 = \bar{P} \cdot \frac{\lambda^2}{(4\pi r)^2} \cdot M$$

SNR 随天线数 $M$ **线性增长**（dB 域中每翻倍增加 ~3dB）。这在远场成立，但在近场会**突破物理能量上限**。

**2. NUSW 向量法 (离散逐天线求和)**

精确计算每个天线到用户的球面波增益，包含幅度衰减和相位变化：

$$\text{SNR}_{\text{NUSW}} = \bar{P} \cdot |\alpha|^2 \cdot \sum_{m=1}^{M} \frac{r}{r_m}$$

其中 $r_m = \sqrt{r^2 + (md)^2 - 2r(md)\cos\theta}$。由于边缘天线的 $r_m$ 更大、增益更低，
SNR 随 $M$ 增长逐渐**饱和**，始终遵守能量守恒。

**3. NUSW 解析公式 (连续孔径近似)**

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

    with st.expander("💻 向量法与解析公式的计算流程对比", expanded=False):
        st.markdown(r"""
**向量法 (NUSW Vector Sum) — 离散精确计算**

逐步计算过程：
1. 生成以阵列中心为原点的天线索引 $m = [-(M-1)/2, \ldots, (M-1)/2]$
2. 对**每一个**天线单元 $m$，用余弦定理精确计算到用户的欧氏距离：$r_m = \sqrt{r^2 + (md)^2 - 2r(md)\cos\theta}$
3. 计算该天线的球面波功率增益：$|a_m|^2 = r / r_m$（幅度衰减的平方）
4. 对所有 $M$ 个天线的功率增益**逐元素求和**：$\sum_{m=1}^{M} |a_m|^2$
5. 乘以路径损耗和发射参数得到最终 SNR

特点：保留了离散阵列的每一个天线贡献，中心天线距离最近、增益最高，
边缘天线距离较远、增益被压低。求和结果是**精确的离散 SNR**。

---

**解析公式法 (Analytical Formula) — 连续积分近似**

逐步计算过程：
1. 将离散天线阵列**等效替换**为一段长度为 $Md$ 的连续孔径（积分代替求和）
2. 对连续孔径上的功率增益做定积分：$\int_{-Md/2}^{Md/2} \frac{r}{r(x)} dx$，其中 $r(x) = \sqrt{r^2 + x^2 - 2rx\cos\theta}$
3. 该积分有**闭式解析解**，化简后得到 $\frac{\lambda^2}{(4\pi)^2 d \cdot r\sin\theta} \cdot \Delta_{\text{span}}$
4. 其中 $\Delta_{\text{span}}$ 是两个 $\arctan$ 的和，反映阵列在用户方向上的"有效角度跨度"

特点：积分是求和的"平滑版本"——它忽略了离散阵列中心天线的峰值采样效应，
因此结果**系统性地略低于**向量法。差异随 $M$ 增大而趋于稳定。

---

**两者差异的物理本质**

| | 向量法 | 解析公式 |
|---|---|---|
| 模型 | 离散 $M$ 个天线 | 连续孔径 |
| 数学 | 有限项求和 $\sum$ | 定积分 $\int$ |
| 计算量 | $O(M)$，逐天线遍历 | $O(1)$，直接代入公式 |
| 精度 | 精确（离散模型下） | 保守估计（略低 ~0.5-1 dB） |
| 适用场景 | 仿真验证、精确评估 | 理论推导、Scaling Law 分析 |
""")

    main_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        st.markdown("##### 参数设置")
        r_user = st.number_input("用户距离 r (m)", 1.0, 500.0, 15.0, step=1.0, key="snr_r")
        theta_deg = st.slider("角度 θ (度)", 0, 180, 90, key="snr_theta")
        P_tx_dB = st.slider("发射功率 (dBm)", 0, 50, 30, key="snr_ptx")
        sigma2_dB = st.slider("噪声功率 (dBm)", -100, 0, -60, key="snr_noise")
        st.markdown("---")
        st.markdown("##### M 扫描范围")
        M_min = st.number_input("M 最小值 (10^x)", 1, 7, 1, key="snr_mmin")
        M_max = st.number_input("M 最大值 (10^x)", 1, 7, 7, key="snr_mmax")

    with main_col:
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

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(M_list, snr_upw_dB, "b-o", linewidth=2, markersize=6, label="UPW (远场)")
        ax.plot(M_list, snr_nusw_dB, "r-s", linewidth=2, markersize=6, label="NUSW 向量法")
        ax.plot(M_list, snr_formula_dB, "g-^", linewidth=2, markersize=6, label="NUSW 解析公式")
        ax.axhline(y=snr_limit_dB, color="k", linestyle="--", linewidth=2,
                   label=f"能量守恒上限 ({snr_limit_dB:.1f} dB)")
        ax.set_xscale("log")
        ax.set_xlabel("天线数量 M")
        ax.set_ylabel("接收 SNR (dB)")
        ax.set_title(f"SNR 缩放律 (r={r_user}m, θ={theta_deg}°, f={fc_global}GHz)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle=":", alpha=0.7)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("UPW 最终 SNR", f"{snr_upw_dB[-1]:.1f} dB",
                  delta="超出上限" if snr_upw_dB[-1] > snr_limit_dB else "符合")
        c2.metric("NUSW 向量法 SNR", f"{snr_nusw_dB[-1]:.1f} dB")
        c3.metric("NUSW 解析公式 SNR", f"{snr_formula_dB[-1]:.1f} dB")
