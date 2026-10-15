"""
近场 XL-MIMO 仿真核心计算模块
提供 NUSW 阵列响应、SNR 缩放律、EDoF 分析、波束聚焦增益等计算函数
"""
import numpy as np


def get_nusw_vector(r, theta, M, d, lam):
    """
    生成近场 NUSW (非均匀球面波) 阵列响应向量

    参数:
        r     : 用户到阵列参考点的距离 (米)
        theta : 用户相对于阵列的角度 (弧度), 0=平行于阵列, pi/2=正对阵列中心
        M     : 天线单元总数
        d     : 天线单元间距
        lam   : 信号波长 (米)

    返回:
        a_nusw       : M 个复数的阵列响应
        amplitude_term: 幅度衰减因子
        phase         : 相位 (弧度)
    """
    m = np.arange(-(M - 1) / 2, (M + 1) / 2)
    delta_m = m * d
    r_m = np.sqrt(r**2 + delta_m**2 - 2 * r * delta_m * np.cos(theta))

    amplitude_term = np.sqrt(r / r_m)
    phase = -2 * np.pi / lam * (r_m - r)
    phase_term = np.exp(1j * phase)

    a_nusw = amplitude_term * phase_term
    return a_nusw, amplitude_term, phase


def compute_dd_rayleigh(M, d, lam, thetas):
    """计算方向相关瑞利距离 r_DDRayl(theta)"""
    D = (M - 1) * d
    return 2 * D**2 * np.sin(thetas)**2 / lam


def compute_upd(M, d, lam, thetas, Upsilon_th=0.9):
    """
    数值搜索均匀功率距离 r_UPD(theta)
    在该距离之外，阵列各单元接收功率的最大波动不超过阈值 Upsilon_th
    """
    D = (M - 1) * d
    r_UPD_vals = []
    m_idx = np.arange(-(M - 1) / 2, (M + 1) / 2)
    delta_m = m_idx * d

    for th in thetas:
        r_search = np.linspace(0.1 * D, 20 * D, 1000)
        R_grid = r_search[:, np.newaxis]
        Delta_grid = delta_m[np.newaxis, :]
        R_m_grid = np.sqrt(R_grid**2 + Delta_grid**2
                           - 2 * R_grid * Delta_grid * np.cos(th))
        Power_grid = 1 / (R_m_grid**2)
        Min_Power = np.min(Power_grid, axis=1)
        Max_Power = np.max(Power_grid, axis=1)
        Ratio = Min_Power / Max_Power

        valid_indices = np.where(Ratio >= Upsilon_th)[0]
        if len(valid_indices) > 0:
            r_UPD_vals.append(r_search[valid_indices[0]])
        else:
            r_UPD_vals.append(r_search[-1])

    return np.array(r_UPD_vals)


def compute_snr_scaling(M_list, r_user, theta, fc, P_tx_dB, sigma2_dB):
    """
    计算 SNR 缩放律: UPW / NUSW向量法(修正版) / NUSW解析公式
    返回各模型的 SNR (线性值) 和能量守恒上限
    """
    c = 3e8
    lam = c / fc
    d = lam / 2
    P_tx = 10**(P_tx_dB / 10)
    sigma2 = 10**(sigma2_dB / 10)
    P_bar = P_tx / sigma2

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cot_theta = cos_theta / sin_theta if np.abs(sin_theta) > 1e-9 else 0

    snr_upw_list = []
    snr_nusw_list = []
    snr_formula_list = []

    for M in M_list:
        m_idx = np.arange(-(M - 1) / 2, (M + 1) / 2)

        # UPW 模型
        phase_upw = -2 * np.pi / lam * m_idx * d * np.cos(theta)
        a_upw = np.exp(1j * phase_upw)
        alpha_upw = np.sqrt((lam / (4 * np.pi))**2 / r_user**2) * np.exp(-1j * 2 * np.pi * r_user / lam)
        snr_upw = P_bar * np.abs(alpha_upw)**2 * np.sum(np.abs(a_upw)**2)
        snr_upw_list.append(snr_upw)

        # NUSW 向量法 (修正版：使用真实路径损耗)
        m_idx2 = np.arange(-(M - 1) / 2, (M + 1) / 2)
        r_m = np.sqrt(r_user**2 + (m_idx2 * d)**2 - 2 * r_user * (m_idx2 * d) * np.cos(theta))
        # 每个天线使用真实距离 r_m 计算路径损耗
        snr_nusw = P_bar * (lam / (4 * np.pi))**2 * np.sum(1 / r_m**2)
        snr_nusw_list.append(snr_nusw)

        # NUSW 解析公式
        term_common = (M * d) / (2 * r_user * sin_theta)
        delta_span = np.arctan(term_common + cot_theta) + np.arctan(term_common - cot_theta)
        coeff = (P_bar * lam**2) / ((4 * np.pi)**2 * d * r_user * sin_theta)
        snr_formula = coeff * delta_span
        snr_formula_list.append(snr_formula)

    eps = 1e-30
    snr_energy_limit = P_bar * lam**2 * np.pi / ((4 * np.pi)**2 * d * r_user * np.sin(theta))

    return (np.array(snr_upw_list), np.array(snr_nusw_list),
            np.array(snr_formula_list), snr_energy_limit)


def compute_edof_sweep(Mt_list, Mr, r_user, fc, threshold=0.3):
    """
    扫描 Mt 计算 EDoF (有效自由度)
    返回仿真 EDoF、理论 EDoF、能量集中度、有效孔径利用率
    """
    c = 3e8
    lam = c / fc
    d = lam / 2

    edof_sim_list = []
    edof_theory_list = []
    energy_concentration_list = []
    eff_aperture_ratio_list = []

    for Mt in Mt_list:
        mr_idx = np.arange(-(Mr - 1) / 2, (Mr + 1) / 2)
        mt_idx = np.arange(-(Mt - 1) / 2, (Mt + 1) / 2)

        dt_grid, dr_grid = np.meshgrid(mt_idx * d, mr_idx * d)
        dist_matrix = np.sqrt(r_user**2 + (dt_grid - dr_grid)**2)

        amp = np.sqrt(r_user / dist_matrix)
        phase = -2 * np.pi / lam * (dist_matrix - r_user)
        H = amp * np.exp(1j * phase)

        singular_values = np.linalg.svd(H, compute_uv=False)
        sv_max = singular_values[0]
        sv_normalized = singular_values / sv_max

        edof_sim = int(np.sum(sv_normalized > threshold))
        edof_sim_list.append(edof_sim)

        Dr = (Mr - 1) * d
        Dt = (Mt - 1) * d
        edof_theory = min((Dr * Dt) / (lam * r_user), Mr, Mt)
        edof_theory_list.append(edof_theory)

        if np.sum(singular_values**2) > 0 and edof_sim > 0:
            energy_ratio = np.sum(singular_values[:edof_sim]**2) / np.sum(singular_values**2)
        else:
            energy_ratio = 0
        energy_concentration_list.append(energy_ratio)

        Dt_phys = (Mt - 1) * d
        if Dr > 0 and lam > 0:
            Dt_eff = edof_sim * lam * r_user / Dr
            ap_ratio = Dt_eff / max(Dt_phys, 1e-10)
        else:
            ap_ratio = 0
        eff_aperture_ratio_list.append(ap_ratio)

    return (edof_sim_list, edof_theory_list,
            energy_concentration_list, eff_aperture_ratio_list)


def compute_beam_gain_grid(r_grid, theta_grid, r_target, theta_target, M, d, lam):
    """
    计算波束聚焦增益网格 (向量化)
    返回近场波束增益矩阵和远场DFT波束增益矩阵
    """
    Nr = len(r_grid)
    Nt = len(theta_grid)

    m_idx = np.arange(-(M - 1) / 2, (M + 1) / 2)
    delta_m = m_idx * d

    # 目标波束成形向量 (近场 NUSW)
    r_m_target = np.sqrt(r_target**2 + delta_m**2
                         - 2 * r_target * delta_m * np.cos(theta_target))
    amp_target = np.sqrt(r_target / r_m_target)
    phase_target = np.exp(-1j * 2 * np.pi / lam * r_m_target)
    w_target = amp_target * phase_target
    w_norm = np.linalg.norm(w_target)

    # 远场 DFT 波束
    phase_dft = 2 * np.pi / lam * delta_m * np.cos(theta_target)
    w_dft = np.exp(1j * phase_dft)
    w_dft_norm = np.linalg.norm(w_dft)

    gain_nf = np.zeros((Nt, Nr))
    gain_ff = np.zeros((Nt, Nr))

    for i, r_obs in enumerate(r_grid):
        # delta_m[:, None] → (M,1), theta_grid[None, :] → (1,Nt) → r_m: (M, Nt)
        r_m = np.sqrt(r_obs**2 + delta_m[:, np.newaxis]**2
                       - 2 * r_obs * delta_m[:, np.newaxis] * np.cos(theta_grid)[np.newaxis, :])
        amp = np.sqrt(r_obs / r_m)                        # (M, Nt)
        phase = np.exp(-1j * 2 * np.pi / lam * r_m)      # (M, Nt)
        a_obs = amp * phase                                # (M, Nt)
        a_norm = np.linalg.norm(a_obs, axis=0)             # (Nt,)

        # w_target.conj() @ a_obs → (Nt,) 对每个 theta 同时计算内积
        gain_nf[:, i] = np.abs(w_target.conj() @ a_obs) / (w_norm * a_norm)
        gain_ff[:, i] = np.abs(w_dft.conj() @ a_obs) / (w_dft_norm * a_norm)

    return gain_nf, gain_ff


def compute_beam_gain_curve(r_curve, theta_target, r_target, M, d, lam):
    """
    计算目标方向上的波束增益随距离变化曲线 (向量化)
    返回近场波束增益和远场DFT波束增益 (线性值)
    """
    m_idx = np.arange(-(M - 1) / 2, (M + 1) / 2)
    delta_m = m_idx * d

    # 目标波束成形向量
    r_m_target = np.sqrt(r_target**2 + delta_m**2
                         - 2 * r_target * delta_m * np.cos(theta_target))
    w_target = np.sqrt(r_target / r_m_target) * np.exp(-1j * 2 * np.pi / lam * r_m_target)
    w_norm = np.linalg.norm(w_target)

    # 远场 DFT 波束
    w_dft = np.exp(1j * 2 * np.pi / lam * delta_m * np.cos(theta_target))
    w_dft_norm = np.linalg.norm(w_dft)

    gain_nf = np.zeros(len(r_curve))
    gain_ff = np.zeros(len(r_curve))

    for i, r_obs in enumerate(r_curve):
        r_m = np.sqrt(r_obs**2 + delta_m**2 - 2 * r_obs * delta_m * np.cos(theta_target))
        a_obs = np.sqrt(r_obs / r_m) * np.exp(-1j * 2 * np.pi / lam * r_m)
        a_norm = np.linalg.norm(a_obs)
        gain_nf[i] = np.abs(w_target.conj() @ a_obs) / (w_norm * a_norm)
        gain_ff[i] = np.abs(w_dft.conj() @ a_obs) / (w_dft_norm * a_norm)

    return gain_nf, gain_ff


def compute_distance_resolution(r_array, M, d, lam, theta_prime=np.pi / 2):
    """
    计算距离分辨率
    返回 r1, r2, delta_r (精确值), delta_r_approx (近似值)
    """
    D = M * d
    r_hp = 0.1 * np.sin(theta_prime)**2 * (2 * D**2 / lam)
    omega = 1.0 / r_hp

    r1 = 1.0 / (1.0 / r_array + omega)
    r2 = 1.0 / (1.0 / r_array - omega)
    delta_r = r2 - r1
    delta_r_approx = 2 * r_array**2 / r_hp

    return r1, r2, delta_r, delta_r_approx, r_hp
