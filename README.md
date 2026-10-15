# 6G 近场 XL-MIMO 仿真系统

基于 Streamlit 的交互式仿真平台，用于验证和分析超大规模 MIMO (XL-MIMO) 在太赫兹/毫米波频段的**近场效应**。

## 功能模块

| 模块 | 内容 |
|------|------|
| NUSW 阵列响应 | 验证近场球面波的相位非线性和幅度非均匀性 |
| 近场区域划分 | 瑞利距离、方向相关瑞利距离、均匀功率距离的边界图 |
| SNR 缩放律 | 对比 UPW / NUSW 向量法 / 解析公式的 SNR 随天线数变化 |
| EDoF 分析 | 近场 LoS MIMO 信道的有效自由度、能量集中度、孔径利用率 |
| 波束聚焦 | 近场 NUSW 波束 (聚光灯) vs 远场 DFT 波束 (手电筒) 的增益分布 |
| 距离分辨率 | 距离聚焦的分辨率极限分析与可视化 |

每个模块均附带可展开的**理论公式说明**，包含完整的数学推导和物理含义解释。

## 项目结构

```
├── app.py              # Streamlit 入口 (侧边栏导航 + 路由)
├── core.py             # 核心计算函数 (NUSW 响应、SNR、EDoF、波束增益等)
├── pages/              # 按模块拆分的页面
│   ├── nusw.py         # NUSW 阵列响应
│   ├── region.py       # 近场区域划分
│   ├── snr.py          # SNR 缩放律
│   ├── edof.py         # EDoF 分析
│   ├── beam.py         # 波束聚焦
│   └── resolution.py   # 距离分辨率
├── requirements.txt    # Python 依赖
└── run.bat             # Windows 一键启动脚本
```

## 环境要求

- Python 3.9+
- 依赖包：`streamlit`, `numpy`, `matplotlib`, `scipy`

## 运行方式

**方式一：双击启动脚本 (Windows)**

```
双击 run.bat
```

**方式二：命令行启动**

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后浏览器会自动打开 `http://localhost:8501`。如未自动打开，请手动访问该地址。

## 使用方式

1. 左侧边栏选择功能模块
2. 调整全局参数（载波频率）
3. 在各模块内调整专属参数，图表实时更新
4. 点击「📖 理论基础与公式定义」查看公式推导
