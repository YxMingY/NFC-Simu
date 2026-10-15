from pages.nusw import render as render_nusw
from pages.region import render as render_region
from pages.snr import render as render_snr
from pages.edof import render as render_edof
from pages.beam import render as render_beam
from pages.resolution import render as render_resolution

MODULE_MAP = {
    "NUSW 阵列响应": render_nusw,
    "近场区域划分": render_region,
    "SNR 缩放律": render_snr,
    "EDoF 分析": render_edof,
    "波束聚焦": render_beam,
    "距离分辨率": render_resolution,
}
