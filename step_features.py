# -*- coding: utf-8 -*-
"""
步态特征提取 —— 仿真端与实测端**共用同一套算法**。

为什么必须共用
--------------
仿真训练的回归模型要落到真机数据上，两边算出来的特征口径必须一致。
各写一套，模型在实测端会因为特征标度不同而失效——而且这种失效是静默的
（预测值看着像那么回事，实际是垃圾）。

为什么不用 scipy
----------------
本项目的运行环境只装了 numpy / pandas。步态信号在 5 Hz 以下，采样 50–100 Hz，
滑动平均低通在这个尺度上足够，不需要 Butterworth。

用法
----
  from step_features import bandpass_acc, step_features, FEATURE_NAMES
  acc_bp = bandpass_acc(acc_mag, fs)          # 去重力后的起伏
  f = step_features(t, acc_mag, acc_bp, gyro_mag, i, j)
"""
import numpy as np

# ★ 特征顺序的【单一来源】：训练脚本、推理脚本、模型 npz 的元数据都引用它。
#   加载模型时会断言两边一致，防止改了顺序却忘了重训。
N_FEAT = 6
FEATURE_NAMES = ["cadence", "acc_mean", "acc_std", "acc_peak", "acc_rms", "gyro_rms"]


def lowpass(x, fs, fc):
    """滑动平均低通。与 07-实测验证/process_field_data.py 的 lowpass 同算法。"""
    w = max(int(fs / max(fc, 0.05)), 1)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def bandpass_acc(acc_mag, fs, fc_hi=5.0, fc_lo=1.0):
    """去重力的步态带通：先低通掉高频抖动，再减去约 1 s 的慢变基线。

    ★ 两步都省不得：只低通不去基线，重力分量（9.8）会淹没起伏；只去基线不低通，
      高频抖动会把峰值特征算成垃圾。这个取舍与实测处理器保持一致。
    """
    a = lowpass(acc_mag, fs, fc_hi)
    return a - lowpass(a, fs, fc_lo)


def step_features(t, acc_mag, acc_bp, gyro_mag, i, j):
    """窗口 [i, j]（含端点）上的 N_FEAT 维特征（**未标准化**）。

    标准化统计量由训练脚本算、随权重一起存盘，推理时复用——否则实测端的
    标度会和仿真端对不上。
    """
    n = len(acc_mag)
    i0 = int(np.clip(i, 0, n - 1))
    j0 = int(np.clip(j, 0, n - 1))
    if j0 < i0:
        i0, j0 = j0, i0
    i1 = j0 + 1

    # 步周期：用窗口两端的时间差。至少给一个下限，避免退化窗口除零。
    dur = float(t[j0] - t[i0]) if j0 > i0 else 0.0
    dur = max(dur, 1e-3)

    seg_mag = acc_mag[i0:i1]
    seg_bp = acc_bp[i0:i1]
    seg_g = gyro_mag[i0:i1]

    return np.array([
        1.0 / dur,                              # cadence (Hz)
        float(np.mean(seg_mag)),                # 加速度模长均值（含重力）
        float(np.std(seg_mag)),                 # 起伏强度
        float(np.max(seg_bp)),                  # 去重力后峰值
        float(np.sqrt(np.mean(seg_bp ** 2))),   # 去重力后 RMS（起伏能量）
        float(np.sqrt(np.mean(seg_g ** 2))),    # 陀螺 RMS（转向剧烈程度）
    ], dtype=float)
