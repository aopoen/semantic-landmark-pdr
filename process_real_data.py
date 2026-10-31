# -*- coding: utf-8 -*-
"""
真机数据后处理：从 Phyphox 传感器数据反推转弯几何（表 3）。

对应论文 4.4 节。输入 real_data/ 下各转角的 Accelerometer.csv / Gyroscope.csv，
输出每个转弯的角度、时长、转弯半径 r 与有效位置偏差 Δ。

几何模型（论文 3.5 节）：
    转弯半径  r = v / ω      v 由步频×步长得到，ω 由转角÷时长得到
    有效位置偏差  Δ ≈ r + s   s 为步长（按身高估算：s = 0.72 × 身高/175）

转弯检测：三轴总角速度 |ω| = sqrt(gx²+gy²+gz²) 找转弯核心（对手臂摆动与手机
朝向不敏感），再用 z 轴角速度把边界外扩到"渐入渐出"段，最后对 z 轴积分得到
转角（z 轴与竖直方向近似对齐，能正确给出 90° 转角）。

用法：
    python process_real_data.py
"""
import os
import csv
import numpy as np

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'real_data')

# 数据集：目录名 / 受试者身高 / 转角类型 / 走廊宽
DATASETS = [
    ('A_175_90_2m',     175, '90°',     '约 2 m'),
    ('B_175_90_1m',     175, '90°',     '约 1 m'),
    ('C_156_90_2m',     156, '90°',     '约 2 m'),
    ('D_175_130_2.5m',  175, '约 130°', '约 2.5 m'),
]


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        rows = [[float(x) for x in row] for row in r]
    return np.array(rows)


def bandpass(x, fs, lo=0.8, hi=3.0):
    """FFT 带通：提取步态频段（去重力与高频噪声）。"""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    m = (f >= lo) & (f <= hi)
    X[~m] = 0
    return np.fft.irfft(X, n)


def detect_steps(t, acc_z, fs):
    """加速度计 z 轴带通后找正峰（步落地），间隔至少 0.25 s。"""
    bp = bandpass(acc_z - np.mean(acc_z), fs)
    peaks = [i for i in range(1, len(bp) - 1)
             if bp[i] > bp[i - 1] and bp[i] >= bp[i + 1] and bp[i] > 0.2]
    kept, last = [], -1e9
    for i in peaks:
        if t[i] - last >= 0.25:
            kept.append(i)
            last = t[i]
    return np.array(kept)


def detect_turns(tg, gx, gy, gz, core=0.65, ex_core=0.5, gap=0.6, expand=0.15):
    """转弯检测：三轴总角速度找核心 + z 轴外扩边界。

    返回 [(start, end, duration, angle_deg), ...]
    """
    w = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    ws = np.convolve(w, np.ones(31) / 31, mode='same')

    # 核心：|ω| 高于 core 进入，低于 ex_core 退出；近段合并
    segs, s = [], None
    for i in range(len(ws)):
        if s is None and ws[i] > core:
            s = i
        elif s is not None and ws[i] < ex_core:
            segs.append((s, i - 1))
            s = None
    if s is not None:
        segs.append((s, len(ws) - 1))
    merged = []
    for s, e in segs:
        if merged and tg[s] - tg[merged[-1][1]] < gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # z 轴外扩边界（去偏后），把转弯的渐入渐出段框进来
    gzc = gz - np.median(gz)
    turns = []
    for s, e in merged:
        while s > 0 and abs(gzc[s - 1]) > expand:
            s -= 1
        while e < len(gzc) - 1 and abs(gzc[e + 1]) > expand:
            e += 1
        ang = np.degrees(np.trapezoid(gzc[s:e + 1], tg[s:e + 1]))
        turns.append((tg[s], tg[e], tg[e] - tg[s], ang))
    return turns


def classify(ang):
    a = abs(ang)
    if 55 <= a <= 120:
        return '直角'
    if 120 < a <= 160:
        return '钝角'
    if 160 < a <= 215:
        return '折返'
    return '其他'


def process(dname, height):
    p = os.path.join(DATA, dname)
    acc = load_csv(os.path.join(p, 'Accelerometer.csv'))
    gyr = load_csv(os.path.join(p, 'Gyroscope.csv'))
    ta, az = acc[:, 0], acc[:, 2]
    tg, gx, gy, gz = gyr[:, 0], gyr[:, 1], gyr[:, 2], gyr[:, 3]

    fs = 1.0 / float(np.median(np.diff(tg)))
    s_len = 0.72 * height / 175.0

    steps = detect_steps(ta, az, fs)
    cadence = 1.0 / np.median(np.diff(ta[steps])) if len(steps) >= 3 else float('nan')
    v = cadence * s_len

    turns = detect_turns(tg, gx, gy, gz)
    return fs, s_len, cadence, v, turns


def main():
    print('=' * 78)
    print('真机转弯几何（对应论文表 3）')
    print('=' * 78)
    for dname, h, turn_type, width in DATASETS:
        fs, s_len, cadence, v, turns = process(dname, h)
        print('\n[%s] 身高 %d cm  %s  %s  步长估 %.2f m  步频 %.2f Hz  步速 %.2f m/s'
              % (dname, h, turn_type, width, s_len, cadence, v))

        # 只取可靠转弯（时长 0.5~5 s，|角度|>40°）
        good = [(ang, dur) for (_, _, dur, ang) in turns
                if 0.5 < dur < 5 and abs(ang) > 40]
        if not good:
            print('  无可靠转弯')
            continue
        for ang, dur in good:
            w = abs(np.radians(ang)) / max(dur, 1e-3)
            r = v / w
            delta = r + s_len
            print('  %s %+6.0f°  %.2f s  r=%.2f m  Δ=%.2f m'
                  % (classify(ang), ang, dur, r, delta))

    print('\n' + '=' * 78)
    print('说明：大角度转弯（130°/折返）的角度测量存在约 25° 的系统偏小，')
    print('      其 r、Δ 为偏大估计；90° 直角不受影响。')
    print('=' * 78)


if __name__ == '__main__':
    main()
