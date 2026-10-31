# -*- coding: utf-8 -*-
"""
语义地标 vs 物理锚点 —— 定位精度对比原型（LLM 语义地标论文的第一个实验）。

问题：LLM 语义地标（零部署，σ≈0.5m，IMU 转角触发）能否逼近物理锚点
     （二维码 0.30m + 地磁指纹 2.5m，需部署）的定位精度？

对照设计（沿用 sim_core 的口径，不偏袒任何一方）：
  方案B  = AI-PDR + 地图约束 + 磁航向（无任何锚点）
  方案C  = 方案B + 物理锚点（二维码 corner_qr + 均匀锚点 anchor_spacing）
  方案S  = 方案B + 语义地标（IMU 转角检测 |dpsi|>阈值 -> 位置约束到最近地标）

语义地标观测模型（关键设计，与物理锚点不同）：
  物理锚点观测 = 当前真实位置 + N(0, σ)      （扫码给出"我在哪"，σ 是观测噪声）
  语义地标观测 = 地标名义坐标 + N(0, σ_landmark)（检测到"在转角"，σ 是地标物理范围）

  触发机制：|dpsi| > turn_thresh（动作检测），不靠"位置靠近"，
  规避了 README 第 344 行警告的触发半径注入误差问题。
  滞回：检测到转角后置 armed=False，|dpsi| 回落到 0.5*阈值才重新武装，
        保证一个转角只触发一次（转弯跨多步，避免重复观测）。

评估口径：末端误差的 p90 / p95（与 facts.py 的 MAIN_P90_END 同口径）。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# numpy<2.0 兼容：sim_core 用了 np.trapezoid（numpy 2.0 才有的新名字），旧版叫 np.trapz
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import (gen_truth, simulate_steps, SensorCfg, in_map,
                      systematic_resample, ENTRY, arc_length, mag_sigma_at)
from step_model import predict_steps_sl, get_model

# ---------------------------------------------------------------- 语义地标库
# 从地图几何（RECTS / WAYPOINTS）提取的"精确语义点"：环形 4 转角 + 折返点 + 2 门。
# 现实中这些点的坐标可从校园地图的 POI/建筑/路网自动提取（LLM 地标生成层）。
LANDMARKS = np.array([
    (58.5, 1.5),    # 右下转角（主走廊 -> 右走廊）
    (58.5, 20.5),   # 右上转角（右走廊 -> 顶走廊）
    (30.0, 20.5),   # 左上转角（顶走廊 -> 支走廊）
    (30.0, 1.5),    # 左下转角（支走廊 -> 主走廊）
    (4.0, 1.5),     # 折返点
    (10.0, 3.5),    # 房间 101 门
    (45.0, 3.5),    # 房间 102 门
], dtype=float)


def run_pf_semantic(seed, steps, cfg, turn_thresh_deg=20.0, landmark_sigma=0.5,
                    use_map=True, use_mag=True, oracle_match=True):
    """粒子滤波 + 语义地标（IMU 转角检测触发，滞回：一个转角只触发一次）。

    oracle_match=True  : 检测到转角后，用真实位置匹配最近地标（上限估计）。
    oracle_match=False : 用滤波器当前估计位置匹配最近地标（更现实，会有误匹配）。
    """
    rng = np.random.default_rng(seed + 555001)
    N = cfg.n_particles
    turn_thresh = np.deg2rad(turn_thresh_deg)

    px = np.full(N, ENTRY[0]) + rng.normal(0, 0.15, N)
    py = np.full(N, ENTRY[1]) + rng.normal(0, 0.15, N)
    ph = np.full(N, 0.0) + rng.normal(0, np.deg2rad(3.0), N)

    est_t, est_x, est_y = [], [], []

    sl_seq = predict_steps_sl(steps)
    proc = cfg.step_sl_proc_std
    if proc is None:
        proc = float(get_model()[1].get("proc_std", 0.03))

    n_lm_hits = 0      # 统计触发了多少次语义地标观测
    turn_armed = True  # 滞回开关：转完才重新武装

    for k, s in enumerate(steps):
        # ---- 状态转移（同 run_particle_filter）----
        ph = ph + s["dpsi"] + rng.normal(0, np.deg2rad(1.2), N)
        sl = sl_seq[k] * (1.0 + proc * rng.standard_normal(N))
        px = px + sl * np.cos(ph)
        py = py + sl * np.sin(ph)

        logw = np.zeros(N)

        # ---- 地图约束（不可穿墙）----
        if use_map:
            ok = in_map(px, py)
            logw += np.where(ok, 0.0, -60.0)

        # ---- 磁力计航向量测（与原 run_particle_filter 方案B 完全一致，不做偏差在线估计）----
        if use_mag:
            sig = mag_sigma_at(np.array([px.mean()]), np.array([py.mean()]),
                               cfg.mag_std_clean, cfg.mag_std_disturb)[0]
            m = s["mag"]  # b_hat 恒为 0，同原方案B
            d = np.arctan2(np.sin(m - ph), np.cos(m - ph))
            logw += -0.5 * (d / sig) ** 2

        # ---- 语义地标观测：IMU 转角检测触发（滞回：一个转角只触发一次）----
        if abs(s["dpsi"]) > turn_thresh and turn_armed:
            turn_armed = False
            if oracle_match:
                # 上限：真实位置匹配最近地标
                dd = np.hypot(LANDMARKS[:, 0] - s["x_true"],
                              LANDMARKS[:, 1] - s["y_true"])
            else:
                # 现实：滤波器估计位置匹配最近地标（会有误匹配风险）
                ex = float(np.mean(px)); ey = float(np.mean(py))
                dd = np.hypot(LANDMARKS[:, 0] - ex, LANDMARKS[:, 1] - ey)
            j = int(np.argmin(dd))
            lx, ly = LANDMARKS[j]
            # 观测 = 地标名义坐标 + σ_landmark（物理范围）
            logw += -0.5 * ((px - lx) / landmark_sigma) ** 2
            logw += -0.5 * ((py - ly) / landmark_sigma) ** 2
            n_lm_hits += 1
        elif abs(s["dpsi"]) < turn_thresh * 0.5:
            turn_armed = True  # 转完，重新武装

        # ---- 权重归一化 ----
        logw -= logw.max()
        w = np.exp(logw)
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 0:
            w = np.ones(N) / N
        else:
            w = w / sw

        # ---- 重采样 ----
        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * N:
            idx = systematic_resample(w, rng)
            px, py, ph = px[idx], py[idx], ph[idx]
            px = px + rng.normal(0, 0.05, N)
            py = py + rng.normal(0, 0.05, N)
            ph = ph + rng.normal(0, np.deg2rad(0.8), N)

        est_t.append(s["t"])
        est_x.append(float(np.mean(px)))
        est_y.append(float(np.mean(py)))

    return (np.array(est_t), np.array(est_x), np.array(est_y)), n_lm_hits


def eval_config(run_fn, n_runs, cfg, seed0, **kw):
    """跑 n_runs 次，返回各 run 的【末端误差】数组（与 facts.py MAIN_P90_END 同口径）。"""
    end_errs = []
    n_hits = 0
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        (te, xe, ye), hits = run_fn(seed, steps, cfg, **kw)
        # 末端误差：估计最后一步步末位置 vs 真值在同刻位置
        tx_end = float(np.interp(te[-1], tr["t"], tr["x"]))
        ty_end = float(np.interp(te[-1], tr["t"], tr["y"]))
        end_errs.append(float(np.hypot(xe[-1] - tx_end, ye[-1] - ty_end)))
        n_hits += hits
    return np.array(end_errs), n_hits / max(n_runs, 1)


def main():
    cfg = SensorCfg()
    n_runs = 40
    seed0 = 1000

    from sim_core import run_particle_filter

    print("=" * 78)
    print("语义地标 vs 物理锚点 末端误差对比（%d 次蒙特卡洛，与 facts.py 同口径）" % n_runs)
    print("=" * 78)

    # ---- 方案B：无锚点基线 ----
    errB, _ = eval_config(
        lambda seed, steps, cfg, **kw: (
            run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                                anchor_spacing=None, corner_qr=False), 0),
        n_runs, cfg, seed0)
    print("方案B  无锚点基线             p90 = %5.2f m   p95 = %5.2f m" %
          (np.percentile(errB, 90), np.percentile(errB, 95)))

    # ---- 方案C：物理锚点（二维码 + 均匀锚点 15m）----
    errC, _ = eval_config(
        lambda seed, steps, cfg, **kw: (
            run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                                anchor_spacing=15.0, corner_qr=True), 0),
        n_runs, cfg, seed0)
    print("方案C  物理锚点(二维码+15m)   p90 = %5.2f m   p95 = %5.2f m" %
          (np.percentile(errC, 90), np.percentile(errC, 95)))

    # ---- 方案S：语义地标（σ 扫描，滞回触发，oracle 匹配）----
    for lm_sigma in (0.30, 0.50, 1.00, 2.00):
        errS, hits = eval_config(run_pf_semantic, n_runs, cfg, seed0,
                                 turn_thresh_deg=20.0, landmark_sigma=lm_sigma,
                                 oracle_match=True)
        print("方案S  语义地标 σ=%.2fm(oracle) p90 = %5.2f m   p95 = %5.2f m  (均触发 %.1f 次)" %
              (lm_sigma, np.percentile(errS, 90), np.percentile(errS, 95), hits))

    # ---- 方案S：现实匹配（滤波器估计位置匹配，含误匹配风险）----
    errS_real, hits = eval_config(run_pf_semantic, n_runs, cfg, seed0,
                                  turn_thresh_deg=20.0, landmark_sigma=0.5,
                                  oracle_match=False)
    print("方案S  语义地标 σ=0.50m(现实) p90 = %5.2f m   p95 = %5.2f m  (均触发 %.1f 次)" %
          (np.percentile(errS_real, 90), np.percentile(errS_real, 95), hits))

    print("=" * 78)


if __name__ == "__main__":
    main()
