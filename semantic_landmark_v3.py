# -*- coding: utf-8 -*-
"""
语义地标 v3 —— 有效位置实验（层2 核心假设验证）。

v2 结论：peak 触发把 p90 从 2.89 压到 2.10m，但仍差物理锚点 1.0m。
根因假设：地标观测用的是"名义坐标"（走廊交点），而 peak 触发发生在转角出口，
        粒子真实位置离名义坐标 ~1m，强观测把粒子拉向名义坐标 = 注入系统偏差。

本实验验证：若地标观测位置换成"有效位置"（实际转弯处），偏差能否消除。
三个变体（都 peak 触发 + 点地标 σ=0.50m）：
  nominal         : 地标 = 名义坐标（v2 peak+point 基线，复现 2.10m）
  effective_true  : 地标 = 触发时刻真实位置（oracle 上限，动态）
  effective_static: 地标 = 一次干净轨迹提取的转弯出口静态位置（LLM 可达，静态）

effective_true  回答"地标位置完美时能到多少"（假设上限）。
effective_static 回答"一个固定的有效位置（不依赖具体运行噪声）能否逼近 oracle"，
                 这才是 LLM 从地图几何能输出的东西。

诊断：额外打印 peak 触发时刻真实位置 vs 名义坐标的偏差，量化"名义坐标错多少"。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import (gen_truth, simulate_steps, SensorCfg, in_map,
                      systematic_resample, ENTRY, mag_sigma_at, WAYPOINTS, _LOOP)
from step_model import predict_steps_sl, get_model

# ---------------------------------------------------------------- 名义坐标（v1/v2）
LANDMARKS = np.array([
    (58.5, 1.5), (58.5, 20.5), (30.0, 20.5), (30.0, 1.5),
    (4.0, 1.5), (10.0, 3.5), (45.0, 3.5),
], dtype=float)


def extract_turn_exit_positions(seed=1000, turn_thresh_deg=20.0):
    """从一条干净真值轨迹提取各转角的 peak 触发时刻真实位置（静态有效位置）。

    逐圈逐转角触发，收集所有触发时刻的 (x_true, y_true)，对每个地标取中位数，
    得到一个"对所有圈都成立"的静态有效位置。这是 LLM 层2 可输出的东西的
    oracle 近似：一个固定的、不依赖单次运行噪声的位置。
    """
    cfg = SensorCfg()
    tr = gen_truth(seed)
    steps = simulate_steps(seed, tr, cfg)
    turn_thresh = np.deg2rad(turn_thresh_deg)

    hits = {j: [] for j in range(len(LANDMARKS))}
    turn_armed, turn_active = True, False

    for s in steps:
        abs_d = abs(s["dpsi"])
        trigger = False
        if abs_d > turn_thresh and turn_armed:
            turn_active, turn_armed = True, False
        elif turn_active and abs_d < turn_thresh * 0.5:
            trigger, turn_active, turn_armed = True, False, True
        if trigger:
            # 用真实位置匹配最近地标（oracle 匹配，只用于"归入哪个地标"）
            d = np.hypot(LANDMARKS[:, 0] - s["x_true"], LANDMARKS[:, 1] - s["y_true"])
            j = int(np.argmin(d))
            hits[j].append((s["x_true"], s["y_true"]))

    eff = np.array(LANDMARKS, dtype=float)
    for j in range(len(LANDMARKS)):
        if hits[j]:
            arr = np.array(hits[j])
            eff[j] = np.median(arr, axis=0)
    return eff


def run_pf_v3(seed, steps, cfg, pos_mode="nominal", static_pos=None,
              turn_thresh_deg=20.0, landmark_sigma=0.5,
              use_map=True, use_mag=True):
    """统一粒子滤波：pos_mode ∈ {nominal, effective_true, effective_static}。"""
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

    n_lm_hits = 0
    turn_armed, turn_active = True, False
    # 诊断：peak 触发时刻 真实位置 vs 名义坐标 的偏差
    diag_dev = []

    for k, s in enumerate(steps):
        abs_d = abs(s["dpsi"])

        ph = ph + s["dpsi"] + rng.normal(0, np.deg2rad(1.2), N)
        sl = sl_seq[k] * (1.0 + proc * rng.standard_normal(N))
        px = px + sl * np.cos(ph)
        py = py + sl * np.sin(ph)

        logw = np.zeros(N)

        if use_map:
            ok = in_map(px, py)
            logw += np.where(ok, 0.0, -60.0)

        if use_mag:
            sig = mag_sigma_at(np.array([px.mean()]), np.array([py.mean()]),
                               cfg.mag_std_clean, cfg.mag_std_disturb)[0]
            m = s["mag"]
            d = np.arctan2(np.sin(m - ph), np.cos(m - ph))
            logw += -0.5 * (d / sig) ** 2

        # ---- peak 触发（同 v2）----
        trigger = False
        if abs_d > turn_thresh and turn_armed:
            turn_active, turn_armed = True, False
        elif turn_active and abs_d < turn_thresh * 0.5:
            trigger, turn_active, turn_armed = True, False, True

        if trigger:
            # 统一 oracle 匹配：用真实位置匹配最近地标（对齐 v2 的 oracle_match=True）。
            # 匹配只决定"这是哪个地标"，观测位置由 pos_mode 决定。
            qx, qy = s["x_true"], s["y_true"]
            if pos_mode == "nominal":
                d = np.hypot(LANDMARKS[:, 0] - qx, LANDMARKS[:, 1] - qy)
                j = int(np.argmin(d))
                lx, ly = LANDMARKS[j]
            elif pos_mode == "effective_true":
                # oracle：地标就在触发时刻的真实位置
                lx, ly = qx, qy
                d = np.hypot(LANDMARKS[:, 0] - qx, LANDMARKS[:, 1] - qy)
                j = int(np.argmin(d))
                diag_dev.append(float(np.hypot(lx - LANDMARKS[j, 0],
                                               ly - LANDMARKS[j, 1])))
            elif pos_mode == "effective_static":
                d = np.hypot(static_pos[:, 0] - qx, static_pos[:, 1] - qy)
                j = int(np.argmin(d))
                lx, ly = static_pos[j]
            else:
                raise ValueError(pos_mode)
            dx = px - lx
            dy = py - ly
            logw += -0.5 * (dx / landmark_sigma) ** 2 + -0.5 * (dy / landmark_sigma) ** 2
            n_lm_hits += 1

        logw -= logw.max()
        w = np.exp(logw)
        sw = w.sum()
        if not np.isfinite(sw) or sw <= 0:
            w = np.ones(N) / N
        else:
            w = w / sw

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

    return (np.array(est_t), np.array(est_x), np.array(est_y)), n_lm_hits, diag_dev


def eval_config(run_fn, n_runs, cfg, seed0, **kw):
    end_errs, n_hits, diag = [], 0, []
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        (te, xe, ye), hits, dg = run_fn(seed, steps, cfg, **kw)
        tx_end = float(np.interp(te[-1], tr["t"], tr["x"]))
        ty_end = float(np.interp(te[-1], tr["t"], tr["y"]))
        end_errs.append(float(np.hypot(xe[-1] - tx_end, ye[-1] - ty_end)))
        n_hits += hits
        diag.extend(dg)
    return np.array(end_errs), n_hits / max(n_runs, 1), diag


def main():
    cfg = SensorCfg()
    n_runs = 40
    seed0 = 1000

    from sim_core import run_particle_filter

    print("=" * 84)
    print("语义地标 v3：有效位置实验（层2 核心假设验证，%d 次蒙特卡洛）" % n_runs)
    print("=" * 84)

    # 方案C 物理锚点（基准）
    errC, _, _ = eval_config(
        lambda seed, steps, cfg, **kw: (
            run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                                anchor_spacing=15.0, corner_qr=True), 0, []),
        n_runs, cfg, seed0)
    print("方案C  物理锚点(二维码+15m)         p90 = %5.2f m  p95 = %5.2f m" %
          (np.percentile(errC, 90), np.percentile(errC, 95)))

    # 静态有效位置（LLM 可达）
    static_pos = extract_turn_exit_positions(seed=1000)

    # nominal：v2 基线（oracle 匹配，对齐 v2）
    errN, hitsN, _ = eval_config(run_pf_v3, n_runs, cfg, seed0,
                                 pos_mode="nominal", landmark_sigma=0.5)
    print("方案S  nominal(名义坐标, v2基线)    p90 = %5.2f m  p95 = %5.2f m  (触发 %.1f)"
          % (np.percentile(errN, 90), np.percentile(errN, 95), hitsN))

    # effective_true：oracle 上限，σ 扫描
    for sg in (0.5, 0.3):
        errT, hitsT, diagT = eval_config(run_pf_v3, n_runs, cfg, seed0,
                                         pos_mode="effective_true",
                                         landmark_sigma=sg)
        print("方案S  effective_true(真实位置oracle,σ=%.2f) p90 = %5.2f m  p95 = %5.2f m  (触发 %.1f)"
              % (sg, np.percentile(errT, 90), np.percentile(errT, 95), hitsT))
    if diagT:
        print("  诊断 peak触发时真实位置 vs 名义坐标 偏差: 均值 %.2f m  中位 %.2f m  max %.2f m"
              % (np.mean(diagT), np.median(diagT), np.max(diagT)))

    # effective_static：静态有效位置，σ 扫描
    for sg in (0.5, 0.3):
        errS, hitsS, _ = eval_config(run_pf_v3, n_runs, cfg, seed0,
                                     pos_mode="effective_static",
                                     static_pos=static_pos, landmark_sigma=sg)
        print("方案S  effective_static(静态有效位置,σ=%.2f) p90 = %5.2f m  p95 = %5.2f m  (触发 %.1f)"
              % (sg, np.percentile(errS, 90), np.percentile(errS, 95), hitsS))

    print("-" * 84)
    print("静态有效位置（提取自 seed=1000 干净轨迹）：")
    for j, (n, s) in enumerate(zip(LANDMARKS, static_pos)):
        print("  地标%2d 名义(%.1f,%.1f) -> 有效(%.2f,%.2f)  偏移 %.2f m"
              % (j, n[0], n[1], s[0], s[1], np.hypot(n[0] - s[0], n[1] - s[1])))
    print("=" * 84)


if __name__ == "__main__":
    main()
