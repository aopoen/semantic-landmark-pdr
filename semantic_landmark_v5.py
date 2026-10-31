# -*- coding: utf-8 -*-
"""
语义地标 v5 —— 诊断：转弯处粒子后验的精度（路径3 可行性前提）。

路径3 假设：粒子滤波在转弯处（地图几何约束）会天然收窄，后验均值可作为
"有效位置"的在线估计，从而精化 LLM 的粗先验（不必让 LLM 直接算精确坐标）。

本诊断量化：无地标观测时（仅地图约束 + 磁力计），转弯触发时刻粒子后验
均值 vs 真值 / vs oracle 有效位置 的误差，以及后验离散程度。
若后验 vs 有效位置 ≤ 0.3m，说明粒子天然收窄，在线精化可行。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import (SensorCfg, gen_truth, simulate_steps, in_map,
                      systematic_resample, ENTRY, mag_sigma_at)
from step_model import predict_steps_sl, get_model
from semantic_landmark_v3 import extract_turn_exit_positions


def run_pf_diag(seed, steps, cfg, static_pos):
    """粒子滤波（地图+磁，无地标观测），诊断转弯触发时刻后验 vs 有效位置。"""
    rng = np.random.default_rng(seed + 555001)
    N = cfg.n_particles
    turn_thresh = np.deg2rad(20.0)

    px = np.full(N, ENTRY[0]) + rng.normal(0, 0.15, N)
    py = np.full(N, ENTRY[1]) + rng.normal(0, 0.15, N)
    ph = np.full(N, 0.0) + rng.normal(0, np.deg2rad(3.0), N)

    sl_seq = predict_steps_sl(steps)
    proc = cfg.step_sl_proc_std
    if proc is None:
        proc = float(get_model()[1].get("proc_std", 0.03))

    turn_armed, turn_active = True, False
    diag = []  # (后验均值 vs 真值, 后验均值 vs 有效位置, 后验离散std)

    for k, s in enumerate(steps):
        abs_d = abs(s["dpsi"])

        ph = ph + s["dpsi"] + rng.normal(0, np.deg2rad(1.2), N)
        sl = sl_seq[k] * (1.0 + proc * rng.standard_normal(N))
        px = px + sl * np.cos(ph)
        py = py + sl * np.sin(ph)

        logw = np.zeros(N)
        ok = in_map(px, py)
        logw += np.where(ok, 0.0, -60.0)
        sig = mag_sigma_at(np.array([px.mean()]), np.array([py.mean()]),
                           cfg.mag_std_clean, cfg.mag_std_disturb)[0]
        m = s["mag"]
        d = np.arctan2(np.sin(m - ph), np.cos(m - ph))
        logw += -0.5 * (d / sig) ** 2

        logw -= logw.max()
        w = np.exp(logw)
        sw = w.sum()
        w = w / sw if (np.isfinite(sw) and sw > 0) else np.ones(N) / N

        ess = 1.0 / np.sum(w ** 2)
        if ess < 0.5 * N:
            idx = systematic_resample(w, rng)
            px, py, ph = px[idx], py[idx], ph[idx]
            px = px + rng.normal(0, 0.05, N)
            py = py + rng.normal(0, 0.05, N)
            ph = ph + rng.normal(0, np.deg2rad(0.8), N)

        # peak 触发（同 v2/v3）
        trigger = False
        if abs_d > turn_thresh and turn_armed:
            turn_active, turn_armed = True, False
        elif turn_active and abs_d < turn_thresh * 0.5:
            trigger, turn_active, turn_armed = True, False, True

        if trigger:
            mx, my = float(np.mean(px)), float(np.mean(py))
            # oracle 匹配：真值位置对应哪个地标
            j = int(np.argmin(np.hypot(static_pos[:, 0] - s["x_true"],
                                      static_pos[:, 1] - s["y_true"])))
            ex, ey = static_pos[j]
            diag.append((float(np.hypot(mx - s["x_true"], my - s["y_true"])),
                         float(np.hypot(mx - ex, my - ey)),
                         float(np.mean(np.hypot(px - mx, py - my)))))

    return diag


def main():
    cfg = SensorCfg()
    n_runs = 40
    seed0 = 1000
    static = extract_turn_exit_positions(seed=1000)

    all_diag = []
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        all_diag.extend(run_pf_diag(seed, steps, cfg, static))

    arr = np.array(all_diag)
    print("=" * 72)
    print("v5 诊断：转弯触发时刻粒子后验精度（无地标观测，%d 次 MC）" % n_runs)
    print("=" * 72)
    print("  后验均值 vs 真值位置:      均值 %.2f m  中位 %.2f m  p90 %.2f m"
          % (arr[:, 0].mean(), np.median(arr[:, 0]), np.percentile(arr[:, 0], 90)))
    print("  后验均值 vs oracle有效位置: 均值 %.2f m  中位 %.2f m  p90 %.2f m"
          % (arr[:, 1].mean(), np.median(arr[:, 1]), np.percentile(arr[:, 1], 90)))
    print("  后验离散(std):             均值 %.2f m  中位 %.2f m"
          % (arr[:, 2].mean(), np.median(arr[:, 2])))
    print("  触发样本数 %d" % len(arr))
    print("-" * 72)
    print("判读：后验 vs 有效位置 ≤ 0.3m → 粒子天然收窄，在线精化可行；")
    print("      否则 → 转弯处粒子发散，需地标观测 + 精化机制。")
    print("=" * 72)


if __name__ == "__main__":
    main()
