# -*- coding: utf-8 -*-
"""
语义地标 v4 —— LLM 推算误差容限实验。

v3 已证：有效位置（oracle 静态提取）σ=0.3 时 p90=1.14m，追平物理锚点 1.07m。
但 effective_static 的位置是 oracle 从真值轨迹提取的。本实验回答：
LLM 独立推算有效位置的误差有多大时，p90 仍追平物理锚点？

方法：oracle 静态有效位置 + 高斯噪声 N(0, σ_pos) 模拟"LLM 推算误差"，
      σ_pos 扫描，看 p90 退化曲线。每个 run 重新抽样 LLM 推算误差（MC 平均），
      模拟"不同建筑 / 不同 LLM 调用得到的推算偏差"。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import SensorCfg, run_particle_filter, gen_truth, simulate_steps
from semantic_landmark_v3 import run_pf_v3, extract_turn_exit_positions, eval_config


def eval_noisy_static(n_runs, cfg, seed0, sigma_pos):
    """oracle 静态有效位置 + N(0,σ_pos) 噪声 → 末端误差数组。"""
    static = extract_turn_exit_positions(seed=1000)
    end_errs = []
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        rng = np.random.default_rng(seed + 900001)
        noisy = static + rng.normal(0, sigma_pos, static.shape)
        (te, xe, ye), _, _ = run_pf_v3(seed, steps, cfg,
                                       pos_mode="effective_static",
                                       static_pos=noisy, landmark_sigma=0.3)
        tx_end = float(np.interp(te[-1], tr["t"], tr["x"]))
        ty_end = float(np.interp(te[-1], tr["t"], tr["y"]))
        end_errs.append(float(np.hypot(xe[-1] - tx_end, ye[-1] - ty_end)))
    return np.array(end_errs)


def main():
    cfg = SensorCfg()
    n_runs = 40
    seed0 = 1000

    print("=" * 84)
    print("语义地标 v4：LLM 推算误差容限（%d 次蒙特卡洛，σ_pos 扫描）" % n_runs)
    print("=" * 84)

    errC, _, _ = eval_config(
        lambda seed, steps, cfg, **kw: (
            run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                                anchor_spacing=15.0, corner_qr=True), 0, []),
        n_runs, cfg, seed0)
    print("方案C  物理锚点(二维码+15m)           p90 = %5.2f m  p95 = %5.2f m" %
          (np.percentile(errC, 90), np.percentile(errC, 95)))

    for sp in (0.0, 0.2, 0.3, 0.5, 0.7, 1.0):
        err = eval_noisy_static(n_runs, cfg, seed0, sp)
        print("方案S  effective_static + σ_pos=%.2f   p90 = %5.2f m  p95 = %5.2f m" %
              (sp, np.percentile(err, 90), np.percentile(err, 95)))

    print("-" * 84)
    print("注：p90 仍 ≤ 物理锚点 1.07m（或接近）的最大 σ_pos = LLM 推算误差容限。")
    print("=" * 84)


if __name__ == "__main__":
    main()
