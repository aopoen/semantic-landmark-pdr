# -*- coding: utf-8 -*-
"""
语义地标 v6 —— 有效位置地标 σ 扫描（补表 5）+ 粒子滤波单步耗时（补表 6）。
复现 v3 的 effective_static，扫描 σ ∈ {0.3, 0.5, 1.0, 2.0}，并测单 run 耗时。
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import SensorCfg, gen_truth, simulate_steps, run_particle_filter
from semantic_landmark_v3 import run_pf_v3, extract_turn_exit_positions, eval_config


def eval_static_sigma(n_runs, cfg, seed0, sigma):
    static = extract_turn_exit_positions(seed=1000)
    errs = []
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        (te, xe, ye), _, _ = run_pf_v3(seed, steps, cfg,
                                       pos_mode="effective_static",
                                       static_pos=static, landmark_sigma=sigma)
        tx_end = float(np.interp(te[-1], tr["t"], tr["x"]))
        ty_end = float(np.interp(te[-1], tr["t"], tr["y"]))
        errs.append(float(np.hypot(xe[-1] - tx_end, ye[-1] - ty_end)))
    return np.array(errs)


def main():
    cfg = SensorCfg()
    n_runs = 40
    seed0 = 1000

    print("=" * 72)
    print("v6：有效位置地标 σ 扫描 + 粒子滤波耗时")
    print("=" * 72)

    for sg in (0.3, 0.5, 1.0, 2.0):
        err = eval_static_sigma(n_runs, cfg, seed0, sg)
        print("  有效位置地标 σ=%.2f  p90 = %5.2f m  p95 = %5.2f m"
              % (sg, np.percentile(err, 90), np.percentile(err, 95)))

    # 耗时：测 5 次单 run，取中位
    static = extract_turn_exit_positions(seed=1000)
    times = []
    for r in range(5):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        t0 = time.perf_counter()
        run_pf_v3(seed, steps, cfg, pos_mode="effective_static",
                  static_pos=static, landmark_sigma=0.3)
        times.append(time.perf_counter() - t0)
    per_run = float(np.median(times))
    n_steps = len(steps)
    print("-" * 72)
    print("  单次粒子滤波 run 耗时 %.2f s（%d 步，1500 粒子）" % (per_run, n_steps))
    print("  单步耗时 %.3f ms" % (per_run / max(n_steps, 1) * 1000))
    print("=" * 72)


if __name__ == "__main__":
    main()
