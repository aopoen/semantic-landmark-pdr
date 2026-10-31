# -*- coding: utf-8 -*-
"""
语义地标 v2 —— 触发时机优化 + 区域建模（拆解两个因素的各自贡献）。

在 v1（点地标 + 转角第一步触发）基础上，拆出两个改进维度：
  trigger_mode:
    "first" = 转角第一步触发（v1 做法，粒子离转角中心最远，注入系统偏差）
    "peak"  = 转角峰值触发（|dpsi| 最大处 ≈ 转角中心，粒子离地标最近）
  landmark_type:
    "point"   = 点地标 + 各向同性 σ（v1 做法，名义坐标 vs 实际区域错配）
    "segment" = 线段区域地标（沿路径方向，粒子约束到线段，反映转角"区域"本质）

据此可拆出 4 个组合，分别验证两个因素各自的贡献：
  (first, point)   = v1 基线
  (peak,  point)   = 只改进触发时机
  (first, segment) = 只改进区域建模
  (peak,  segment) = 两者都改进
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from sim_core import (gen_truth, simulate_steps, SensorCfg, in_map,
                      systematic_resample, ENTRY, arc_length, mag_sigma_at)
from step_model import predict_steps_sl, get_model

# ---------------------------------------------------------------- 点地标（v1）
LANDMARKS = np.array([
    (58.5, 1.5), (58.5, 20.5), (30.0, 20.5), (30.0, 1.5),
    (4.0, 1.5), (10.0, 3.5), (45.0, 3.5),
], dtype=float)

# ---------------------------------------------------------------- 线段地标（v2 区域建模）
# 每行 (x0, y0, x1, y1)，沿路径方向的 ~2m 线段。
# 语义：转角/门不是"点"，是沿路径的一段区域。粒子约束到线段（沿方向自由，垂直方向严格）。
SEG_LANDMARKS = np.array([
    (58.5, 0.5, 58.5, 2.5),    # 右下转角（垂直，右走廊方向）
    (57.5, 20.5, 59.5, 20.5),  # 右上转角（水平，顶走廊方向）
    (30.0, 19.5, 30.0, 21.5),  # 左上转角（垂直，支走廊方向）
    (29.0, 1.5, 31.0, 1.5),    # 左下转角（水平，主走廊方向）
    (3.0, 1.5, 5.0, 1.5),      # 折返点（水平）
    (10.0, 3.0, 10.0, 4.0),    # 房间 101 门（垂直）
    (45.0, 3.0, 45.0, 4.0),    # 房间 102 门（垂直）
], dtype=float)


def point_segment_dist(px, py, x0, y0, x1, y1):
    """向量化点到线段距离（px, py 为 (N,) 数组）。"""
    dx = x1 - x0
    dy = y1 - y0
    L2 = dx * dx + dy * dy
    t = ((px - x0) * dx + (py - y0) * dy) / max(L2, 1e-9)
    t = np.clip(t, 0.0, 1.0)
    cx = x0 + t * dx
    cy = y0 + t * dy
    return np.hypot(px - cx, py - cy)


def _match(qx, qy, landmark_type):
    """匹配最近地标，返回索引。qx, qy 是标量（真实位置或估计位置）。"""
    if landmark_type == "point":
        d = np.hypot(LANDMARKS[:, 0] - qx, LANDMARKS[:, 1] - qy)
    else:  # segment
        d = np.array([point_segment_dist(np.array([qx]), np.array([qy]),
                                         x0, y0, x1, y1)[0]
                      for (x0, y0, x1, y1) in SEG_LANDMARKS])
    return int(np.argmin(d))


def run_pf_landmark(seed, steps, cfg, trigger_mode="first", landmark_type="point",
                    turn_thresh_deg=20.0, landmark_sigma=0.5,
                    use_map=True, use_mag=True, oracle_match=True):
    """统一粒子滤波：trigger_mode × landmark_type 四组合。"""
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
    turn_armed = True
    turn_active = False
    prev_abs_d = 0.0

    for k, s in enumerate(steps):
        abs_d = abs(s["dpsi"])

        # ---- 状态转移 ----
        ph = ph + s["dpsi"] + rng.normal(0, np.deg2rad(1.2), N)
        sl = sl_seq[k] * (1.0 + proc * rng.standard_normal(N))
        px = px + sl * np.cos(ph)
        py = py + sl * np.sin(ph)

        logw = np.zeros(N)

        # ---- 地图约束 ----
        if use_map:
            ok = in_map(px, py)
            logw += np.where(ok, 0.0, -60.0)

        # ---- 磁力计（同原方案B）----
        if use_mag:
            sig = mag_sigma_at(np.array([px.mean()]), np.array([py.mean()]),
                               cfg.mag_std_clean, cfg.mag_std_disturb)[0]
            m = s["mag"]
            d = np.arctan2(np.sin(m - ph), np.cos(m - ph))
            logw += -0.5 * (d / sig) ** 2

        # ---- 转角触发（first / peak 两种时机）----
        if trigger_mode == "first":
            # 进入转角第一步触发，转完重新武装
            trigger = abs_d > turn_thresh and turn_armed
            if trigger:
                turn_armed = False
            elif abs_d < turn_thresh * 0.5:
                turn_armed = True  # 转完，重新武装
        else:  # peak：转角出口步触发（粒子在转角出口，离地标比入口近）
            if abs_d > turn_thresh and turn_armed:
                turn_active = True
                turn_armed = False
                trigger = False
            elif turn_active and abs_d < turn_thresh * 0.5:
                trigger = True
                turn_active = False
                turn_armed = True
            else:
                trigger = False
        prev_abs_d = abs_d

        # ---- 语义地标观测（point / segment）----
        if trigger:
            if oracle_match:
                qx, qy = s["x_true"], s["y_true"]
            else:
                qx, qy = float(np.mean(px)), float(np.mean(py))
            j = _match(qx, qy, landmark_type)
            if landmark_type == "point":
                lx, ly = LANDMARKS[j]
                dx = px - lx
                dy = py - ly
                logw += -0.5 * (dx / landmark_sigma) ** 2 + -0.5 * (dy / landmark_sigma) ** 2
            else:  # segment
                x0, y0, x1, y1 = SEG_LANDMARKS[j]
                dist = point_segment_dist(px, py, x0, y0, x1, y1)
                logw += -0.5 * (dist / landmark_sigma) ** 2
            n_lm_hits += 1

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
    """各 run 末端误差数组（与 facts.py MAIN_P90_END 同口径）。"""
    end_errs = []
    n_hits = 0
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        (te, xe, ye), hits = run_fn(seed, steps, cfg, **kw)
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

    print("=" * 84)
    print("语义地标 v2：触发时机 × 区域建模 拆解（%d 次蒙特卡洛，末端 p90/p95）" % n_runs)
    print("=" * 84)

    # 方案C 物理锚点（基准）
    errC, _ = eval_config(
        lambda seed, steps, cfg, **kw: (
            run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                                anchor_spacing=15.0, corner_qr=True), 0),
        n_runs, cfg, seed0)
    print("方案C  物理锚点(二维码+15m)         p90 = %5.2f m  p95 = %5.2f m" %
          (np.percentile(errC, 90), np.percentile(errC, 95)))

    # 四个组合（σ=0.50m）
    combos = [
        ("first",  "point",   "v1 基线：第一步+点地标"),
        ("peak",   "point",   "只改进触发时机：峰值+点地标"),
        ("first",  "segment", "只改进区域建模：第一步+线段地标"),
        ("peak",   "segment", "两者都改：峰值+线段地标"),
    ]
    for trig, lmtype, label in combos:
        errS, hits = eval_config(run_pf_landmark, n_runs, cfg, seed0,
                                 trigger_mode=trig, landmark_type=lmtype,
                                 landmark_sigma=0.5, oracle_match=True)
        print("方案S  %-28s p90 = %5.2f m  p95 = %5.2f m  (触发 %.1f)" %
              (label, np.percentile(errS, 90), np.percentile(errS, 95), hits))

    print("-" * 84)
    print("注：峰值触发 + 线段地标，若 p90 逼近方案C，则说明两因素均有效。")
    print("=" * 84)


if __name__ == "__main__":
    main()
