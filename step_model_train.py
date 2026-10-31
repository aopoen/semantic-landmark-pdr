# -*- coding: utf-8 -*-
"""
训练步态速度回归模型（纯 numpy MLP）。

流程
----
  1. 在**多个步速**上生成仿真轨迹，逐步收集 (步态特征, 真值步长)
  2. 按**轨迹**划分训练/验证（同一条轨迹的相邻步不跨集，避免泄漏）
  3. 训练 StepLengthMLP，验证集早停
  4. 在**未参与训练的速度** v_nom = 1.30（主实验工作点）上评估泛化
  5. 与"常数步长"基线对比，出图、写报告、存 out/step_model.npz

为什么要跨速度训练
------------------
真值步长与步速物理耦合（sim_core.sl_true_from_v）。如果训练速度只有一个，
模型只要记住一个常数就能通关——等于没学到东西，也没法用到实测的不同速度档上。

为什么测试速度刻意不含在训练里
------------------------------
主实验用 v_nom = 1.30。把 1.30 排除在训练速度之外，测出来的才是**对未见
工作点的泛化**，而不是对训练点的插值。

用法
----
  python step_model_train.py
"""
import os
import sys
import json
import numpy as np

from sim_core import SensorCfg, gen_truth, simulate_steps
from step_model import StepLengthMLP
from step_features import FEATURE_NAMES, N_FEAT

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ★ 训练速度刻意不含 1.30（主实验工作点）；1.75 用于测真外推
TRAIN_V = [0.80, 0.95, 1.10, 1.25, 1.45, 1.60]
TEST_V = [1.30]
EXTRA_V = [1.75]
SEEDS_PER_V = 8
# ★ 种子区段与主实验（1000+）、锚点扫描（2000+）、各专项（3000/5000/6000）
#   完全不相交，防止训练集与评估集数据泄漏。
TRAIN_SEED0 = 90000
TEST_SEED0 = 95000


def collect(v_list, cfg, seeds_per_v, seed_base):
    """在给定步速列表上生成轨迹，逐步收集 (特征, 真值步长, (速度序号, 种子序号, 速度))。"""
    X, y, g = [], [], []
    for vi, v in enumerate(v_list):
        for s in range(seeds_per_v):
            seed = seed_base + vi * 1000 + s
            tr = gen_truth(seed, v_nom=v)
            st = simulate_steps(seed, tr, cfg)
            for step in st:
                X.append(step["feat"])
                y.append(step["sl_true"])
                g.append((vi, s, float(v)))
    return np.array(X, float), np.array(y, float), g


def rmse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def make_fig(model, cfg, vv, pred_means):
    from plot_util import Chart, C_A, C_C, C_D
    vs = np.linspace(0.70, 1.80, 80)
    sl_truth = cfg.true_sl_mean * (vs / cfg.v_ref) ** cfg.sl_expo

    c = Chart(w=1280, h=780, pad=(112, 78, 52, 104),
              title="图12  步态回归模型：从加速度特征恢复步长–步速关系",
              sub="训练数据为仿真多速度轨迹；主实验工作点 v=1.30 未参与训练（held-out 速度）")
    c.set_x(0.70, 1.80, ticks=np.arange(0.7, 1.81, 0.1), fmt="{:.1f}", label="步速 (m/s)")
    c.set_y(0.40, 0.90, ticks=np.arange(0.4, 0.91, 0.1), fmt="{:.2f}", label="步长 (m)")
    c.grid_and_axes()
    c.line(vs, sl_truth, C_C, width=4)
    c.hline(0.74, color=C_A, dash=9, width=3)
    c.hline(0.70, color=(150, 156, 164), dash=9, width=2)
    for v, m in zip(vv, pred_means):
        c.marker(v, m, C_D, r=8)
    c.legend_items = [(C_C, "真值耦合律 sl = 0.70·(v/1.30)^0.55"),
                      (C_D, "模型预测（各速度上的均值）"),
                      (C_A, "常数 0.74 m（A 方案固定步长）"),
                      ((150, 156, 164), "常数 0.70 m（真值均值）")]
    c.legend(loc="upper left")
    return c.save(os.path.join(OUT, "fig12_step_model.png"))


def main():
    cfg = SensorCfg()
    L = []
    A = L.append
    A("=" * 84)
    A("步态速度回归模型 —— 训练与泛化评估")
    A("=" * 84)
    A("模型结构：纯 numpy 手写 MLP   %d → 32(ReLU) → 16(ReLU) → 1(线性)" % N_FEAT)
    A("训练目标：log(步长) 标准化后做 MSE（在 log 空间里等价于优化相对误差）")
    A("特征：%s" % ", ".join(FEATURE_NAMES))
    A("")
    A("训练步速：%s" % TRAIN_V)
    A("泛化测试：%s（主实验工作点，未参与训练）" % TEST_V)
    A("外推测试：%s（超出训练范围）" % EXTRA_V)
    A("")

    # ---------------- 1. 造数据 ----------------
    A("[1/5] 生成训练数据 ...")
    Xall, yall, g = collect(TRAIN_V, cfg, SEEDS_PER_V, TRAIN_SEED0)
    va = np.array([gi[1] >= SEEDS_PER_V - 2 for gi in g])
    Xtr, ytr = Xall[~va], yall[~va]
    Xva, yva = Xall[va], yall[va]
    A("  样本：训练 %d / 验证 %d（按轨迹划分，同轨迹不跨集）" % (len(Xtr), len(Xva)))
    A("  真值步长范围 %.3f – %.3f m，均值 %.3f m" % (yall.min(), yall.max(), yall.mean()))
    A("")

    # ---------------- 2. 训练 ----------------
    A("[2/5] 训练（Adam，验证集早停）...")
    model = StepLengthMLP(seed=0)
    hist = model.fit(Xtr, ytr, Xva, yva, epochs=400, lr=3e-3, l2=1e-5,
                     batch=256, patience=40, seed=0, verbose=True)
    if hist:
        A("  训练结束，验证 MSE(log 空间) = %.6f" % hist[-1])
    A("")

    # ---------------- 3. 泛化评估 ----------------
    A("[3/5] 泛化评估（测试集用独立种子区段，与训练不相交）...")
    rows = []
    for label, vlist in [("训练速度（内插）", TRAIN_V),
                         ("未见速度 1.30（主实验工作点）", TEST_V),
                         ("外推 1.75（超出训练范围）", EXTRA_V)]:
        Xe, ye, _ = collect(vlist, cfg, SEEDS_PER_V, TEST_SEED0)
        pred = model.predict_sl(Xe)
        e = pred - ye
        rows.append(dict(label=label, n=len(ye),
                         rmse=rmse(pred, ye),
                         mae=float(np.mean(np.abs(e))),
                         bias=float(np.mean(e)),
                         rel_pct=float(np.mean(np.abs(e)) / np.mean(ye) * 100),
                         true_mean=float(np.mean(ye)),
                         pred_mean=float(np.mean(pred))))
    A("%-34s%8s%10s%10s%11s%10s" % ("测试集", "N", "RMSE(m)", "MAE(m)", "系统偏差(m)", "相对%"))
    A("-" * 86)
    for r in rows:
        A("%-34s%8d%10.4f%10.4f%+11.4f%9.2f%%"
          % (r["label"], r["n"], r["rmse"], r["mae"], r["bias"], r["rel_pct"]))
    A("")

    # ---------------- 4. 常数基线对照 ----------------
    A("[4/5] 与常数步长基线对比（同一测试集，v=1.30）...")
    Xe, ye, _ = collect(TEST_V, cfg, SEEDS_PER_V, TEST_SEED0)
    pred = model.predict_sl(Xe)
    model_rmse = rmse(pred, ye)
    model_rel = float(np.mean(np.abs(pred - ye)) / np.mean(ye) * 100)
    base = []
    for name, c0 in [("常数 0.70 m（真值均值）", 0.70),
                     ("常数 0.74 m（A 方案固定步长）", 0.74)]:
        p = np.full_like(ye, c0)
        base.append((name, rmse(p, ye), float(np.mean(np.abs(p - ye)) / np.mean(ye) * 100)))
    A("%-34s%12s%12s" % ("方法（同一测试集，v=1.30）", "RMSE(m)", "相对%"))
    A("-" * 62)
    A("%-34s%12.4f%11.2f%%" % ("本模型（步态回归）", model_rmse, model_rel))
    for name, r, rel in base:
        A("%-34s%12.4f%11.2f%%" % (name, r, rel))
    imp = (base[0][1] - model_rmse) / base[0][1] * 100
    A("")
    A("  相对「常数 0.70 m」基线，RMSE 改善 %.1f%%" % imp)
    A("")

    # ---------------- 5. 保存 ----------------
    meta = dict(
        feature_names=list(FEATURE_NAMES), n_feat=int(N_FEAT),
        train_v=TRAIN_V, test_v=TEST_V, extra_v=EXTRA_V,
        seeds=dict(train_base=TRAIN_SEED0, test_base=TEST_SEED0, per_v=SEEDS_PER_V),
        arch="MLP %d-32-16-1 ReLU, Adam, target=log(sl)" % N_FEAT,
        test_v130_rmse_m=round(model_rmse, 5),
        test_v130_rel_pct=round(model_rel, 3),
        test_v130_bias_m=round(rows[1]["bias"], 5),
        # ★ 粒子滤波里逐粒子步长过程噪声的取值依据：模型在主实验工作点上的
        #   相对 RMSE。写进元数据，仿真端直接读，避免手填一个和模型不符的数。
        proc_std=round(model_rmse / max(float(np.mean(ye)), 1e-6), 5),
        constant_baseline_rmse_m=round(base[0][1], 5),
        rmse_improve_pct=round(imp, 2),
        trained="2026-10-07",
        note="训练数据全部来自仿真；真机数据用于域校准，见 field_step_calib.py。",
    )
    path = model.save(os.path.join(OUT, "step_model.npz"), meta)
    A("[5/5] 模型已保存 -> %s" % path)
    A("      测试速度上的 RMSE = %.4f m（相对 %.2f%%），系统偏差 %+.4f m"
      % (model_rmse, model_rel, rows[1]["bias"]))
    A("")

    # ---------------- 图 ----------------
    vv = sorted(set(TRAIN_V + TEST_V + EXTRA_V))
    Xg, yg, gg = collect(vv, cfg, 6, TEST_SEED0)
    pg = model.predict_sl(Xg)
    v_arr = np.array([gi[2] for gi in gg])
    means = [float(np.mean(pg[v_arr == v])) for v in vv]
    fig = make_fig(model, cfg, vv, means)
    A("图 -> %s" % fig)

    txt = "\n".join(L)
    with open(os.path.join(OUT, "step_model_train.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    try:
        print(txt)
    except UnicodeEncodeError:
        print(txt.encode("utf-8", "replace").decode("utf-8", "replace"))
    print("\n报告 -> %s" % os.path.join(OUT, "step_model_train.txt"))


if __name__ == "__main__":
    main()
