# 无基础设施室内定位中语义地标有效位置偏差 —— 复现代码

论文《无基础设施室内定位中语义地标有效位置偏差的量化与影响》的复现代码（仿真 + 真机数据）。

## 复现什么

本仓库复现论文的核心实验（表 2–4 与 4.9 节），即"语义地标有效位置偏差"的量化：

- 名义位置 vs 有效位置地标对定位精度的影响（p90：2.10 m → 1.49 m → 1.14 m）
- 有效位置误差敏感度（追平物理锚点需容限 0.2 m）
- LLM 直接推算 / 几何解析 / 在线后验估计三条获取路径的检验（1.79 m / 0.76 m 诊断）

论文表 1（方案 0/A/B/C 三段式递进：483 / 22.2 / 5.3 / 1.07 m）来自基础 PDR 仿真工程，对应 `sim_core.py` 的 `run_speed_only` / `run_pure_dead_reckoning` / `run_particle_filter`；方案 C 的 p90 = 1.07 m 亦由 `semantic_landmark_v3.py` 作为基线打印。表 3（真机验证，多受试者多角度）来自 `real_data/` 下的 Phyphox 实测数据，处理脚本为 `process_real_data.py`。

## 环境

- Python 3.8+
- numpy（脚本已内置 `np.trapezoid = np.trapz` 兼容，适配 numpy 1.26.4 及以下）
- Pillow（仅 `make_figs.py` 绘图需要）

```bash
pip install numpy pillow
```

## 快速开始

核心结果（表 2，40 次蒙特卡洛，约 3–5 分钟）：

```bash
python semantic_landmark_v3.py
```

输出：方案 C（物理锚点基线）、名义位置地标、有效位置地标（oracle 与静态提取）的末端 p90 / p95。

## 脚本 → 论文章节映射

| 脚本 | 复现内容 | 论文章节 |
|------|---------|---------|
| `semantic_landmark_study.py` (v1) | 物理锚点 vs 语义地标基线（方案 B/C/S） | 4.3 前置分析 |
| `semantic_landmark_v2.py` | 触发时机 × 区域建模拆解（peak 触发 2.89→2.10 m） | 讨论 |
| `semantic_landmark_v3.py` | **核心**：名义/有效位置地标（2.10 / 1.49 / 1.14 / 1.07 m） | 表 2、4.3 |
| `semantic_landmark_v4.py` | LLM 推算误差容限 σ_pos 扫描（0→1.14 / 0.2→1.51 / 0.5→1.74 / 1.0→1.83 m） | 表 4、4.5 |
| `semantic_landmark_v5.py` | 转弯处粒子后验精度诊断（1.79 m / 0.76 m） | 4.6 |
| `semantic_landmark_v6.py` | 观测强度 σ 扫描（1.14/1.49/2.61/4.38 m）+ 单步耗时 | 4.9 |
| `step_model_train.py` | 训练步态回归模型（步长误差改善 53.9%） | 3.2 |
| `process_real_data.py` | 真机转弯几何（r/Δ，多受试者 × 多角度） | 表 3、4.4 |

## 目录结构

```
sim_core.py               仿真核心：平面图、真值轨迹、传感器、粒子滤波、蒙特卡洛
step_features.py          步态特征提取（仿真端与实测端共用）
step_model.py             步态回归 MLP（纯 numpy 手写，6-32-16-1）
step_model_train.py       训练脚本（跨速度训练 + 验证集早停）
plot_util.py              Pillow 绘图工具
semantic_landmark_*.py    语义地标实验 v1–v6
process_real_data.py      真机数据处理（转弯几何 → r/Δ）
make_figs.py              图 1（方法框架）与图 2（楼层平面）
real_data/                真机 Phyphox 原始数据（A/B/C/D 四转角）
out/step_model.npz        已训练的模型权重
```

## 说明

- 论文主体为仿真研究；真机验证（表 3）的 Phyphox 原始数据在 `real_data/` 下，运行 `process_real_data.py` 复现 r/Δ。磁力计数据未纳入（论文结论：窄走廊 + 好陀螺仪下磁力计是净负担）。
- 大角度转弯（130°/折返）的角度测量有约 25° 的系统偏小，其 r/Δ 为偏大估计；90° 直角不受影响。
- 模型权重 `out/step_model.npz` 已随仓库提供；若缺失，运行 `step_model_train.py` 重新训练。
- `make_figs.py` 生成图 1、图 2；图 3（转角几何）为单独绘制的示意图。
