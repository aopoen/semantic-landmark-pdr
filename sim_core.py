# -*- coding: utf-8 -*-
"""
室内定位定量仿真 —— 核心引擎
================================================
三种方案对比：
  A  纯航位推算        : 陀螺仪积分航向 + 固定步长模型，无地图、无锚点
  B  AI-PDR + 地图约束 : 粒子滤波；磁力计航向 + 学习型速度回归 + 不可穿墙约束
  C  B + 稀疏绝对锚点  : 二维码(0.3m) + 地磁指纹地标(2.5m)

关键设计原则（保证结论可信）：
  * 三种方法共享同一条真值轨迹、同一组 IMU 噪声、同一个入口已知点
  * 误差评价按"时间对齐"，不是按步数对齐（避免步数不同造成的偏袒）
  * 全部结果做蒙特卡洛，报 p50 / p90，不报单车最好成绩

步态速度回归（2026-10-07 起）
--------------------------------
B/C 方案的步长不再是一条"常数+噪声"公式，而是由 out/step_model.npz 里的
神经网络（纯 numpy 手写 MLP）逐步推理给出。模型输入是每步的步态特征
（步频、加速度起伏等，见 step_features.py），训练脚本是 step_model_train.py。
真值步长与步速之间物理耦合（次线性幂律），这是"加速度特征能反推步长"的前提。
"""
import numpy as np

# numpy < 2.0 只有 np.trapz，没有 np.trapezoid；统一补齐，保证旧环境可跑。
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from step_features import bandpass_acc, step_features
from step_model import get_model as _get_step_model, predict_steps_sl

# ============================================================
# 1. 平面图（轴对齐矩形并集 = 可行走区域，其余即为墙体）
# ============================================================
C1 = (0.0, 0.0, 60.0, 3.0)      # 主走廊   x:0..60,  y:0..3
C2 = (28.0, 3.0, 32.0, 22.0)    # 支走廊   x:28..32, y:3..22
C3 = (28.0, 19.0, 60.0, 22.0)   # 顶部走廊 x:28..60, y:19..22
C4 = (57.0, 3.0, 60.0, 22.0)    # 右侧走廊 x:57..60, y:3..22
R1 = (5.0, 4.0, 15.0, 11.5)     # 房间 101
D1 = (9.0, 3.0, 11.0, 4.0)      # 房间 101 的门
R2 = (40.0, 4.0, 50.0, 11.5)    # 房间 102
D2 = (44.0, 3.0, 46.0, 4.0)     # 房间 102 的门
RECTS = [C1, C2, C3, C4, R1, D1, R2, D2]

# 入口（闸机/大门）—— 三种方法共享的已知起点
ENTRY = (0.5, 1.5)

# 真值行走路线：绕楼环形巡逻 N 圈 + 最后进房间
#  * 环形（不原路折返）-> 误差单调累积
#  * 多圈 + 约 6 分钟 -> 才能回答"能撑多久"
#  * 折返点放在 x=4.0，避免在入口做掉头时把轨迹甩出墙外
N_LAPS = 3
TURNAROUND = (4.0, 1.5)
_LOOP = [(58.5, 1.5), (58.5, 20.5), (30.0, 20.5), (30.0, 1.5), TURNAROUND]
WAYPOINTS = [ENTRY]
for _ in range(N_LAPS):
    WAYPOINTS += _LOOP
WAYPOINTS += [(45.0, 1.5), (45.0, 7.5)]


def in_map(x, y):
    """向量化：判断点是否在可行走区域内"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.zeros(x.shape, dtype=bool)
    for (x0, y0, x1, y1) in RECTS:
        m |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return m


def out_of_map_penalty(x, y):
    """返回点到可行走区域的距离（在外面为正）——用于轨迹质量评估"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = np.full(x.shape, np.inf)
    for (x0, y0, x1, y1) in RECTS:
        dx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
        best = np.minimum(best, np.hypot(dx, dy))
    return best


# ============================================================
# 2. 真值轨迹（独轮车模型 + 转向角速度限幅 = 真实转弯半径）
# ============================================================
def gen_truth(seed, dt=0.02, v_nom=1.30, max_turn_rate=np.deg2rad(120.0)):
    rng = np.random.default_rng(seed)
    x, y = WAYPOINTS[0]
    heading = np.arctan2(WAYPOINTS[1][1] - y, WAYPOINTS[1][0] - x)
    wp = 1
    T, X, Y, H, V = [], [], [], [], []
    t = 0.0
    for _ in range(200000):
        if wp >= len(WAYPOINTS):
            break
        # 自然步速波动（±6%，约 0.15 Hz）
        v = v_nom * (1.0 + 0.06 * np.sin(2 * np.pi * 0.15 * t) + 0.01 * rng.standard_normal())
        # 平滑转向
        tx, ty = WAYPOINTS[wp]
        dx, dy = tx - x, ty - y
        if np.hypot(dx, dy) < 0.25:
            wp += 1
            continue
        desired = np.arctan2(dy, dx)
        err = np.arctan2(np.sin(desired - heading), np.cos(desired - heading))
        omega = np.clip(err / max(dt, 1e-9), -max_turn_rate, max_turn_rate)
        omega = np.clip(err * 6.0, -max_turn_rate, max_turn_rate)   # 比例控制
        heading += omega * dt
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        T.append(t); X.append(x); Y.append(y); H.append(heading); V.append(v)
        t += dt
    return dict(t=np.array(T), x=np.array(X), y=np.array(Y),
                h=np.array(H), v=np.array(V))


def arc_length(tr):
    d = np.hypot(np.diff(tr["x"]), np.diff(tr["y"]))
    return np.concatenate([[0.0], np.cumsum(d)])


# ============================================================
# 3. 传感器 / 步态仿真
# ============================================================
class SensorCfg:
    # --- 陀螺仪 ---
    gyro_bias_std   = np.deg2rad(0.15)   # 逐次运行的常值零偏 -> 约 13.5°/90s
    gyro_noise_std  = np.deg2rad(0.30)   # 白噪声（单步积分后大幅衰减）
    gyro_scale_err  = 0.02               # 2% 标度因数误差（转弯时才是致命项）
    # --- 步长 ---
    true_sl_mean    = 0.70               # 真值平均步长 (m)
    true_sl_var     = 0.05               # 真值步长逐次波动 5%
    fixed_sl        = 0.74               # A 方案假设的固定步长 -> +5.7% 系统性偏差
    # --- A 方案（纯推算）无法观测的误差 ---
    A_sl_random     = 0.08               # 固定步长模型的随机误差
    # --- B/C 方案的步态速度回归（2026-10-07 起为真实模型）---
    # ★ AI_sl_bias / AI_sl_random 已退役：它们曾经是一条"假装有 AI"的常数+噪声。
    #   现在步长由 out/step_model.npz 推理给出。字段保留仅为不破坏旧脚本 import。
    AI_sl_bias      = 0.02               # [deprecated，不再被任何代码读取]
    AI_sl_random    = 0.035              # [deprecated，不再被任何代码读取]

    # 真值步长与步速的物理耦合：sl = true_sl_mean * (v/v_ref)^sl_expo
    #   v_ref=1.30 / true_sl_mean=0.70 保持主实验工作点不变（隐含步频仍为 1.857 Hz）
    #   sl_expo=0.55（次线性增长）在 v≈0.70 m/s 处给出步长≈0.498 m、步频≈1.41 Hz，
    #   与 2026-10-05 实测慢走 0.513 m @ 1.27–1.46 Hz 吻合。
    v_ref           = 1.30
    sl_expo         = 0.55

    # 加速度计（步态回归的可观测量来源）：垂直起伏幅值随步速增大
    acc_amp_ref     = 2.60               # v_ref 处的垂直起伏幅值 (m/s^2)
    acc_amp_expo    = 1.10               # 幅值随速度的幂次
    acc_noise_std   = 0.15               # 加速度计白噪声 (m/s^2)
    acc_xy_ratio    = (0.50, 0.35)       # x / y 轴起伏相对 z 轴的比例
    grav            = 9.81               # 重力加速度 (m/s^2)

    # 模型残差：粒子滤波里逐粒子的步长过程噪声。
    #   None = 自动取自模型元数据（out/step_model.npz 记录的相对 RMSE）；
    #   专项脚本可以覆写成具体数值做敏感性分析（见 bottleneck_study.py）。
    step_sl_proc_std = None
    # --- 磁力计（航向）---
    # 说明：混凝土+钢筋结构内，磁航向的实测精度普遍比室外差得多，
    #       开阔处 12° RMS、钢结构附近 45° RMS 是偏乐观但仍合理的取值
    mag_std_clean   = np.deg2rad(12.0)   # 开阔走廊的随机噪声
    mag_std_disturb = np.deg2rad(45.0)   # 钢门/电梯/配电间附近
    # ★ 关键：室内磁偏差的主体不是白噪声，而是钢筋结构造成的
    #   "沿路径缓慢漂移"的空间相关误差。不建模这一项，B 方案会被严重高估。
    mag_bias_std    = np.deg2rad(15.0)   # 缓慢漂移磁偏差的幅度
    mag_corr_len    = 15.0               # 空间相关长度 (m)
    # --- 锚点 ---
    qr_std          = 0.30               # 二维码 / 闸机
    mag_lm_std      = 2.50               # 地磁指纹地标
    # --- 3D 打印磁标记（铁粉/钕铁硼线材）---
    # 物理机制：标记处由"已知几何的人工磁场"主导，因此
    #   (a) 航向可用标记几何直接解算，不受钢筋结构的空间相关偏差污染；
    #   (b) 顺带得到一个比自然磁指纹精确得多的位置修正；
    #   (c) 经过标记时可以估出"当地环境磁偏差"，并沿路径按相关长度衰减地补偿。
    marker_influence   = 1.5             # 影响半径 (m)（旧的二值模型用）
    marker_heading_std = np.deg2rad(4.0) # 影响区内的航向精度
    marker_pos_std     = 0.40            # 标记给出的位置修正精度 (m)
    # --- 3D 打印磁标记：真实磁场模型 ---
    # ★ 标记是磁偶极子，场按 1/r^3 衰减，不存在"干净的边界"：
    #   标记场 >> 环境偏差 -> 可解算航向；
    #   标记场 ≈ 环境偏差 -> 两者叠加，且标记场的建模误差会额外污染航向；
    #   标记场 << 环境偏差 -> 无影响。
    marker_A1        = 100.0             # 1 m 处的等效罗差幅度 (deg)，决定场有多强
    marker_model_err = 0.15              # 标记场建模的相对误差 -> 过渡区的残差来源
    marker_floor     = np.deg2rad(4.0)   # 标记主导、航向可解算时的误差地板
    marker_ratio_hi  = 3.0               # A_marker / A_ambient 超过此值 -> 标记主导
    marker_ratio_lo  = 0.30              # 低于此值 -> 标记不可检测
    # ★ 标记贴在墙上，不在人行进路径的正上方。横向偏置决定了"最近能靠多近"，
    #   是决定标记能不能被检测到、以及能不能主导航向的关键几何量。
    marker_standoff  = 1.00              # 标记到行进路径的横向距离 (m)
    # --- 粒子滤波 ---
    n_particles     = 1500


def disturbed_zones():
    """磁场受扰区（钢门、电梯井、配电间附近），格式 (x0, x1, y0, y1)"""
    return [
        (7.5, 12.5, 1.5, 6.0),      # 房间 101 钢门
        (42.5, 47.5, 1.5, 6.0),     # 房间 102 钢门
        (27.0, 33.0, 19.0, 22.5),   # 支走廊/顶走廊交叉口（设备间）
        (56.0, 60.0, 18.0, 22.0),   # 右走廊/顶走廊交叉口（配电间）
    ]


def mag_sigma_at(x, y, base_clean, base_dist):
    s = np.full(np.shape(x), base_clean, dtype=float)
    for (x0, x1, y0, y1) in disturbed_zones():
        m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        s = np.where(m, base_dist, s)
    return s


# ============================================================
# 3b. 步态：真值步长-步速耦合 + 加速度计仿真（2026-10-07 新增）
# ============================================================
def sl_true_from_v(v, cfg):
    """真值步长随步速的物理耦合：次线性幂律 sl = true_sl_mean * (v/v_ref)^sl_expo。

    在 v = 1.30 m/s（巡逻档）处给出 true_sl_mean = 0.70 m，保持主实验工作点不变；
    在 v ≈ 0.70 m/s 处给出约 0.50 m，与 2026-10-05 实测慢走档吻合。
    物理上：步速 = 步长 × 步频，两者都随步速上升，但步长上升得更慢。
    """
    return cfg.true_sl_mean * (v / cfg.v_ref) ** cfg.sl_expo


def simulate_accel(t, v, sl_seq, cfg, rng):
    """生成与真值同一时间栅格的加速度模长（含重力）。

    ★ 相位用"连续积分"而不是每步重置：步与步之间的起伏波形才能自然衔接。
      每步重置会在步边界制造人造跳变，被峰值特征当成假信号捕捉。

    ★ 起伏幅值只与步速 v 耦合——这正是"加速度特征能反推步速/步长"的物理来源。
      没有这层耦合，模型什么也学不到（旧版真值步长是纯随机的，与任何观测量无关）。
    """
    dt = float(np.median(np.diff(t)))
    f_step = v / np.maximum(sl_seq, 1e-6)             # 瞬时步频 (Hz)
    phi = 2.0 * np.pi * np.cumsum(f_step) * dt         # 连续相位
    amp = cfg.acc_amp_ref * (v / cfg.v_ref) ** cfg.acc_amp_expo
    n = len(t)
    rx, ry = cfg.acc_xy_ratio
    ax = rx * amp * np.sin(phi + 0.7) + rng.normal(0, cfg.acc_noise_std, n)
    ay = ry * amp * np.sin(phi + 1.9) + rng.normal(0, cfg.acc_noise_std, n)
    az = cfg.grav + amp * np.sin(phi) + rng.normal(0, cfg.acc_noise_std, n)
    return np.sqrt(ax ** 2 + ay ** 2 + az ** 2)


def simulate_steps(seed, tr, cfg=SensorCfg(), markers=None):
    """从真值轨迹生成"可观测的"步级测量量

    markers=None                        : 自然磁环境（磁偏差全量污染航向）
    markers=dict(spacing=S, debias=B)   : 沿途每 S 米有一个 3D 打印磁标记
        - 影响半径内：标记场主导 -> 航向无环境偏差、精度大幅提高
        - debias=True：经过标记时估出当地磁偏差，并按相关长度衰减地补偿后续读数
    """
    rng = np.random.default_rng(seed + 90001)
    t, x, y, h, v = tr["t"], tr["x"], tr["y"], tr["h"], tr["v"]
    omega_true = np.gradient(np.unwrap(h), t)

    # 陀螺仪读数
    gyro = omega_true * (1.0 + cfg.gyro_scale_err) + rng.normal(0, cfg.gyro_bias_std) \
           + rng.normal(0, cfg.gyro_noise_std, size=t.shape)

    # 磁力计读数 = 真值航向 + 空间相关磁偏差 + 位置相关随机噪声
    s_arc = arc_length(tr)
    grid = np.arange(0.0, max(s_arc[-1], 1.0), 0.5)
    white = rng.standard_normal(len(grid))
    ksig = cfg.mag_corr_len / 2.0
    half = max(int(4 * ksig / 0.5), 1)
    kern = np.exp(-0.5 * ((np.arange(-half, half + 1) * 0.5) / ksig) ** 2)
    kern /= kern.sum()
    bias_grid = np.convolve(white, kern, mode="same")
    bias_grid = bias_grid / (bias_grid.std() + 1e-12) * cfg.mag_bias_std
    mag_bias = np.interp(s_arc, grid, bias_grid)

    sig = mag_sigma_at(x, y, cfg.mag_std_clean, cfg.mag_std_disturb)

    if markers is None:
        mag = h + mag_bias + rng.normal(0, 1.0, size=t.shape) * sig
        detect = np.ones(t.shape, dtype=bool)
    elif markers.get("model") == "dipole":
        S = float(markers["spacing"])
        delta = np.mod(s_arc, S)                       # 距上一个标记的沿路径距离
        along = np.minimum(delta, S - delta)           # 沿路径到最近标记的距离
        # 标记贴墙 -> 实际距离 = sqrt(横向偏置^2 + 沿路径距离^2)
        r = np.maximum(np.sqrt(cfg.marker_standoff ** 2 + along ** 2), 0.25)
        A_mk = cfg.marker_A1 * (1.0 / r) ** 3          # 等效罗差幅度 (deg)，偶极子 1/r^3
        # 当地环境扰动水平：偏差与噪声共同决定"背景有多吵"
        amb = np.sqrt(np.rad2deg(mag_bias) ** 2 + np.rad2deg(sig) ** 2)
        ratio = A_mk / np.maximum(amb, 1.0)
        dom = ratio > cfg.marker_ratio_hi              # 标记主导 -> 航向可解算
        detect = ratio > cfg.marker_ratio_lo           # 可检测 -> 才能给位置修正
        # 残差 = 环境偏差 + 标记场建模误差（幅度随标记场强增长）
        mk_err = cfg.marker_model_err * np.deg2rad(A_mk) * rng.standard_normal(t.shape)
        resid = mag_bias + mk_err
        if markers.get("debias"):
            resid = resid - mag_bias * np.exp(-delta / cfg.mag_corr_len)
        bias_eff = np.where(dom, rng.standard_normal(t.shape) * cfg.marker_floor, resid)
        mag = h + bias_eff + rng.normal(0, 1.0, size=t.shape) * sig
    else:
        S = float(markers["spacing"])
        infl = cfg.marker_influence
        # 距"上一个标记"的沿路径距离
        delta = np.mod(s_arc, S)
        near = np.minimum(delta, S - delta) < infl
        detect = np.ones(t.shape, dtype=bool)
        if markers.get("debias"):
            # 刚过标记时偏差被完全估出，随距离按相关长度衰减
            resid = mag_bias * (1.0 - np.exp(-delta / cfg.mag_corr_len))
        else:
            resid = mag_bias
        if markers.get("heading", True):
            bias_eff = np.where(near, 0.0, resid)
            noise_eff = np.where(near, cfg.marker_heading_std, sig)
        else:
            # 消融：标记只给位置修正，不改善航向
            bias_eff, noise_eff = resid, sig
        mag = h + bias_eff + rng.normal(0, 1.0, size=t.shape) * noise_eff

    # 真值步长序列：与步速物理耦合 + 逐步波动
    # ★ 这里仍然只消费一次 rng.standard_normal(t.shape)，与旧版完全一致——
    #   刻意不打乱主随机数流，使陀螺仪/磁力计读数不受本次改动影响。
    sl_base = sl_true_from_v(v, cfg)
    sl_true = sl_base * (1.0 + cfg.true_sl_var * rng.standard_normal(t.shape))

    # 加速度计：用**独立**的随机流，进一步避免扰动上面的抽样序列
    rng_acc = np.random.default_rng(seed + 770001)
    acc_mag = simulate_accel(t, v, sl_true, cfg, rng_acc)
    fs_sim = 1.0 / max(float(np.median(np.diff(t))), 1e-9)
    acc_bp = bandpass_acc(acc_mag, fs_sim)
    gyro_mag = np.abs(gyro)

    # 步分割：步间隔 = 步长 / 速度
    steps = []
    tt = t[0]
    k = 0
    while tt < t[-1]:
        i = int(np.searchsorted(t, tt))
        if i >= len(t) - 1:
            break
        vv = v[i]
        s = sl_true[i]
        dt_step = s / max(vv, 0.3)
        h_next = tt + dt_step
        j = min(int(np.searchsorted(t, h_next)), len(t) - 1)
        # 该步内陀螺仪积分的航向变化
        dpsi = np.trapezoid(gyro[i:j + 1], t[i:j + 1]) if j > i else 0.0
        # ★ 时间戳必须与位置对齐：滤波器处理完第 k 步后，状态对应的是
        #   "该步结束时刻"的位置，所以时间戳要用 h_next（步末），
        #   而不是 tt（步首）。用 tt 会给所有按时间评价的误差注入
        #   约一个步长（~0.7 m）的恒定滞后，压平各配置之间的真实差异。
        # 步态特征（回归模型的输入）：在 i..j 窗口上算。仿真端与实测端共用
        # 同一个 step_features.step_features()，口径必须一致。
        feat = step_features(t, acc_mag, acc_bp, gyro_mag, i, j)
        steps.append(dict(t=h_next, i=i, j=j, dpsi=dpsi,
                          mag=mag[j], x_true=x[j], y_true=y[j], h_true=h[j],
                          sl_true=s, detectable=bool(detect[j]),
                          feat=feat, dur=float(h_next - tt)))
        tt = h_next
        k += 1
        if k > 20000:
            break
    return steps


# ============================================================
# 4. 粒子滤波（地图约束 + 可选航向/位置量测）
# ============================================================
def systematic_resample(w, rng):
    n = len(w)
    pos = (np.arange(n) + rng.random()) / n
    cum = np.cumsum(w)
    cum[-1] = 1.0
    return np.clip(np.searchsorted(cum, pos), 0, n - 1)


def run_particle_filter(seed, steps, cfg, use_map=True, use_mag=True,
                        anchor_spacing=None, anchor_sigma=None,
                        corner_qr=False, false_anchor_prob=0.0,
                        mag_sigma_scale=1.0, mag_bias_adapt=0.0,
                        mag_bias_rw=0.0):
    """返回逐步估计 (t, x, y)

    corner_qr      : 是否布置"现实版"二维码锚点（入口+各转角/岔路口，σ=0.30 m）
    anchor_spacing : 沿路径每 S 米布置一个均匀锚点（按物理位置去重）
    anchor_sigma   : 均匀锚点的定位精度（m），None 则用 cfg.mag_lm_std
    """
    rng = np.random.default_rng(seed + 555001)
    N = cfg.n_particles
    sig_anchor = cfg.mag_lm_std if anchor_sigma is None else anchor_sigma

    # ---- 锚点定义 ----
    # 二维码：贴在入口与各转角/岔路口（现实中就是指示牌的位置）
    qr = [(0.5, 1.5), (58.5, 1.5), (58.5, 20.5), (30.0, 20.5),
          (30.0, 1.5), (45.0, 3.5)] if corner_qr else []

    # 均匀锚点：沿路径每 S 米（真实里程）获得一次绝对定位修正。
    # ★ 模型要点：量测值是"我此刻的真实位置 + 精度 σ 的噪声"，
    #   而不是"锚点的名义坐标"。后者的触发半径会强行注入约 2 m 误差，
    #   给所有配置加一个恒定地板，从而掩盖间距与精度的真实差异。
    #   物理上对应：具备位姿估计能力的视觉标记会给出自身位置；
    #   地磁指纹匹配的是"当前所在位置"。
    fix_steps = {}
    if anchor_spacing is not None:
        acc, prev_p, next_at = 0.0, None, anchor_spacing
        for k, s in enumerate(steps):
            p = (s["x_true"], s["y_true"])
            if prev_p is not None:
                acc += float(np.hypot(p[0] - prev_p[0], p[1] - prev_p[1]))
            prev_p = p
            if acc >= next_at:
                fix_steps[k] = p
                next_at += anchor_spacing

    # ---- 误检（幽灵锚点）----
    # 钢门 / 电梯 / 配电间产生的自然磁异常，会被误认成标记。
    # 系统于是把"我在某个标记处"的修正打在了错误的位置上 —— 误差量级等于
    # 人到最近标记的距离，可能有好几米。这是磁标记方案最现实的失效模式。
    ghost_at = {}
    if false_anchor_prob > 0 and fix_steps:
        mk_xy = np.array(list(fix_steps.values()))
        armed = True
        for k, s in enumerate(steps):
            p = np.array([s["x_true"], s["y_true"]])
            inz = False
            for (x0, x1, y0, y1) in disturbed_zones():
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    inz = True
                    break
            if inz and armed:
                armed = False
                if rng.random() < false_anchor_prob:
                    d = np.hypot(mk_xy[:, 0] - p[0], mk_xy[:, 1] - p[1])
                    jj = int(np.argmin(d))
                    ghost_at[k] = (float(mk_xy[jj, 0]), float(mk_xy[jj, 1]))
            elif not inz:
                armed = True

    # 触发开关（滞回：靠近触发一次，走远后重新武装 —— 保证每圈都生效）
    qr_armed = [True] * len(qr)

    px = np.full(N, ENTRY[0]) + rng.normal(0, 0.15, N)
    py = np.full(N, ENTRY[1]) + rng.normal(0, 0.15, N)
    ph = np.full(N, 0.0) + rng.normal(0, np.deg2rad(3.0), N)

    est_t, est_x, est_y = [], [], []
    b_hat = 0.0          # 慢变磁偏差的在线估计
    bias_p = np.zeros(N)  # 逐粒子的磁偏差状态（联合估计用）

    # 步态速度回归：整条轨迹一次性推理（矢量化的，不逐步调模型）。
    # ★ 模型文件缺失会直接报错退出，不静默回退常数式——否则又回到"假装有 AI"。
    sl_seq = predict_steps_sl(steps)
    # 逐粒子步长过程噪声：默认用模型自身的相对 RMSE（记录在模型元数据里）
    proc = cfg.step_sl_proc_std
    if proc is None:
        proc = float(_get_step_model()[1].get("proc_std", 0.03))

    for k, s in enumerate(steps):
        # ---- 状态转移 ----
        # 航向：陀螺仪积分 + 过程噪声
        ph = ph + s["dpsi"] + rng.normal(0, np.deg2rad(1.2), N)
        # 速度/步长：模型推理输出 + 逐粒子过程噪声（代表模型残差）
        sl = sl_seq[k] * (1.0 + proc * rng.standard_normal(N))
        px = px + sl * np.cos(ph)
        py = py + sl * np.sin(ph)

        logw = np.zeros(N)

        # ---- 地图约束（不可穿墙） ----
        if use_map:
            ok = in_map(px, py)
            # 少量"软惩罚"而非硬杀，避免粒子瞬间耗尽
            logw += np.where(ok, 0.0, -60.0)

        # ---- 磁力计航向量测 ----
        if use_mag:
            # 慢变磁偏差的在线估计：用"磁读数 − 当前航向估计"的残差做低通跟踪，
            # 再从量测里减掉。目的是把"系统性偏差"和"随机噪声"分开处理——
            # 前者平均不掉、必须估计；后者交给滤波。
            # 增益必须小，否则会把真实的航向漂移一起吸进来。
            if mag_bias_adapt > 0.0:
                ph_bar = float(np.arctan2(np.mean(np.sin(ph)), np.mean(np.cos(ph))))
                resid_h = np.arctan2(np.sin(s["mag"] - ph_bar), np.cos(s["mag"] - ph_bar))
                b_hat += mag_bias_adapt * resid_h
            sig = mag_sigma_at(np.array([px.mean()]), np.array([py.mean()]),
                               cfg.mag_std_clean, cfg.mag_std_disturb)[0] * mag_sigma_scale
            if mag_bias_rw > 0.0:
                # ★ 把磁偏差作为粒子的状态变量做联合估计：
                #   偏差与航向单独看不可观测，但"带着错误偏差的粒子会撞墙"，
                #   地图约束因此能挑出偏差估计正确的那些粒子。
                bias_p = bias_p + rng.normal(0, np.deg2rad(mag_bias_rw), N)
                m = s["mag"] - bias_p
            else:
                m = s["mag"] - b_hat
            d = np.arctan2(np.sin(m - ph), np.cos(m - ph))
            logw += -0.5 * (d / sig) ** 2

        # ---- 二维码锚点 ----
        # 关键：是否"扫到码"由人的真实位置决定（人真的走到牌子前才扫），
        #       绝不能由滤波器自己的估计决定，否则就是自我确认偏差
        for qi, (qx, qy) in enumerate(qr):
            dist = float(np.hypot(s["x_true"] - qx, s["y_true"] - qy))
            if qr_armed[qi] and dist < 1.2:
                qr_armed[qi] = False
                logw += -0.5 * ((px - s["x_true"]) / cfg.qr_std) ** 2
                logw += -0.5 * ((py - s["y_true"]) / cfg.qr_std) ** 2
            elif dist > 3.0:
                qr_armed[qi] = True

        # ---- 均匀绝对锚点：每 S 米一次"当前真实位置"的绝对观测 ----
        # 只有当地可检测时才拿得到修正（磁标记在钢门附近会被背景淹没）
        if k in fix_steps and s.get("detectable", True):
            logw += -0.5 * ((px - s["x_true"]) / sig_anchor) ** 2
            logw += -0.5 * ((py - s["y_true"]) / sig_anchor) ** 2

        # ---- 误检：把修正打在了错误的位置 ----
        if k in ghost_at:
            gx, gy = ghost_at[k]
            logw += -0.5 * ((px - gx) / sig_anchor) ** 2
            logw += -0.5 * ((py - gy) / sig_anchor) ** 2

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
            if mag_bias_rw > 0.0:
                bias_p = bias_p[idx]      # 偏差状态必须跟着粒子一起重采样
            # 粗糙化，防止粒子贫化
            px = px + rng.normal(0, 0.05, N)
            py = py + rng.normal(0, 0.05, N)
            ph = ph + rng.normal(0, np.deg2rad(0.8), N)

        est_t.append(s["t"])
        est_x.append(float(np.mean(px)))
        est_y.append(float(np.mean(py)))

    return np.array(est_t), np.array(est_x), np.array(est_y)


def run_pure_dead_reckoning(seed, steps, cfg):
    """
    A 方案：纯航位推算
      - 航向：只用陀螺仪积分（零偏 + 标度因数误差全部保留）
      - 步长：固定值 0.74 m（与真值均值 0.70 m 有 +5.7% 系统偏差）
      - 无地图约束、无磁力计、无锚点
    起点与初始航向已知（入口是已知的，三种方法都一样，不偏袒）
    """
    rng = np.random.default_rng(seed + 777001)
    x, y, h = ENTRY[0], ENTRY[1], 0.0
    est_t, est_x, est_y = [], [], []
    for s in steps:
        h += s["dpsi"]                                    # 陀螺仪积分（偏差原样累积）
        sl = cfg.fixed_sl * (1.0 + cfg.A_sl_random * rng.standard_normal())  # 固定步长模型
        x += sl * np.cos(h)
        y += sl * np.sin(h)
        est_t.append(s["t"]); est_x.append(x); est_y.append(y)
    return np.array(est_t), np.array(est_x), np.array(est_y)


def run_speed_only(seed, steps, cfg):
    """
    方案 0：只有"入口位置 + 走了多远"，没有任何方向信息
           —— 这正是最初设想的原始形态，用来量化"缺方向"的代价
    只能假设方向恒定（朝入口朝向），于是轨迹是一条直线
    """
    rng = np.random.default_rng(seed + 333001)
    x, y, h = ENTRY[0], ENTRY[1], 0.0
    est_t, est_x, est_y = [], [], []
    for s in steps:
        sl = cfg.fixed_sl * (1.0 + cfg.A_sl_random * rng.standard_normal())
        x += sl * np.cos(h)
        y += sl * np.sin(h)
        est_t.append(s["t"]); est_x.append(x); est_y.append(y)
    return np.array(est_t), np.array(est_x), np.array(est_y)


# ============================================================
# 5. 蒙特卡洛评估
# ============================================================
def interp_series(t_src, x_src, y_src, t_grid):
    return (np.interp(t_grid, t_src, x_src), np.interp(t_grid, t_src, y_src))


def run_monte_carlo(n_runs=40, anchor_spacing=15.0, cfg=SensorCfg(),
                    methods=("A", "B", "C"), t_grid=None, seed0=1000):
    """
    返回 dict：
      t_grid          公共时间栅格
      err[method]     (n_runs, len(t_grid)) 误差矩阵
      dist_grid       真值累计里程栅格（等长于 t_grid，取各次运行中位数）
    """
    errs = {m: [] for m in methods}
    dists = []
    for r in range(n_runs):
        seed = seed0 + r
        tr = gen_truth(seed)
        steps = simulate_steps(seed, tr, cfg)
        if len(steps) < 10:
            continue
        t_end = steps[-1]["t"]
        if t_grid is None:
            t_grid_local = np.linspace(0.0, t_end, 200)
        else:
            t_grid_local = t_grid[t_grid <= t_end]
            if len(t_grid_local) < 10:
                continue

        # 真值在时间栅格上的位置
        tx, ty = interp_series(tr["t"], tr["x"], tr["y"], t_grid_local)
        # 真值累计里程
        dd = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(tx), np.diff(ty)))])
        dists.append(dd)

        if "A" in methods:
            ta, xa, ya = run_pure_dead_reckoning(seed, steps, cfg)
            xi, yi = interp_series(ta, xa, ya, t_grid_local)
            errs["A"].append(np.hypot(xi - tx, yi - ty))

        if "B" in methods:
            tb, xb, yb = run_particle_filter(seed, steps, cfg, use_map=True,
                                             use_mag=True, anchor_spacing=None,
                                             qr_points=False)
            xi, yi = interp_series(tb, xb, yb, t_grid_local)
            errs["B"].append(np.hypot(xi - tx, yi - ty))

        if "C" in methods:
            tc, xc, yc = run_particle_filter(seed, steps, cfg, use_map=True,
                                             use_mag=True,
                                             anchor_spacing=anchor_spacing,
                                             qr_points=True)
            xi, yi = interp_series(tc, xc, yc, t_grid_local)
            errs["C"].append(np.hypot(xi - tx, yi - ty))

    out = dict(t_grid=t_grid_local,
               dist_grid=np.median(np.array(dists), axis=0),
               err={m: np.array(v) for m, v in errs.items()})
    return out
