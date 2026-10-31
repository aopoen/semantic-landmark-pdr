# -*- coding: utf-8 -*-
"""
步态速度回归模型 —— 纯 numpy 手写的单隐层 MLP。

这是作品里"AI"的实体：输入一步的步态特征，输出该步的步长（米）。训练脚本
见 step_model_train.py，权重存为 out/step_model.npz。

为什么是纯 numpy
----------------
运行环境只有 numpy / pandas（无 sklearn / torch），且不应该为了一个 801 参数的
网络去装深度学习框架。手写前向/反向总共不到 120 行，可复现性反而更好——
评委能逐行读完。

设计要点
--------
* 目标取 **log(步长)** 再标准化：步长的观测噪声本质是乘性的（±5% 相对波动），
  在 log 空间里它才是等方差的，用 MSE 才等价于优化相对误差。
* 输出层**不接激活**，线性输出。
* 验证集早停 + L2，防止 3 万样本上过拟合。

用法
----
  from step_model import StepLengthMLP, get_model, predict_steps_sl
  model, meta = get_model()               # 进程内单例；文件缺失则报错退出
  sl = predict_steps_sl(steps)            # steps[k]["feat"] -> 每步平均步长
"""
import os
import json
import numpy as np

from step_features import N_FEAT, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "out", "step_model.npz")


class StepLengthMLP:
    """6 → 32 (ReLU) → 16 (ReLU) → 1 (线性)，参数 801。"""

    def __init__(self, n_in=N_FEAT, hidden=(32, 16), seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in] + list(hidden) + [1]
        self.W, self.b = [], []
        for k in range(len(sizes) - 1):
            # He 初始化：ReLU 需要按 fan_in 缩放
            self.W.append(rng.standard_normal((sizes[k], sizes[k + 1]))
                          * np.sqrt(2.0 / sizes[k]))
            self.b.append(np.zeros(sizes[k + 1]))
        # 标准化统计量（训练时填充）
        self.feat_mean = np.zeros(n_in)
        self.feat_std = np.ones(n_in)
        self.y_mean = 0.0
        self.y_std = 1.0

    # ---------------------------------------------------------------- 前向
    def _forward(self, X):
        Zs, As = [], [X]
        A = X
        for k in range(len(self.W) - 1):
            Z = A @ self.W[k] + self.b[k]
            A = np.maximum(Z, 0.0)          # ReLU
            Zs.append(Z)
            As.append(A)
        Y = (A @ self.W[-1] + self.b[-1])[:, 0]
        return Y, Zs, As

    def _loss_grad(self, Xn, yn, l2):
        """一个 minibatch 上的 MSE 损失与梯度（手写反向传播）。"""
        Y, Zs, As = self._forward(Xn)
        m = len(Xn)
        loss = float(np.mean((Y - yn) ** 2))

        grads = [None] * len(self.W)
        dZ = (2.0 * (Y - yn) / m)[:, None]      # dL/dZ_out
        for k in range(len(self.W) - 1, -1, -1):
            gW = As[k].T @ dZ + l2 * self.W[k]
            gb = dZ.sum(0)
            grads[k] = (gW, gb)
            if k > 0:
                dA = dZ @ self.W[k].T
                dZ = dA * (Zs[k - 1] > 0)       # ReLU 的导数
        return loss, grads

    def _mse(self, X, y):
        Xn = (X - self.feat_mean) / self.feat_std
        ly = np.log(np.maximum(y, 1e-6))
        yn = (ly - self.y_mean) / self.y_std
        Y, _, _ = self._forward(Xn)
        return float(np.mean((Y - yn) ** 2))

    # ---------------------------------------------------------------- 训练
    def fit(self, X, y, Xva=None, yva=None, epochs=300, lr=3e-3, l2=1e-5,
            batch=256, patience=30, seed=0, verbose=False):
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        self.feat_mean = X.mean(0)
        self.feat_std = X.std(0) + 1e-9
        ly = np.log(np.maximum(y, 1e-6))
        self.y_mean = float(ly.mean())
        self.y_std = float(ly.std() + 1e-9)

        Xn = (X - self.feat_mean) / self.feat_std
        yn = (ly - self.y_mean) / self.y_std

        rng = np.random.default_rng(seed)
        m = len(Xn)

        mW = [np.zeros_like(w) for w in self.W]
        vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]
        vb = [np.zeros_like(b) for b in self.b]
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0

        best = np.inf
        best_params = None
        bad = 0
        hist = []

        for ep in range(epochs):
            idx = rng.permutation(m)
            for s0 in range(0, m, batch):
                bi = idx[s0:s0 + batch]
                _, grads = self._loss_grad(Xn[bi], yn[bi], l2)
                step += 1
                for k in range(len(self.W)):
                    gW, gb = grads[k]
                    mW[k] = b1 * mW[k] + (1 - b1) * gW
                    vW[k] = b2 * vW[k] + (1 - b2) * gW ** 2
                    mb[k] = b1 * mb[k] + (1 - b1) * gb
                    vb[k] = b2 * vb[k] + (1 - b2) * gb ** 2
                    mWh = mW[k] / (1 - b1 ** step)
                    vWh = vW[k] / (1 - b2 ** step)
                    mbh = mb[k] / (1 - b1 ** step)
                    vbh = vb[k] / (1 - b2 ** step)
                    self.W[k] -= lr * mWh / (np.sqrt(vWh) + eps)
                    self.b[k] -= lr * mbh / (np.sqrt(vbh) + eps)

            if Xva is not None:
                vl = self._mse(Xva, yva)
                hist.append(vl)
                if verbose and (ep % 25 == 0 or ep == epochs - 1):
                    print("    epoch %3d  验证 MSE(log空间) = %.6f" % (ep, vl))
                if vl < best - 1e-7:
                    best = vl
                    best_params = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        if verbose:
                            print("    早停于 epoch %d（验证损失连续 %d 轮未改善）" % (ep, patience))
                        break

        if best_params is not None:
            self.W, self.b = best_params
        return hist

    # ---------------------------------------------------------------- 推理
    def predict_log(self, X):
        X = np.asarray(X, float).reshape(-1, self.W[0].shape[0])
        Xn = (X - self.feat_mean) / self.feat_std
        Y, _, _ = self._forward(Xn)
        return Y * self.y_std + self.y_mean

    def predict_sl(self, X, calib=None):
        """返回步长（米）。calib=(a, b) 为实测域仿射校准，可选。"""
        sl = np.exp(self.predict_log(X))
        if calib is not None:
            a, b = float(calib[0]), float(calib[1])
            sl = a * sl + b
        return sl

    # ---------------------------------------------------------------- 存读
    def save(self, path=None, meta=None):
        path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = {}
        for k, w in enumerate(self.W):
            d["W%d" % k] = w
        for k, b in enumerate(self.b):
            d["b%d" % k] = b
        d["feat_mean"] = np.asarray(self.feat_mean, float)
        d["feat_std"] = np.asarray(self.feat_std, float)
        d["y_mean"] = np.float64(self.y_mean)
        d["y_std"] = np.float64(self.y_std)
        d["meta"] = np.array(json.dumps(meta or {}, ensure_ascii=False))
        np.savez_compressed(path, **d)
        return path

    @staticmethod
    def load(path=None):
        path = path or DEFAULT_PATH
        if not os.path.exists(path):
            raise SystemExit(
                "步态回归模型不存在：%s\n请先运行 step_model_train.py 训练模型。" % path)
        z = np.load(path, allow_pickle=False)
        n_layers = sum(1 for k in z.files if k.startswith("W"))
        m = StepLengthMLP.__new__(StepLengthMLP)
        m.W = [z["W%d" % k] for k in range(n_layers)]
        m.b = [z["b%d" % k] for k in range(n_layers)]
        m.feat_mean = z["feat_mean"]
        m.feat_std = z["feat_std"]
        m.y_mean = float(z["y_mean"])
        m.y_std = float(z["y_std"])
        meta = json.loads(str(z["meta"]))

        # ★ 特征顺序/维度的守卫：顺序改了却没重训，会得到静默错误的结果。
        names = meta.get("feature_names")
        if names is not None and list(names) != list(FEATURE_NAMES):
            raise SystemExit(
                "模型的特征顺序 %s 与 step_features.FEATURE_NAMES %s 不一致，请重训。"
                % (names, FEATURE_NAMES))
        return m, meta


# ---------------------------------------------------------------- 进程内单例
_MODEL = None
_META = None


def get_model(path=None):
    """加载模型（进程内只读一次）。文件缺失或特征不一致时直接报错退出——
    **不做静默回退**，否则又退回"假装有 AI"。"""
    global _MODEL, _META
    if _MODEL is None:
        _MODEL, _META = StepLengthMLP.load(path or DEFAULT_PATH)
    return _MODEL, _META


def predict_steps_sl(steps, calib=None):
    """整条轨迹一次性预测每步的平均步长，返回 np.ndarray[len(steps)]。

    矢量化的：粒子滤波每帧都要用步长，不能在这里逐步调模型。
    """
    if len(steps) == 0:
        return np.zeros(0)
    feats = np.array([s["feat"] for s in steps], float)
    model, meta = get_model()
    if calib is None:
        calib = meta.get("calib")
    return model.predict_sl(feats, calib)
