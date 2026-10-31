# -*- coding: utf-8 -*-
"""
生成论文图：图 1 方法闭环图 + 图 2 楼层平面图（Pillow 绘制，中文微软雅黑）。
输出到 论文/figs/。
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf']
FONT_PATH = next((f for f in _FONT_CANDIDATES if os.path.exists(f)), None)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)


def _font(size):
    if FONT_PATH is None:
        return ImageFont.load_default()
    return ImageFont.truetype(FONT_PATH, size)


# ---- 平面图数据（同 sim_core.py）----
RECTS = [
    ('C1', 0.0, 0.0, 60.0, 3.0, 'corridor'),
    ('C2', 28.0, 3.0, 32.0, 22.0, 'corridor'),
    ('C3', 28.0, 19.0, 60.0, 22.0, 'corridor'),
    ('C4', 57.0, 3.0, 60.0, 22.0, 'corridor'),
    ('R1', 5.0, 4.0, 15.0, 11.5, 'room'),
    ('D1', 9.0, 3.0, 11.0, 4.0, 'door'),
    ('R2', 40.0, 4.0, 50.0, 11.5, 'room'),
    ('D2', 44.0, 3.0, 46.0, 4.0, 'door'),
]
ENTRY = (0.5, 1.5)
LANDMARKS = np.array([
    (58.5, 1.5), (58.5, 20.5), (30.0, 20.5), (30.0, 1.5),
    (4.0, 1.5), (10.0, 3.5), (45.0, 3.5),
], dtype=float)
LM_LABEL = ['转角1', '转角2', '转角3', '转角4', '折返', '门101', '门102']

from sim_core import gen_truth  # noqa: E402


def draw_floorplan():
    scale = 26
    x_min, x_max, y_min, y_max = -2.0, 63.0, -2.0, 24.0
    W = int((x_max - x_min) * scale)
    H = int((y_max - y_min) * scale)
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    def tx(x):
        return (x - x_min) * scale

    def ty(y):
        return (y_max - y) * scale

    for name, x0, y0, x1, y1, kind in RECTS:
        px0, py0 = tx(x0), ty(y1)
        px1, py1 = tx(x1), ty(y0)
        if kind == 'corridor':
            fill, outline, w = '#e8f0f8', '#5b7ea0', 3
        elif kind == 'room':
            fill, outline, w = '#f0ede4', '#b8a88a', 3
        else:
            fill, outline, w = '#ffffff', '#999999', 3
        d.rectangle([px0, py0, px1, py1], fill=fill, outline=outline, width=w)

    tr = gen_truth(1000)
    pts = [(tx(x), ty(y)) for x, y in zip(tr['x'], tr['y'])]
    d.line(pts, fill='#c0392b', width=4)

    for (lx, ly), lab in zip(LANDMARKS, LM_LABEL):
        px, py = tx(lx), ty(ly)
        r = 6
        d.ellipse([px - r, py - r, px + r, py + r],
                  fill='#e67e22', outline='#7f3f00', width=2)
        d.text((px + 8, py + 8), lab, fill='#7f3f00', font=_font(16))

    ex, ey = tx(ENTRY[0]), ty(ENTRY[1])
    d.rectangle([ex - 8, ey - 8, ex + 8, ey + 8], fill='#27ae60', outline='#0b5a2a', width=3)
    d.text((ex - 22, ey - 30), '入口', fill='#0b5a2a', font=_font(18))

    img.save(os.path.join(OUT, 'fig2_floorplan.png'))
    print('图 2 已生成 fig2_floorplan.png', img.size)


def _multiline(d, cx, cy, text, fill, size, anchor='mm'):
    f = _font(size)
    lines = text.split('\n')
    # 简单居中：逐行绘制，行高约 1.4*size
    lh = int(size * 1.5)
    total_h = lh * len(lines)
    y0 = cy - total_h // 2
    for i, ln in enumerate(lines):
        bbox = d.textbbox((0, 0), ln, font=f)
        w = bbox[2] - bbox[0]
        d.text((cx - w // 2, y0 + i * lh), ln, fill=fill, font=f)


def draw_pipeline():
    W, H = 2000, 1040
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    def box(cx, cy, w, h, text, fill, outline):
        d.rounded_rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                            radius=18, fill=fill, outline=outline, width=4)
        _multiline(d, cx, cy, text, '#222222', 40)

    def arrow(x1, y1, x2, y2, label=None):
        d.line([x1, y1, x2, y2], fill='#333333', width=5)
        # 箭头三角形
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        L = 22
        for da in (math.radians(155), math.radians(-155)):
            ax = x2 + L * math.cos(ang + da)
            ay = y2 + L * math.sin(ang + da)
            d.line([x2, y2, ax, ay], fill='#333333', width=5)
        if label:
            _multiline(d, (x1 + x2) // 2, (y1 + y2) // 2 - 18, label, '#555555', 28)

    # 主链
    box(330, 520, 420, 200, '惯性传感器\n(陀螺/加速度计)', '#e8f0f8', '#5b7ea0')
    box(790, 520, 420, 200, 'PDR\n步态回归+航向积分', '#e8f0f8', '#5b7ea0')
    box(1260, 520, 420, 200, '粒子滤波\n(1500 粒子)', '#fdf2e3', '#b8860b')
    box(1700, 520, 360, 200, '位置估计', '#eafaf1', '#1e8449')

    arrow(540, 520, 580, 520, '步长/航向')
    arrow(1000, 520, 1050, 520, '状态转移')
    arrow(1470, 520, 1520, 520, '后验均值')

    # LLM（上方）
    box(1260, 200, 440, 180, 'LLM 语义地标\n(离线生成)', '#f5eef8', '#7d3c98')
    arrow(1260, 290, 1260, 420, '地标观测')

    # 约束（下方）
    box(1260, 850, 440, 180, '墙体不可穿约束\n磁力计航向量测', '#fdedec', '#922b21')
    arrow(1260, 760, 1260, 620, '权重更新')

    img.save(os.path.join(OUT, 'fig1_pipeline.png'))
    print('图 1 已生成 fig1_pipeline.png', img.size)


if __name__ == '__main__':
    draw_floorplan()
    draw_pipeline()
