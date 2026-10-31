# -*- coding: utf-8 -*-
"""极简绘图工具（Pillow 实现，零额外依赖）"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_R = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
_fc = {}


def F(size, bold=False):
    key = (size, bold)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size)
    return _fc[key]


# 配色（对色盲友好、打印可辨）
C_A = (214, 69, 65)      # 红  - 方案0 / 差
C_B = (230, 145, 56)     # 橙
C_C = (56, 108, 176)     # 蓝
C_D = (46, 139, 87)      # 绿
GRID = (226, 230, 235)
AXIS = (60, 66, 74)


class Chart:
    def __init__(self, w=1240, h=740, pad=(104, 66, 48, 92), title="", sub=""):
        self.w, self.h, self.pad = w, h, pad
        self.img = Image.new("RGB", (w, h), "white")
        self.d = ImageDraw.Draw(self.img)
        self.title, self.sub = title, sub
        self.xr, self.yr = (0, 1), (0, 1)
        self.logy = False
        self.xlab = self.ylab = ""
        self._xt = self._yt = None
        self._xfmt = "{:.0f}"
        self._yfmt = "{:.1f}"
        self.legend_items = []

    # --- 坐标设置 ---
    def set_x(self, lo, hi, ticks=None, fmt="{:.0f}", label=""):
        self.xr = (lo, hi); self._xt = ticks; self._xfmt = fmt; self.xlab = label

    def set_y(self, lo, hi, ticks=None, fmt="{:.1f}", label="", log=False):
        self.yr = (lo, hi); self._yt = ticks; self._yfmt = fmt
        self.ylab = label; self.logy = log

    # --- 映射 ---
    def X(self, v):
        l, r = self.pad[0], self.w - self.pad[2]
        return l + (np.asarray(v, float) - self.xr[0]) / (self.xr[1] - self.xr[0]) * (r - l)

    def Y(self, v):
        t, b = self.pad[1], self.h - self.pad[3]
        v = np.asarray(v, float)
        if self.logy:
            v = np.log10(np.maximum(v, 1e-9))
            lo, hi = np.log10(self.yr[0]), np.log10(self.yr[1])
        else:
            lo, hi = self.yr
        return b - (v - lo) / (hi - lo) * (b - t)

    # --- 绘制 ---
    def grid_and_axes(self):
        d = self.d
        # y 网格 + 刻度
        yt = self._yt if self._yt is not None else np.linspace(
            np.log10(self.yr[0]) if self.logy else self.yr[0],
            np.log10(self.yr[1]) if self.logy else self.yr[1], 6)
        for v in yt:
            py = float(self.Y(10 ** v if self.logy else v))
            d.line([(self.pad[0], py), (self.w - self.pad[2], py)], fill=GRID, width=1)
            lab = self._yfmt.format(10 ** v if self.logy else v)
            d.text((self.pad[0] - 12, py), lab, font=F(19), fill=AXIS, anchor="rm")
        # x 网格 + 刻度
        xt = self._xt if self._xt is not None else np.linspace(*self.xr, 6)
        for v in xt:
            px = float(self.X(v))
            d.line([(px, self.pad[1]), (px, self.h - self.pad[3])], fill=GRID, width=1)
            d.text((px, self.h - self.pad[3] + 12), self._xfmt.format(v),
                   font=F(19), fill=AXIS, anchor="ma")
        # 轴框
        d.rectangle([self.pad[0], self.pad[1], self.w - self.pad[2], self.h - self.pad[3]],
                    outline=AXIS, width=2)
        # 轴标题
        d.text(((self.pad[0] + self.w - self.pad[2]) / 2, self.h - 32), self.xlab,
               font=F(21, True), fill=AXIS, anchor="mm")
        tmp = Image.new("RGB", (620, 34), "white")
        ImageDraw.Draw(tmp).text((310, 17), self.ylab, font=F(21, True), fill=AXIS, anchor="mm")
        self.img.paste(tmp.rotate(90, expand=True), (6, int((self.pad[1] + self.h - self.pad[3]) / 2) - 310))
        # 标题
        d.text((self.pad[0], 16), self.title, font=F(27, True), fill=(25, 29, 34))
        if self.sub:
            d.text((self.pad[0], 46), self.sub, font=F(17), fill=(110, 118, 128))

    def line(self, xs, ys, color, width=3):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        pts = [(float(self.X(a)), float(self.Y(b))) for a, b in zip(xs, ys)]
        if len(pts) > 1:
            self.d.line(pts, fill=color, width=width, joint="curve")

    def band(self, xs, lo, hi, color, alpha=0.20):
        xs = np.asarray(xs, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        top = [(float(self.X(a)), float(self.Y(b))) for a, b in zip(xs, hi)]
        bot = [(float(self.X(a)), float(self.Y(b))) for a, b in zip(xs, lo)][::-1]
        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        ImageDraw.Draw(ov).polygon(top + bot, fill=color + (int(255 * alpha),))
        self.img.paste(Image.alpha_composite(self.img.convert("RGBA"), ov).convert("RGB"), (0, 0))
        self.d = ImageDraw.Draw(self.img)

    def hline(self, y, color=(150, 156, 164), dash=8, width=1):
        py = float(self.Y(y))
        x = self.pad[0]
        while x < self.w - self.pad[2]:
            self.d.line([(x, py), (min(x + dash, self.w - self.pad[2]), py)], fill=color, width=width)
            x += dash * 2

    def vline(self, x, color=(150, 156, 164), dash=8, width=1):
        px = float(self.X(x))
        y = self.pad[1]
        while y < self.h - self.pad[3]:
            self.d.line([(px, y), (px, min(y + dash, self.h - self.pad[3]))], fill=color, width=width)
            y += dash * 2

    def marker(self, x, y, color, r=5):
        px, py = float(self.X(x)), float(self.Y(y))
        self.d.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)

    def text(self, x, y, s, color=(25, 29, 34), size=19, bold=False, anchor="lm"):
        self.d.text((x, y), s, font=F(size, bold), fill=color, anchor=anchor)

    def legend(self, loc="upper left", ncol=1):
        items = self.legend_items
        if not items:
            return
        pad, lh = 14, 30
        cw = int(max(F(19).getlength(t) for _, t in items)) + 62
        nrow = (len(items) + ncol - 1) // ncol
        bw, bh = cw * ncol + pad * 2, lh * nrow + pad * 2
        if loc == "below":
            bx = (self.w - bw) // 2
            by = self.h - self.pad[3] + 44
        elif loc == "upper right":
            bx, by = self.w - self.pad[2] - bw - 18, self.pad[1] + 16
        else:
            bx, by = self.pad[0] + 18, self.pad[1] + 16
        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle([bx, by, bx + bw, by + bh], 8,
                                             fill=(255, 255, 255, 238),
                                             outline=(186, 192, 201, 255), width=1)
        self.img.paste(Image.alpha_composite(self.img.convert("RGBA"), ov).convert("RGB"), (0, 0))
        self.d = ImageDraw.Draw(self.img)
        for i, (col, txt) in enumerate(items):
            r, c = divmod(i, ncol)
            x = bx + pad + c * cw
            y = by + pad + r * lh + lh // 2
            self.d.line([(x, y), (x + 30, y)], fill=col, width=4)
            self.d.text((x + 40, y), txt, font=F(19), fill=(35, 40, 46), anchor="lm")

    def save(self, path):
        self.img.save(path)
        return path
