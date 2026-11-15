"""Minimal chart rendering with Pillow only (no matplotlib).

Rationale: the figures this package needs are all "a handful of series against
a log axis".  Pulling in matplotlib for that would make the repository heavier
than the model it contains, and would break on machines that only have numpy.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189),
    (255, 127, 14), (23, 190, 207), (140, 86, 75), (227, 119, 194),
]


def _font(size: int = 14, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if step >= raw:
            break
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-12:
        out.append(v)
        v += step
    return out


def _log_ticks(lo: float, hi: float) -> List[float]:
    out = []
    lo_e = math.floor(math.log10(lo))
    hi_e = math.ceil(math.log10(hi))
    for e in range(lo_e, hi_e + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo <= v <= hi:
                out.append(v)
    return out


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e6 or a < 1e-3:
        e = int(math.floor(math.log10(a)))
        m = v / 10 ** e
        return f"{m:.0f}e{e}" if abs(m - round(m)) < 1e-9 else f"{m:.1f}e{e}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return f"{v:.2f}".rstrip("0").rstrip(".")


class Figure:
    def __init__(self, width=900, height=560, title="", xlabel="", ylabel="",
                 xlog=False, ylog=False, pad=(92, 60, 42, 150)):
        self.w, self.h = width, height
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.xlog, self.ylog = xlog, ylog
        self.left, self.top, self.right_pad, self.bottom = pad
        self.series: List[dict] = []
        self.hlines: List[Tuple[float, str]] = []
        self.vlines: List[Tuple[float, str]] = []
        self.annots: List[Tuple[float, float, str]] = []

    # -- primitives
    def line(self, xs, ys, label="", color=None, width=3, marker=False):
        self.series.append(dict(xs=list(xs), ys=list(ys), label=label,
                                color=color, width=width, kind="line", marker=marker))
        return self

    def scatter(self, xs, ys, label="", color=None, r=6):
        self.series.append(dict(xs=list(xs), ys=list(ys), label=label,
                                color=color, width=0, kind="scatter", r=r))
        return self

    def hline(self, y, label=""):
        self.hlines.append((y, label))
        return self

    def vline(self, x, label=""):
        self.vlines.append((x, label))
        return self

    def annotate(self, x, y, text):
        self.annots.append((x, y, text))
        return self

    # -- rendering
    def _bounds(self):
        xs = [v for s in self.series for v in s["xs"]] + [v for v, _ in self.vlines]
        ys = [v for s in self.series for v in s["ys"]] + [v for v, _ in self.hlines]
        if not xs or not ys:
            raise ValueError("empty figure")
        if self.xlog:
            xs = [v for v in xs if v > 0]
        if self.ylog:
            ys = [v for v in ys if v > 0]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if self.xlog:
            x0, x1 = math.log10(x0), math.log10(x1)
        if self.ylog:
            y0, y1 = math.log10(y0), math.log10(y1)
        xpad = (x1 - x0) * 0.06 or 0.5
        ypad = (y1 - y0) * 0.10 or 0.5
        return x0 - xpad, x1 + xpad, max(y0 - ypad, -1e9), y1 + ypad

    def _sx(self, v, b):
        lo, hi = b[0], b[1]
        t = math.log10(v) if self.xlog else v
        return self.left + (t - lo) / (hi - lo) * (self.w - self.left - self.right_pad)

    def _sy(self, v, b):
        lo, hi = b[2], b[3]
        t = math.log10(v) if self.ylog else v
        return self.h - self.bottom - (t - lo) / (hi - lo) * (self.h - self.top - self.bottom)

    def save(self, path: str):
        b = self._bounds()
        img = Image.new("RGB", (self.w, self.h), "white")
        d = ImageDraw.Draw(img)
        f_t = _font(20, True)
        f_l = _font(15)
        f_s = _font(13)

        # --- grid + ticks
        xt = _log_ticks(10 ** b[0], 10 ** b[1]) if self.xlog else _nice_ticks(b[0], b[1])
        yt = _log_ticks(10 ** b[2], 10 ** b[3]) if self.ylog else _nice_ticks(b[2], b[3])
        for xv in xt:
            px = self._sx(xv, b)
            d.line([(px, self.top), (px, self.h - self.bottom)], fill=(232, 232, 232))
            lab = _fmt(xv)
            tw = d.textlength(lab, font=f_s)
            d.text((px - tw / 2, self.h - self.bottom + 8), lab, fill=(60, 60, 60), font=f_s)
        for yv in yt:
            py = self._sy(yv, b)
            d.line([(self.left, py), (self.w - self.right_pad, py)], fill=(232, 232, 232))
            lab = _fmt(yv)
            tw = d.textlength(lab, font=f_s)
            d.text((self.left - tw - 8, py - 7), lab, fill=(60, 60, 60), font=f_s)

        d.rectangle([self.left, self.top, self.w - self.right_pad, self.h - self.bottom],
                    outline=(90, 90, 90), width=2)

        # --- reference lines
        for yv, lab in self.hlines:
            py = self._sy(yv, b)
            d.line([(self.left, py), (self.w - self.right_pad, py)],
                   fill=(150, 150, 150), width=2)
            if lab:
                d.text((self.left + 8, py - 18), lab, fill=(110, 110, 110), font=f_s)
        for xv, lab in self.vlines:
            px = self._sx(xv, b)
            d.line([(px, self.top), (px, self.h - self.bottom)],
                   fill=(170, 170, 170), width=1)
            if lab:
                d.text((px + 5, self.top + 4), lab, fill=(110, 110, 110), font=f_s)

        # --- series
        for i, s in enumerate(self.series):
            col = s["color"] or PALETTE[i % len(PALETTE)]
            pts = [(self._sx(x, b), self._sy(y, b)) for x, y in zip(s["xs"], s["ys"])]
            if s["kind"] == "line" and len(pts) > 1:
                d.line(pts, fill=col, width=s["width"])
            if s["kind"] == "scatter" or s["marker"]:
                r = s.get("r", 5)
                for (px, py) in pts:
                    d.ellipse([px - r, py - r, px + r, py + r], fill="white",
                              outline=col, width=3)

        for x, y, text in self.annots:
            d.text((self._sx(x, b) + 8, self._sy(y, b) - 18), text,
                   fill=(70, 70, 70), font=f_s)

        # --- labels
        if self.title:
            tw = d.textlength(self.title, font=f_t)
            d.text(((self.w - tw) / 2, 14), self.title, fill=(20, 20, 20), font=f_t)
        if self.xlabel:
            tw = d.textlength(self.xlabel, font=f_l)
            d.text(((self.w - tw) / 2, self.h - 30), self.xlabel, fill=(30, 30, 30), font=f_l)
        if self.ylabel:
            tw = d.textlength(self.ylabel, font=f_l)
            tmp = Image.new("RGBA", (int(tw) + 8, 24), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((2, 2), self.ylabel, fill=(30, 30, 30), font=f_l)
            tmp = tmp.rotate(90, expand=True)
            img.paste(tmp, (12, (self.h - tmp.size[1]) // 2), tmp)

        # --- legend
        items = [(s["color"] or PALETTE[i % len(PALETTE)], s["label"])
                 for i, s in enumerate(self.series) if s["label"]]
        if items:
            lw = max(d.textlength(lbl, font=f_s) for _, lbl in items) + 58
            lh = 26 * len(items) + 14
            x0 = self.w - self.right_pad - lw - 8
            y0 = self.top + 10
            d.rounded_rectangle([x0, y0, x0 + lw, y0 + lh], 8, fill=(255, 255, 255),
                                outline=(210, 210, 210), width=1)
            for k, (col, lbl) in enumerate(items):
                yy = y0 + 16 + k * 26
                d.line([(x0 + 12, yy + 6), (x0 + 40, yy + 6)], fill=col, width=4)
                d.text((x0 + 48, yy - 2), lbl, fill=(40, 40, 40), font=f_s)

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        img.save(path)
        return path


# --------------------------------------------------------------------------
# Ready-made figures
# --------------------------------------------------------------------------
def pred_vs_measured(pred, meas, path, title="Predicted vs measured step time"):
    lo = min(min(pred), min(meas)) * 0.85
    hi = max(max(pred), max(meas)) * 1.15
    fig = Figure(720, 640, title=title, xlabel="measured step time (ms)",
                 ylabel="predicted (ms)", xlog=True, ylog=True)
    fig.line([lo, hi], [lo, hi], label="y = x", color=(150, 150, 150), width=2)
    fig.scatter(meas, pred, label="configs", color=PALETTE[0], r=6)
    return fig.save(path)


def cp_scaling(curves: dict, path, title="CP scaling", ideal=True):
    fig = Figure(900, 560, title=title, xlabel="CP degree",
                 ylabel="speedup vs CP=1", xlog=True, ylog=False)
    cps = sorted(next(iter(curves.values()))[0])
    if ideal:
        fig.line(cps, cps, label="ideal (linear)", color=(200, 200, 200), width=2)
    for i, (label, (xs, ys)) in enumerate(curves.items()):
        fig.line(xs, ys, label=label, color=PALETTE[i % len(PALETTE)], marker=True)
    fig.hline(1.0, "no gain")
    return fig.save(path)


def efficiency_curve(params, path, cp_list=(1, 2, 4, 8), T=8192,
                     title="Kernel efficiency vs chunk length"):
    from .cost import efficiency

    fig = Figure(820, 540, title=title, xlabel="chunk length c = T / CP (tokens)",
                 ylabel="efficiency (fraction of peak)", xlog=True, ylog=False)
    cs = [T / cp for cp in cp_list]
    lo, hi = int(min(cs) / 2), int(max(cs) * 2)
    xs = [lo * (hi / lo) ** (k / 120.0) for k in range(121)]
    fig.line(xs, [efficiency(params, c) for c in xs], label="eff(c)", width=4)
    fig.scatter(cs, [efficiency(params, c) for c in cs],
                label="CP configs at T=%d" % T, r=6)
    for c in cs:
        fig.vline(c, f"CP={int(round(T / c))}")
    return fig.save(path)
