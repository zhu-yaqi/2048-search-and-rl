"""绘图工具：统一保存训练/进化/评估曲线，便于写进实验报告。

字体处理说明：
    旧版代码用 try/except 选字体是个 bug —— 给 rcParams 赋值永远不会抛错，
    matplotlib 只有在真正渲染文本时才会回退到 DejaVu Sans 并出现中文方块。
    新版改成用 matplotlib.font_manager 真正枚举系统已安装字体，从候选列表
    里挑第一个 *确实* 装了的字体设进 rcParams。
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 无界面后端，便于在脚本中直接保存图片
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm


def _configure_chinese_font() -> Optional[str]:
    """从候选列表中挑一个真正已安装的中文字体，配置进 rcParams。

    返回最终生效的字体名（用于调试）；若没有任何候选可用，返回 None。
    """
    candidates = [
        "Microsoft YaHei",       # Windows 默认
        "SimHei",                # Windows
        "Noto Sans CJK SC",      # Linux 常见
        "Noto Sans SC",
        "PingFang SC",           # macOS
        "Heiti SC",              # macOS
        "WenQuanYi Zen Hei",     # Linux 常见
        "Source Han Sans SC",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    chosen: Optional[str] = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen is not None:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


_ACTIVE_FONT = _configure_chinese_font()


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_curves(xs: Sequence, series: dict, title: str, xlabel: str,
                ylabel: str, out_path: str):
    """series: {标签: y 序列}。"""
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 4.5))
    for label, ys in series.items():
        plt.plot(xs, ys, marker="o", markersize=3, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图已保存: {out_path}")


def plot_score_hist(scores: Sequence, title: str, out_path: str,
                    bins: int = 20):
    _ensure_dir(out_path)
    plt.figure(figsize=(7, 4.5))
    plt.hist(scores, bins=bins, color="#4c72b0", alpha=0.85, edgecolor="white")
    plt.title(title)
    plt.xlabel("分数")
    plt.ylabel("局数")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图已保存: {out_path}")


def plot_tile_distribution(tile_dist: dict, title: str, out_path: str):
    """画最大方块分布柱状图。tile_dist 形如 {64: 9, 128: 25, ...}。"""
    _ensure_dir(out_path)
    items = sorted(tile_dist.items(), key=lambda x: int(x[0]))
    labels = [str(int(k)) for k, _ in items]
    counts = [v for _, v in items]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts, color="#dd8452", alpha=0.9, edgecolor="white")
    for bar, c in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(c), ha="center", va="bottom", fontsize=9)
    plt.title(title)
    plt.xlabel("最大方块")
    plt.ylabel("局数")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"图已保存: {out_path}")
