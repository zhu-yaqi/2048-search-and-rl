"""根据 eval_lookahead.py 的输出，生成监督网络"纯网络 vs 前瞻"的对比图。

生成：
    results/supervised/lookahead_compare.png        指标对比柱状图（仅需汇总）
    results/supervised/lookahead_score_hist.png      前瞻版分数分布（需原始 scores）
    results/supervised/lookahead_tile_dist.png       前瞻版最大方块分布（需原始数据）

用法：
    python scripts/plot_lookahead.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.plotting import plot_score_hist, plot_tile_distribution, _ACTIVE_FONT

COMPARE_JSON = "results/supervised/lookahead_compare.json"


def plot_compare_bars(base: dict, look: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # 左：分数类指标
    metrics = ["平均分", "中位数分", "最高分"]
    base_vals = [base["avg_score"], base["median_score"], base["max_score"]]
    look_vals = [look["avg_score"], look["median_score"], look["max_score"]]
    x = np.arange(len(metrics))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, base_vals, w, label="纯网络", color="#4c72b0")
    b2 = axes[0].bar(x + w / 2, look_vals, w, label="网络+前瞻", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel("分数")
    axes[0].set_title("分数指标对比")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    for bars in (b1, b2):
        for bar in bars:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)

    # 右：达成率
    rates = ["≥512", "≥1024", "≥2048"]
    base_r = [base["rate_512"] * 100, base["rate_1024"] * 100, base["rate_2048"] * 100]
    look_r = [look["rate_512"] * 100, look["rate_1024"] * 100, look["rate_2048"] * 100]
    x2 = np.arange(len(rates))
    r1 = axes[1].bar(x2 - w / 2, base_r, w, label="纯网络", color="#4c72b0")
    r2 = axes[1].bar(x2 + w / 2, look_r, w, label="网络+前瞻", color="#dd8452")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(rates)
    axes[1].set_ylabel("达成率 (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("最大方块达成率对比")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    for bars in (r1, r2):
        for bar in bars:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{bar.get_height():.0f}%", ha="center", va="bottom", fontsize=8)

    n = look.get("n_games", "?")
    fig.suptitle(f"监督网络推理增强对比（同 {n} 局）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"图已保存: {out_path}")


def main():
    print(f"使用中文字体: {_ACTIVE_FONT}")
    if not os.path.exists(COMPARE_JSON):
        raise FileNotFoundError(f"找不到 {COMPARE_JSON}，请先跑 scripts/eval_lookahead.py")
    with open(COMPARE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    base = data.get("baseline")
    look = data.get("lookahead")

    if base and look:
        plot_compare_bars(base, look, "results/supervised/lookahead_compare.png")

    # 前瞻版分布图（需要原始数据，重跑后才有）
    if look and look.get("scores"):
        n = look.get("n_games", len(look["scores"]))
        plot_score_hist(look["scores"], f"网络+前瞻 {n} 局分数分布",
                        "results/supervised/lookahead_score_hist.png", bins=12)
    else:
        print("[skip] lookahead 无原始 scores，跳过分数分布图（重跑 eval_lookahead 后再画）")

    if look and look.get("tile_distribution"):
        plot_tile_distribution({int(k): v for k, v in look["tile_distribution"].items()},
                               "网络+前瞻 最大方块分布",
                               "results/supervised/lookahead_tile_dist.png")
    else:
        print("[skip] lookahead 无 tile_distribution，跳过最大方块分布图")

    print("完成。")


if __name__ == "__main__":
    main()
