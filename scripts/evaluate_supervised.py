"""评估当前 supervised 网络模型并把完整 scores 写入 summary.json/supervised。

为什么需要这个脚本：
  - results/summary.json 的 supervised 段是早期 15k 样本模型的结果（avg≈3645），
    与 results/supervised/policy.pt（最新 120k 样本模型，avg≈5353）已经不匹配；
  - 评估完成后会把"scores、tile_distribution、aggregate 指标"等完整数据合并进
    summary.json，使 regenerate_plots.py 能基于 *真实* 数据出图。

用法：
    python scripts/evaluate_supervised.py --games 100
"""
import argparse
import json
import os
import time

import _bootstrap  # noqa: F401

import numpy as np
import torch

from src.supervised.model import NetAgent, load_net
from src.common.runner import evaluate_agent
from src.common.plotting import plot_score_hist, plot_tile_distribution


SUMMARY_PATH = os.path.join("results", "summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="results/supervised/policy.pt")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=10000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}, model = {args.model}")
    net = load_net(args.model, device=device)
    agent = NetAgent(net, device=device)

    t0 = time.time()
    summ = evaluate_agent(agent, n_games=args.games, base_seed=args.base_seed, verbose=False)
    elapsed = time.time() - t0
    print(summ)
    print(f"耗时: {elapsed:.1f}s")

    # 派生指标
    tiles = np.array([t for t in [int(k) for k in summ.tile_distribution
                                  for _ in range(summ.tile_distribution[k])]])

    eval_block = {
        "n_games": summ.n_games,
        "avg_score": summ.avg_score,
        "median_score": float(np.median(summ.scores)),
        "std_score": summ.std_score,
        "max_score": summ.max_score,
        "avg_max_tile": summ.avg_max_tile,
        "avg_steps": summ.avg_steps,
        "rate_512": float(np.mean(tiles >= 512)) if len(tiles) else 0.0,
        "rate_1024": float(np.mean(tiles >= 1024)) if len(tiles) else 0.0,
        "rate_2048": float(np.mean(tiles >= 2048)) if len(tiles) else 0.0,
        "max_tile_best": int(tiles.max()) if len(tiles) else 0,
        "tile_distribution": {int(k): int(v) for k, v in summ.tile_distribution.items()},
        "scores": [int(s) for s in summ.scores],
    }

    # 合并到 summary.json
    data = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    sup = data.get("supervised", {})
    sup["model"] = args.model
    sup["eval"] = eval_block
    data["supervised"] = sup
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写入 {SUMMARY_PATH}")

    # 直接画图
    plot_score_hist(summ.scores, "监督学习网络对局分数分布",
                    "results/supervised/policy_score_hist.png", bins=15)
    plot_tile_distribution(summ.tile_distribution,
                           "监督学习网络最大方块分布",
                           "results/supervised/policy_tile_dist.png")


if __name__ == "__main__":
    main()
