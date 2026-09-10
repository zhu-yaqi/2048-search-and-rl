"""第 3 部分：用遗传算法优化启发式权重，保存最优权重并绘制进化曲线。

用法：
    python scripts/run_evolution.py --pop 20 --gens 15 --games 4
"""
import argparse
import os

import _bootstrap  # noqa: F401

import numpy as np

from src.evolution.ga import GeneticOptimizer, GAConfig
from src.heuristic.evaluation import DEFAULT_WEIGHTS, FEATURE_NAMES
from src.heuristic.search import GreedyAgent
from src.common.runner import evaluate_agent
from src.common.plotting import plot_curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--gens", type=int, default=15)
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="results/evolution")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = GAConfig(pop_size=args.pop, n_generations=args.gens,
                   games_per_eval=args.games, seed=args.seed)
    opt = GeneticOptimizer(cfg)
    best_weights, hist = opt.run(verbose=True)

    np.save(os.path.join(args.outdir, "best_weights.npy"), best_weights)
    print("\n最优权重:")
    for name, w in zip(FEATURE_NAMES, best_weights):
        print(f"  {name:14s}: {w:+.3f}")

    gens = list(range(1, len(hist.best_fitness) + 1))
    plot_curves(gens, {"每代最优": hist.best_fitness, "每代平均": hist.avg_fitness},
                "遗传算法进化曲线（适应度=贪心平均得分）", "代数", "平均得分",
                os.path.join(args.outdir, "ga_curve.png"))

    # 对比：默认权重 vs 进化权重（更多局，更稳）
    print("\n=== 进化前后对比(贪心策略, 各 30 局) ===")
    print("默认权重:")
    base = evaluate_agent(GreedyAgent(DEFAULT_WEIGHTS), n_games=30)
    print(base)
    print("进化权重:")
    evolved = evaluate_agent(GreedyAgent(best_weights), n_games=30)
    print(evolved)


if __name__ == "__main__":
    main()
