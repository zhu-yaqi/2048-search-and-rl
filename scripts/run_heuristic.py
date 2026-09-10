"""评估第 2 部分的启发式策略（贪心 与 Expectimax），输出统计与分数分布图。

用法：
    python scripts/run_heuristic.py --games 30
    python scripts/run_heuristic.py --games 10 --expectimax-depth 2
"""
import argparse
import os

import _bootstrap  # noqa: F401

import numpy as np

from src.heuristic.search import GreedyAgent, ExpectimaxAgent
from src.common.runner import evaluate_agent
from src.common.plotting import plot_score_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--expectimax-games", type=int, default=10)
    p.add_argument("--expectimax-depth", type=int, default=2)
    p.add_argument("--weights", default=None, help="可选: 进化得到的权重 .npy")
    p.add_argument("--outdir", default="results/heuristic")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    weights = np.load(args.weights) if args.weights else None

    print("=== 随机策略基线 ===")
    rng_summ = evaluate_agent(lambda b: int(np.random.choice(b.available_moves())),
                              n_games=args.games)
    print(rng_summ)

    print("\n=== 贪心(一步前瞻)策略 ===")
    greedy = GreedyAgent(weights=weights) if weights is not None else GreedyAgent()
    g_summ = evaluate_agent(greedy, n_games=args.games, verbose=True)
    print(g_summ)
    plot_score_hist(g_summ.scores, "贪心策略分数分布",
                    os.path.join(args.outdir, "greedy_hist.png"))

    print(f"\n=== Expectimax(depth={args.expectimax_depth})策略 ===")
    exp = (ExpectimaxAgent(depth=args.expectimax_depth, weights=weights)
           if weights is not None else ExpectimaxAgent(depth=args.expectimax_depth))
    e_summ = evaluate_agent(exp, n_games=args.expectimax_games, verbose=True)
    print(e_summ)
    plot_score_hist(e_summ.scores, f"Expectimax(depth={args.expectimax_depth})分数分布",
                    os.path.join(args.outdir, "expectimax_hist.png"))


if __name__ == "__main__":
    main()
