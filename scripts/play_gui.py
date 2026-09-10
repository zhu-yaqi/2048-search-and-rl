"""启动 2048 图形界面。

用法示例：
    python scripts/play_gui.py                 # 人类游玩(方向键)
    python scripts/play_gui.py --agent greedy  # 观看贪心 AI 自动演示
    python scripts/play_gui.py --agent expectimax --depth 2
    python scripts/play_gui.py --agent net --model results/supervised/policy.pt
"""
import argparse

import _bootstrap  # noqa: F401

import numpy as np

from src.game.gui import Game2048GUI
from src.heuristic.search import GreedyAgent, ExpectimaxAgent


def build_agent(args):
    if args.agent == "human":
        return None
    if args.agent == "greedy":
        weights = np.load(args.weights) if args.weights else None
        return GreedyAgent(weights=weights) if weights is not None else GreedyAgent()
    if args.agent == "expectimax":
        weights = np.load(args.weights) if args.weights else None
        kw = {"depth": args.depth}
        if weights is not None:
            kw["weights"] = weights
        return ExpectimaxAgent(**kw)
    if args.agent == "net":
        from src.supervised.model import load_net, NetAgent
        return NetAgent(load_net(args.model))
    raise ValueError(args.agent)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="human",
                   choices=["human", "greedy", "expectimax", "net"])
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--weights", default=None, help="启发式权重 .npy 路径(可选)")
    p.add_argument("--model", default="results/supervised/policy.pt")
    p.add_argument("--delay", type=int, default=120, help="AI 每步间隔毫秒")
    args = p.parse_args()

    agent = build_agent(args)
    gui = Game2048GUI(agent=agent, ai_delay_ms=args.delay)
    gui.run()


if __name__ == "__main__":
    main()
