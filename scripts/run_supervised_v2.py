"""运行监督学习 v2。

用法示例：

CPU:
    python scripts/run_supervised_v2.py --samples 80000 --epochs 50 --depth 3

GPU:
    python scripts/run_supervised_v2.py --samples 120000 --epochs 60 --depth 3 --device cuda

只评估更多局：
    python scripts/run_supervised_v2.py --samples 80000 --epochs 50 --eval-games 200
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 确保从项目根目录运行时能导入 src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.supervised.generate_data import generate_dataset
from src.supervised.train import TrainConfig, train_supervised
from src.supervised.model import NetAgent
from src.supervised.eval_policy import evaluate_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-out", type=str, default="data/supervised_v2_dataset.npz")
    parser.add_argument("--model-out", type=str, default="models/policy_supervised_v2.pt")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--explore-prob", type=float, default=0.07)
    parser.add_argument("--temperature", type=float, default=0.35)
    args = parser.parse_args()

    X, y, extras = generate_dataset(
        n_samples=args.samples,
        expert_depth=args.depth,
        out_path=args.data_out,
        base_seed=args.seed,
        verbose=True,
        temperature=args.temperature,
        explore_prob=args.explore_prob,
        return_extras=True,
    )

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        augment=not args.no_augment,
    )

    net, history = train_supervised(
        X,
        y,
        config=config,
        model_out=args.model_out,
        target_probs=extras["target_probs"],
        legal_mask=extras["legal_mask"],
        game_id=extras["game_id"],
    )

    agent = NetAgent(net, device=args.device)
    evaluate_agent(agent, n_games=args.eval_games, base_seed=10000, verbose=True)


if __name__ == "__main__":
    main()
