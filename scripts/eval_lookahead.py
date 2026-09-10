"""对比评估：纯监督网络 NetAgent vs 网络+前瞻 NetLookaheadAgent。

用法：
    python scripts/eval_lookahead.py --games 30 --rollout-len 8 --n-rollouts 3
    python scripts/eval_lookahead.py --games 30 --baseline-only   # 只跑纯网络基线
"""
import os
import sys
import time
import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.game.board import Board
from src.supervised.model import NetAgent, load_net
from src.supervised.lookahead import NetLookaheadAgent

DEFAULT_MODEL = "results/supervised/policy.pt"


def evaluate_agent(agent, n_games: int = 30, base_seed: int = 10000, verbose: bool = True):
    """对局评估，并保留每局原始分数/最大方块，便于画分布图。"""
    scores, max_tiles, steps_list = [], [], []
    for i in range(n_games):
        board = Board(rng=np.random.default_rng(base_seed + i))
        steps = 0
        while not board.is_game_over():
            action = agent.select_action(board)
            if action is None:
                break
            if not board.move(action):
                break
            steps += 1
        scores.append(int(board.score))
        max_tiles.append(int(board.max_tile()))
        steps_list.append(steps)

    scores_np = np.array(scores, dtype=np.float64)
    tiles_np = np.array(max_tiles, dtype=np.int64)
    steps_np = np.array(steps_list, dtype=np.float64)
    tile_distribution = {int(k): int(v) for k, v in sorted(Counter(max_tiles).items())}

    result = {
        "n_games": n_games,
        "avg_score": float(scores_np.mean()),
        "median_score": float(np.median(scores_np)),
        "max_score": int(scores_np.max()),
        "avg_steps": float(steps_np.mean()),
        "avg_max_tile": float(tiles_np.mean()),
        "rate_512": float(np.mean(tiles_np >= 512)),
        "rate_1024": float(np.mean(tiles_np >= 1024)),
        "rate_2048": float(np.mean(tiles_np >= 2048)),
        "max_tile_best": int(tiles_np.max()),
        "scores": scores,
        "max_tiles": max_tiles,
        "tile_distribution": tile_distribution,
    }
    if verbose:
        brief = {k: v for k, v in result.items()
                 if k not in ("scores", "max_tiles", "tile_distribution")}
        print("[eval]", brief, flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--rollout-len", type=int, default=8)
    p.add_argument("--n-rollouts", type=int, default=3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--base-seed", type=int, default=10000)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--lookahead-only", action="store_true")
    p.add_argument("--out", type=str, default="results/supervised/lookahead_compare.json")
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动切换到 CPU", flush=True)
        device = "cpu"

    net = load_net(args.model, device=device, channels=args.channels,
                   n_blocks=args.blocks, dropout=args.dropout)

    out = {"games": args.games, "base_seed": args.base_seed}

    if not args.lookahead_only:
        print("=" * 70)
        print(f"[基线] 纯 NetAgent，{args.games} 局", flush=True)
        print("=" * 70)
        t0 = time.time()
        base = evaluate_agent(NetAgent(net, device=device), n_games=args.games,
                              base_seed=args.base_seed, verbose=True)
        base["seconds"] = round(time.time() - t0, 1)
        out["baseline"] = base

    if not args.baseline_only:
        print("=" * 70)
        print(f"[增强] NetLookaheadAgent rollout_len={args.rollout_len} "
              f"n_rollouts={args.n_rollouts}，{args.games} 局", flush=True)
        print("=" * 70)
        t0 = time.time()
        la = NetLookaheadAgent(net, device=device, rollout_len=args.rollout_len,
                               n_rollouts=args.n_rollouts, seed=0)
        look = evaluate_agent(la, n_games=args.games, base_seed=args.base_seed, verbose=True)
        look["seconds"] = round(time.time() - t0, 1)
        out["lookahead"] = look

    if "baseline" in out and "lookahead" in out:
        b, l = out["baseline"], out["lookahead"]
        print("=" * 70)
        print("[对比汇总]")
        print(f"  平均分:   {b['avg_score']:.0f}  ->  {l['avg_score']:.0f}  "
              f"({(l['avg_score']/max(b['avg_score'],1)-1)*100:+.1f}%)")
        print(f"  平均最大块: {b['avg_max_tile']:.0f}  ->  {l['avg_max_tile']:.0f}")
        print(f"  到 1024 率: {b['rate_1024']*100:.0f}%  ->  {l['rate_1024']*100:.0f}%")
        print(f"  到 2048 率: {b['rate_2048']*100:.0f}%  ->  {l['rate_2048']*100:.0f}%")
        print("=" * 70)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
