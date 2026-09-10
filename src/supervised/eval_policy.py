"""真实对局评估工具。

不要只看 val_acc。2048 的监督学习最终一定要看：
1. 平均分
2. 中位数分
3. 最大方块
4. 到达 512/1024/2048 的比例
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from ..game.board import Board


def evaluate_agent(agent, n_games: int = 100, base_seed: int = 10000, verbose: bool = True) -> Dict[str, Any]:
    scores = []
    max_tiles = []
    steps_list = []

    for i in range(n_games):
        board = Board(rng=np.random.default_rng(base_seed + i))
        steps = 0

        while not board.is_game_over():
            action = agent.select_action(board)
            if action is None:
                break
            moved = board.move(action)
            if not moved:
                break
            steps += 1

        scores.append(board.score)
        max_tiles.append(board.max_tile())
        steps_list.append(steps)

    scores_np = np.array(scores, dtype=np.float64)
    tiles_np = np.array(max_tiles, dtype=np.int64)
    steps_np = np.array(steps_list, dtype=np.float64)

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
    }

    if verbose:
        print("[eval]", result, flush=True)

    return result
