"""对局运行与多局评估。

提供统一接口：给定一个"决策函数 / 智能体"，跑完整对局并统计分数、最大方块、
步数；以及对多局求平均，得到稳定的性能指标（平均分、最大方块分布、胜率等），
作为各方案横向比较的统一标准。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ..game.board import Board, ACTIONS, simulate_move

# 决策器：可以是带 select_action 方法的对象，也可以是 grid->action 的函数
Policy = Union["HasSelect", Callable[[Board], Optional[int]]]


@dataclass
class GameResult:
    score: int
    max_tile: int
    steps: int


@dataclass
class EvalSummary:
    n_games: int
    avg_score: float
    std_score: float
    max_score: int
    avg_max_tile: float
    avg_steps: float
    tile_distribution: Dict[int, int] = field(default_factory=dict)
    scores: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        dist = ", ".join(f"{k}:{v}" for k, v in sorted(self.tile_distribution.items()))
        return (
            f"局数={self.n_games} 平均分={self.avg_score:.1f}(±{self.std_score:.1f}) "
            f"最高分={self.max_score} 平均最大块={self.avg_max_tile:.0f} "
            f"平均步数={self.avg_steps:.1f}\n最大方块分布: {{{dist}}}"
        )


def _get_action(policy: Policy, board: Board) -> Optional[int]:
    if hasattr(policy, "select_action"):
        return policy.select_action(board)
    return policy(board)


def play_game(policy: Policy, seed: Optional[int] = None,
              max_steps: int = 800) -> GameResult:
    """用给定策略跑完一整局，返回结果。"""
    rng = np.random.default_rng(seed)
    board = Board(rng=rng)
    steps = 0
    while not board.is_game_over() and steps < max_steps:
        action = _get_action(policy, board)
        if action is None:
            break
        moved, _ = board.move(action)
        if not moved:
            # 策略给了非法动作：退而求其次选一个合法动作，避免死循环
            legal = board.available_moves()
            if not legal:
                break
            board.move(int(legal[0]))
        steps += 1
    return GameResult(score=board.score, max_tile=board.max_tile(), steps=steps)


def evaluate_agent(policy: Policy, n_games: int = 20, base_seed: int = 0,
                   verbose: bool = False) -> EvalSummary:
    """多局评估，返回平均统计。"""
    results: List[GameResult] = []
    for i in range(n_games):
        r = play_game(policy, seed=base_seed + i)
        results.append(r)
        if verbose:
            print(f"  game {i + 1}/{n_games}: score={r.score} max={r.max_tile} steps={r.steps}")
    scores = [r.score for r in results]
    tiles = [r.max_tile for r in results]
    dist: Dict[int, int] = {}
    for t in tiles:
        dist[t] = dist.get(t, 0) + 1
    return EvalSummary(
        n_games=n_games,
        avg_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        max_score=int(np.max(scores)),
        avg_max_tile=float(np.mean(tiles)),
        avg_steps=float(np.mean([r.steps for r in results])),
        tile_distribution=dist,
        scores=scores,
    )
