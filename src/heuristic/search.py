"""基于启发式评价函数的决策智能体（运行在高性能指数引擎上）。

包含两种策略：
1. GreedyAgent —— 一步前瞻(1-ply)：分别模拟上/下/左/右后得到的局面，用评价
   函数打分，选择评分最高的动作。计算极快，用于进化算法的适应度评估。
2. ExpectimaxAgent —— 期望最大化搜索：把环境随机生成新块建模为"机会节点"，
   智能体节点取最大、机会节点取期望，向前搜索若干层。它能更好地处理 2048 的
   随机性，是更强的决策策略，同时作为监督学习的"专家"。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..game.board import Board
from ..game.fast import ACTIONS, move_exp, is_terminal_exp, to_exp
from .evaluation import DEFAULT_WEIGHTS, evaluate_exp, build_line_table, evaluate_fast
from .evaluator import evaluate_board


class HeuristicAgent:
    name = "heuristic"

    def select_action(self, board: Board) -> Optional[int]:
        raise NotImplementedError


class GreedyAgent(HeuristicAgent):
    """一步前瞻贪心策略。score_weight 控制立即得分在评价中的占比。"""

    name = "greedy"

    def __init__(self, weights: np.ndarray = DEFAULT_WEIGHTS, score_weight: float = 0.0):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.score_weight = score_weight

    def select_action(self, board: Board) -> Optional[int]:
        E = to_exp(board.grid)
        best_a, best_v = None, -np.inf
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, a)
            if not moved:
                continue
            v = evaluate_exp(newE, self.weights) + self.score_weight * gained
            if v > best_v:
                best_v, best_a = v, a
        return best_a


class ExpectimaxAgent(HeuristicAgent):
    """期望最大化搜索策略（强基线 + 监督学习专家）。

    参数：
        depth: 搜索的最大深度。
        prob_cutoff: 机会节点剪枝阈值，累计概率过小的分支不再展开。
        adaptive: 是否根据空格数量自适应调整深度（空格越少越往深处搜）。
    """

    name = "expectimax"

    def __init__(self, weights: np.ndarray = DEFAULT_WEIGHTS, depth: int = 2,
                 prob_cutoff: float = 0.003, adaptive: bool = True,
                 score_weight: float = 0.0):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.depth = depth
        self.prob_cutoff = prob_cutoff
        self.adaptive = adaptive
        self.score_weight = score_weight
        # 为当前固定权重预构建逐行评分表，加速叶子评价
        self._line_table = build_line_table(self.weights)

    def _exp_to_grid(self, E: np.ndarray) -> np.ndarray:
        grid = np.zeros_like(E, dtype=np.int64)
        mask = E > 0
        grid[mask] = 2 ** E[mask]
        return grid


    def _eval(self, E: np.ndarray) -> float:
        base_score = evaluate_fast(E, self._line_table, self.weights)

        grid = self._exp_to_grid(E)
        structure_score = evaluate_board(grid)

        # 关键：不要让你新增的角落/蛇形评分把原评分完全盖掉
        return base_score + 0.05 * structure_score




    def _max_value(self, E: np.ndarray, depth: int, prob: float) -> float:
        if depth <= 0 or is_terminal_exp(E):
            return self._eval(E)
        best = -np.inf
        any_move = False
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, a)
            if not moved:
                continue
            any_move = True
            v = self.score_weight * gained + self._chance_value(newE, depth, prob)
            if v > best:
                best = v
        if not any_move:
            return self._eval(E)
        return best

    def _chance_value(self, E: np.ndarray, depth: int, prob: float) -> float:
        empties = np.argwhere(E == 0)
        n = len(empties)
        if n == 0:
            return self._eval(E)
        branch_prob = prob / n
        if branch_prob < self.prob_cutoff:
            return self._eval(E)
        ev = 0.0
        for (r, c) in empties:
            for val_exp, p in ((1, 0.9), (2, 0.1)):  # 指数1=2, 指数2=4
                E2 = E.copy()
                E2[r, c] = val_exp
                ev += (p / n) * self._max_value(E2, depth - 1, branch_prob * p)
        return ev

    def _current_depth(self, E: np.ndarray) -> int:
        if not self.adaptive:
            return self.depth
        empties = int(np.count_nonzero(E == 0))
        if empties >= 6:
            return max(1, self.depth - 1)
        if empties >= 3:
            return self.depth
        return self.depth + 1

    def select_action(self, board: Board) -> Optional[int]:
        E = to_exp(board.grid)
        depth = self._current_depth(E)
        best_a, best_v = None, -np.inf
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, a)
            if not moved:
                continue
            v = self.score_weight * gained + self._chance_value(newE, depth, 1.0)
            if v > best_v:
                best_v, best_a = v, a
        return best_a
