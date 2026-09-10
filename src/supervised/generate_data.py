"""监督学习 v2 数据生成。

相比原始版本，这个版本不只保存一个硬标签 y，还额外保存：
1. target_probs: Expectimax 对四个动作的软标签分布。
2. legal_mask: 当前局面哪些动作合法。
3. game_id: 每条样本来自哪一局，用于按整局划分训练/验证集。
4. expert_scores: 专家对四个动作的原始评分，方便分析。

同时支持 DAgger 式扰动采样：
    大部分时间按专家动作走；
    少量时间随机走一个合法动作；
    但每个状态的标签仍然由专家标注。
这样训练集会包含更多“模型走歪后可能遇到的状态”，比纯专家轨迹更稳。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np

from ..game.board import Board
from ..common.encoding import encode_board
from ..heuristic.search import ExpectimaxAgent
from ..game.fast import ACTIONS, move_exp, to_exp


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _softmax_legal(scores: np.ndarray, legal_mask: np.ndarray, temperature: float = 0.35) -> np.ndarray:
    """只在合法动作上做 softmax。"""
    probs = np.zeros_like(scores, dtype=np.float64)
    legal_idx = np.where(legal_mask > 0)[0]
    if len(legal_idx) == 0:
        return probs

    s = scores[legal_idx].astype(np.float64)
    s = s / max(temperature, 1e-6)
    s = s - np.max(s)
    e = np.exp(s)
    probs[legal_idx] = e / np.sum(e)
    return probs


def expert_action_scores(expert: ExpectimaxAgent, board: Board) -> Tuple[np.ndarray, np.ndarray]:
    """返回专家对四个动作的评分和合法动作 mask。

    这里复用 ExpectimaxAgent 内部搜索逻辑，和 select_action 的评分保持一致。
    """
    E = to_exp(board.grid)
    depth = expert._current_depth(E)

    scores = np.full(4, -1e9, dtype=np.float64)
    legal_mask = np.zeros(4, dtype=np.float32)

    for a in ACTIONS:
        newE, gained, moved = move_exp(E, a)
        if not moved:
            continue

        legal_mask[int(a)] = 1.0
        scores[int(a)] = expert.score_weight * gained + expert._chance_value(newE, depth, 1.0)

    return scores, legal_mask


def generate_dataset(
    n_samples: int = 50000,
    expert_depth: int = 3,
    out_path: Optional[str] = None,
    base_seed: int = 0,
    verbose: bool = True,
    temperature: float = 0.35,
    explore_prob: float = 0.07,
    max_steps_per_game: int = 5000,
    return_extras: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """生成监督学习数据。

    返回：
        X: (N, 16, 4, 4)
        y: (N,)
        extras 可选：
            target_probs: (N, 4)
            legal_mask: (N, 4)
            game_id: (N,)
            expert_scores: (N, 4)

    参数建议：
        n_samples=80000 起步，效果比 15000 稳很多。
        expert_depth=3 起步，如果太慢再降到 2。
        explore_prob=0.05~0.10，覆盖非专家状态。
    """
    rng = np.random.default_rng(base_seed)
    expert = ExpectimaxAgent(depth=expert_depth)

    X_list = []
    y_list = []
    target_probs_list = []
    legal_mask_list = []
    scores_list = []
    game_id_list = []

    game_idx = 0

    while len(X_list) < n_samples:
        board = Board(rng=np.random.default_rng(base_seed + game_idx))
        steps = 0

        while (
            not board.is_game_over()
            and len(X_list) < n_samples
            and steps < max_steps_per_game
        ):
            scores, legal_mask = expert_action_scores(expert, board)
            legal = np.where(legal_mask > 0)[0]

            if len(legal) == 0:
                break

            best_action = int(legal[np.argmax(scores[legal])])
            target_probs = _softmax_legal(scores, legal_mask, temperature=temperature)

            X_list.append(encode_board(board.grid))
            y_list.append(best_action)
            target_probs_list.append(target_probs.astype(np.float32))
            legal_mask_list.append(legal_mask.astype(np.float32))
            scores_list.append(scores.astype(np.float32))
            game_id_list.append(game_idx)

            # DAgger 式扰动：少量时间不走专家最优，用于收集“走歪后”的局面。
            if rng.random() < explore_prob:
                action = int(rng.choice(legal))
            else:
                action = best_action

            board.move(action)
            steps += 1

        if verbose:
            print(
                f"  [data-v2] 已完成 {game_idx + 1} 局, "
                f"累计样本 {len(X_list)}/{n_samples}, "
                f"本局 score={board.score}, max={board.max_tile()}",
                flush=True,
            )

        game_idx += 1

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    extras = {
        "target_probs": np.stack(target_probs_list).astype(np.float32),
        "legal_mask": np.stack(legal_mask_list).astype(np.float32),
        "game_id": np.array(game_id_list, dtype=np.int64),
        "expert_scores": np.stack(scores_list).astype(np.float32),
    }

    if out_path:
        dir_name = os.path.dirname(out_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        np.savez_compressed(out_path, X=X, y=y)


    if return_extras:
        return X, y, extras

    return X, y
