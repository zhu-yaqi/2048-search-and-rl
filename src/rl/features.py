"""强化学习专用棋盘特征。

DQN 本质上只要求 Q 网络是可微函数近似器，不限定必须使用 CNN。2048 的棋盘只有
4×4，空格数、单调性、平滑性、蛇形排列、可合并数等结构特征对长期收益很关键。
本文件把这些信息编码成紧凑向量，用于更快、更稳定的 Dueling Double DQN 训练。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..game.fast import ACTIONS, move_exp

# 8 种蛇形模板：四个角起步 + 左右/上下翻转。数值越大的格子越希望放大块。
_BASE_SNAKE = np.array([
    [15, 14, 13, 12],
    [8, 9, 10, 11],
    [7, 6, 5, 4],
    [0, 1, 2, 3],
], dtype=np.float32)
_SNAKES = []
for m in (_BASE_SNAKE, np.fliplr(_BASE_SNAKE), np.flipud(_BASE_SNAKE), np.flipud(np.fliplr(_BASE_SNAKE))):
    _SNAKES.append(m)
    _SNAKES.append(m.T)
SNAKE_TEMPLATES = np.stack(_SNAKES, axis=0) / 15.0

# 与启发式基线同方向的势函数权重；仅用于 reward shaping 和安全护栏的叶节点估值。
POTENTIAL_WEIGHTS = np.array([0.70, 0.12, 0.04, 0.35, 1.20, 0.20], dtype=np.float32)

# 特征维度：16 指数 + 16 是否为空 + 15 全局结构 + 8 蛇形分数 + 4 合法动作。
FEATURE_DIM = 59


def _legal_mask(E: np.ndarray) -> np.ndarray:
    out = np.zeros(4, dtype=np.float32)
    for a in ACTIONS:
        _, _, moved = move_exp(E, int(a))
        out[int(a)] = 1.0 if moved else 0.0
    return out


def legal_mask_batch(exp_boards: np.ndarray) -> np.ndarray:
    boards = np.asarray(exp_boards, dtype=np.uint8)
    mask = np.zeros((len(boards), 4), dtype=bool)
    for i, E in enumerate(boards):
        for a in ACTIONS:
            _, _, moved = move_exp(E, int(a))
            if moved:
                mask[i, int(a)] = True
    return mask


def _structure_features(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回全局结构特征和蛇形模板特征。"""
    Ef = E.astype(np.float32)
    nonzero = Ef > 0
    empty = float(np.count_nonzero(E == 0))
    max_exp = float(E.max())
    total_exp = float(Ef.sum())
    mean_exp = total_exp / 16.0
    nonzero_count = max(1.0, float(np.count_nonzero(nonzero)))
    mean_nonzero = total_exp / nonzero_count

    # 单调性：分别保留横向/纵向的最佳单调方向，越接近 0 越好。
    dh = Ef[:, 1:] - Ef[:, :-1]
    dv = Ef[1:, :] - Ef[:-1, :]
    inc_r = np.where(dh > 0, dh, 0.0).sum(axis=1)
    dec_r = np.where(dh < 0, -dh, 0.0).sum(axis=1)
    inc_c = np.where(dv > 0, dv, 0.0).sum(axis=0)
    dec_c = np.where(dv < 0, -dv, 0.0).sum(axis=0)
    mono_row = -float(np.minimum(inc_r, dec_r).sum())
    mono_col = -float(np.minimum(inc_c, dec_c).sum())
    mono_total = mono_row + mono_col

    # 平滑性：只统计相邻非空格。
    mask_h = (E[:, 1:] > 0) & (E[:, :-1] > 0)
    mask_v = (E[1:, :] > 0) & (E[:-1, :] > 0)
    smooth_h = -float(np.abs(dh)[mask_h].sum())
    smooth_v = -float(np.abs(dv)[mask_v].sum())
    smooth_total = smooth_h + smooth_v

    merges = float(((E[:, 1:] == E[:, :-1]) & (E[:, :-1] > 0)).sum() + ((E[1:, :] == E[:-1, :]) & (E[:-1, :] > 0)).sum())

    max_pos = np.unravel_index(int(np.argmax(E)), E.shape)
    corner = 1.0 if tuple(max_pos) in {(0, 0), (0, 3), (3, 0), (3, 3)} else 0.0
    edge = 1.0 if (max_pos[0] in (0, 3) or max_pos[1] in (0, 3)) else 0.0

    snake_scores = (Ef[None, :, :] * SNAKE_TEMPLATES).sum(axis=(1, 2)) / (15.0 * 16.0)
    snake_best = float(snake_scores.max())

    # 归一化到相近尺度，便于 MLP/Q 学习。
    global_feats = np.array([
        empty / 16.0,
        max_exp / 15.0,
        mean_exp / 15.0,
        mean_nonzero / 15.0,
        np.log2(float((1 << E[E > 0].astype(np.int64)).sum()) + 1.0) / 16.0 if np.any(E > 0) else 0.0,
        mono_row / 64.0,
        mono_col / 64.0,
        mono_total / 128.0,
        smooth_h / 96.0,
        smooth_v / 96.0,
        smooth_total / 192.0,
        merges / 24.0,
        corner,
        edge,
        snake_best,
    ], dtype=np.float32)
    return global_feats, snake_scores.astype(np.float32)


def extract_feature_exp(E: np.ndarray) -> np.ndarray:
    """单个指数棋盘 -> DQN 特征向量。"""
    E = np.asarray(E, dtype=np.uint8)
    exp_flat = E.reshape(-1).astype(np.float32) / 15.0
    empty_flat = (E.reshape(-1) == 0).astype(np.float32)
    global_feats, snake_scores = _structure_features(E)
    legal = _legal_mask(E)
    feat = np.concatenate([exp_flat, empty_flat, global_feats, snake_scores, legal], axis=0).astype(np.float32)
    if feat.shape[0] != FEATURE_DIM:
        raise RuntimeError(f"FEATURE_DIM 不一致: {feat.shape[0]} != {FEATURE_DIM}")
    return feat


def extract_feature_batch_exp(exp_boards: np.ndarray) -> np.ndarray:
    boards = np.asarray(exp_boards, dtype=np.uint8)
    out = np.empty((len(boards), FEATURE_DIM), dtype=np.float32)
    for i, E in enumerate(boards):
        out[i] = extract_feature_exp(E)
    return out


def board_potential_exp(E: np.ndarray) -> float:
    """2048 势函数 Φ(s)：越大代表局面越有长期潜力。

    这是 reward shaping 使用的状态势能，而不是最终策略本身。它鼓励 RL 在获得分数
    的同时保留空格、维持单调/平滑结构，并把最大块固定在角落附近。
    """
    E = np.asarray(E, dtype=np.uint8)
    global_feats, snake_scores = _structure_features(E)
    empty = global_feats[0] * 16.0
    max_exp = global_feats[1] * 15.0
    mono = global_feats[7] * 128.0
    smooth = global_feats[10] * 192.0
    merges = global_feats[11] * 24.0
    corner = global_feats[12]
    snake_best = float(snake_scores.max()) * 10.0
    # mono/smooth 为负值，正权重会惩罚破坏结构。
    return float(
        POTENTIAL_WEIGHTS[0] * empty
        + POTENTIAL_WEIGHTS[1] * mono
        + POTENTIAL_WEIGHTS[2] * smooth
        + POTENTIAL_WEIGHTS[3] * max_exp
        + POTENTIAL_WEIGHTS[4] * corner
        + POTENTIAL_WEIGHTS[5] * merges
        + 0.65 * snake_best
    )
