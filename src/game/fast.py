"""高性能 2048 引擎（指数表示 + 行查找表）。

为支撑 Expectimax 深层搜索、进化算法适应度评估、强化学习海量 rollout，纯
Python 的逐格循环太慢。这里采用 2048 高性能实现的经典技巧：

1. 用"指数表示"代替数值：方块 2^k 用整数 k 表示（空格为 0）。这样一行 4 个
   格子可编码为一个 16 位整数（每格 4 bit）。
2. 预先枚举所有 65536 种行，离线计算其"向左滑动+合并"的结果行与得分，存成
   查找表。运行时一次移动只需对 4 行做表查询，再用转置/翻转把四个方向归约为
   "向左"，整体由 numpy 向量化完成，速度比逐格循环快一到两个数量级。

棋盘在本模块内统一用 4x4 的 uint8 指数矩阵 E 表示。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = (UP, DOWN, LEFT, RIGHT)
SIZE = 4
MAX_EXP = 15  # 支持到 2^15


def _build_tables():
    """枚举所有行，构建向左移动的结果表与得分表。"""
    n = 1 << 16
    result = np.zeros(n, dtype=np.uint16)
    score = np.zeros(n, dtype=np.float64)
    moved = np.zeros(n, dtype=bool)
    for idx in range(n):
        cells = [(idx >> (4 * k)) & 0xF for k in range(4)]
        tiles = [c for c in cells if c != 0]
        merged = []
        s = 0.0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                nv = tiles[i] + 1  # 指数 +1 即数值翻倍
                merged.append(nv)
                s += float(1 << nv)  # 合并所得数值 2^nv
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        merged.extend([0] * (4 - len(merged)))
        out_idx = merged[0] | (merged[1] << 4) | (merged[2] << 8) | (merged[3] << 12)
        result[idx] = out_idx
        score[idx] = s
        moved[idx] = (out_idx != idx)
    return result, score, moved


LEFT_RESULT, LEFT_SCORE, LEFT_MOVED = _build_tables()


def to_exp(value_grid: np.ndarray) -> np.ndarray:
    """数值棋盘 -> 指数矩阵。"""
    E = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask = value_grid > 0
    E[mask] = np.log2(value_grid[mask]).astype(np.uint8)
    return E


def to_value(E: np.ndarray) -> np.ndarray:
    """指数矩阵 -> 数值棋盘。"""
    V = np.zeros((SIZE, SIZE), dtype=np.int64)
    mask = E > 0
    V[mask] = (1 << E[mask].astype(np.int64))
    return V


def _row_indices(work: np.ndarray) -> np.ndarray:
    w = work.astype(np.uint32)
    return (w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12))


def _move_left_exp(work: np.ndarray) -> Tuple[np.ndarray, float]:
    idx = _row_indices(work)
    res = LEFT_RESULT[idx]
    score = float(LEFT_SCORE[idx].sum())
    newE = np.empty((SIZE, SIZE), dtype=np.uint8)
    newE[:, 0] = res & 0xF
    newE[:, 1] = (res >> 4) & 0xF
    newE[:, 2] = (res >> 8) & 0xF
    newE[:, 3] = (res >> 12) & 0xF
    return newE, score


def move_exp(E: np.ndarray, action: int) -> Tuple[np.ndarray, float, bool]:
    """对指数矩阵执行一次移动，返回(新矩阵, 得分, 是否变化)。"""
    if action == LEFT:
        work = E
    elif action == RIGHT:
        work = np.fliplr(E)
    elif action == UP:
        work = E.T
    elif action == DOWN:
        work = np.fliplr(E.T)
    else:
        raise ValueError(action)
    work = np.ascontiguousarray(work)
    new_work, score = _move_left_exp(work)
    if action == LEFT:
        result = new_work
    elif action == RIGHT:
        result = np.fliplr(new_work)
    elif action == UP:
        result = new_work.T
    else:
        result = np.fliplr(new_work).T
    result = np.ascontiguousarray(result)
    moved = not np.array_equal(result, E)
    return result, score, moved


def is_terminal_exp(E: np.ndarray) -> bool:
    if np.any(E == 0):
        return False
    for a in ACTIONS:
        _, _, moved = move_exp(E, a)
        if moved:
            return False
    return True


def spawn_exp(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在空位生成新块（2 占 90%，4 占 10%），返回新矩阵（原地修改副本）。"""
    empties = np.argwhere(E == 0)
    if len(empties) == 0:
        return E
    r, c = empties[rng.integers(len(empties))]
    E2 = E.copy()
    E2[r, c] = 1 if rng.random() < 0.9 else 2  # 指数 1=2, 2=4
    return E2
