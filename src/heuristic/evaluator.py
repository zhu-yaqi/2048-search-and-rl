from __future__ import annotations

import numpy as np


_BASE_SNAKE = np.array([
    [15, 14, 13, 12],
    [8,   9, 10, 11],
    [7,   6,  5,  4],
    [0,   1,  2,  3],
], dtype=np.float64)


SNAKE_WEIGHTS = [
    _BASE_SNAKE,
    np.fliplr(_BASE_SNAKE),
    np.flipud(_BASE_SNAKE),
    np.flipud(np.fliplr(_BASE_SNAKE)),
]


def log_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.float64)
    mask = grid > 0
    out[mask] = np.log2(grid[mask])
    return out


def empty_score(grid: np.ndarray) -> float:
    return float(np.sum(grid == 0))


def corner_score(grid: np.ndarray) -> float:
    max_tile = np.max(grid)
    if max_tile <= 0:
        return 0.0

    corners = [
        grid[0, 0],
        grid[0, 3],
        grid[3, 0],
        grid[3, 3],
    ]

    value = float(np.log2(max_tile))

    if max_tile in corners:
        return value
    else:
        return -value


def snake_score(grid: np.ndarray) -> float:
    lg = log_grid(grid)
    return max(float(np.sum(lg * w)) for w in SNAKE_WEIGHTS)


def smoothness_score(grid: np.ndarray) -> float:
    lg = log_grid(grid)
    penalty = 0.0

    for r in range(4):
        for c in range(4):
            if lg[r, c] == 0:
                continue

            if r + 1 < 4 and lg[r + 1, c] > 0:
                penalty -= abs(lg[r, c] - lg[r + 1, c])

            if c + 1 < 4 and lg[r, c + 1] > 0:
                penalty -= abs(lg[r, c] - lg[r, c + 1])

    return float(penalty)


def merge_score(grid: np.ndarray) -> float:
    score = 0.0

    for r in range(4):
        for c in range(4):
            v = grid[r, c]
            if v == 0:
                continue

            if r + 1 < 4 and grid[r + 1, c] == v:
                score += np.log2(v)

            if c + 1 < 4 and grid[r, c + 1] == v:
                score += np.log2(v)

    return float(score)


def evaluate_board(grid: np.ndarray) -> float:
    """
    温和版结构评分：
    1. 空格多更好
    2. 最大块在角落更好
    3. 蛇形递减结构更好
    4. 相邻数字平滑更好
    5. 可合并数字更多更好
    """
    return (
        80.0 * empty_score(grid)
        + 120.0 * corner_score(grid)
        + 1.0 * snake_score(grid)
        + 20.0 * smoothness_score(grid)
        + 40.0 * merge_score(grid)
    )
