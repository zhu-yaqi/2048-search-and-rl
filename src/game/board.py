"""2048 游戏核心引擎。

棋盘使用 4x4 的 numpy 整数矩阵表示，0 表示空格，其余为方块上的数值
(2, 4, 8, ...)。本模块只负责"游戏规则"本身（移动、合并、生成新块、判定
结束、计分），不包含任何 AI 策略，保证规则与智能体解耦。

四个动作统一编码为：
    0 = UP(上)   1 = DOWN(下)   2 = LEFT(左)   3 = RIGHT(右)

实现要点：把任意方向的移动都归约为"向左移动"这一基本操作，再通过转置/
翻转把棋盘旋转回去，这样只需要正确实现一行的滑动合并即可，逻辑清晰且不易出错。
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np

# 动作编码
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_NAMES = {UP: "上", DOWN: "下", LEFT: "左", RIGHT: "右"}

SIZE = 4


def _compress_merge_left(row: np.ndarray) -> Tuple[np.ndarray, int]:
    """对一行(长度 4)执行"向左滑动 + 合并"，返回(新行, 本行得分)。

    规则：先把非零块压紧到左侧，相邻且相等的块合并为一个翻倍块并加分，
    每个块在一次移动中最多只能参与一次合并。
    """
    tiles = [int(x) for x in row if x != 0]
    merged: List[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            new_val = tiles[i] * 2
            merged.append(new_val)
            gained += new_val
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (SIZE - len(merged)))
    return np.array(merged, dtype=np.int64), gained


def _move_grid(grid: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
    """对整盘执行一次移动，返回(新棋盘, 得分, 是否发生变化)。

    通过翻转/转置把四个方向统一归约为"向左"。
    """
    g = grid.copy()
    if action == LEFT:
        work = g
    elif action == RIGHT:
        work = np.fliplr(g)
    elif action == UP:
        work = g.T
    elif action == DOWN:
        work = np.fliplr(g.T)
    else:
        raise ValueError(f"非法动作: {action}")

    gained = 0
    new_rows = []
    for r in range(SIZE):
        new_row, g_score = _compress_merge_left(work[r])
        new_rows.append(new_row)
        gained += g_score
    new_work = np.stack(new_rows, axis=0)

    # 逆变换还原方向
    if action == LEFT:
        result = new_work
    elif action == RIGHT:
        result = np.fliplr(new_work)
    elif action == UP:
        result = new_work.T
    else:  # DOWN
        result = np.fliplr(new_work).T

    moved = not np.array_equal(result, grid)
    return result, gained, moved


class Board:
    """2048 棋盘与游戏状态。"""

    def __init__(self, grid: Optional[np.ndarray] = None, score: int = 0,
                 rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        if grid is None:
            self.grid = np.zeros((SIZE, SIZE), dtype=np.int64)
            self.score = 0
            # 初始随机生成 2 个方块
            self.spawn_tile()
            self.spawn_tile()
        else:
            self.grid = np.array(grid, dtype=np.int64)
            self.score = score

    # ---------- 基础操作 ----------
    def clone(self) -> "Board":
        """深拷贝当前局面（不共享随机数生成器，便于搜索模拟）。"""
        b = Board.__new__(Board)
        b.grid = self.grid.copy()
        b.score = self.score
        b.rng = self.rng
        return b

    def empty_cells(self) -> List[Tuple[int, int]]:
        cells = np.argwhere(self.grid == 0)
        return [tuple(c) for c in cells]

    def spawn_tile(self) -> bool:
        """在空位随机生成新块：2 的概率 90%，4 的概率 10%。返回是否成功。"""
        empties = self.empty_cells()
        if not empties:
            return False
        idx = self.rng.integers(len(empties))
        r, c = empties[idx]
        self.grid[r, c] = 2 if self.rng.random() < 0.9 else 4
        return True

    def move(self, action: int, spawn: bool = True) -> Tuple[bool, int]:
        """执行一次玩家动作。

        参数 spawn 为 True 时，在有效移动后自动生成一个新方块。
        返回(是否为有效移动, 本步得分)。
        """
        new_grid, gained, moved = _move_grid(self.grid, action)
        if not moved:
            return False, 0
        self.grid = new_grid
        self.score += gained
        if spawn:
            self.spawn_tile()
        return True, gained

    # ---------- 查询 ----------
    def available_moves(self) -> List[int]:
        """返回当前所有"能产生变化"的合法动作。"""
        valid = []
        for a in ACTIONS:
            _, _, moved = _move_grid(self.grid, a)
            if moved:
                valid.append(a)
        return valid

    def is_game_over(self) -> bool:
        """无空位且四个方向都无法移动/合并时游戏结束。"""
        if len(self.empty_cells()) > 0:
            return False
        return len(self.available_moves()) == 0

    def max_tile(self) -> int:
        return int(self.grid.max())

    def __repr__(self) -> str:
        lines = [f"Score: {self.score}  Max: {self.max_tile()}"]
        for r in range(SIZE):
            lines.append(" ".join(f"{int(v):5d}" if v else "    ." for v in self.grid[r]))
        return "\n".join(lines)


# ---------- 供搜索/环境使用的纯函数接口（不依赖 Board 实例，便于高效模拟） ----------

def simulate_move(grid: np.ndarray, action: int) -> Tuple[np.ndarray, int, bool]:
    """纯函数式移动：不生成新块，返回(新棋盘, 得分, 是否变化)。"""
    return _move_grid(grid, action)


def is_terminal(grid: np.ndarray) -> bool:
    if np.any(grid == 0):
        return False
    for a in ACTIONS:
        _, _, moved = _move_grid(grid, a)
        if moved:
            return False
    return True
