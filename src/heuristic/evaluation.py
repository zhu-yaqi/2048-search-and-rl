"""启发式评价函数。

核心思想：把一个棋盘局面映射为若干个可解释的"特征"，再用一组权重做线性
加权求和，得到该局面的"好坏分数"。智能体在决策时模拟每个动作的后继局面并用
该函数打分，从而选择评分较高的动作。

之所以采用"特征 + 线性权重"的结构，是因为：
1. 每个特征都对应 2048 中公认有效的人类经验（保持空格、单调排列、相邻平滑、
   大数靠角等），方案具有可解释性；
2. 权重向量天然成为后续"进化计算"优化的对象（决策变量），便于第 3 部分自动调参。

所有特征都在 log2（指数）棋盘上计算，使不同量级的方块处于可比的数值尺度。
为提高速度，特征提取使用 numpy 向量化实现。
"""
from __future__ import annotations

import numpy as np

from ..game.fast import to_exp

# 特征顺序固定，权重向量与之一一对应
FEATURE_NAMES = [
    "empty",          # 空格数量（越多越灵活）
    "monotonicity",   # 单调性（行列尽量单调，<=0，越接近0越好）
    "smoothness",     # 平滑性（相邻方块数值接近，<=0，越接近0越好）
    "max_tile",       # 最大方块的 log2 值
    "corner",         # 最大方块是否位于角落（0/1）
    "merges",         # 当前可直接合并的相邻同值对数量
]

# 手工调好的一组默认权重（作为基线，也作为进化算法的初始参考）
DEFAULT_WEIGHTS = np.array([2.7, 1.0, 0.1, 1.0, 2.0, 0.7], dtype=np.float64)

TERMINAL_PENALTY = -1e6


def extract_features_exp(E: np.ndarray) -> np.ndarray:
    """从指数矩阵提取特征向量（向量化），顺序与 FEATURE_NAMES 一致。"""
    Ef = E.astype(np.float64)

    empty = float(np.count_nonzero(E == 0))

    # 单调性：行、列方向取"上升违背量"和"下降违背量"的较小者求和
    dh = Ef[:, 1:] - Ef[:, :-1]
    inc_r = np.where(dh > 0, dh, 0.0).sum(axis=1)
    dec_r = np.where(dh < 0, -dh, 0.0).sum(axis=1)
    dv = Ef[1:, :] - Ef[:-1, :]
    inc_c = np.where(dv > 0, dv, 0.0).sum(axis=0)
    dec_c = np.where(dv < 0, -dv, 0.0).sum(axis=0)
    mono = -(np.minimum(inc_r, dec_r).sum() + np.minimum(inc_c, dec_c).sum())

    # 平滑性：相邻两格都非空时，累加其指数差绝对值的相反数
    mask_h = (E[:, 1:] > 0) & (E[:, :-1] > 0)
    mask_v = (E[1:, :] > 0) & (E[:-1, :] > 0)
    smooth = -(np.abs(dh)[mask_h].sum() + np.abs(dv)[mask_v].sum())

    max_log = float(E.max())

    max_pos = np.unravel_index(np.argmax(E), E.shape)
    corner = 1.0 if tuple(max_pos) in {(0, 0), (0, 3), (3, 0), (3, 3)} else 0.0

    eq_h = ((E[:, 1:] == E[:, :-1]) & (E[:, :-1] > 0)).sum()
    eq_v = ((E[1:, :] == E[:-1, :]) & (E[:-1, :] > 0)).sum()
    merges = float(eq_h + eq_v)

    return np.array([empty, mono, smooth, max_log, corner, merges], dtype=np.float64)


def evaluate_exp(E: np.ndarray, weights: np.ndarray = DEFAULT_WEIGHTS) -> float:
    """对指数矩阵局面打分：特征向量与权重向量的点积。"""
    return float(np.dot(extract_features_exp(E), weights))


# ---------- 固定权重下的高速评价（逐行评分查找表） ----------
# 单调性/平滑性/可合并数都可按"单行"分解：整盘 = 4 行 + 4 列各自的行评分之和。
# 因此对一个固定的权重向量，可离线枚举所有 65536 种行的评分，运行时只需 8 次
# 查表 + 3 个全局特征(空格/最大块/角落)，把评价函数再加速一个数量级，专供
# Expectimax 这类需要海量调用评价函数的搜索使用。

_IDX = np.arange(1 << 16, dtype=np.uint32)
_CELLS = np.stack([(_IDX >> (4 * k)) & 0xF for k in range(4)], axis=1).astype(np.float64)
_CORNERS = {(0, 0), (0, 3), (3, 0), (3, 3)}


def _line_components():
    cells = _CELLS
    d = cells[:, 1:] - cells[:, :-1]
    inc = np.where(d > 0, d, 0.0).sum(axis=1)
    dec = np.where(d < 0, -d, 0.0).sum(axis=1)
    mono = -np.minimum(inc, dec)
    both = (cells[:, 1:] > 0) & (cells[:, :-1] > 0)
    smooth = -(np.abs(d) * both).sum(axis=1)
    merges = ((cells[:, 1:] == cells[:, :-1]) & (cells[:, :-1] > 0)).sum(axis=1).astype(np.float64)
    return mono, smooth, merges


_MONO, _SMOOTH, _MERGES = _line_components()


def build_line_table(weights: np.ndarray) -> np.ndarray:
    """根据权重构建 (65536,) 的逐行评分表（仅含单调/平滑/合并三项）。"""
    w = np.asarray(weights, dtype=np.float64)
    return w[1] * _MONO + w[2] * _SMOOTH + w[5] * _MERGES


def evaluate_fast(E: np.ndarray, line_table: np.ndarray, weights: np.ndarray) -> float:
    """用预构建的逐行评分表快速评价；结果与 evaluate_exp 完全一致。"""
    Ew = E.astype(np.uint32)
    ridx = Ew[:, 0] | (Ew[:, 1] << 4) | (Ew[:, 2] << 8) | (Ew[:, 3] << 12)
    Et = Ew.T
    cidx = Et[:, 0] | (Et[:, 1] << 4) | (Et[:, 2] << 8) | (Et[:, 3] << 12)
    s = line_table[ridx].sum() + line_table[cidx].sum()
    empty = float(np.count_nonzero(E == 0))
    mx = float(E.max())
    pos = np.unravel_index(np.argmax(E), E.shape)
    corner = 1.0 if (int(pos[0]), int(pos[1])) in _CORNERS else 0.0
    return float(s + weights[0] * empty + weights[3] * mx + weights[4] * corner)


# ---------- 兼容数值棋盘的接口（供报告/可视化使用） ----------

def extract_features(grid: np.ndarray) -> np.ndarray:
    return extract_features_exp(to_exp(grid))


def evaluate(grid: np.ndarray, weights: np.ndarray = DEFAULT_WEIGHTS) -> float:
    return evaluate_exp(to_exp(grid), weights)
