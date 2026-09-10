"""棋盘的神经网络输入编码。

把 4x4 棋盘编码为 one-hot 张量，形状为 (C, 4, 4)，其中通道 k 表示该格子的
数值是否为 2^k（k=0 表示空格）。这样的表示对神经网络更友好：
- 不同量级的方块被映射到不同通道，避免了数值大小对网络的非线性干扰；
- 卷积核可以直接学习"相邻方块的空间关系"（单调、可合并等）。
"""
from __future__ import annotations

from typing import List

import numpy as np

# 通道数：支持到 2^15=32768，足够覆盖正常对局
NUM_CHANNELS = 16


def encode_board(grid: np.ndarray) -> np.ndarray:
    """单个棋盘 -> (NUM_CHANNELS, 4, 4) 的 float32 one-hot 张量。"""
    out = np.zeros((NUM_CHANNELS, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = int(grid[i, j])
            k = 0 if v == 0 else int(np.log2(v))
            k = min(k, NUM_CHANNELS - 1)
            out[k, i, j] = 1.0
    return out


def encode_exp(E: np.ndarray) -> np.ndarray:
    """指数矩阵(uint8) -> (NUM_CHANNELS, 4, 4) one-hot，供高性能环境直接使用。"""
    out = np.zeros((NUM_CHANNELS, 4, 4), dtype=np.float32)
    k = np.clip(E.astype(np.int64), 0, NUM_CHANNELS - 1)
    ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    out[k.ravel(), ii.ravel(), jj.ravel()] = 1.0
    return out


def encode_batch(grids: List[np.ndarray]) -> np.ndarray:
    """批量编码 -> (N, NUM_CHANNELS, 4, 4)。"""
    return np.stack([encode_board(g) for g in grids], axis=0)
