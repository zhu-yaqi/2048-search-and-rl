"""监督学习 v2：更强的 2048 策略网络。

输入：
    X.shape = (N, 16, 4, 4)
    16 个通道通常表示：空格、2、4、8、... 的 one-hot 编码。

输出：
    logits.shape = (N, 4)
    四个动作的分数，上/下/左/右。

特点：
1. 小型 ResNet，比普通 CNN 更稳定。
2. 自动从 one-hot 棋盘中提取空格数、最大块、蛇形结构等辅助特征。
3. NetAgent 推理时只在合法动作中选择最高分动作。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..common.encoding import NUM_CHANNELS, encode_board
from ..game.board import Board


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BoardFeatureLayer(nn.Module):
    """从 one-hot 棋盘中提取少量 2048 结构特征。

    这里假设通道编号就是指数：
        channel 0 = 空格
        channel 1 = 2
        channel 2 = 4
        channel 3 = 8
        ...
    如果你的 encode_board 不是这个规则，需要同步修改这里。
    """

    def __init__(self, num_channels: int = NUM_CHANNELS):
        super().__init__()
        exps = torch.arange(num_channels, dtype=torch.float32).view(1, num_channels, 1, 1)
        self.register_buffer("exps", exps)

        base_snake = torch.tensor([
            [15, 14, 13, 12],
            [8,   9, 10, 11],
            [7,   6,  5,  4],
            [0,   1,  2,  3],
        ], dtype=torch.float32)

        snakes = torch.stack([
            base_snake,
            torch.flip(base_snake, dims=[1]),
            torch.flip(base_snake, dims=[0]),
            torch.flip(base_snake, dims=[0, 1]),
        ], dim=0)

        snakes = snakes / snakes.max().clamp_min(1.0)
        self.register_buffer("snakes", snakes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 4, 4)
        B = x.size(0)
        exp_grid = (x * self.exps).sum(dim=1)  # (B, 4, 4)

        empty_ratio = x[:, 0].sum(dim=(1, 2)) / 16.0
        max_exp = exp_grid.amax(dim=(1, 2)) / 15.0
        mean_exp = exp_grid.sum(dim=(1, 2)) / (16.0 * 15.0)

        # 蛇形结构得分：四个角都允许，取最大
        snake_scores = (exp_grid.unsqueeze(1) * self.snakes.unsqueeze(0)).sum(dim=(2, 3))
        snake_best = snake_scores.max(dim=1).values / (15.0 * 16.0)

        # 平滑性：相邻指数差越小越好；这里取负惩罚并归一化
        right_mask = (exp_grid[:, :, :-1] > 0) & (exp_grid[:, :, 1:] > 0)
        down_mask = (exp_grid[:, :-1, :] > 0) & (exp_grid[:, 1:, :] > 0)
        right_diff = torch.abs(exp_grid[:, :, :-1] - exp_grid[:, :, 1:]) * right_mask
        down_diff = torch.abs(exp_grid[:, :-1, :] - exp_grid[:, 1:, :]) * down_mask
        smoothness = -(right_diff.sum(dim=(1, 2)) + down_diff.sum(dim=(1, 2))) / 180.0

        # 可合并潜力：相邻相同且非空
        right_merge = ((exp_grid[:, :, :-1] == exp_grid[:, :, 1:]) & right_mask).float().sum(dim=(1, 2))
        down_merge = ((exp_grid[:, :-1, :] == exp_grid[:, 1:, :]) & down_mask).float().sum(dim=(1, 2))
        merge_ratio = (right_merge + down_merge) / 24.0

        return torch.stack([
            empty_ratio,
            max_exp,
            mean_exp,
            snake_best,
            smoothness,
            merge_ratio,
        ], dim=1)


class PolicyNet(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        n_actions: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, dropout=dropout) for _ in range(n_blocks)]
        )

        self.feature_layer = BoardFeatureLayer(NUM_CHANNELS)
        self.feature_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(channels * 4 * 4 + 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.blocks(z)
        z = torch.flatten(z, 1)

        f = self.feature_layer(x)
        f = self.feature_mlp(f)

        return self.head(torch.cat([z, f], dim=1))


class NetAgent:
    """用策略网络选择动作的智能体。"""

    name = "net"

    def __init__(self, net: PolicyNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.device = device

    def action_scores(self, board: Board) -> np.ndarray:
        x = torch.from_numpy(encode_board(board.grid)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.net(x)[0].detach().cpu().numpy()
        return logits

    def select_action(self, board: Board) -> Optional[int]:
        legal = board.available_moves()
        if not legal:
            return None

        scores = self.action_scores(board)

        # 只在合法动作中选最高分，避免网络输出非法方向导致空走。
        return int(max(legal, key=lambda a: scores[a]))


def load_net(
    path: str,
    device: str = "cpu",
    channels: int = 128,
    n_blocks: int = 4,
    dropout: float = 0.05,
) -> PolicyNet:
    net = PolicyNet(channels=channels, n_blocks=n_blocks, dropout=dropout)
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    return net
