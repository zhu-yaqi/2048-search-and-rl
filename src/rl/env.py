"""2048 强化学习环境。

本模块把 2048 游戏封装为极简 Gym-like 环境，供 DQN、Double DQN、Dueling
DQN、Prioritized Replay、n-step TD 等强化学习算法与游戏持续交互。

动作空间固定为四个离散动作：
    0=上, 1=下, 2=左, 3=右

奖励定义：
1. 真实合并得分 ``gained * reward_scale``：保证优化目标与 2048 原始分数一致；
2. potential-based reward shaping ``scale * (gamma*Φ(s') - Φ(s))``：鼓励保留空格、
   单调/平滑结构、最大块靠角等长期有利局面；
3. 终局惩罚：避免智能体为了短期合并过早死亡。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..common.encoding import encode_exp
from ..game.fast import ACTIONS, is_terminal_exp, move_exp, spawn_exp
from .features import board_potential_exp


class Game2048Env:
    """Gym-like 的 2048 强化学习环境。"""

    def __init__(
        self,
        seed: Optional[int] = None,
        reward_scale: float = 1.0 / 16.0,
        invalid_penalty: float = 2.0,
        terminal_penalty: float = 10.0,
        potential_scale: float = 0.20,
        shaping_gamma: float = 0.99,
    ):
        self.rng = np.random.default_rng(seed)
        self.reward_scale = float(reward_scale)
        self.invalid_penalty = float(invalid_penalty)
        self.terminal_penalty = float(terminal_penalty)
        self.potential_scale = float(potential_scale)
        self.shaping_gamma = float(shaping_gamma)
        self.E = np.zeros((4, 4), dtype=np.uint8)
        self.score = 0.0
        self.reset()

    def reset(self) -> np.ndarray:
        """重置环境并返回初始 one-hot 观测。"""
        self.E = np.zeros((4, 4), dtype=np.uint8)
        self.E = spawn_exp(self.E, self.rng)
        self.E = spawn_exp(self.E, self.rng)
        self.score = 0.0
        return self.obs()

    def obs(self) -> np.ndarray:
        return encode_exp(self.E)

    def legal_actions(self) -> List[int]:
        legal: List[int] = []
        for action in ACTIONS:
            _, _, moved = move_exp(self.E, action)
            if moved:
                legal.append(int(action))
        return legal

    def max_tile(self) -> int:
        m = int(self.E.max())
        return 0 if m <= 0 else int(1 << m)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步动作，返回 ``(obs, reward, done, info)``。"""
        action = int(action)
        phi_before = board_potential_exp(self.E)

        newE, gained, moved = move_exp(self.E, action)
        if not moved:
            done = is_terminal_exp(self.E)
            reward = -max(1.0, self.invalid_penalty)
            if done:
                reward -= self.terminal_penalty
            return self.obs(), float(reward), bool(done), {
                "invalid": True,
                "score": float(self.score),
                "gained": 0.0,
                "potential_before": float(phi_before),
                "potential_after": float(phi_before),
            }

        # 真实 2048 规则：有效移动后随机生成一个 2/4 方块。
        self.E = spawn_exp(newE, self.rng)
        self.score += float(gained)
        done = is_terminal_exp(self.E)

        phi_after = board_potential_exp(self.E)
        reward = float(gained) * self.reward_scale
        reward += self.potential_scale * (self.shaping_gamma * phi_after - phi_before)
        if done:
            reward -= self.terminal_penalty

        return self.obs(), float(reward), bool(done), {
            "invalid": False,
            "score": float(self.score),
            "gained": float(gained),
            "max_tile": self.max_tile(),
            "potential_before": float(phi_before),
            "potential_after": float(phi_after),
        }
