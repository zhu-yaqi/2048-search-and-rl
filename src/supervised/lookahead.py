"""监督网络 + 前瞻搜索（推理时增强，无需重训）。

动机：
    监督网络 NetAgent 只做一次前向、直接 argmax，缺少"前瞻"，因此在长对局中
    少量错误会累积，导致棋盘结构崩坏。本模块在**不改动、不重训**网络的前提下，
    用网络自身做浅层前瞻，把"直觉策略"升级为"带搜索的策略"。

两种价值估计（均为纯网络，不掺启发式）：
    1. rollout（默认）：对每个候选动作，先真实执行 move + 随机 spawn，再用网络
       策略把后继局面继续玩 rollout_len 步，以这段真实累计得分作为该动作价值，
       多次采样取平均以平滑随机性。价值信号是真实游戏得分，可靠且量纲一致。
    2. 这等价于经典的 "policy rollout / one-step lookahead"，是 AlphaGo 之前
       就广泛使用的、用策略网络增强决策质量的标准做法。

速度：
    每个决策步约 len(legal) × n_rollouts × rollout_len 次网络前向。CPU 上比纯
    NetAgent 慢，但评估几十局完全可行。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from ..game.board import Board
from ..game.fast import ACTIONS, move_exp, spawn_exp, is_terminal_exp, to_exp
from ..common.encoding import encode_exp
from .model import PolicyNet


class NetLookaheadAgent:
    """用监督网络做策略先验 + 真实 rollout 前瞻的智能体。"""

    name = "net_lookahead"

    def __init__(
        self,
        net: PolicyNet,
        device: str = "cpu",
        rollout_len: int = 8,
        n_rollouts: int = 3,
        score_scale: float = 1.0,
        seed: int = 0,
    ):
        self.net = net.to(device).eval()
        self.device = device
        self.rollout_len = rollout_len
        self.n_rollouts = n_rollouts
        self.score_scale = score_scale
        self.rng = np.random.default_rng(seed)

    def _legal_exp(self, E: np.ndarray) -> List[int]:
        return [a for a in ACTIONS if move_exp(E, a)[2]]

    def _net_pick(self, E: np.ndarray, legal: List[int]) -> int:
        """用网络在合法动作中选最高分动作（rollout 内部策略）。"""
        x = torch.from_numpy(encode_exp(E)).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.net(x)[0].cpu().numpy()
        return int(max(legal, key=lambda a: logits[a]))

    def _rollout(self, E: np.ndarray) -> float:
        """从 E 出发，用网络策略玩 rollout_len 步，返回真实累计合并得分。"""
        total = 0.0
        cur = E
        for _ in range(self.rollout_len):
            if is_terminal_exp(cur):
                break
            legal = self._legal_exp(cur)
            if not legal:
                break
            a = self._net_pick(cur, legal)
            newE, gained, moved = move_exp(cur, a)
            if not moved:
                break
            total += float(gained)
            cur = spawn_exp(newE, self.rng)
        return total

    def select_action(self, board: Board) -> Optional[int]:
        E = to_exp(board.grid)
        legal = self._legal_exp(E)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best_a, best_q = None, -np.inf
        for a in legal:
            newE, gained, _ = move_exp(E, a)
            samples = []
            for _ in range(self.n_rollouts):
                s2 = spawn_exp(newE, self.rng)
                samples.append(float(gained) + self._rollout(s2))
            q = self.score_scale * float(np.mean(samples))
            if q > best_q:
                best_q, best_a = q, a
        return best_a
