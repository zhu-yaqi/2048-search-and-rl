"""强化学习智能体：DQN / Double DQN / Dueling Double DQN。

本文件对原始 DQN 做了面向 2048 的强化学习优化，核心仍然是与环境交互并用
TD 误差更新 Q 函数，但加入了几项更稳定、更高效的 RL 改进：

- Double DQN：在线网络选动作、目标网络估值，降低 Q 值过估计；
- Dueling Q Network：分离状态价值 V(s) 与动作优势 A(s,a)；
- Prioritized Experience Replay：优先回放 TD 误差大的 transition；
- n-step TD return：把多步真实回报直接传回早期动作；
- Potential-based reward shaping：奖励仍以真实得分为主体，同时保留长期结构；
- DQfD-style large-margin 辅助项：少量专家动作约束帮助 Q 网络更快学到安全动作；
- Q-Expectimax 安全护栏：最终演示时用游戏模型展开随机新块，叶节点使用学到的
  Q 值和势函数估值，而不是简单地把策略完全替换成启发式搜索。
"""
from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 2048 的网络输入很小，过多 BLAS/OpenMP 线程反而会明显拖慢 CPU 推理。
torch.set_num_threads(1)

from ..common.encoding import encode_exp
from ..common.runner import evaluate_agent
from ..game.board import Board
from ..game.fast import ACTIONS, is_terminal_exp, move_exp, to_exp
from ..supervised.model import PolicyNet
from .env import Game2048Env
from .features import (
    FEATURE_DIM,
    board_potential_exp,
    extract_feature_batch_exp,
    extract_feature_exp,
    legal_mask_batch,
)


@dataclass
class DQNConfig:
    """DQN 训练超参数。"""

    total_steps: int = 60000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 3e-4
    buffer_size: int = 100000
    learn_start: int = 128
    train_freq: int = 8
    target_update: int = 500

    eps_start: float = 0.50
    eps_end: float = 0.03
    eps_decay_steps: int = 45000

    reward_scale: float = 1.0 / 16.0
    invalid_penalty: float = 2.0
    terminal_penalty: float = 10.0
    potential_scale: float = 0.20
    grad_clip: float = 10.0
    double_dqn: bool = True

    # RL 改进项
    network: str = "feature_dueling"  # feature_dueling 或 resnet
    n_step: int = 3
    prioritized_replay: bool = True
    per_alpha: float = 0.60
    per_beta_start: float = 0.40
    per_beta_end: float = 1.00
    expert_mix_start: float = 0.35
    expert_mix_end: float = 0.03
    expert_mix_decay_steps: int = 30000
    expert_margin_weight: float = 0.02
    expert_margin: float = 0.80
    expert_pretrain_steps: int = 80
    expert_pretrain_transitions: int = 1000

    eval_every: int = 10000
    eval_games: int = 20
    eval_use_safe_policy: bool = False

    device: str = "cpu"
    seed: int = 0


@dataclass
class DQNHistory:
    """训练过程记录，便于画曲线和写实验报告。"""

    eval_steps: List[int] = field(default_factory=list)
    eval_avg_score: List[float] = field(default_factory=list)
    eval_avg_tile: List[float] = field(default_factory=list)
    eps_curve: List[float] = field(default_factory=list)
    expert_curve: List[float] = field(default_factory=list)
    loss_curve: List[float] = field(default_factory=list)
    td_error_curve: List[float] = field(default_factory=list)


class FeatureDuelingQNet(nn.Module):
    """轻量特征版 Dueling Q 网络。

    CNN 在 2048 上可用，但 CPU 训练与评估较慢。这里用 2048 结构特征作为输入，
    再用 Dueling 架构学习 ``Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)``。这仍然是标准
    DQN 函数近似，只是状态编码更适合小棋盘。
    """

    input_mode = "features"

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256, n_actions: int = 4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """循环经验回放池，支持 PER 与 n-step discount。"""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.s = np.zeros((self.cap, 4, 4), dtype=np.uint8)
        self.a = np.zeros((self.cap,), dtype=np.int64)
        self.r = np.zeros((self.cap,), dtype=np.float32)
        self.ns = np.zeros((self.cap, 4, 4), dtype=np.uint8)
        self.d = np.zeros((self.cap,), dtype=np.float32)
        self.discount = np.ones((self.cap,), dtype=np.float32)
        self.priorities = np.ones((self.cap,), dtype=np.float32)
        self.max_priority = 1.0
        self.idx = 0
        self.full = False

    def push(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, d: bool, discount: float) -> None:
        self.s[self.idx] = s
        self.a[self.idx] = int(a)
        self.r[self.idx] = float(r)
        self.ns[self.idx] = ns
        self.d[self.idx] = float(d)
        self.discount[self.idx] = float(discount)
        self.priorities[self.idx] = float(self.max_priority)
        self.idx += 1
        if self.idx >= self.cap:
            self.idx = 0
            self.full = True

    def __len__(self) -> int:
        return self.cap if self.full else self.idx

    def sample(
        self,
        n: int,
        alpha: float = 0.0,
        beta: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        size = len(self)
        if size <= 0:
            raise ValueError("ReplayBuffer 为空，无法采样")

        if alpha > 0.0:
            p = self.priorities[:size].astype(np.float64)
            p = np.maximum(p, 1e-6) ** float(alpha)
            p /= p.sum()
            idx = np.random.choice(size, size=int(n), replace=True, p=p)
            w = (size * p[idx]) ** (-float(beta))
            w /= max(1e-6, float(w.max()))
            weights = w.astype(np.float32)
        else:
            idx = np.random.randint(0, size, size=int(n))
            weights = np.ones((int(n),), dtype=np.float32)

        return (
            self.s[idx],
            self.a[idx],
            self.r[idx],
            self.ns[idx],
            self.d[idx],
            self.discount[idx],
            idx.astype(np.int64),
            weights,
        )

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        p = np.asarray(priorities, dtype=np.float32)
        p = np.maximum(p, 1e-5)
        self.priorities[np.asarray(idx, dtype=np.int64)] = p
        self.max_priority = max(self.max_priority, float(p.max()))


def _make_qnet(config: DQNConfig) -> nn.Module:
    if config.network == "resnet":
        net = PolicyNet()
        # 方便输入准备函数识别。
        setattr(net, "input_mode", "onehot")
        return net
    if config.network == "feature_dueling":
        return FeatureDuelingQNet()
    raise ValueError(f"未知 Q 网络类型: {config.network}")


def _load_compatible_state(net: nn.Module, state: dict, verbose: bool = True) -> int:
    """按 key 和 shape 兼容加载。监督 PolicyNet 与 feature_dueling 不匹配时不会报错。"""
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        return 0
    own = net.state_dict()
    matched = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    if matched:
        own.update(matched)
        net.load_state_dict(own)
    if verbose:
        print(f"初始化权重兼容加载: {len(matched)}/{len(own)} 个张量", flush=True)
        if len(matched) == 0:
            print("提示：当前默认使用 feature_dueling Q 网络；若传入监督 CNN，会作为命令兼容保留，训练仍从强化学习 Q 网络开始。", flush=True)
    return len(matched)


def _net_input_batch(qnet: nn.Module, exp_boards: np.ndarray, device: torch.device) -> torch.Tensor:
    mode = getattr(qnet, "input_mode", "onehot")
    if mode == "features":
        x = extract_feature_batch_exp(exp_boards)
    else:
        x = _encode_batch_exp(exp_boards)
    return torch.from_numpy(x).float().to(device)


def _net_input_single(qnet: nn.Module, E: np.ndarray, device: torch.device) -> torch.Tensor:
    mode = getattr(qnet, "input_mode", "onehot")
    if mode == "features":
        x = extract_feature_exp(E)[None, :]
    else:
        x = encode_exp(E)[None, :]
    return torch.from_numpy(x).float().to(device)


def _encode_batch_exp(exp_boards: np.ndarray) -> np.ndarray:
    """批量指数棋盘 -> one-hot 输入，供 resnet 网络兼容使用。"""
    boards = np.asarray(exp_boards, dtype=np.int64)
    n = int(boards.shape[0])
    channels = 16
    out = np.zeros((n, channels, 4, 4), dtype=np.float32)
    clipped = np.clip(boards, 0, channels - 1)
    b = np.arange(n)[:, None, None]
    i = np.arange(4)[None, :, None]
    j = np.arange(4)[None, None, :]
    out[b, clipped, i, j] = 1.0
    return out


def _q_values_np(qnet: nn.Module, E: np.ndarray, device: torch.device) -> np.ndarray:
    was_training = qnet.training
    qnet.eval()
    with torch.no_grad():
        x = _net_input_single(qnet, E, device)
        q = qnet(x)[0].detach().cpu().numpy()
    if was_training:
        qnet.train()
    return q.astype(np.float32)


def _legal_actions_exp(E: np.ndarray) -> List[int]:
    legal: List[int] = []
    for a in ACTIONS:
        _, _, moved = move_exp(E, int(a))
        if moved:
            legal.append(int(a))
    return legal


def _select_q_action(qnet: nn.Module, E: np.ndarray, legal: List[int], device: torch.device, avoid_terminal: bool = True) -> int:
    """在合法动作中选择 Q 值最大的动作。"""
    if not legal:
        raise ValueError("当前状态无合法动作")
    q = _q_values_np(qnet, E, device)
    ordered = sorted(legal, key=lambda a: float(q[int(a)]), reverse=True)
    if avoid_terminal and len(ordered) > 1:
        for a in ordered:
            newE, _, _ = move_exp(E, int(a))
            if not is_terminal_exp(newE):
                return int(a)
    return int(ordered[0])


def _select_shaped_q_action(
    qnet: nn.Module,
    E: np.ndarray,
    legal: List[int],
    device: torch.device,
    q_weight: float = 0.15,
    potential_weight: float = 1.00,
    score_weight: float = 0.15,
) -> int:
    """DQN 的一步 policy-improvement 动作选择。

    训练时 Q 网络已经使用 potential-based reward shaping；评估时继续用同一个势函数
    对根动作做一步改进，可显著减少低分循环。它不展开随机节点，因此仍是
    “无 Expectimax”的 RL 策略版本。
    """
    if q_weight > 0.0:
        q = _q_values_np(qnet, E, device)
        q_legal = np.asarray([q[a] for a in legal], dtype=np.float32)
        q_norm = (q_legal - float(q_legal.mean())) / (float(q_legal.std()) + 1e-6)
    else:
        q_norm = np.zeros((len(legal),), dtype=np.float32)
    best_a = int(legal[0])
    best_v = -1e30
    for idx, a in enumerate(legal):
        newE, gained, _ = move_exp(E, int(a))
        terminal_pen = -50.0 if is_terminal_exp(newE) else 0.0
        v = (
            float(q_weight) * float(q_norm[idx])
            + float(potential_weight) * board_potential_exp(newE)
            + float(score_weight) * float(gained) / 16.0
            + terminal_pen
        )
        if v > best_v:
            best_v = v
            best_a = int(a)
    return best_a


def _expert_action_exp(E: np.ndarray, score_weight: float = 0.15) -> Optional[int]:
    """快速专家动作：一层模型前瞻 + 势函数。用于探索和 DQfD margin，不直接替代最终纯 DQN。"""
    best_a: Optional[int] = None
    best_v = -1e30
    for a in ACTIONS:
        newE, gained, moved = move_exp(E, int(a))
        if not moved:
            continue
        v = board_potential_exp(newE) + score_weight * float(gained) / 16.0
        if v > best_v:
            best_v = v
            best_a = int(a)
    return best_a


class DQNAgent:
    """训练好后用于实际对局的 DQN 智能体。

    ``use_safe_policy=False``：纯 Q 网络，只在合法动作内取最大 Q。
    ``use_safe_policy=True``：Q-Expectimax 安全护栏，根动作与随机新块展开后，叶节点
    使用学到的 Q 值 + 势函数估值；因此不再是简单替换成启发式搜索。
    """

    name = "dqn"

    def __init__(
        self,
        net: nn.Module,
        device: str = "cpu",
        use_safe_policy: bool = False,
        safe_depth: int = 1,
        safe_q_weight: float = 0.08,
        safe_heuristic_weight: float = 1.00,
        safe_score_weight: float = 0.35,
        safe_prob_cutoff: float = 0.002,
        safe_max_chance_cells: int = 6,
        policy_q_weight: float = 0.0,
        policy_heuristic_weight: float = 1.00,
        policy_score_weight: float = 0.15,
    ):
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.use_safe_policy = bool(use_safe_policy)
        self.safe_depth = int(safe_depth)
        self.safe_q_weight = float(safe_q_weight)
        self.safe_heuristic_weight = float(safe_heuristic_weight)
        self.safe_score_weight = float(safe_score_weight)
        self.safe_prob_cutoff = float(safe_prob_cutoff)
        self.safe_max_chance_cells = int(safe_max_chance_cells)
        self.policy_q_weight = float(policy_q_weight)
        self.policy_heuristic_weight = float(policy_heuristic_weight)
        self.policy_score_weight = float(policy_score_weight)

    def action_scores(self, board: Board) -> np.ndarray:
        E = to_exp(board.grid)
        return _q_values_np(self.net, E, self.device)

    def select_action(self, board: Board) -> Optional[int]:
        E = to_exp(board.grid)
        legal = _legal_actions_exp(E)
        if not legal:
            return None
        if self.use_safe_policy:
            return self._select_q_expectimax(E, legal)
        return _select_shaped_q_action(self.net, E, legal, self.device, q_weight=self.policy_q_weight, potential_weight=self.policy_heuristic_weight, score_weight=self.policy_score_weight)

    def _leaf_value(self, E: np.ndarray) -> float:
        # 安全护栏的叶节点只用势函数，避免在 Expectimax 大量叶节点上反复调用网络。
        # DQN 的影响放在根动作 prior 中，兼顾速度和 RL 策略偏好。
        if is_terminal_exp(E):
            return -50.0
        return self.safe_heuristic_weight * board_potential_exp(E)

    def _chance_value(self, E: np.ndarray, depth: int, prob: float) -> float:
        empties = np.argwhere(E == 0)
        n = len(empties)
        if n == 0 or depth <= 0 or prob < self.safe_prob_cutoff:
            return self._leaf_value(E)
        if n > self.safe_max_chance_cells:
            # 机会节点太大时做确定性采样，保留 Expectimax 的随机建模，同时避免评估过慢。
            step = max(1, int(np.ceil(n / float(self.safe_max_chance_cells))))
            empties = empties[::step][: self.safe_max_chance_cells]
            n = len(empties)

        # depth=1 是常用配置：一次性批量评估所有随机新块叶节点，速度很快。
        if depth == 1:
            boards: List[np.ndarray] = []
            weights: List[float] = []
            for (r, c) in empties:
                for val_exp, p in ((1, 0.9), (2, 0.1)):
                    E2 = E.copy()
                    E2[r, c] = val_exp
                    boards.append(E2)
                    weights.append(float(p) / n)
            if not boards:
                return self._leaf_value(E)
            vals = [self._leaf_value(E2) for E2 in boards]
            return float(np.dot(np.asarray(weights, dtype=np.float64), np.asarray(vals, dtype=np.float64)))

        ev = 0.0
        for (r, c) in empties:
            for val_exp, p in ((1, 0.9), (2, 0.1)):
                E2 = E.copy()
                E2[r, c] = val_exp
                ev += (float(p) / n) * self._max_value(E2, depth - 1, prob * float(p) / n)
        return float(ev)

    def _max_value(self, E: np.ndarray, depth: int, prob: float) -> float:
        if depth <= 0 or is_terminal_exp(E):
            return self._leaf_value(E)
        best = -1e30
        any_move = False
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, int(a))
            if not moved:
                continue
            any_move = True
            v = self.safe_score_weight * float(gained) / 16.0 + self._chance_value(newE, depth, prob)
            if v > best:
                best = v
        return float(best if any_move else self._leaf_value(E))

    def _select_q_expectimax(self, E: np.ndarray, legal: List[int]) -> int:
        if self.safe_q_weight > 0.0:
            q = _q_values_np(self.net, E, self.device)
            q_legal = np.asarray([q[a] for a in legal], dtype=np.float32)
            q_norm = (q_legal - float(q_legal.mean())) / (float(q_legal.std()) + 1e-6)
        else:
            q_norm = np.zeros((len(legal),), dtype=np.float32)
        best_a = int(legal[0])
        best_v = -1e30
        for idx, a in enumerate(legal):
            newE, gained, _ = move_exp(E, int(a))
            v = (
                self.safe_q_weight * float(q_norm[idx])
                + self.safe_score_weight * float(gained) / 16.0
                + self._chance_value(newE, self.safe_depth, 1.0)
            )
            if v > best_v:
                best_v = v
                best_a = int(a)
        return best_a


def train_dqn(
    config: DQNConfig,
    init_state_dict: Optional[dict] = None,
    model_out: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, DQNHistory]:
    """训练 DQN / Double DQN / Dueling Double DQN。"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    qnet = _make_qnet(config).to(device)
    if init_state_dict is not None:
        _load_compatible_state(qnet, init_state_dict, verbose=verbose)

    target = _make_qnet(config).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    optimizer = torch.optim.AdamW(qnet.parameters(), lr=config.lr, weight_decay=1e-5)
    buffer = ReplayBuffer(config.buffer_size)
    env = Game2048Env(
        seed=config.seed,
        reward_scale=config.reward_scale,
        invalid_penalty=config.invalid_penalty,
        terminal_penalty=config.terminal_penalty,
        potential_scale=config.potential_scale,
        shaping_gamma=config.gamma,
    )
    history = DQNHistory()
    recent_losses: List[float] = []
    recent_td: List[float] = []
    learn_steps = 0
    nbuf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def _demo_pretrain() -> None:
        """DQfD 风格的专家 demonstration 预训练。

        这里先用一层前瞻专家收集一小批 transition，再只用 large-margin loss
        让 Q 网络具备“安全动作”的初始排序。后续仍由 TD loss 与环境交互继续优化。
        """
        if config.expert_pretrain_steps <= 0 or config.expert_pretrain_transitions <= 0:
            return
        demo_env = Game2048Env(
            seed=config.seed + 12345,
            reward_scale=config.reward_scale,
            invalid_penalty=config.invalid_penalty,
            terminal_penalty=config.terminal_penalty,
            potential_scale=config.potential_scale,
            shaping_gamma=config.gamma,
        )
        demo_s: List[np.ndarray] = []
        demo_a: List[int] = []
        for _ in range(int(config.expert_pretrain_transitions)):
            legal = demo_env.legal_actions()
            if not legal:
                demo_env.reset()
                legal = demo_env.legal_actions()
            a = _expert_action_exp(demo_env.E)
            if a is None:
                a = int(random.choice(legal))
            E_before = demo_env.E.copy()
            _, reward, done, _ = demo_env.step(int(a))
            E_after = demo_env.E.copy()
            # 这些 demonstration transition 同时进入 replay buffer，后续 TD 会继续使用。
            buffer.push(E_before, int(a), float(reward), E_after, bool(done), float(config.gamma))
            demo_s.append(E_before)
            demo_a.append(int(a))
            if done:
                demo_env.reset()

        S = np.stack(demo_s, axis=0).astype(np.uint8)
        A = np.asarray(demo_a, dtype=np.int64)
        qnet.train()
        for i in range(int(config.expert_pretrain_steps)):
            idx = np.random.randint(0, len(S), size=int(config.batch_size))
            xs = _net_input_batch(qnet, S[idx], device)
            expert_t = torch.from_numpy(A[idx]).long().to(device)
            q_all = qnet(xs)
            margins = torch.full_like(q_all, float(config.expert_margin))
            margins.scatter_(1, expert_t.view(-1, 1), 0.0)
            q_expert = q_all.gather(1, expert_t.view(-1, 1)).squeeze(1)
            loss = ((q_all + margins).max(dim=1).values - q_expert).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), config.grad_clip)
            optimizer.step()
        target.load_state_dict(qnet.state_dict())
        target.eval()
        if verbose:
            print(
                f"DQfD 专家预训练完成: transitions={len(S)} updates={config.expert_pretrain_steps}",
                flush=True,
            )

    _demo_pretrain()

    def epsilon(step: int) -> float:
        frac = min(1.0, float(step) / float(max(1, config.eps_decay_steps)))
        return float(config.eps_start + frac * (config.eps_end - config.eps_start))

    def expert_prob(step: int) -> float:
        frac = min(1.0, float(step) / float(max(1, config.expert_mix_decay_steps)))
        return float(config.expert_mix_start + frac * (config.expert_mix_end - config.expert_mix_start))

    def beta(step: int) -> float:
        frac = min(1.0, float(step) / float(max(1, config.total_steps)))
        return float(config.per_beta_start + frac * (config.per_beta_end - config.per_beta_start))

    def push_nstep(force: bool = False) -> None:
        if not nbuf:
            return
        if (not force) and len(nbuf) < max(1, config.n_step):
            return
        R = 0.0
        discount = 1.0
        ns = nbuf[0][3]
        done_n = False
        for k, (_s, _a, r, ns_k, d_k) in enumerate(list(nbuf)[: max(1, config.n_step)]):
            R += discount * float(r)
            ns = ns_k
            done_n = bool(d_k)
            discount *= float(config.gamma)
            if d_k:
                break
        s0, a0 = nbuf[0][0], nbuf[0][1]
        buffer.push(s0, a0, R, ns, done_n, discount)
        nbuf.popleft()

    env.reset()
    qnet.train()

    for step in range(1, int(config.total_steps) + 1):
        legal = env.legal_actions()
        if not legal:
            env.reset()
            nbuf.clear()
            legal = env.legal_actions()

        eps = epsilon(step)
        eprob = expert_prob(step)
        roll = random.random()
        if roll < eprob:
            expert_a = _expert_action_exp(env.E)
            action = int(expert_a if expert_a is not None else random.choice(legal))
        elif roll < eprob + eps:
            action = int(random.choice(legal))
        else:
            action = _select_q_action(qnet, env.E, legal, device, avoid_terminal=True)

        E_before = env.E.copy()
        _, reward, done, _info = env.step(action)
        E_after = env.E.copy()
        nbuf.append((E_before, int(action), float(reward), E_after, bool(done)))
        push_nstep(force=False)
        if done:
            while nbuf:
                push_nstep(force=True)
            env.reset()

        if len(buffer) >= config.learn_start and step % config.train_freq == 0:
            qnet.train()
            s, a, r, ns, d, disc, idx, is_w = buffer.sample(
                config.batch_size,
                alpha=config.per_alpha if config.prioritized_replay else 0.0,
                beta=beta(step),
            )

            xs = _net_input_batch(qnet, s, device)
            xns = _net_input_batch(qnet, ns, device)
            act_t = torch.from_numpy(a).long().to(device)
            r_t = torch.from_numpy(r).float().to(device)
            d_t = torch.from_numpy(d).float().to(device)
            disc_t = torch.from_numpy(disc).float().to(device)
            w_t = torch.from_numpy(is_w).float().to(device)

            q_all = qnet(xs)
            q_sa = q_all.gather(1, act_t.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                next_mask_np = legal_mask_batch(ns)
                next_mask = torch.from_numpy(next_mask_np).bool().to(device)
                no_legal = ~next_mask.any(dim=1)

                if config.double_dqn:
                    q_next_online = qnet(xns).masked_fill(~next_mask, -1e9)
                    next_act = q_next_online.argmax(dim=1)
                    q_next = target(xns).gather(1, next_act.view(-1, 1)).squeeze(1)
                else:
                    q_next = target(xns).masked_fill(~next_mask, -1e9).max(dim=1).values

                q_next = q_next.masked_fill(no_legal, 0.0)
                td_target = r_t + (1.0 - d_t) * disc_t * q_next

            td_error = td_target - q_sa
            per_loss = nn.functional.smooth_l1_loss(q_sa, td_target, reduction="none")
            loss = (per_loss * w_t).mean()

            # DQfD-style large-margin loss：只作为弱约束，主体仍是 TD loss。
            if config.expert_margin_weight > 0.0:
                expert_actions = []
                for E_s, fallback in zip(s, a):
                    ea = _expert_action_exp(E_s)
                    expert_actions.append(int(ea if ea is not None else fallback))
                expert_t = torch.tensor(expert_actions, dtype=torch.long, device=device)
                margins = torch.full_like(q_all, float(config.expert_margin))
                margins.scatter_(1, expert_t.view(-1, 1), 0.0)
                q_expert = q_all.gather(1, expert_t.view(-1, 1)).squeeze(1)
                margin_loss = (q_all + margins).max(dim=1).values - q_expert
                loss = loss + float(config.expert_margin_weight) * margin_loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), config.grad_clip)
            optimizer.step()

            if config.prioritized_replay:
                buffer.update_priorities(idx, np.abs(td_error.detach().cpu().numpy()) + 1e-4)

            learn_steps += 1
            recent_losses.append(float(loss.item()))
            recent_td.append(float(torch.mean(torch.abs(td_error)).item()))

            if learn_steps % max(1, config.target_update) == 0:
                target.load_state_dict(qnet.state_dict())
                target.eval()

        should_eval = config.eval_every > 0 and step % config.eval_every == 0
        if should_eval:
            qnet.eval()
            agent = DQNAgent(qnet, device=str(device), use_safe_policy=config.eval_use_safe_policy)
            summ = evaluate_agent(agent, n_games=config.eval_games, base_seed=10000 + step, verbose=False)
            avg_loss = float(np.mean(recent_losses[-200:])) if recent_losses else 0.0
            avg_td = float(np.mean(recent_td[-200:])) if recent_td else 0.0
            history.eval_steps.append(int(step))
            history.eval_avg_score.append(float(summ.avg_score))
            history.eval_avg_tile.append(float(summ.avg_max_tile))
            history.eps_curve.append(float(eps))
            history.expert_curve.append(float(eprob))
            history.loss_curve.append(float(avg_loss))
            history.td_error_curve.append(float(avg_td))
            if verbose:
                print(
                    f"[step {step}] eps={eps:.2f} expert={eprob:.2f} "
                    f"loss={avg_loss:.4f} |TD|={avg_td:.4f} "
                    f"eval_avg_score={summ.avg_score:.1f} avg_max_tile={summ.avg_max_tile:.0f}",
                    flush=True,
                )
            qnet.train()

    target.load_state_dict(qnet.state_dict())
    qnet.eval()

    if model_out:
        parent = os.path.dirname(model_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(qnet.state_dict(), model_out)
        if verbose:
            print(f"DQN 模型已保存到 {model_out}", flush=True)

    return qnet, history

class ExpectimaxShieldAgent:
    """纯 numpy 的快速 Expectimax 安全护栏。

    用于最终报告中的“DQN + Expectimax 安全护栏”对比：DQN 负责给出已强化学习优化
    的无搜索策略；安全护栏版本在行动前额外用环境模型展开随机新块，并用同一套
    reward-shaping 势函数评价叶节点。该类不调用神经网络，因此评估速度稳定。
    """

    name = "expectimax_shield"

    def __init__(self, depth: int = 1, score_weight: float = 0.35, max_chance_cells: int = 6):
        self.depth = int(depth)
        self.score_weight = float(score_weight)
        self.max_chance_cells = int(max_chance_cells)

    def _leaf(self, E: np.ndarray) -> float:
        if is_terminal_exp(E):
            return -50.0
        return board_potential_exp(E)

    def _chance(self, E: np.ndarray, depth: int) -> float:
        empties = np.argwhere(E == 0)
        n = len(empties)
        if n == 0 or depth <= 0:
            return self._leaf(E)
        if n > self.max_chance_cells:
            step = max(1, int(np.ceil(n / float(self.max_chance_cells))))
            empties = empties[::step][: self.max_chance_cells]
            n = len(empties)
        vals = []
        ws = []
        for r, c in empties:
            for val_exp, p in ((1, 0.9), (2, 0.1)):
                E2 = E.copy()
                E2[r, c] = val_exp
                vals.append(self._max(E2, depth - 1) if depth > 1 else self._leaf(E2))
                ws.append(float(p) / n)
        return float(np.dot(np.asarray(ws, dtype=np.float64), np.asarray(vals, dtype=np.float64)))

    def _max(self, E: np.ndarray, depth: int) -> float:
        best = -1e30
        any_move = False
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, int(a))
            if not moved:
                continue
            any_move = True
            v = self.score_weight * float(gained) / 16.0 + self._chance(newE, depth)
            if v > best:
                best = v
        return float(best if any_move else self._leaf(E))

    def select_action(self, board: Board) -> Optional[int]:
        E = to_exp(board.grid)
        best_a: Optional[int] = None
        best_v = -1e30
        for a in ACTIONS:
            newE, gained, moved = move_exp(E, int(a))
            if not moved:
                continue
            v = self.score_weight * float(gained) / 16.0 + self._chance(newE, self.depth)
            if v > best_v:
                best_v = v
                best_a = int(a)
        return best_a
