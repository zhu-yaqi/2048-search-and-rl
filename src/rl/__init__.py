"""强化学习模块：2048 环境、DQN/Double DQN/Dueling DQN 智能体与训练入口。"""

from .env import Game2048Env
from .dqn import DQNAgent, DQNConfig, DQNHistory, ExpectimaxShieldAgent, FeatureDuelingQNet, ReplayBuffer, train_dqn

__all__ = [
    "Game2048Env",
    "DQNAgent",
    "DQNConfig",
    "DQNHistory",
    "FeatureDuelingQNet",
    "ExpectimaxShieldAgent",
    "ReplayBuffer",
    "train_dqn",
]
