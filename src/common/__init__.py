"""通用工具：对局运行与评估、棋盘编码。"""
from .runner import play_game, evaluate_agent, GameResult
from .encoding import encode_board, encode_exp, encode_batch, NUM_CHANNELS

__all__ = [
    "play_game",
    "evaluate_agent",
    "GameResult",
    "encode_board",
    "encode_exp",
    "encode_batch",
    "NUM_CHANNELS",
]
