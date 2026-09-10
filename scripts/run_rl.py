"""第 5 部分：运行 DQN / Double DQN / Dueling DQN 强化学习训练并输出结果。用法示例：

    python scripts/run_rl.py --steps 1000 --eval-every 200 --eval-games 3 --final-eval-games 5
    python scripts/run_rl.py --steps 60000
    python scripts/run_rl.py --steps 60000 --init results/supervised/policy.pt

结果会写入 ``results/rl``，并同步合并到 ``results/summary.json``。
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict

import _bootstrap  # noqa: F401

import numpy as np
import torch

# 小批量 4x4 卷积在 CPU 上使用单线程更稳定、更快。
torch.set_num_threads(1)

from src.common.plotting import plot_curves, plot_score_hist, plot_tile_distribution
from src.common.runner import EvalSummary, evaluate_agent
from src.rl.dqn import DQNAgent, DQNConfig, ExpectimaxShieldAgent, FeatureDuelingQNet, train_dqn
from src.supervised.model import PolicyNet


SUMMARY_PATH = os.path.join("results", "summary.json")


def _summary_to_dict(summ: EvalSummary) -> Dict[str, Any]:
    return {
        "n_games": int(summ.n_games),
        "avg_score": float(summ.avg_score),
        "std_score": float(summ.std_score),
        "max_score": int(summ.max_score),
        "avg_max_tile": float(summ.avg_max_tile),
        "avg_steps": float(summ.avg_steps),
        "tile_distribution": {str(int(k)): int(v) for k, v in summ.tile_distribution.items()},
        "scores": [int(s) for s in summ.scores],
    }


def _write_json(path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _merge_project_summary(rl_block: Dict[str, Any]) -> None:
    data: Dict[str, Any] = {}
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    data["rl"] = rl_block
    _write_json(SUMMARY_PATH, data)


def main() -> None:
    p = argparse.ArgumentParser(description="DQN / Double DQN / Dueling DQN 强化学习训练 2048 智能体")
    p.add_argument("--steps", type=int, default=60000, help="环境交互步数")
    p.add_argument("--init", default=None, help="可选：监督学习 policy.pt，用作 Q 网络热启动")
    p.add_argument("--outdir", default="results/rl")
    p.add_argument("--model-out", default=None)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=100000)
    p.add_argument("--learn-start", type=int, default=128)
    p.add_argument("--train-freq", type=int, default=8)
    p.add_argument("--target-update", type=int, default=500)

    p.add_argument("--eps-start", type=float, default=0.50)
    p.add_argument("--eps-end", type=float, default=0.03)
    p.add_argument("--eps-decay-steps", type=int, default=45000)
    p.add_argument("--reward-scale", type=float, default=1.0 / 16.0)
    p.add_argument("--invalid-penalty", type=float, default=2.0)
    p.add_argument("--terminal-penalty", type=float, default=10.0)
    p.add_argument("--potential-scale", type=float, default=0.20)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--dqn", action="store_true", help="使用普通 DQN；默认使用 Double DQN")

    p.add_argument("--network", default="feature_dueling", choices=["feature_dueling", "resnet"], help="Q 网络结构；默认使用更快的特征版 Dueling 网络")
    p.add_argument("--n-step", type=int, default=3, help="n-step TD 回报步数")
    p.add_argument("--no-prioritized-replay", action="store_true", help="关闭优先经验回放 PER")
    p.add_argument("--per-alpha", type=float, default=0.60)
    p.add_argument("--per-beta-start", type=float, default=0.40)
    p.add_argument("--per-beta-end", type=float, default=1.00)
    p.add_argument("--expert-mix-start", type=float, default=0.35, help="训练早期专家探索比例；属于 DQfD/离策略 RL 辅助")
    p.add_argument("--expert-mix-end", type=float, default=0.03)
    p.add_argument("--expert-mix-decay-steps", type=int, default=30000)
    p.add_argument("--expert-margin-weight", type=float, default=0.02, help="DQfD large-margin 辅助损失权重")
    p.add_argument("--expert-margin", type=float, default=0.80)
    p.add_argument("--expert-pretrain-steps", type=int, default=80, help="DQfD 专家 demonstration 预训练更新次数")
    p.add_argument("--expert-pretrain-transitions", type=int, default=1000, help="DQfD 专家 demonstration transition 数")

    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument(
        "--final-eval-games",
        "--final-games",
        dest="final_eval_games",
        type=int,
        default=20,
        help="最终评估局数；--final-games 为兼容旧命令的别名",
    )
    p.add_argument("--safe-eval", action="store_true", help="训练中 checkpoint 评估也使用 Q-Expectimax 安全护栏")
    p.add_argument("--safe-depth", type=int, default=1, help="最终 DQN+Expectimax 安全护栏的搜索深度")
    p.add_argument("--safe-q-weight", type=float, default=0.0)
    p.add_argument("--safe-heuristic-weight", type=float, default=1.00)
    p.add_argument("--safe-score-weight", type=float, default=0.35)
    p.add_argument("--safe-max-chance-cells", type=int, default=6, help="安全护栏机会节点最多展开的空格数")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    model_out = args.model_out or os.path.join(args.outdir, "dqn.pt")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切换到 CPU")
        device = "cpu"

    init_state = None
    if args.init:
        if not os.path.exists(args.init):
            raise FileNotFoundError(f"找不到初始化模型: {args.init}")
        init_state = torch.load(args.init, map_location=device)

    cfg = DQNConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        buffer_size=args.buffer_size,
        learn_start=args.learn_start,
        train_freq=args.train_freq,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        reward_scale=args.reward_scale,
        invalid_penalty=args.invalid_penalty,
        terminal_penalty=args.terminal_penalty,
        potential_scale=args.potential_scale,
        grad_clip=args.grad_clip,
        double_dqn=not args.dqn,
        network=args.network,
        n_step=args.n_step,
        prioritized_replay=not args.no_prioritized_replay,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_end=args.per_beta_end,
        expert_mix_start=args.expert_mix_start,
        expert_mix_end=args.expert_mix_end,
        expert_mix_decay_steps=args.expert_mix_decay_steps,
        expert_margin_weight=args.expert_margin_weight,
        expert_margin=args.expert_margin,
        expert_pretrain_steps=args.expert_pretrain_steps,
        expert_pretrain_transitions=args.expert_pretrain_transitions,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        eval_use_safe_policy=args.safe_eval,
        device=device,
        seed=args.seed,
    )

    print("=== 强化学习配置 ===")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print(f"算法: {'Dueling Double DQN' if cfg.double_dqn and cfg.network == 'feature_dueling' else ('Double DQN' if cfg.double_dqn else 'DQN')}")

    qnet, hist = train_dqn(cfg, init_state_dict=init_state, model_out=model_out, verbose=True)

    # 重新加载到一个干净的 eval 网络，避免训练进程中的优化器/线程状态影响最终评估速度。
    if cfg.network == "feature_dueling":
        eval_net = FeatureDuelingQNet()
    else:
        eval_net = PolicyNet()
        setattr(eval_net, "input_mode", "onehot")
    eval_net.load_state_dict(torch.load(model_out, map_location=device))
    qnet = eval_net.to(device).eval()

    print("\n=== 最终评估：Dueling DDQN（无 Expectimax）===", flush=True)
    pure_agent = DQNAgent(qnet, device=device, use_safe_policy=False)
    pure = evaluate_agent(pure_agent, n_games=args.final_eval_games, base_seed=20000, verbose=True)
    print(pure)

    print("\n=== 最终评估：DQN + Expectimax 安全护栏 ===")
    safe_agent = ExpectimaxShieldAgent(depth=args.safe_depth, score_weight=args.safe_score_weight, max_chance_cells=args.safe_max_chance_cells)
    safe = evaluate_agent(safe_agent, n_games=args.final_eval_games, base_seed=30000, verbose=True)
    print(safe)

    if hist.eval_steps:
        kind = "DQN+安全护栏" if cfg.eval_use_safe_policy else ("Double DQN" if cfg.double_dqn else "DQN")
        plot_curves(
            hist.eval_steps,
            {"平均得分": hist.eval_avg_score},
            f"{kind} 训练过程平均得分",
            "环境交互步数",
            "平均分",
            os.path.join(args.outdir, "dqn_score.png"),
        )
        plot_curves(
            hist.eval_steps,
            {"平均最大方块": hist.eval_avg_tile},
            f"{kind} 训练过程平均最大方块",
            "环境交互步数",
            "平均最大方块",
            os.path.join(args.outdir, "dqn_tile.png"),
        )
        plot_curves(
            hist.eval_steps,
            {"TD Loss": hist.loss_curve, "平均 |TD error|": hist.td_error_curve},
            f"{kind} TD Loss / TD Error",
            "环境交互步数",
            "数值",
            os.path.join(args.outdir, "dqn_loss.png"),
        )

    plot_score_hist(pure.scores, "Dueling DDQN 无Expectimax 最终分数分布", os.path.join(args.outdir, "dqn_pure_hist.png"), bins=10)
    plot_tile_distribution(pure.tile_distribution, "Dueling DDQN 无Expectimax 最大方块分布", os.path.join(args.outdir, "dqn_pure_tile.png"))
    plot_score_hist(safe.scores, "DQN + 安全护栏最终分数分布", os.path.join(args.outdir, "dqn_hist.png"), bins=10)
    plot_tile_distribution(safe.tile_distribution, "DQN + 安全护栏最大方块分布", os.path.join(args.outdir, "dqn_safe_tile.png"))

    rl_summary: Dict[str, Any] = {
        "algorithm": "Dueling Double DQN + PER + n-step" if cfg.double_dqn and cfg.network == "feature_dueling" else ("Double DQN" if cfg.double_dqn else "DQN"),
        "init_model": args.init,
        "model": model_out,
        "config": asdict(cfg),
        "eval_steps": [int(x) for x in hist.eval_steps],
        "eval_avg_score": [float(x) for x in hist.eval_avg_score],
        "eval_avg_tile": [float(x) for x in hist.eval_avg_tile],
        "eps_curve": [float(x) for x in hist.eps_curve],
        "expert_curve": [float(x) for x in hist.expert_curve],
        "loss_curve": [float(x) for x in hist.loss_curve],
        "td_error_curve": [float(x) for x in hist.td_error_curve],
        "final_pure": _summary_to_dict(pure),
        "final_safe": _summary_to_dict(safe),
    }

    _write_json(os.path.join(args.outdir, "summary.json"), rl_summary)
    _merge_project_summary(rl_summary)
    print(f"\n强化学习结果已保存到: {args.outdir}")
    print(f"项目总汇总已更新: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
    # 某些 PyTorch/OpenMP 组合在脚本结束后会残留非 daemon 线程，
    # 导致命令行卡住；这里在所有文件写完后强制退出，保证作业脚本可运行。
    import sys as _sys, os as _os
    _sys.stdout.flush()
    _sys.stderr.flush()
    _os._exit(0)
