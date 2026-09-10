import os
import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.supervised.generate_data import generate_dataset
from src.supervised.train import train_supervised, TrainConfig
from src.supervised.model import NetAgent, load_net
from src.supervised.eval_policy import evaluate_agent


DEFAULT_DATA = "results/supervised/data.npz"
DEFAULT_MODEL = "results/supervised/policy.pt"
DEFAULT_SUMMARY = "results/supervised/summary.json"


def save_full_dataset(path: str, X, y, extras):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez_compressed(
        path,
        X=X,
        y=y,
        target_probs=extras["target_probs"],
        legal_mask=extras["legal_mask"],
        game_id=extras["game_id"],
        expert_scores=extras["expert_scores"],
    )

    print(f"[save data] 完整专家数据已保存到 {path}", flush=True)

    data = np.load(path)
    print("[check data keys]", data.files, flush=True)
    for k in data.files:
        print(" ", k, data[k].shape, data[k].dtype, flush=True)


def load_dataset(path: str):
    data = np.load(path)

    required = ["X", "y", "target_probs", "legal_mask", "game_id"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(
            f"{path} 缺少字段 {missing}。"
            "请重新生成完整 v2 数据，不能只用旧版 X/y 数据。"
        )

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    target_probs = data["target_probs"].astype(np.float32)
    legal_mask = data["legal_mask"].astype(np.float32)
    game_id = data["game_id"].astype(np.int64)

    print("[data]", path, flush=True)
    print(" X:", X.shape, X.dtype, flush=True)
    print(" y:", y.shape, y.dtype, flush=True)
    print(" target_probs:", target_probs.shape, target_probs.dtype, flush=True)
    print(" legal_mask:", legal_mask.shape, legal_mask.dtype, flush=True)
    print(" game_id:", game_id.shape, game_id.dtype, flush=True)
    print(" y counts:", np.bincount(y, minlength=4), flush=True)

    bad = legal_mask[np.arange(len(y)), y] <= 0
    print("[check] 专家动作被 legal_mask 判为非法的数量:", int(bad.sum()), flush=True)
    print("[check] 占比:", float(bad.mean()), flush=True)

    return X, y, target_probs, legal_mask, game_id


def generate_or_load_data(args):
    if args.reuse_data and os.path.exists(args.data_out):
        print(f"[reuse data] 使用已有数据: {args.data_out}", flush=True)
        return load_dataset(args.data_out)

    print("=" * 80)
    print("[1/3] 生成 Expectimax 专家数据", flush=True)
    print("=" * 80)

    X, y, extras = generate_dataset(
        n_samples=args.samples,
        expert_depth=args.depth,
        out_path=None,
        base_seed=args.seed,
        verbose=True,
        temperature=args.temperature,
        explore_prob=args.explore_prob,
        return_extras=True,
    )

    save_full_dataset(args.data_out, X, y, extras)

    return (
        X.astype(np.float32),
        y.astype(np.int64),
        extras["target_probs"].astype(np.float32),
        extras["legal_mask"].astype(np.float32),
        extras["game_id"].astype(np.int64),
    )


def train_policy(args, X, y, target_probs, legal_mask, game_id):
    print("=" * 80)
    print("[2/3] 监督学习训练策略网络", flush=True)
    print("=" * 80)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动切换到 CPU", flush=True)
        device = "cpu"

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        device=device,
        seed=args.seed,

        channels=args.channels,
        n_blocks=args.blocks,
        dropout=args.dropout,

        # 不要开 label smoothing：
        # 它和非法动作 mask=-1e9 结合会让 loss 异常变大。
        label_smoothing=0.0,

        soft_loss_weight=args.soft_weight,
        hard_loss_weight=args.hard_weight,

        augment=args.augment,
        early_stop_patience=args.early_stop_patience,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
    )

    net, history = train_supervised(
        X,
        y,
        config=config,
        model_out=args.model_out,
        target_probs=target_probs,
        legal_mask=legal_mask,
        game_id=game_id,
    )

    return net, history, device


def eval_policy(args, net, device):
    print("=" * 80)
    print("[3/3] 真实对局评估", flush=True)
    print("=" * 80)

    agent = NetAgent(net, device=device)
    result = evaluate_agent(agent, n_games=args.eval_games, verbose=True)
    return result


def save_summary(args, history, result, device):
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    summary = {
        "data": args.data_out,
        "model": args.model_out,
        "best_val_acc": max(history.val_acc) if history.val_acc else None,
        "last_val_acc": history.val_acc[-1] if history.val_acc else None,
        "best_val_loss": min(history.val_loss) if history.val_loss else None,
        "last_val_loss": history.val_loss[-1] if history.val_loss else None,
        "eval": result,
        "config": {
            "samples": args.samples,
            "depth": args.depth,
            "explore_prob": args.explore_prob,
            "temperature": args.temperature,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "channels": args.channels,
            "blocks": args.blocks,
            "dropout": args.dropout,
            "lr": args.lr,
            "soft_weight": args.soft_weight,
            "hard_weight": args.hard_weight,
            "augment": args.augment,
            "device": device,
        },
    }

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[save summary] {args.summary_out}", flush=True)


def eval_saved_model(args):
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动切换到 CPU", flush=True)
        device = "cpu"

    net = load_net(
        args.model_out,
        device=device,
        channels=args.channels,
        n_blocks=args.blocks,
        dropout=args.dropout,
    )

    agent = NetAgent(net, device=device)
    evaluate_agent(agent, n_games=args.eval_games, verbose=True)


def main():
    parser = argparse.ArgumentParser(description="监督学习：生成专家数据并训练 2048 策略网络")

    parser.add_argument("--samples", type=int, default=80000)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore-prob", type=float, default=0.07)
    parser.add_argument("--temperature", type=float, default=0.35)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # 最终采用的 5300+ 分模型参数
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--soft-weight", type=float, default=0.3)
    parser.add_argument("--hard-weight", type=float, default=0.7)

    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--data-out", type=str, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--summary-out", type=str, default=DEFAULT_SUMMARY)

    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="复用已有 results/supervised/data.npz，不重新生成专家数据",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只评估已有模型，不训练",
    )

    args = parser.parse_args()

    os.makedirs("results/supervised", exist_ok=True)

    if args.eval_only:
        eval_saved_model(args)
        return

    X, y, target_probs, legal_mask, game_id = generate_or_load_data(args)
    net, history, device = train_policy(args, X, y, target_probs, legal_mask, game_id)
    result = eval_policy(args, net, device)
    save_summary(args, history, result, device)


if __name__ == "__main__":
    main()
