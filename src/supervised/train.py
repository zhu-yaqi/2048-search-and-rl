"""监督学习 v2 训练代码。

核心改进：
1. 支持 Expectimax 软标签 target_probs，不再只学一个硬动作。
2. 训练时屏蔽非法动作，避免网络把概率浪费在非法方向。
3. 按 game_id 整局划分训练/验证集，避免相邻状态泄漏。
4. 可选棋盘翻转增强，并同步变换动作标签。
5. AdamW + CosineLR + label smoothing + gradient clipping + early stopping。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import PolicyNet


# 重要：这里假设动作编号是：
# 0 = 上, 1 = 下, 2 = 左, 3 = 右
# 如果你的 ACTIONS 编号不是这个顺序，必须改下面四个常量。
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    device: str = "cpu"
    seed: int = 0

    channels: int = 128
    n_blocks: int = 4
    dropout: float = 0.05

    label_smoothing: float = 0.0
    soft_loss_weight: float = 0.4
    hard_loss_weight: float = 0.6

    augment: bool = True
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    num_workers: int = 0


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _accuracy(logits: torch.Tensor, y: torch.Tensor, legal_mask: Optional[torch.Tensor] = None) -> float:
    if legal_mask is not None:
        logits = logits.masked_fill(legal_mask <= 0, -1e9)
    return (logits.argmax(dim=1) == y).float().mean().item()


def _remap_action_array(y: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.empty_like(y, dtype=np.int64)
    for i, a in enumerate(y):
        out[i] = mapping[int(a)]
    return out


def _remap_action_matrix(m: Optional[np.ndarray], mapping: dict[int, int]) -> Optional[np.ndarray]:
    if m is None:
        return None
    out = np.zeros_like(m)
    for old_a, new_a in mapping.items():
        out[:, new_a] = m[:, old_a]
    return out


def augment_2048_data(
    X: np.ndarray,
    y: np.ndarray,
    target_probs: Optional[np.ndarray] = None,
    legal_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """对训练集做翻转增强，并同步变换动作标签。

    只做翻转，不做 90 度旋转，避免动作编号不确定时太容易出错。
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int64)

    xs = [X]
    ys = [y]
    ps = [target_probs] if target_probs is not None else None
    ms = [legal_mask] if legal_mask is not None else None

    # 水平翻转：left <-> right
    map_h = {
        ACTION_UP: ACTION_UP,
        ACTION_DOWN: ACTION_DOWN,
        ACTION_LEFT: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_LEFT,
    }
    xs.append(np.flip(X, axis=3).copy())
    ys.append(_remap_action_array(y, map_h))
    if ps is not None:
        ps.append(_remap_action_matrix(target_probs, map_h))
    if ms is not None:
        ms.append(_remap_action_matrix(legal_mask, map_h))

    # 垂直翻转：up <-> down
    map_v = {
        ACTION_UP: ACTION_DOWN,
        ACTION_DOWN: ACTION_UP,
        ACTION_LEFT: ACTION_LEFT,
        ACTION_RIGHT: ACTION_RIGHT,
    }
    xs.append(np.flip(X, axis=2).copy())
    ys.append(_remap_action_array(y, map_v))
    if ps is not None:
        ps.append(_remap_action_matrix(target_probs, map_v))
    if ms is not None:
        ms.append(_remap_action_matrix(legal_mask, map_v))

    # 180 度翻转：up <-> down, left <-> right
    map_hv = {
        ACTION_UP: ACTION_DOWN,
        ACTION_DOWN: ACTION_UP,
        ACTION_LEFT: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_LEFT,
    }
    xs.append(np.flip(np.flip(X, axis=2), axis=3).copy())
    ys.append(_remap_action_array(y, map_hv))
    if ps is not None:
        ps.append(_remap_action_matrix(target_probs, map_hv))
    if ms is not None:
        ms.append(_remap_action_matrix(legal_mask, map_hv))

    X_aug = np.concatenate(xs, axis=0).astype(np.float32)
    y_aug = np.concatenate(ys, axis=0).astype(np.int64)
    p_aug = np.concatenate(ps, axis=0).astype(np.float32) if ps is not None else None
    m_aug = np.concatenate(ms, axis=0).astype(np.float32) if ms is not None else None

    return (
        np.ascontiguousarray(X_aug),
        np.ascontiguousarray(y_aug),
        np.ascontiguousarray(p_aug) if p_aug is not None else None,
        np.ascontiguousarray(m_aug) if m_aug is not None else None,
    )


def _split_indices(
    n: int,
    val_ratio: float,
    seed: int,
    game_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if game_id is None:
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        return idx[n_val:], idx[:n_val]

    games = np.unique(game_id)
    rng.shuffle(games)
    n_val_games = max(1, int(len(games) * val_ratio))
    val_games = set(games[:n_val_games].tolist())

    val_mask = np.array([int(g) in val_games for g in game_id], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]

    return train_idx, val_idx


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    target_probs: Optional[np.ndarray],
    legal_mask: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    device: str,
    num_workers: int,
) -> DataLoader:
    if target_probs is None:
        target_probs = np.zeros((len(X), 4), dtype=np.float32)
    if legal_mask is None:
        legal_mask = np.ones((len(X), 4), dtype=np.float32)

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
        torch.from_numpy(target_probs).float(),
        torch.from_numpy(legal_mask).float(),
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )



def _make_soft_targets(
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    把 target_probs 处理成真正的概率分布。

    支持两种输入：
    1. 已经是概率：[0.7, 0.2, 0.1, 0.0]
    2. Expectimax 原始分数：[123000, 98000, 125000, 50000]

    如果是原始分数，就先做每行标准化，再 softmax。
    """
    target_probs = target_probs.float()
    legal_mask = legal_mask.float()

    row_sum = target_probs.sum(dim=1, keepdim=True)

    looks_like_probs = (
        (target_probs >= -1e-6).all()
        and (target_probs <= 1.0 + 1e-6).all()
        and torch.all(torch.abs(row_sum - 1.0) < 1e-3)
    )

    if looks_like_probs:
        probs = target_probs * legal_mask
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(eps)
        return probs

    # 否则认为 target_probs 是 Expectimax 原始动作评分
    scores = target_probs.clone()

    # 非法动作不给概率
    scores = scores.masked_fill(legal_mask <= 0, -1e9)

    # 对每一行做标准化，避免原始分数尺度太大
    legal_count = legal_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    safe_scores = scores.masked_fill(legal_mask <= 0, 0.0)
    mean = safe_scores.sum(dim=1, keepdim=True) / legal_count

    var = (((safe_scores - mean) * legal_mask) ** 2).sum(dim=1, keepdim=True) / legal_count
    std = torch.sqrt(var + eps)

    z = (safe_scores - mean) / std
    z = z.masked_fill(legal_mask <= 0, -1e9)

    probs = torch.softmax(z / temperature, dim=1)
    return probs


def _loss_fn(
    logits: torch.Tensor,
    y: torch.Tensor,
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    ce_loss: nn.Module,
    config: TrainConfig,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(legal_mask <= 0, -1e9)

    hard_loss = ce_loss(masked_logits, y)

    has_soft = target_probs.abs().sum(dim=1).mean() > 0
    if not has_soft:
        return hard_loss

    soft_targets = _make_soft_targets(
        target_probs=target_probs,
        legal_mask=legal_mask,
        temperature=1.5,
    )

    logp = torch.log_softmax(masked_logits, dim=1)
    soft_loss = -(soft_targets * logp).sum(dim=1).mean()

    return config.hard_loss_weight * hard_loss + config.soft_loss_weight * soft_loss



def train_supervised(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig = TrainConfig(),
    model_out: Optional[str] = None,
    target_probs: Optional[np.ndarray] = None,
    legal_mask: Optional[np.ndarray] = None,
    game_id: Optional[np.ndarray] = None,
) -> Tuple[PolicyNet, TrainHistory]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = config.device

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    print("[check] X shape:", X.shape)
    print("[check] y shape:", y.shape)
    print("[check] y min/max:", y.min(), y.max())
    print("[check] y counts:", np.bincount(y, minlength=4))
    if legal_mask is not None:
        legal_mask_check = np.asarray(legal_mask, dtype=np.float32)
        bad = legal_mask_check[np.arange(len(y)), y] <= 0
        print("[check] 专家动作被 legal_mask 判为非法的数量:", int(bad.sum()))
        print("[check] 占比:", float(bad.mean()))

    target_probs = None if target_probs is None else np.asarray(target_probs, dtype=np.float32)
    legal_mask = None if legal_mask is None else np.asarray(legal_mask, dtype=np.float32)
    game_id = None if game_id is None else np.asarray(game_id, dtype=np.int64)

    n = len(X)
    train_idx, val_idx = _split_indices(n, config.val_ratio, config.seed, game_id=game_id)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    p_train = target_probs[train_idx] if target_probs is not None else None
    p_val = target_probs[val_idx] if target_probs is not None else None

    m_train = legal_mask[train_idx] if legal_mask is not None else None
    m_val = legal_mask[val_idx] if legal_mask is not None else None

    if config.augment:
        X_train, y_train, p_train, m_train = augment_2048_data(
            X_train, y_train, target_probs=p_train, legal_mask=m_train
        )
        print(f"[augment] 训练集增强后: X={X_train.shape}, y={y_train.shape}", flush=True)

    train_loader = _make_loader(
        X_train, y_train, p_train, m_train,
        batch_size=config.batch_size,
        shuffle=True,
        device=device,
        num_workers=config.num_workers,
    )
    val_loader = _make_loader(
        X_val, y_val, p_val, m_val,
        batch_size=1024,
        shuffle=False,
        device=device,
        num_workers=config.num_workers,
    )

    net = PolicyNet(
        channels=config.channels,
        n_blocks=config.n_blocks,
        dropout=config.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    ce_loss = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, config.epochs),
        eta_min=config.lr * 0.05,
    )

    history = TrainHistory()
    best_val_loss = float("inf")
    best_state = None

    best_val_acc = -1.0
    best_acc_state = None

    bad_epochs = 0

    for epoch in range(config.epochs):
        net.train()
        losses, accs = [], []

        for xb, yb, pb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = net(xb)
            loss = _loss_fn(logits, yb, pb, mb, ce_loss, config)
            loss.backward()

            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)

            opt.step()

            losses.append(loss.item())
            accs.append(_accuracy(logits.detach(), yb, mb))

        net.eval()
        vlosses, vaccs = [], []

        with torch.no_grad():
            for xb, yb, pb, mb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pb = pb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)

                logits = net(xb)
                loss = _loss_fn(logits, yb, pb, mb, ce_loss, config)

                vlosses.append(loss.item())
                vaccs.append(_accuracy(logits, yb, mb))

        scheduler.step()

        train_loss = float(np.mean(losses)) if losses else 0.0
        train_acc = float(np.mean(accs)) if accs else 0.0
        val_loss = float(np.mean(vlosses)) if vlosses else 0.0
        val_acc = float(np.mean(vaccs)) if vaccs else 0.0

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)

        lr_now = opt.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch + 1:02d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"lr={lr_now:.2e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"[best acc] val_acc={best_val_acc:.3f}", flush=True)

        if bad_epochs >= config.early_stop_patience:
            print(f"[early stop] 验证集 {config.early_stop_patience} 轮没有提升，停止训练。", flush=True)
            break

    if model_out:
        _ensure_parent_dir(model_out)

        base, ext = os.path.splitext(model_out)
        if ext == "":
            ext = ".pt"

        if best_state is not None:
            torch.save(best_state, f"{base}_best_loss{ext}")
            print(f"[save] best_loss 模型已保存到 {base}_best_loss{ext}", flush=True)

        if best_acc_state is not None:
            torch.save(best_acc_state, f"{base}_best_acc{ext}")
            print(f"[save] best_acc 模型已保存到 {base}_best_acc{ext}", flush=True)

        # 默认用 best_acc，因为真实策略更看重动作选择准确率
        if best_acc_state is not None:
            net.load_state_dict(best_acc_state)
            torch.save(net.state_dict(), model_out)
            print(f"[save] 默认模型使用 best_acc，已保存到 {model_out}", flush=True)
        elif best_state is not None:
            net.load_state_dict(best_state)
            torch.save(net.state_dict(), model_out)
            print(f"[save] 默认模型使用 best_loss，已保存到 {model_out}", flush=True)
    else:
        if best_acc_state is not None:
            net.load_state_dict(best_acc_state)
        elif best_state is not None:
            net.load_state_dict(best_state)

    return net, history
