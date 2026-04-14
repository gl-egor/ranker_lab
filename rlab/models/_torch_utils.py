"""
Общие утилиты для нейронных ранкеров (DCN-v2, DeepFM и следующих).

Содержит:
    - RankTableDataset      — DataFrame → тензоры
    - GroupBatchSampler     — батчи целыми группами (для listwise loss)
    - group_softmax_loss    — listwise log-softmax по группе
    - build_param_groups    — раздельные lr для эмбеддингов и плотных слоёв
    - build_scheduler       — фабрика lr-шедулеров (cosine / plateau / none)
    - train_neural_ranker   — общий training loop с early stopping

Принцип: всё, что относится к стандартной тренировке (loop / optimizer /
scheduler / валидация / early stop), живёт здесь. В обёртках конкретных
моделей (dcnv2_ranker, deepfm_ranker) остаётся только архитектура.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from rlab.models.base import FeatureSpec


# ═════════════════════════════════════════════════════════════════════════════
# Dataset / Sampler / Loss (как было)
# ═════════════════════════════════════════════════════════════════════════════
class RankTableDataset(Dataset):
    """
    Оборачивает rank_table DataFrame в тензоры.
    Контракт: df имеет user_idx, item_idx, numerical_cols, label, group_id;
    группы идут подряд (инвариант loader.py); scaler уже зафитчен на train.
    """
    def __init__(self, df: pd.DataFrame, feature_spec: FeatureSpec, scaler: StandardScaler):
        self.user_idx = torch.from_numpy(df["user_idx"].values.astype(np.int64))
        self.item_idx = torch.from_numpy(df["item_idx"].values.astype(np.int64))

        num_cols = feature_spec.numerical_cols
        if num_cols:
            num = df[num_cols].values.astype(np.float32)
            num = scaler.transform(num).astype(np.float32)
            self.num_feat = torch.from_numpy(num)
        else:
            self.num_feat = torch.zeros((len(df), 0), dtype=torch.float32)

        self.labels   = torch.from_numpy(df[feature_spec.target_col].values.astype(np.float32))
        self.group_id = torch.from_numpy(df[feature_spec.group_col].values.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_idx[idx], self.item_idx[idx], self.num_feat[idx],
            self.labels[idx],   self.group_id[idx],
        )


class GroupBatchSampler(torch.utils.data.Sampler):
    """
    Батчи индексов целыми группами. Реальный размер батча =
    groups_per_batch * (1 + n_neg).
    """
    def __init__(self, group_ids: np.ndarray, groups_per_batch: int,
                 shuffle: bool, seed: int = 0):
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        order = np.argsort(group_ids, kind="stable")
        sorted_groups = group_ids[order]
        _, first_idx = np.unique(sorted_groups, return_index=True)
        bounds = np.append(first_idx, len(order))
        self.group_slices: list[np.ndarray] = [
            order[bounds[i]:bounds[i + 1]] for i in range(len(first_idx))
        ]

    def __iter__(self):
        idxs = np.arange(len(self.group_slices))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for i in range(0, len(idxs), self.groups_per_batch):
            chunk = idxs[i:i + self.groups_per_batch]
            yield np.concatenate([self.group_slices[j] for j in chunk]).tolist()

    def __len__(self):
        return (len(self.group_slices) + self.groups_per_batch - 1) \
               // self.groups_per_batch


def group_softmax_loss(
    scores: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
) -> torch.Tensor:
    """Listwise: -log P(positive) под log-softmax по группе, среднее по группам."""
    unique = torch.unique(groups)
    losses = []
    for gid in unique:
        mask = groups == gid
        log_prob = F.log_softmax(scores[mask], dim=0)
        losses.append(-(log_prob * labels[mask]).sum())
    return torch.stack(losses).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Optimizer / Scheduler factories
# ═════════════════════════════════════════════════════════════════════════════
def build_param_groups(
    model: nn.Module,
    lr: float,
    emb_lr_mult: float = 1.0,
    emb_weight_decay: float | None = None,
    weight_decay: float = 1e-5,
) -> list[dict[str, Any]]:
    """
    Делит параметры модели на группы для оптимизатора.

    Группа 'emb' — все nn.Embedding слои (распознаём по 'emb' в имени).
    Группа 'dense' — всё остальное (Linear, LayerNorm, биасы и т.д.).

    Параметры:
        lr               : базовый lr (для dense-группы).
        emb_lr_mult      : множитель lr для эмбеддингов.
                           1.0  → одинаковый lr (выключено).
                           0.1  → эмбеддинги учатся в 10 раз медленнее.
        emb_weight_decay : отдельный wd для эмбеддингов. None = тот же,
                           что и для dense. Полезно ставить БОЛЬШЕ для emb
                           (например, 1e-2), чтобы давить редкие
                           id-эмбеддинги к нулю.
        weight_decay     : wd для dense-группы.

    Зачем разделять:
        - Embedding-таблицы у нас огромные (миллионы строк), но
          в каждом батче активны только сотни. Большой lr на них
          приводит к шумным обновлениям тех немногих строк, что
          попали в батч; маленький — стабилизирует.
        - Dense-слои наоборот хорошо переваривают стандартный AdamW lr.
    """
    emb_params, dense_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "emb" in name:
            emb_params.append(param)
        else:
            dense_params.append(param)

    if emb_weight_decay is None:
        emb_weight_decay = weight_decay

    return [
        {"params": emb_params,
         "lr": lr * emb_lr_mult,
         "weight_decay": emb_weight_decay},
        {"params": dense_params,
         "lr": lr,
         "weight_decay": weight_decay},
    ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    n_epochs: int,
    n_steps_per_epoch: int = 1,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """
    Фабрика lr-шедулеров.

    kind:
        'none'    → None, lr константный.
        'cosine'  → CosineAnnealingLR, плавный спуск с lr_max до eta_min
                    за n_epochs * n_steps_per_epoch шагов. Шаги делаются
                    каждый батч (см. n_steps_per_epoch).
        'plateau' → ReduceLROnPlateau по val NDCG, factor=0.5, patience=1.
                    Шагается раз в эпоху, требует передачи метрики.
                    Удобен, когда заранее непонятно, сколько эпох будет
                    (early stop может выключить раньше cosine).
        'warmup_cosine' → линейный warmup на warmup_epochs эпох до lr_max,
                          затем cosine до eta_min до конца.

    kwargs:
        eta_min        : минимальный lr (для cosine), default 1e-6.
        warmup_epochs  : сколько эпох разогрева для warmup_cosine.
    """
    if kind == "none":
        return None

    if kind == "cosine":
        total_steps = n_epochs * n_steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get("eta_min", 1e-6),
        )

    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 1),
        )

    if kind == "warmup_cosine":
        warmup_epochs = kwargs.get("warmup_epochs", 1)
        total_steps   = n_epochs * n_steps_per_epoch
        warmup_steps  = warmup_epochs * n_steps_per_epoch
        eta_min       = kwargs.get("eta_min", 1e-6)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            # cosine от 1.0 до eta_min/lr_base за оставшиеся шаги
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler kind '{kind}'. "
                     f"Use one of: none, cosine, plateau, warmup_cosine.")


# ═════════════════════════════════════════════════════════════════════════════
# Универсальный training loop
# ═════════════════════════════════════════════════════════════════════════════
def train_neural_ranker(
    *,
    model: nn.Module,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    scaler: StandardScaler,
    params: dict[str, Any],
    seed: int,
    device: str,
    forward_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """
    Универсальный training loop для нейронных ранкеров.

    Контракт модели:
        forward_fn(model, user_idx, item_idx, num_feat) -> scores (B,)
        Если None — используется model(user_idx, item_idx, num_feat).
        Через forward_fn можно адаптировать модель с другой сигнатурой.

    Все гиперпараметры берутся из params (с дефолтами):
        lr, weight_decay, emb_lr_mult, emb_weight_decay
        max_epochs, patience, groups_per_batch, grad_clip
        scheduler ('none' | 'cosine' | 'plateau' | 'warmup_cosine')
        scheduler_kwargs (dict, передаётся в build_scheduler)

    Возвращает:
        train_history     : list[dict]
        train_time_sec    : float
        best_val_ndcg     : float
        best_state_dict   : dict с весами лучшей эпохи (уже загружен в model)
    """
    torch.manual_seed(seed)

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_ds = RankTableDataset(train_df, feature_spec, scaler)
    valid_ds = RankTableDataset(valid_df, feature_spec, scaler)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupBatchSampler(
            train_df[feature_spec.group_col].values,
            groups_per_batch=params["groups_per_batch"],
            shuffle=True, seed=seed,
        ),
        num_workers=0, pin_memory=(device == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_sampler=GroupBatchSampler(
            valid_df[feature_spec.group_col].values,
            groups_per_batch=params["groups_per_batch"],
            shuffle=False, seed=seed,
        ),
        num_workers=0,
    )

    # ── Optimizer с param groups ────────────────────────────────────────
    param_groups = build_param_groups(
        model,
        lr=params["lr"],
        emb_lr_mult=params.get("emb_lr_mult", 1.0),
        emb_weight_decay=params.get("emb_weight_decay", None),
        weight_decay=params.get("weight_decay", 1e-5),
    )
    opt = torch.optim.AdamW(param_groups)

    # ── Scheduler ───────────────────────────────────────────────────────
    sched_kind = params.get("scheduler", "none")
    sched_kwargs = params.get("scheduler_kwargs", {})
    scheduler = build_scheduler(
        opt, kind=sched_kind,
        n_epochs=params["max_epochs"],
        n_steps_per_epoch=len(train_loader),
        **sched_kwargs,
    )
    # Шагать ли scheduler каждый батч (cosine/warmup_cosine — да)
    # или раз в эпоху (plateau)?
    step_each_batch = sched_kind in ("cosine", "warmup_cosine")

    # ── Early stopping bookkeeping ──────────────────────────────────────
    best_ndcg = -1.0
    best_state: dict | None = None
    patience = 0
    history: list[dict] = []

    # импорт здесь — чтобы избежать циклической зависимости
    from rlab.eval.metrics import ranking_metrics

    if forward_fn is None:
        forward_fn = lambda m, u, i, n: m(u, i, n)

    t0 = time.time()
    for epoch in trange(1, params["max_epochs"] + 1, desc="epoch", unit="ep"):
        # ── train ───────────────────────────────────────────────────────
        model.train()
        losses = []
        for u, i, n, y, g in tqdm(train_loader, desc=f"  train e{epoch}",
                                   leave=False, unit="batch"):
            u, i, n = u.to(device), i.to(device), n.to(device)
            y, g    = y.to(device), g.to(device)

            scores = forward_fn(model, u, i, n)
            loss = group_softmax_loss(scores, y, g)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            opt.step()
            if scheduler is not None and step_each_batch:
                scheduler.step()
            losses.append(loss.item())

        # ── eval ────────────────────────────────────────────────────────
        val_scores, val_labels, val_groups = _predict_loader(
            model, valid_loader, device, forward_fn
        )
        metrics = ranking_metrics(val_scores, val_labels, val_groups, k=10)

        # текущий lr (после возможного step) — для логов
        cur_lrs = [g["lr"] for g in opt.param_groups]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_NDCG@10": metrics["NDCG"],
            "val_HR@10":   metrics["HR"],
            "lr_emb":   cur_lrs[0],
            "lr_dense": cur_lrs[1] if len(cur_lrs) > 1 else cur_lrs[0],
        })
        print(f"  [e{epoch}] loss={np.mean(losses):.4f} "
              f"val NDCG@10={metrics['NDCG']:.4f} "
              f"lr_emb={cur_lrs[0]:.2e}")

        # plateau шагает раз в эпоху по метрике
        if scheduler is not None and not step_each_batch:
            scheduler.step(metrics["NDCG"])

        # early stopping по val NDCG
        if metrics["NDCG"] > best_ndcg:
            best_ndcg = metrics["NDCG"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= params["patience"]:
                print(f"  early stop at epoch {epoch} "
                      f"(best NDCG@10={best_ndcg:.4f})")
                break

    # восстанавливаем лучшие веса
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_time_sec": time.time() - t0,
        "best_val_ndcg":  best_ndcg,
        "train_history":  history,
    }


def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    forward_fn: Callable,
):
    """Прогон по DataLoader, возвращает (scores, labels, groups) numpy."""
    model.eval()
    S, L, G = [], [], []
    with torch.no_grad():
        for u, i, n, y, g in loader:
            u, i, n = u.to(device), i.to(device), n.to(device)
            S.append(forward_fn(model, u, i, n).cpu().numpy())
            L.append(y.numpy()); G.append(g.numpy())
    return np.concatenate(S), np.concatenate(L), np.concatenate(G)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: предсказание на произвольном DataFrame (для Ranker.predict)
# ═════════════════════════════════════════════════════════════════════════════
def predict_dataframe(
    model: nn.Module,
    df: pd.DataFrame,
    feature_spec: FeatureSpec,
    scaler: StandardScaler,
    device: str,
    forward_fn: Callable | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Предсказания на тесте/любом DataFrame, без листового батчевания."""
    if forward_fn is None:
        forward_fn = lambda m, u, i, n: m(u, i, n)

    ds = RankTableDataset(df, feature_spec, scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    out = []
    with torch.no_grad():
        for u, i, n, _, _ in loader:
            u, i, n = u.to(device), i.to(device), n.to(device)
            out.append(forward_fn(model, u, i, n).cpu().numpy())
    return np.concatenate(out).astype(np.float32)