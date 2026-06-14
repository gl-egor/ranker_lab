"""
Contextual adaptive stacking: gating network + weighted combination.

GatingNetwork(context) -> softmax weights over experts.
Final score = sum(w_i * calibrated_score_i).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from rlab.models._torch_utils import GroupBatchSampler, group_softmax_loss
from rlab.models.base import FeatureSpec


DEFAULT_CONTEXT_FEATURES = [
    "history_len",
    "item_popularity_log",
    "user_interaction_count",
    "item_mean_rating",
    "user_mean_rating",
]


class GatingNetwork(nn.Module):
    """Context features -> expert weights via MLP + softmax."""

    def __init__(
        self,
        context_dim: int,
        n_experts: int,
        hidden: int = 32,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = context_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, n_experts))
        self.mlp = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(context), dim=-1)


class MetaRankTableDataset(Dataset):
    """Dataset: base scores + context features + labels + groups."""

    def __init__(
        self,
        base_scores: np.ndarray,
        context: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray,
    ):
        self.base_scores = torch.from_numpy(base_scores.astype(np.float32))
        self.context = torch.from_numpy(context.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.groups = torch.from_numpy(groups.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.base_scores[idx],
            self.context[idx],
            self.labels[idx],
            self.groups[idx],
        )


def _predict_loader(
    model: GatingNetwork,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores_list, weights_list, labels_list, groups_list = [], [], [], []
    with torch.no_grad():
        for base_s, ctx, y, g in loader:
            base_s = base_s.to(device)
            ctx = ctx.to(device)
            w = model(ctx)
            final = (w * base_s).sum(dim=-1)
            scores_list.append(final.cpu().numpy())
            weights_list.append(w.cpu().numpy())
            labels_list.append(y.numpy())
            groups_list.append(g.numpy())
    return (
        np.concatenate(scores_list),
        np.concatenate(weights_list),
        np.concatenate(labels_list),
        np.concatenate(groups_list),
    )


def train_gating_ranker(
    *,
    train_base_scores: np.ndarray,
    train_context: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    valid_base_scores: np.ndarray,
    valid_context: np.ndarray,
    valid_labels: np.ndarray,
    valid_groups: np.ndarray,
    n_experts: int,
    params: dict[str, Any],
    seed: int,
    device: str = "cpu",
) -> tuple[GatingNetwork, StandardScaler, dict[str, Any]]:
    """
    Обучает gating network на OOF train-скорах, early stopping по valid NDCG@10.
    """
    torch.manual_seed(seed)
    context_scaler = StandardScaler()
    train_ctx = context_scaler.fit_transform(train_context).astype(np.float32)
    valid_ctx = context_scaler.transform(valid_context).astype(np.float32)

    context_dim = train_ctx.shape[1]
    model = GatingNetwork(
        context_dim=context_dim,
        n_experts=n_experts,
        hidden=params.get("gate_hidden", 32),
        n_layers=params.get("gate_layers", 2),
        dropout=params.get("gate_dropout", 0.1),
    ).to(device)

    train_ds = MetaRankTableDataset(
        train_base_scores, train_ctx, train_labels, train_groups,
    )
    valid_ds = MetaRankTableDataset(
        valid_base_scores, valid_ctx, valid_labels, valid_groups,
    )
    groups_per_batch = params.get("groups_per_batch", 256)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=GroupBatchSampler(
            train_groups, groups_per_batch, shuffle=True, seed=seed,
        ),
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_sampler=GroupBatchSampler(
            valid_groups, groups_per_batch, shuffle=False, seed=seed,
        ),
        num_workers=0,
    )

    lr = params.get("gate_lr", 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    max_epochs = params.get("gate_epochs", 50)
    patience = params.get("gate_patience", 10)
    grad_clip = params.get("grad_clip", 5.0)

    from rlab.eval.metrics import ranking_metrics

    best_ndcg = -1.0
    best_state: dict | None = None
    wait = 0
    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for base_s, ctx, y, g in train_loader:
            base_s = base_s.to(device)
            ctx = ctx.to(device)
            y = y.to(device)
            g = g.to(device)

            w = model(ctx)
            scores = (w * base_s).sum(dim=-1)
            loss = group_softmax_loss(scores, y, g)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())

        val_scores, _, val_labels, val_groups_arr = _predict_loader(
            model, valid_loader, device,
        )
        metrics = ranking_metrics(val_scores, val_labels, val_groups_arr, k=10)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_NDCG@10": metrics["NDCG"],
        })

        if metrics["NDCG"] > best_ndcg:
            best_ndcg = metrics["NDCG"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    meta = {
        "train_time_sec": time.time() - t0,
        "best_val_ndcg": best_ndcg,
        "train_history": history,
        "n_gate_params": sum(p.numel() for p in model.parameters()),
    }
    return model, context_scaler, meta


def predict_gating_ranker(
    *,
    model: GatingNetwork,
    context_scaler: StandardScaler,
    base_scores: np.ndarray,
    context: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
    groups_per_batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (final_scores, expert_weights) both of length n_rows.
    expert_weights shape: (n_rows, n_experts).
    """
    ctx = context_scaler.transform(context).astype(np.float32)
    ds = MetaRankTableDataset(
        base_scores, ctx,
        labels.astype(np.float32),
        groups.astype(np.int64),
    )
    loader = DataLoader(
        ds,
        batch_sampler=GroupBatchSampler(
            groups, groups_per_batch, shuffle=False, seed=0,
        ),
        num_workers=0,
    )
    scores, weights, _, _ = _predict_loader(model, loader, device)
    return scores, weights


def extract_context_matrix(
    df: pd.DataFrame,
    context_features: list[str],
) -> np.ndarray:
    missing = [c for c in context_features if c not in df.columns]
    if missing:
        raise KeyError(f"Context features missing from DataFrame: {missing}")
    return df[context_features].values.astype(np.float32)
