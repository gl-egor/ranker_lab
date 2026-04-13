"""
DeepFM под интерфейс Ranker.

Классический DeepFM (Guo et al., 2017): разделяемые эмбеддинги для
категориальных фичей идут параллельно в:
  - FM-часть (линейный член + попарные dot-product взаимодействия),
  - Deep-часть (MLP по конкатенации всех эмбеддингов и numerical фичей).
Финальный скор = линейный + FM + deep.

У нас 2 категориальных поля (user_idx, item_idx), поэтому FM-часть
сводится к user·item + линейные члены. На больших датасетах это
эквивалентно MF + биасы, а общий вклад модели даёт deep-часть.

Реализация переиспользует RankTableDataset/GroupBatchSampler/loss
из _torch_utils — только архитектура своя.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from rlab.models._torch_utils import (
    GroupBatchSampler,
    RankTableDataset,
    group_softmax_loss,
)
from rlab.models.base import FeatureSpec, Ranker, register_model


# ═════════════════════════════════════════════════════════════════════════════
# 1. Архитектура
# ═════════════════════════════════════════════════════════════════════════════
class DeepFM(nn.Module):
    """
    Параметры:
        n_users, n_items : размер таблиц эмбеддингов (max_idx + 1)
        emb_dim          : размерность embedding для FM и deep
        n_num_features   : numerical фичи (добавятся в deep-часть)
        mlp_dims         : скрытые слои MLP
        dropout          : dropout в MLP

    Выход: scalar score на пару (user, item) с учётом numerical-фичей.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_num_features: int,
        mlp_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        # ── FM-часть ────────────────────────────────────────────────────
        # Эмбеддинги для попарных взаимодействий (user × item).
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        # Линейные члены (одномерные биасы по id).
        self.user_bias = nn.Embedding(n_users, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items, 1, padding_idx=0)
        # Линейный скор по numerical-фичам (вклад в «первый порядок»).
        self.num_linear = nn.Linear(n_num_features, 1) if n_num_features > 0 else None
        # Глобальный bias.
        self.global_bias = nn.Parameter(torch.zeros(1))

        # ── Deep-часть ──────────────────────────────────────────────────
        # MLP поверх конкатенации эмбеддингов и numerical.
        deep_in = 2 * emb_dim + n_num_features
        self.ln = nn.LayerNorm(deep_in)
        mlp: list[nn.Module] = []
        prev = deep_in
        for h in mlp_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        mlp.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        num_feat: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)           # (B, d)
        i = self.item_emb(item_idx)           # (B, d)

        # FM: линейные члены + попарное dot-product взаимодействие.
        # Для 2 полей формула sum-square-sum упрощается до u·i.
        fm_interact = (u * i).sum(dim=-1)                  # (B,)
        linear = (
            self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
            + self.global_bias
        )                                                  # (B,)
        if self.num_linear is not None:
            linear = linear + self.num_linear(num_feat).squeeze(-1)

        # Deep: MLP по конкатенации.
        deep_in = torch.cat([u, i, num_feat], dim=-1)
        deep_in = self.ln(deep_in)
        deep_out = self.mlp(deep_in).squeeze(-1)           # (B,)

        return linear + fm_interact + deep_out


# ═════════════════════════════════════════════════════════════════════════════
# 2. Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS: dict[str, Any] = {
    "emb_dim":        32,
    "mlp_dims":       (256, 128),
    "dropout":        0.1,
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "max_epochs":     15,
    "patience":       3,
    "groups_per_batch": 128,
    "grad_clip":      1.0,
}


@register_model("deepfm")
class DeepFMRanker(Ranker):
    """
    Обёртка DeepFM. Структурно идентична DCNv2Ranker — единственное
    отличие в архитектуре модели и дефолтах гиперпараметров.
    """
    def __init__(self):
        self._model: DeepFM | None = None
        self._scaler: StandardScaler | None = None
        self._feature_spec: FeatureSpec | None = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        p = {**DEFAULT_PARAMS, **params}
        torch.manual_seed(seed)

        # Scaler: fit на train, transform на всех.
        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = StandardScaler().fit(train_df[num_cols].values)
        else:
            self._scaler = StandardScaler().fit(np.zeros((1, 0)))
        self._feature_spec = feature_spec

        n_users = feature_spec.cardinalities.get("user_idx", 0)
        n_items = feature_spec.cardinalities.get("item_idx", 0)
        self._model = DeepFM(
            n_users=n_users, n_items=n_items,
            emb_dim=p["emb_dim"],
            n_num_features=len(num_cols),
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
        ).to(self._device)

        train_ds = RankTableDataset(train_df, feature_spec, self._scaler)
        valid_ds = RankTableDataset(valid_df, feature_spec, self._scaler)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=GroupBatchSampler(
                train_df[feature_spec.group_col].values,
                groups_per_batch=p["groups_per_batch"],
                shuffle=True, seed=seed,
            ),
            num_workers=0, pin_memory=(self._device == "cuda"),
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_sampler=GroupBatchSampler(
                valid_df[feature_spec.group_col].values,
                groups_per_batch=p["groups_per_batch"],
                shuffle=False, seed=seed,
            ),
            num_workers=0,
        )

        opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=p["lr"], weight_decay=p["weight_decay"],
        )

        from rlab.eval.metrics import ranking_metrics

        best_ndcg = -1.0
        best_state = None
        patience = 0
        history = []

        t0 = time.time()
        for epoch in trange(1, p["max_epochs"] + 1, desc="epoch", unit="ep"):
            self._model.train()
            losses = []
            for u, i, n, y, g in tqdm(train_loader, desc=f"  train e{epoch}",
                                       leave=False, unit="batch"):
                u, i, n = u.to(self._device), i.to(self._device), n.to(self._device)
                y, g    = y.to(self._device), g.to(self._device)

                scores = self._model(u, i, n)
                loss = group_softmax_loss(scores, y, g)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), p["grad_clip"])
                opt.step()
                losses.append(loss.item())

            val_scores, val_labels, val_groups = self._predict_loader(valid_loader)
            metrics = ranking_metrics(val_scores, val_labels, val_groups, k=10)
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_NDCG@10": metrics["NDCG"],
                "val_HR@10":   metrics["HR"],
            })
            print(f"  [e{epoch}] loss={np.mean(losses):.4f} "
                  f"val NDCG@10={metrics['NDCG']:.4f}")

            if metrics["NDCG"] > best_ndcg:
                best_ndcg = metrics["NDCG"]
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self._model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= p["patience"]:
                    print(f"  early stop at epoch {epoch} "
                          f"(best NDCG@10={best_ndcg:.4f})")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return {
            "train_time_sec": time.time() - t0,
            "best_val_ndcg":  best_ndcg,
            "train_history":  history,
        }

    # ─────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, feature_spec: FeatureSpec) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        ds = RankTableDataset(df, feature_spec, self._scaler)
        loader = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=0)

        self._model.eval()
        scores = []
        with torch.no_grad():
            for u, i, n, _, _ in loader:
                u = u.to(self._device); i = i.to(self._device); n = n.to(self._device)
                scores.append(self._model(u, i, n).cpu().numpy())
        return np.concatenate(scores).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def n_params(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())

    # ─────────────────────────────────────────────────────────────────────
    def _predict_loader(self, loader: DataLoader):
        self._model.eval()
        S, L, G = [], [], []
        with torch.no_grad():
            for u, i, n, y, g in loader:
                u = u.to(self._device); i = i.to(self._device); n = n.to(self._device)
                S.append(self._model(u, i, n).cpu().numpy())
                L.append(y.numpy()); G.append(g.numpy())
        return np.concatenate(S), np.concatenate(L), np.concatenate(G)
