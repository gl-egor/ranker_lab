"""
DCN-v2 под интерфейс Ranker.
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
class CrossLayer(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l. Один слой из DCN-V2."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCNv2(nn.Module):
    """
    Parallel DCN-V2: Cross и Deep сети идут параллельно от общего x0,
    потом их выходы конкатенируются и идут в скалярный скор.

    Принимает:
        user_idx: (B,) long
        item_idx: (B,) long
        num_feat: (B, n_num) float — уже отскейленные
    Возвращает:
        scores:   (B,) float
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_num_features: int,
        n_cross: int = 2,
        mlp_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)

        x0_dim = 2 * emb_dim + n_num_features
        self.ln = nn.LayerNorm(x0_dim)
        self.drop = nn.Dropout(dropout)

        self.cross_layers = nn.ModuleList(
            [CrossLayer(x0_dim) for _ in range(n_cross)]
        )

        mlp: list[nn.Module] = []
        prev = x0_dim
        for h in mlp_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*mlp)

        self.head = nn.Linear(x0_dim + prev, 1)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        num_feat: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x0 = torch.cat([u, i, num_feat], dim=-1)
        x0 = self.drop(self.ln(x0))

        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        deep_out = self.mlp(x0)
        combined = torch.cat([xl, deep_out], dim=-1)
        return self.head(combined).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS: dict[str, Any] = {
    "emb_dim":        32,
    "n_cross":        2,
    "mlp_dims":       (256, 128),
    "dropout":        0.1,
    "lr":             1e-3,
    "weight_decay":   1e-5,
    "max_epochs":     15,
    "patience":       3,
    "groups_per_batch": 128,
    "grad_clip":      1.0,
}


@register_model("dcnv2")
class DCNv2Ranker(Ranker):
    def __init__(self):
        self._model: DCNv2 | None = None
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

        # ── Scaler по train ──────────────────────────────────────────────
        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = StandardScaler().fit(train_df[num_cols].values)
        else:
            # dummy: даёт identity transform
            self._scaler = StandardScaler().fit(np.zeros((1, 0)))
        self._feature_spec = feature_spec

        # ── Модель ───────────────────────────────────────────────────────
        # cardinalities в feature_spec — это "max_idx + 1", т.е. размер
        # таблицы эмбеддингов.
        n_users = feature_spec.cardinalities.get("user_idx", 0)
        n_items = feature_spec.cardinalities.get("item_idx", 0)
        self._model = DCNv2(
            n_users=n_users, n_items=n_items,
            emb_dim=p["emb_dim"],
            n_num_features=len(num_cols),
            n_cross=p["n_cross"],
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
        ).to(self._device)

        # ── Data loaders ─────────────────────────────────────────────────
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

        # ── Optimizer ────────────────────────────────────────────────────
        opt = torch.optim.AdamW(
            self._model.parameters(),
            lr=p["lr"], weight_decay=p["weight_decay"],
        )

        # ── Training loop с early stopping ───────────────────────────────
        # Метрика для early stopping — NDCG@k на valid.
        # Импорт здесь, чтобы не создавать цикла.
        from rlab.eval.metrics import ranking_metrics

        best_ndcg = -1.0
        best_state = None
        patience = 0
        history = []

        t0 = time.time()
        for epoch in trange(1, p["max_epochs"] + 1, desc="epoch", unit="ep"):
            # train
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

            # eval
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

            # early stopping
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

        # восстанавливаем лучшие веса
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
        # большой батч — просто по позициям, group-структура на predict не важна
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
        """Внутренний helper: прогон по DataLoader с group-батчами,
        возвращает (scores, labels, groups) для расчёта метрик."""
        self._model.eval()
        S, L, G = [], [], []
        with torch.no_grad():
            for u, i, n, y, g in loader:
                u = u.to(self._device); i = i.to(self._device); n = n.to(self._device)
                S.append(self._model(u, i, n).cpu().numpy())
                L.append(y.numpy()); G.append(g.numpy())
        return np.concatenate(S), np.concatenate(L), np.concatenate(G)
