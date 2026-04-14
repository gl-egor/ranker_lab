"""
DeepFM под интерфейс Ranker.

Архитектура: линейный + FM (для 2 категориальных полей сводится к u·i)
+ Deep MLP. Тренировка — через общий train_neural_ranker.

Гиперпараметры через ModelConfig.params: те же, что у DCN-v2, кроме
n_cross (нет cross-сети). См. DEFAULT_PARAMS.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from rlab.models._torch_utils import (
    predict_dataframe,
    train_neural_ranker,
)
from rlab.models.base import FeatureSpec, Ranker, register_model


# ═════════════════════════════════════════════════════════════════════════════
# Архитектура
# ═════════════════════════════════════════════════════════════════════════════
class DeepFM(nn.Module):
    """
    Линейный (биасы по user/item + Linear по numerical) +
    FM-взаимодействие (u·i для двух категориальных полей) +
    Deep MLP по конкатенации эмбеддингов и numerical-фичей.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_num_features: int,
        mlp_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        # FM-эмбеддинги (для попарного взаимодействия)
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        # Линейные биасы по id
        self.user_bias = nn.Embedding(n_users, 1, padding_idx=0)
        self.item_bias = nn.Embedding(n_items, 1, padding_idx=0)
        # Линейный вклад numerical-фичей
        self.num_linear = nn.Linear(n_num_features, 1) if n_num_features > 0 else None
        # Глобальный bias
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Deep-часть
        deep_in = 2 * emb_dim + n_num_features
        self.ln = nn.LayerNorm(deep_in) if layer_norm else nn.Identity()
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
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)

        # FM: попарное взаимодействие u·i + линейные члены
        fm_interact = (u * i).sum(dim=-1)
        linear = (
            self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
            + self.global_bias
        )
        if self.num_linear is not None:
            linear = linear + self.num_linear(num_feat).squeeze(-1)

        # Deep
        deep_in = torch.cat([u, i, num_feat], dim=-1)
        deep_in = self.ln(deep_in)
        deep_out = self.mlp(deep_in).squeeze(-1)

        return linear + fm_interact + deep_out


# ═════════════════════════════════════════════════════════════════════════════
# Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS: dict[str, Any] = {
    # архитектура
    "emb_dim":           32,
    "mlp_dims":          (256, 128),
    "dropout":           0.1,
    "layer_norm":        True,
    # оптимизация
    "lr":                1e-3,
    "emb_lr_mult":       1.0,
    "emb_weight_decay":  None,
    "weight_decay":      1e-5,
    "grad_clip":         1.0,
    # обучение
    "max_epochs":        15,
    "patience":          3,
    "groups_per_batch":  128,
    # scheduler
    "scheduler":         "none",
    "scheduler_kwargs":  {},
}


@register_model("deepfm")
class DeepFMRanker(Ranker):
    def __init__(self):
        self._model: DeepFM | None = None
        self._scaler: StandardScaler | None = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        p = {**DEFAULT_PARAMS, **params}

        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = StandardScaler().fit(train_df[num_cols].values)
        else:
            self._scaler = StandardScaler().fit(np.zeros((1, 0)))

        self._model = DeepFM(
            n_users=feature_spec.cardinalities.get("user_idx", 0),
            n_items=feature_spec.cardinalities.get("item_idx", 0),
            emb_dim=p["emb_dim"],
            n_num_features=len(num_cols),
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
        ).to(self._device)

        return train_neural_ranker(
            model=self._model,
            train_df=train_df, valid_df=valid_df,
            feature_spec=feature_spec, scaler=self._scaler,
            params=p, seed=seed, device=self._device,
        )

    def predict(self, df: pd.DataFrame, feature_spec: FeatureSpec) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return predict_dataframe(
            self._model, df, feature_spec, self._scaler, self._device
        )

    def n_params(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())