"""
DCN-v2 под интерфейс Ranker.

Архитектура — Cross Network + Deep Network (parallel).
Тренировка — общий train_neural_ranker из _torch_utils.

Все гиперпараметры выставляются через ModelConfig.params:
    Архитектура:
        emb_dim          : размерность эмбеддингов (default 32)
        n_cross          : число слоёв Cross Network (default 2)
        mlp_dims         : tuple размерностей скрытых слоёв (default (256, 128))
        dropout          : dropout в MLP и после конкатенации (default 0.1)
        layer_norm       : использовать ли LayerNorm на входе (default True)
    Оптимизация:
        lr               : базовый lr (default 1e-3)
        emb_lr_mult      : множитель lr для эмбеддингов (default 1.0)
        emb_weight_decay : wd для эмбеддингов; None = как у dense (default None)
        weight_decay     : wd для dense-параметров (default 1e-5)
        grad_clip        : max_norm для clip_grad_norm (default 1.0)
    Обучение:
        max_epochs       : default 15
        patience         : early stopping (default 3)
        groups_per_batch : сколько query-групп в батч (default 128)
    Scheduler:
        scheduler        : 'none' | 'cosine' | 'plateau' | 'warmup_cosine'
                           (default 'none')
        scheduler_kwargs : dict для шедулера (eta_min, warmup_epochs, ...)
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
from functools import partial
from rlab.data.features import make_feature_row, TrainAggregates, compute_train_aggregates
from collections import Counter

def make_feature_row_wrapper(
    user_idx: int,
    item_idx: int,
    label: int,
    group_id: int,
    *,
    user_histories: dict[int, list[int]],
    user_counters: dict[int, dict[int, int]],
    agg: TrainAggregates,
):
    history_items = user_histories.get(user_idx, [])
    history_counter = user_counters.get(user_idx, {})

    return make_feature_row(
        user_idx=user_idx,
        item_idx=item_idx,
        history_items=history_items,
        history_counter=history_counter,
        agg=agg,
        label=label,
        group_id=group_id,
    )

def _build_make_row_fn(
    self,
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> callable:
    """
    Callback (user_idx, item_idx, label, group_id) -> dict для hard mining.

    Использует train_aggregates из feature_spec (посчитаны в loader.py
    по raw interactions с rating — здесь их реконструировать невозможно,
    т.к. rank_table содержит только label).

    Истории юзеров восстанавливаем из позитивов rank_table — для этого
    rating не нужен.
    """
    if feature_spec.train_aggregates is None:
        raise RuntimeError(
            "hard_mining=True требует feature_spec.train_aggregates. "
            "Убедитесь, что loader.py заполняет это поле при построении "
            "FeatureSpec (см. compute_train_aggregates в features.py)."
        )

    pos_df = train_df[train_df[feature_spec.target_col] == 1]

    # История юзера = items, которые он реально взаимодействовал в train.
    # groupby.agg(list) — каноничная идиома "собери значения в списки по группе".
    user_histories: dict[int, list[int]] = (
        pos_df.groupby("user_idx")["item_idx"].agg(list).to_dict()
    )

    # Counter считает частоты за один проход на C-уровне — быстрее,
    # чем ручное d[k] = d.get(k, 0) + 1 в цикле Python.
    user_counters: dict[int, dict[int, int]] = {
        int(u): dict(Counter(items)) for u, items in user_histories.items()
    }

    # Приводим ключи к int для консистентности (groupby может вернуть numpy.int64,
    # а make_feature_row внутри, возможно, делает history_counter.get(item_idx))
    user_histories = {int(k): [int(x) for x in v] for k, v in user_histories.items()}

    return partial(
        make_feature_row_wrapper,
        user_histories=user_histories,
        user_counters=user_counters,
        agg=feature_spec.train_aggregates,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Архитектура
# ═════════════════════════════════════════════════════════════════════════════
class CrossLayer(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l. Один слой DCN-V2."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCNv2(nn.Module):
    """
    Parallel DCN-V2: Cross и Deep сети идут параллельно от общего x0,
    их выходы конкатенируются и идут в скалярный скор.
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
        layer_norm: bool = True,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)

        x0_dim = 2 * emb_dim + n_num_features
        self.ln = nn.LayerNorm(x0_dim) if layer_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

        self.cross_layers = nn.ModuleList(
            [CrossLayer(x0_dim) for _ in range(n_cross)]
        )

        mlp: list[nn.Module] = []
        prev = x0_dim
        for h in mlp_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()

        head_in = x0_dim + (prev if mlp_dims else x0_dim)
        self.head = nn.Linear(head_in, 1)

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
# Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS: dict[str, Any] = {
    # архитектура
    "emb_dim":           32,
    "n_cross":           2,
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


@register_model("dcnv2")
class DCNv2Ranker(Ranker):
    def __init__(self):
        self._model: DCNv2 | None = None
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

        # Scaler
        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = StandardScaler().fit(train_df[num_cols].values)
        else:
            self._scaler = StandardScaler().fit(np.zeros((1, 0)))

        # Модель
        self._model = DCNv2(
            n_users=feature_spec.cardinalities.get("user_idx", 0),
            n_items=feature_spec.cardinalities.get("item_idx", 0),
            emb_dim=p["emb_dim"],
            n_num_features=len(num_cols),
            n_cross=p["n_cross"],
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
        ).to(self._device)

        make_row_fn = None
        if p.get("hard_mining", False):
            make_row_fn = _build_make_row_fn(train_df, feature_spec)

        # Тренировка через общий loop
        return train_neural_ranker(
            model=self._model,
            train_df=train_df,
            valid_df=valid_df,
            feature_spec=feature_spec,
            scaler=self._scaler,
            params=p,
            seed=seed,
            device=self._device,
            make_row_fn=make_row_fn,   # None при выключенном hard mining
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