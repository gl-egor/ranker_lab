"""
DeepFM Enhanced — DeepFM с PLE + MultihashEmbedding.

Аналог dcnv2_enhanced_ranker, но с архитектурой DeepFM
(линейный + FM-взаимодействие + Deep MLP) вместо Cross Network.

Зачем нужен отдельно от dcnv2_enhanced:
─────────────────────────────────────────
Чтобы в sweep'е честно сравнить архитектуры при одинаковых
кодированиях. Если PLE+Multihash дать только DCN-v2, мы не
сможем отделить эффект кодирований от эффекта Cross Network.

    grid={"model.kind": ["dcnv2_enhanced", "deepfm_enhanced"]}
    → разница = чистый эффект Cross Network vs FM.

Архитектура:
    [MultihashEmb(user) | MultihashEmb(item) | PLE(num_features)]
                           ↓
        ┌──────── Linear + FM ──────────┐
        │ user_bias + item_bias         │
        │ + dot(u_emb, i_emb)           │
        │ + Linear(num_feat)            │
        └──────────┬────────────────────┘
                   +
        ┌──────── Deep MLP ─────────────┐
        │ concat(u, i, num) → MLP → 1   │
        └──────────┬────────────────────┘
                   ↓
                 score
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rlab.models._torch_utils import (
    predict_dataframe,
    train_neural_ranker,
)
from rlab.models.base import FeatureSpec, Ranker, register_model
from rlab.models.encodings import MultihashEmbedding, PiecewiseLinearScaler


# ═════════════════════════════════════════════════════════════════════════════
# Архитектура
# ═════════════════════════════════════════════════════════════════════════════

class DeepFMEnhanced(nn.Module):
    """
    DeepFM с MultihashEmbedding + PLE.

    Отличие от обычного DeepFM:
      - user_emb / item_emb = MultihashEmbedding (общая таблица + хеши)
      - num_feat приходит уже в PLE-формате (256 dims вместо 8)
      - FM-взаимодействие работает так же: dot(user_emb, item_emb)
      - num_linear принимает PLE-вектор, не сырые 8 чисел

    Сигнатура forward идентична базовому DeepFM — это позволяет
    использовать тот же train_neural_ranker без изменений.
    """

    def __init__(
        self,
        emb_dim: int,
        ple_output_dim: int,
        mlp_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        layer_norm: bool = True,
        multihash_cardinality: int = 65536,
        multihash_num_hashes: int = 3,
    ):
        super().__init__()

        # ── FM-эмбеддинги через хеширование ──────────────────────────
        self.user_emb = MultihashEmbedding(
            cardinality=multihash_cardinality,
            emb_dim=emb_dim,
            num_hashes=multihash_num_hashes,
            seed_offset=0,
        )
        self.item_emb = MultihashEmbedding(
            cardinality=multihash_cardinality,
            emb_dim=emb_dim,
            num_hashes=multihash_num_hashes,
            seed_offset=10000,
        )

        # ── Линейная часть (биасы) ───────────────────────────────────
        # В MultihashEmbedding нет padding_idx, поэтому используем
        # отдельные маленькие Embedding для биасов — они лёгкие.
        # Альтернатива: вычислять bias как среднее хеш-биасов,
        # но это усложняет код без ощутимого выигрыша.
        self.user_bias = nn.Embedding(multihash_cardinality, 1, padding_idx=0)
        self.item_bias = nn.Embedding(multihash_cardinality, 1, padding_idx=0)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Линейный вклад PLE-фичей
        self.num_linear = (
            nn.Linear(ple_output_dim, 1) if ple_output_dim > 0 else None
        )

        # ── Deep часть ───────────────────────────────────────────────
        deep_in = 2 * emb_dim + ple_output_dim
        self.ln = nn.LayerNorm(deep_in) if layer_norm else nn.Identity()

        mlp: list[nn.Module] = []
        prev = deep_in
        for h in mlp_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        mlp.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        num_feat: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)   # (B, emb_dim)
        i = self.item_emb(item_idx)   # (B, emb_dim)

        # FM: попарное взаимодействие u·i
        fm_interact = (u * i).sum(dim=-1)

        # Линейные члены (упрощённые биасы через хеш-позиции)
        # Берём первую хеш-позицию для bias — точное значение
        # не критично, это просто сдвиг скора.
        u_hash0 = (user_idx + self.user_emb.seeds[0]) % self.user_emb.cardinality
        i_hash0 = (item_idx + self.item_emb.seeds[0]) % self.item_emb.cardinality
        linear = (
            self.user_bias(u_hash0).squeeze(-1)
            + self.item_bias(i_hash0).squeeze(-1)
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
    "emb_dim":           32,
    "mlp_dims":          (256, 128),
    "dropout":           0.1,
    "layer_norm":        True,
    # PLE
    "ple_n_bins":        32,
    # MultihashEmbedding
    "multihash_cardinality": 65536,
    "multihash_num_hashes":  3,
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


@register_model("deepfm_enhanced")
class DeepFMEnhancedRanker(Ranker):

    def __init__(self):
        self._model: DeepFMEnhanced | None = None
        self._scaler: PiecewiseLinearScaler | None = None
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
            self._scaler = PiecewiseLinearScaler(n_bins=p["ple_n_bins"])
            self._scaler.fit(train_df[num_cols].values.astype(np.float32))
            ple_output_dim = self._scaler.output_dim
        else:
            self._scaler = PiecewiseLinearScaler(n_bins=1)
            self._scaler.fit(np.zeros((1, 0), dtype=np.float32))
            ple_output_dim = 0

        self._model = DeepFMEnhanced(
            emb_dim=p["emb_dim"],
            ple_output_dim=ple_output_dim,
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
            multihash_cardinality=p["multihash_cardinality"],
            multihash_num_hashes=p["multihash_num_hashes"],
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