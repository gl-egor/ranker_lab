"""
DCN-v2 Enhanced — вариант DCN-v2 с продвинутыми кодированиями из семинара.

Отличия от базового dcnv2_ranker.py:
──────────────────────────────────────
1. PiecewiseLinearScaler вместо StandardScaler для числовых признаков.
   Каждый из 8 числовых признаков превращается в вектор из ~32 чисел
   (вместо одного числа). Итого input_dim растёт с 2*emb_dim + 8
   до 2*emb_dim + ~256. Это даёт MLP и Cross Network намного больше
   информации о форме распределения каждого признака.

2. MultihashEmbedding вместо nn.Embedding для user_idx/item_idx.
   Все ID хешируются в общую таблицу — tail-айтемы получают более
   осмысленные представления за счёт коллизий с популярными.

Зачем отдельная модель, а не параметр в dcnv2_ranker?
──────────────────────────────────────────────────────
Чтобы в sweep'е можно было честно сравнить:
    grid={"model.kind": ["dcnv2", "dcnv2_enhanced"]}
и увидеть эффект кодирований при прочих равных (та же архитектура
Cross+Deep, тот же training loop, те же гиперпараметры).

Дополнительные гиперпараметры (поверх стандартных DCN-v2):
    ple_n_bins              : число бинов PLE (default 32)
    multihash_cardinality   : размер общей таблицы хешей (default 65536)
    multihash_num_hashes    : число хеш-функций (default 3)
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
# Архитектура (та же Cross+Deep структура, но с новыми кодированиями)
# ═════════════════════════════════════════════════════════════════════════════

class CrossLayer(nn.Module):
    """x_{l+1} = x0 * (W x_l + b) + x_l — идентична семинарной."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(xl) + xl


class DCNv2Enhanced(nn.Module):
    """
    Parallel DCN-V2 с MultihashEmbedding + PLE.

    Архитектура:
        [MultihashEmb(user) | MultihashEmb(item) | PLE(num_features)]
                               ↓ x0
                          LayerNorm → Dropout
                          ┌────┴────┐
                   Cross Network   Deep Network (MLP)
                          └────┬────┘
                             concat
                               ↓
                         Linear → score

    Размерность x0 = 2 * emb_dim + ple_output_dim.
    При emb_dim=32 и 8 фичей × 32 бина: x0 = 64 + 256 = 320.
    Для сравнения, в базовом DCN-v2: x0 = 64 + 8 = 72.
    Больший input_dim → Cross Network моделирует больше взаимодействий.
    """

    def __init__(
        self,
        emb_dim: int,
        ple_output_dim: int,
        n_cross: int = 2,
        mlp_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        layer_norm: bool = True,
        # MultihashEmbedding параметры
        multihash_cardinality: int = 65536,
        multihash_num_hashes: int = 3,
    ):
        super().__init__()

        # ── Эмбеддинги через хеширование ─────────────────────────────
        # seed_offset=0 для user, seed_offset=10000 для item —
        # чтобы одинаковые ID (user_42 vs item_42) не попадали
        # в одни и те же позиции таблицы.
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

        # ── Входная размерность ──────────────────────────────────────
        # PLE уже расширила числовые фичи до ple_output_dim
        x0_dim = 2 * emb_dim + ple_output_dim

        self.ln = nn.LayerNorm(x0_dim) if layer_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

        # ── Cross Network ────────────────────────────────────────────
        self.cross_layers = nn.ModuleList(
            [CrossLayer(x0_dim) for _ in range(n_cross)]
        )

        # ── Deep Network (параллельная ветка) ────────────────────────
        mlp: list[nn.Module] = []
        prev = x0_dim
        for h in mlp_dims:
            mlp += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.mlp = nn.Sequential(*mlp) if mlp else nn.Identity()

        # ── Выходной слой ────────────────────────────────────────────
        head_in = x0_dim + (prev if mlp_dims else x0_dim)
        self.head = nn.Linear(head_in, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init для Linear, нормальный init для эмбеддингов."""
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
        """
        Сигнатура forward идентична базовому DCNv2 —
        это позволяет использовать тот же train_neural_ranker
        и predict_dataframe без изменений.
        """
        u = self.user_emb(user_idx)       # (B, emb_dim)
        i = self.item_emb(item_idx)       # (B, emb_dim)
        # num_feat уже закодирован PLE: (B, ple_output_dim)

        x0 = torch.cat([u, i, num_feat], dim=-1)
        x0 = self.drop(self.ln(x0))

        # Cross
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        # Deep
        deep_out = self.mlp(x0)

        # Concat → score
        combined = torch.cat([xl, deep_out], dim=-1)
        return self.head(combined).squeeze(-1)


# ═════════════════════════════════════════════════════════════════════════════
# Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS: dict[str, Any] = {
    # архитектура (те же, что у базового DCN-v2)
    "emb_dim":           32,
    "n_cross":           2,
    "mlp_dims":          (256, 128),
    "dropout":           0.1,
    "layer_norm":        True,
    # PLE
    "ple_n_bins":        32,
    # MultihashEmbedding
    "multihash_cardinality": 65536,
    "multihash_num_hashes":  3,
    # оптимизация (идентична базовому)
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


@register_model("dcnv2_enhanced")
class DCNv2EnhancedRanker(Ranker):
    """
    Обёртка, подключающая DCNv2Enhanced к системе экспериментов.

    Ключевое отличие от DCNv2Ranker: вместо StandardScaler
    используется PiecewiseLinearScaler. Это меняет размерность
    входа num_feat, но forward-сигнатура модели остаётся той же,
    поэтому train_neural_ranker и predict_dataframe работают
    без изменений — им всё равно, какой scaler подготовил данные.
    """

    def __init__(self):
        self._model: DCNv2Enhanced | None = None
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

        # ── PLE Scaler вместо StandardScaler ─────────────────────────
        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = PiecewiseLinearScaler(n_bins=p["ple_n_bins"])
            self._scaler.fit(train_df[num_cols].values.astype(np.float32))
            ple_output_dim = self._scaler.output_dim
        else:
            # Фоллбэк: нет числовых фичей
            self._scaler = PiecewiseLinearScaler(n_bins=1)
            self._scaler.fit(np.zeros((1, 0), dtype=np.float32))
            ple_output_dim = 0

        # ── Модель ───────────────────────────────────────────────────
        self._model = DCNv2Enhanced(
            emb_dim=p["emb_dim"],
            ple_output_dim=ple_output_dim,
            n_cross=p["n_cross"],
            mlp_dims=tuple(p["mlp_dims"]),
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
            multihash_cardinality=p["multihash_cardinality"],
            multihash_num_hashes=p["multihash_num_hashes"],
        ).to(self._device)

        # ── Тренировка через общий loop ──────────────────────────────
        # train_neural_ranker создаёт RankTableDataset, который зовёт
        # scaler.transform(). Наш PiecewiseLinearScaler имеет тот же
        # метод .transform() — всё работает прозрачно.
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