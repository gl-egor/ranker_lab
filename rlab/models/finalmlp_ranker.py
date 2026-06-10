"""
FinalMLP — two-stream MLP ranker (Mao et al., 2023) под общий training loop.

Отличия от базовой версии на ветке experiment:
──────────────────────────────────────────────
1. MultihashEmbedding вместо nn.Embedding — tail-айтемы делят таблицу
   через хеш-коллизии.
2. PiecewiseLinearScaler (PLE) вместо StandardScaler для числовых признаков.
3. loss_temperature — делитель scores перед softmax (tau < 1 → острее).
4. Опциональные режимы перевзвешивания групп в loss (взаимоисключающие):
   - tail_aware_alpha > 0: weight_g = 1 + alpha * (1 - pop_norm)
   - ips_beta > 0:         weight_g ∝ 1 / (count_pos + 1)^beta (IPS)

Запуск:

    model=ModelConfig(
        kind="finalmlp",
        params={"tail_aware_alpha": 0.3, "loss_temperature": 0.7},
    )

    # или IPS:
    model=ModelConfig(kind="finalmlp", params={"ips_beta": 0.3})
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rlab.models._torch_utils import predict_dataframe, train_neural_ranker
from rlab.models.base import FeatureSpec, Ranker, register_model
from rlab.models.dcnv2_ranker import build_make_row_fn
from rlab.models.dcnv2_reg import (
    compute_tail_aware_group_weights,
    group_softmax_loss_tail_aware,
)
from rlab.models.encodings import MultihashEmbedding, PiecewiseLinearScaler


# ═════════════════════════════════════════════════════════════════════════════
# Архитектура FinalMLP
# ═════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Plain feed-forward block used as one FinalMLP stream."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: tuple[int, ...],
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeatureSelectionGate(nn.Module):
    """Bottleneck feature gating (Mao et al., 2023 — FinalMLP)."""

    def __init__(self, input_dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(input_dim // reduction, 16)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.sigmoid(self.gate(x)) * 2.0
        return x * weight


class MultiHeadBilinearFusion(nn.Module):
    """Multi-head bilinear fusion from FinalMLP."""

    def __init__(self, dim1: int, dim2: int, num_heads: int = 4):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_heads, dim1, dim2))
        self.c = nn.Parameter(torch.empty(num_heads, 1))
        self.p1 = nn.Parameter(torch.empty(dim1, 1))
        self.p2 = nn.Parameter(torch.empty(dim2, 1))
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.W, mean=0.0, std=np.sqrt(2.0 / (dim1 + dim2)))
        nn.init.normal_(self.c, mean=0.0, std=0.01)
        nn.init.normal_(self.p1, mean=0.0, std=0.01)
        nn.init.normal_(self.p2, mean=0.0, std=0.01)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        interactions = torch.einsum("bi,kij,bj->bk", x1, self.W, x2)
        fusion_out = torch.matmul(interactions, self.c)
        skip1 = torch.matmul(x1, self.p1)
        skip2 = torch.matmul(x2, self.p2)
        return (fusion_out + skip1 + skip2 + self.bias).squeeze(-1)


class FinalMLP(nn.Module):
    """
    Two-stream MLP с MultihashEmbedding + PLE-кодированными числовыми фичами.

    Каждый поток получает свою feature-gated версию входа x0, затем
    multi-head bilinear fusion объединяет выходы потоков в скор.
    """

    def __init__(
        self,
        emb_dim: int,
        ple_output_dim: int,
        mlp1_hidden: tuple[int, ...] = (256, 128, 64),
        mlp2_hidden: tuple[int, ...] = (512, 256, 64),
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_norm: bool = True,
        gate_reduction: int = 4,
        multihash_cardinality: int = 65536,
        multihash_num_hashes: int = 3,
    ):
        super().__init__()

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

        input_dim = 2 * emb_dim + ple_output_dim
        self.ln = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

        self.gate1 = FeatureSelectionGate(input_dim, reduction=gate_reduction)
        self.gate2 = FeatureSelectionGate(input_dim, reduction=gate_reduction)

        self.stream1 = MLP(input_dim, mlp1_hidden, dropout)
        self.stream2 = MLP(input_dim, mlp2_hidden, dropout)
        self.fusion = MultiHeadBilinearFusion(
            dim1=self.stream1.output_dim,
            dim2=self.stream2.output_dim,
            num_heads=num_heads,
        )

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
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        x = torch.cat([u, i, num_feat], dim=-1)
        x = self.drop(self.ln(x))

        out1 = self.stream1(self.gate1(x))
        out2 = self.stream2(self.gate2(x))
        return self.fusion(out1, out2)


# ═════════════════════════════════════════════════════════════════════════════
# IPS group weights (предвычисление, как tail_aware в dcnv2_reg)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ips_group_weights(
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    beta: float,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    IPS-веса групп: w_g = 1 / (count_pos + 1)^beta, нормировка к mean=1, clip.

    count_pos — число позитивов айтема в train (без утечки).
    При beta=0 все веса = 1.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0 for IPS weights")

    label_col = feature_spec.target_col
    group_col = feature_spec.group_col

    positives = (
        train_df.loc[train_df[label_col] == 1, [group_col, "item_idx"]]
        .sort_values(group_col)
        .drop_duplicates(group_col, keep="first")
    )

    item_pop = (
        train_df.loc[train_df[label_col] == 1]
        .groupby("item_idx")
        .size()
    )

    pos_pop = positives["item_idx"].map(item_pop).fillna(1).astype(np.float64)
    raw_weights = 1.0 / (pos_pop.to_numpy() + 1.0) ** beta
    raw_weights = raw_weights.astype(np.float64)
    mean_w = float(raw_weights.mean()) if len(raw_weights) else 1.0
    if mean_w > 0:
        raw_weights = raw_weights / mean_w
    raw_weights = np.clip(raw_weights, None, max_weight).astype(np.float32)

    n_groups = int(positives[group_col].max()) + 1
    out = np.ones(n_groups, dtype=np.float32)
    out[positives[group_col].to_numpy()] = raw_weights
    return torch.from_numpy(out)


# ═════════════════════════════════════════════════════════════════════════════
# Обёртка Ranker
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS: dict[str, Any] = {
    # архитектура
    "emb_dim":           32,
    "mlp1_hidden":       (256, 128, 64),
    "mlp2_hidden":       (512, 256, 64),
    "num_heads":         4,
    "dropout":           0.1,
    "layer_norm":        True,
    "gate_reduction":    4,
    # PLE
    "ple_n_bins":        32,
    # MultihashEmbedding
    "multihash_cardinality": 65536,
    "multihash_num_hashes":  3,
    # loss
    "loss_temperature":  1.0,
    "tail_aware_alpha":  0.0,   # 0 = выключено
    "ips_beta":          0.0,   # 0 = выключено; при >0 имеет приоритет над tail_aware
    "ips_max_weight":    10.0,
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


@register_model("finalmlp")
class FinalMLPRanker(Ranker):
    """
    FinalMLP с MultihashEmbedding + PLE и опциональным tail-aware / IPS loss.
    """

    def __init__(self):
        self._model: FinalMLP | None = None
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

        alpha = float(p.get("tail_aware_alpha", 0))
        ips_beta = float(p.get("ips_beta", 0))
        loss_temperature = float(
            p.get("loss_temperature", p.get("temperature", 1.0))
        )

        if alpha > 0 and ips_beta > 0:
            raise ValueError(
                "Укажите только один режим: tail_aware_alpha > 0 ИЛИ ips_beta > 0."
            )

        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = PiecewiseLinearScaler(n_bins=p["ple_n_bins"])
            self._scaler.fit(train_df[num_cols].values.astype(np.float32))
            ple_output_dim = self._scaler.output_dim
        else:
            self._scaler = PiecewiseLinearScaler(n_bins=1)
            self._scaler.fit(np.zeros((1, 0), dtype=np.float32))
            ple_output_dim = 0

        self._model = FinalMLP(
            emb_dim=p["emb_dim"],
            ple_output_dim=ple_output_dim,
            mlp1_hidden=tuple(p["mlp1_hidden"]),
            mlp2_hidden=tuple(p["mlp2_hidden"]),
            num_heads=p["num_heads"],
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
            gate_reduction=p.get("gate_reduction", 4),
            multihash_cardinality=p["multihash_cardinality"],
            multihash_num_hashes=p["multihash_num_hashes"],
        ).to(self._device)

        make_row_fn = None
        if p.get("hard_mining", False) or p.get("resample_each_epoch", False):
            make_row_fn = build_make_row_fn(train_df, feature_spec)

        group_weights = None
        loss_fn = None
        loss_mode = "default"

        if ips_beta > 0:
            loss_mode = "ips"
            group_weights = compute_ips_group_weights(
                train_df, feature_spec, ips_beta,
                max_weight=float(p.get("ips_max_weight", 10.0)),
            ).to(self._device)
            loss_fn = partial(
                group_softmax_loss_tail_aware,
                temperature=loss_temperature,
            )
        elif alpha > 0:
            loss_mode = "tail_aware"
            group_weights = compute_tail_aware_group_weights(
                train_df, feature_spec, alpha,
            ).to(self._device)
            loss_fn = partial(
                group_softmax_loss_tail_aware,
                temperature=loss_temperature,
            )

        meta = train_neural_ranker(
            model=self._model,
            train_df=train_df,
            valid_df=valid_df,
            feature_spec=feature_spec,
            scaler=self._scaler,
            params=p,
            seed=seed,
            device=self._device,
            make_row_fn=make_row_fn,
            loss_fn=loss_fn,
            group_weights=group_weights,
        )
        meta["loss_temperature"] = loss_temperature
        meta["loss_mode"] = loss_mode
        meta["tail_aware_alpha"] = alpha
        meta["ips_beta"] = ips_beta
        if group_weights is not None:
            meta["group_weight_min"] = float(group_weights.min().cpu())
            meta["group_weight_max"] = float(group_weights.max().cpu())
        return meta

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
