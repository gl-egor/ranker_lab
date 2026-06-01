"""FinalMLP ranker for the shared neural training loop."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from rlab.models._torch_utils import predict_dataframe, train_neural_ranker
from rlab.models.base import FeatureSpec, Ranker, register_model
from rlab.models.dcnv2_ranker import build_make_row_fn


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
    """Механизм гейтирования признаков из статьи FinalMLP."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Умножаем на 2, чтобы среднее значение весов после сигмоиды было около 1.
        # Это сохраняет масштаб исходных признаков на старте обучения.
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

        # He-style init для W — учитывает оба измерения входа
        nn.init.normal_(self.W, mean=0.0, std=np.sqrt(2.0 / (dim1 + dim2)))
        # Малый std для проекционных весов — стабильный старт fusion
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
    Two-stream MLP over concatenated user/item embeddings and numerical features,
    followed by multi-head bilinear fusion.

    Each stream receives its own feature-gated view of the shared input,
    allowing the model to learn stream-specific feature importance independently.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int,
        n_num_features: int,
        mlp1_hidden: tuple[int, ...] = (256, 128, 64),
        mlp2_hidden: tuple[int, ...] = (512, 256, 64),
        num_heads: int = 4,
        dropout: float = 0.1,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)

        input_dim = 2 * emb_dim + n_num_features
        self.ln = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

        # Независимые гейты: каждый поток учится выбирать свои признаки
        self.gate1 = FeatureSelectionGate(input_dim)
        self.gate2 = FeatureSelectionGate(input_dim)

        self.stream1 = MLP(input_dim, mlp1_hidden, dropout)
        self.stream2 = MLP(input_dim, mlp2_hidden, dropout)
        self.fusion = MultiHeadBilinearFusion(
            dim1=self.stream1.output_dim,
            dim2=self.stream2.output_dim,
            num_heads=num_heads,
        )

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

        # Каждый поток получает свою взвешенную версию входа
        out1 = self.stream1(self.gate1(x))
        out2 = self.stream2(self.gate2(x))
        return self.fusion(out1, out2)


DEFAULT_PARAMS: dict[str, Any] = {
    "emb_dim": 32,
    "mlp1_hidden": (256, 128, 64),
    "mlp2_hidden": (512, 256, 64),
    "num_heads": 4,
    "dropout": 0.1,
    "layer_norm": True,
    "lr": 1e-3,
    "emb_lr_mult": 1.0,
    "emb_weight_decay": None,
    "weight_decay": 1e-5,
    "grad_clip": 1.0,
    "max_epochs": 15,
    "patience": 3,
    "groups_per_batch": 128,
    "scheduler": "none",
    "scheduler_kwargs": {},
}


@register_model("finalmlp")
class FinalMLPRanker(Ranker):
    def __init__(self):
        self._model: FinalMLP | None = None
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

        self._model = FinalMLP(
            n_users=feature_spec.cardinalities.get("user_idx", 0),
            n_items=feature_spec.cardinalities.get("item_idx", 0),
            emb_dim=p["emb_dim"],
            n_num_features=len(num_cols),
            mlp1_hidden=tuple(p["mlp1_hidden"]),
            mlp2_hidden=tuple(p["mlp2_hidden"]),
            num_heads=p["num_heads"],
            dropout=p["dropout"],
            layer_norm=p["layer_norm"],
        ).to(self._device)

        make_row_fn = None
        if (
            p.get("hard_mining", False)
            or p.get("pop_weighted_sampler", False)
            or p.get("resample_each_epoch", False)
        ):
            make_row_fn = build_make_row_fn(train_df, feature_spec)

        return train_neural_ranker(
            model=self._model,
            train_df=train_df,
            valid_df=valid_df,
            feature_spec=feature_spec,
            scaler=self._scaler,
            params=p,
            seed=seed,
            device=self._device,
            make_row_fn=make_row_fn,
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