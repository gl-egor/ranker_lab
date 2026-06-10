"""
DCN-v2 Reg — enhanced-кодирования + tail-aware listwise loss.

Отличия от dcnv2_ranker.py / dcnv2_enhanced_ranker.py:
──────────────────────────────────────────────────────────
1. MultihashEmbedding + PiecewiseLinearScaler (как в dcnv2_enhanced):
   tail-айтемы делят embedding-таблицу через хеш-коллизии;
   числовые фичи кодируются PLE (~32 бина на признак).

2. Tail-aware listwise loss:
       weight_g = 1 + alpha * (1 - pop_norm(positive_item))
   Head-группы получают вес ~1.0, tail — до 1 + alpha.
   При alpha=0 loss совпадает с обычным group_softmax_loss.

3. Temperature в softmax loss (loss_temperature):
       log_softmax(scores / tau). tau < 1 → более «острый» loss.
       Рекомендуемый sweep: [0.5, 0.7, 1.0].

Запуск на books5, train_size=60k, ids_only:

    from rlab.configs import DataConfig, EvalConfig, ExperimentConfig, ModelConfig
    from rlab.runner import run_experiment

    cfg = ExperimentConfig(
        name="dcnv2_reg_60k_ids",
        data=DataConfig(
            dataset="books5",
            train_size=60_000,
            feature_set="ids_only",
        ),
        model=ModelConfig(
            kind="dcnv2_reg",
            params={"tail_aware_alpha": 0.3},
        ),
        eval=EvalConfig(stratify_by_pop=True),
        seed=42,
    )
    print(run_experiment(cfg).summary())
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from rlab.models._torch_utils import (
    predict_dataframe,
    train_neural_ranker,
)
from rlab.models.base import FeatureSpec, Ranker, register_model
from rlab.models.dcnv2_enhanced_ranker import DCNv2Enhanced
from rlab.models.dcnv2_ranker import build_make_row_fn
from rlab.models.encodings import PiecewiseLinearScaler


def group_softmax_loss_tail_aware(
    scores: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    group_weights: torch.Tensor | None,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Listwise loss с мягким бонусом для tail-групп.

    group_weights: (max_group_id + 1,) — вес на группу.
    weight = 1 + alpha * (1 - normalized_popularity) позитивного айтема.
    temperature: делитель для scores перед softmax (tau < 1 → острее).
    При alpha=0 все веса = 1 → эквивалент group_softmax_loss с temperature.
    """
    t = max(float(temperature), 1e-8)
    unique = torch.unique(groups)
    losses = []
    for gid in unique:
        mask = groups == gid
        log_prob = F.log_softmax(scores[mask] / t, dim=0)
        loss_g = -(log_prob * labels[mask]).sum()
        if group_weights is not None:
            w = group_weights[gid.long()]
            loss_g = loss_g * w
        losses.append(loss_g)
    return torch.stack(losses).mean()


def compute_tail_aware_group_weights(
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    alpha: float,
) -> torch.Tensor:
    """
    Предвычисляет веса групп из популярности позитивного айтема.

    Популярность = число позитивов айтема в train (без утечки).
    Нормализация min-max по всем айтемам train → pop_norm ∈ [0, 1].
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for tail-aware weights")

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
    pmin = float(item_pop.min())
    pmax = float(item_pop.max())

    pos_pop = positives["item_idx"].map(item_pop).fillna(pmin).astype(np.float64)
    if pmax > pmin:
        pop_norm = (pos_pop - pmin) / (pmax - pmin)
    else:
        pop_norm = pd.Series(np.zeros(len(pos_pop), dtype=np.float64))

    pop_norm_arr = pop_norm.to_numpy(dtype=np.float64)
    weights = 1.0 + alpha * (1.0 - pop_norm_arr)

    n_groups = int(positives[group_col].max()) + 1
    out = np.ones(n_groups, dtype=np.float32)
    out[positives[group_col].to_numpy()] = weights.astype(np.float32)
    return torch.from_numpy(out)


DEFAULT_PARAMS: dict[str, Any] = {
    # архитектура
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
    # tail-aware loss
    "tail_aware_alpha":  0.3,
    "loss_temperature":  1.0,
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


@register_model("dcnv2_reg")
class DCNv2RegRanker(Ranker):
    """
    DCN-v2 с MultihashEmbedding + PLE и tail-aware listwise loss.

    Архитектура совпадает с dcnv2_enhanced; отличие — перевзвешивание
    групп в loss по популярности позитивного айтема.
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
        alpha = float(p.get("tail_aware_alpha", 0))
        loss_temperature = float(
            p.get("loss_temperature", p.get("temperature", 1.0))
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

        make_row_fn = None
        if p.get("hard_mining", False):
            make_row_fn = build_make_row_fn(train_df, feature_spec)

        group_weights = None
        loss_fn = None
        if alpha > 0:
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
        meta["tail_aware_alpha"] = alpha
        meta["loss_temperature"] = loss_temperature
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
