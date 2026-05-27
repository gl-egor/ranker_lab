"""
DCN-v2 + popularity-aware training:
  1) popularity-weighted negative sampler
  2) popularity regularization in loss
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from rlab.models._torch_utils import predict_dataframe, train_neural_ranker
from rlab.models.base import FeatureSpec, Ranker, register_model
from rlab.models.dcnv2_ranker import DCNv2, build_make_row_fn, DEFAULT_PARAMS as BASE_DEFAULT_PARAMS


DEFAULT_PARAMS: dict[str, Any] = {
    **BASE_DEFAULT_PARAMS,
    # Popularity-weighted sampler
    "pop_weighted_sampler": True,
    "pop_resample_each_epoch": False,
    # Loss = (1-a)*LTR + a*RegTerm (+ weight_decay in optimizer)
    "pop_reg_alpha": 0.15,
}


@register_model("dcnv2_reg")
class DCNv2RegRanker(Ranker):
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

        num_cols = feature_spec.numerical_cols
        if num_cols:
            self._scaler = StandardScaler().fit(train_df[num_cols].values)
        else:
            self._scaler = StandardScaler().fit(np.zeros((1, 0)))

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
        return predict_dataframe(self._model, df, feature_spec, self._scaler, self._device)

    def n_params(self) -> int:
        if self._model is None:
            return 0
        return sum(p.numel() for p in self._model.parameters())

