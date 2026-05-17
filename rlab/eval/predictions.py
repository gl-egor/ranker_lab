"""Сбор и сохранение построчных предсказаний на тесте."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rlab.models.base import FeatureSpec


def build_predictions_df(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """DataFrame (group_id, item_idx, label, pred) для анализа и diversity-метрик."""
    if "item_idx" not in test_df.columns:
        raise ValueError(
            "test_df must contain 'item_idx' (use feature_set that includes 'ids')"
        )
    return pd.DataFrame({
        "group_id": test_df[feature_spec.group_col].values,
        "item_idx": test_df["item_idx"].values,
        "label": test_df[feature_spec.target_col].values,
        "pred": np.asarray(scores, dtype=np.float32),
    })


def save_predictions_df(df_preds: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(path, index=False)
    return path
