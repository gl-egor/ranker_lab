"""Сбор и сохранение построчных предсказаний (train / test / valid)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rlab.models.base import FeatureSpec

if TYPE_CHECKING:
    from rlab.models.base import Ranker


def build_predictions_df(
    df: pd.DataFrame,
    scores: np.ndarray,
    feature_spec: FeatureSpec,
) -> pd.DataFrame:
    """
    Минимум: group_id, label, pred.
    user_idx / item_idx — если есть в df (метаданные из loader).
    """
    data: dict[str, np.ndarray] = {
        "group_id": df[feature_spec.group_col].values,
        "label": df[feature_spec.target_col].values,
        "pred": np.asarray(scores, dtype=np.float32),
    }
    if "user_idx" in df.columns:
        data["user_idx"] = df["user_idx"].values
    if "item_idx" in df.columns:
        data["item_idx"] = df["item_idx"].values
    return pd.DataFrame(data)


def save_predictions_df(df_preds: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_parquet(path, index=False)
    return path


def save_split_predictions(
    model: "Ranker",
    df: pd.DataFrame,
    feature_spec: FeatureSpec,
    run_dir: Path,
    model_kind: str,
    split: str,
    scores: np.ndarray | None = None,
) -> Path:
    """Parquet: runs/<run_id>/preds_<model>_<split>.parquet."""
    if scores is None:
        scores = model.predict(df, feature_spec)
    df_preds = build_predictions_df(df, scores, feature_spec)
    path = run_dir / f"preds_{model_kind}_{split}.parquet"
    save_predictions_df(df_preds, path)
    print(f"[eval] predictions saved → {path} ({len(df_preds)} rows)")
    return path
