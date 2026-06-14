"""
Генерация OOF-скоров базовых моделей для stacking.

K-fold OOF на train предотвращает переобучение мета-ранкера на train-скорах.
На valid/test — полное обучение на train и инференс.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlab.configs import ExperimentConfig
from rlab.models.base import FeatureSpec, build_model
from rlab.runner import set_global_seed


def _score_col(model_kind: str) -> str:
    return f"score_{model_kind}"


def _split_groups_kfold(
    group_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[np.ndarray]:
    """Разбивает уникальные group_id на K непересекающихся фолдов."""
    unique = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return np.array_split(unique, n_folds)


def _subset_by_groups(
    df: pd.DataFrame,
    group_col: str,
    groups: np.ndarray,
) -> pd.DataFrame:
    return df[df[group_col].isin(groups)].reset_index(drop=True)


def _train_and_predict(
    model_kind: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    params: dict[str, Any],
    seed: int,
) -> np.ndarray:
    set_global_seed(seed)
    model = build_model(model_kind)
    model.fit(train_df, valid_df, feature_spec, params, seed)
    return model.predict(predict_df, feature_spec)


def generate_oof_scores(
    *,
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    model_kind: str,
    params: dict[str, Any],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """
    K-fold OOF-скоры на train_df. Длина = len(train_df), порядок = порядок строк.
    """
    group_col = feature_spec.group_col
    group_ids = train_df[group_col].values
    oof = np.full(len(train_df), np.nan, dtype=np.float32)

    folds = _split_groups_kfold(group_ids, n_folds, seed)
    for fold_idx, val_groups in enumerate(folds):
        if len(val_groups) == 0:
            continue
        train_mask = ~train_df[group_col].isin(val_groups)
        fold_train = train_df[train_mask].reset_index(drop=True)
        fold_val = train_df[train_df[group_col].isin(val_groups)].reset_index(drop=True)

        scores = _train_and_predict(
            model_kind, fold_train, fold_val, fold_val,
            feature_spec, params, seed + fold_idx,
        )
        val_mask = train_df[group_col].isin(val_groups).values
        oof[val_mask] = scores

    if np.isnan(oof).any():
        raise RuntimeError(f"OOF scores incomplete for {model_kind}")
    return oof


def generate_full_train_scores(
    *,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    model_kind: str,
    params: dict[str, Any],
    seed: int,
) -> dict[str, np.ndarray]:
    """Полное обучение на train, инференс на valid и test."""
    set_global_seed(seed)
    model = build_model(model_kind)
    model.fit(train_df, valid_df, feature_spec, params, seed)
    return {
        "valid": model.predict(valid_df, feature_spec),
        "test": model.predict(test_df, feature_spec),
    }


def stacking_cache_path(cfg: ExperimentConfig, model_kind: str) -> Path:
    cache_dir = Path(cfg.output_dir) / "stacking" / cfg.data.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{cfg.hash()}__{model_kind}__oof{cfg.stacking.n_oof_folds}.parquet"
    return cache_dir / key


def load_or_generate_base_scores(
    *,
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    model_kind: str,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """
    Возвращает dict с ключами 'train', 'valid', 'test' — numpy-массивы скоров.
    Кеширует в results/stacking/{dataset}/.
    """
    cache_path = stacking_cache_path(cfg, model_kind)
    if cache_path.exists() and not force_recompute:
        cached = pd.read_parquet(cache_path)
        return {
            "train": cached["train_score"].values.astype(np.float32),
            "valid": cached["valid_score"].values.astype(np.float32),
            "test": cached["test_score"].values.astype(np.float32),
        }

    params = cfg.stacking.base_model_params.get(model_kind, cfg.model.params)
    n_folds = cfg.stacking.n_oof_folds

    print(f"  [base_scores] OOF {model_kind} ({n_folds} folds)...")
    train_scores = generate_oof_scores(
        train_df=train_df,
        feature_spec=feature_spec,
        model_kind=model_kind,
        params=params,
        n_folds=n_folds,
        seed=cfg.seed,
    )

    print(f"  [base_scores] full fit {model_kind} on valid/test...")
    split_scores = generate_full_train_scores(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_spec=feature_spec,
        model_kind=model_kind,
        params=params,
        seed=cfg.seed,
    )
    valid_scores = split_scores["valid"]
    test_scores = split_scores["test"]

    cache_df = pd.DataFrame({
        "train_score": train_scores,
        "valid_score": valid_scores,
        "test_score": test_scores,
    })
    cache_df.to_parquet(cache_path, index=False)
    return {"train": train_scores, "valid": valid_scores, "test": test_scores}


def build_score_matrix(
    score_dicts: dict[str, dict[str, np.ndarray]],
    split: str,
    model_kinds: list[str],
) -> np.ndarray:
    """(n_rows, n_experts) матрица скоров для split='train'|'valid'|'test'."""
    cols = [score_dicts[k][split] for k in model_kinds]
    return np.stack(cols, axis=1).astype(np.float32)
