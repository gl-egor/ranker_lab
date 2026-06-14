"""
Генерация OOF-скоров базовых моделей для stacking.

K-fold OOF на train предотвращает переобучение мета-ранкера на train-скорах.
На valid/test — полное обучение на train и инференс.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlab.configs import ExperimentConfig
from rlab.models.base import FeatureSpec, build_model
from rlab.runner import set_global_seed

# v1: один parquet с train/valid/test колонками (баг — разная длина)
# v2: три parquet-файла
# v3: один scores.npz (текущий)
STACKING_CACHE_VERSION = 3
_SCORES_FILE = "scores.npz"
_META_FILE = "meta.json"


def check_stacking_cache_fix() -> bool:
    """
    True, если установлена версия с исправленным кешем.
    Вызовите в Colab перед экспериментом.
    """
    return STACKING_CACHE_VERSION >= 3


def _flatten_scores(scores: np.ndarray, expected_len: int, name: str) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(flat) != expected_len:
        raise ValueError(
            f"Score length mismatch for {name}: got {len(flat)}, expected {expected_len}"
        )
    return flat


def _split_groups_kfold(
    group_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[np.ndarray]:
    unique = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return list(np.array_split(unique, n_folds))


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
    scores = model.predict(predict_df, feature_spec)
    return _flatten_scores(scores, len(predict_df), model_kind)


def generate_oof_scores(
    *,
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    model_kind: str,
    params: dict[str, Any],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    group_col = feature_spec.group_col
    oof = np.full(len(train_df), np.nan, dtype=np.float32)

    folds = _split_groups_kfold(train_df[group_col].values, n_folds, seed)
    for fold_idx, val_groups in enumerate(folds):
        if len(val_groups) == 0:
            continue
        val_groups_set = set(val_groups.tolist())
        fold_train = train_df[~train_df[group_col].isin(val_groups_set)].reset_index(drop=True)
        fold_val = train_df[train_df[group_col].isin(val_groups_set)].reset_index(drop=True)

        scores = _train_and_predict(
            model_kind, fold_train, fold_val, fold_val,
            feature_spec, params, seed + fold_idx,
        )
        val_idx = np.flatnonzero(train_df[group_col].isin(val_groups_set).values)
        if len(val_idx) != len(scores):
            raise ValueError(
                f"OOF fold {fold_idx} for {model_kind}: "
                f"{len(scores)} scores vs {len(val_idx)} rows"
            )
        oof[val_idx] = scores

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
    set_global_seed(seed)
    model = build_model(model_kind)
    model.fit(train_df, valid_df, feature_spec, params, seed)
    return {
        "valid": _flatten_scores(model.predict(valid_df, feature_spec), len(valid_df), f"{model_kind}/valid"),
        "test": _flatten_scores(model.predict(test_df, feature_spec), len(test_df), f"{model_kind}/test"),
    }


def stacking_cache_dir(cfg: ExperimentConfig, model_kind: str) -> Path:
    cache_root = Path(cfg.output_dir) / "stacking" / cfg.data.dataset
    key = f"{cfg.hash()}__{model_kind}__oof{cfg.stacking.n_oof_folds}"
    path = cache_root / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_legacy_cache_files(cache_root: Path) -> None:
    """Удаляет старый битый формат: один .parquet на модель."""
    for legacy in cache_root.glob("*__oof*.parquet"):
        try:
            legacy.unlink()
            print(f"  [base_scores] removed legacy cache file: {legacy.name}")
        except OSError:
            pass


def _load_cached_scores(
    cache_dir: Path,
    *,
    expected_lengths: dict[str, int],
) -> dict[str, np.ndarray] | None:
    npz_path = cache_dir / _SCORES_FILE
    if not npz_path.exists():
        return None

    loaded = np.load(npz_path)
    scores = {
        split: loaded[split].astype(np.float32)
        for split in ("train", "valid", "test")
    }
    for split, expected in expected_lengths.items():
        if len(scores[split]) != expected:
            print(f"  [base_scores] cache stale for {split}: "
                  f"{len(scores[split])} vs {expected}, recomputing")
            return None
    return scores


def _save_cached_scores(cache_dir: Path, scores: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        cache_dir / _SCORES_FILE,
        train=scores["train"],
        valid=scores["valid"],
        test=scores["test"],
    )
    meta_path = cache_dir / _META_FILE
    meta_path.write_text(
        pd.Series({
            "version": STACKING_CACHE_VERSION,
            "train_len": len(scores["train"]),
            "valid_len": len(scores["valid"]),
            "test_len": len(scores["test"]),
        }).to_json(),
        encoding="utf-8",
    )


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
    if not check_stacking_cache_fix():
        raise RuntimeError(
            "Установлена старая версия rlab.stacking.base_scores. "
            "В Colab: git pull + Runtime → Restart session, "
            "или проверьте check_stacking_cache_fix()."
        )

    cache_root = Path(cfg.output_dir) / "stacking" / cfg.data.dataset
    cache_root.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_cache_files(cache_root)

    expected_lengths = {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }
    cache_dir = stacking_cache_dir(cfg, model_kind)

    if not force_recompute:
        cached = _load_cached_scores(cache_dir, expected_lengths=expected_lengths)
        if cached is not None:
            print(f"  [base_scores] cache HIT: {model_kind}")
            return cached

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

    scores = {
        "train": _flatten_scores(train_scores, expected_lengths["train"], f"{model_kind}/train"),
        "valid": split_scores["valid"],
        "test": split_scores["test"],
    }
    _save_cached_scores(cache_dir, scores)
    return scores


def build_score_matrix(
    score_dicts: dict[str, dict[str, np.ndarray]],
    split: str,
    model_kinds: list[str],
) -> np.ndarray:
    cols = [score_dicts[k][split] for k in model_kinds]
    return np.stack(cols, axis=1).astype(np.float32)
