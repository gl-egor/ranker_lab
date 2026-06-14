"""Unit tests for stacking pipeline (synthetic data, no raw files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rlab.models.base import FeatureSpec
from rlab.stacking.base_scores import (
    _load_cached_scores,
    _save_cached_scores,
    check_stacking_cache_fix,
)
from rlab.stacking.calibration import MultiModelCalibrator
from rlab.stacking.gating_ranker import (
    extract_context_matrix,
    predict_gating_ranker,
    train_gating_ranker,
)
from rlab.eval.stratified import compute_all_segments


def _make_synthetic_rank_table(n_groups: int = 50, n_neg: int = 5) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        u = g % 10 + 1
        pos_item = g % 20 + 1
        rows.append({
            "group_id": g, "label": 1, "user_idx": u, "item_idx": pos_item,
            "history_len": g % 15, "user_mean_rating": 3.5,
            "user_interaction_count": g % 10 + 1,
            "item_popularity": g % 100, "item_popularity_log": np.log1p(g % 100),
            "item_mean_rating": 4.0, "user_item_prev_count": 0,
            "user_item_seen_before": 0,
        })
        for j in range(n_neg):
            rows.append({
                "group_id": g, "label": 0, "user_idx": u, "item_idx": (pos_item + j + 1) % 50 + 1,
                "history_len": g % 15, "user_mean_rating": 3.5,
                "user_interaction_count": g % 10 + 1,
                "item_popularity": (g + j) % 100, "item_popularity_log": np.log1p((g + j) % 100),
                "item_mean_rating": 4.0, "user_item_prev_count": 0,
                "user_item_seen_before": 0,
            })
    return pd.DataFrame(rows)


CONTEXT_FEATURES = [
    "history_len", "item_popularity_log", "user_interaction_count",
    "item_mean_rating", "user_mean_rating",
]


def test_calibration_and_gating():
    df = _make_synthetic_rank_table(n_groups=40)
    n = len(df)
    rng = np.random.default_rng(0)
    base_scores = rng.standard_normal((n, 3)).astype(np.float32)
    base_scores[df["label"].values == 1] += 1.0

    cal = MultiModelCalibrator(["m1", "m2", "m3"])
    cal.fit(base_scores, df["label"].values)
    calibrated = cal.transform(base_scores)

    assert calibrated.min() >= 0.0 and calibrated.max() <= 1.0

    ctx = extract_context_matrix(df, CONTEXT_FEATURES)
    split = 30
    train_g = df["group_id"].unique()[:split]
    valid_g = df["group_id"].unique()[split:]
    train_mask = df["group_id"].isin(train_g).values
    valid_mask = df["group_id"].isin(valid_g).values

    model, scaler, meta = train_gating_ranker(
        train_base_scores=calibrated[train_mask],
        train_context=ctx[train_mask],
        train_labels=df["label"].values[train_mask],
        train_groups=df["group_id"].values[train_mask],
        valid_base_scores=calibrated[valid_mask],
        valid_context=ctx[valid_mask],
        valid_labels=df["label"].values[valid_mask],
        valid_groups=df["group_id"].values[valid_mask],
        n_experts=3,
        params={"gate_epochs": 5, "gate_patience": 3, "groups_per_batch": 8},
        seed=42,
        device="cpu",
    )
    assert meta["best_val_ndcg"] >= 0.0

    scores, weights = predict_gating_ranker(
        model=model, context_scaler=scaler,
        base_scores=calibrated[valid_mask],
        context=ctx[valid_mask],
        groups=df["group_id"].values[valid_mask],
        labels=df["label"].values[valid_mask],
        device="cpu",
    )
    assert len(scores) == valid_mask.sum()
    assert weights.shape == (valid_mask.sum(), 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)


def test_segment_analysis():
    df = _make_synthetic_rank_table(n_groups=40)
    n = len(df)
    rng = np.random.default_rng(1)
    scores = rng.standard_normal(n).astype(np.float32)
    scores[df["label"].values == 1] += 0.5

    spec = FeatureSpec(
        categorical_cols=["user_idx", "item_idx"],
        numerical_cols=["history_len", "item_popularity"],
    )
    segments = compute_all_segments(
        scores, df["label"].values, df["group_id"].values,
        df, df, spec, k=10,
    )
    assert "popularity" in segments
    assert "history_len" in segments


def test_base_scores_cache_different_lengths(tmp_path):
    """train/valid/test имеют разную длину — кеш не должен падать."""
    assert check_stacking_cache_fix()
    scores = {
        "train": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "valid": np.array([0.5, 0.6], dtype=np.float32),
        "test": np.array([0.1], dtype=np.float32),
    }
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _save_cached_scores(cache_dir, scores)
    loaded = _load_cached_scores(
        cache_dir,
        expected_lengths={"train": 4, "valid": 2, "test": 1},
    )
    assert loaded is not None
    for split in scores:
        np.testing.assert_array_equal(loaded[split], scores[split])
