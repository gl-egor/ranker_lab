"""
Оркестрация contextual adaptive stacking experiment.

Pipeline:
  1. Load data
  2. Generate OOF + valid/test scores for each base model
  3. Calibrate base scores (Platt scaling)
  4. Train gating network on OOF train scores
  5. Evaluate on test with segment analysis + gating weight breakdown
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rlab.configs import ExperimentConfig
from rlab.models.base import RunRecord, append_run_record
from rlab.runner import set_global_seed

import rlab.models.catboost_ranker  # noqa: F401
import rlab.models.dcnv2_enhanced_ranker  # noqa: F401
import rlab.models.finalmlp_ranker  # noqa: F401

from rlab.stacking.base_scores import (
    build_score_matrix,
    load_or_generate_base_scores,
)
from rlab.stacking.calibration import MultiModelCalibrator
from rlab.stacking.gating_ranker import (
    extract_context_matrix,
    predict_gating_ranker,
    train_gating_ranker,
)


def _fixed_blend_scores(
    score_matrix: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Простое усреднение (baseline для сравнения)."""
    if weights is None:
        weights = [1.0 / score_matrix.shape[1]] * score_matrix.shape[1]
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    return (score_matrix * w).sum(axis=1)


def run_stacking_experiment(
    cfg: ExperimentConfig,
    *,
    skip_if_exists: bool = True,
) -> RunRecord:
    """
    Полный pipeline contextual adaptive stacking.

    model в RunRecord = 'adaptive_stacking'.
    extras содержит segment breakdown и gating weights.
    """
    set_global_seed(cfg.seed)
    stacking = cfg.stacking
    model_kinds = stacking.base_models

    run_id = cfg.run_id()
    runs_parquet = os.path.join(cfg.output_dir, "runs.parquet")
    run_dir = Path(cfg.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json(), encoding="utf-8")

    if skip_if_exists and os.path.exists(runs_parquet):
        existing = pd.read_parquet(runs_parquet)
        if run_id in set(existing["run_id"].values):
            print(f"[skip] {run_id} already in runs.parquet")
            row = existing[existing["run_id"] == run_id].iloc[0].to_dict()
            return _row_to_record(row)

    from rlab.data.loader import load_dataset

    train_df, valid_df, test_df, feature_spec = load_dataset(
        cfg.data, cache_dir=cfg.cache_dir, seed=cfg.seed,
    )

    # ── 1. Base model scores ──────────────────────────────────────────────
    print(f"[{run_id}] generating base scores for {model_kinds}...")
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    for kind in model_kinds:
        all_scores[kind] = load_or_generate_base_scores(
            cfg=cfg,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_spec=feature_spec,
            model_kind=kind,
            force_recompute=stacking.force_recompute,
        )

    train_matrix = build_score_matrix(all_scores, "train", model_kinds)
    valid_matrix = build_score_matrix(all_scores, "valid", model_kinds)
    test_matrix = build_score_matrix(all_scores, "test", model_kinds)

    train_labels = train_df[feature_spec.target_col].values
    valid_labels = valid_df[feature_spec.target_col].values
    test_labels = test_df[feature_spec.target_col].values
    train_groups = train_df[feature_spec.group_col].values
    valid_groups = valid_df[feature_spec.group_col].values
    test_groups = test_df[feature_spec.group_col].values

    # ── 2. Calibration ────────────────────────────────────────────────────
    if stacking.calibrate:
        print(f"[{run_id}] calibrating base scores (Platt scaling)...")
        calibrator = MultiModelCalibrator(model_kinds)
        calibrator.fit(train_matrix, train_labels)
        train_matrix = calibrator.transform(train_matrix)
        valid_matrix = calibrator.transform(valid_matrix)
        test_matrix = calibrator.transform(test_matrix)

    train_context = extract_context_matrix(train_df, stacking.context_features)
    valid_context = extract_context_matrix(valid_df, stacking.context_features)
    test_context = extract_context_matrix(test_df, stacking.context_features)

    # ── 3. Train gating network ───────────────────────────────────────────
    print(f"[{run_id}] training gating network...")
    gate_params = {
        "gate_hidden": stacking.gate_hidden,
        "gate_layers": stacking.gate_layers,
        "gate_lr": stacking.gate_lr,
        "gate_epochs": stacking.gate_epochs,
        "gate_patience": stacking.gate_patience,
        "gate_dropout": stacking.gate_dropout,
        "groups_per_batch": stacking.groups_per_batch,
    }
    gate_model, context_scaler, gate_meta = train_gating_ranker(
        train_base_scores=train_matrix,
        train_context=train_context,
        train_labels=train_labels,
        train_groups=train_groups,
        valid_base_scores=valid_matrix,
        valid_context=valid_context,
        valid_labels=valid_labels,
        valid_groups=valid_groups,
        n_experts=len(model_kinds),
        params=gate_params,
        seed=cfg.seed,
        device=cfg.device,
    )

    # ── 4. Test evaluation ────────────────────────────────────────────────
    test_df_eval = test_df
    test_matrix_eval = test_matrix
    test_context_eval = test_context
    test_labels_eval = test_labels
    test_groups_eval = test_groups

    n_total_groups = test_df_eval[feature_spec.group_col].nunique()
    n_eval_groups = n_total_groups

    if cfg.eval.eval_warm_only and "user_idx" in train_df.columns:
        warm_users = set(train_df["user_idx"].unique())
        warm_groups = set(
            test_df_eval.loc[test_df_eval["user_idx"].isin(warm_users), feature_spec.group_col]
        )
        keep = test_df_eval[feature_spec.group_col].isin(warm_groups).values
        test_df_eval = test_df_eval[keep].reset_index(drop=True)
        test_matrix_eval = test_matrix_eval[keep]
        test_context_eval = test_context_eval[keep]
        test_labels_eval = test_labels_eval[keep]
        test_groups_eval = test_groups_eval[keep]
        n_eval_groups = test_df_eval[feature_spec.group_col].nunique()
        print(f"[eval] warm filter: {n_eval_groups}/{n_total_groups} groups")

    test_scores, expert_weights = predict_gating_ranker(
        model=gate_model,
        context_scaler=context_scaler,
        base_scores=test_matrix_eval,
        context=test_context_eval,
        groups=test_groups_eval,
        labels=test_labels_eval,
        device=cfg.device,
    )

    from rlab.eval.metrics import ranking_metrics, per_group_ndcg
    from rlab.eval.bootstrap import bootstrap_ci
    from rlab.eval.stratified import (
        compute_all_segments,
        gating_weights_by_segment,
        make_popularity_bins,
        ndcg_by_popularity_bin,
    )

    k = cfg.eval.k
    overall = ranking_metrics(test_scores, test_labels_eval, test_groups_eval, k=k)

    ci_low = ci_high = None
    if cfg.eval.bootstrap_n > 0:
        per_group = per_group_ndcg(test_scores, test_labels_eval, test_groups_eval, k=k)
        ci_low, ci_high = bootstrap_ci(
            per_group,
            n_boot=cfg.eval.bootstrap_n,
            alpha=cfg.eval.bootstrap_alpha,
            seed=cfg.seed,
        )

    ndcg_by_bin = {}
    n_groups_by_bin = {}
    if cfg.eval.stratify_by_pop:
        bin_fn = make_popularity_bins(
            train_df, feature_spec, n_bins=cfg.eval.n_pop_bins,
        )
        ndcg_by_bin, n_groups_by_bin = ndcg_by_popularity_bin(
            test_scores, test_labels_eval, test_groups_eval, test_df_eval,
            feature_spec=feature_spec, bin_fn=bin_fn, k=k,
        )

    all_segments = compute_all_segments(
        test_scores, test_labels_eval, test_groups_eval,
        train_df, test_df_eval, feature_spec,
        k=k, n_pop_bins=cfg.eval.n_pop_bins,
    )

    pop_fn = make_popularity_bins(train_df, feature_spec, n_bins=cfg.eval.n_pop_bins)
    gate_weights_by_seg = gating_weights_by_segment(
        expert_weights, test_groups_eval, test_df_eval, feature_spec,
        segment_fn=lambda gid: pop_fn(
            int(test_df_eval[test_df_eval[feature_spec.group_col] == gid]["item_idx"].iloc[0])
        ),
        model_kinds=model_kinds,
    )

    blend_scores = _fixed_blend_scores(test_matrix_eval)
    blend_metrics = ranking_metrics(blend_scores, test_labels_eval, test_groups_eval, k=k)
    base_metrics = {}
    for i, kind in enumerate(model_kinds):
        base_metrics[kind] = ranking_metrics(
            test_matrix_eval[:, i], test_labels_eval, test_groups_eval, k=k,
        )

    extras = {
        **gate_meta,
        "base_models": model_kinds,
        "calibrated": stacking.calibrate,
        "gate_params": gate_params,
        "segments": all_segments,
        "gating_weights_by_pop_bin": gate_weights_by_seg,
        "baseline_equal_blend": blend_metrics,
        "baseline_base_models": base_metrics,
    }

    # Save gating artifacts
    torch_path = run_dir / "gating_model.pt"
    import torch
    torch.save({
        "state_dict": gate_model.state_dict(),
        "context_scaler_mean": context_scaler.mean_,
        "context_scaler_scale": context_scaler.scale_,
        "model_kinds": model_kinds,
        "context_features": stacking.context_features,
    }, torch_path)

    with open(run_dir / "stacking_extras.json", "w", encoding="utf-8") as f:
        json.dump(extras, f, indent=2, ensure_ascii=False, default=str)

    record = RunRecord(
        run_id=run_id,
        name=cfg.name,
        config_hash=cfg.hash(),
        dataset=cfg.data.dataset,
        model="adaptive_stacking",
        feature_set=cfg.data.feature_set,
        train_size=train_df[feature_spec.group_col].nunique(),
        seed=cfg.seed,
        ndcg_at_k=overall["NDCG"],
        hr_at_k=overall["HR"],
        mrr_at_k=overall["MRR"],
        k=k,
        ndcg_ci_low=ci_low,
        ndcg_ci_high=ci_high,
        ndcg_by_pop_bin=ndcg_by_bin,
        n_groups_by_pop_bin=n_groups_by_bin,
        train_time_sec=float(gate_meta.get("train_time_sec", 0.0)),
        n_params=int(gate_meta.get("n_gate_params", 0)),
        n_eval_groups=n_eval_groups,
        n_total_groups=n_total_groups,
        extras=extras,
    )

    append_run_record(record, runs_parquet)
    print(record.summary())
    print(f"  Baseline equal blend NDCG@{k}={blend_metrics['NDCG']:.4f}")
    for kind, m in base_metrics.items():
        print(f"  Base {kind} NDCG@{k}={m['NDCG']:.4f}")
    return record


def _row_to_record(row: dict) -> RunRecord:
    import json
    extras = json.loads(row.get("extras", "{}") or "{}")
    ndcg_bins = {
        k.replace("ndcg_", ""): v
        for k, v in row.items()
        if k.startswith("ndcg_") and k not in (
            "ndcg_at_k", "ndcg_ci_low", "ndcg_ci_high",
        ) and v is not None and not (isinstance(v, float) and np.isnan(v))
    }
    n_groups_bins = {
        k.replace("n_groups_", ""): int(v)
        for k, v in row.items()
        if k.startswith("n_groups_") and v is not None
        and not (isinstance(v, float) and np.isnan(v))
    }
    return RunRecord(
        run_id=row["run_id"], name=row["name"],
        config_hash=row["config_hash"], dataset=row["dataset"],
        model=row["model"], feature_set=row["feature_set"],
        train_size=int(row["train_size"]), seed=int(row["seed"]),
        ndcg_at_k=float(row["ndcg_at_k"]), hr_at_k=float(row["hr_at_k"]),
        mrr_at_k=float(row["mrr_at_k"]), k=int(row["k"]),
        ndcg_ci_low=row.get("ndcg_ci_low"),
        ndcg_ci_high=row.get("ndcg_ci_high"),
        ndcg_by_pop_bin=ndcg_bins, n_groups_by_pop_bin=n_groups_bins,
        train_time_sec=float(row.get("train_time_sec", 0.0)),
        n_params=int(row.get("n_params", 0)),
        extras=extras,
    )
