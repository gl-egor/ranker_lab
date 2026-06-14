#!/usr/bin/env python3
"""
Эксперименты для диплома: Contextual Adaptive Stacking Ranker.

Запуск из корня ranker_lab:
    python scripts/run_stacking_experiments.py --quick
    python scripts/run_stacking_experiments.py --full
    python scripts/run_stacking_experiments.py --ablation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# rlab package root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlab.configs import DataConfig, EvalConfig, ExperimentConfig, ModelConfig, StackingConfig
from rlab.stacking.stacking_runner import run_stacking_experiment


# Гиперпараметры базовых моделей (из sweep'ов на ветке ttb_novelty)
BASE_MODEL_PARAMS = {
    "catboost": {},
    "dcnv2_enhanced": {
        "max_epochs": 20,
        "patience": 3,
        "groups_per_batch": 256,
        "lr": 1e-3,
        "hard_mining": False,
    },
    "finalmlp": {
        "max_epochs": 20,
        "patience": 3,
        "groups_per_batch": 256,
        "lr": 1e-3,
        "tail_aware_alpha": 0.3,
        "loss_temperature": 0.7,
    },
}


def _base_cfg(
    name: str,
    *,
    train_size: int = 30_000,
    dataset: str = "books5",
    seed: int = 42,
    n_oof_folds: int = 5,
    calibrate: bool = True,
    bootstrap_n: int = 1000,
    max_users: int | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        data=DataConfig(
            dataset=dataset,
            train_size=train_size,
            valid_size=5_000,
            test_size=5_000,
            feature_set="full",
            max_users=max_users,
        ),
        model=ModelConfig(kind="adaptive_stacking"),
        eval=EvalConfig(
            k=10,
            stratify_by_pop=True,
            n_pop_bins=4,
            bootstrap_n=bootstrap_n,
            eval_warm_only=False,
        ),
        stacking=StackingConfig(
            base_models=["catboost", "dcnv2_enhanced", "finalmlp"],
            base_model_params=BASE_MODEL_PARAMS,
            n_oof_folds=n_oof_folds,
            calibrate=calibrate,
            gate_hidden=32,
            gate_layers=2,
            gate_epochs=50,
            gate_patience=10,
        ),
        seed=seed,
        device="cuda",
    )


def run_quick():
    """Быстрая отладка: малый train, 3 OOF folds, без bootstrap."""
    cfg = _base_cfg(
        "stacking_quick",
        train_size=3_000,
        n_oof_folds=3,
        bootstrap_n=0,
        max_users=2_000,
    )
    cfg.stacking.gate_epochs = 20
    cfg.stacking.base_model_params = {
        "catboost": {"iterations": 100},
        "dcnv2_enhanced": {**BASE_MODEL_PARAMS["dcnv2_enhanced"], "max_epochs": 5},
        "finalmlp": {**BASE_MODEL_PARAMS["finalmlp"], "max_epochs": 5},
    }
    return run_stacking_experiment(cfg)


def run_main():
    """
    Основной эксперимент для диплома (books5, train_size=30k, 3 seeds).
    Сравнивает adaptive stacking vs equal blend vs каждый base model.
    """
    records = []
    for seed in [42, 43, 44]:
        cfg = _base_cfg(f"stacking_main_s{seed}", train_size=30_000, seed=seed)
        records.append(run_stacking_experiment(cfg))
    return records


def run_ablation():
    """
    Ablation study для новизны:
      1. adaptive stacking (full)
      2. без calibration
      3. equal blend (baseline — смотреть extras)
    """
    records = []

    cfg_full = _base_cfg("stacking_ablation_full", calibrate=True)
    records.append(run_stacking_experiment(cfg_full))

    cfg_nocal = _base_cfg("stacking_ablation_nocal", calibrate=False)
    records.append(run_stacking_experiment(cfg_nocal))

    return records


def run_segment_analysis():
    """
    Расширенный сегментный анализ: warm-only eval + все бины.
    Результаты в extras['segments'] и extras['gating_weights_by_pop_bin'].
    """
    cfg = _base_cfg("stacking_segments", train_size=30_000)
    cfg.eval.eval_warm_only = True
    cfg.eval.bootstrap_n = 1000
    return run_stacking_experiment(cfg)


def run_h1_scale():
    """H1-style: stacking при разных train_size."""
    records = []
    for train_size in [10_000, 30_000]:
        cfg = _base_cfg(
            f"stacking_h1_ts{train_size}",
            train_size=train_size,
            bootstrap_n=500,
        )
        records.append(run_stacking_experiment(cfg))
    return records


EXPERIMENTS = {
    "quick": ("Быстрая отладка (~30 мин CPU/GPU)", run_quick),
    "main": ("Основной эксперiment: 3 seeds, books5 30k", run_main),
    "ablation": ("Ablation: calibration on/off", run_ablation),
    "segments": ("Сегментный анализ + warm users", run_segment_analysis),
    "h1_scale": ("Stacking при train_size 10k/30k", run_h1_scale),
    "full": ("Все эксперименты подряд", None),
}


def main():
    parser = argparse.ArgumentParser(description="Stacking experiments for diploma")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        default="quick",
        help="Which experiment suite to run",
    )
    parser.add_argument("--quick", action="store_true", help="Alias for --experiment quick")
    parser.add_argument("--full", action="store_true", help="Run all experiments")
    parser.add_argument("--ablation", action="store_true", help="Alias for --experiment ablation")
    args = parser.parse_args()

    if args.quick:
        args.experiment = "quick"
    elif args.ablation:
        args.experiment = "ablation"
    elif args.full:
        args.experiment = "full"

    print("=" * 60)
    print("Contextual Adaptive Stacking — эксперименты для диплома")
    print("=" * 60)

    if args.experiment == "full":
        for name, (desc, fn) in EXPERIMENTS.items():
            if name == "full" or fn is None:
                continue
            print(f"\n>>> {name}: {desc}")
            fn()
    else:
        desc, fn = EXPERIMENTS[args.experiment]
        print(f"\n>>> {args.experiment}: {desc}")
        fn()

    print("\nРезультаты: results/runs.parquet")
    print("Детали stacking: results/runs/<run_id>/stacking_extras.json")
    print("\nДля диплома смотрите в extras:")
    print("  - segments: NDCG по tail/head, short/long history, cold/warm, sparse/dense")
    print("  - gating_weights_by_pop_bin: средние веса экспертов по популярности")
    print("  - baseline_equal_blend vs baseline_base_models")


if __name__ == "__main__":
    main()
