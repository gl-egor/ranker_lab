"""
Оркестрация одного эксперимента.

run_experiment(cfg) — единственная публичная функция.
Делает всё: данные → модель → fit → eval → сохранение RunRecord.

Намеренно процедурный стиль, без классов. Одна функция — один запуск,
читается сверху вниз как рецепт.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from rlab.configs import ExperimentConfig
from rlab.models.base import (
    RunRecord,
    append_run_record,
    build_model,
)

import rlab.models.catboost_ranker  # noqa: F401
import rlab.models.dcnv2_ranker     # noqa: F401
import rlab.models.deepfm_ranker    # noqa: F401
import rlab.models.dcnv2_enhanced_ranker # noqa: F401
import rlab.models.deepfm_enhanced_ranker # noqa: F401


def set_global_seed(seed: int) -> None:
    """Фиксируем все источники случайности, чтобы результат был воспроизводим."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN determinism — пока выключено: сильно замедляет, а для
        # sweep'а (3 сида × N конфигов) важнее скорость, чем полная
        # битовая воспроизводимость на GPU. Добавим флагом, если понадобится.
    except ImportError:
        pass


def run_experiment(cfg: ExperimentConfig, *, skip_if_exists: bool = True) -> RunRecord:
    """
    Прогнать один эксперимент по конфигу.

    Параметры:
        cfg             : ExperimentConfig, целиком описывающий запуск.
        skip_if_exists  : если в runs.parquet уже есть запись с таким
                          run_id (= hash конфига), пропустить и вернуть её.
                          Критично для sweep'а: повторный запуск не
                          переобучит одно и то же.

    Возвращает:
        RunRecord — уже записан в results/runs.parquet.
    """
    # ─── 0. Инициализация окружения ──────────────────────────────────────
    set_global_seed(cfg.seed)

    run_id = cfg.run_id()
    runs_parquet = os.path.join(cfg.output_dir, "runs.parquet")
    run_dir = Path(cfg.output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Кладём конфиг рядом с артефактами — чтобы по run_id можно было
    # воспроизвести запуск через месяц.
    (run_dir / "config.json").write_text(cfg.to_json(), encoding="utf-8")

    # ─── 1. Skip, если уже считали ───────────────────────────────────────
    if skip_if_exists and os.path.exists(runs_parquet):
        import pandas as pd
        existing = pd.read_parquet(runs_parquet)
        if run_id in set(existing["run_id"].values):
            print(f"[skip] {run_id} already in runs.parquet")
            row = existing[existing["run_id"] == run_id].iloc[0].to_dict()
            # лёгкая реконструкция RunRecord — достаточно для логов
            return _row_to_record(row)

    # ─── 2. Данные ───────────────────────────────────────────────────────
    # load_dataset вернёт три DataFrame + FeatureSpec. Внутри:
    #   - читает сырые данные (Amazon Books / Yelp / ...);
    #   - делает train/valid/test split по time;
    #   - строит rank_table (позитивы + негативы по cfg.data.neg_strategy);
    #   - применяет feature_set (для H3);
    #   - кеширует в cfg.cache_dir для быстрых повторов.
    #
    # Реализация loader.py — в следующем шаге (отдельный файл).
    from rlab.data.loader import load_dataset

    train_df, valid_df, test_df, feature_spec = load_dataset(
        cfg.data, cache_dir=cfg.cache_dir, seed=cfg.seed
    )

    # ─── 3. Модель ───────────────────────────────────────────────────────
    model = build_model(cfg.model.kind)
    print(f"[{run_id}] fitting {cfg.model.kind} on {len(train_df)} rows "
          f"({train_df[feature_spec.group_col].nunique()} groups)")

    fit_meta = model.fit(
        train_df=train_df,
        valid_df=valid_df,
        feature_spec=feature_spec,
        params=cfg.model.params,
        seed=cfg.seed,
    )

    # ─── 4. Инференс на тесте ────────────────────────────────────────────
    test_scores = model.predict(test_df, feature_spec)

    def _apply_warm_filter(train_df, test_df, test_scores, cfg):
        """
        Если eval_warm_only=True, оставляем в тесте только группы,
        где user_idx встречался хотя бы в одной train-группе.
    
        Возвращает отфильтрованные (test_df, test_scores, n_total, n_warm).
    
        Почему фильтруем по group_id, а не по строкам:
            Каждая группа = 1 позитив + 50 негативов для одного юзера.
            Если убрать группу частично, NDCG посчитается неправильно.
            Поэтому убираем целые группы.
    
        Почему НЕ фильтруем train/valid:
            train — по определению «тёплый» (все юзеры в нём есть).
            valid — используется для early stopping, фильтровать его
            опасно (модель может остановиться на другой эпохе). Если
            нужно — можно добавить аналогичный фильтр, но для диплома
            достаточно фильтрации теста.
        """
        import numpy as np
    
        n_total = test_df[cfg.eval.group_col if hasattr(cfg.eval, 'group_col')
                        else "group_id"].nunique()
    
        if not cfg.eval.eval_warm_only:
            return test_df, test_scores, n_total, n_total
    
        # Юзеры, встретившиеся в train
        # train_df содержит колонку user_idx (если feature_set включает ids)
        if "user_idx" not in train_df.columns:
            print("[warn] eval_warm_only=True, но user_idx нет в train_df. "
                "feature_set должна включать 'ids'.")
            return test_df, test_scores, n_total, n_total
    
        warm_users = set(train_df["user_idx"].unique())
    
        # Группы с тёплыми юзерами
        group_col = "group_id"
        warm_mask = test_df["user_idx"].isin(warm_users)
        warm_groups = set(test_df.loc[warm_mask, group_col].unique())
    
        # Фильтруем целые группы
        keep_mask = test_df[group_col].isin(warm_groups).values
        test_df_warm = test_df[keep_mask].reset_index(drop=True)
        test_scores_warm = test_scores[keep_mask]
    
        n_warm = test_df_warm[group_col].nunique()
        print(f"[eval] warm filter: {n_warm}/{n_total} groups "
            f"({100*n_warm/n_total:.1f}% coverage)")
    
        return test_df_warm, test_scores_warm, n_total, n_warm
    
    test_df, test_scores, n_total_groups, n_eval_groups = _apply_warm_filter(train_df, test_df, test_scores, cfg)
    test_labels = test_df[feature_spec.target_col].values
    test_groups = test_df[feature_spec.group_col].values

    # ─── 5. Метрики ──────────────────────────────────────────────────────
    # Считаем в три прохода: общие, CI через bootstrap, стратификация.
    # Реализация — rlab/eval/metrics.py и rlab/eval/stratified.py.
    from rlab.eval.metrics import ranking_metrics, per_group_ndcg
    from rlab.eval.bootstrap import bootstrap_ci
    from rlab.eval.stratified import (
        make_popularity_bins, ndcg_by_popularity_bin,
    )

    k = cfg.eval.k
    overall = ranking_metrics(test_scores, test_labels, test_groups, k=k)

    ci_low = ci_high = None
    if cfg.eval.bootstrap_n > 0:
        per_group = per_group_ndcg(test_scores, test_labels, test_groups, k=k)
        ci_low, ci_high = bootstrap_ci(
            per_group,
            n_boot=cfg.eval.bootstrap_n,
            alpha=cfg.eval.bootstrap_alpha,
            seed=cfg.seed,
        )

    ndcg_by_bin = {}
    n_groups_by_bin = {}
    if cfg.eval.stratify_by_pop:
        # Бинирование ПО TRAIN — чтобы не было утечки: популярность
        # в тесте может сильно отличаться, особенно на tail'е.
        bin_fn = make_popularity_bins(
            train_df, feature_spec, n_bins=cfg.eval.n_pop_bins
        )
        ndcg_by_bin, n_groups_by_bin = ndcg_by_popularity_bin(
            test_scores, test_labels, test_groups, test_df,
            feature_spec=feature_spec, bin_fn=bin_fn, k=k,
        )

    # ─── 6. Собираем RunRecord ───────────────────────────────────────────
    record = RunRecord(
        run_id=run_id,
        name=cfg.name,
        config_hash=cfg.hash(),
        dataset=cfg.data.dataset,
        model=cfg.model.kind,
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
        train_time_sec=float(fit_meta.get("train_time_sec", 0.0)),
        n_params=model.n_params(),
        n_eval_groups=n_eval_groups,
        n_total_groups=n_total_groups,
        extras=fit_meta,
    )

    # ─── 7. Сохраняем ────────────────────────────────────────────────────
    append_run_record(record, runs_parquet)
    print(record.summary())
    return record


def _row_to_record(row: dict) -> RunRecord:
    """Восстанавливает RunRecord из строки parquet (для skip-ветки)."""
    import json
    # достаём pop-bin'ы обратно из плоских колонок
    ndcg_bins = {
        k.replace("ndcg_", ""): v
        for k, v in row.items()
        if k.startswith("ndcg_") and k not in (
            "ndcg_at_k", "ndcg_ci_low", "ndcg_ci_high"
        ) and v is not None and not (isinstance(v, float) and np.isnan(v))
    }
    n_groups_bins = {
        k.replace("n_groups_", ""): int(v)
        for k, v in row.items()
        if k.startswith("n_groups_") and v is not None
        and not (isinstance(v, float) and np.isnan(v))
    }
    extras = json.loads(row.get("extras", "{}") or "{}")

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
