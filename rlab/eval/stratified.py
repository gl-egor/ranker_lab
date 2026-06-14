"""
Стратифицированная оценка по популярности айтемов (H2).

Идея:
  1. По train-выборке считаем количество появлений каждого айтема.
  2. Бьём айтемы на бины по квантилям популярности (tail → head).
  3. Для каждой group'ы теста определяем бин ПО ЕЁ ПОЗИТИВНОМУ айтему.
  4. Считаем среднее NDCG@k отдельно на каждом бине.

Важно: биннинг считается по TRAIN, а не по test — иначе утечка
(на тесте могут появиться айтемы из «тёплой» части распределения,
которых в train не было, и они ошибочно попадут в head).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlab.eval.metrics import _metrics_for_rank, _rank_of_positive
from rlab.models.base import FeatureSpec


# Имена бинов в порядке возрастания популярности.
# Для n_bins=4 получим tail / q2 / q3 / head.
_BIN_NAMES_BY_N: dict[int, list[str]] = {
    2: ["tail", "head"],
    3: ["tail", "mid", "head"],
    4: ["tail", "q2", "q3", "head"],
    5: ["tail", "q2", "mid", "q4", "head"],
}


def make_popularity_bins(
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    n_bins: int = 4,
):
    """
    Возвращает функцию bin_fn(item_idx) -> str (имя бина).

    Считаем популярность ТОЛЬКО по позитивам train'а (label==1):
    строки train_df включают и негативы, их в счёт брать нельзя.

    Бьём на квантили равного размера (по числу айтемов, не по
    общему числу показов) — так tail-бин всегда содержит ~25%
    уникальных айтемов, а не «все айтемы с 1 показом».
    """
    if n_bins not in _BIN_NAMES_BY_N:
        raise ValueError(f"n_bins must be one of {list(_BIN_NAMES_BY_N)}")

    item_col = "item_idx"  # соглашение проекта; если поменяем — правим здесь
    label_col = feature_spec.target_col

    positives = train_df[train_df[label_col] == 1]
    pop = positives.groupby(item_col).size()

    # qcut по уникальным айтемам — даёт равные по ёмкости корзины
    # для «словаря айтемов», что интерпретируется как «нижний квартиль
    # айтемов по популярности».
    bin_names = _BIN_NAMES_BY_N[n_bins]
    try:
        binned = pd.qcut(pop.values, q=n_bins, labels=bin_names, duplicates="drop")
    except ValueError:
        # слишком много айтемов с одинаковой популярностью (например,
        # все = 1 на мини-датасете) → бьём по rank, а не по значению
        ranks = pop.rank(method="first")
        binned = pd.qcut(ranks, q=n_bins, labels=bin_names)

    # item_idx → bin_name
    item_to_bin: dict[int, str] = dict(zip(pop.index.values, map(str, binned)))
    # айтемы, не встреченные в train — по умолчанию tail
    default_bin = bin_names[0]

    def bin_fn(item_idx: int) -> str:
        return item_to_bin.get(int(item_idx), default_bin)

    return bin_fn


def ndcg_by_popularity_bin(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    bin_fn,
    k: int = 10,
) -> tuple[dict[str, float], dict[str, int]]:
    """
    NDCG@k по бинам популярности + число групп в каждом бине.

    Группу относим к бину по её позитивному айтему — это естественно,
    потому что NDCG определяется тем, как высоко всплывает позитив.

    Возвращает:
        ndcg_by_bin:     {'tail': 0.08, 'q2': 0.13, ..., 'head': 0.22}
        n_groups_by_bin: {'tail': 2340, ...}   ← размеры для интерпретации
    """
    item_col = "item_idx"
    label_col = feature_spec.target_col
    group_col = feature_spec.group_col

    # собираем DataFrame со всеми нужными колонками
    df = pd.DataFrame({
        "pred":     scores,
        "label":    labels,
        "group_id": groups,
        item_col:   test_df[item_col].values,
    })

    # для каждой группы: какой бин у её позитива?
    positives = df[df["label"] == 1].drop_duplicates("group_id")
    positives["bin"] = positives[item_col].map(bin_fn)
    group_to_bin = dict(zip(positives["group_id"].values, positives["bin"].values))

    # NDCG по группам
    ndcg_and_bin: list[tuple[str, float]] = []
    for gid, g in df.groupby("group_id", sort=True):
        rank = _rank_of_positive(g)
        ndcg, _, _ = _metrics_for_rank(rank, k)
        bin_name = group_to_bin.get(gid, "tail")
        ndcg_and_bin.append((bin_name, ndcg))

    # агрегируем по бинам
    tmp = pd.DataFrame(ndcg_and_bin, columns=["bin", "ndcg"])
    ndcg_by_bin = tmp.groupby("bin")["ndcg"].mean().to_dict()
    n_groups_by_bin = tmp.groupby("bin").size().to_dict()

    # приводим к плавающим / int — чтобы RunRecord сериализовался в parquet
    ndcg_by_bin = {str(k): float(v) for k, v in ndcg_by_bin.items()}
    n_groups_by_bin = {str(k): int(v) for k, v in n_groups_by_bin.items()}
    return ndcg_by_bin, n_groups_by_bin


def make_history_len_bins(
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    n_bins: int = 3,
):
    """
    Бинning групп по history_len позитива: short / medium / long.
    Квантили считаются по позитивам test_df.
    """
    if n_bins not in _BIN_NAMES_BY_N:
        raise ValueError(f"n_bins must be one of {list(_BIN_NAMES_BY_N)}")

    label_col = feature_spec.target_col
    group_col = feature_spec.group_col
    positives = test_df[test_df[label_col] == 1].drop_duplicates(group_col)
    values = positives["history_len"].values
    bin_names = _BIN_NAMES_BY_N[n_bins]
    try:
        binned = pd.qcut(values, q=n_bins, labels=bin_names, duplicates="drop")
    except ValueError:
        ranks = pd.Series(values).rank(method="first")
        binned = pd.qcut(ranks, q=n_bins, labels=bin_names)

    group_to_bin = dict(zip(positives[group_col].values, map(str, binned)))
    default_bin = bin_names[len(bin_names) // 2]

    def bin_fn(group_id: int) -> str:
        return group_to_bin.get(int(group_id), default_bin)

    return bin_fn


def make_user_warmth_bins(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
):
    """
    cold = user не встречался в train, warm = встречался.
    """
    if "user_idx" not in train_df.columns or "user_idx" not in test_df.columns:
        def bin_fn(_group_id: int) -> str:
            return "unknown"
        return bin_fn

    warm_users = set(train_df["user_idx"].unique())
    group_col = feature_spec.group_col
    label_col = feature_spec.target_col
    positives = test_df[test_df[label_col] == 1].drop_duplicates(group_col)
    group_to_warm = {
        int(gid): ("warm" if int(uid) in warm_users else "cold")
        for gid, uid in zip(positives[group_col], positives["user_idx"])
    }

    def bin_fn(group_id: int) -> str:
        return group_to_warm.get(int(group_id), "cold")

    return bin_fn


def make_item_density_bins(
    train_df: pd.DataFrame,
    feature_spec: FeatureSpec,
):
    """
    sparse / dense по медиане item_popularity (train positives).
    """
    label_col = feature_spec.target_col
    positives = train_df[train_df[label_col] == 1]
    pop = positives.groupby("item_idx")["item_popularity"].first()
    median_pop = float(pop.median())
    item_to_bin = {
        int(item): ("dense" if p >= median_pop else "sparse")
        for item, p in pop.items()
    }

    def bin_fn(item_idx: int) -> str:
        return item_to_bin.get(int(item_idx), "sparse")

    return bin_fn


def ndcg_by_group_segment(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    group_bin_fn,
    k: int = 10,
) -> tuple[dict[str, float], dict[str, int]]:
    """NDCG@k по сегментам, где группа относится к сегменту через group_bin_fn(group_id)."""
    group_col = feature_spec.group_col
    df = pd.DataFrame({
        "pred": scores,
        "label": labels,
        "group_id": groups,
    })

    ndcg_and_bin: list[tuple[str, float]] = []
    for gid, g in df.groupby(group_col, sort=True):
        rank = _rank_of_positive(g)
        ndcg, _, _ = _metrics_for_rank(rank, k)
        bin_name = group_bin_fn(int(gid))
        ndcg_and_bin.append((bin_name, ndcg))

    tmp = pd.DataFrame(ndcg_and_bin, columns=["bin", "ndcg"])
    ndcg_by_bin = {str(k): float(v) for k, v in tmp.groupby("bin")["ndcg"].mean().to_dict().items()}
    n_groups_by_bin = {str(k): int(v) for k, v in tmp.groupby("bin").size().to_dict().items()}
    return ndcg_by_bin, n_groups_by_bin


def ndcg_by_item_segment(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    item_bin_fn,
    k: int = 10,
) -> tuple[dict[str, float], dict[str, int]]:
    """NDCG@k по сегментам позитивного item (popularity, density)."""
    return ndcg_by_popularity_bin(
        scores, labels, groups, test_df, feature_spec, item_bin_fn, k=k,
    )


def gating_weights_by_segment(
    expert_weights: np.ndarray,
    groups: np.ndarray,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    segment_fn,
    model_kinds: list[str],
) -> dict[str, dict[str, float]]:
    """
    Средние веса экспертов по сегментам (на уровне групп).

    Returns: {segment: {model_kind: avg_weight}}
    """
    group_col = feature_spec.group_col
    label_col = feature_spec.target_col
    positives = test_df[test_df[label_col] == 1].drop_duplicates(group_col)
    pos_indices = positives.index.values

    group_ids = positives[group_col].values
    segments = [segment_fn(int(gid)) for gid in group_ids]

    pos_weights = expert_weights[pos_indices]
    tmp = pd.DataFrame(pos_weights, columns=model_kinds)
    tmp["segment"] = segments

    result: dict[str, dict[str, float]] = {}
    for seg, grp in tmp.groupby("segment"):
        result[str(seg)] = {
            kind: float(grp[kind].mean()) for kind in model_kinds
        }
    return result


def compute_all_segments(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    k: int = 10,
    n_pop_bins: int = 4,
) -> dict[str, dict[str, float]]:
    """Сводная таблица NDCG@k по всем сегментам для диплома."""
    segments: dict[str, dict[str, float]] = {}

    pop_fn = make_popularity_bins(train_df, feature_spec, n_bins=n_pop_bins)
    ndcg_pop, _ = ndcg_by_item_segment(
        scores, labels, groups, test_df, feature_spec, pop_fn, k=k,
    )
    segments["popularity"] = ndcg_pop

    hist_fn = make_history_len_bins(test_df, feature_spec, n_bins=3)
    ndcg_hist, _ = ndcg_by_group_segment(
        scores, labels, groups, test_df, feature_spec, hist_fn, k=k,
    )
    segments["history_len"] = ndcg_hist

    warm_fn = make_user_warmth_bins(train_df, test_df, feature_spec)
    ndcg_warm, _ = ndcg_by_group_segment(
        scores, labels, groups, test_df, feature_spec, warm_fn, k=k,
    )
    segments["user_warmth"] = ndcg_warm

    density_fn = make_item_density_bins(train_df, feature_spec)
    ndcg_dense, _ = ndcg_by_item_segment(
        scores, labels, groups, test_df, feature_spec, density_fn, k=k,
    )
    segments["item_density"] = ndcg_dense

    return segments