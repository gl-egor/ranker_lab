"""
Метрики ранжирования для задачи с одним позитивом на группу.

Формат датасета: каждая группа (query) = один позитив + N негативов.
В этом частном случае NDCG@k сводится к:
    1 / log2(rank_of_positive + 1)   if rank ≤ k else 0
HR@k  = (rank ≤ k), MRR@k = 1/rank if rank ≤ k else 0.

Код портирован из ranking_metrics_one_positive в version2.ipynb,
переписан на векторизованный pandas для скорости.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_of_positive(group: pd.DataFrame) -> int:
    """
    Ранг позитивного айтема в группе (1-indexed) по убыванию score.
    Ties разрешаются стабильной сортировкой (порядок появления в группе).
    Если позитива нет — возвращаем len(group)+1, такая группа даст все нули.
    """
    sorted_group = group.sort_values("pred", ascending=False, kind="stable")
    pos_mask = sorted_group["label"].values == 1
    if not pos_mask.any():
        return len(group) + 1
    return int(np.argmax(pos_mask)) + 1  # 1-indexed


def _metrics_for_rank(rank: int, k: int) -> tuple[float, float, float]:
    """(NDCG@k, HR@k, MRR@k) для одной группы с данным рангом позитива."""
    if rank > k:
        return 0.0, 0.0, 0.0
    return 1.0 / np.log2(rank + 1), 1.0, 1.0 / rank


def per_group_ndcg(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    NDCG@k для каждой группы отдельно. 1-D массив длины n_groups.
    Нужно для bootstrap CI — ресэмплируем именно эти значения.
    Порядок — по возрастанию group_id (sort=True), чтобы стратификация
    по popularity_bins (см. stratified.py) могла джойниться по индексу.
    """
    df = pd.DataFrame({"pred": scores, "label": labels, "group_id": groups})
    ndcgs = []
    for _, g in df.groupby("group_id", sort=True):
        rank = _rank_of_positive(g)
        ndcg, _, _ = _metrics_for_rank(rank, k)
        ndcgs.append(ndcg)
    return np.asarray(ndcgs, dtype=np.float64)


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
) -> dict[str, float]:
    """
    Усреднённые по группам NDCG@k / HR@k / MRR@k + n_groups.

    Контракт входа:
        scores, labels, groups — 1-D массивы одной длины,
        одна строка = один (user, item) кандидат.
        label ∈ {0, 1}, позитив в каждой группе ровно один.
    """
    df = pd.DataFrame({"pred": scores, "label": labels, "group_id": groups})
    ndcgs, hrs, mrrs = [], [], []
    for _, g in df.groupby("group_id", sort=True):
        rank = _rank_of_positive(g)
        ndcg, hr, mrr = _metrics_for_rank(rank, k)
        ndcgs.append(ndcg); hrs.append(hr); mrrs.append(mrr)
    return {
        "NDCG":     float(np.mean(ndcgs)),
        "HR":       float(np.mean(hrs)),
        "MRR":      float(np.mean(mrrs)),
        "n_groups": len(ndcgs),
    }