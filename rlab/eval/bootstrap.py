"""
Bootstrap CI для метрик ранжирования.

Единица ресэмплинга — ГРУППА (query), а не строка. Это важно:
строки внутри одной группы сильно коррелированы (один и тот же юзер
с одним и тем же позитивом), ресэмплинг по строкам занизит дисперсию
и даст оптимистично узкий CI.

Используется в H1: сравниваем CatBoost vs DCN-v2 на разных train_size
и смотрим, пересекаются ли их 95% CI.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    per_group_values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    agg: str = "mean",
) -> tuple[float, float]:
    """
    Percentile bootstrap CI.

    Параметры:
        per_group_values : 1-D массив значений метрики по группам
                           (из per_group_ndcg или аналога).
        n_boot           : число ресэмплов. 1000 — хватает для 95% CI
                           с разумной стабильностью.
        alpha            : уровень значимости (0.05 → 95% CI).
        seed             : для воспроизводимости.
        agg              : 'mean' | 'median'. Обычно mean (он и есть
                           средняя метрика по датасету).

    Возвращает (low, high) — percentile bootstrap quantiles.
    """
    if len(per_group_values) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(per_group_values)

    agg_fn = np.mean if agg == "mean" else np.median
    stats = np.empty(n_boot, dtype=np.float64)

    # Индексы сразу в матрице (n_boot, n) — по памяти ок для наших
    # размеров (10k групп × 1000 бутстрапов = 10M int — ~80MB).
    # Если когда-нибудь не хватит — можно свернуть в цикл.
    idx = rng.integers(0, n, size=(n_boot, n))
    resampled = per_group_values[idx]              # (n_boot, n)
    stats = agg_fn(resampled, axis=1)              # (n_boot,)

    low  = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    return low, high


def ci_overlap(
    ci_a: tuple[float, float],
    ci_b: tuple[float, float],
) -> bool:
    """
    Пересекаются ли два CI. Используется для H1:
        если CI CatBoost и CI DCN пересекаются — «ничья», гипотеза
        «разница в пользу X вне CI» не подтверждена.
    Note: непересечение CI — более строгий критерий, чем p < 0.05,
    но для наших целей (грубое сравнение моделей) этого хватает.
    """
    low_a, high_a = ci_a
    low_b, high_b = ci_b
    return not (high_a < low_b or high_b < low_a)