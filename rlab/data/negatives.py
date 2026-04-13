"""
Стратегии негативного сэмплирования.

В v1 поддерживаем две: 'random' (uniform) и 'popularity' (∝ item_popularity).
Hard-mining отложен — он требует тёплого старта модели и ломает
идемпотентность sweep'а.

Контракт любого сэмплера:
    sample_negatives(pos_item, forbidden, item_pool, n_neg, rng, **kwargs)
        → np.ndarray shape (n_neg,) с item_idx негативов.

forbidden — set, куда уже входят pos_item и вся история юзера:
не хотим сэмплить то, что юзер уже видел.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sample_negatives_random(
    pos_item: int,
    forbidden: set[int],
    item_pool: np.ndarray,
    n_neg: int,
    rng: np.random.Generator,
    **_kwargs,
) -> np.ndarray:
    """
    Равномерный сэмплинг. Берём candidates пачками n_neg*8 и фильтруем
    forbidden — на больших item_pool'ах это быстрее, чем одиночный reject sampling.
    """
    negatives: list[int] = []
    used: set[int] = set()
    pool_size = len(item_pool)
    # максимум 20 попыток × 8n — защита от бесконечного цикла
    # на микро-датасетах, где |item_pool| ~ n_neg.
    for _ in range(20):
        if len(negatives) >= n_neg:
            break
        candidates = rng.choice(item_pool, size=min(n_neg * 8, pool_size), replace=True)
        for c in candidates:
            c_int = int(c)
            if c_int in forbidden or c_int in used:
                continue
            negatives.append(c_int)
            used.add(c_int)
            if len(negatives) >= n_neg:
                break
    return np.asarray(negatives[:n_neg], dtype=np.int64)


def sample_negatives_popularity(
    pos_item: int,
    forbidden: set[int],
    item_pool: np.ndarray,
    n_neg: int,
    rng: np.random.Generator,
    pop_probs: np.ndarray,   # shape (len(item_pool),), ∑=1
    **_kwargs,
) -> np.ndarray:
    """
    Сэмплинг с весами, пропорциональными популярности.
    pop_probs должен быть выровнен по item_pool (индекс-в-индекс).

    Зачем: популярные айтемы — более «информативные» негативы
    (их модель чаще путает с позитивом). Это стандартный приём,
    применяется в word2vec negative sampling и многих рекомендерах.
    Для H2 полезно сравнить: не схлопнется ли различие head/tail,
    если negatives будут искусственно head-heavy.
    """
    negatives: list[int] = []
    used: set[int] = set()
    pool_size = len(item_pool)
    for _ in range(20):
        if len(negatives) >= n_neg:
            break
        candidates = rng.choice(item_pool, size=min(n_neg * 8, pool_size),
                                replace=True, p=pop_probs)
        for c in candidates:
            c_int = int(c)
            if c_int in forbidden or c_int in used:
                continue
            negatives.append(c_int)
            used.add(c_int)
            if len(negatives) >= n_neg:
                break
    return np.asarray(negatives[:n_neg], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Фабрика по имени стратегии
# ─────────────────────────────────────────────────────────────────────────────
_SAMPLERS: dict[str, Callable] = {
    "random":     sample_negatives_random,
    "popularity": sample_negatives_popularity,
}


def get_sampler(strategy: str) -> Callable:
    if strategy not in _SAMPLERS:
        raise KeyError(
            f"Unknown neg_strategy '{strategy}'. "
            f"Available: {list(_SAMPLERS)}"
        )
    return _SAMPLERS[strategy]


def build_popularity_probs(
    item_popularity: dict[int, int],
    item_pool: np.ndarray,
    alpha: float = 0.75,
) -> np.ndarray:
    """
    Вероятности сэмплинга для popularity-стратегии.

    p(i) ∝ pop(i) ** alpha. alpha=0.75 — как в word2vec: сглаживает
    распределение, не давая топ-айтемам полностью доминировать.

    Возвращает массив длины len(item_pool), выровненный с item_pool
    (нужен именно такой порядок для rng.choice(..., p=...)).
    """
    pops = np.asarray(
        [item_popularity.get(int(i), 0) for i in item_pool],
        dtype=np.float64,
    )
    # +1 чтобы айтемы, которые не встречались в train, имели ненулевой
    # шанс попасть в негативы (редкость, но возможно на корнер-кейсах).
    weights = (pops + 1.0) ** alpha
    return weights / weights.sum()