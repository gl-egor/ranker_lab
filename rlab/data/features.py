"""
Фича-инжиниринг и feature_groups.

Ключ к H3: фичи объединены в именованные группы, и feature_set
(имя в DataConfig) разворачивается в список колонок через FEATURE_SETS.
Убирая группу 'cross' — мы отключаем user_item_prev_count и seen_before,
что и проверяет H3: сохранится ли преимущество CatBoost без них.

Агрегаты (item_popularity, user_mean_rating и т.п.) считаются ТОЛЬКО
по train-части (чтобы не было утечки во время eval).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Группы фичей — семантический слой
# ─────────────────────────────────────────────────────────────────────────────
# При добавлении новой фичи: (1) сгенерировать её в build_rank_table_rows,
# (2) вписать в нужную группу ниже.
FEATURE_GROUPS: dict[str, list[str]] = {
    # ids: метаданные для сохранения предсказаний при любом feature_set
    "ids":        ["user_idx", "item_idx"],
    "user_stats": ["history_len", "user_mean_rating", "user_interaction_count"],
    "item_stats": ["item_popularity", "item_popularity_log", "item_mean_rating"],
    "cross":      ["user_item_prev_count", "user_item_seen_before"],  # ← H3
}


# Именованные наборы для DataConfig.feature_set
FEATURE_SETS: dict[str, list[str]] = {
    "full":            ["ids", "user_stats", "item_stats", "cross"],
    "no_cross":        ["ids", "user_stats", "item_stats"],
    "no_cross_no_ids": ["user_stats", "item_stats"],
    "ids_only":        ["ids"],                # для sanity-check нейронок
}


def resolve_feature_set(name: str) -> list[str]:
    """
    'full' → ['user_idx', 'item_idx', 'history_len', ...].
    Сохраняет порядок групп и колонок внутри группы.
    """
    if name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature_set '{name}'. Available: {list(FEATURE_SETS)}")
    cols: list[str] = []
    seen: set[str] = set()
    for group_name in FEATURE_SETS[name]:
        for col in FEATURE_GROUPS[group_name]:
            if col not in seen:
                cols.append(col)
                seen.add(col)
    return cols


# Алиас для loader / post-hoc анализа (совместимость)
METADATA_COLS: list[str] = FEATURE_GROUPS["ids"]


def categorical_cols_in(feature_cols: list[str]) -> list[str]:
    """Какие из выбранных колонок — категориальные (только user_idx/item_idx)."""
    return [c for c in feature_cols if c in FEATURE_GROUPS["ids"]]


def numerical_cols_in(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c not in FEATURE_GROUPS["ids"]]


# ─────────────────────────────────────────────────────────────────────────────
# Агрегаты по train
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainAggregates:
    """
    Всё, что посчитано по train и нужно для генерации фичей на всех сплитах.
    Единое место хранения — проще кешировать и передавать.
    """
    item_popularity:     dict[int, int]
    item_mean_rating:    dict[int, float]
    user_mean_rating:    dict[int, float]
    user_interaction_count: dict[int, int]
    global_rating_mean:  float


def compute_train_aggregates(train_interactions: pd.DataFrame) -> TrainAggregates:
    """
    train_interactions: полный train до сплита query/candidate — одна строка
    на взаимодействие с колонками user_idx, item_idx, rating.

    Считаем: популярность айтема (кол-во взаимодействий), средний рейтинг
    айтема, средний рейтинг юзера, общий средний рейтинг.
    """
    item_popularity = train_interactions.groupby("item_idx").size().to_dict()
    item_mean_rating = train_interactions.groupby("item_idx")["rating"].mean().to_dict()
    user_mean_rating = train_interactions.groupby("user_idx")["rating"].mean().to_dict()
    user_interaction_count = train_interactions.groupby("user_idx").size().to_dict()
    global_rating_mean = float(train_interactions["rating"].mean())

    return TrainAggregates(
        item_popularity={int(k): int(v) for k, v in item_popularity.items()},
        item_mean_rating={int(k): float(v) for k, v in item_mean_rating.items()},
        user_mean_rating={int(k): float(v) for k, v in user_mean_rating.items()},
        user_interaction_count={int(k): int(v) for k, v in user_interaction_count.items()},
        global_rating_mean=global_rating_mean,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Генерация строки rank_table (positive / negative)
# ─────────────────────────────────────────────────────────────────────────────
def make_feature_row(
    user_idx: int,
    item_idx: int,
    history_items: list[int],
    history_counter: dict[int, int],
    agg: TrainAggregates,
    label: int,
    group_id: int,
) -> dict:
    """
    Одна строка rank_table. Генерируется одинаково для позитивов и негативов —
    разница только в label и в том, что для позитива seen_before у тебя в
    ноутбуке было жёстко 1, а здесь мы честно смотрим на history_counter
    (seen_before = 1, если айтем уже был в истории, иначе 0).

    Это небольшое отклонение от version2.ipynb, но оно честнее:
    так seen_before перестаёт быть leakage-фичей «позитив всегда 1»
    и превращается в реальный сигнал (для H3 это принципиально —
    иначе признак seen_before сам по себе выдаёт позитив).
    """
    prev_count = int(history_counter.get(int(item_idx), 0))
    pop = int(agg.item_popularity.get(int(item_idx), 0))
    return {
        "group_id":              int(group_id),
        "label":                 int(label),

        "user_idx":              int(user_idx),
        "item_idx":              int(item_idx),

        "history_len":           len(history_items),
        "user_mean_rating":      float(agg.user_mean_rating.get(int(user_idx),
                                                                 agg.global_rating_mean)),
        "user_interaction_count": int(agg.user_interaction_count.get(int(user_idx), 0)),

        "item_popularity":       pop,
        "item_popularity_log":   float(np.log1p(pop)),
        "item_mean_rating":      float(agg.item_mean_rating.get(int(item_idx),
                                                                 agg.global_rating_mean)),

        "user_item_prev_count":  prev_count,
        "user_item_seen_before": int(prev_count > 0),
    }