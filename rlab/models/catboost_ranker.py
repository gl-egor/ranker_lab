"""
Обёртка CatBoostRanker под интерфейс Ranker.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from rlab.models.base import FeatureSpec, Ranker, register_model


#Любой параметр переопределяется через
# ModelConfig.params в конфиге эксперимента.
DEFAULT_PARAMS: dict[str, Any] = {
    "loss_function": "YetiRank",
    "eval_metric": "NDCG:top=10",
    "iterations": 500,
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 5,
    "verbose": 50,
    "use_best_model": True,
    "early_stopping_rounds": 50,  # добавил: при sweep'е train_size=10k
                                   # без ES оверфитимся и теряем время
}


@register_model("catboost")
class CatBoostRankerWrapper(Ranker):
    """
    Тонкая обёртка над catboost.CatBoostRanker.

    Важное:
      - user_idx и item_idx мы передаём как cat_features. CatBoost
        строит CTR-фичи автоматически — именно это даёт ему сильный
        сигнал на head-айтемах (релевантно для H2).
      - Группы в train_df/valid_df ДОЛЖНЫ идти подряд и быть отсортированы
        по group_id. Это инвариант всей системы (см. loader.py).
    """

    def __init__(self):
        self._model = None  # тип: CatBoostRanker | None
        self._feature_cols: list[str] | None = None
        self._cat_cols: list[str] | None = None

    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """
        Обучение. Сливает DEFAULT_PARAMS с пользовательскими params,
        собирает Pool'ы, зовёт .fit, запоминает best_iteration.
        """
        from catboost import CatBoostRanker, Pool

        self._feature_cols = feature_spec.feature_cols
        self._cat_cols = list(feature_spec.categorical_cols)

        # ── собираем финальные параметры ──────────────────────────────
        # Порядок: defaults ← user params ← seed (пользователь не должен
        # случайно перебить seed в params).
        effective = {**DEFAULT_PARAMS, **params, "random_seed": seed}

        # ── Pool'ы ────────────────────────────────────────────────────
        # хранит X + y + group_id + cat_features вместе.
        train_pool = Pool(
            data=train_df[self._feature_cols],
            label=train_df[feature_spec.target_col].values,
            group_id=train_df[feature_spec.group_col].values,
            cat_features=self._cat_cols,
        )
        valid_pool = Pool(
            data=valid_df[self._feature_cols],
            label=valid_df[feature_spec.target_col].values,
            group_id=valid_df[feature_spec.group_col].values,
            cat_features=self._cat_cols,
        )

        # ── обучение ──────────────────────────────────────────────────
        self._model = CatBoostRanker(**effective)
        t0 = time.time()
        self._model.fit(train_pool, eval_set=valid_pool)
        train_time = time.time() - t0

        return {
            "train_time_sec": train_time,
            "best_iteration": int(self._model.get_best_iteration() or 0),
            "tree_count": int(self._model.tree_count_),
        }

    # ─────────────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, feature_spec: FeatureSpec) -> np.ndarray:
        """Скоры. Группы/лейблы здесь не нужны — зовём .predict напрямую."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        from catboost import Pool

        pool = Pool(
            data=df[self._feature_cols],
            cat_features=self._cat_cols,
        )
        return np.asarray(self._model.predict(pool), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────
    def n_params(self) -> int:
        """
        Для деревьев "параметров" нет в строгом смысле. Возвращаем
        tree_count × средний листьев в дереве (грубая оценка ёмкости).
        Если нужна честная цифра — можно дёрнуть get_tree_leaf_counts().
        """
        if self._model is None:
            return 0
        try:
            leaf_counts = self._model.get_tree_leaf_counts()
            return int(sum(leaf_counts))
        except Exception:
            # fallback: для старых версий catboost
            return int(self._model.tree_count_) * (2 ** 8)
