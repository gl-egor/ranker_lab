"""
Базовый контракт модели и структура записи результата эксперимента.

Ranker  — абстрактный интерфейс. Все модели (CatBoost, DCN-v2, DeepFM)
          реализуют эти три метода и становятся взаимозаменяемыми.
RunRecord — одна строка в results/runs.parquet. Всё, что нам нужно
            для анализа гипотез H1/H2/H3, должно в неё помещаться.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rlab.configs import ExperimentConfig
    from rlab.data.features import TrainAggregates


# ─────────────────────────────────────────────────────────────────────────────
# Feature spec — описание колонок, которое модели используют, чтобы понять,
# что категориально, что численно, и где таргет/группа.
# Заполняется в rlab/data/loader.py один раз на датасет.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FeatureSpec:
    target_col: str = "label"
    group_col: str = "group_id"
    categorical_cols: list[str] = field(default_factory=list)
    numerical_cols: list[str] = field(default_factory=list)
    cardinalities: dict[str, int] = field(default_factory=dict)  # для nn.Embedding
    train_aggregates: "TrainAggregates | None" = None

    @property
    def feature_cols(self) -> list[str]:
        """Все фичи подряд (cat + num), в порядке для DataFrame."""
        return self.categorical_cols + self.numerical_cols


# ─────────────────────────────────────────────────────────────────────────────
# Ranker — абстрактный базовый класс
# ─────────────────────────────────────────────────────────────────────────────
class Ranker(ABC):
    """
    Все модели реализуют этот интерфейс. Runner видит только его.

    Данные:
      - train_df/valid_df/test_df — pd.DataFrame, уже с фичами,
        одна строка = (user, item) кандидат, с колонками label/group_id.
      - Группы идут подряд, отсортированы по group_id (требование CatBoost
        и удобно для батчевания в нейронках).
      - feature_spec говорит, что категориально/численно.

    Predict:
      - на вход тот же формат DataFrame;
      - на выход 1-D numpy массив длины len(df) со скорами;
      - порядок скоров = порядок строк входа. Это критично для метрик.
    """

    name: str = "base"  # переопределяется в наследниках

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        feature_spec: FeatureSpec,
        params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """
        Обучает модель.

        Возвращает dict с произвольной метой обучения (опционально):
          best_iteration, train_time_sec, train_history, ...
        Эта мета потом попадает в RunRecord.extras — для анализа
        кривых обучения и времени.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_spec: FeatureSpec) -> np.ndarray:
        """Скоры длины len(df). Больше — релевантнее."""
        ...

    @abstractmethod
    def n_params(self) -> int:
        """
        Число обучаемых параметров. Для CatBoost — сумма листьев
        (приближённо), для nn — sum(p.numel() for p in parameters).
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Реестр моделей: строка из ModelConfig.kind → класс
# Регистрацию делают сами модули
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[Ranker]] = {}


def register_model(kind: str):
    """
    Декоратор для регистрации модели. Используется так:

        @register_model("catboost")
        class CatBoostRanker(Ranker):
            ...

    Потом runner делает build_model("catboost") и получает класс.
    Это лучше, чем if/elif — добавление модели не трогает runner.
    """
    def _decorator(cls: type[Ranker]) -> type[Ranker]:
        if kind in _REGISTRY:
            raise ValueError(f"Model '{kind}' already registered")
        _REGISTRY[kind] = cls
        cls.name = kind
        return cls
    return _decorator


def build_model(kind: str) -> Ranker:
    """Фабрика: по имени возвращает новый инстанс модели."""
    if kind not in _REGISTRY:
        raise KeyError(
            f"Unknown model '{kind}'. Registered: {list(_REGISTRY)}. "
            f"Did you forget to import rlab.models.{kind}_ranker?"
        )
    return _REGISTRY[kind]()


# ─────────────────────────────────────────────────────────────────────────────
# RunRecord — одна строка в runs.parquet
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RunRecord:
    """
    Метаданные об экспериментах
    """
    # ─── идентификация ───────────────────────────────────────────────────────
    run_id: str
    name: str
    config_hash: str

    # ─── что за эксперимент ──────────────────────────────────────────────────
    dataset: str
    model: str
    feature_set: str
    train_size: int
    seed: int

    # ─── общие метрики (весь тест) ───────────────────────────────────────────
    ndcg_at_k: float
    hr_at_k: float
    mrr_at_k: float
    k: int

    # ─── H1: bootstrap CI ────────────────────────────────────────────────────
    ndcg_ci_low: float | None = None
    ndcg_ci_high: float | None = None

    # ─── H2: стратификация по popularity ─────────────────────────────────────
    # Ключи: 'tail', 'q2', 'q3', 'head' (при n_pop_bins=4).
    ndcg_by_pop_bin: dict[str, float] = field(default_factory=dict)
    n_groups_by_pop_bin: dict[str, int] = field(default_factory=dict)

    # ─── мета обучения ───────────────────────────────────────────────────────
    train_time_sec: float = 0.0
    n_params: int = 0

    #warm validation
    n_eval_groups: int = 0
    n_total_groups: int = 0

    # ─── произвольные поля от конкретной модели ───────────
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Плоский dict для записи в parquet."""
        d = asdict(self)
        # разворачиваем dict'ы с метриками в отдельные колонки
        for bin_name, val in d.pop("ndcg_by_pop_bin").items():
            d[f"ndcg_{bin_name}"] = val
        for bin_name, val in d.pop("n_groups_by_pop_bin").items():
            d[f"n_groups_{bin_name}"] = val
        # extras оставляем как JSON-строку, чтобы parquet переварил
        import json
        d["extras"] = json.dumps(d["extras"], ensure_ascii=False)
        return d

    def summary(self) -> str:
        """Строчка для print после run_experiment."""
        lines = [
            f"[{self.run_id}]",
            f"  {self.model} on {self.dataset} "
            f"(train_size={self.train_size}, fs={self.feature_set}, seed={self.seed})",
            f"  NDCG@{self.k}={self.ndcg_at_k:.4f}  "
            f"HR@{self.k}={self.hr_at_k:.4f}  "
            f"MRR@{self.k}={self.mrr_at_k:.4f}",
        ]
        if self.ndcg_ci_low is not None:
            lines.append(
                f"  95% CI NDCG: [{self.ndcg_ci_low:.4f}, {self.ndcg_ci_high:.4f}]"
            )
        if self.ndcg_by_pop_bin:
            bins = "  ".join(
                f"{k}={v:.4f}" for k, v in self.ndcg_by_pop_bin.items()
            )
            lines.append(f"  NDCG by pop: {bins}")
        lines.append(
            f"  time={self.train_time_sec:.1f}s  params={self.n_params:,}"
        )
        return "\n".join(lines)


def append_run_record(record: RunRecord, runs_parquet_path: str) -> None:
    """
    Дописывает строчку в results/runs.parquet
    Реализовано как read→concat→write
    """
    import os
    row = pd.DataFrame([record.to_dict()])
    if os.path.exists(runs_parquet_path):
        existing = pd.read_parquet(runs_parquet_path)
        # если такой run_id уже есть — перезаписываем
        existing = existing[existing["run_id"] != record.run_id]
        row = pd.concat([existing, row], ignore_index=True)
    os.makedirs(os.path.dirname(runs_parquet_path) or ".", exist_ok=True)
    row.to_parquet(runs_parquet_path, index=False)
