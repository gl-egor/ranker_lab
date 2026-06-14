"""
Типизированные конфиги эксперимента.

Идея: один ExperimentConfig = один воспроизводимый запуск.
Всё, что влияет на результат, должно лежать здесь (и больше нигде).
Это даёт две вещи:
  1) хэш конфига = идентификатор запуска (нет дублей в sweep);
  2) сериализация в JSON = артефакт, по которому можно повторить.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# DataConfig — всё, что относится к подготовке датасета
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    """
    dataset      : ключ датасета, по нему loader.py выбирает реализацию
                   ('books5', 'yelp', 'ml1m', ...).
    train_size   : число query-групп в train. None = использовать всё.
                   Именно этот параметр sweep'им для H1.
    valid_size   : фиксируем, чтобы по разным train_size сравнивать
                   на одной и той же валидации.
    test_size    : аналогично.
    feature_set  : имя набора фичей из feature_groups (см. FeatureConfig).
                   Для H3 sweep'им между 'full' и 'no_cross'.
                   'no_hcf' — ids + user_mean_rating, item_popularity,
                   item_mean_rating (честное сравнение с CatBoost).
    n_neg_train  : кол-во негативов на позитив в train.
    n_neg_eval   : то же для valid/test (обычно больше, например 50-100).
    neg_strategy : 'random' | 'popularity' | 'hard'. Hard требует тёплого старта.
    """
    dataset: str = "books5"
    train_size: int | None = 30_000
    valid_size: int = 10_000
    test_size: int = 10_000
    feature_set: str = "full"
    n_neg_train: int = 5
    n_neg_eval: int = 50
    neg_strategy: str = "random"
    # Ограничение на число уникальных юзеров. None = все.
    # Работает ДО сплита: случайно сэмплируем N юзеров и оставляем
    # только их взаимодействия. Сильно ускоряет отладку на Books
    # (22M взаимодействий → ~1M при max_users=10_000).
    max_users: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# ModelConfig — что за модель и с какими гиперпараметрами
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    kind   : 'catboost' | 'dcnv2' | 'deepfm'. По этому ключу
             runner выбирает реализацию из rlab.models.
    params : словарь гиперпараметров модели. Каждая модель сама знает,
             какие ключи она ждёт (см. её from_config).
             Намеренно dict, а не вложенный dataclass: так sweep по
             гиперпараметрам не требует менять типы.
    """
    kind: str = "catboost"
    params: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# EvalConfig — как мерить качество
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    """
    k               : отсечка для NDCG@k / HR@k / MRR@k.
    stratify_by_pop : считать ли NDCG отдельно по head/mid/tail бинам (H2).
    n_pop_bins      : число бинов популярности (4 = квартили).
    bootstrap_n     : число bootstrap-ресэмплов для CI (H1). 0 = выключить.
    bootstrap_alpha : уровень значимости (0.05 = 95% CI).
    """
    k: int = 10
    stratify_by_pop: bool = True
    n_pop_bins: int = 4
    bootstrap_n: int = 1000
    bootstrap_alpha: float = 0.05
    eval_warm_only: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# StackingConfig — contextual adaptive stacking meta-ranker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StackingConfig:
    """
    base_models       : список kind базовых ранкеров для ensemble.
    base_model_params : гиперпараметры по kind (если пусто — cfg.model.params).
    n_oof_folds       : K для OOF-скоров на train.
    calibrate         : Platt scaling перед gating network.
    context_features  : контекстные фичи для gating (user/item stats).
    gate_hidden       : размер скрытого слоя gating MLP.
    gate_layers       : число скрытых слоёв.
    gate_lr           : learning rate gating network.
    gate_epochs       : max эпох обучения gating.
    gate_patience     : early stopping patience.
    gate_dropout      : dropout в gating MLP.
    groups_per_batch  : групп на батч при обучении gating.
    force_recompute   : пересчитать base scores даже если есть кеш.
    """
    base_models: list[str] = field(
        default_factory=lambda: ["catboost", "dcnv2_enhanced", "finalmlp"]
    )
    base_model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    n_oof_folds: int = 5
    calibrate: bool = True
    context_features: list[str] = field(default_factory=lambda: [
        "history_len",
        "item_popularity_log",
        "user_interaction_count",
        "item_mean_rating",
        "user_mean_rating",
    ])
    gate_hidden: int = 32
    gate_layers: int = 2
    gate_lr: float = 1e-3
    gate_epochs: int = 50
    gate_patience: int = 10
    gate_dropout: float = 0.1
    groups_per_batch: int = 256
    force_recompute: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentConfig — корень
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """
    name       : человекочитаемая метка, попадает в run_id и в логи.
    data       : DataConfig.
    model      : ModelConfig.
    eval       : EvalConfig.
    seed       : единый seed для numpy/torch/catboost — воспроизводимость.
    output_dir : куда писать runs.parquet и артефакты каждого запуска.
    cache_dir  : куда кешировать подготовленные датасеты
                 (parquet с фичами). Сильно ускоряет sweep.
    device     : 'cuda' | 'cpu'. Для catboost не важно, для нейронок — да.
    """
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    output_dir: str = "./results"
    cache_dir: str = "./cache"
    device: str = "cuda"
    stacking: StackingConfig = field(default_factory=StackingConfig)

    # ─── сериализация ────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Превращает вложенные dataclass'ы в обычный dict (для JSON)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON-строка для сохранения рядом с артефактами запуска."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def hash(self) -> str:
        """
        Детерминистический короткий хэш конфига.
        Два одинаковых конфига → одинаковый hash → можно пропустить в sweep.
        Исключаем из хэша output_dir / cache_dir — они не влияют на результат.
        """
        payload = self.to_dict()
        payload.pop("output_dir", None)
        payload.pop("cache_dir", None)
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.md5(blob).hexdigest()[:10]

    def run_id(self) -> str:
        """Уникальный ID запуска: имя + хэш конфига."""
        return f"{self.name}__{self.hash()}"
