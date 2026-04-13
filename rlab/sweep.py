"""
Sweep — прогон сетки экспериментов.

run_sweep(sweep_cfg) берёт базовый ExperimentConfig и перебирает по
grid'у, где каждое значение — путь вида 'data.train_size' → список.
Для каждой комбинации строится модифицированный конфиг и вызывается
run_experiment. Идемпотентно за счёт skip_if_exists.

Пример:
    sweep = SweepConfig(
        name="H1_scale",
        base=base_cfg,
        grid={
            "data.train_size": [10_000, 30_000, 100_000],
            "model.kind": ["catboost", "dcnv2"],
            "seed": [42, 43, 44],
        },
    )
    run_sweep(sweep)   # 3 × 2 × 3 = 18 запусков
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any

from rlab.configs import ExperimentConfig
from rlab.models.base import RunRecord
from rlab.runner import run_experiment


@dataclass
class SweepConfig:
    """
    name            : метка sweep'а (используется в run_id: name + grid-значения).
    base            : базовый ExperimentConfig — начальная точка.
    grid            : dict 'dotted.path' → список значений.
                      Путь разрешается через __getattr__/setattr:
                      'data.train_size' → base.data.train_size.
                      Поддерживается 2 уровня вложенности (data/model/eval).
                      Для model.params.* используем путь 'model.params.iterations'.
    skip_if_exists  : прокидывается в run_experiment.
    name_template   : как именовать каждый подзапуск. По умолчанию — склейка
                      сокращений всех варьируемых параметров.
    """
    name: str
    base: ExperimentConfig
    grid: dict[str, list[Any]] = field(default_factory=dict)
    skip_if_exists: bool = True
    name_template: str | None = None


def _set_by_path(cfg: ExperimentConfig, path: str, value: Any) -> None:
    """
    Проставляет значение по dotted-пути. Поддерживает:
        'seed'                      → cfg.seed
        'data.train_size'           → cfg.data.train_size
        'model.params.iterations'   → cfg.model.params['iterations']

    Для последнего сегмента: если контейнер — dict, пишем по ключу;
    если dataclass — через setattr.
    """
    parts = path.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p) if not isinstance(obj, dict) else obj[p]
    leaf = parts[-1]
    if isinstance(obj, dict):
        obj[leaf] = value
    else:
        setattr(obj, leaf, value)


def _short_name_for_combo(combo: dict[str, Any]) -> str:
    """
    Короткое имя эксперимента из варьируемых значений.
    'data.train_size'=30000 → 'ts30000'
    'model.kind'='catboost' → 'catboost'
    'seed'=42               → 's42'
    """
    parts = []
    for path, val in combo.items():
        leaf = path.split(".")[-1]
        # несколько ручных сокращений для читаемости
        abbrev = {
            "train_size": "ts", "feature_set": "fs",
            "kind": "", "seed": "s", "n_neg_train": "nn",
        }.get(leaf, leaf)
        parts.append(f"{abbrev}{val}")
    return "_".join(p for p in parts if p)


def run_sweep(sweep_cfg: SweepConfig) -> list[RunRecord]:
    """
    Прогоняет все комбинации grid'а. Возвращает список RunRecord'ов.

    Порядок перебора: itertools.product по значениям grid в порядке
    объявления ключей. Это полезно для наблюдения: ставишь seed'ы
    последним — сначала видишь разнообразие моделей, потом сиды.
    """
    keys = list(sweep_cfg.grid.keys())
    value_lists = [sweep_cfg.grid[k] for k in keys]

    combos = list(itertools.product(*value_lists))
    print(f"[sweep] {sweep_cfg.name}: {len(combos)} runs")

    records: list[RunRecord] = []
    for i, values in enumerate(combos, 1):
        combo = dict(zip(keys, values))
        cfg = copy.deepcopy(sweep_cfg.base)

        # применяем точечные изменения к копии конфига
        for path, val in combo.items():
            _set_by_path(cfg, path, val)

        # имя эксперимента: sweep_name + сокращения варьируемых
        if sweep_cfg.name_template:
            cfg.name = sweep_cfg.name_template.format(**combo)
        else:
            cfg.name = f"{sweep_cfg.name}__{_short_name_for_combo(combo)}"

        print(f"\n[sweep {i}/{len(combos)}] {cfg.name}")
        try:
            rec = run_experiment(cfg, skip_if_exists=sweep_cfg.skip_if_exists)
            records.append(rec)
        except Exception as e:
            # в sweep'е один падающий конфиг не должен убивать остальные.
            # Но Colab, увидев print, не потеряет контекст.
            print(f"[sweep] FAILED {cfg.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[sweep] done: {len(records)}/{len(combos)} succeeded")
    return records
