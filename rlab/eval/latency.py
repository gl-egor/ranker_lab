"""Замер времени инференса для Ranker.predict(df, feature_spec)."""

from __future__ import annotations

import statistics
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from rlab.models.base import FeatureSpec, Ranker

_PYTORCH_KINDS = frozenset({
    "dcnv2", "deepfm", "dcnv2_enhanced", "deepfm_enhanced",
})


@contextmanager
def inference_timer(name: str = "Inference"):
    """Контекстный менеджер: perf_counter, вывод в миллисекундах."""
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    print(f"[{name}] Заняло: {elapsed * 1000:.2f} ms")


def is_pytorch_ranker(model: "Ranker", model_kind: str | None = None) -> bool:
    """Нужна ли torch.cuda.synchronize() после predict."""
    if model_kind and model_kind in _PYTORCH_KINDS:
        return True
    device = getattr(model, "_device", None)
    return device in ("cuda", "mps")


def _sync_after_predict(use_cuda_sync: bool) -> None:
    if not use_cuda_sync:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def make_latency_sample(
    df: pd.DataFrame,
    feature_spec: "FeatureSpec",
    n_groups: int = 1,
) -> pd.DataFrame:
    """
    Срез DataFrame на первые n_groups query-групп.
    Одна группа ≈ 1 позитив + n_neg_eval кандидатов — типичная единица инференса.
    """
    group_col = feature_spec.group_col
    groups = df[group_col].unique()[:n_groups]
    return df[df[group_col].isin(groups)].copy().reset_index(drop=True)


def measure_inference_latency(
    model: "Ranker",
    sample_df: pd.DataFrame,
    feature_spec: "FeatureSpec",
    *,
    num_runs: int = 100,
    warmup_runs: int = 10,
    model_kind: str | None = None,
    name: str = "Inference",
) -> dict[str, Any]:
    """
    Warm-up + num_runs замеров predict на одном и том же sample_df.

    Возвращает:
        mean_ms, p95_ms, n_runs, sample_rows, sample_groups
    """
    use_sync = is_pytorch_ranker(model, model_kind)

    print(f"[latency] Прогрев ({warmup_runs} runs, {len(sample_df)} rows)...")
    for _ in range(warmup_runs):
        model.predict(sample_df, feature_spec)
        _sync_after_predict(use_sync)

    latencies: list[float] = []
    print(f"[latency] Запуск {num_runs} итераций ({name})...")
    for _ in range(num_runs):
        start = perf_counter()
        model.predict(sample_df, feature_spec)
        _sync_after_predict(use_sync)
        latencies.append((perf_counter() - start) * 1000)

    mean_ms = float(statistics.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    print(f"[latency] Mean: {mean_ms:.2f} ms  P95: {p95_ms:.2f} ms")

    return {
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "n_runs": num_runs,
        "sample_rows": len(sample_df),
        "sample_groups": int(sample_df[feature_spec.group_col].nunique()),
    }
