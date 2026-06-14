"""
Platt scaling для нормализации скоров базовых моделей.

Каждая базовая модель имеет свой масштаб (YetiRank vs logits).
Калибровка переводит скоры в [0, 1] перед gating network.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Per-model Platt scaling: P(y=1) = sigmoid(a * score + b)."""

    def __init__(self):
        self._lr: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        X = scores.reshape(-1, 1)
        y = labels.astype(int)
        self._lr = LogisticRegression(C=1e10, max_iter=1000, solver="lbfgs")
        self._lr.fit(X, y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError("Calibrator not fitted")
        X = scores.reshape(-1, 1)
        return self._lr.predict_proba(X)[:, 1].astype(np.float32)


class MultiModelCalibrator:
    """Набор калибраторов — по одному на каждого эксперта."""

    def __init__(self, model_kinds: list[str]):
        self.model_kinds = model_kinds
        self.calibrators: dict[str, PlattCalibrator] = {
            k: PlattCalibrator() for k in model_kinds
        }

    def fit(
        self,
        score_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> "MultiModelCalibrator":
        """score_matrix: (n_rows, n_experts)."""
        for i, kind in enumerate(self.model_kinds):
            self.calibrators[kind].fit(score_matrix[:, i], labels)
        return self

    def transform(self, score_matrix: np.ndarray) -> np.ndarray:
        out = np.empty_like(score_matrix)
        for i, kind in enumerate(self.model_kinds):
            out[:, i] = self.calibrators[kind].transform(score_matrix[:, i])
        return out
