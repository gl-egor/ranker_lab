"""
Продвинутые кодирования признаков из семинара Яндекса.

Два компонента:

1) PiecewiseLinearScaler — замена StandardScaler.
   Вместо простого (x - mean) / std каждый числовой признак
   разбивается на n_bins «участков» по квантилям train-данных.
   Значение кодируется как вектор длины n_bins, где большинство
   элементов = 0 или 1, а в «активном» бине — дробное число
   от 0 до 1 (линейная интерполяция). Результат: вместо 1 числа
   на признак модель видит n_bins чисел — гораздо богаче для MLP.

   Интерфейс совместим со StandardScaler:
       scaler = PiecewiseLinearScaler(n_bins=32)
       scaler.fit(X_train)          # X_train: np.ndarray (N, n_features)
       X_encoded = scaler.transform(X)  # → (N, n_features * n_bins)

2) MultihashEmbedding — замена пары nn.Embedding(n_users) + nn.Embedding(n_items).
   Вместо отдельной строки в таблице для каждого ID мы хешируем ID
   в num_hashes позиций общей таблицы размера cardinality и усредняем
   полученные эмбеддинги. Выгода: tail-айтемы, у которых мало данных
   для обучения собственного вектора, «делят» параметры через хеш-
   коллизии → получают более осмысленные представления.

   Источник: "Unified Embedding: Battle-Tested Feature Representations
   for Web-Scale ML Systems" (Shi et al., Google DeepMind, 2024).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# 1. PiecewiseLinearScaler
# ═════════════════════════════════════════════════════════════════════════════

class PiecewiseLinearScaler:
    """
    Кусочно-линейное кодирование числовых признаков.

    Как это работает (на примере одного признака с 4 бинами):
    ────────────────────────────────────────────────────────────
    Допустим, квантили разбили признак на участки:
        bin_edges = [0, 10, 30, 70, 100]   (5 границ → 4 бина)

    Для значения x = 25:
        - бин 0 [0,  10]: x > 10  → полностью пройден  → 1.0
        - бин 1 [10, 30]: x = 25  → частично            → (25-10)/(30-10) = 0.75
        - бин 2 [30, 70]: x < 30  → не достигнут        → 0.0
        - бин 3 [70,100]: x < 70  → не достигнут        → 0.0

    Итого вектор: [1.0, 0.75, 0.0, 0.0]

    Это как «термометр»: заполняется слева направо, в активном
    бине — плавный переход. MLP видит и «сколько» (дробная часть),
    и «где» (какие бины заполнены) — намного информативнее, чем
    одно число после StandardScaler.

    Параметры:
        n_bins : число бинов на каждый признак (default 32).
                 Больше бинов = точнее, но растёт размерность входа.
                 32 — хороший баланс (по статье и семинару).
    """

    def __init__(self, n_bins: int = 32):
        self.n_bins = n_bins
        # Заполняются в fit():
        self._bin_edges: list[np.ndarray] | None = None  # длина n_features
        self._n_features: int = 0

    # ── fit: вычисляем границы бинов по квантилям train ───────────────
    def fit(self, X: np.ndarray) -> "PiecewiseLinearScaler":
        """
        X: np.ndarray shape (N, n_features).
        Для каждого признака считаем (n_bins + 1) квантиль
        от 0% до 100% — это и будут границы бинов.
        """
        self._n_features = X.shape[1]
        self._bin_edges = []

        # Квантили равномерно от 0 до 1
        quantile_levels = np.linspace(0.0, 1.0, self.n_bins + 1)

        for col_idx in range(self._n_features):
            col = X[:, col_idx].astype(np.float64)
            edges = np.quantile(col, quantile_levels)
            # np.unique убирает дубли (если много одинаковых значений)
            edges = np.unique(edges)
            self._bin_edges.append(edges)

        return self

    # ── transform: кодируем каждый признак в n_bins чисел ─────────────
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, n_features) → выход: (N, total_bins).
        total_bins = sum(len(edges)-1 для каждого признака).

        Обычно total_bins ≈ n_features * n_bins, но может быть
        чуть меньше, если у какого-то признака мало уникальных
        значений (тогда np.unique сократил число границ).
        """
        if self._bin_edges is None:
            raise RuntimeError("Сначала вызови fit()!")

        N = X.shape[0]
        encoded_parts: list[np.ndarray] = []

        for col_idx in range(self._n_features):
            edges = self._bin_edges[col_idx]
            n_bins_col = len(edges) - 1

            if n_bins_col == 0:
                # Если все значения одинаковые — один бин, всегда 1.0
                encoded_parts.append(np.ones((N, 1), dtype=np.float32))
                continue

            col = X[:, col_idx].astype(np.float64)

            # Для каждого бина k считаем:
            #   weight[k] = 1 / (right - left)
            #   encoded[k] = clamp((x - left) * weight, 0, 1)
            result = np.zeros((N, n_bins_col), dtype=np.float32)
            for k in range(n_bins_col):
                left = edges[k]
                right = edges[k + 1]
                width = right - left
                if width < 1e-12:
                    # Вырожденный бин — ставим 1 где x >= left
                    result[:, k] = (col >= left).astype(np.float32)
                else:
                    result[:, k] = np.clip(
                        (col - left) / width, 0.0, 1.0
                    ).astype(np.float32)

            encoded_parts.append(result)

        return np.concatenate(encoded_parts, axis=1)

    # ── Утилиты ───────────────────────────────────────────────────────
    @property
    def output_dim(self) -> int:
        """Сколько колонок выдаёт transform. Нужно модели для input_dim."""
        if self._bin_edges is None:
            raise RuntimeError("Сначала вызови fit()!")
        return sum(max(len(e) - 1, 1) for e in self._bin_edges)


# ═════════════════════════════════════════════════════════════════════════════
# 2. MultihashEmbedding
# ═════════════════════════════════════════════════════════════════════════════

class MultihashEmbedding(nn.Module):
    """
    Хеш-эмбеддинг для одного категориального поля (user ИЛИ item).

    Как это работает (на примере item_idx = 42, num_hashes = 3):
    ────────────────────────────────────────────────────────────
    1. Берём 3 «соли» (seeds): [1337, 7777, 2342]
    2. Вычисляем 3 хеш-позиции в таблице:
         h0 = (42 + 1337) % cardinality = 1379
         h1 = (42 + 7777) % cardinality = 7819
         h2 = (42 + 2342) % cardinality = 2384
    3. Достаём 3 эмбеддинга из ОДНОЙ таблицы:
         e0 = table[1379],  e1 = table[7819],  e2 = table[2384]
    4. Усредняем: embedding = (e0 + e1 + e2) / 3

    Почему это помогает tail-айтемам:
    ─────────────────────────────────
    При обычном nn.Embedding(n_items, dim) редкий айтем (2 примера
    в train) получает свой вектор, который почти не обучается.
    С хешированием он «делит» 3 строки таблицы с другими айтемами,
    которые попали в те же хеш-позиции. Эти соседи могут быть
    популярными → их градиенты «подтягивают» общие параметры →
    редкий айтем получает более осмысленный вектор.

    Параметры:
        cardinality : размер общей таблицы эмбеддингов.
                      Чем больше — тем меньше коллизий (и ближе
                      к обычному nn.Embedding). 65536 — разумный
                      дефолт для датасетов с 100k–200k айтемов.
        emb_dim     : размерность каждого вектора.
        num_hashes  : число хеш-функций. 2–3 хватает.
        seed_offset : сдвиг для генерации солей (чтобы user и item
                      использовали разные хеши).
    """

    def __init__(
        self,
        cardinality: int = 65536,
        emb_dim: int = 32,
        num_hashes: int = 3,
        seed_offset: int = 0,
    ):
        super().__init__()
        self.cardinality = cardinality
        self.num_hashes = num_hashes
        self.emb_dim = emb_dim

        # Общая таблица — единственный набор обучаемых параметров
        self.embedding = nn.Embedding(cardinality, emb_dim, padding_idx=0)

        # Соли для хеш-функций — фиксированные, не обучаются.
        # Разные seed_offset для user и item гарантируют, что
        # user_42 и item_42 не попадут в одни и те же позиции.
        seeds = torch.tensor(
            [1337 + 1000 * k + seed_offset for k in range(num_hashes)],
            dtype=torch.long,
        )
        # register_buffer → сохраняется в state_dict, переезжает на GPU
        # вместе с моделью, но не участвует в optimizer.step()
        self.register_buffer("seeds", seeds)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (batch_size,) — исходные ID (user_idx или item_idx).
        Возвращает: (batch_size, emb_dim) — усреднённый эмбеддинг.
        """
        # idx: (B,) → (B, 1) + seeds: (H,) → broadcast → (B, H)
        hash_positions = (idx.unsqueeze(1) + self.seeds) % self.cardinality

        # Lookup: (B, H) → (B, H, emb_dim)
        embeddings = self.embedding(hash_positions)

        # Усреднение по хешам: (B, H, emb_dim) → (B, emb_dim)
        return embeddings.mean(dim=1)