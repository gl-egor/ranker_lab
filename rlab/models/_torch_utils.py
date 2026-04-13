"""
Общие утилиты для нейронных ранкеров (DCN-v2, DeepFM и следующих).

Вынесено отдельно, чтобы не копипастить между dcnv2_ranker и deepfm_ranker:
    - RankTableDataset   — превращает DataFrame в тензоры
    - GroupBatchSampler  — батчи целыми группами (для listwise loss)
    - group_softmax_loss — listwise log-softmax по группе

Ни один из этих компонентов не знает про конкретную архитектуру модели.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from rlab.models.base import FeatureSpec


class RankTableDataset(Dataset):
    """
    Оборачивает rank_table DataFrame в тензоры.


    - df имеет колонки user_idx, item_idx (cat) + numerical_cols + label + group_id.
    - group_id идут подряд (инвариант loader.py).
    - scaler — ObservedStandardScaler, уже зафитченный на train.
    """
    def __init__(self, df: pd.DataFrame, feature_spec: FeatureSpec, scaler: StandardScaler):
        self.user_idx = torch.from_numpy(df["user_idx"].values.astype(np.int64))
        self.item_idx = torch.from_numpy(df["item_idx"].values.astype(np.int64))

        num_cols = feature_spec.numerical_cols
        if num_cols:
            num = df[num_cols].values.astype(np.float32)
            num = scaler.transform(num).astype(np.float32)
            self.num_feat = torch.from_numpy(num)
        else:
            # для feature_set='ids_only' — пустой тензор корректной формы
            self.num_feat = torch.zeros((len(df), 0), dtype=torch.float32)

        self.labels   = torch.from_numpy(df[feature_spec.target_col].values.astype(np.float32))
        self.group_id = torch.from_numpy(df[feature_spec.group_col].values.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_idx[idx], self.item_idx[idx], self.num_feat[idx],
            self.labels[idx],   self.group_id[idx],
        )


class GroupBatchSampler(torch.utils.data.Sampler):
    """
    Отдаёт батчи индексов так, чтобы каждый батч содержал целое число
    групп (query). Это нужно для listwise loss (softmax по группе) —
    хотим все кандидаты одной query в одном батче.

    Реальный размер батча:  groups_per_batch * (1 + n_neg).
    """
    def __init__(self, group_ids: np.ndarray, groups_per_batch: int,
                 shuffle: bool, seed: int = 0):
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        order = np.argsort(group_ids, kind="stable")
        sorted_groups = group_ids[order]
        _, first_idx = np.unique(sorted_groups, return_index=True)
        bounds = np.append(first_idx, len(order))
        self.group_slices: list[np.ndarray] = [
            order[bounds[i]:bounds[i + 1]] for i in range(len(first_idx))
        ]

    def __iter__(self):
        idxs = np.arange(len(self.group_slices))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for i in range(0, len(idxs), self.groups_per_batch):
            chunk = idxs[i:i + self.groups_per_batch]
            yield np.concatenate([self.group_slices[j] for j in chunk]).tolist()

    def __len__(self):
        return (len(self.group_slices) + self.groups_per_batch - 1) \
               // self.groups_per_batch


def group_softmax_loss(
    scores: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor,
) -> torch.Tensor:
    """
    Listwise: для каждой группы log-softmax по скорам, берём -log P(позитив),
    усредняем по группам.

    Предполагаем: в каждой группе ровно один label==1, остальные 0
    (инвариант loader.py).
    """
    unique = torch.unique(groups)
    losses = []
    for gid in unique:
        mask = groups == gid
        log_prob = F.log_softmax(scores[mask], dim=0)
        losses.append(-(log_prob * labels[mask]).sum())
    return torch.stack(losses).mean()
