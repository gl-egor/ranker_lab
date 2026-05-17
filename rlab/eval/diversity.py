"""Diversity-метрики: Coverage@k и EPC@k по top-K рекомендациям."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_top_k_recs(df_preds: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Топ-K кандидатов на группу по убыванию pred."""
    return (
        df_preds.sort_values(["group_id", "pred"], ascending=[True, False])
        .groupby("group_id", sort=True)
        .head(k)
    )


def calculate_coverage(top_k_df: pd.DataFrame, total_catalog_items: set[int]) -> float:
    """Доля уникальных айтемов в выдаче от всего каталога."""
    if not total_catalog_items:
        return 0.0
    recommended = set(top_k_df["item_idx"].unique())
    return len(recommended) / len(total_catalog_items)


def calculate_epc(
    top_k_df: pd.DataFrame,
    train_item_popularity: dict[int, int],
) -> float:
    """
    Expected Popularity Complement: дисконтированная «новизна» по позициям в топе.
    train_item_popularity: {item_idx: число взаимодействий в train}.
    """
    if top_k_df.empty:
        return 0.0

    df = top_k_df.copy()
    df["rank"] = df.groupby("group_id").cumcount() + 1
    df["item_pop"] = df["item_idx"].map(train_item_popularity).fillna(0)

    max_pop = max(train_item_popularity.values()) if train_item_popularity else 1
    df["novelty_score"] = 1.0 - (
        np.log2(df["item_pop"] + 1) / np.log2(max_pop + 1)
    )
    df["discounted_novelty"] = df["novelty_score"] / np.log2(df["rank"] + 1)

    return float(df.groupby("group_id")["discounted_novelty"].sum().mean())


def diversity_metrics(
    df_preds: pd.DataFrame,
    k: int,
    catalog_items: set[int],
    train_item_popularity: dict[int, int],
) -> dict[str, float]:
    """Coverage@k и EPC@k на top-K из кандидатов теста."""
    top_k = get_top_k_recs(df_preds, k=k)
    return {
        "coverage_at_k": calculate_coverage(top_k, catalog_items),
        "epc_at_k": calculate_epc(top_k, train_item_popularity),
    }
