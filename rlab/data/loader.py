"""
Загрузка датасета и подготовка train/valid/test rank_table.

Основной pipeline (load_dataset):
    raw interactions
      → idx mapping (user_id → user_idx, item_id → item_idx)
      → leave-last-two-out split по времени per user
      → train aggregates (item_popularity, user_mean_rating, ...)
      → build_rank_table(train/valid/test)    ← позитив + N негативов
      → apply feature_set                     ← для H3
      → (train, valid, test, FeatureSpec)

Кеширование:
    Ключ кеша = hash релевантных полей DataConfig (dataset, размеры,
    neg_strategy, n_neg_*, seed). feature_set НЕ входит в ключ кеша —
    он применяется как post-filter колонок, а сама rank_table общая.
    Это экономит 80% времени при sweep'е по feature_set.

Добавление датасета:
    Реализовать загрузчик, возвращающий DataFrame с колонками
    (user_id, item_id, rating, timestamp), и зарегистрировать в
    RAW_READERS. Всё остальное универсально.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rlab.configs import DataConfig
from rlab.data.features import (
    FEATURE_GROUPS,
    TrainAggregates,
    categorical_cols_in,
    compute_train_aggregates,
    METADATA_COLS,
    make_feature_row,
    numerical_cols_in,
    resolve_feature_set,
)
from rlab.data.negatives import (
    build_popularity_probs,
    get_sampler,
)
from rlab.models.base import FeatureSpec


# ═════════════════════════════════════════════════════════════════════════════
# 1. Реестр сырых загрузчиков (добавление датасета = добавление функции сюда)
# ═════════════════════════════════════════════════════════════════════════════
RAW_READERS: dict[str, callable] = {}


def register_raw_reader(name: str):
    """Декоратор для регистрации. Читатель возвращает DataFrame с
    колонками user_id, item_id, rating, timestamp."""
    def _d(fn):
        RAW_READERS[name] = fn
        return fn
    return _d


@register_raw_reader("books5")
def _read_books5(raw_dir: str) -> pd.DataFrame:
    """
    Amazon Books 5-core. Ожидаем файл raw_dir/books5_interactions.parquet
    с колонками user_id, item_id, rating, timestamp.

    Если у тебя другой формат (например, CSV с другими колонками) —
    подправь здесь одно место, всё остальное универсально.
    """
    path = os.path.join(raw_dir, "books5_interactions.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Books5 raw data not found at {path}. "
            f"Положи файл с колонками user_id, item_id, rating, timestamp."
        )
    df = pd.read_parquet(path)
    expected = {"user_id", "item_id", "rating", "timestamp"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


@register_raw_reader("ml1m")
def _read_ml1m(raw_dir: str) -> pd.DataFrame:
    """
    MovieLens-1M. Ожидаем ratings.dat в формате:
        UserID::MovieID::Rating::Timestamp
    Это оригинальный формат со страницы GroupLens — скачай и распакуй
    как есть, не конвертируя:
        wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
        unzip ml-1m.zip

    Ищем файл по двум стандартным путям: raw_dir/ml-1m/ratings.dat
    или raw_dir/ratings.dat (если распаковал flat).
    """
    candidates = [
        os.path.join(raw_dir, "ml-1m", "ratings.dat"),
        os.path.join(raw_dir, "ratings.dat"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"MovieLens-1M ratings.dat не найден. Искал: {candidates}. "
            f"Скачай ml-1m.zip с GroupLens и распакуй в {raw_dir}."
        )

    # Формат: '1::1193::5::978300760\n' — ID как строки (для консистентности
    # с books5, где они тоже строковые после idx mapping).
    df = pd.read_csv(
        path, sep="::", engine="python", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str,
               "rating": np.float32, "timestamp": np.int64},
    )
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. Вспомогательные: idx mapping, сплит, истории
# ═════════════════════════════════════════════════════════════════════════════
def _build_idx_mapping(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """
    Переводим строковые user_id/item_id в плотные int-индексы.
    0 резервируем под padding/unknown → индексы с 1.
    """
    user_ids = pd.Categorical(df["user_id"])
    item_ids = pd.Categorical(df["item_id"])
    df = df.copy()
    df["user_idx"] = user_ids.codes.astype(np.int64) + 1
    df["item_idx"] = item_ids.codes.astype(np.int64) + 1
    user2idx = {u: i + 1 for i, u in enumerate(user_ids.categories)}
    item2idx = {i: j + 1 for j, i in enumerate(item_ids.categories)}
    return df, user2idx, item2idx


def _split_last_two_out(
    df: pd.DataFrame,
    min_history: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leave-last-two-out по юзеру:
      последнее по времени  → test (1 строка),
      предпоследнее         → valid,
      остальное             → train.
    Юзеры с <3 взаимодействиями выкидываются (не хватает на все три сплита).
    """
    df = df.sort_values(["user_idx", "timestamp"], kind="stable").reset_index(drop=True)

    # для каждой записи — её обратный rank в истории юзера
    df["rev_rank"] = df.groupby("user_idx").cumcount(ascending=False)
    # число записей у юзера
    user_counts = df.groupby("user_idx").size()
    keep_users = set(user_counts[user_counts >= min_history].index)
    df = df[df["user_idx"].isin(keep_users)].reset_index(drop=True)

    test  = df[df["rev_rank"] == 0].drop(columns=["rev_rank"]).reset_index(drop=True)
    valid = df[df["rev_rank"] == 1].drop(columns=["rev_rank"]).reset_index(drop=True)
    train = df[df["rev_rank"] >= 2].drop(columns=["rev_rank"]).reset_index(drop=True)
    return train, valid, test


def _build_histories(
    train_interactions: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Для каждой query-записи присоединяем list of item_idx из train-истории
    данного юзера. Если юзера нет в train — пустой список.
    """
    hist_map = (
        train_interactions.sort_values(["user_idx", "timestamp"])
                          .groupby("user_idx")["item_idx"]
                          .apply(lambda s: s.astype(int).tolist())
                          .to_dict()
    )
    query_df = query_df.copy()
    query_df["history_items"] = query_df["user_idx"].map(
        lambda u: hist_map.get(int(u), [])
    )
    return query_df


# ═════════════════════════════════════════════════════════════════════════════
# 3. Сборка rank_table
# ═════════════════════════════════════════════════════════════════════════════
def _build_rank_table(
    query_df: pd.DataFrame,
    agg: TrainAggregates,
    item_pool: np.ndarray,
    n_neg: int,
    neg_strategy: str,
    seed: int,
    desc: str = "rank_table",
) -> pd.DataFrame:
    """
    Для каждой query (строка query_df с user_idx, item_idx, history_items)
    генерим 1 позитив + n_neg негативов. Итог — длинный DataFrame,
    отсортированный по (group_id, label DESC).

    Порядок строк важен:
      - CatBoost требует, чтобы группы шли подряд;
      - metrics.py группирует по group_id, порядок внутри неважен;
      - нейронки нормально работают с любым разумным порядком.
    """
    rng = np.random.default_rng(seed)
    sampler = get_sampler(neg_strategy)

    # для popularity-сэмплера нужны веса
    sampler_kwargs: dict = {}
    if neg_strategy == "popularity":
        sampler_kwargs["pop_probs"] = build_popularity_probs(
            agg.item_popularity, item_pool
        )

    rows: list[dict] = []
    for gid, row in enumerate(
        tqdm(query_df.itertuples(index=False),
             total=len(query_df), desc=desc, unit="q")
    ):
        user_idx = int(row.user_idx)
        pos_item = int(row.item_idx)
        history = list(row.history_items) if row.history_items is not None else []
        hist_counter: dict[int, int] = {}
        for x in history:
            hist_counter[int(x)] = hist_counter.get(int(x), 0) + 1

        # позитив
        rows.append(make_feature_row(
            user_idx=user_idx, item_idx=pos_item,
            history_items=history, history_counter=hist_counter,
            agg=agg, label=1, group_id=gid,
        ))

        # негативы
        forbidden = set(history) | {pos_item}
        negs = sampler(
            pos_item=pos_item, forbidden=forbidden,
            item_pool=item_pool, n_neg=n_neg, rng=rng,
            **sampler_kwargs,
        )
        for neg in negs:
            rows.append(make_feature_row(
                user_idx=user_idx, item_idx=int(neg),
                history_items=history, history_counter=hist_counter,
                agg=agg, label=0, group_id=gid,
            ))

    out = pd.DataFrame(rows)
    out = out.sort_values(["group_id", "label"], ascending=[True, False]) \
             .reset_index(drop=True)
    out["group_id"] = out["group_id"].astype("int32")
    out["label"]    = out["label"].astype("int8")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Кеш
# ═════════════════════════════════════════════════════════════════════════════
def _cache_key(data_cfg: DataConfig, seed: int) -> str:
    """
    Ключ кеша содержит всё, что влияет на rank_table, КРОМЕ feature_set
    (он применяется post-factum выбором колонок).
    """
    payload = {
        "dataset":       data_cfg.dataset,
        "train_size":    data_cfg.train_size,
        "valid_size":    data_cfg.valid_size,
        "test_size":     data_cfg.test_size,
        "n_neg_train":   data_cfg.n_neg_train,
        "n_neg_eval":    data_cfg.n_neg_eval,
        "neg_strategy":  data_cfg.neg_strategy,
        "seed":          seed,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def _cache_paths(cache_dir: str, key: str) -> dict[str, str]:
    base = Path(cache_dir) / f"{key}"
    base.mkdir(parents=True, exist_ok=True)
    return {
        "train": str(base / "train.parquet"),
        "valid": str(base / "valid.parquet"),
        "test":  str(base / "test.parquet"),
        "meta":  str(base / "meta.json"),
        "agg":   str(base / "agg.pkl"), #Для hard mining
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. Главная функция
# ═════════════════════════════════════════════════════════════════════════════
def load_dataset(
    data_cfg: DataConfig,
    cache_dir: str,
    seed: int,
    raw_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureSpec]:
    """
    Главная точка входа. Возвращает (train_df, valid_df, test_df, FeatureSpec).

    raw_dir: откуда читать сырые данные. Если None — ищем в переменной
             окружения RLAB_RAW_DIR, иначе в ./raw.
    """
    if raw_dir is None:
        raw_dir = os.environ.get("RLAB_RAW_DIR", "./raw")

    key = _cache_key(data_cfg, seed)
    paths = _cache_paths(cache_dir, key)

    # ── Кеш-hit ──────────────────────────────────────────────────────────
    cache_files = ("train", "valid", "test", "meta", "agg")
    cache_hit = all(os.path.exists(paths[k]) for k in cache_files)

    if cache_hit:
        print(f"[loader] cache HIT: {key}")
        train = pd.read_parquet(paths["train"])
        valid = pd.read_parquet(paths["valid"])
        test  = pd.read_parquet(paths["test"])
        with open(paths["meta"]) as f:
            meta = json.load(f)
        with open(paths["agg"], "rb") as f:
            agg: TrainAggregates = pickle.load(f)
    else:
        print(f"[loader] cache MISS: {key} — building from scratch")
        train, valid, test, meta, agg = _build_from_scratch(data_cfg, seed, raw_dir)
        train.to_parquet(paths["train"], index=False)
        valid.to_parquet(paths["valid"], index=False)
        test.to_parquet(paths["test"],  index=False)
        with open(paths["meta"], "w") as f:
            json.dump(meta, f)
        with open(paths["agg"], "wb") as f:
            pickle.dump(agg, f)

    # ── Apply feature_set (пост-фильтр колонок) ──────────────────────────
    # METADATA_COLS остаются для сохранения предсказаний при любом feature_set.
    feature_cols = resolve_feature_set(data_cfg.feature_set)
    keep_cols = list(dict.fromkeys(
        ["group_id", "label"] + METADATA_COLS + feature_cols
    ))
    train = train[keep_cols].copy()
    valid = valid[keep_cols].copy()
    test  = test[keep_cols].copy()

    spec = FeatureSpec(
        target_col="label",
        group_col="group_id",
        categorical_cols=categorical_cols_in(feature_cols),
        numerical_cols=numerical_cols_in(feature_cols),
        cardinalities={
            "user_idx": int(meta["n_users"]) + 1,   # +1 под 0=padding
            "item_idx": int(meta["n_items"]) + 1,
        },
        train_aggregates=agg, 
    )
    return train, valid, test, spec


def _build_from_scratch(
    data_cfg: DataConfig,
    seed: int,
    raw_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Вся нерезидентная работа: чтение, idx mapping, сплит, агрегаты,
    rank_table. Вынесено отдельно, чтобы load_dataset остался линейным.
    """
    if data_cfg.dataset not in RAW_READERS:
        raise KeyError(
            f"Unknown dataset '{data_cfg.dataset}'. "
            f"Registered: {list(RAW_READERS)}"
        )

    # 1. raw → idx
    raw = RAW_READERS[data_cfg.dataset](raw_dir)
    raw, _, _ = _build_idx_mapping(raw)
    n_users = int(raw["user_idx"].max())
    n_items = int(raw["item_idx"].max())

    # 2. split
    train_int, valid_q, test_q = _split_last_two_out(raw)
    print(f"[loader] split: train={len(train_int)} valid={len(valid_q)} "
          f"test={len(test_q)}")

    # 3. aggregates (ONLY from train)
    agg = compute_train_aggregates(train_int)

    # 4. subsample queries по target размерам
    rng = np.random.default_rng(seed)

    def _subsample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
        if n is None or n >= len(df):
            return df.reset_index(drop=True)
        idx = rng.choice(len(df), size=n, replace=False)
        return df.iloc[sorted(idx)].reset_index(drop=True)

    # train queries: берём позитивные взаимодействия train'а как queries,
    # но генерим по ним rank_table. На практике N_train взаимодействий
    # намного больше, чем нам нужно query'ей.
    train_q = _subsample(train_int, data_cfg.train_size)
    valid_q = _subsample(valid_q,   data_cfg.valid_size)
    test_q  = _subsample(test_q,    data_cfg.test_size)

    # 5. histories (на основе полного train_int, не сабсемплированного)
    train_q = _build_histories(train_int, train_q)
    valid_q = _build_histories(train_int, valid_q)
    test_q  = _build_histories(train_int, test_q)

    # 6. rank_table для каждого сплита
    item_pool = np.asarray(sorted(agg.item_popularity.keys()), dtype=np.int64)

    train_rt = _build_rank_table(
        train_q, agg, item_pool,
        n_neg=data_cfg.n_neg_train, neg_strategy=data_cfg.neg_strategy,
        seed=seed, desc="train",
    )
    valid_rt = _build_rank_table(
        valid_q, agg, item_pool,
        n_neg=data_cfg.n_neg_eval, neg_strategy=data_cfg.neg_strategy,
        seed=seed + 1, desc="valid",
    )
    test_rt = _build_rank_table(
        test_q, agg, item_pool,
        n_neg=data_cfg.n_neg_eval, neg_strategy=data_cfg.neg_strategy,
        seed=seed + 2, desc="test",
    )

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_train_queries": len(train_q),
        "n_valid_queries": len(valid_q),
        "n_test_queries":  len(test_q),
        "neg_strategy":    data_cfg.neg_strategy,
    }
    return train_rt, valid_rt, test_rt, meta, agg
