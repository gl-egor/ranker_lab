"""
Общие утилиты для нейронных ранкеров (DCN-v2, DeepFM и следующих).

Содержит:
    - RankTableDataset      — DataFrame → тензоры
    - GroupBatchSampler     — батчи целыми группами (для listwise loss)
    - group_softmax_loss    — listwise log-softmax по группе
    - build_param_groups    — раздельные lr для эмбеддингов и плотных слоёв
    - build_scheduler       — фабрика lr-шедулеров (cosine / plateau / none)
    - train_neural_ranker   — общий training loop с early stopping и hard mining

Hard mining (v2):
    После warmup_epochs эпох пересэмплируем негативы в train_df на основе
    текущих скоров модели. Valid/Test остаются фиксированными — для честного
    сравнения с CatBoost.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from rlab.models.base import FeatureSpec


# ═════════════════════════════════════════════════════════════════════════════
# Dataset / Sampler / Loss
# ═════════════════════════════════════════════════════════════════════════════
class RankTableDataset(Dataset):
    """
    Оборачивает rank_table DataFrame в тензоры.
    Контракт: df имеет user_idx, item_idx, numerical_cols, label, group_id;
    группы идут подряд (инвариант loader.py); scaler уже зафитчен на train.
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
    Батчи индексов целыми группами. Реальный размер батча =
    groups_per_batch * (1 + n_neg).
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
    """Listwise: -log P(positive) под log-softmax по группе, среднее по группам."""
    unique = torch.unique(groups)
    losses = []
    for gid in unique:
        mask = groups == gid
        log_prob = F.log_softmax(scores[mask], dim=0)
        losses.append(-(log_prob * labels[mask]).sum())
    return torch.stack(losses).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Hard Negative Mining (v2: per-user, с пересборкой фичей)
# ═════════════════════════════════════════════════════════════════════════════
def _mine_hard_negatives_batched(
    model: nn.Module,
    forward_fn: Callable,
    device: str,
    user_indices: np.ndarray,   # (n_groups,)
    pos_items: np.ndarray,      # (n_groups,)
    item_pool: np.ndarray,
    n_hard: int,
    n_candidates: int,
    num_feat_dim: int,
    rng: np.random.Generator,
    scoring_batch: int = 65536,
) -> np.ndarray:
    """
    Per-user hard mining батчевым скорингом.

    Идея: для каждой группы (юзера) сэмплим n_candidates случайных items,
    скорим их моделью разом, берём top-n_hard.

    num_feat передаём как нули: нам нужен относительный порядок items
    для фиксированного юзера, а не абсолютный скор. Это дешёвый прокси,
    стандартный в литературе по hard mining. Вы не учите на этих скорах,
    только ранжируете кандидатов.

    Возвращает: (n_groups, n_hard) с item_idx.
    """
    n_groups = len(user_indices)

    # Случайные кандидаты — один np.choice вместо цикла
    candidates = rng.choice(
        item_pool, size=(n_groups, n_candidates), replace=True,
    )  # (n_groups, n_candidates)

    users_flat = np.repeat(user_indices, n_candidates)
    items_flat = candidates.reshape(-1)

    model.eval()
    scores = np.empty(len(users_flat), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(users_flat), scoring_batch):
            e = s + scoring_batch
            u = torch.from_numpy(users_flat[s:e]).long().to(device)
            i = torch.from_numpy(items_flat[s:e]).long().to(device)
            n = torch.zeros(e - s if e <= len(users_flat) else len(users_flat) - s,
                            num_feat_dim, device=device)
            scores[s:e] = forward_fn(model, u, i, n[:len(u)]).cpu().numpy()

    scores = scores.reshape(n_groups, n_candidates)

    # Маскируем позитив на случай, если он случайно попал в кандидаты
    pos_mask = candidates == pos_items[:, None]
    scores = np.where(pos_mask, -np.inf, scores)

    # argpartition — O(n) top-k вместо O(n log n) у argsort.
    # Для каждой строки берём индексы n_hard наибольших скоров.
    top_idx = np.argpartition(-scores, n_hard, axis=1)[:, :n_hard]
    return np.take_along_axis(candidates, top_idx, axis=1)


def resample_negatives_hard(
    *,
    train_df: pd.DataFrame,
    model: nn.Module,
    forward_fn: Callable,
    feature_spec: FeatureSpec,
    make_row_fn: Callable[[int, int, int, int], dict],
    n_neg: int,
    hard_ratio: float,
    device: str,
    seed: int,
    n_candidates: int = 200,
) -> pd.DataFrame:
    """
    Пересэмплинг негативов с per-user hard mining и пересчётом item-фичей.

    Параметры:
        train_df    : текущая rank_table (нужны только позитивы для каркаса).
        make_row_fn : (user_idx, item_idx, label, group_id) -> dict.
                      Обязательно должен корректно считать все item-фичи
                      (popularity, mean_rating) из train-агрегатов.
                      Предоставляется вызывающей стороной (dcnv2_ranker/deepfm_ranker),
                      обычно через functools.partial поверх features.make_feature_row.
        hard_ratio  : доля hard среди n_neg. Остальное — random для diversity.
        n_candidates: размер пула для hard mining на каждого юзера.
    """
    rng = np.random.default_rng(seed)

    # Каркас из позитивов — одна строка на группу
    pos_df = (
        train_df[train_df[feature_spec.target_col] == 1]
        [["user_idx", "item_idx", feature_spec.group_col]]
        .sort_values(feature_spec.group_col)
        .reset_index(drop=True)
    )
    user_idx_arr = pos_df["user_idx"].to_numpy()
    pos_item_arr = pos_df["item_idx"].to_numpy()
    group_id_arr = pos_df[feature_spec.group_col].to_numpy()

    item_pool = np.sort(train_df["item_idx"].unique())
    num_feat_dim = len(feature_spec.numerical_cols)

    n_hard = int(round(n_neg * hard_ratio))
    n_random = n_neg - n_hard

    # ── 1. Hard негативы per-user, батчевым скорингом ───────────────────
    if n_hard > 0:
        hard_negs = _mine_hard_negatives_batched(
            model=model, forward_fn=forward_fn, device=device,
            user_indices=user_idx_arr, pos_items=pos_item_arr,
            item_pool=item_pool,
            n_hard=n_hard, n_candidates=n_candidates,
            num_feat_dim=num_feat_dim, rng=rng,
        )  # (n_groups, n_hard)
    else:
        hard_negs = np.empty((len(pos_df), 0), dtype=np.int64)

    # ── 2. Random негативы батчем с фильтрацией forbidden ───────────────
    # Для каждой группы forbidden = {pos} ∪ hard_negs[group].
    # Берём пул с запасом и отбрасываем коллизии.
    random_negs = np.empty((len(pos_df), n_random), dtype=np.int64)
    if n_random > 0:
        over_sample = max(n_random * 4, 16)
        raw = rng.choice(item_pool, size=(len(pos_df), over_sample), replace=True)
        for g in range(len(pos_df)):
            forbidden = {int(pos_item_arr[g])}
            forbidden.update(int(x) for x in hard_negs[g])
            chosen: list[int] = []
            for c in raw[g]:
                c_int = int(c)
                if c_int in forbidden:
                    continue
                forbidden.add(c_int)  # и от дублей внутри группы
                chosen.append(c_int)
                if len(chosen) == n_random:
                    break
            # страховка для микро-пулов
            while len(chosen) < n_random:
                c_int = int(rng.choice(item_pool))
                if c_int not in forbidden:
                    forbidden.add(c_int)
                    chosen.append(c_int)
            random_negs[g] = chosen

    # ── 3. Пересборка rank_table через make_row_fn ──────────────────────
    # Именно здесь пересчитываются item-зависимые фичи (popularity и т.д.).
    rows: list[dict] = []
    for g in range(len(pos_df)):
        u = int(user_idx_arr[g])
        gid = int(group_id_arr[g])
        rows.append(make_row_fn(u, int(pos_item_arr[g]), 1, gid))
        for it in hard_negs[g]:
            rows.append(make_row_fn(u, int(it), 0, gid))
        for it in random_negs[g]:
            rows.append(make_row_fn(u, int(it), 0, gid))

    out = pd.DataFrame(rows)
    # Сохраняем инвариант loader.py: группы идут подряд
    out = out.sort_values([feature_spec.group_col, feature_spec.target_col],
                          ascending=[True, False]).reset_index(drop=True)
    return out
# ═════════════════════════════════════════════════════════════════════════════
# Optimizer / Scheduler factories
# ═════════════════════════════════════════════════════════════════════════════
def build_param_groups(
    model: nn.Module,
    lr: float,
    emb_lr_mult: float = 1.0,
    emb_weight_decay: float | None = None,
    weight_decay: float = 1e-5,
) -> list[dict[str, Any]]:
    """
    Делит параметры модели на группы для оптимизатора.

    Группа 'emb' — все nn.Embedding слои (распознаём по 'emb' в имени).
    Группа 'dense' — всё остальное (Linear, LayerNorm, биасы и т.д.).

    Параметры:
        lr               : базовый lr (для dense-группы).
        emb_lr_mult      : множитель lr для эмбеддингов.
                           1.0  → одинаковый lr (выключено).
                           0.1  → эмбеддинги учатся в 10 раз медленнее.
        emb_weight_decay : отдельный wd для эмбеддингов. None = тот же,
                           что и для dense. Полезно ставить БОЛЬШЕ для emb
                           (например, 1e-2), чтобы давить редкие
                           id-эмбеддинги к нулю.
        weight_decay     : wd для dense-группы.
    """
    emb_params, dense_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "emb" in name:
            emb_params.append(param)
        else:
            dense_params.append(param)

    if emb_weight_decay is None:
        emb_weight_decay = weight_decay

    return [
        {"params": emb_params,
         "lr": lr * emb_lr_mult,
         "weight_decay": emb_weight_decay},
        {"params": dense_params,
         "lr": lr,
         "weight_decay": weight_decay},
    ]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    kind: str,
    n_epochs: int,
    n_steps_per_epoch: int = 1,
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """
    Фабрика lr-шедулеров.

    kind:
        'none'    → None, lr константный.
        'cosine'  → CosineAnnealingLR до eta_min.
        'plateau' → ReduceLROnPlateau по val NDCG.
        'warmup_cosine' → линейный warmup + cosine decay.
    """
    if kind == "none":
        return None

    if kind == "cosine":
        total_steps = n_epochs * n_steps_per_epoch
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get("eta_min", 1e-6),
        )

    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 1),
        )

    if kind == "warmup_cosine":
        warmup_epochs = kwargs.get("warmup_epochs", 1)
        total_steps   = n_epochs * n_steps_per_epoch
        warmup_steps  = warmup_epochs * n_steps_per_epoch
        eta_min       = kwargs.get("eta_min", 1e-6)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown scheduler kind '{kind}'.")


# ═════════════════════════════════════════════════════════════════════════════
# Универсальный training loop с hard mining
# ═════════════════════════════════════════════════════════════════════════════
def train_neural_ranker(
    *,
    model: nn.Module,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_spec: FeatureSpec,
    scaler: StandardScaler,
    params: dict[str, Any],
    seed: int,
    device: str,
    forward_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    make_row_fn: Callable | None = None, #for hard mining
) -> dict[str, Any]:
    """
    Универсальный training loop для нейронных ранкеров.

    Hard mining (новое):
        params["hard_mining"]         : bool, включить ли (default: False)
        params["hard_mining_warmup"]  : int, эпох warmup (default: 2)
        params["hard_ratio"]          : float, доля hard негативов (default: 0.5)
    
    После warmup эпох:
        1. Вычисляем скоры модели для всех items
        2. Пересэмплируем негативы в train_df (hard + random mix)
        3. Пересоздаём DataLoader
    
    Valid остаётся фиксированным — для честного сравнения с CatBoost.

    Возвращает:
        train_history, train_time_sec, best_val_ndcg, best_state_dict
    """
    torch.manual_seed(seed)

    # Hard mining params
    use_hard_mining = params.get("hard_mining", False)
    hard_mining_warmup = params.get("hard_mining_warmup", 2)
    hard_ratio = params.get("hard_ratio", 0.5)
    n_candidates = params.get("hard_mining_n_candidates", 200)

    if use_hard_mining and make_row_fn is None:
        raise ValueError(
            "hard_mining=True требует make_row_fn для корректного "
            "пересчёта item-фичей. Передайте callback из features.py."
        )
    
    # Определяем n_neg из train_df
    group_sizes = train_df.groupby("group_id").size()
    n_neg = int(group_sizes.iloc[0]) - 1  # размер группы - 1 позитив
    
    # n_items для compute_item_scores
    n_items = feature_spec.cardinalities.get("item_idx", train_df["item_idx"].max() + 1)

    # ── Создаём DataLoaders ─────────────────────────────────────────────
    def _make_loader(df: pd.DataFrame, shuffle: bool, loader_seed: int) -> DataLoader:
        ds = RankTableDataset(df, feature_spec, scaler)
        return DataLoader(
            ds,
            batch_sampler=GroupBatchSampler(
                df[feature_spec.group_col].values,
                groups_per_batch=params["groups_per_batch"],
                shuffle=shuffle, seed=loader_seed,
            ),
            num_workers=0, pin_memory=(device == "cuda"),
        )

    # Train loader будет пересоздаваться при hard mining
    current_train_df = train_df.copy()
    train_loader = _make_loader(current_train_df, shuffle=True, loader_seed=seed)
    
    # Valid loader фиксирован
    valid_loader = _make_loader(valid_df, shuffle=False, loader_seed=seed)

    # ── Optimizer ───────────────────────────────────────────────────────
    param_groups = build_param_groups(
        model,
        lr=params["lr"],
        emb_lr_mult=params.get("emb_lr_mult", 1.0),
        emb_weight_decay=params.get("emb_weight_decay", None),
        weight_decay=params.get("weight_decay", 1e-5),
    )
    opt = torch.optim.AdamW(param_groups)

    # ── Scheduler ───────────────────────────────────────────────────────
    sched_kind = params.get("scheduler", "none")
    sched_kwargs = params.get("scheduler_kwargs", {})
    scheduler = build_scheduler(
        opt, kind=sched_kind,
        n_epochs=params["max_epochs"],
        n_steps_per_epoch=len(train_loader),
        **sched_kwargs,
    )
    step_each_batch = sched_kind in ("cosine", "warmup_cosine")

    # ── Early stopping ──────────────────────────────────────────────────
    best_ndcg = -1.0
    best_state: dict | None = None
    patience = 0
    history: list[dict] = []

    from rlab.eval.metrics import ranking_metrics

    if forward_fn is None:
        forward_fn = lambda m, u, i, n: m(u, i, n)

    t0 = time.time()
    for epoch in trange(1, params["max_epochs"] + 1, desc="epoch", unit="ep"):
        
        # ── Hard mining: пересэмплируем негативы после warmup ───────────
        if use_hard_mining and epoch > hard_mining_warmup:
            print(f"  [hard mining] epoch {epoch}: пересэмплинг негативов...")
            current_train_df = resample_negatives_hard(
                train_df=train_df,             # исходный (с warm-start позитивами)
                model=model,
                forward_fn=forward_fn,
                feature_spec=feature_spec,
                make_row_fn=make_row_fn,
                n_neg=n_neg,
                hard_ratio=hard_ratio,
                device=device,
                seed=seed + epoch,
                n_candidates=n_candidates,
            )
            train_loader = _make_loader(current_train_df, shuffle=True,
                                        loader_seed=seed + epoch)

        # ── Train ───────────────────────────────────────────────────────
        model.train()
        losses = []
        for u, i, n, y, g in tqdm(train_loader, desc=f"  train e{epoch}",
                                   leave=False, unit="batch"):
            u, i, n = u.to(device), i.to(device), n.to(device)
            y, g    = y.to(device), g.to(device)

            scores = forward_fn(model, u, i, n)
            loss = group_softmax_loss(scores, y, g)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"])
            opt.step()
            if scheduler is not None and step_each_batch:
                scheduler.step()
            losses.append(loss.item())

        # ── Eval ────────────────────────────────────────────────────────
        val_scores, val_labels, val_groups = _predict_loader(
            model, valid_loader, device, forward_fn
        )
        metrics = ranking_metrics(val_scores, val_labels, val_groups, k=10)

        cur_lrs = [g["lr"] for g in opt.param_groups]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_NDCG@10": metrics["NDCG"],
            "val_HR@10":   metrics["HR"],
            "lr_emb":   cur_lrs[0],
            "lr_dense": cur_lrs[1] if len(cur_lrs) > 1 else cur_lrs[0],
            "hard_mining_active": use_hard_mining and epoch > hard_mining_warmup,
        })
        
        hm_status = " [HM]" if (use_hard_mining and epoch > hard_mining_warmup) else ""
        print(f"  [e{epoch}]{hm_status} loss={np.mean(losses):.4f} "
              f"val NDCG@10={metrics['NDCG']:.4f} "
              f"lr_emb={cur_lrs[0]:.2e}")

        if scheduler is not None and not step_each_batch:
            scheduler.step(metrics["NDCG"])

        # Early stopping
        if metrics["NDCG"] > best_ndcg:
            best_ndcg = metrics["NDCG"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= params["patience"]:
                print(f"  early stop at epoch {epoch} "
                      f"(best NDCG@10={best_ndcg:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_time_sec": time.time() - t0,
        "best_val_ndcg":  best_ndcg,
        "train_history":  history,
    }


def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    forward_fn: Callable,
):
    """Прогон по DataLoader, возвращает (scores, labels, groups) numpy."""
    model.eval()
    S, L, G = [], [], []
    with torch.no_grad():
        for u, i, n, y, g in loader:
            u, i, n = u.to(device), i.to(device), n.to(device)
            S.append(forward_fn(model, u, i, n).cpu().numpy())
            L.append(y.numpy()); G.append(g.numpy())
    return np.concatenate(S), np.concatenate(L), np.concatenate(G)


# ═════════════════════════════════════════════════════════════════════════════
# Helper: предсказание на произвольном DataFrame
# ═════════════════════════════════════════════════════════════════════════════
def predict_dataframe(
    model: nn.Module,
    df: pd.DataFrame,
    feature_spec: FeatureSpec,
    scaler: StandardScaler,
    device: str,
    forward_fn: Callable | None = None,
    batch_size: int = 8192,
) -> np.ndarray:
    """Предсказания на тесте/любом DataFrame."""
    if forward_fn is None:
        forward_fn = lambda m, u, i, n: m(u, i, n)

    ds = RankTableDataset(df, feature_spec, scaler)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    out = []
    with torch.no_grad():
        for u, i, n, _, _ in loader:
            u, i, n = u.to(device), i.to(device), n.to(device)
            out.append(forward_fn(model, u, i, n).cpu().numpy())
    return np.concatenate(out).astype(np.float32)