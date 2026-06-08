"""
Глава 4: Popularity-aware training & IPS-reweighted loss.

Эксперименты:
  4.1  Ablation study — что именно помогает DCNv2 на хвосте
  4.2  IPS-reweighted loss — новизна диплома
  4.3  FinalMLP (fixed) vs DCNv2/CatBoost/DeepFM на 10k/60k/300k

Запуск в Colab:
  1. Убедись, что обновлённые файлы finalmlp_ranker.py и _torch_utils.py
     загружены на Drive (или обновлены в git).
  2. Скопируй содержимое этого файла в ячейки Colab.
  3. Запускай ячейки последовательно.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1: Setup
# ══════════════════════════════════════════════════════════════════════════════

import sys, os

from google.colab import drive          # type: ignore
drive.mount("/content/drive")

RLAB_ROOT  = "/content/drive/MyDrive/ranker_lab"
DATA_ROOT  = "/content/drive/MyDrive/ranker_lab_data"
RESULTS    = f"{DATA_ROOT}/results"
CACHE      = f"{DATA_ROOT}/cache"

sys.path.insert(0, RLAB_ROOT)
os.environ["RLAB_RAW_DIR"] = f"{DATA_ROOT}/raw"

from rlab.configs import ExperimentConfig, DataConfig, ModelConfig, EvalConfig
from rlab.sweep import SweepConfig, run_sweep
from rlab.runner import run_experiment

# ══════════════════════════════════════════════════════════════════════════════
# Cell 2: Shared eval config & base params
# ══════════════════════════════════════════════════════════════════════════════

EVAL_CFG = EvalConfig(
    k=10,
    stratify_by_pop=True,
    n_pop_bins=4,
    bootstrap_n=1000,
    eval_warm_only=True,
    save_predictions=True,
    measure_inference_latency=True,
)

# --- DCNv2 base (совпадает с твоей успешной базой, emb_dim=64) ---------------
DCN_BASE_300K = {
    'emb_dim': 64,
    'dropout': 0.2,
    'layer_norm': True,
    'lr': 5e-4,
    'emb_lr_mult': 3.0,
    'emb_weight_decay': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'max_epochs': 12,
    'patience': 3,
    'groups_per_batch': 256,
    'scheduler': "warmup_cosine",
    'scheduler_kwargs': {"warmup_epochs": 2, "eta_min": 1e-5},
    'n_cross': 2,
    'mlp_dims': (512, 256, 128),
}

# --- FinalMLP 300k: emb_dim=32, умеренные MLP, сильнее emb WD ---------------
# emb_dim=16 на 60k давало val~0.3 и переобучение — мало ёмкости при больших MLP.
# emb_dim=64 на 300k → 81M params и ранний early stop. Компромисс: 32.
FMLP_BASE_300K = {
    'emb_dim': 32,
    'dropout': 0.22,
    'layer_norm': True,
    'lr': 5e-4,
    'emb_lr_mult': 2.5,
    'emb_weight_decay': 2e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'max_epochs': 15,
    'patience': 4,
    'groups_per_batch': 256,
    'scheduler': "warmup_cosine",
    'scheduler_kwargs': {"warmup_epochs": 2, "eta_min": 1e-5},
    'mlp1_hidden': (384, 192, 96),
    'mlp2_hidden': (192, 96),
    'num_heads': 4,
    'gate_reduction': 4,
    'pop_weighted_sampler': True,
    'pop_reg_alpha': 0.01,
}

# Быстрый вариант: popularity-негативы при сборке датасета, без runtime resample
FMLP_BASE_300K_FAST = {
    **FMLP_BASE_300K,
    'pop_weighted_sampler': False,
    'pop_reg_alpha': 0.01,
}

FMLP_BASE_60K = {
    'emb_dim': 32,
    'dropout': 0.28,
    'layer_norm': True,
    'lr': 7e-4,
    'emb_lr_mult': 2.0,
    'emb_weight_decay': 5e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'max_epochs': 18,
    'patience': 4,
    'groups_per_batch': 128,
    'scheduler': "warmup_cosine",
    'scheduler_kwargs': {"warmup_epochs": 2, "eta_min": 1e-5},
    'mlp1_hidden': (256, 128),
    'mlp2_hidden': (128, 64),
    'num_heads': 4,
    'gate_reduction': 4,
    'pop_weighted_sampler': True,
    'pop_reg_alpha': 0.01,
}

# DataConfig для 300k — два режима пересэмплирования
DATA_CFG_300K = DataConfig(
    dataset="books5",
    train_size=300_000,
    valid_size=10_000,
    test_size=10_000,
    feature_set="no_hc_features",
    n_neg_train=5,
    n_neg_eval=50,
    neg_strategy="random",          # runtime pop resample в train_neural_ranker
)

DATA_CFG_300K_FAST = DataConfig(
    dataset="books5",
    train_size=300_000,
    valid_size=10_000,
    test_size=10_000,
    feature_set="no_hc_features",
    n_neg_train=5,
    n_neg_eval=50,
    neg_strategy="popularity",
)

FMLP_BASE_10K = {
    'emb_dim': 32,
    'dropout': 0.3,
    'layer_norm': True,
    'lr': 1e-3,
    'emb_lr_mult': 2.0,
    'emb_weight_decay': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'max_epochs': 25,
    'patience': 5,
    'groups_per_batch': 64,
    'scheduler': "warmup_cosine",
    'scheduler_kwargs': {"warmup_epochs": 3, "eta_min": 1e-5},
    'mlp1_hidden': (256, 128),
    'mlp2_hidden': (128, 64),
    'num_heads': 4,
    'gate_reduction': 4,
    'pop_weighted_sampler': True,
    'pop_reg_alpha': 0.01,
}


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3: §4.1 — Ablation study: что помогает DCNv2 на tail
# ══════════════════════════════════════════════════════════════════════════════
#
# Все эксперименты используют model.kind="dcnv2_reg" (он всегда создаёт
# make_row_fn), но мы переключаем pop_weighted_sampler и pop_reg_alpha.
#
# Сравниваем 6 конфигураций:
#   1. DCNv2 vanilla          — ни sampler, ни reg
#   2. DCNv2 + pop sampler    — только пересэмплинг негативов
#   3. DCNv2 + pop reg        — только регуляризатор в loss
#   4. DCNv2 full (pop+reg)   — обе техники вместе
#   5. FinalMLP vanilla       — без pop-aware training
#   6. FinalMLP full (pop+reg)— с обоими компонентами
#
# → Разница (4)-(1) = суммарный эффект popularity-aware training
# → Разница (4)-(5)/(6) = вклад архитектуры Cross Network vs FinalMLP

def run_ablation_study():
    """§4.1: Ablation study на train_size=300k, no_hc_features."""

    data_cfg = DataConfig(
        dataset="books5",
        train_size=300_000,
        valid_size=10_000,
        test_size=10_000,
        feature_set="no_hc_features",
        n_neg_train=5,
        n_neg_eval=50,
    )

    ablation_configs = [
        # --- DCNv2 варианты ---
        ("H10_ablation_dcn_vanilla", "dcnv2_reg", {
            **DCN_BASE_300K,
            'pop_weighted_sampler': False,
            'pop_reg_alpha': 0.0,
        }),
        ("H10_ablation_dcn_pop_sampler", "dcnv2_reg", {
            **DCN_BASE_300K,
            'pop_weighted_sampler': True,
            'pop_resample_each_epoch': False,
            'pop_reg_alpha': 0.0,
        }),
        ("H10_ablation_dcn_pop_reg", "dcnv2_reg", {
            **DCN_BASE_300K,
            'pop_weighted_sampler': False,
            'pop_reg_alpha': 0.01,
        }),
        ("H10_ablation_dcn_full", "dcnv2_reg", {
            **DCN_BASE_300K,
            'pop_weighted_sampler': True,
            'pop_resample_each_epoch': False,
            'pop_reg_alpha': 0.01,
        }),
        # --- FinalMLP варианты ---
        ("H10_ablation_fmlp_vanilla", "finalmlp", {
            **FMLP_BASE_300K,
            'pop_weighted_sampler': False,
            'pop_reg_alpha': 0.0,
        }),
        ("H10_ablation_fmlp_full", "finalmlp", {
            **FMLP_BASE_300K,
            'pop_weighted_sampler': True,
            'pop_reg_alpha': 0.01,
        }),
    ]

    records = []
    for name, kind, params in ablation_configs:
        for seed in [42, 43, 44]:
            cfg = ExperimentConfig(
                name=f"{name}__ts300000_s{seed}",
                data=data_cfg,
                model=ModelConfig(kind=kind, params=params),
                eval=EVAL_CFG,
                seed=seed,
                output_dir=RESULTS,
                cache_dir=CACHE,
            )
            print(f"\n{'='*60}")
            print(f"  {cfg.name}")
            print(f"{'='*60}")
            rec = run_experiment(cfg, skip_if_exists=True)
            records.append(rec)

    print("\n\n" + "="*60)
    print("  ABLATION STUDY — SUMMARY")
    print("="*60)
    for r in records:
        pop = r.ndcg_by_pop_bin
        tail = pop.get("tail", 0)
        head = pop.get("head", 0)
        print(f"  {r.name:50s}  NDCG={r.ndcg_at_k:.4f}  "
              f"tail={tail:.4f}  head={head:.4f}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4: §4.2 — IPS-reweighted loss
# ══════════════════════════════════════════════════════════════════════════════
#
# IPS (Inverse Propensity Scoring) — вес каждой группы = 1/(count+1)^beta.
# Группы с непопулярным позитивом получают больший вес в loss.
#
# Sweep по beta ∈ {0.1, 0.3, 0.5} для DCNv2_reg и FinalMLP.
# Также тестируем IPS без pop-aware training (standalone эффект).

def run_ips_experiments():
    """§4.2: IPS-reweighted loss на train_size=300k."""

    data_cfg = DataConfig(
        dataset="books5",
        train_size=300_000,
        valid_size=10_000,
        test_size=10_000,
        feature_set="no_hc_features",
        n_neg_train=5,
        n_neg_eval=50,
    )

    ips_configs = []

    # DCNv2_reg + IPS (поверх pop-aware training)
    for beta in [0.1, 0.3, 0.5]:
        ips_configs.append((
            f"H11_ips_dcn_beta{beta}", "dcnv2_reg", {
                **DCN_BASE_300K,
                'pop_weighted_sampler': True,
                'pop_resample_each_epoch': False,
                'pop_reg_alpha': 0.01,
                'ips_beta': beta,
            }
        ))

    # FinalMLP + IPS (поверх pop-aware training)
    for beta in [0.1, 0.3, 0.5]:
        ips_configs.append((
            f"H11_ips_fmlp_beta{beta}", "finalmlp", {
                **FMLP_BASE_300K,
                'pop_weighted_sampler': True,
                'pop_reg_alpha': 0.01,
                'ips_beta': beta,
            }
        ))

    # DCNv2_reg с IPS alone (без pop-aware, чтобы изолировать эффект IPS)
    ips_configs.append((
        "H11_ips_dcn_standalone_beta03", "dcnv2_reg", {
            **DCN_BASE_300K,
            'pop_weighted_sampler': False,
            'pop_reg_alpha': 0.0,
            'ips_beta': 0.3,
        }
    ))

    records = []
    for name, kind, params in ips_configs:
        for seed in [42]:
            cfg = ExperimentConfig(
                name=f"{name}__ts300000_s{seed}",
                data=data_cfg,
                model=ModelConfig(kind=kind, params=params),
                eval=EVAL_CFG,
                seed=seed,
                output_dir=RESULTS,
                cache_dir=CACHE,
            )
            print(f"\n{'='*60}")
            print(f"  {cfg.name}")
            print(f"{'='*60}")
            rec = run_experiment(cfg, skip_if_exists=True)
            records.append(rec)

    print("\n\n" + "="*60)
    print("  IPS EXPERIMENTS — SUMMARY")
    print("="*60)
    for r in records:
        pop = r.ndcg_by_pop_bin
        tail = pop.get("tail", 0)
        head = pop.get("head", 0)
        print(f"  {r.name:50s}  NDCG={r.ndcg_at_k:.4f}  "
              f"tail={tail:.4f}  head={head:.4f}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5: FinalMLP (fixed) — сравнительные прогоны на 10k / 60k / 300k
# ══════════════════════════════════════════════════════════════════════════════
#
# Конфигурации подобраны для fair comparison с имеющимися результатами
# DCNv2, CatBoost, DeepFM.
#
# Ключевые отличия от старого FinalMLP:
#   - emb_dim=32 вместо 64 (≈4x меньше параметров → нет overfitting)
#   - Bottleneck gate (reduction=4) вместо однослойного
#   - pop_weighted_sampler + pop_reg_alpha для честного сравнения с DCNv2_reg
#   - Адаптированные MLP/dropout/lr под каждый train_size

def run_finalmlp_comparison():
    """FinalMLP на 10k/60k/300k × 3 seeds для сравнения с остальными моделями."""

    configs_by_size = {
        10_000:  FMLP_BASE_10K,
        60_000:  FMLP_BASE_60K,
        300_000: FMLP_BASE_300K,
    }

    records = []
    for train_size, params in configs_by_size.items():
        data_cfg = DataConfig(
            dataset="books5",
            train_size=train_size,
            valid_size=10_000,
            test_size=10_000,
            feature_set="no_hc_features",
            n_neg_train=5,
            n_neg_eval=50,
        )

        for seed in [42, 43, 44]:
            cfg = ExperimentConfig(
                name=f"H12_fmlp_fixed_no_hcf__ts{train_size}_s{seed}",
                data=data_cfg,
                model=ModelConfig(kind="finalmlp", params=params),
                eval=EVAL_CFG,
                seed=seed,
                output_dir=RESULTS,
                cache_dir=CACHE,
            )
            print(f"\n{'='*60}")
            print(f"  {cfg.name}")
            print(f"{'='*60}")
            rec = run_experiment(cfg, skip_if_exists=True)
            records.append(rec)

    print("\n\n" + "="*60)
    print("  FINALMLP COMPARISON — SUMMARY")
    print("="*60)
    for r in records:
        pop = r.ndcg_by_pop_bin
        tail = pop.get("tail", 0)
        head = pop.get("head", 0)
        print(f"  ts={r.train_size:>6d} s={r.seed}  "
              f"NDCG={r.ndcg_at_k:.4f}  tail={tail:.4f}  head={head:.4f}  "
              f"params={r.n_params:,}")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Cell 5b: Один прогон FinalMLP на 300k (рекомендуемая конфигурация)
# ══════════════════════════════════════════════════════════════════════════════
#
# Два режима:
#   fast=True  — neg_strategy="popularity" в loader, pop_weighted_sampler=False
#                Самый быстрый старт (~минуты до epoch 1).
#   fast=False — runtime pop resample с tqdm (после оптимизации ~5-15 мин).
#
# Рекомендация: fast=True для первого прогона, fast=False если нужен
# точный ablation pop_weighted_sampler vs loader.

def run_finalmlp_300k(seed: int = 42, fast: bool = True):
    """FinalMLP на train_size=300k, no_hc_features."""

    data_cfg = DATA_CFG_300K_FAST if fast else DATA_CFG_300K
    params = FMLP_BASE_300K_FAST if fast else FMLP_BASE_300K
    mode = "fast" if fast else "runtime_pop"

    cfg = ExperimentConfig(
        name=f"H12_fmlp_300k_{mode}__s{seed}",
        data=data_cfg,
        model=ModelConfig(kind="finalmlp", params=params),
        eval=EVAL_CFG,
        seed=seed,
        output_dir=RESULTS,
        cache_dir=CACHE,
    )
    print(f"\n{'='*60}")
    print(f"  {cfg.name}")
    print(f"  params={params}")
    print(f"{'='*60}")
    return run_experiment(cfg, skip_if_exists=False)


# ══════════════════════════════════════════════════════════════════════════════
# Cell 6: Запуск всех экспериментов
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Раскомментируй нужные блоки (или запускай ячейками в Colab)

    # --- §4.1: Ablation study ---
    ablation_records = run_ablation_study()

    # --- §4.2: IPS experiments ---
    ips_records = run_ips_experiments()

    # --- FinalMLP comparison (10k/60k/300k) ---
    fmlp_records = run_finalmlp_comparison()
