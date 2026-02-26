# Hybrid Encoder HPO Run Plan

**Date:** 2026-02-25
**Target hardware:** 1x NVIDIA H200 (80GB)
**Encoder:** Hybrid ESM-C + EGNN (default)
**Estimated duration:** 3–5 days

---

## Overview

Run Optuna hyperparameter optimization on the hybrid (ESM-C + EGNN) encoder using 10% of the training data and 20 trials. The study uses TPE sampling with MedianPruner for early stopping of underperforming trials.

## Data

| Split | Full Size | 10% Subset | Path Key |
|-------|-----------|------------|----------|
| Train | 358,784 seqs | ~35,878 seqs | `TRAIN_DATA_PATH` |
| Validation | 44,848 seqs | ~4,485 seqs | `VAL_DATA_PATH` |
| Test | skipped during HPO | — | — |

Pre-computed data required:
- **Graph index:** `data/processed/graph_index.json` (34MB, 16-shard archive)
- **Graph archives:** `data/processed/graphs.shard-{0000..0015}-of-0016.pngrph`
- **Label embeddings:** `data/embeddings/go_frozen_label_embeddings_2025_02_06.pt`
- **GO annotations:** `data/annotations/go_annotations_2025_02_06.pkl`

## Search Space

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| `LEARNING_RATE` | float | [1e-5, 1e-2] | log |
| `WEIGHT_DECAY` | float | [1e-6, 1e-2] | log |
| `OPTIMIZER` | categorical | {Adam, AdamW} | — |
| `GRADIENT_ACCUMULATION_STEPS` | categorical | {1, 2, 4} | — |
| `CLIP_VALUE` | float | [0.5, 5.0] | uniform |
| `NUM_EPOCHS` | int | [10, 30] | uniform |
| `FOCAL_LOSS_ALPHA` | categorical | {-1, 0.25, 0.5, 0.75} | — |
| `MAX_ATOMS_PER_BATCH` | categorical | {10000, 15000, 20000, 30000} | — |

**Optimization target:** `validation_f1_macro` (maximize)

## Optuna Settings

| Setting | Value |
|---------|-------|
| `n_trials` | 20 |
| `n_startup_trials` | 5 (random sampling before TPE) |
| `n_warmup_steps` | 5 (epochs before pruner activates) |
| Sampler | TPESampler (seed=42) |
| Pruner | MedianPruner |
| Storage | SQLite (`outputs/hybrid_hpo_hpo.db`) |

## Command

```bash
pixi run python bin/hpo.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names=null \
    run.full_path_name=FULL_DATA_PATH \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    params.TRAIN_SUBSET_FRACTION=0.1 \
    params.VALIDATION_SUBSET_FRACTION=0.1 \
    run.wandb_project=null \
    run.name=hybrid_hpo \
    hpo.n_trials=20
```

## Fixed Parameters (Not Searched)

These use defaults from `configs/params/default.yaml`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LOSS_FN` | FocalLoss | Standard for GO annotation imbalance |
| `FOCAL_LOSS_GAMMA` | 2 | Standard focal loss |
| `LATENT_EMBEDDING_DIM` | 1024 | Architecture constant |
| `OUTPUT_MLP_NUM_LAYERS` | 3 | Architecture constant |
| `PROJECTION_HEAD_NUM_LAYERS` | 4 | Architecture constant |
| `EGNN_N_LAYERS` | 4 | Architecture constant |
| `EGNN_HIDDEN_DIM` | 256 | Architecture constant |
| `LORA` | true | Memory efficiency |
| `LORA_RANK` | 4 | Memory efficiency |
| `AUGMENT_RESIDUE_PROBABILITY` | 0.1 | Standard augmentation |
| `LABEL_EMBEDDING_NOISING_ALPHA` | 20.0 | Standard regularization |

## Time Estimate

Based on benchmarking with 1% data on an A5000 (24GB):

- 1% data, batch_size=2, A5000: ~24 min/epoch
- H200 is ~3–4x faster than A5000 (FP16 TFLOPS: 990 vs 262)
- H200 has 80GB → can fit larger batches → further speedup

**Per-trial estimate on H200 with 10% data:**
- ~15–30 min/epoch (depending on `MAX_ATOMS_PER_BATCH`)
- Good trial (10–30 epochs): ~3–15 hours
- Pruned trial (~5 epochs): ~1.5–2.5 hours

**Total 20 trials:** ~3–5 days (assuming ~50% pruned)

## After HPO

Once the study completes, retrieve best params and retrain on full data:

```bash
# Check results
pixi run python -c "
import optuna
study = optuna.load_study(study_name='hybrid_hpo_hpo', storage='sqlite:///outputs/hybrid_hpo_hpo.db')
print(f'Best trial: #{study.best_trial.number}')
print(f'Best f1_macro: {study.best_value:.4f}')
for k, v in study.best_params.items():
    print(f'  {k}: {v}')
"

# Retrain with best params on full data
pixi run python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.name=hybrid_best_params \
    params.LEARNING_RATE=<best> \
    params.WEIGHT_DECAY=<best> \
    params.OPTIMIZER=<best> \
    params.GRADIENT_ACCUMULATION_STEPS=<best> \
    params.CLIP_VALUE=<best> \
    params.NUM_EPOCHS=<best> \
    params.MAX_ATOMS_PER_BATCH=<best>
```

## Notes

- The study persists to SQLite and can be resumed if interrupted. Re-running the same command with the same `study_name` will continue from where it left off.
- OOM trials are automatically pruned (not counted as failures).
- NaN trials are automatically pruned.
- The `output_layer` MLP runs in fp32 (autocast disabled) to prevent NaN from fp16 overflow in BatchNorm1d — this was fixed on `fix/metafix` and cherry-picked to `feat/optuna`.
