# Optuna Hyperparameter Optimization — Design Document

**Date:** 2026-02-25
**Status:** Approved

## Goal

Integrate Optuna into ProtNote's training pipeline to automatically search for optimal core training hyperparameters, using TPE sampling, MedianPruner for early stopping of bad trials, and SQLite persistence for study resumption.

## Approach

**In-process integration (Approach B):** A new `bin/hpo.py` entry point imports and calls the existing `train_validate_test` function directly. A lightweight callback in `ProtNoteTrainer` reports validation metrics to Optuna after each epoch, enabling MedianPruner.

## Search Space

Shared across both encoder modes:

| Parameter | Type | Range | Scale |
|-----------|------|-------|-------|
| `LEARNING_RATE` | float | [1e-5, 1e-2] | log |
| `WEIGHT_DECAY` | float | [1e-6, 1e-2] | log |
| `OPTIMIZER` | categorical | {Adam, AdamW} | — |
| `GRADIENT_ACCUMULATION_STEPS` | categorical | {1, 2, 4} | — |
| `CLIP_VALUE` | float | [0.5, 5.0] | uniform |
| `NUM_EPOCHS` | int | [10, 30] | — |

Encoder-specific batching:

| Encoder | Parameter | Type | Range |
|---------|-----------|------|-------|
| Hybrid (ESM-C + EGNN) | `MAX_ATOMS_PER_BATCH` | categorical | {10000, 15000, 20000, 30000} |
| Legacy (ProteInfer) | `TRAIN_BATCH_SIZE` | categorical | {2, 4, 8} |

**Optimization target:** `validation_f1_macro` (maximize)

## Architecture

### New Files

**`bin/hpo.py`** — Hydra entry point that:
1. Creates/resumes an Optuna study backed by SQLite
2. Defines an objective function that suggests hyperparameters, overrides the Hydra config, and calls `train_validate_test` via `mp.spawn`
3. Uses `mp.Value` (shared memory float) to retrieve `best_val_metric` from the rank-0 training process
4. Logs best trial params at completion

**`configs/hpo/default.yaml`** — HPO-specific configuration:
```yaml
n_trials: 50
timeout: null
study_name: null
n_startup_trials: 5
n_warmup_steps: 5
```

### Modifications to Existing Code

**`ProtNoteTrainer.__init__`**: Accept optional `trial_callback=None` parameter.

**`ProtNoteTrainer.train`**: After validation call (~line 1004), invoke:
```python
if self.trial_callback is not None and self.is_master:
    self.trial_callback(epoch, self.best_val_metric)
```

**`bin/main.py` / `train_validate_test`**: Thread `trial_callback` parameter through to `ProtNoteTrainer`. Default is `None` so regular training is unaffected.

### Objective Function Flow

```
objective(trial, cfg)
  ├── Suggest 7 hyperparameters via trial.suggest_*()
  ├── Override cfg.params in-memory
  ├── Create trial_callback that calls trial.report() + trial.should_prune()
  ├── Create mp.Value for metric return
  ├── try:
  │     mp.spawn(train_validate_test, args=(..., trial_callback, shared_metric))
  │   except TrialPruned: raise
  │   except RuntimeError (CUDA OOM): return 0.0
  │   finally: cleanup GPU memory + DDP
  └── return shared_metric.value
```

### DDP Handling

- `trial_callback` runs only on `is_master` (rank 0)
- When `TrialPruned` is raised on rank 0, DDP process group is destroyed cleanly
- Non-master processes detect disconnection and exit

### Study Configuration

- **Sampler:** TPESampler (Optuna default, Bayesian optimization)
- **Pruner:** MedianPruner with `n_startup_trials=5` (random trials before TPE), `n_warmup_steps=5` (epochs before pruning activates)
- **Storage:** SQLite at `<output_dir>/<study_name>.db`
- **Resumption:** `load_if_exists=True` — kill and restart without losing completed trials

### W&B Integration

Each trial creates a separate W&B run named `{run_name}_trial_{trial.number}`. `wandb.finish()` is called per trial.

## Error Handling

1. **CUDA OOM**: Catch `RuntimeError` with "out of memory" → `torch.cuda.empty_cache()`, return 0.0
2. **NaN loss**: Detect NaN in training → raise `TrialPruned()` to skip config
3. **Process cleanup**: `finally` block destroys DDP process group and frees GPU memory per trial
4. **W&B cleanup**: `wandb.finish()` per trial to prevent run leakage

## Usage

```bash
# Run 50 trials
python bin/hpo.py run.gpus=2 hpo.n_trials=50

# Resume a previous study
python bin/hpo.py run.gpus=2 hpo.study_name=my_study hpo.n_trials=100

# Quick test with subset data
python bin/hpo.py run.gpus=1 hpo.n_trials=3 params.TRAIN_SUBSET_FRACTION=0.1

# Train with best params (printed at end of HPO)
python bin/main.py params.LEARNING_RATE=0.00042 params.OPTIMIZER=AdamW ...
```

## Files Changed

| File | Change |
|------|--------|
| `bin/hpo.py` | **New** — HPO entry point |
| `configs/hpo/default.yaml` | **New** — HPO config |
| `configs/config.yaml` | Add `hpo` to Hydra defaults |
| `protnote/models/ProtNoteTrainer.py` | Add `trial_callback` param (~4 lines) |
| `bin/main.py` | Thread `trial_callback` through `train_validate_test` |
