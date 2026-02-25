# Optuna Hyperparameter Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Optuna-based hyperparameter optimization via a new `bin/hpo.py` entry point, with MedianPruner and SQLite persistence.

**Architecture:** Separate `bin/hpo.py` creates an Optuna study, defines an objective that overrides Hydra config params in-memory and calls `train_validate_test` via `mp.spawn`. A lightweight callback added to `ProtNoteTrainer` reports validation metrics for pruning. Metric is returned to the objective via `mp.Value`.

**Tech Stack:** Optuna (TPESampler, MedianPruner, SQLite storage), Hydra/OmegaConf, PyTorch DDP

---

## Task 1: Add Optuna Dependency

**Files:**
- Modify: `pyproject.toml:18-32`

**Step 1: Add optuna to the `[project.optional-dependencies] ml` list**

In `pyproject.toml`, add `"optuna>=3.0,<5"` to the `ml` extras list (after the last entry):

```toml
[project.optional-dependencies]
ml = [
    "torchmetrics==1.4.0",
    "torcheval==0.0.7",
    "tensorboard==2.16.2",
    "transformers==4.40.0",
    "wandb==0.15.11",
    "sacremoses==0.0.53",
    "pynvml==11.5.0",
    "azureml-mlflow==1.53.0",
    "loralib==0.1.2",
    "umap-learn==0.5.4",
    "atomworks[ml,openbabel]>=2.2.0,<3",
    "esm @ git+https://github.com/KagurazakaHinagi/esm.git",
    "optuna>=3.0,<5",
]
```

**Step 2: Install the dependency**

Run: `pixi install` or `pip install optuna`

**Step 3: Verify installation**

Run: `python -c "import optuna; print(optuna.__version__)"`
Expected: version number prints without error

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optuna dependency for hyperparameter optimization"
```

---

## Task 2: Add HPO Hydra Config

**Files:**
- Create: `configs/hpo/default.yaml`
- Modify: `configs/config.yaml`

**Step 1: Create `configs/hpo/default.yaml`**

```yaml
# Optuna hyperparameter optimization settings
n_trials: 50            # Number of HPO trials to run
timeout: null           # Timeout in seconds (null = no timeout)
study_name: null        # Study name (null = auto-generate from run.name + timestamp)
n_startup_trials: 5     # Random sampling trials before TPE kicks in
n_warmup_steps: 5       # Validation epochs before pruner activates per trial
```

**Step 2: Add `hpo` to Hydra defaults in `configs/config.yaml`**

Add `- hpo: default` to the defaults list (after `- remote: default`):

```yaml
defaults:
  - params: default
  - encoder/proteinfer: default
  - encoder/structural: default
  - paths: default
  - remote: default
  - hpo: default
  - _self_
```

**Step 3: Verify Hydra can load the config**

Run: `python -c "import hydra; from omegaconf import DictConfig; print('ok')"`
Expected: `ok` (basic sanity; full validation happens when we run `hpo.py`)

**Step 4: Commit**

```bash
git add configs/hpo/default.yaml configs/config.yaml
git commit -m "config: add Optuna HPO defaults to Hydra config"
```

---

## Task 3: Add Trial Callback to ProtNoteTrainer

**Files:**
- Modify: `protnote/models/ProtNoteTrainer.py:92-106` (\_\_init\_\_ signature)
- Modify: `protnote/models/ProtNoteTrainer.py:994-1004` (train loop after validate)

**Step 1: Add `trial_callback` parameter to `__init__`**

In `ProtNoteTrainer.__init__` (line 92), add `trial_callback=None` to the signature and store it:

Change the `__init__` signature from:
```python
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        rank: int,
        config: dict,
        logger: logging.Logger,
        timestamp: str,
        run_name: str,
        loss_fn: torch.nn.Module,
        use_wandb: bool = False,
        use_amlt: bool = False,
        is_master: bool = True,
        starting_epoch: int = 1,
    ):
```

To:
```python
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        rank: int,
        config: dict,
        logger: logging.Logger,
        timestamp: str,
        run_name: str,
        loss_fn: torch.nn.Module,
        use_wandb: bool = False,
        use_amlt: bool = False,
        is_master: bool = True,
        starting_epoch: int = 1,
        trial_callback=None,
    ):
```

Then, inside `__init__` after `self.starting_epoch = starting_epoch` (line 132), add:
```python
        self.trial_callback = trial_callback
```

**Step 2: Invoke callback after validation in `train` method**

In the `train` method, after the validation block (after line 1004 — after `self.validate(...)` returns), add:

```python
                # Optuna trial callback: report metric and check for pruning
                if self.trial_callback is not None and self.is_master:
                    self.trial_callback(epoch, self.best_val_metric)
```

This goes right after the `self.validate(...)` call block (line 1004) and before the logging line (line 1006). The indentation should match the `if epoch % self.EPOCHS_PER_VALIDATION == 0:` block (8 spaces + 4 for the inner if).

**Step 3: Verify regular training is unaffected**

Since `trial_callback` defaults to `None`, the condition `self.trial_callback is not None` is always `False` for normal training. No behavior change.

**Step 4: Commit**

```bash
git add protnote/models/ProtNoteTrainer.py
git commit -m "feat: add optional trial_callback to ProtNoteTrainer for HPO"
```

---

## Task 4: Thread trial_callback Through main.py

**Files:**
- Modify: `bin/main.py:63` (mp.spawn call)
- Modify: `bin/main.py:66` (train_validate_test signature)
- Modify: `bin/main.py:410-422` (Trainer initialization)

**Step 1: Update `train_validate_test` signature**

Change line 66 from:
```python
def train_validate_test(gpu, cfg, world_size):
```
To:
```python
def train_validate_test(gpu, cfg, world_size, trial_callback=None, result_metric=None):
```

- `trial_callback`: optional callback for Optuna pruning (called after each validation)
- `result_metric`: optional `mp.Value` to write the best metric back to the parent process

**Step 2: Update `mp.spawn` call in `main` function**

Change line 63 from:
```python
    mp.spawn(train_validate_test, nprocs=run.gpus, args=(cfg, world_size))
```
To:
```python
    mp.spawn(train_validate_test, nprocs=run.gpus, args=(cfg, world_size, None, None))
```

This keeps regular training unaffected (passes `None` for both new args).

**Step 3: Pass `trial_callback` to ProtNoteTrainer**

In the `ProtNoteTrainer` initialization (lines 410-422), add `trial_callback=trial_callback`:

```python
    Trainer = ProtNoteTrainer(
        model=model,
        device=device,
        rank=rank,
        config=config,
        logger=logger,
        timestamp=timestamp,
        run_name=run.name,
        use_wandb=run.wandb_project is not None and is_master,
        use_amlt=run.amlt,
        loss_fn=loss_fn,
        is_master=is_master,
        trial_callback=trial_callback,
    )
```

**Step 4: Write best metric to shared value after training**

After the `Trainer.train(...)` call (line 468), add:

```python
        # Write best validation metric to shared value for HPO
        if result_metric is not None and is_master:
            result_metric.value = Trainer.best_val_metric
```

**Step 5: Commit**

```bash
git add bin/main.py
git commit -m "feat: thread trial_callback and result_metric through train_validate_test"
```

---

## Task 5: Create bin/hpo.py

**Files:**
- Create: `bin/hpo.py`

**Step 1: Write the HPO entry point**

Create `bin/hpo.py` with the following content:

```python
"""Optuna hyperparameter optimization for ProtNote.

Usage:
    python bin/hpo.py run.gpus=2 hpo.n_trials=50
    python bin/hpo.py run.gpus=1 hpo.n_trials=3 params.TRAIN_SUBSET_FRACTION=0.1
    python bin/hpo.py hpo.study_name=my_study hpo.n_trials=100  # resume
"""

import os
import socket
import logging
import multiprocessing as mp

import hydra
import optuna
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from bin.main import train_validate_test


def _make_objective(cfg):
    """Create an Optuna objective function that wraps the training pipeline."""

    def objective(trial):
        # --- Suggest hyperparameters ---
        suggested = {
            "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-5, 1e-2, log=True),
            "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-2, log=True),
            "OPTIMIZER": trial.suggest_categorical("OPTIMIZER", ["Adam", "AdamW"]),
            "GRADIENT_ACCUMULATION_STEPS": trial.suggest_categorical(
                "GRADIENT_ACCUMULATION_STEPS", [1, 2, 4]
            ),
            "CLIP_VALUE": trial.suggest_float("CLIP_VALUE", 0.5, 5.0),
            "NUM_EPOCHS": trial.suggest_int("NUM_EPOCHS", 15, 50),
            "TRAIN_BATCH_SIZE": trial.suggest_categorical(
                "TRAIN_BATCH_SIZE", [4, 8, 16, 32]
            ),
        }

        # --- Override config with suggested values ---
        trial_cfg = cfg.copy()
        for key, value in suggested.items():
            OmegaConf.update(trial_cfg, f"params.{key}", value)

        run = trial_cfg.run
        trial_number = trial.number

        # Tag W&B run with trial number
        if run.wandb_project is not None:
            OmegaConf.update(
                trial_cfg, "run.name", f"{run.name}_trial_{trial_number}"
            )

        # --- Define pruning callback ---
        def trial_callback(epoch, metric):
            trial.report(metric, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # --- Shared metric for returning result from DDP rank 0 ---
        result_metric = mp.Value("d", 0.0)

        # --- Set up DDP environment ---
        world_size = run.gpus * run.nodes
        if not run.amlt:
            os.environ["MASTER_ADDR"] = "localhost"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

        # --- Run training ---
        try:
            mp.spawn(
                train_validate_test,
                nprocs=run.gpus,
                args=(trial_cfg, world_size, trial_callback, result_metric),
            )
        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logging.warning(
                    f"Trial {trial_number} OOM with params: {suggested}"
                )
                return 0.0
            raise
        except Exception as e:
            if "nan" in str(e).lower():
                logging.warning(
                    f"Trial {trial_number} NaN with params: {suggested}"
                )
                raise optuna.exceptions.TrialPruned()
            raise

        return result_metric.value

    return objective


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    hpo = cfg.hpo
    run = cfg.run

    # --- Study name ---
    study_name = hpo.study_name
    if study_name is None:
        study_name = f"{run.name}_hpo"

    # --- SQLite storage ---
    output_dir = cfg.paths.get("OUTPUT_MODEL_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    storage = f"sqlite:///{os.path.join(output_dir, study_name)}.db"

    # --- Create or resume study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg.params.SEED,
            n_startup_trials=hpo.n_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=hpo.n_startup_trials,
            n_warmup_steps=hpo.n_warmup_steps,
        ),
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        logging.info(
            f"Resuming study '{study_name}' with {n_existing} existing trials. "
            f"Best so far: {study.best_value:.4f}"
        )

    # --- Run optimization ---
    objective = _make_objective(cfg)
    study.optimize(
        objective,
        n_trials=hpo.n_trials,
        timeout=hpo.timeout,
    )

    # --- Report results ---
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Study: {study_name}")
    print(f"Storage: {storage}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best value (validation_f1_macro): {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nTo train with best params:")
    params_str = " ".join(
        f"params.{k}={v}" for k, v in study.best_params.items()
    )
    print(f"  python bin/main.py {params_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('bin/hpo.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add bin/hpo.py
git commit -m "feat: add Optuna HPO entry point (bin/hpo.py)"
```

---

## Task 6: Handle TrialPruned in DDP Workers

**Files:**
- Modify: `bin/main.py:546-582` (cleanup section)

The `TrialPruned` exception raised inside `trial_callback` will propagate through the DDP training loop. We need the `train_validate_test` function to handle cleanup properly when pruning occurs.

**Step 1: Wrap the training+cleanup in a try/finally**

In `train_validate_test`, wrap the training section (from the `Trainer.train(...)` call through the cleanup section) in a try/finally so that DDP is always cleaned up:

Find the cleanup section at the end of `train_validate_test` (lines 575-582):
```python
    # Loggers
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    # Torch
    torch.cuda.empty_cache()
    dist.destroy_process_group()
```

Wrap the entire training/validation/test/cleanup block in try/finally. After the line `logger.info(OmegaConf.to_yaml(params))` (line 131), add a try block. Move the cleanup into a `finally`:

```python
    # Log the params
    logger.info(OmegaConf.to_yaml(params))

    try:
        # ... all existing code from label_tokenizer init through metric logging ...
```

Then at the very end (replacing lines 575-582):
```python
    finally:
        # Loggers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        # Torch
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
```

The `dist.is_initialized()` guard protects against double-destroy if an exception occurs before DDP init.

**Step 2: Commit**

```bash
git add bin/main.py
git commit -m "fix: ensure DDP cleanup on TrialPruned exception in train_validate_test"
```

---

## Task 7: Smoke Test

**Files:** None (manual verification)

**Step 1: Verify hpo.py loads without errors**

Run: `python bin/hpo.py --help`
Expected: Hydra help output showing config groups including `hpo`

**Step 2: Run a minimal 1-trial test (if data is available)**

Run:
```bash
python bin/hpo.py \
    run.gpus=1 \
    hpo.n_trials=1 \
    params.TRAIN_SUBSET_FRACTION=0.1 \
    params.VALIDATION_SUBSET_FRACTION=0.1 \
    params.NUM_EPOCHS=2
```

Expected: One trial completes (or fails with data-not-found if on a machine without data). Check that:
- Optuna study is created (SQLite `.db` file exists)
- Best params are printed
- No leftover DDP processes

**Step 3: Verify regular training still works**

Run: `python bin/main.py --help`
Expected: Same Hydra help as before (no regressions from our changes)

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address issues from smoke testing"
```

---

## Task 8: Final Commit and Summary

**Step 1: Verify all changes**

Run: `git log --oneline feat/optuna ^fix/metafix`
Expected: 5-7 commits covering dependency, config, trainer callback, main.py threading, hpo.py, DDP cleanup

**Step 2: Review the diff**

Run: `git diff fix/metafix..feat/optuna --stat`
Expected:
- `pyproject.toml` — 1 line added
- `configs/hpo/default.yaml` — new file (~6 lines)
- `configs/config.yaml` — 1 line added
- `protnote/models/ProtNoteTrainer.py` — ~5 lines added
- `bin/main.py` — ~15 lines modified
- `bin/hpo.py` — new file (~140 lines)
- `docs/plans/` — design + plan docs
