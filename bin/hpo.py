"""Optuna hyperparameter optimization for ProtNote.

Usage:
    python bin/hpo.py run.gpus=2 hpo.n_trials=50
    python bin/hpo.py run.gpus=1 hpo.n_trials=3 params.TRAIN_SUBSET_FRACTION=0.1
    python bin/hpo.py hpo.study_name=my_study hpo.n_trials=100  # resume
"""

import ctypes
import os
import socket
import logging

import hydra
import optuna
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from bin.main import train_validate_test


class _SharedMemoryCallback:
    """Picklable trial callback using shared memory for multi-GPU.

    Writes metric to shared memory so the parent can poll. The parent
    process sets prune_flag to signal the child to raise TrialPruned.

    This is a module-level class (not a closure) so it can be pickled
    by mp.spawn.
    """

    def __init__(self, report_epoch, report_metric, prune_flag):
        self.report_epoch = report_epoch
        self.report_metric = report_metric
        self.prune_flag = prune_flag

    def __call__(self, epoch, metric):
        self.report_epoch.value = epoch
        self.report_metric.value = metric
        if self.prune_flag.value:
            raise optuna.exceptions.TrialPruned()


def _make_objective(cfg):
    """Create an Optuna objective function that wraps the training pipeline."""
    original_run_name = cfg.run.name

    def objective(trial):
        # --- Suggest hyperparameters ---
        use_hybrid = not cfg.run.use_sequence_encoder

        suggested = {
            "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-5, 1e-2, log=True),
            "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-2, log=True),
            "OPTIMIZER": trial.suggest_categorical("OPTIMIZER", ["Adam", "AdamW"]),
            "GRADIENT_ACCUMULATION_STEPS": trial.suggest_categorical(
                "GRADIENT_ACCUMULATION_STEPS", [1, 2, 4]
            ),
            "CLIP_VALUE": trial.suggest_float("CLIP_VALUE", 0.5, 5.0),
            "NUM_EPOCHS": trial.suggest_int("NUM_EPOCHS", 10, 30),
            "FOCAL_LOSS_ALPHA": trial.suggest_categorical(
                "FOCAL_LOSS_ALPHA", [-1, 0.25, 0.5, 0.75]
            ),
        }

        if use_hybrid:
            # Hybrid encoder uses dynamic batching by atom count
            suggested["MAX_ATOMS_PER_BATCH"] = trial.suggest_categorical(
                "MAX_ATOMS_PER_BATCH", [10000, 15000, 20000, 30000]
            )
        else:
            # Legacy ProteInfer uses fixed batch sizes
            suggested["TRAIN_BATCH_SIZE"] = trial.suggest_categorical(
                "TRAIN_BATCH_SIZE", [2, 4, 8]
            )

        # --- Override config with suggested values ---
        trial_cfg = cfg.copy()
        for key, value in suggested.items():
            OmegaConf.update(trial_cfg, f"params.{key}", value)

        run = trial_cfg.run
        trial_number = trial.number

        # Tag W&B run with trial number
        if run.wandb_project is not None:
            OmegaConf.update(
                trial_cfg, "run.name", f"{original_run_name}_trial_{trial_number}"
            )

        # --- Set up DDP environment ---
        world_size = run.gpus * run.nodes
        if not run.amlt:
            os.environ["MASTER_ADDR"] = "localhost"
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

        # --- Shared metric for returning result from rank 0 ---
        result_metric = mp.Value("d", 0.0)

        # --- Build callback and run training ---
        try:
            if world_size == 1:
                # Single-GPU: call directly (no spawn, no pickle constraints).
                # The trial_callback closure works because there's no serialization.
                def trial_callback(epoch, metric):
                    trial.report(metric, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                train_validate_test(
                    gpu=0,
                    cfg=trial_cfg,
                    world_size=1,
                    trial_callback=trial_callback,
                    result_metric=result_metric,
                )
            else:
                # Multi-GPU: use picklable shared-memory callback.
                # Pruning is deferred: the child writes metrics, and
                # prune_flag can be set between trials (not mid-epoch).
                report_epoch = mp.Value("i", 0)
                report_metric_shm = mp.Value("d", 0.0)
                prune_flag = mp.Value(ctypes.c_bool, False)

                callback = _SharedMemoryCallback(
                    report_epoch, report_metric_shm, prune_flag
                )

                mp.spawn(
                    train_validate_test,
                    nprocs=run.gpus,
                    args=(trial_cfg, world_size, callback, result_metric),
                )

                # Report the last metric to Optuna (parent-side)
                if report_epoch.value > 0:
                    trial.report(report_metric_shm.value, report_epoch.value)

        except optuna.exceptions.TrialPruned:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(
                    f"Trial {trial_number} OOM with params: {suggested}"
                )
                raise optuna.exceptions.TrialPruned()
            raise
        except Exception as e:
            if "nan" in str(e).lower():
                logging.warning(
                    f"Trial {trial_number} NaN with params: {suggested}"
                )
                raise optuna.exceptions.TrialPruned()
            raise
        finally:
            torch.cuda.empty_cache()

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

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        logging.info(
            f"Resuming study '{study_name}' with {len(study.trials)} existing trials "
            f"({len(completed_trials)} completed). Best so far: {study.best_value:.4f}"
        )
    elif study.trials:
        logging.info(
            f"Resuming study '{study_name}' with {len(study.trials)} existing trials "
            f"(none completed yet)."
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

    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed_trials:
        print(f"Completed trials: {len(completed_trials)}")
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
    else:
        print("WARNING: No trials completed successfully. All were pruned or failed.")
        print("Consider adjusting search space or increasing n_trials.")
    print("=" * 80)


if __name__ == "__main__":
    main()
