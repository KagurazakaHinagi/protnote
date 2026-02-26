# ToxinNote Training Guide

This guide covers environment setup, configuration, training, evaluation, and inference for ToxinNote models.

For data preparation (downloading, preprocessing, embedding generation), see the [Data Preparation Guide](data_guide.md).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Configuration System](#3-configuration-system)
4. [Training](#4-training)
5. [Evaluation and Testing](#5-evaluation-and-testing)
6. [Inference](#6-inference)
7. [Advanced Topics](#7-advanced-topics)

---

## 1. Prerequisites

- **Hardware**: NVIDIA GPU with CUDA 12.9+
- **Software**: Python 3.10+, [Pixi](https://pixi.sh/) package manager
- **Platform**: Linux (x86_64)
- **Data**: Complete the [Data Preparation Guide](data_guide.md) before training

## 2. Environment Setup

The project defines two pixi environments:

| Environment | Command | Platforms | Includes | Use for |
|-------------|---------|-----------|----------|---------|
| `default` | `pixi shell` | Linux (x86_64) | All dependencies (PyTorch, CUDA, torch-geometric, etc.) | Training, evaluation, inference |
| `dataprep` | `pixi shell -e dataprep` | Linux, macOS, Windows | Core packages only (no GPU deps) | Data downloading and preprocessing |

```bash
# Activate the full environment for training (Linux with GPU)
pixi shell
```

This installs PyTorch 2.7.1, Hydra, OmegaConf, transformers, ESM-C, AtomWorks, and all other dependencies defined in `pyproject.toml`.

## 3. Configuration System

ToxinNote uses [Hydra](https://hydra.cc/) for configuration management.

### 3.1 Config Structure

```
configs/
  config.yaml                  # Root config: defaults + runtime settings
  params/
    default.yaml               # Training hyperparameters
  encoder/
    proteinfer/
      default.yaml             # ProteInfer CNN constants
    structural/
      default.yaml             # EGNN structural encoder settings
  paths/
    default.yaml               # Data and output paths (relative)
  remote/
    default.yaml               # Remote URLs (HuggingFace, AlphaFold DB)
  <!--hpo/
    default.yaml               # Optuna HPO settings-->
```

### 3.2 Runtime Settings (`config.yaml` → `run:`)

These control what the training script does:

| Setting | Default | Description |
|---------|---------|-------------|
| `run.name` | `ToxinNote` | Experiment name (used in checkpoints, logs, W&B) |
| `run.train_path_name` | `TRAIN_DATA_PATH` | Config key for training FASTA (set to `null` to skip training) |
| `run.validation_path_name` | `VAL_DATA_PATH` | Config key for validation FASTA (set to `null` to skip) |
| `run.test_paths_names` | `[TEST_DATA_PATH]` | List of config keys for test FASTAs (set to `null` to skip) |
| `run.full_path_name` | `FULL_DATA_PATH` | Config key for full dataset (vocabulary generation) |
| `run.annotations_path_name` | `GO_ANNOTATIONS_PATH` | Config key for GO/EC annotations |
| `run.base_label_embedding_name` | `GO_BASE_LABEL_EMBEDDING_PATH` | Config key for label embeddings |
| `run.wandb_project` | `toxinnote` | W&B project name (set to `null` to disable logging) |
| `run.wandb_entity` | `cs527-toxinnote` | W&B entity/team name |
| `run.model_file` | `null` | Checkpoint to load (filename in `data/models/ProtNote/`) |
| `run.from_checkpoint` | `false` | Resume training from checkpoint (restores optimizer state) |
| `run.save_prediction_results` | `false` | Save predictions as HDF5 |
| `run.save_embeddings` | `false` | Save joint embeddings |
| `run.save_val_test_metrics` | `false` | Append val/test metrics to a JSON file |
| `run.save_val_test_metrics_file` | `val_test_metrics.json` | JSON file for accumulated metrics |
| `run.eval_only_represented_labels` | `false` | Only evaluate labels represented in the dataset |
| `run.use_sequence_encoder` | `false` | Use legacy ProteInfer CNN (default: hybrid EGNN) |
| `run.gpus` | `1` | Number of GPUs per node |
| `run.nodes` | `1` | Number of nodes |
| `run.nr` | `0` | Node rank (set automatically when using AMLT) |
| `run.amlt` | `false` | Running on AMLT (Azure ML) |
| `run.mlflow` | `false` | Enable MLFlow logging (requires AMLT) |

### 3.3 Key Hyperparameters (`params/default.yaml`)

**Batch sizes and sampling:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | `8` | Per-GPU training batch size (used when `MAX_ATOMS_PER_BATCH` is null) |
| `VALIDATION_BATCH_SIZE` | `8` | Per-GPU validation batch size (used when `MAX_ATOMS_PER_BATCH` is null) |
| `TEST_BATCH_SIZE` | `8` | Per-GPU test batch size (used when `MAX_ATOMS_PER_BATCH` is null) |
| `MAX_ATOMS_PER_BATCH` | `null` | Max total atoms per batch for dynamic batching (null = use fixed batch size) |
| `WEIGHTED_SAMPLING` | `true` | Over-sample rare sequences by inverse label frequency |
| `INV_FREQUENCY_POWER` | `0.5` | Power for inverse frequency weighting |
| `TRAIN_LABEL_SAMPLE_SIZE` | `null` | Sample K labels per batch (null = all) |

**Optimization:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | `0.0003` | Learning rate |
| `OPTIMIZER` | `Adam` | Optimizer (Adam or AdamW) |
| `WEIGHT_DECAY` | `0.001` | Weight decay (AdamW/SGD only) |
| `NUM_EPOCHS` | `46` | Total training epochs |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Accumulate gradients over N steps |
| `CLIP_VALUE` | `1` | Gradient clipping max norm (null = off) |

**Architecture:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PROTEIN_EMBEDDING_DIM` | `1100` | Protein encoder output dimension |
| `LABEL_EMBEDDING_DIM` | `1024` | Label encoder output dimension |
| `LATENT_EMBEDDING_DIM` | `1024` | Shared latent space dimension |
| `FEATURE_FUSION` | `concatenation` | Fusion method: `concatenation`, `similarity`, `concatenation_diff`, `concatenation_prod` |
| `OUTPUT_MLP_NUM_LAYERS` | `3` | Output MLP depth |
| `PROJECTION_HEAD_NUM_LAYERS` | `4` | Projection head depth |

**Encoder settings:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_PLM_EMBEDDINGS` | `true` | Use ESM-C embeddings (false = learned AA embeddings) |
| `EGNN_HIDDEN_DIM` | `256` | EGNN hidden dimension |
| `EGNN_N_LAYERS` | `4` | Number of EGNN layers |
| `EGNN_OUT_DIM` | `256` | EGNN output dimension |
| `TRAIN_PROTEIN_ENCODER` | `true` | Train the hybrid encoder (EGNN layers) |
| `PRETRAINED_SEQUENCE_ENCODER` | `false` | Load pretrained ProteInfer weights (legacy encoder only) |
| `TRAIN_SEQUENCE_ENCODER` | `true` | Train ProteInfer CNN (legacy encoder only; `false` = freeze) |
| `LORA` | `true` | Enable LoRA for label encoder |
| `LORA_RANK` | `4` | LoRA rank |
| `LABEL_ENCODER_NUM_TRAINABLE_LAYERS` | `0` | Trainable layers in E5 (0 = frozen) |

**Loss function:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LOSS_FN` | `FocalLoss` | Loss: `BCE`, `FocalLoss`, `WeightedBCE`, `BatchWeightedBCE`, `RGDBCE`, `CBLoss` |
| `FOCAL_LOSS_GAMMA` | `2` | Focal loss focusing parameter |
| `FOCAL_LOSS_ALPHA` | `-1` | Focal loss balancing (-1 = no balancing) |

**Augmentation:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AUGMENT_RESIDUE_PROBABILITY` | `0.1` | Probability of BLOSUM62-based AA substitution per residue |
| `LABEL_EMBEDDING_NOISING_ALPHA` | `20.0` | Additive noise on label embeddings during training |

**Evaluation:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPTIMIZATION_METRIC_NAME` | `f1_macro` | Metric to optimize for model selection |
| `DECISION_TH` | `0.5` | Decision threshold (null = tune on validation set) |
| `ESTIMATE_MAP` | `false` | Compute mean average precision |

### 3.4 Data Path Keys (`paths/default.yaml`)

Path names referenced in CLI arguments map to relative paths under `data/`:

| Key | Path |
|-----|------|
| `TRAIN_DATA_PATH` | `sequences/uniprot_sprot_2025_01_with_struct/splits/train_GO.fasta` |
| `VAL_DATA_PATH` | `sequences/uniprot_sprot_2025_01_with_struct/splits/dev_GO.fasta` |
| `TEST_DATA_PATH` | `sequences/uniprot_sprot_2025_01_with_struct/splits/test_GO.fasta` |
| `FULL_DATA_PATH` | `sequences/uniprot_sprot_2025_01_with_struct/full_GO.fasta` |
| `GO_ANNOTATIONS_PATH` | `annotations/go_annotations_2025_02_06.pkl` |
| `GO_BASE_LABEL_EMBEDDING_PATH` | `embeddings/go_frozen_label_embeddings_2025_02_06.pt` |
| `PROTEINFER_GO_WEIGHTS_PATH` | `models/proteinfer/GO_model_weights13703706.pkl` |

### 3.5 Overriding Config at the Command Line

All parameters can be overridden via Hydra CLI syntax:

```bash
# Override a single parameter
python bin/main.py params.LEARNING_RATE=0.001

# Override multiple parameters
python bin/main.py params.TRAIN_BATCH_SIZE=8 params.NUM_EPOCHS=100

# Override run settings
python bin/main.py run.name=my_experiment run.wandb_project=my_project

# Print the resolved config without running
python bin/main.py --cfg job
```

---

## 4. Training

### 4.1 Supervised Training (Hybrid Encoder — Default)

The default encoder is the hybrid ESM-C + EGNN structure encoder. This requires pre-computed ESM-C embeddings and atom-level graphs (see [Data Preparation Guide](data_guide.md)).

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.name=toxinnote_hybrid \
    run.wandb_project=toxinnote
```

**What happens:**
1. Config is loaded and paths resolved
2. DDP process group is initialized (one process per GPU)
3. Label tokenizer and encoder (E5) are loaded
4. Graph index and archive are loaded
5. `ProteinStructureDataset` is created for train/val/test
6. `StructuralProteinEncoder` (ESM-C + EGNN) is initialized
7. `ProtNote` model is assembled and wrapped in DDP
8. Loss function and weighted sampler are configured
9. Training loop runs for `NUM_EPOCHS` epochs
10. Best checkpoint is saved based on `OPTIMIZATION_METRIC_NAME`
11. Final evaluation on validation and test sets

### 4.2 Supervised Training (Legacy ProteInfer Encoder)

For sequence-only training without structure data. There are two modes:

**4.2a. Train ProteInfer from scratch (default for `use_sequence_encoder`):**

All ProteInfer CNN weights are randomly initialized and trained end-to-end (141.6M trainable params).

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.use_sequence_encoder=true \
    run.name=protnote_cnn
```

**4.2b. Frozen pretrained ProteInfer (original ProtNote setup):**

Load pretrained ProteInfer weights and freeze them — only projection heads and output MLP are trained (75.8M trainable params). This requires the pretrained weights file at `data/models/proteinfer/GO_model_weights13703706.pkl` (configured via `PROTEINFER_GO_WEIGHTS_PATH` in `paths/default.yaml`).

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.full_path_name=FULL_DATA_PATH \
    run.use_sequence_encoder=true \
    params.PRETRAINED_SEQUENCE_ENCODER=true \
    params.TRAIN_SEQUENCE_ENCODER=false \
    run.name=protnote_cnn_frozen
```

| Mode | `PRETRAINED_SEQUENCE_ENCODER` | `TRAIN_SEQUENCE_ENCODER` | Trainable Params |
|------|-------------------------------|--------------------------|------------------|
| Train from scratch | `false` (default) | `true` (default) | 141.6M |
| Frozen pretrained | `true` | `false` | 75.8M |

### 4.3 Zero-Shot Training

Train on a label-disjoint split to evaluate zero-shot generalization:

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH_ZERO_SHOT \
    run.validation_path_name=VAL_DATA_PATH_ZERO_SHOT \
    run.test_paths_names='[TEST_DATA_PATH_ZERO_SHOT]' \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.name=zero_shot_experiment \
    params.EXTRACT_VOCABULARIES_FROM=null
```

Note: `EXTRACT_VOCABULARIES_FROM=null` ensures vocabularies are built from the training set rather than the full dataset (important for proper zero-shot evaluation).

### 4.4 Multi-GPU Training

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.gpus=4 \
    run.name=multi_gpu_experiment
```

For multi-node training, also set `run.nodes` and `run.nr` (or use AMLT with `run.amlt=true`).

### 4.5 Resume from Checkpoint

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    run.model_file=2024-01-15_toxinnote_best_val_metric.pt \
    run.from_checkpoint=true \
    run.name=resumed_experiment
```

The `model_file` path is relative to `data/models/ProtNote/`. Setting `from_checkpoint=true` restores the optimizer state and epoch counter.

### 4.6 Checkpoints

During training, the trainer saves checkpoints to `outputs/checkpoints/`:

| Checkpoint | Filename Pattern | When |
|------------|-----------------|------|
| Best metric | `{timestamp}_{name}_best_val_metric.pt` | When `OPTIMIZATION_METRIC_NAME` improves |
| Best loss | `{timestamp}_{name}_best_val_loss.pt` | When validation loss improves |
| Last epoch | `{timestamp}_{name}_last_epoch.pt` | After every epoch |
| Periodic | `{timestamp}_{name}_epoch_{N}.pt` | Every 10 epochs |

Each checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_metric`.

### 4.7 Logging

- **Console + file**: Logs are written to `outputs/logs/{timestamp}_{name}.log`
- **Weights & Biases**: Enabled by default (`run.wandb_project=toxinnote`). Set to `null` to disable.
- **MLFlow**: Set `run.mlflow=true` (requires `run.amlt=true`)

---

## 5. Evaluation and Testing

### 5.1 Evaluate a Trained Model

Run inference and compute metrics on a test set:

```bash
python bin/main.py \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    run.annotations_path_name=GO_ANNOTATIONS_PATH \
    run.base_label_embedding_name=GO_BASE_LABEL_EMBEDDING_PATH \
    run.save_prediction_results=true \
    run.name=eval_run
```

Include a validation set to automatically tune the decision threshold:

```bash
python bin/main.py \
    run.validation_path_name=VAL_DATA_PATH \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    run.save_prediction_results=true \
    params.DECISION_TH=null \
    run.name=eval_with_threshold_tuning
```

Setting `DECISION_TH=null` triggers threshold optimization on the validation set before testing.

### 5.2 Metrics

The following metrics are computed:

| Metric | Description |
|--------|-------------|
| `f1_micro` | F1 score, micro-averaged across all labels |
| `f1_macro` | F1 score, macro-averaged (mean per-label F1) |
| `f1_weighted` | F1 score, weighted by label support |
| `precision_micro/macro` | Precision |
| `recall_micro/macro` | Recall |
| `map_micro` | Mean average precision (micro) |
| `map_macro` | Mean average precision (macro) |
| `bce_loss` | Binary cross-entropy loss |
| `focal_loss` | Focal loss |

### 5.3 ProteInfer Baseline

Test the standalone ProteInfer CNN baseline (no text encoder):

```bash
python bin/test_proteinfer.py \
    --validation-path-name VAL_DATA_PATH \
    --test-paths-names TEST_DATA_PATH \
    --proteinfer-weights GO \
    --threshold 0.5
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--validation-path-name` | `None` | Config key for validation FASTA |
| `--test-paths-names` | `None` | Config key(s) for test FASTAs |
| `--train-path-name` | `None` | Config key for training FASTA |
| `--proteinfer-weights` | `GO` | Which model weights to use: `GO` or `EC` |
| `--threshold` | `0.5` | Decision threshold for predictions |
| `--model-weights-id` | `None` | Model variant ID (if multiple weight files exist) |
| `--name` | `ProteInfer` | Name for the run |
| `--annotations-path-name` | `GO_ANNOTATIONS_PATH` | Config key for annotations |
| `--base-label-embedding-name` | `GO_BASE_LABEL_EMBEDDING_PATH` | Config key for label embeddings |
| `--save-prediction-results` | off | Save predictions and ground truth |
| `--only-inference` | off | Predict without computing metrics |
| `--only-represented-labels` | off | Only predict labels represented in the dataset |
| `--override` | `None` | Override config parameters as key-value pairs |

### 5.4 BLAST Baseline

Run sequence similarity baseline using BLAST:

```bash
python bin/run_blast.py \
    --test-data-path data/sequences/uniprot_sprot_2025_01_with_struct/splits/test_GO.fasta
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--test-data-path` | — | Path to the test FASTA file (required) |
| `--train-data-path` | `TRAIN_DATA_PATH` from config | Path to the training FASTA (used as BLAST database) |
| `--top-k-hits` | `1` | Number of top hits per query |
| `--max-evalue` | `0.05` | E-value threshold (hits above this are dropped) |
| `--cache` | off | Use cached results if available |
| `--save-runtime-info` | off | Save search/parse duration to CSV |

### 5.5 Batch Model Testing

Test multiple models across pre-defined dataset configurations:

```bash
python bin/test_models.py \
    --model-files model1.pt model2.pt \
    --test-paths-names TEST_DATA_PATH_ZERO_SHOT TEST_DATA_PATH
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model-files` | — | List of `.pt` model checkpoints to test (required) |
| `--test-paths-names` | all defined tests | List of test path names (keys from `TEST_COMMANDS` dict) |
| `--test-type` | `all` | What to run: `all`, `baseline`, or `model` |
| `--save-prediction-results` | off | Save predictions for each test |
| `--save-embeddings` | off | Save embeddings for each test |
| `--save-val-test-metrics` | off | Append metrics to JSON file |
| `--save-val-test-metrics-file` | `val_test_metrics.json` | JSON file for accumulated metrics |

### 5.6 Save Predictions

Predictions are saved as HDF5 files in `outputs/results/`:

```
outputs/results/{data_loader_name}_results_{run_name}.h5
```

Each file contains:
- `sequence_ids`: Protein identifiers
- `logits`: Raw model outputs
- `labels`: Ground truth multi-hot vectors
- `label_vocabulary`: Ordered label list

---

## 6. Inference

### 6.1 Inference on New Proteins

To run inference on a custom FASTA file:

1. Add the FASTA path to `configs/paths/default.yaml` under `data_paths`:
   ```yaml
   MY_CUSTOM_DATA: path/to/my_proteins.fasta
   ```

2. If using the hybrid encoder, generate ESM-C embeddings and atom graphs for the new proteins (see [Data Preparation Guide](data_guide.md), Sections 9 and 10).

3. Run inference:
   ```bash
   python bin/main.py \
       run.test_paths_names='[MY_CUSTOM_DATA]' \
       run.model_file=best_model.pt \
       run.save_prediction_results=true \
       run.name=custom_inference
   ```

### 6.2 FASTA Format

Input FASTA files should have GO/EC annotations in the header:

```
>SEQUENCE_ID GO:0006412 GO:0003735
MAKQKTEVVRIVGRPFAYTLKDSQAKLR...
```

For inference-only (no ground truth), annotations in headers are optional.

---

## 7. Advanced Topics

### 7.1 Model Architecture Overview

ToxinNote uses a two-tower architecture:

```
Protein branch                         Text branch
──────────────                         ───────────
Structure + Sequence                   GO term description
        │                                      │
   ESM-C (frozen)                    Multilingual E5
   960-dim/residue                   (frozen + LoRA)
        │                                      │
   + atom-type one-hot (37)                    │
   = 997-dim/atom                              │
        │                                      │
   EGNN (4 layers)                             │
   256-dim/atom                                │
        │                                      │
   Global pooling                              │
   → 1100-dim                          → 1024-dim
        │                                      │
   Projection head                    Projection head
   → 1024-dim                         → 1024-dim
        │                                      │
        └──────── Concatenation ───────────────┘
                        │
                   Output MLP (3 layers)
                        │
                   sigmoid → probability
```

### 7.2 Feature Fusion Options

| Method | Description |
|--------|-------------|
| `concatenation` | `[protein_emb, label_emb]` → MLP (default) |
| `concatenation_diff` | `[protein_emb, label_emb, protein_emb - label_emb]` → MLP |
| `concatenation_prod` | `[protein_emb, label_emb, protein_emb * label_emb]` → MLP |
| `similarity` | Cosine similarity / temperature |

### 7.3 Loss Functions

| Loss | When to Use |
|------|-------------|
| `FocalLoss` | Default. Handles extreme class imbalance in GO annotation |
| `BCE` | Standard binary cross-entropy baseline |
| `WeightedBCE` | Per-label weighting by inverse frequency |
| `CBLoss` | Class-balanced loss for long-tailed distributions |
| `RGDBCE` | Reweighted gradient descent BCE |

### 7.4 Data Augmentation

**Sequence augmentation** (`AUGMENT_RESIDUE_PROBABILITY=0.1`): Each residue has a 10% chance of being substituted according to BLOSUM62 substitution matrix probabilities. Only applied during training.

**Label embedding noising** (`LABEL_EMBEDDING_NOISING_ALPHA=20.0`): Additive Gaussian noise is applied to label embeddings during training, scaled by `alpha / sqrt(embedding_dim)`. This regularizes the text encoder pathway.

### 7.5 Interpretability

ToxinNote supports residue-level interpretability via Integrated Gradients:

```bash
python bin/main.py \
    run.test_paths_names='[TEST_DATA_PATH]' \
    run.model_file=best_model.pt \
    params.INTERPRETABILITY_METHOD=IntegratedGradient \
    params.INTERPRETABILITY_WEIGHT=0.1 \
    params.INTERPRETABILITY_N_STEPS=50
```

This computes gradient-input attributions and can be trained jointly with a site ground truth MSE loss.

### 7.6 Rapid Prototyping

Use subset fractions to train on smaller data for quick iteration:

```bash
python bin/main.py \
    run.train_path_name=TRAIN_DATA_PATH \
    run.validation_path_name=VAL_DATA_PATH \
    params.TRAIN_SUBSET_FRACTION=0.1 \
    params.VALIDATION_SUBSET_FRACTION=0.1 \
    params.NUM_EPOCHS=5 \
    run.name=quick_test
```

### 7.7 Common Override Recipes

**Enable dynamic batching by atom count (prevents OOM from large proteins):**
```bash
params.MAX_ATOMS_PER_BATCH=15000
# Batch size varies per batch — small proteins get larger batches, large proteins get smaller ones
# Requires atom-level mode with n_atoms in graph_index.json (produced by prepare_graph_data.py)
```

**Increase batch size with gradient accumulation (for limited GPU memory):**
```bash
params.TRAIN_BATCH_SIZE=2 params.GRADIENT_ACCUMULATION_STEPS=4
# Effective batch size = 2 * 4 = 8
```

**Disable weighted sampling:**
```bash
params.WEIGHTED_SAMPLING=false
```

**Use BioGPT as label encoder:**
```bash
params.LABEL_ENCODER_CHECKPOINT=microsoft/biogpt
```

**Unfreeze label encoder layers:**
```bash
params.LABEL_ENCODER_NUM_TRAINABLE_LAYERS=2 params.LORA=false
```

**Use learned AA embeddings instead of ESM-C (for de novo proteins):**
```bash
params.USE_PLM_EMBEDDINGS=false params.LEARNED_AA_EMBEDDING_DIM=128
```
