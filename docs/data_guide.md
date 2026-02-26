# Data Preparation Guide

This guide covers the complete data preparation pipeline for ToxinNote — from downloading pre-processed data to building atom-level graphs for the structural encoder.

For training, evaluation, and inference, see the [Training Guide](training_guide.md).

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Download Pre-processed Data (Quickstart)](#3-download-pre-processed-data-quickstart)
4. [Create Datasets from Raw Sources](#4-create-datasets-from-raw-sources)
5. [Split Dataset into Train/Val/Test](#5-split-dataset-into-trainvaltest)
6. [Generate Parenthood JSON](#6-generate-parenthood-json)
7. [Download Protein Structures](#7-download-protein-structures)
8. [Generate Label Embeddings](#8-generate-label-embeddings)
9. [Generate ESM-C Sequence Embeddings](#9-generate-esm-c-sequence-embeddings)
10. [Build Atom-Level Graphs](#10-build-atom-level-graphs)
11. [Summary](#11-summary)

---

## 1. Prerequisites

- **Software**: Python 3.10+, [Pixi](https://pixi.sh/) package manager
- **Platform**: Linux, macOS (Intel or Apple Silicon), or Windows for steps 3–7. Linux (x86_64) with NVIDIA GPU for steps 8–10.
- **Hardware**: No GPU required for steps 3–7. Steps 8–10 (embedding generation and graph building) require an NVIDIA GPU.

## 2. Environment Setup

The project provides a lightweight, cross-platform `dataprep` environment that excludes GPU-dependent packages (PyTorch, CUDA, torch-geometric, etc.). It runs on Linux, macOS, and Windows, making it suitable for data downloading and preprocessing on any machine:

```bash
# Install and activate the dataprep environment (no GPU deps, cross-platform)
pixi shell -e dataprep
```

For steps that require a GPU (label embeddings, ESM-C embeddings, graph building), use the default environment (Linux only):

```bash
# Install and activate the full environment (includes GPU deps, Linux only)
pixi shell
```

## 3. Download Pre-processed Data (Quickstart)

```bash
pixi run dataprep
```

This downloads the complete dataset from HuggingFace (`cs527-toxinnote/protnote_data`) into the `data/` directory, including:
- FASTA files (train/val/test splits)
- GO/EC annotations
- Pre-computed label embeddings
- ProteInfer model weights
- Vocabularies

After this step you can skip directly to [Section 9](#9-generate-esm-c-sequence-embeddings) if using the hybrid encoder, or go straight to training (see [Training Guide](training_guide.md)) if using the legacy ProteInfer encoder.

## 4. Create Datasets from Raw Sources

If you need to build datasets from scratch:


**From SwissProt releases:**

```bash
python bin/make_dataset_from_swissprot.py \
    --latest-swissprot-file "path to swissprot .dat" \
    --parsed-latest-swissprot-file "path to parsed SwissProt .pkl or new .pkl" \
    --output-file-path "output path to .fasta" \
    --annotations-file "annotations .pkl in data/annotations/" \
    --parenthood-file "parenthood .json in data/vocabularies/" \
    --label-vocabulary "new | all | proteinfer" \
    --sequence-vocabulary "new | all | proteinfer_test | proteinfer_train"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--latest-swissprot-file` | — | Path to the SwissProt `.dat` file (relative to `data/swissprot/`) |
| `--output-file-path` | — | Output FASTA file path |
| `--annotations-file` | — | Annotations pickle in `data/annotations/` (e.g., `go_annotations_jul_2024.pkl`) |
| `--parenthood-file` | — | Parenthood JSON in `data/vocabularies/` for GO term hierarchy |
| `--parsed-latest-swissprot-file` | — | Cached parsed SwissProt pickle in `data/swissprot/` (created automatically on first run) |
| `--label-vocabulary` | — | Label set to use: `proteinfer` (original vocab), `all` (all observed terms), `new` (terms added since baseline) |
| `--sequence-vocabulary` | — | Sequence set: `all`, `new` (exclude train/val seqs), `proteinfer_test`, `proteinfer_train` |
| `--baseline-annotations-file` | `None` | Baseline annotations pickle for computing new labels (required when `--label-vocabulary=new`) |
| `--only-leaf-nodes` | off | Restrict to leaf nodes of the GO/EC hierarchy |
| `--keywords` | `None` | Filter SwissProt records by keywords (e.g., `Toxin`) |
| `--structure-filter` | `none` | Filter by structure availability: `none`, `any`, `pdb`, `afdb` |
| `--min-plddt` | `None` | Minimum average pLDDT for AFDB entries (requires `--plddt-file`) |
| `--plddt-file` | `None` | Path to pLDDT values: JSON or pickle with `seq_id` + `afdb_avg_plddt` columns |
| `--no-cache` | off | Re-parse SwissProt from scratch instead of using cached pickle |

**For zero-shot evaluation (label-disjoint splits):**

```bash
python bin/make_zero_shot_datasets_from_proteinfer.py
```

Creates 80/10/10 train/val/test splits where labels are disjoint across splits.

**From ProteInfer TFRecords:**

```bash
python bin/make_proteinfer_dataset.py \
    --dataset-type random \
    --annotation-types GO
```

This converts TFRecord files into FASTA format with GO annotations in the sequence headers.

## 5. Split Dataset into Train/Val/Test

After creating a full FASTA dataset (e.g., from SwissProt), split it into train/dev/test sets:

```bash
# Using a config key from paths/default.yaml
python bin/split_dataset.py \
    --input-path-name "config name for full-set .fasta path" \
    --output-dir "output path to train_*.fasta, dev_*.fasta, test_*.fasta" \
    --prefix "EC | GO | any label prefix"

# Using a direct file path
python bin/split_dataset.py \
    --input-path "path to full-set .fasta file" \
    --output-dir "output path to train_*.fasta, dev_*.fasta, test_*.fasta" \
    --prefix "EC | GO | any label prefix"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--input-path-name` | — | Config key from `paths/default.yaml` (mutually exclusive with `--input-path`) |
| `--input-path` | — | Direct path to input FASTA file (mutually exclusive with `--input-path-name`) |
| `--output-dir` | — | Output directory for split files (relative to project root or absolute) |
| `--prefix` | — | Label prefix for output filenames (e.g., `EC` → `train_EC.fasta`) |
| `--train-ratio` | `0.8` | Training set fraction |
| `--val-ratio` | `0.1` | Validation set fraction |
| `--test-ratio` | `0.1` | Test set fraction |
| `--seed` | `42` | Random seed for reproducibility |

Output: `train_{prefix}.fasta`, `dev_{prefix}.fasta`, `test_{prefix}.fasta` in the specified output directory.

## 6. Generate Parenthood JSON

Generate the GO/EC label hierarchy used for label normalization during evaluation:

```bash
# Auto-detect source files (latest .obo + enzyme files in data/annotations/)
python bin/generate_parenthood.py

# Or specify files explicitly
python bin/generate_parenthood.py \
    --go-obo "path to gene ontology .obo" \
    --enzyme-dat "path to enzyme commission number .dat" \
    --enzclass-txt "path to enzyme commission number class .txt" \
    --output "output path to .json vocabulary"
```

This replaces the proteinfer `parenthood_bin.py` dependency. Source files needed:
- **GO**: A `.obo` file — download with `python bin/download_GO_annotations.py`
- **EC**: `enzyme.dat` + `enzclass.txt` from [ExPASy](https://ftp.expasy.org/databases/enzyme/) (optional, use `--skip-ec` for GO only)

Output: `data/vocabularies/parenthood_{date}.json` — referenced in config as `PARENTHOOD_LIB_PATH`.

## 7. Download Protein Structures

Required only if you need structure files for new proteins not already in `data/structures/`.

```bash
python bin/download_structures.py \
    "path to .pkl file generated by @bin/make_dataset_from_swissprot"
```

Downloads CIF/PDB files from AlphaFold DB or RCSB PDB into `data/structures/`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `uniprot_pkl_file` | — | Positional. Path to UniProt pickle with `seq_id`, `struct_expr`, `struct_afdb` columns |
| `--alphafolddb` | off | Download only AlphaFoldDB structures (skip PDB lookup) |
| `--override` | off | Re-download and overwrite existing structure files |

Output: CIF files in `data/structures/pdb/` and `data/structures/alphafolddb/`, plus `data/structures/structure_index.json`.

> Note: if you need to prepare structures in large quantity, download directly from [AlphaFoldDB Compression](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/) and prepare the graph using `--local-afdb` option in Step 10.

## 8. Generate Label Embeddings

> **Requires GPU.** Use the default environment (`pixi shell`).

Pre-compute text embeddings for all GO term descriptions using Multilingual E5:

```bash
python bin/generate_label_embeddings.py \
    --add-instruction \
    --account-for-sos
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pooling-method` | `mean` | Embedding pooling: `mean`, `last_token`, or `all` |
| `--label-encoder-checkpoint` | `intfloat/multilingual-e5-large-instruct` | HuggingFace model |
| `--base-label-embedding-path` | `GO_BASE_LABEL_EMBEDDING_PATH` | Config key for output path |
| `--annotations-path-name` | `GO_ANNOTATIONS_2019_UPDATED_PATH` | Config key for annotations. **Note**: This default is from the original ProtNote codebase — in this fork, use `GO_ANNOTATIONS_PATH` instead. |
| `--add-instruction` | off | Format input for instruction-tuned model |
| `--account-for-sos` | off | Ignore SOS token in pooling |

**For EC labels:**

```bash
python bin/generate_label_embeddings.py \
    --base-label-embedding-path EC_BASE_LABEL_EMBEDDING_PATH \
    --annotations-path-name EC_ANNOTATIONS_PATH \
    --add-instruction --account-for-sos
```

Output: `data/embeddings/frozen_E5_multiling_inst_label_embeddings_mean.pt` (+ corresponding index file).

## 9. Generate ESM-C Sequence Embeddings

> **Requires GPU.** Use the default environment (`pixi shell`).

Required for the hybrid (ESM-C + EGNN) encoder. Generates per-residue embeddings (960-dim) for each protein:

```bash
python bin/generate_sequence_embeddings.py \
    --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH \
    --batch-size 16
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fasta-path-names` | `TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH` | Config key names for FASTA files |
| `--model-name` | config `ESMC_MODEL_NAME` (`esmc_300m`) | ESM-C model variant |
| `--batch-size` | `8` (fallback; overridable via `ESMC_BATCH_SIZE` in params config) | Inference batch size |
| `--max-sequence-length` | `5000` | Skip sequences longer than this |

Output: Individual `.pt` files in `data/embeddings/esmc/` plus `esmc_index.json`.

## 10. Build Atom-Level Graphs

> **Requires GPU.** Use the default environment (`pixi shell`).

Combines structure files + ESM-C embeddings into atom-level graphs for the EGNN encoder:

```bash
python bin/prepare_graph_data.py \
    --fasta-path-names TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH \
    --num-workers 8 \
    --knn-k 20
```

**What it does:**
1. Parses CIF/PDB structure files (via AtomWorks)
2. Loads pre-computed ESM-C per-residue embeddings
3. Builds atom-level graphs (covalent bonds + k-NN spatial edges)
4. Aligns ESM-C residue embeddings to structure atoms
5. Saves as individual `.pt` graphs and consolidates into a `.pngrph` archive

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--fasta-path-names` | `TRAIN_DATA_PATH VAL_DATA_PATH TEST_DATA_PATH` | Config key names for FASTA files (used for sequence lookup) |
| `--knn-k` | config `KNN_K` (`20`) | Number of k-nearest-neighbor edges per atom |
| `--num-workers` | `min(8, cpu_count)` | Parallel processing workers (use `1` for sequential/debug mode) |
| `--chunksize` | `10` | Chunk size for multiprocessing |
| `--no-consolidate` | off | Skip consolidation into a single archive file (consolidation is on by default) |
| `--num-shards` | `1` | Number of archive shards (e.g., `16` for large datasets) |
| `--keep-individual-files` | off | Keep individual `.pt` files after archiving (default: delete them) |
| `--timeout` | `300` | Per-protein timeout in seconds (`0` to disable) |
| `--local-afdb` | off | Use a local folder of AFDB PDB files instead of downloaded CIF files |

**Using local AlphaFoldDB structures:**

If you have a local copy of AlphaFold DB structure files (e.g., from a bulk download), you can use them instead of individually downloaded CIF files. Set the config keys in `configs/paths/default.yaml`:

```yaml
LOCAL_AFDB_DIR: structures/local_afdb/    # folder containing the PDB files
LOCAL_AFDB_SUFFIX: v4                      # model version suffix
```

Then pass `--local-afdb`:

```bash
python bin/prepare_graph_data.py \
    --num-workers 8 \
    --local-afdb
```

Files are expected to be named `AF-<UNIPROT_ID>-F1-model_<SUFFIX>.pdb` (e.g., `AF-P12345-F1-model_v4.pdb`). This only affects proteins whose structure index entry has `"source": "alphafolddb"` — PDB-sourced structures are resolved normally.

Output: `data/processed/graphs.pngrph` (archive) + `data/processed/graph_index.json`.

**Graph index format**: Each entry in `graph_index.json` is `{"filename": "SEQ_ID.pt", "n_atoms": 1234}`. The `n_atoms` field is used by the dynamic batch sampler to group proteins by atom count.

## 11. Summary

| Step | Script | Environment | Required For | Output |
|------|--------|-------------|--------------|--------|
| Download data | `pixi run dataprep` | `dataprep` | All | `data/` directory |
| Create datasets | `make_proteinfer_dataset.py` | `dataprep` | Custom datasets | FASTA files |
| Split dataset | `split_dataset.py` | `dataprep` | Custom datasets | `train_*.fasta`, `dev_*.fasta`, `test_*.fasta` |
| Parenthood JSON | `generate_parenthood.py` | `dataprep` | Evaluation (label normalization) | `data/vocabularies/parenthood_*.json` |
| Download structures | `download_structures.py` | `dataprep` | Hybrid encoder | `data/structures/` |
| Label embeddings | `generate_label_embeddings.py` | `default` | All | `data/embeddings/*.pt` |
| ESM-C embeddings | `generate_sequence_embeddings.py` | `default` | Hybrid encoder | `data/embeddings/esmc/*.pt` |
| Atom graphs | `prepare_graph_data.py` | `default` | Hybrid encoder | `data/processed/graphs.pngrph` |
