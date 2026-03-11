#!/usr/bin/env bash
# Evaluate ProteNote models on MPNN-designed toxin sequences.
#
# Usage:
#   # Hybrid model (ESM-C + EGNN) — requires structure data prepared via paths=mpnn_toxin
#   bash bin/eval_mpnn_toxin.sh <model_file>
#
#   # Sequence-only model (ProteInfer CNN)
#   bash bin/eval_mpnn_toxin.sh <model_file> --seq-only
#
# Examples:
#   bash bin/eval_mpnn_toxin.sh 2026-01-18_22-19-56_protnote_toxin_best_val_metric.pt
#   bash bin/eval_mpnn_toxin.sh 2026-01-18_22-19-56_protnote_toxin_best_val_metric.pt --seq-only

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_file> [--seq-only]"
    echo "  model_file: checkpoint filename in data/models/ProtNote/"
    exit 1
fi

MODEL_FILE="$1"
# Extract just the filename (no directory) and strip .pt for use in run.name
MODEL_NAME="$(basename "${MODEL_FILE}" .pt)"
SEQ_ONLY=false

if [ "${2:-}" = "--seq-only" ]; then
    SEQ_ONLY=true
fi

COMMON_OVERRIDES=(
    "paths=mpnn_toxin"
    "run.train_path_name=null"
    "run.validation_path_name=null"
    "run.test_paths_names=[TEST_DATA_PATH]"
    "run.model_file=${MODEL_FILE}"
    "run.save_prediction_results=true"
    "run.save_val_test_metrics=true"
    "run.name=mpnn_toxin_eval_${MODEL_NAME}"
    "params.EXTRACT_VOCABULARIES_FROM=null"
    "params.DECISION_TH=0.5"
    "params.ESTIMATE_MAP=false"
    "params.OPTIMIZATION_METRIC_NAME=f1_macro"
    "params.TEST_BATCH_SIZE=8"
)

if [ "$SEQ_ONLY" = true ]; then
    echo "=== Evaluating sequence-only model on MPNN toxin test set ==="
    pixi run python bin/main.py \
        "${COMMON_OVERRIDES[@]}" \
        "run.use_sequence_encoder=true"
else
    echo "=== Evaluating hybrid model on MPNN toxin test set ==="
    pixi run python bin/main.py \
        "${COMMON_OVERRIDES[@]}"
fi
