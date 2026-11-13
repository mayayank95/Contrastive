#!/usr/bin/env bash
# Frozen compact SALAD (544-d, dino_salad_512_32) kappa head for the paper tables.
# Same SF-XL place batches as MixVPR: 24×20, min 20, batch leave-one-out vMF.
# VAR_HEAD=agg (default): randomly initialized SALAD aggregator copy. Its forward
# returns the pre-L2 descriptor. Retrieval stays intra-normalized.
# VAR_HEAD=desc: no aggregator. Linear(544, 1) on the pre-L2 token and cluster sum.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SALAD_CKPT="${SALAD_CKPT:?Set SALAD_CKPT to the released dino_salad_512_32 checkpoint}"
VAR_HEAD="${VAR_HEAD:-agg}"

case "${VAR_HEAD}" in
  agg)
    VAR_HEAD_ARGS=(--var_head_type agg)
    RUN_TAG="salad544-sfxl-freeze-agg-random"
    ;;
  desc)
    VAR_HEAD_ARGS=(--var_head_type descriptor)
    RUN_TAG="salad544-sfxl-freeze-desc"
    ;;
  *)
    echo "ERROR: VAR_HEAD must be agg or desc, got ${VAR_HEAD}" >&2
    exit 1
    ;;
esac

exec python train.py \
  --config configs/datasets_salad.json \
  --device cuda \
  --freeze_model \
  --freeze_batchnorm \
  --method salad \
  --model_mode uncertainty \
  --descriptors_dimension 544 \
  --resume_model "${SALAD_CKPT}" \
  "${VAR_HEAD_ARGS[@]}" \
  --var_head_linear 1 \
  --variance_activation softplus \
  --image_size 322 \
  --num_workers 4 \
  --use_labels \
  --use_wandb \
  --wandb_run_name "${WANDB_RUN_NAME:-${RUN_TAG}-24x20}"
