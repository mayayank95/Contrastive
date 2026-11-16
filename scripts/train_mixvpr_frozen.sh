#!/usr/bin/env bash
# Frozen MixVPR-512 kappa head for the paper tables.
# SF-XL place batches, 24×20, min 20, batch leave-one-out vMF.
# The aggregator inside the kappa head stays at its build-time initialization.
# USE_WANDB=1 turns on Weights & Biases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MIXVPR_CKPT="${MIXVPR_CKPT:?Set MIXVPR_CKPT to the released MixVPR-512 checkpoint}"

ARGS=(
  --config configs/datasets_mixvpr.json
  --device cuda
  --freeze_model
  --freeze_batchnorm
  --method mixvpr
  --model_mode uncertainty
  --descriptors_dimension 512
  --resume_model "${MIXVPR_CKPT}"
  --var_head_type agg
  --var_head_linear 1
  --variance_activation softplus
  --image_size 320
  --num_workers 4
  --use_labels
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  ARGS+=(--use_wandb --wandb_run_name "${WANDB_RUN_NAME:-mixvpr512-sfxl-freeze-agg-random-24x20}")
fi

exec python train.py "${ARGS[@]}"
