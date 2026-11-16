#!/usr/bin/env bash
# Frozen compact SALAD (544-d, dino_salad_512_32) kappa head for the paper tables.
# Same SF-XL place batches as MixVPR: 24×20, min 20, batch leave-one-out vMF.
# Descriptor head: no aggregator. Linear(544, 1) on the pre-L2 token and cluster sum.
# Early stopping watches validation ECE@1 with patience 15.
# Retrieval stays intra-normalized. USE_WANDB=1 turns on Weights & Biases.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SALAD_CKPT="${SALAD_CKPT:?Set SALAD_CKPT to the released dino_salad_512_32 checkpoint}"
RUN_TAG="salad544-sfxl-freeze-desc"

ARGS=(
  --config configs/datasets_salad.json
  --device cuda
  --freeze_model
  --freeze_batchnorm
  --method salad
  --model_mode uncertainty
  --descriptors_dimension 544
  --resume_model "${SALAD_CKPT}"
  --var_head_linear 1
  --variance_activation softplus
  --image_size 322
  --num_workers 4
  --use_labels
)

if [[ "${USE_WANDB:-0}" == "1" ]]; then
  ARGS+=(--use_wandb --wandb_run_name "${WANDB_RUN_NAME:-${RUN_TAG}-24x20}")
fi

exec python train.py "${ARGS[@]}"
