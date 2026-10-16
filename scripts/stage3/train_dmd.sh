#!/bin/bash
# S7 -> model ⑤: 4-step streaming rgb2thermal by self-forcing DMD (fresh LoRA on the S6 AR model, teacher = exported
# stage-1 teacher). Config: configs/thermal/stage3_dmd.yaml (inside the LongLive checkout).
#   source scripts/env.sh
#   STAGE2_CKPT=$THERMAL_RUNS/stage2_ar/snapshots/step_XXXXXX.pt bash scripts/stage3/train_dmd.sh [run_name]
# Resumes from the logdir. In-training evals (short val clips + the 4-min val stream) -> <logdir>/eval, wandb.
RUN=${1:-stage3_dmd}; shift || true
: "${STAGE2_CKPT:?set STAGE2_CKPT to the chosen S6 checkpoint (EMA snapshot or checkpoint_model_*/model.pt)}"
export STAGE2_CKPT
CFG=${STAGE3_CONFIG:-configs/thermal/stage3_dmd.yaml}
source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh" "$@"
