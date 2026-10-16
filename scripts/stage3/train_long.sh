#!/bin/bash
# S8 -> model ⑥: long-rollout tuning of the S7 LoRA (student rolls far past the window; teacher scores the last one).
# Config: configs/thermal/stage3_long.yaml. Needs the 481-frame latents (scripts/data/encode_all.sh stage3).
#   source scripts/env.sh
#   STAGE2_CKPT=<same as S7> STAGE3_LORA=$THERMAL_RUNS/stage3_dmd/checkpoint_model_XXXXXX/model.pt \
#     bash scripts/stage3/train_long.sh [run_name]
RUN=${1:-stage3_long}; shift || true
: "${STAGE2_CKPT:?set STAGE2_CKPT (the S6 checkpoint S7 started from)}"
: "${STAGE3_LORA:?set STAGE3_LORA to the chosen S7 checkpoint_model_*/model.pt}"
export STAGE2_CKPT STAGE3_LORA
CFG=${STAGE3_CONFIG:-configs/thermal/stage3_long.yaml}
source "$(dirname "${BASH_SOURCE[0]}")/_launch.sh" "$@"
