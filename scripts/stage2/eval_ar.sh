#!/bin/bash
# Streaming rgb2thermal eval of a stage-2 checkpoint (EMA weights) on the 81f val latents; with no checkpoint it
# evaluates the exported stage-1 teacher run causally. Writes mp4s (RGB | prediction | real) + metrics.json.
#   source scripts/env.sh && bash scripts/stage2/eval_ar.sh <out_dir> [ckpt.pt] [--steps 30] [--n 8]
# Bidirectional reference (the teacher as one block): INFER_CFG=configs/thermal/stage2_infer_bidir.yaml
set -euo pipefail
OUT=${1:?out dir}; shift; CKPT=""
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then CKPT=$1; shift; fi
export LLV2_MODEL_DIR=${LLV2_MODEL_DIR:-$THERMAL_RUNS/stage2_teacher}
export LLV2_PROMPT_EMBEDS=${LLV2_PROMPT_EMBEDS:-$LLV2_MODEL_DIR/prompt_embeds.pt}
export THERMAL_V2V_REPO=$REPO
cd "$LONGLIVE_DIR"
ARGS=(--config ${INFER_CFG:-configs/thermal/stage2_infer.yaml} --latents ${VAL_LATENTS:-$THERMAL_DATA/latents/tartanrgbt_81f/val} --out "$OUT")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT" --use_ema)
exec $PY_LL thermal_eval.py "${ARGS[@]}" "$@"
