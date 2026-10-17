#!/bin/bash
# Evaluate a ⑤ / ⑥ checkpoint (4-step streaming; EMA LoRA when saved) on the 81f val clips and the 4-min val stream.
#   source scripts/env.sh
#   STAGE2_CKPT=<S6 checkpoint> bash scripts/stage3/eval.sh <S7|S8 checkpoint_model_*/model.pt> <out_dir> [long]
# "long" = model ⑥: the stream is generated with the 1-pair frame sink it was trained with.
set -euo pipefail
CK=${1:?S7/S8 checkpoint}; OUT=${2:?out dir}; MODE=${3:-short}
: "${STAGE2_CKPT:?set STAGE2_CKPT to the S6 checkpoint the LoRA was trained on}"
CFG=configs/thermal/stage3_infer.yaml; [ "$MODE" = long ] && CFG=configs/thermal/stage3_infer_long.yaml
LONG=${THERMAL_LONG_VAL:-$( (ls "$THERMAL_DATA"/latents/tartanrgbt_long/val/*.safetensors 2>/dev/null || true) | head -1)}
INFER_CFG=$CFG bash "$REPO/scripts/stage2/eval_ar.sh" "$OUT/val_clips" "$STAGE2_CKPT" --n ${NUM_CLIPS:-8} --steps 4 --lora_ckpt "$CK"
if [ -n "$LONG" ]; then
  INFER_CFG=$CFG bash "$REPO/scripts/stage2/eval_ar.sh" "$OUT/long_stream" "$STAGE2_CKPT" --files "$LONG" --steps 4 --lora_ckpt "$CK"
else
  echo "[stage3/eval] no long val stream (scripts/data/encode_all.sh stage3) -> short clips only"
fi
