#!/bin/bash
# shared launcher for S7 / S8 (LongLive DMD trainer). Sourced by train_dmd.sh / train_long.sh with RUN and CFG set.
set -euo pipefail
: "${LONGLIVE_DIR:?source scripts/env.sh first}"
LOGDIR=$THERMAL_RUNS/$RUN; mkdir -p "$LOGDIR"
export LLV2_MODEL_DIR=${LLV2_MODEL_DIR:-$THERMAL_RUNS/stage2_teacher}           # real score / critic = exported teacher
export LLV2_PROMPT_EMBEDS=${LLV2_PROMPT_EMBEDS:-$LLV2_MODEL_DIR/prompt_embeds.pt}
export THERMAL_V2V_REPO=$REPO WANDB_DIR=$LOGDIR
export THERMAL_LONG_VAL=${THERMAL_LONG_VAL:-$( (ls "$THERMAL_DATA"/latents/tartanrgbt_long/val/*.safetensors 2>/dev/null || true) | head -1)}
NPROC=${NPROC:-$(bash "$REPO/scripts/_ngpu.sh")}; NNODES=${NNODES:-1}
if [ "$NNODES" -gt 1 ]; then
  RDZV=(--nnodes "$NNODES" --node_rank "${NODE_RANK:?}" --master_addr "${MASTER_ADDR:?}" --master_port "${MASTER_PORT:-29500}")
else
  RDZV=(--standalone --nnodes 1)
fi
cd "$LONGLIVE_DIR"
cp -f "$CFG" "$LOGDIR/config.yaml"
$PY_LL -m torch.distributed.run "${RDZV[@]}" --nproc_per_node "$NPROC" \
  train.py --config_path "$CFG" --logdir "$LOGDIR" "$@" 2>&1 | tee -a "$LOGDIR/console.log"
exit ${PIPESTATUS[0]}
