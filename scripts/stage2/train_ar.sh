#!/bin/bash
# Stage 2: causal AR teacher forcing (rgb2thermal) on LongLive, initialised from the exported stage-1 teacher.
#   source scripts/env.sh && bash scripts/stage2/train_ar.sh [run_name] [extra LongLive train.py args]
# Needs: scripts/stage2/setup_longlive.sh, the 81f latents (scripts/data/encode_all.sh stage2) and
#        scripts/stage2/export_teacher.py -> $THERMAL_RUNS/stage2_teacher. Resumes from the logdir.
# Multi-GPU / multi-node: NPROC, NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT as in stage 1 (FSDP).
set -euo pipefail
: "${LONGLIVE_DIR:?source scripts/env.sh first}"
RUN=${1:-stage2_ar}; shift || true
LOGDIR=$THERMAL_RUNS/$RUN; mkdir -p "$LOGDIR"
export LLV2_MODEL_DIR=${LLV2_MODEL_DIR:-$THERMAL_RUNS/stage2_teacher}
export LLV2_PROMPT_EMBEDS=${LLV2_PROMPT_EMBEDS:-$LLV2_MODEL_DIR/prompt_embeds.pt}
export THERMAL_V2V_REPO=$REPO WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$LOGDIR
CFG=${STAGE2_CONFIG:-configs/thermal/stage2_ar.yaml}
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}; NNODES=${NNODES:-1}
if [ "$NNODES" -gt 1 ]; then
  RDZV=(--nnodes "$NNODES" --node_rank "${NODE_RANK:?}" --master_addr "${MASTER_ADDR:?}" --master_port "${MASTER_PORT:-29500}")
else
  RDZV=(--standalone --nnodes 1)
fi
cd "$LONGLIVE_DIR"
cp -f "$CFG" "$LOGDIR/config.yaml"
exec $PY_LL -m torch.distributed.run "${RDZV[@]}" --nproc_per_node "$NPROC" \
  train.py --config_path "$CFG" --logdir "$LOGDIR" --disable-wandb "$@"
