#!/bin/bash
# Stage 1: bidirectional RGB <-> thermal teacher (configs/stage1/bidir_33f.yaml), torchrun on 1..N nodes.
#   source scripts/env.sh && bash scripts/stage1/train.sh [key=value overrides...]
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT on every node (slurm / k8s provide them).
# Resumes automatically from $THERMAL_RUNS/stage1_bidir_33f/ckpt. Smoke test: see scripts/smoke_test.sh.
set -euo pipefail
: "${THERMAL_RUNS:?source scripts/env.sh first}"
cd "$REPO"
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}; NNODES=${NNODES:-1}
if [ "$NNODES" -gt 1 ]; then
  RDZV=(--nnodes "$NNODES" --node_rank "${NODE_RANK:?}" --master_addr "${MASTER_ADDR:?}" --master_port "${MASTER_PORT:-29500}")
else
  RDZV=(--standalone --nnodes 1)
fi
# console log appended under the run dir (train_log.jsonl, eval/eval_log.jsonl and eval/*.mp4 come from the trainer;
# wandb: online with WANDB_API_KEY / `wandb login`, else offline in <run>/wandb)
LOG=${STAGE1_LOG:-$THERMAL_RUNS/stage1_bidir_33f/console.log}; mkdir -p "$(dirname "$LOG")"
$PY -m torch.distributed.run "${RDZV[@]}" --nproc_per_node "$NPROC" \
  src/train_icl.py --config ${STAGE1_CONFIG:-configs/stage1/bidir_33f.yaml} "$@" 2>&1 | tee -a "$LOG"
exit ${PIPESTATUS[0]}
