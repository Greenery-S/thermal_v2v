#!/bin/bash
# Evaluate a stage-1 checkpoint on the val latents, one direction at a time (single GPU).
#   source scripts/env.sh && bash scripts/stage1/eval.sh <ckpt file name, e.g. step_3000.pt> [rgb2thermal|thermal2rgb] [extra infer_icl.py args]
set -euo pipefail
CK=${1:?ckpt name under <run>/ckpt}; DIR=${2:-rgb2thermal}; shift $(( $# >= 2 ? 2 : 1 ))
RUN=${STAGE1_RUN:-$THERMAL_RUNS/stage1_bidir_33f}
cd "$REPO"
exec $PY src/infer_icl.py --run_dir "$RUN" --ckpt "$CK" --direction "$DIR" --num_clips ${NUM_CLIPS:-8} --steps 30 --cfg 2.0 \
  --out "$RUN/eval/${CK%.pt}_$DIR" "$@"
