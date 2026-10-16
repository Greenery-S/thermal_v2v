#!/bin/bash
# Evaluate the 3-step stage-1b student (EMA weights) on the stage-1 val latents, rgb2thermal, no CFG.
#   source scripts/env.sh && bash scripts/stage1b/eval.sh <step, e.g. 3000> [extra infer_icl.py args]
# Teacher baselines on the same clips: bash scripts/stage1/eval.sh <stage-1 ckpt> rgb2thermal (30 steps, CFG 2).
set -euo pipefail
STEP=${1:?checkpoint step}; shift
OUT_DIR=${OUT:-$THERMAL_RUNS/stage1b_dmd3}
TEACHER=${TEACHER:-$THERMAL_RUNS/stage1b_teacher/Wan2.2-TI2V-5B-Diffusers}
STU=$OUT_DIR/checkpoint-${STEP}_weight_only/ema/generator_ema.safetensors
[ -f "$STU" ] || { echo "no EMA student at $STU"; exit 1; }
cd "$REPO"
exec $PY src/infer_icl.py --run_dir ${STAGE1_RUN:-$THERMAL_RUNS/stage1_bidir_33f} --direction rgb2thermal \
  --num_clips ${NUM_CLIPS:-8} --fv_teacher "$TEACHER" --fv_student "$STU" --cfg 1.0 --out "$OUT_DIR/eval/student_ema_s$STEP" "$@"
