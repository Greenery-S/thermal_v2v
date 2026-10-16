#!/bin/bash
# End-to-end smoke test on ONE GPU with a few clips per source (~1.5 h). It checks the plumbing, not the quality:
#   encode -> stage 1 (20 steps) -> stage-1 eval (both directions)
#   -> stage 1b: merge teacher, parquet, FastVideo DMD (10 steps), 3-step eval
#   -> stage 2: export teacher, LongLive AR (10 iterations, no checkpoint), streaming eval
#   source scripts/env.sh && bash scripts/smoke_test.sh
#   SMOKE_STEPS="1b" bash scripts/smoke_test.sh          # only some parts: any of  data 1 1b 2  (default: all)
# Needs the raw datasets under $THERMAL_DATA/raw (at least one sequence of each source), and the stage-1b / stage-2
# setups (scripts/stage1b/setup_fastvideo.sh, scripts/stage2/setup_longlive.sh).
# Everything goes to $THERMAL_RUNS/smoke, so a real run's latents / checkpoints are never touched.
set -euo pipefail
: "${THERMAL_DATA:?source scripts/env.sh first}"
export NGPU=1 NPROC=1
STEPS=" ${SMOKE_STEPS:-data 1 1b 2} "
want() { [[ "$STEPS" == *" $1 "* ]]; }
SM=${SMOKE_DIR:-$THERMAL_RUNS/smoke}; mkdir -p "$SM/data"
ln -sfn "$THERMAL_DATA/raw" "$SM/data/raw"       # raw data shared, latents / runs separate
export THERMAL_DATA=$SM/data THERMAL_RUNS=$SM
step() { echo; echo "=== $(date +%T) $*"; }
need() {  # need <eval dir> <what>: the in-training eval must have written an mp4 and an eval_log.jsonl line
  local n; n=$(find "$1" -name '*.mp4' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && [ -s "$1/eval_log.jsonl" ] || { echo "FAILED: no $2 output under $1"; exit 1; }
  echo "ok: $2 -> $n mp4, $(wc -l < "$1/eval_log.jsonl") metric lines ($1)"
}

if want data; then
  step "data: encode a few clips per source (33f + 81f)"
  bash "$REPO/scripts/data/encode_all.sh" all --limit_seqs 1 --limit 6
fi

if want 1; then
  step "stage 1: 20 steps"
  bash "$REPO/scripts/stage1/train.sh" training.max_steps=20 training.val_every=10 training.save_every=10 \
    training.batch_size=1 training.warmup_steps=2 training.log_every=1 data.val_limit=2 data.num_workers=2 \
    training.eval_clips=1 training.eval_steps=10
  need "$SM/stage1_bidir_33f/eval" "stage-1 in-training eval"
  step "stage-1 eval (2 clips, both directions, 10 steps)"
  NUM_CLIPS=2 bash "$REPO/scripts/stage1/eval.sh" step_20.pt rgb2thermal --steps 10
  NUM_CLIPS=2 bash "$REPO/scripts/stage1/eval.sh" step_20.pt thermal2rgb --steps 10
fi

if want 1b; then
  step "stage 1b: merge teacher -> parquet -> FastVideo DMD 10 steps -> 3-step eval"
  $PY "$REPO/scripts/stage1b/merge_teacher.py" --run_dir "$SM/stage1_bidir_33f" --ckpt step_20.pt \
    --out "$SM/stage1b_teacher/Wan2.2-TI2V-5B-Diffusers"
  $PY_FV "$REPO/scripts/stage1b/latents_to_parquet.py" --prompt_embeds "$SM/stage1_bidir_33f/prompt_embeds.pt" \
    --src "$THERMAL_DATA/latents/tartanrgbt_33f/train:0.6" --src "$THERMAL_DATA/latents/kaist_33f/train:0.25" \
    --src "$THERMAL_DATA/latents/llvip_33f/train:0.15" --out "$THERMAL_DATA/fastvideo/rgb2thermal_33f/train"
  THERMAL_EVAL_N=1 MAX_STEPS=10 CKPT_EVERY=10 bash "$REPO/scripts/stage1b/train_dmd.sh"
  need "$SM/stage1b_dmd3/eval" "stage-1b in-training eval"
  NUM_CLIPS=2 bash "$REPO/scripts/stage1b/eval.sh" 10
fi

if want 2; then
  step "stage 2: export the stage-1 teacher for LongLive"
  $PY "$REPO/scripts/stage2/export_teacher.py" --run_dir "$SM/stage1_bidir_33f" --ckpt step_20.pt --out "$SM/stage2_teacher"
  step "stage 2: AR teacher forcing, 10 iterations (no checkpoint: a full stage-2 checkpoint is ~50 GB)"
  STAGE2_CONFIG=configs/thermal/stage2_ar_smoke.yaml bash "$REPO/scripts/stage2/train_ar.sh" stage2_ar
  need "$SM/stage2_ar/eval" "stage-2 in-training eval"
  step "stage-2 streaming eval of the exported teacher, run causally (2 clips, 4 steps)"
  bash "$REPO/scripts/stage2/eval_ar.sh" "$SM/stage2_ar/eval_smoke" --n 2 --steps 4
fi

step "SMOKE TEST PASSED ($STEPS)"; echo "outputs: $SM"
