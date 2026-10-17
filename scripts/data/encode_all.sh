#!/bin/bash
# Encode every latent set both stages need, one process per local GPU (resumable: existing clips are skipped).
#   source scripts/env.sh && bash scripts/data/encode_all.sh [stage1|stage2|stage3|all]
# Multi-node: run on each node with NNODES / NODE_RANK set; clips are sharded across all NNODES x NGPU processes.
# Extra args after the stage go to every precompute_latents.py call (e.g. --limit 8 --limit_seqs 1 for a smoke test).
set -euo pipefail
: "${THERMAL_DATA:?source scripts/env.sh first}"
WHAT=${1:-all}; shift || true
EXTRA=("$@")
RAW=$THERMAL_DATA/raw; LAT=$THERMAL_DATA/latents
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=(${CUDA_VISIBLE_DEVICES//,/ }); else GPUS=($(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 )))); fi
NG=${NGPU:-${#GPUS[@]}}; NNODES=${NNODES:-1}; N=$((NG * NNODES))
if [ "$NNODES" -gt 1 ]; then : "${NODE_RANK:?set NODE_RANK (0..NNODES-1) on every node}"; else NODE_RANK=${NODE_RANK:-0}; fi
mkdir -p "$LAT/logs"

run() {  # run <name> <split> <precompute args...>
  local name=$1 split=$2; shift 2
  local pids=()
  for g in $(seq 0 $((NG - 1))); do
    local s=$((NODE_RANK * NG + g))
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} $PY "$REPO/src/precompute_latents.py" --split "$split" --out "$LAT/$name/$split" \
      --model_dir "$WAN22_5B_DIR" --shard $s --num_shards $N "$@" "${EXTRA[@]}" \
      > "$LAT/logs/${name}_${split}_shard$s.log" 2>&1 &
    pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] || { echo "FAILED $name/$split (see $LAT/logs/${name}_${split}_shard*.log)"; exit 1; }
  echo "$name/$split: $(ls "$LAT/$name/$split" | wc -l) clips"
}

cd "$REPO"
TARTAN=(--source tartanrgbt --root "$RAW/TartanRGBT")
KAIST=(--source kaist --root "$RAW/KAIST" --frame_step 2)      # 20 Hz -> 10 Hz
LLVIP=(--source llvip --root "$RAW/LLVIP" --every 5)

if [ "$WHAT" = stage1 ] || [ "$WHAT" = all ]; then           # 33 frames (stage-1 bidirectional teacher)
  run tartanrgbt_33f train "${TARTAN[@]}" --num_frames 33 --stride 16
  run tartanrgbt_33f val   "${TARTAN[@]}" --num_frames 33 --stride 33
  run kaist_33f      train "${KAIST[@]}"  --num_frames 33 --stride 16
  run llvip_33f      train "${LLVIP[@]}"  --num_frames 33
fi
if [ "$WHAT" = stage2 ] || [ "$WHAT" = all ]; then           # 81 frames (stage-2 AR, video only)
  run tartanrgbt_81f train "${TARTAN[@]}" --num_frames 81 --stride 40
  run tartanrgbt_81f val   "${TARTAN[@]}" --num_frames 81 --stride 81
  run kaist_81f      train "${KAIST[@]}"  --num_frames 81 --stride 40
fi
if [ "$WHAT" = stage3 ] || [ "$WHAT" = all ]; then           # stage 3 (S7/S8 few-step streaming)
  # S8 long-rollout tuning: 481-frame (~48 s) video clips. FFC windows are kept: DMD conditions on the RGB only.
  run tartanrgbt_481f train "${TARTAN[@]}" --num_frames 481 --stride 480 --keep_bad --num_workers 1
  run kaist_481f      train "${KAIST[@]}"  --num_frames 481 --stride 480 --keep_bad --num_workers 1
  # long-stream eval: the 4-min TartanRGBT val sequence (day5, 2389 frames at 10 Hz) as ONE clip
  run tartanrgbt_long val "${TARTAN[@]}" --num_frames 2389 --stride 100000 --keep_bad --num_workers 1
fi
