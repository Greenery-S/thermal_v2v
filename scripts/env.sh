#!/bin/bash
# Environment for every thermal_v2v script. Edit the paths once for your cluster (or export them before sourcing):
#   source scripts/env.sh
# (bash or zsh: $0 is the sourced file in zsh)
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}
export WAN22_5B_DIR=${WAN22_5B_DIR:-/path/to/Wan2.2-TI2V-5B-Diffusers}   # HF Wan-AI/Wan2.2-TI2V-5B-Diffusers
export THERMAL_DATA=${THERMAL_DATA:-/path/to/thermal_v2v_data}            # raw/ (downloads) and latents/ (encoded)
export THERMAL_RUNS=${THERMAL_RUNS:-/path/to/thermal_v2v_runs}            # checkpoints, logs, evals
export THERMAL_EXT=${THERMAL_EXT:-$THERMAL_DATA/external}                  # LongLive checkout (stage 2)
# conda envs created by scripts/setup/create_envs.sh
export CONDA_ENV_THERMAL=${CONDA_ENV_THERMAL:-thermal_v2v}                 # stage 1 + data
export CONDA_ENV_LONGLIVE=${CONDA_ENV_LONGLIVE:-thermal_v2v_longlive}      # stage 2
export CONDA_ENV_FASTVIDEO=${CONDA_ENV_FASTVIDEO:-thermal_v2v_fastvideo}   # stage 1b
# python of each env. Explicitly exported PY / PY_LL / PY_FV win; otherwise they are (re)resolved from the conda envs on
# every `source` (so sourcing again after scripts/setup/* picks up the new envs). An env that does not exist yet leaves
# its variable empty (with a warning) instead of silently falling back to some other python.
_thermal_py() {  # _thermal_py <VAR> <conda env>
  local var=$1 env=$2 cur isauto
  eval "cur=\${$var:-}; isauto=\${_THERMAL_AUTO_$var:-}"      # (no ${!var}: works in zsh too)
  if [ -n "$cur" ] && [ -z "$isauto" ]; then return; fi             # set by the user: keep
  local p; p=$(conda run -n "$env" which python 2>/dev/null || true)
  export "$var=$p" "_THERMAL_AUTO_$var=1"
  [ -n "$p" ] || echo "[env.sh] conda env '$env' not found yet -> $var empty (run its setup script, then source env.sh again)" >&2
}
_thermal_py PY "$CONDA_ENV_THERMAL"; _thermal_py PY_LL "$CONDA_ENV_LONGLIVE"; _thermal_py PY_FV "$CONDA_ENV_FASTVIDEO"
# launchers default NPROC to the GPUs this process may use: bash $REPO/scripts/_ngpu.sh (honours CUDA_VISIBLE_DEVICES)
export FASTVIDEO_DIR=${FASTVIDEO_DIR:-$THERMAL_EXT/FastVideo}
export LONGLIVE_DIR=${LONGLIVE_DIR:-$THERMAL_EXT/LongLive/LongLive2.0}
export PYTHONPATH=$REPO:${PYTHONPATH:-} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=${TMPDIR:-/tmp} HF_HOME=${HF_HOME:-$THERMAL_DATA/hf_home}
# torch.compile / triton caches must be on an exec-able local disk
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/inductor_cache_$USER}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TMPDIR/triton_cache_$USER}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
