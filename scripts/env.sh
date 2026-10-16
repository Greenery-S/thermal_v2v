#!/bin/bash
# Environment for every thermal_v2v script. Edit the paths once for your cluster (or export them before sourcing):
#   source scripts/env.sh
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export WAN22_5B_DIR=${WAN22_5B_DIR:-/path/to/Wan2.2-TI2V-5B-Diffusers}   # HF Wan-AI/Wan2.2-TI2V-5B-Diffusers
export THERMAL_DATA=${THERMAL_DATA:-/path/to/thermal_v2v_data}            # raw/ (downloads) and latents/ (encoded)
export THERMAL_RUNS=${THERMAL_RUNS:-/path/to/thermal_v2v_runs}            # checkpoints, logs, evals
export THERMAL_EXT=${THERMAL_EXT:-$THERMAL_DATA/external}                  # LongLive checkout (stage 2)
# conda envs created by scripts/setup/create_envs.sh
export CONDA_ENV_THERMAL=${CONDA_ENV_THERMAL:-thermal_v2v}                 # stage 1 + data
export CONDA_ENV_LONGLIVE=${CONDA_ENV_LONGLIVE:-thermal_v2v_longlive}      # stage 2
export CONDA_ENV_FASTVIDEO=${CONDA_ENV_FASTVIDEO:-thermal_v2v_fastvideo}   # stage 1b
export PY=${PY:-$(conda run -n "$CONDA_ENV_THERMAL" which python 2>/dev/null || which python)}
export PY_LL=${PY_LL:-$(conda run -n "$CONDA_ENV_LONGLIVE" which python 2>/dev/null || which python)}
export PY_FV=${PY_FV:-$(conda run -n "$CONDA_ENV_FASTVIDEO" which python 2>/dev/null || which python)}
export FASTVIDEO_DIR=${FASTVIDEO_DIR:-$THERMAL_EXT/FastVideo}
export LONGLIVE_DIR=${LONGLIVE_DIR:-$THERMAL_EXT/LongLive/LongLive2.0}
export PYTHONPATH=$REPO:${PYTHONPATH:-} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=${TMPDIR:-/tmp} HF_HOME=${HF_HOME:-$THERMAL_DATA/hf_home}
# torch.compile / triton caches must be on an exec-able local disk
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/inductor_cache_$USER}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TMPDIR/triton_cache_$USER}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
