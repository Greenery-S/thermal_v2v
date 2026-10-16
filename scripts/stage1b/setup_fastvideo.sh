#!/bin/bash
# Stage 1b setup: FastVideo (hao-ai-lab) at the pinned upstream commit + our ICL patch, in its own conda env.
# FastVideo pins torch 2.12.0; its installer picks the CUDA build via uv's torch backend (cu126 works with CUDA 12.x drivers).
#   source scripts/env.sh && bash scripts/stage1b/setup_fastvideo.sh
set -euo pipefail
: "${THERMAL_EXT:?source scripts/env.sh first}"
UPSTREAM=2b164405     # hao-ai-lab/FastVideo main, 2026-10 (the patch was made against it)
mkdir -p "$THERMAL_EXT"; cd "$THERMAL_EXT"
if [ ! -d FastVideo ]; then
  git clone https://github.com/hao-ai-lab/FastVideo.git
  git -C FastVideo checkout -b thermal-icl $UPSTREAM
  git -C FastVideo -c user.name=thermal_v2v -c user.email=none am "$REPO/scripts/stage1b/fastvideo_thermal_icl.patch"
fi
if ! conda env list | grep -q "^$CONDA_ENV_FASTVIDEO "; then
  conda create -y -n "$CONDA_ENV_FASTVIDEO" python=3.10
  conda run -n "$CONDA_ENV_FASTVIDEO" pip install uv
  # fastvideo-kernel (sparse-attention kernels) is not needed: we train with FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
  (cd FastVideo && UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu126} conda run -n "$CONDA_ENV_FASTVIDEO" uv pip install -e .)
  conda run -n "$CONDA_ENV_FASTVIDEO" uv pip install pyarrow safetensors omegaconf
fi
conda run -n "$CONDA_ENV_FASTVIDEO" python -c "import torch, fastvideo; print('torch', torch.__version__, 'fastvideo ok')"
echo "FastVideo ready at $THERMAL_EXT/FastVideo"
