#!/bin/bash
# Create the two conda environments (CUDA 12.8 wheels; needs an NVIDIA driver >= 570 / CUDA 12.8 capable).
#   stage 1 + data : $CONDA_ENV_THERMAL  (torch 2.7.1, diffusers 0.39)
#   stage 2        : $CONDA_ENV_LONGLIVE (torch 2.8.0 + flash-attn 2.8.3, LongLive's pinned diffusers) -- created by
#                    scripts/stage2/setup_longlive.sh because it needs the LongLive checkout's requirements.txt
#   source scripts/env.sh && bash scripts/setup/create_envs.sh
set -euo pipefail
: "${CONDA_ENV_THERMAL:?source scripts/env.sh first}"
conda create -y -n "$CONDA_ENV_THERMAL" python=3.10
conda run -n "$CONDA_ENV_THERMAL" pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n "$CONDA_ENV_THERMAL" pip install -r "$REPO/requirements.txt"
conda run -n "$CONDA_ENV_THERMAL" python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'diffusers', diffusers.__version__)"
