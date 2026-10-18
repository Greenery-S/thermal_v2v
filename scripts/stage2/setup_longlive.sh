#!/bin/bash
# Stage 2 setup: NVlabs/LongLive at the pinned upstream commit + our patch, its conda env, and the original-format
# Wan2.2-TI2V-5B files LongLive loads (VAE .pth, UMT5 tokenizer, config).
#   source scripts/env.sh && bash scripts/stage2/setup_longlive.sh
set -euo pipefail
: "${THERMAL_EXT:?source scripts/env.sh first}"
UPSTREAM=c6a27e7      # NVlabs/LongLive main, 2026-10 (the patch was made against it)
mkdir -p "$THERMAL_EXT"; cd "$THERMAL_EXT"
if [ ! -d LongLive ]; then
  git clone https://github.com/NVlabs/LongLive.git
  git -C LongLive checkout -b thermal-v2v $UPSTREAM
  git -C LongLive -c user.name=thermal_v2v -c user.email=none am "$REPO/scripts/stage2/longlive_thermal_v2v.patch"
fi
LL=$THERMAL_EXT/LongLive/LongLive2.0

# conda env (LongLive pins its own diffusers/peft stack; torch 2.8 cu128 + prebuilt flash-attn 2.8.3)
if ! conda env list | grep -q "^$CONDA_ENV_LONGLIVE "; then
  conda create -y -n "$CONDA_ENV_LONGLIVE" python=3.10
  RUN=(conda run -n "$CONDA_ENV_LONGLIVE")
  "${RUN[@]}" pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
  grep -v -E '^(git\+|#|dashscope|flask|starlette|onnx|pycocotools|dominate)' "$LL/requirements.txt" | sed 's/#.*//' \
    | grep -v '^\s*$' > "$TMPDIR/ll_reqs.txt"
  "${RUN[@]}" pip install -r "$TMPDIR/ll_reqs.txt"
  "${RUN[@]}" pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
  "${RUN[@]}" pip install safetensors pyyaml "peft==0.17.1" imageio-ffmpeg scikit-image   # newer peft rejects torchao 0.13
fi

# original-format Wan2.2-TI2V-5B pieces (VAE, tokenizer, config); the DiT weights come from the exported teacher
W=$LL/wan_models/Wan2.2-TI2V-5B; mkdir -p "$W"
# `conda run` does not forward stdin (a heredoc would silently not run): call the env's python directly
PYLL=${PY_LL:-$(conda run -n "$CONDA_ENV_LONGLIVE" which python | tail -1)}
"$PYLL" - "$W" <<'PY'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
w = sys.argv[1]
hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth", local_dir=w)
hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "config.json", local_dir=w)
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", allow_patterns=["google/umt5-xxl/*"], local_dir=w)
print("wan_models ready:", w)
PY
for f in Wan2.2_VAE.pth config.json google/umt5-xxl; do
  [ -e "$W/$f" ] || { echo "[setup_longlive] missing $W/$f" >&2; exit 1; }
done
echo "LongLive ready at $LL"
