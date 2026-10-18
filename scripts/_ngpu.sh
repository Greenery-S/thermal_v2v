#!/bin/bash
# Number of GPUs this process may use: CUDA_VISIBLE_DEVICES if set (e.g. a partial-node allocation), else all GPUs.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .; else nvidia-smi -L | wc -l; fi
