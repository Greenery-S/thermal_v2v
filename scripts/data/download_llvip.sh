#!/bin/bash
# LLVIP (ICCVW 2021): 15,488 registered visible/infrared pairs, 1280x1024, 26 static street cameras (mostly night),
# 8-bit white-hot LWIR. License: non-commercial research use, cite LLVIP; do not redistribute.
# Registered set LLVIP.zip (~4.0 GB) from the Hugging Face copy (same bytes as the official Google Drive file;
# fallback: Drive id 1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv via drive.usercontent.google.com ...&confirm=t).
#   bash scripts/data/download_llvip.sh <out_dir>        -> <out_dir>/LLVIP/{visible,infrared}/{train,test}/*.jpg
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"; cd "$OUT"
SIZE=4003858172
URLS=("https://huggingface.co/datasets/jsonhash/LLVIP/resolve/main/LLVIP.zip"
      "https://drive.usercontent.google.com/download?id=1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv&export=download&confirm=t")
for i in $(seq 1 20); do
  [ -f LLVIP.zip ] && [ "$(stat -c %s LLVIP.zip)" = "$SIZE" ] && break
  curl -L -C - --retry 5 --retry-delay 30 -o LLVIP.zip "${URLS[$(( i % 2 == 1 ? 0 : 1 ))]}" || sleep 60
done
[ "$(stat -c %s LLVIP.zip)" = "$SIZE" ] || { echo "download incomplete"; exit 1; }
unzip -q -o LLVIP.zip && rm -f LLVIP.zip && echo "extracted to $OUT/LLVIP"
