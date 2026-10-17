#!/bin/bash
# KAIST Multispectral Pedestrian (CVPR 2015), official full release: set00-11, consecutive 20 Hz frames, 640x512,
# beam-splitter aligned RGB + 8-bit LWIR. License: CC BY-NC-SA 4.0 (non-commercial).
# Official Google Drive tar (~39 GB), downloaded with curl (resumable, no gdown needed), md5-checked, extracted.
#   bash scripts/data/download_kaist.sh <out_dir>        -> <out_dir>/kaist-cvpr15/images/setXX/VYYY/{visible,lwir}/
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"; cd "$OUT"
ID=1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc
MD5=fcfc7cfcc1237ad414af64b972f8dc97
URL="https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t"
SIZE=38999029760
for i in $(seq 1 50); do
  [ -f kaist-cvpr15.tar ] && [ "$(stat -c %s kaist-cvpr15.tar)" = "$SIZE" ] && break
  curl -fL -C - --retry 5 --retry-delay 30 -o kaist-cvpr15.tar "$URL" || { echo "[retry $i] (Drive quota / network), sleeping"; sleep 120; }
done
[ "$(stat -c %s kaist-cvpr15.tar)" = "$SIZE" ] || { echo "download incomplete"; exit 1; }
echo "$MD5  kaist-cvpr15.tar" | md5sum -c -
tar -xf kaist-cvpr15.tar && echo "extracted to $OUT/kaist-cvpr15 (delete kaist-cvpr15.tar to save 39 GB)"
