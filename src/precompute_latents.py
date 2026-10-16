"""Encode paired RGB / thermal clips with the Wan2.2 VAE -> one safetensors file per clip.

Each file holds `cond` (RGB) and `target` (thermal, per-clip normalised, replicated to 3 channels) RAW Wan2.2 VAE
latents [48, f, h, w] (fp16, un-normalised; trainers apply (z - mean) / std), plus metadata "meta" (JSON: source,
seq, start, frame_step, sensor, media, lo, hi) that the trainers turn into the structured caption.

Sources (datasets/rgbt_sources.py): tartanrgbt (10 Hz video, radiometric), kaist (20 Hz video -> --frame_step 2 for
10 Hz, 8-bit AGC), llvip (stills, 8-bit AGC; each image -> one static clip, --every N keeps every N-th image).

  python src/precompute_latents.py --source kaist --root $DATA/raw/KAIST --split train \
      --num_frames 33 --stride 16 --frame_step 2 --out $DATA/latents/kaist_33f/train --model_dir $WAN22_5B_DIR

Resumable (existing files are skipped). Sharding: --shard i --num_shards n (one process per GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from datasets.rgbt_sources import SOURCES, build_clips, load_clip  # noqa: E402


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, src, clips, num_frames, frame_step):
        self.src, self.clips, self.n, self.step = src, clips, num_frames, frame_step

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        rgb, th, meta = load_clip(self.src, self.clips[i], self.n, self.step)
        cond = torch.from_numpy(rgb).permute(3, 0, 1, 2).float() / 127.5 - 1.0          # 3, T, H, W
        target = torch.from_numpy(th).unsqueeze(0).expand(3, -1, -1, -1).contiguous()   # 3, T, H, W
        return {"cond": cond, "target": target, "meta": meta}


def make_source(a):
    cls = SOURCES[a.source]
    kw = {}
    if a.source == "tartanrgbt":
        kw = {"train_list": a.train_list, "val_list": a.val_list}
    elif a.source == "llvip":
        kw = {"every": a.every}
    return cls(a.root, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--root", required=True, help="raw dataset root")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_dir", required=True, help="Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--num_frames", type=int, default=33, help="4k+1")
    ap.add_argument("--stride", type=int, default=16, help="clip start stride, in (sub-sampled) frames")
    ap.add_argument("--frame_step", type=int, default=1, help="2 = take every other frame (20 Hz -> 10 Hz)")
    ap.add_argument("--every", type=int, default=5, help="llvip: keep every N-th image")
    ap.add_argument("--train_list", default="configs/data/tartanrgbt_train34.txt")
    ap.add_argument("--val_list", default="configs/data/tartanrgbt_debug_val.txt")
    ap.add_argument("--limit_seqs", type=int, default=0, help="smoke tests: only the first N sequences")
    ap.add_argument("--limit", type=int, default=0, help="smoke tests: only the first N clips")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=6)
    a = ap.parse_args()
    assert (a.num_frames - 1) % 4 == 0, "num_frames must be 4k+1 for the Wan VAE"

    src = make_source(a)
    clips = build_clips(src, a.split, a.num_frames, a.stride, a.frame_step, limit_seqs=a.limit_seqs)
    if a.limit:
        clips = clips[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    name_of = lambda c: f"{a.source}__{c['seq'].replace('/', '__')}__{c['start']:06d}.safetensors"
    todo = [c for c in clips[a.shard::a.num_shards] if not os.path.exists(os.path.join(a.out, name_of(c)))]
    print(f"{a.source}/{a.split}: {len(clips)} clips, shard {a.shard}/{a.num_shards}: {len(todo)} to encode", flush=True)
    if not todo:
        return

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.model_dir, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    vae.requires_grad_(False)
    loader = torch.utils.data.DataLoader(ClipDataset(src, todo, a.num_frames, a.frame_step), batch_size=1,
                                         num_workers=a.num_workers, collate_fn=lambda b: b[0])
    t0 = time.time()
    for i, item in enumerate(loader):
        m = item["meta"]
        with torch.no_grad():
            z_c = vae.encode(item["cond"][None].cuda()).latent_dist.mode()[0]
            z_t = vae.encode(item["target"][None].cuda()).latent_dist.mode()[0]
        path = os.path.join(a.out, f"{a.source}__{m['seq'].replace('/', '__')}__{m['start']:06d}.safetensors")
        tmp = path + ".tmp"
        save_file({"cond": z_c.half().cpu().contiguous(), "target": z_t.half().cpu().contiguous()}, tmp,
                  metadata={"meta": json.dumps(m), "num_frames": str(a.num_frames)})
        os.replace(tmp, path)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] {(time.time() - t0) / (i + 1):.2f}s/clip", flush=True)
    print(f"DONE {a.source}/{a.split} shard {a.shard}: {len(todo)} clips, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
