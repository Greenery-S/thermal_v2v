"""Stage 1b data: our clip latents (src/precompute_latents.py) -> FastVideo ICL parquet rows (rgb2thermal).

Columns = FastVideo's t2v parquet schema + cond_latent_{bytes,shape,dtype}:
  vae_latent     <- thermal target latent [48, f, h, w] (raw VAE, float32)
  cond_latent    <- RGB condition latent  [48, f, h, w] (raw VAE, float32)
  text_embedding <- the clip's structured rgb2thermal caption (datasets/captions.py), UNPADDED [L, 4096] bf16,
                    looked up in the stage-1 prompt cache (FastVideo pads to 512)
Several sources are mixed by weight: each source contributes round(weight * total) rows (drawn with replacement when
a source has fewer clips than its share), so the uniform FastVideo sampler sees the stage-1 mixing ratios.

  python scripts/stage1b/latents_to_parquet.py --prompt_embeds $THERMAL_RUNS/stage1_bidir_33f/prompt_embeds.pt \
      --src $THERMAL_DATA/latents/tartanrgbt_33f/train:0.6 --src $THERMAL_DATA/latents/kaist_33f/train:0.25 \
      --src $THERMAL_DATA/latents/llvip_33f/train:0.15 --out $THERMAL_DATA/fastvideo/rgb2thermal_33f/train
"""
import argparse
import glob
import importlib.util
import json
import os
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open

_spec = importlib.util.spec_from_file_location(   # by path: `datasets` clashes with the Hugging Face library
    "thermal_v2v_captions", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "captions.py"))
captions = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(captions)

ap = argparse.ArgumentParser()
ap.add_argument("--src", action="append", required=True, help="latent_dir[:weight] (repeatable)")
ap.add_argument("--prompt_embeds", required=True, help="stage-1 run's prompt_embeds.pt (all structured captions)")
ap.add_argument("--out", required=True)
ap.add_argument("--rows_per_file", type=int, default=256)
ap.add_argument("--limit", type=int, default=0, help="smoke tests: at most N clips per source")
ap.add_argument("--fps", type=float, default=10.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cache = torch.load(a.prompt_embeds)
text_cache = {}


def text_for(cap):
    if cap not in text_cache:
        pe = cache[cap]
        L = int((pe.float().abs().sum(-1) > 0).sum())
        t = pe[:L].contiguous().to(torch.bfloat16)
        text_cache[cap] = (t.view(torch.uint16).numpy().tobytes(), list(t.shape))
    return text_cache[cap]


srcs = []
for s in a.src:
    d, w = s, "1"
    if ":" in s:
        head, tail = s.rsplit(":", 1)
        try:
            float(tail); d, w = head, tail
        except ValueError:
            pass
    fs = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if a.limit:
        fs = fs[:a.limit]
    srcs.append((d, float(w), fs))
total = sum(len(fs) for _, _, fs in srcs)
wsum = sum(w for _, w, _ in srcs)
rng = random.Random(a.seed)
files = []
for d, w, fs in srcs:
    n = max(1, round(total * w / wsum))
    pick = rng.sample(fs, n) if n <= len(fs) else [rng.choice(fs) for _ in range(n)]
    files += pick
    print(f"{d}: {len(fs)} clips -> {n} rows (weight {w})")
rng.shuffle(files)

fields = ["id", "vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype", "text_embedding_bytes", "text_embedding_shape",
          "text_embedding_dtype", "file_name", "caption", "media_type", "width", "height", "num_frames", "duration_sec", "fps",
          "cond_latent_bytes", "cond_latent_shape", "cond_latent_dtype"]
schema = pa.schema([
    pa.field("id", pa.string()), pa.field("vae_latent_bytes", pa.binary()), pa.field("vae_latent_shape", pa.list_(pa.int64())),
    pa.field("vae_latent_dtype", pa.string()), pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())), pa.field("text_embedding_dtype", pa.string()),
    pa.field("file_name", pa.string()), pa.field("caption", pa.string()), pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()), pa.field("height", pa.int64()), pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()), pa.field("fps", pa.float64()),
    pa.field("cond_latent_bytes", pa.binary()), pa.field("cond_latent_shape", pa.list_(pa.int64())),
    pa.field("cond_latent_dtype", pa.string()),
])
os.makedirs(a.out, exist_ok=True)
rows = {k: [] for k in fields}
n_file = 0


def flush():
    global rows, n_file
    if rows["id"]:
        pq.write_table(pa.table(rows, schema=schema), os.path.join(a.out, f"data_{n_file:05d}.parquet"))
        n_file += 1
        rows = {k: [] for k in fields}


for i, f in enumerate(files):
    with safe_open(f, "pt") as sf:
        md = sf.metadata() or {}
        meta = json.loads(md.get("meta", "{}"))
        nf = int(md.get("num_frames", 33))
        cond = sf.get_tensor("cond").float().numpy()
        tgt = sf.get_tensor("target").float().numpy()
    cap = captions.caption("rgb2thermal", meta.get("sensor", "radiometric"), meta.get("media", "video"))
    tb, ts = text_for(cap)
    name = f"{i:06d}_{os.path.basename(f)}"
    rows["id"].append(name); rows["file_name"].append(name); rows["caption"].append(cap); rows["media_type"].append("video")
    rows["vae_latent_bytes"].append(tgt.astype(np.float32).tobytes()); rows["vae_latent_shape"].append(list(tgt.shape))
    rows["vae_latent_dtype"].append("float32")
    rows["cond_latent_bytes"].append(cond.astype(np.float32).tobytes()); rows["cond_latent_shape"].append(list(cond.shape))
    rows["cond_latent_dtype"].append("float32")
    rows["text_embedding_bytes"].append(tb); rows["text_embedding_shape"].append(ts); rows["text_embedding_dtype"].append("bfloat16")
    rows["width"].append(int(cond.shape[-1] * 16)); rows["height"].append(int(cond.shape[-2] * 16)); rows["num_frames"].append(nf)
    rows["duration_sec"].append(nf / a.fps); rows["fps"].append(a.fps)
    if len(rows["id"]) >= a.rows_per_file:
        flush()
flush()
print(f"wrote {len(files)} rows in {n_file} files -> {a.out}; captions used: {len(text_cache)}")
