"""Round-trip check of encoded latents: decode one clip per latent dir back to pixels and compare it with the raw
source frames it claims to come from (seq / start / frame_step in its metadata). This verifies that RGB and thermal
are paired, that sub-sampled frames line up in time, that stills are repeated, and that the VAE round trip is sane.

  python scripts/data/check_latents.py --out $THERMAL_RUNS/check_latents $THERMAL_DATA/latents/*/train
Writes <out>/<dir name>.png (rows: frames 0 / mid / last; columns: raw RGB | decoded RGB | raw thermal | decoded
thermal) and prints PSNR(decoded vs raw) per modality. Expect roughly > 25 dB RGB and > 30 dB thermal; anything
much lower means mis-pairing or misalignment.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from datasets.rgbt_sources import SOURCES, load_clip  # noqa: E402


def psnr(a, b):
    return float(10 * np.log10(4.0 / max(float(((a - b) ** 2).mean()), 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=os.path.join(os.environ.get("THERMAL_DATA", ""), "raw"))
    ap.add_argument("--model_dir", default=os.environ.get("WAN22_5B_DIR"))
    a = ap.parse_args()
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(a.model_dir, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    roots = {"tartanrgbt": "TartanRGBT", "kaist": "KAIST", "llvip": "LLVIP"}
    os.makedirs(a.out, exist_ok=True)
    ok = True
    for d in a.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        if not files:
            print(f"SKIP {d}: empty"); continue
        f = files[len(files) // 2]
        with safe_open(f, "pt") as sf:
            md = sf.metadata() or {}
            meta = json.loads(md.get("meta", "{}"))
            nf = int(md.get("num_frames"))
            z = {k: sf.get_tensor(k).float()[None].cuda() for k in ("cond", "target")}
        src = SOURCES[meta["source"]](os.path.join(a.raw, roots[meta["source"]]))
        rgb_raw, th_raw, _ = load_clip(src, {"seq": meta["seq"], "start": meta["start"]}, nf, meta.get("frame_step", 1))
        with torch.no_grad():
            dec = {k: vae.decode(v).sample[0].float().clamp(-1, 1).permute(1, 2, 3, 0).cpu().numpy() for k, v in z.items()}
        rgb = rgb_raw.astype(np.float32) / 127.5 - 1
        p_rgb, p_th = psnr(dec["cond"], rgb), psnr(dec["target"].mean(-1), th_raw)
        static = ""
        if meta.get("media") == "image":
            static = f", raw frames identical: {bool(np.all(rgb_raw == rgb_raw[0]))}"
        good = p_rgb > 22 and p_th > 25
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {os.path.basename(os.path.dirname(d.rstrip('/')))}/{os.path.basename(d.rstrip('/'))}: "
              f"{os.path.basename(f)} ({meta['source']}, {meta.get('sensor')}, {meta.get('media')}, step {meta.get('frame_step', 1)}, "
              f"{nf}f) PSNR rgb {p_rgb:.1f} dB, thermal {p_th:.1f} dB{static}", flush=True)
        u8 = lambda x: ((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8)
        g3 = lambda x: np.repeat(u8(x)[..., None], 3, -1)
        rows = [np.concatenate([u8(rgb[i]), u8(dec["cond"][i]), g3(th_raw[i]), g3(dec["target"][i].mean(-1))], 1)
                for i in (0, nf // 2, nf - 1)]
        Image.fromarray(np.concatenate(rows, 0)).resize((4 * 320, 3 * 256)).save(
            os.path.join(a.out, f"{os.path.basename(os.path.dirname(d.rstrip('/')))}_{os.path.basename(d.rstrip('/'))}.png"))
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
