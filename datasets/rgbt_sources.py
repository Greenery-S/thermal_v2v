"""Paired RGB / thermal sources behind one interface, so precompute_latents.py can encode any of them into the same
clip format.

A source exposes sequences, each an ordered list of (rgb_path, thermal_path) frame pairs, plus per-frame flags for
frames to avoid (TartanRGBT FFC/NUC shutter frames). Image datasets (media="image") are a "sequence" per image; a clip
repeats that image `num_frames` times (a static clip). Every source carries the caption fields:
  sensor: "radiometric" (16-bit, absolute) | "agc8" (8-bit per-frame auto gain)
  media:  "video" | "image"

Geometry: every frame is center-cropped to the 5:4 aspect and resized to 640x512 (W x H), the TartanRGBT/KAIST native
size. Thermal is reduced to one channel and normalised per clip with percentiles (clip_norm), for 16-bit and 8-bit
alike, then replicated to 3 channels for the RGB VAE.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SIZE = (640, 512)  # W, H
Pair = Tuple[str, str]


def read_list(path: str) -> List[str]:
    out = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            out.append(line)
    return out


def _fit(im: Image.Image, size=SIZE, resample=Image.BILINEAR) -> Image.Image:
    """Center-crop to the target aspect, then resize."""
    W, H = im.size
    tw, th = size
    if W * th > H * tw:          # too wide
        nw = round(H * tw / th); x0 = (W - nw) // 2
        im = im.crop((x0, 0, x0 + nw, H))
    elif W * th < H * tw:        # too tall
        nh = round(W * th / tw); y0 = (H - nh) // 2
        im = im.crop((0, y0, W, y0 + nh))
    return im if im.size == size else im.resize(size, resample)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(_fit(Image.open(path).convert("RGB")), dtype=np.uint8)


def load_thermal(path: str) -> np.ndarray:
    """-> float32 [H, W] in the sensor's raw units (16-bit counts or 8-bit gray)."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        a = np.asarray(im).astype(np.float32)
        return np.asarray(_fit(Image.fromarray(a, mode="F")), dtype=np.float32)
    return np.asarray(_fit(im.convert("L")), dtype=np.float32)


def clip_norm(t: np.ndarray, valid: Optional[np.ndarray] = None, pct=(0.5, 99.5), margin=0.05):
    """[T, H, W] raw -> [-1, 1] with one (lo, hi) for the whole clip (never per frame: no added flicker)."""
    ref = t[valid] if valid is not None and valid.any() else t
    lo, hi = np.percentile(ref, pct)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - margin * span, hi + margin * span
    x = (t - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32), (float(lo), float(hi))


class Source:
    name = "base"
    sensor = "agc8"
    media = "video"

    def __init__(self, root: str):
        self.root = root

    def sequences(self, split: str) -> List[str]:
        raise NotImplementedError

    def frames(self, seq: str) -> List[Pair]:
        raise NotImplementedError

    def cached_frames(self, seq: str) -> List[Pair]:
        cache = self.__dict__.setdefault("_frames_cache", {})
        if seq not in cache:
            cache[seq] = self.frames(seq)
        return cache[seq]

    def bad_frames(self, seq: str, n: int) -> np.ndarray:
        return np.zeros(n, dtype=bool)


class TartanRGBT(Source):
    """<root>/dayX/<seq>/{rgb_in_thermal/*_rgb_in_thermal.png, thermal_left_rect_16/*.png, thermal_left_ffc/data.txt}.
    10 Hz, 640x512, RGB warped into the thermal frame, 16-bit radiometric thermal. Splits by sequence lists."""
    name, sensor, media = "tartanrgbt", "radiometric", "video"

    def __init__(self, root, train_list="configs/data/tartanrgbt_train34.txt", val_list="configs/data/tartanrgbt_debug_val.txt"):
        super().__init__(root)
        self.lists = {"train": train_list, "val": val_list}

    def sequences(self, split):
        return [s for s in read_list(self.lists[split]) if os.path.isdir(os.path.join(self.root, s, "thermal_left_rect_16"))]

    def frames(self, seq):
        d = os.path.join(self.root, seq)
        t = sorted(glob.glob(os.path.join(d, "thermal_left_rect_16", "*.png")))
        r = sorted(glob.glob(os.path.join(d, "rgb_in_thermal", "*.png")))
        n = min(len(t), len(r))  # thermal occasionally has 1-2 extra frames
        return list(zip(r[:n], t[:n]))

    def bad_frames(self, seq, n):
        p = os.path.join(self.root, seq, "thermal_left_ffc", "data.txt")
        out = np.zeros(n, dtype=bool)
        if os.path.isfile(p):
            f = np.array([int(x.strip()) for x in open(p) if x.strip()], dtype=np.int64)[:n] > 0
            out[:len(f)] = f
        return out


class KAIST(Source):
    """KAIST Multispectral Pedestrian: setXX/VYYY/{visible,lwir}/IZZZZZ.jpg, consecutive 20 Hz frames, 640x512,
    beam-splitter aligned, 8-bit white-hot LWIR. Accepts the official release (<root>/kaist-cvpr15/images/set00-11,
    scripts/data/download_kaist.sh) or the HF mirror koifisharriet/KAIST-Multispectral-Pedestrian-Benchmark
    (<root>/kaist_train/set00-05 only). Day: set00-02, 06-08; night: set03-05, 09-11. Val = one day + one night
    sequence."""
    name, sensor, media = "kaist", "agc8", "video"
    VAL = ("set02/V000", "set05/V000")

    def __init__(self, root):
        super().__init__(root)
        for sub in ("kaist-cvpr15/images", "images", "kaist_train"):
            if os.path.isdir(os.path.join(root, sub)):
                self.base = os.path.join(root, sub)
                break
        else:
            raise FileNotFoundError(f"no KAIST set folders under {root}")

    def sequences(self, split):
        seqs = sorted(os.path.relpath(d, self.base) for d in glob.glob(os.path.join(self.base, "set*", "V*")))
        return [s for s in seqs if (s in self.VAL) == (split == "val")]

    def frames(self, seq):
        d = os.path.join(self.base, seq)
        v = sorted(glob.glob(os.path.join(d, "visible", "*.jpg")))
        names = {os.path.basename(p) for p in glob.glob(os.path.join(d, "lwir", "*.jpg"))}
        return [(p, os.path.join(d, "lwir", os.path.basename(p))) for p in v if os.path.basename(p) in names]


class LLVIP(Source):
    """LLVIP registered stills: <root>/LLVIP/{visible,infrared}/{train,test}/SSNNNN.jpg (SS = scene, 26 static
    surveillance cameras, mostly night), 1280x1024, homography-registered, 8-bit white-hot. Each image becomes one
    static clip. Images of a scene are near-duplicates, so `every` keeps every N-th image per split."""
    name, sensor, media = "llvip", "agc8", "image"

    def __init__(self, root, every: int = 5):
        super().__init__(root)
        self.every = int(every)

    def sequences(self, split):
        return ["test" if split == "val" else "train"]

    def frames(self, seq):
        d = os.path.join(self.root, "LLVIP")
        v = sorted(glob.glob(os.path.join(d, "visible", seq, "*.jpg")))[::self.every]
        return [(p, os.path.join(d, "infrared", seq, os.path.basename(p))) for p in v
                if os.path.exists(os.path.join(d, "infrared", seq, os.path.basename(p)))]


SOURCES: Dict[str, type] = {"tartanrgbt": TartanRGBT, "kaist": KAIST, "llvip": LLVIP}


def build_clips(src: Source, split: str, num_frames: int, stride: int, frame_step: int = 1,
                skip_bad: bool = True, limit_seqs: int = 0) -> List[Dict]:
    """Clip index: [{seq, start}] windows of num_frames frames taken every frame_step frames (frame_step=2 turns a
    20 Hz source into 10 Hz). Image sources: one clip per image (start = image index)."""
    clips = []
    seqs = src.sequences(split)
    if limit_seqs:
        seqs = seqs[:limit_seqs]
    for seq in seqs:
        fr = src.cached_frames(seq)
        if src.media == "image":
            clips += [{"seq": seq, "start": i} for i in range(len(fr))]
            continue
        bad = src.bad_frames(seq, len(fr))
        span = (num_frames - 1) * frame_step + 1
        for s in range(0, len(fr) - span + 1, stride * frame_step):
            if skip_bad and bad[s:s + span:frame_step].any():
                continue
            clips.append({"seq": seq, "start": s})
    return clips


def load_clip(src: Source, clip: Dict, num_frames: int, frame_step: int = 1):
    """-> rgb uint8 [T, H, W, 3], thermal float32 [T, H, W] in [-1, 1], meta dict."""
    fr = src.cached_frames(clip["seq"])
    if src.media == "image":
        pairs = [fr[clip["start"]]]
    else:
        pairs = fr[clip["start"]: clip["start"] + (num_frames - 1) * frame_step + 1: frame_step]
    rgb = np.stack([load_rgb(r) for r, _ in pairs])
    th_raw = np.stack([load_thermal(t) for _, t in pairs])
    valid = None
    if src.media != "image":
        valid = ~src.bad_frames(clip["seq"], len(fr))[clip["start"]: clip["start"] + len(pairs) * frame_step: frame_step]
    th, (lo, hi) = clip_norm(th_raw, valid)
    if src.media == "image":
        rgb, th = np.repeat(rgb, num_frames, 0), np.repeat(th, num_frames, 0)
    meta = {"source": src.name, "seq": clip["seq"], "start": clip["start"], "frame_step": frame_step,
            "sensor": src.sensor, "media": src.media, "lo": lo, "hi": hi,
            # clip-relative FFC/NUC shutter frames (only in --keep_bad clips); evals leave them out of the GT metrics
            "ffc": [] if valid is None else np.flatnonzero(~valid).tolist()}
    return rgb, th, meta
