"""In-training evaluation helpers shared by all stages (stage 1 trainer, FastVideo stage 1b, LongLive stage 2).

Pure numpy / imageio / wandb, so the stage-1b and stage-2 code (separate envs, separate repos) can load this file by
path (THERMAL_V2V_REPO) without importing anything else from thermal_v2v.

  timer = EvalTimer(minutes=60)                 # time-based trigger; same decision on every rank
  if timer.due(): ... run the stage's sampler on a few fixed val clips ...
  log_eval_clip(out_dir, step, tag, cond, pred, gt, run=wandb_run_or_None)

Each call writes <out_dir>/step_XXXXXX/<tag>.mp4 ([condition | prediction | real], grayscale thermal) and appends one
JSON line (step, tag, metrics) to <out_dir>/eval_log.jsonl; with a wandb run it also logs the video and the metrics
under eval/<tag>/... at the same step.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np


# ----------------------------------------------------------------------------- trigger
class EvalTimer:
    """Fires every `minutes` of wall-clock training time (and once at the first check if at_start). Under
    torch.distributed the decision is taken on rank 0 and broadcast, so every rank enters the eval together (needed
    when the model is FSDP-sharded)."""

    def __init__(self, minutes: float = 60.0, at_start: bool = True):
        self.period = float(minutes) * 60.0
        self.last = None if at_start else time.time()

    def due(self) -> bool:
        if self.period <= 0:
            return False
        fire = self.last is None or (time.time() - self.last) >= self.period
        try:
            import torch
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
                t = torch.tensor([1 if fire else 0], device=dev)
                dist.broadcast(t, 0)
                fire = bool(t.item())
        except ImportError:
            pass
        if fire:
            self.last = time.time()
        return fire


# ----------------------------------------------------------------------------- frames / metrics
def to_u8(x) -> np.ndarray:
    """[T, H, W] or [T, H, W, 3] (or channel-first [T, 3, H, W]) in [-1, 1] -> uint8 [T, H, W, 3]."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 4 and x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(0, 2, 3, 1)
    if x.ndim == 3:
        x = x[..., None]
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, -1)
    return ((np.clip(x, -1, 1) + 1) * 127.5).round().astype(np.uint8)


def gray(x) -> np.ndarray:
    return to_u8(x).astype(np.float32).mean(-1) / 127.5 - 1.0


def metrics(pred, gt) -> Dict[str, float]:
    """Thermal-style metrics on grayscale [T, H, W] in [-1, 1]: per-clip affine-aligned PSNR / SSIM (the temperature
    range of a clip is not recoverable from RGB), Pearson correlation, gain over the constant prediction, and the
    4-frame chunk-boundary jump ratio (~1 = no seams)."""
    p, g = gray(pred).astype(np.float64), gray(gt).astype(np.float64)
    pf, gf = p.reshape(-1), g.reshape(-1)
    pc, gc = pf - pf.mean(), gf - gf.mean()
    a = (pc * gc).sum() / max((pc * pc).sum(), 1e-12)
    b = gf.mean() - a * pf.mean()
    pa = np.clip(a * p + b, -1, 1)
    mse = float(((pa - g) ** 2).mean())
    psnr = 10 * np.log10(4.0 / max(mse, 1e-10))
    psnr_const = 10 * np.log10(4.0 / max(float(g.var()), 1e-10))
    pear = float((pc * gc).sum() / max(np.sqrt((pc * pc).sum() * (gc * gc).sum()), 1e-12))
    out = {"psnr_affine": float(psnr), "gain_db": float(psnr - psnr_const), "pearson": pear}
    try:
        from skimage.metrics import structural_similarity
        idx = range(0, len(g), max(1, len(g) // 4))
        out["ssim_affine"] = float(np.mean([structural_similarity(pa[i], g[i], data_range=2.0) for i in idx]))
    except ImportError:
        pass
    if len(p) > 8:
        d = np.abs(np.diff(p, axis=0)).mean(axis=(1, 2))
        k = np.arange(len(d))
        out["jump_ratio"] = float(d[k % 4 == 0].mean() / max(d[k % 4 != 0].mean(), 1e-8))
    return out


# ----------------------------------------------------------------------------- logging
def log_eval_clip(out_dir: str, step: int, tag: str, cond, pred, gt, run=None, fps: int = 10,
                  extra: Optional[Dict] = None, thermal_target: bool = True) -> Dict[str, float]:
    """cond / pred / gt: [T, H, W(, 3)] in [-1, 1]. thermal_target=False (thermal->rgb) computes metrics on the
    grayscale of the RGB prediction too, which is still a sane progress signal."""
    import imageio
    frames = np.concatenate([to_u8(cond), to_u8(pred), to_u8(gt)], axis=2)
    path = os.path.join(out_dir, f"step_{int(step):06d}", f"{tag}.mp4")   # tag may contain "/" (task/clipN)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, quality=8, macro_block_size=1)
    m = metrics(pred, gt)
    rec = {"step": int(step), "tag": tag, "time": time.strftime("%F %T"), "mp4": path, **m, **(extra or {})}
    with open(os.path.join(out_dir, "eval_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    if run is not None:
        try:  # the mp4 file itself (raw frames would need moviepy); wandb trouble never stops training
            import wandb
            run.log({f"eval/{tag}/video": wandb.Video(path, format="mp4"),
                     **{f"eval/{tag}/{k}": v for k, v in m.items()}}, step=int(step))
        except Exception as e:
            print(f"[eval] wandb log failed ({e!r}); local mp4 + eval_log.jsonl are written", flush=True)
    print(f"[eval] step {step} {tag}: " + " ".join(f"{k}={v:.3f}" for k, v in m.items()) + f" -> {path}", flush=True)
    return m


def wandb_mode() -> str:
    """online if WANDB_API_KEY is set or `wandb login` stored a key (~/.netrc), else offline; WANDB_MODE overrides."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_DISABLED", "").lower() == "true":
        return "disabled"
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = os.path.expanduser(os.environ.get("NETRC", "~/.netrc"))
    try:
        return "online" if "api.wandb.ai" in open(netrc).read() else "offline"
    except OSError:
        return "offline"


def wandb_run_id(out_dir: str) -> str:
    """One wandb id per run directory, stored in <out_dir>/wandb_id.txt: restarts resume the same run, while a new run
    directory with the same name (e.g. after the smoke test) gets a fresh run instead of colliding."""
    os.makedirs(out_dir, exist_ok=True)
    f = os.path.join(out_dir, "wandb_id.txt")
    if os.environ.get("WANDB_RUN_ID"):
        return os.environ["WANDB_RUN_ID"]
    if not os.path.exists(f):
        import uuid
        with open(f, "w") as fh:
            fh.write(uuid.uuid4().hex[:12])
    return open(f).read().strip()


def wandb_init(project: str, name: str, out_dir: str, config: Optional[dict] = None):
    """wandb run (see wandb_mode / wandb_run_id). Returns None if wandb is unavailable."""
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed; local logs only", flush=True)
        return None
    return wandb.init(project=os.environ.get("WANDB_PROJECT", project), entity=os.environ.get("WANDB_ENTITY") or None,
                      name=name, dir=out_dir, config=config, mode=wandb_mode(), resume="allow",
                      id=wandb_run_id(out_dir))
