"""ICL V2V trainer for Wan2.2-TI2V-5B: [cond clean t=0 | target noisy], shared RoPE, flow matching.

Single-direction (rgb2thermal) or bidirectional (UniVidX-style stochastic condition masking) is just the
`directions` weights in the config. Trainable modes: `lora` (token-group gated LoRA, UniVidX recipe) or
`full` (bf16 weights + fp32 master + 8-bit AdamW, UnityVideo recipe). Reads pre-encoded latents from
src/precompute_latents.py. Launch with torchrun:

  torchrun --standalone --nproc_per_node=1 src/train_icl.py --config configs/train/rgb2thermal_lora_debug.yaml \
      output_dir=$THERMAL_RUNS/exp1  [key=value overrides]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from models.icl.transformer import WanIclV2VTransformer3DModel
from models.icl.gated_lora import inject_gated_lora, set_token_groups, lora_parameters, lora_state_dict
from models.icl.position_utils import patch_grid_from_latents, token_count
from tools.flow_schedule import FlowMatchScheduler

MODALITIES = ("rgb", "thermal")
# direction -> (cond modality, target modality); "joint" = both noisy (no cond)
DIRECTIONS = {"rgb2thermal": ("rgb", "thermal"), "thermal2rgb": ("thermal", "rgb"), "joint": (None, None)}


# ----------------------------------------------------------------------------- data
class LatentPairs(Dataset):
    """Pre-encoded clips: keys cond(rgb)/target(thermal) [48,f,h,w] fp16 raw VAE latents.

    `dirs` entries are paths or {dir, weight} dicts (multi-source mixing: a source's clips share its weight,
    so a source with weight w is drawn w/sum(w) of the time regardless of its size). Each file's metadata
    "meta" JSON may carry sensor ("radiometric" | "agc8") and media ("video" | "image") for the structured
    caption; files without them (phase-1 TartanRGBT latents) count as radiometric video."""

    def __init__(self, dirs, limit: int = 0):
        self.files, self.weights = [], []
        for d in dirs:
            path, w = (d, 1.0) if isinstance(d, str) else (str(d["dir"]), float(d.get("weight", 1.0)))
            fs = sorted(glob.glob(os.path.join(path, "*.safetensors")))
            if limit:
                fs = fs[:limit]
            self.files += fs
            self.weights += [w / max(len(fs), 1)] * len(fs)
        if not self.files:
            raise FileNotFoundError(f"no latents under {dirs}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from safetensors import safe_open
        d = load_file(self.files[i])
        with safe_open(self.files[i], "pt") as fh:
            meta = json.loads((fh.metadata() or {}).get("meta", "{}"))
        return {"rgb": d["cond"].float(), "thermal": d["target"].float(), "path": self.files[i],
                "sensor": meta.get("sensor", "radiometric"), "media": meta.get("media", "video")}


class WeightedDistributedSampler(torch.utils.data.Sampler):
    """Per-epoch multinomial draw (with replacement) by per-sample weight, sharded across ranks."""

    def __init__(self, weights, num_replicas, rank, seed=0):
        self.w = torch.tensor(weights, dtype=torch.double)
        self.n = len(weights) // num_replicas
        self.num_replicas, self.rank, self.seed, self.epoch = num_replicas, rank, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.w, self.n * self.num_replicas, replacement=True, generator=g)
        return iter(idx[self.rank::self.num_replicas].tolist())


class FakeLatentPairs(Dataset):
    def __init__(self, n=8, shape=(48, 2, 4, 6)):
        self.n, self.shape = n, shape

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {"rgb": torch.randn(*self.shape, generator=g), "thermal": torch.randn(*self.shape, generator=g) * 0.5, "path": f"fake{i}",
                "sensor": "radiometric", "media": "video"}


def collate(b):
    return {"rgb": torch.stack([x["rgb"] for x in b]), "thermal": torch.stack([x["thermal"] for x in b]),
            "path": [x["path"] for x in b], "sensor": [x["sensor"] for x in b], "media": [x["media"] for x in b]}


# ----------------------------------------------------------------------------- model wrapper
class IclSystem(torch.nn.Module):
    def __init__(self, transformer: WanIclV2VTransformer3DModel, cfg):
        super().__init__()
        self.transformer = transformer
        D = transformer.inner_dim
        # zero-init modality embeddings (added to every token of that stream; role-agnostic)
        self.modality_embed = torch.nn.ParameterDict({m: torch.nn.Parameter(torch.zeros(D)) for m in MODALITIES})
        self.cfg = cfg

    def forward(self, noisy_target, cond, t, text, cond_mod, target_mod):
        """noisy_target/cond: [B,48,f,h,w] normalised latents (cond may be None for joint/unconditional)."""
        B = noisy_target.shape[0]
        grid = patch_grid_from_latents(noisy_target, self.transformer.config.patch_size)
        n = token_count(grid)
        groups = []
        if cond is not None:
            groups.append((cond_mod, n, False))
        groups.append((target_mod, n, True))
        if self.cfg.model.trainable == "lora":
            set_token_groups(self.transformer, groups)
        out = self.transformer(
            noisy_target, t, text, cond_latents=cond,
            # "thermal@student" etc. = extra LoRA sets of the same modality (DMD roles); embeddings are per base modality
            cond_token_bias=(self.modality_embed[cond_mod.split("@")[0]].expand(B, -1) if cond is not None else None),
            target_token_bias=self.modality_embed[target_mod.split("@")[0]].expand(B, -1),
            attn_mode=self.cfg.model.attn_mode, shared_positions=bool(self.cfg.model.shared_positions),
            return_dict=False,
        )[0]
        # NOTE: do not clear the token groups here -- activation-checkpoint recomputation runs during
        # backward and must see the same groups as the original forward. They are overwritten next call.
        return out


def trainable_state(system: IclSystem, cfg) -> Dict[str, torch.Tensor]:
    sd = {f"modality_embed.{k}": v.detach().cpu() for k, v in system.modality_embed.items()}
    if cfg.model.trainable == "lora":
        sd.update({f"transformer.{k}": v.detach().cpu() for k, v in lora_state_dict(system.transformer).items()})
    else:
        sd.update({f"transformer.{k}": v.detach().to(torch.bfloat16).cpu() for k, v in system.transformer.state_dict().items()})
    return sd


# ----------------------------------------------------------------------------- text
def structured(cfg) -> bool:
    return str(cfg.get("captions", {}).get("mode", "fixed")) == "structured"


def all_prompts(cfg):
    if structured(cfg):
        from datasets.captions import all_captions
        return all_captions()
    return sorted(set(list(cfg.prompts.values()) + [""]))


def build_prompt_cache(cfg, model_dir: str, path: str, device):
    """Encode the few fixed prompts once (UMT5-XXL), save, drop the encoder."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    prompts = all_prompts(cfg)
    tok = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16).to(device).eval()
    out = {}
    with torch.no_grad():
        for p in prompts:
            ti = tok([p], padding="max_length", max_length=int(cfg.model.max_text_len), truncation=True,
                     add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
            ids, mask = ti.input_ids.to(device), ti.attention_mask.to(device)
            e = enc(ids, mask).last_hidden_state[0]
            L = int(mask[0].sum())
            e = torch.cat([e[:L], e.new_zeros(e.shape[0] - L, e.shape[1])])
            out[p] = e.to(torch.bfloat16).cpu()
    del enc
    torch.cuda.empty_cache()
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, extras = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    use_cuda = torch.cuda.is_available()
    import datetime  # long timeout: rank 0 runs the in-training sample eval while the other ranks wait at a barrier
    dist.init_process_group("nccl" if use_cuda else "gloo", timeout=datetime.timedelta(minutes=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.enable_cudnn_sdp(False)  # torch 2.5+/sm90 cudnn-SDPA segfaults (SkyWander note)
        torch.backends.cuda.matmul.allow_tf32 = True
    tiny = bool(cfg.get("debug_tiny", False))  # CPU dry run: random tiny transformer + fake latents, no weights/text encoder
    is_main = rank == 0
    seed = int(cfg.seed) + rank
    torch.manual_seed(seed); random.seed(seed)

    out_dir = str(cfg.output_dir)
    ckpt_dir = os.path.join(out_dir, "ckpt")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(out_dir, "config.yaml"))
    dist.barrier()
    log_f = open(os.path.join(out_dir, "train_log.jsonl"), "a") if is_main else None
    from tools.eval_video import EvalTimer, log_eval_clip, wandb_init
    wb = (wandb_init(str(cfg.get("wandb_project", "thermal_v2v")), os.path.basename(out_dir.rstrip("/")),
                     os.path.join(out_dir, "wandb"), OmegaConf.to_container(cfg, resolve=True))
          if is_main and not bool(cfg.get("debug_tiny", False)) else None)

    def log(msg, **kv):
        if is_main:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
            if kv:
                log_f.write(json.dumps(kv) + "\n"); log_f.flush()
                if wb is not None and "step" in kv:  # same numbers to wandb (nested dicts flattened)
                    flat = {}
                    for k, v in kv.items():
                        if isinstance(v, dict):
                            flat.update({f"{k}/{kk}": vv for kk, vv in v.items()})
                        elif isinstance(v, (int, float)) and k != "step":
                            flat[f"train/{k}"] = v
                    wb.log(flat, step=int(kv["step"]))

    # ---- text embeddings (fixed prompts)
    pc_path = os.path.join(out_dir, "prompt_embeds.pt")
    if tiny:
        prompt_embeds = {p: torch.randn(int(cfg.model.max_text_len), 32, device=device) for p in all_prompts(cfg)}
        lat_mean, lat_std = torch.zeros(1, 48, 1, 1, 1, device=device), torch.ones(1, 48, 1, 1, 1, device=device)
    else:
        if is_main and not os.path.exists(pc_path):
            log("encoding fixed prompts with UMT5-XXL ...")
            build_prompt_cache(cfg, cfg.model_dir, pc_path, device)
        dist.barrier()
        prompt_embeds = {k: v.to(device) for k, v in torch.load(pc_path).items()}
        # ---- VAE latent stats (no VAE weights needed)
        vcfg = json.load(open(os.path.join(cfg.model_dir, "vae", "config.json")))
        lat_mean = torch.tensor(vcfg["latents_mean"], device=device).view(1, -1, 1, 1, 1)
        lat_std = torch.tensor(vcfg["latents_std"], device=device).view(1, -1, 1, 1, 1)

    # ---- model
    weight_dtype = torch.bfloat16 if use_cuda else torch.float32
    if tiny:
        transformer = WanIclV2VTransformer3DModel(num_attention_heads=2, attention_head_dim=128, in_channels=48, out_channels=48,
                                                  text_dim=32, freq_dim=32, ffn_dim=128, num_layers=2).to(weight_dtype)
    else:
        transformer = WanIclV2VTransformer3DModel.from_pretrained(cfg.model_dir, subfolder="transformer", torch_dtype=weight_dtype)
    transformer.requires_grad_(False)
    if cfg.training.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if cfg.model.trainable == "lora":
        inject_gated_lora(transformer, MODALITIES, rank=int(cfg.model.lora.rank), alpha=float(cfg.model.lora.alpha),
                          target_modules=list(cfg.model.lora.target_modules), mode=str(cfg.model.lora.gating), dropout=float(cfg.model.lora.dropout))
        for p in lora_parameters(transformer):
            p.data = p.data.float(); p.requires_grad_(True)  # LoRA in fp32 (tiny), base bf16
    elif cfg.model.trainable == "full":
        transformer.requires_grad_(True)
    else:
        raise ValueError(cfg.model.trainable)
    system = IclSystem(transformer, cfg).to(device)
    for p in system.modality_embed.parameters():
        p.data = p.data.float()

    params = [p for p in system.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    log(f"trainable params: {n_train / 1e6:.1f}M mode={cfg.model.trainable} attn={cfg.model.attn_mode} shared_pos={cfg.model.shared_positions}")

    # ---- optimizer
    lr = float(cfg.training.lr)
    if cfg.model.trainable == "full":
        import bitsandbytes as bnb
        master = [p.detach().clone().float().requires_grad_(True) for p in params]
        opt = bnb.optim.AdamW8bit(master, lr=lr, betas=(0.9, 0.95), weight_decay=float(cfg.training.weight_decay))
    else:
        master = None
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=float(cfg.training.weight_decay))
    warm, total = int(cfg.training.warmup_steps), int(cfg.training.max_steps)
    min_ratio = float(cfg.training.lr_min_ratio)

    def lr_at(step):
        if step < warm:
            return lr * (step + 1) / max(warm, 1)
        pr = min(max((step - warm) / max(total - warm, 1), 0.0), 1.0)
        return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * pr)))

    # ---- resume
    step = 0
    latest = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    if cfg.training.resume and latest:
        ck = torch.load(latest[-1], map_location="cpu")
        missing = system.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["optimizer"]); step = int(ck["step"])
        if master is not None:
            for m_, p_ in zip(master, ck["master"]): m_.data.copy_(p_)
        log(f"resumed from {latest[-1]} at step {step} (unexpected keys: {len(missing.unexpected_keys)})")

    # gated LoRA: the clean-condition modality's LoRA (and, per step, the inactive direction's params) receive no
    # gradient -> DDP must tolerate unused parameters.
    ddp = torch.nn.parallel.DistributedDataParallel(system, device_ids=[local_rank] if use_cuda else None, find_unused_parameters=True,
                                                    gradient_as_bucket_view=True) if world > 1 else system

    # ---- data
    if tiny:
        train_ds, val_ds = FakeLatentPairs(8), FakeLatentPairs(2)
    else:
        to_py = lambda xs: [x if isinstance(x, str) else OmegaConf.to_container(x, resolve=True) for x in xs]
        train_ds = LatentPairs(to_py(cfg.data.train_latent_dirs), limit=int(cfg.data.get("limit", 0)))
        val_ds = LatentPairs(to_py(cfg.data.val_latent_dirs), limit=int(cfg.data.val_limit))
    if not tiny and any(not isinstance(d, str) for d in cfg.data.train_latent_dirs):
        sampler = WeightedDistributedSampler(train_ds.weights, world, rank, seed=int(cfg.seed))
    else:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=int(cfg.seed), drop_last=True)
    loader = DataLoader(train_ds, batch_size=int(cfg.training.batch_size), sampler=sampler, num_workers=int(cfg.data.num_workers),
                        collate_fn=collate, pin_memory=use_cuda, drop_last=True, persistent_workers=int(cfg.data.num_workers) > 0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)
    log(f"train clips {len(train_ds)} | val clips {len(val_ds)} | world {world} | bs/gpu {cfg.training.batch_size} x accum {cfg.training.accumulate}")

    # ---- flow matching
    sched = FlowMatchScheduler(shift=float(cfg.training.flow_shift), sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(1000, training=True)
    timesteps = sched.timesteps.to(device)
    dir_names = list(cfg.directions.keys()); dir_w = [float(cfg.directions[k]) for k in dir_names]
    p_cond_drop = float(cfg.training.cond_dropout)
    p_text_drop = float(cfg.training.text_dropout)

    def norm(z):
        return (z.to(device) - lat_mean) / lat_std

    def sample_t(B):
        if cfg.training.timestep_sampling == "logit_normal":
            u = torch.sigmoid(torch.randn(B, device=device))
            idx = (u * (len(timesteps) - 1)).round().long()
        else:
            idx = torch.randint(0, len(timesteps), (B,), device=device)
        return timesteps[idx]

    def text_for(batch, direction, train=True):
        """[B, L, D] text embeddings. Fixed mode: one prompt per direction. Structured mode: per-sample caption
        from the clip's sensor/media metadata (datasets/captions.py). Whole-caption dropout (-> "", the CFG
        unconditional) with p_text_drop in both modes."""
        B = len(batch["path"])
        if train and random.random() < p_text_drop:
            return prompt_embeds[""].unsqueeze(0).expand(B, -1, -1)
        if not structured(cfg):
            return prompt_embeds[cfg.prompts[direction]].unsqueeze(0).expand(B, -1, -1)
        from datasets.captions import caption
        caps = [caption(direction, s, m) for s, m in zip(batch["sensor"], batch["media"])]
        return torch.stack([prompt_embeds[c] for c in caps])

    def make_batch(batch, direction, train=True):
        """Returns (noisy_target, cond, t, text, cond_mod, target_mod, v_target, weight)."""
        cond_mod, target_mod = DIRECTIONS[direction]
        if direction == "joint":
            # joint generation = both streams noisy; implemented as target=thermal with a NOISY rgb cond at the same t
            cond_mod, target_mod = "rgb", "thermal"
        x_t = norm(batch[target_mod]); B = x_t.shape[0]
        t = sample_t(B)
        noise = torch.randn_like(x_t)
        noisy = sched.add_noise(x_t, noise, t)
        v = sched.training_target(x_t, noise, t)
        w = sched.training_weight(t)
        cond = norm(batch[cond_mod])
        if direction == "joint":
            cond = sched.add_noise(cond, torch.randn_like(cond), t)
        elif train and random.random() < p_cond_drop:
            cond = torch.zeros_like(cond)  # null condition for CFG
        text = text_for(batch, direction, train)
        return noisy.to(weight_dtype), cond.to(weight_dtype), t, text, cond_mod, target_mod, v, w

    @torch.no_grad()
    def validate():
        ddp.eval(); losses = {}
        for d in dir_names:
            if float(cfg.directions[d]) <= 0:
                continue
            tot, n = 0.0, 0
            for batch in val_loader:
                for tfix in (200.0, 500.0, 800.0):
                    cond_mod, target_mod = DIRECTIONS[d] if d != "joint" else ("rgb", "thermal")
                    x_t = norm(batch[target_mod]); noise = torch.randn_like(x_t)
                    t = torch.full((1,), tfix, device=device)
                    noisy = sched.add_noise(x_t, noise, t); v = sched.training_target(x_t, noise, t)
                    cond = norm(batch[cond_mod])
                    if d == "joint": cond = sched.add_noise(cond, torch.randn_like(cond), t)
                    text = text_for(batch, d, train=False)
                    with torch.autocast("cuda", dtype=weight_dtype, enabled=use_cuda):
                        pred = ddp(noisy.to(weight_dtype), cond.to(weight_dtype), t, text, cond_mod, target_mod)
                    tot += torch.nn.functional.mse_loss(pred.float(), v.float()).item(); n += 1
            tot_t = torch.tensor([tot, n], device=device); dist.all_reduce(tot_t)
            losses[d] = (tot_t[0] / tot_t[1]).item()
        ddp.train(); return losses

    # ---- ticket deadline (GPU-farm protocol): stop early, save, leave the marker -> requeued on the next ticket
    deadline = float(os.environ.get("FVF_TICKET_DEADLINE", os.environ.get("TICKET_DEADLINE", "0")) or 0)
    reserve = float(os.environ.get("FVF_CKPT_RESERVE", "600"))
    marker = os.environ.get("FVF_INCOMPLETE_MARKER", "")
    if deadline:
        log(f"ticket deadline in {deadline - time.time():.0f}s (reserve {reserve:.0f}s)")

    def out_of_time():
        flag = torch.tensor([1.0 if (deadline and time.time() > deadline - reserve) else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # all ranks must agree
        return flag.item() > 0

    # ---- in-training sample eval: every eval_every_minutes, a few fixed val clips per direction -> mp4 + metrics
    eval_timer = EvalTimer(float(cfg.training.get("eval_every_minutes", 60)), bool(cfg.training.get("eval_at_start", True)))
    eval_state = {}

    @torch.no_grad()
    def sample_eval(at_step):
        if tiny:
            return
        if is_main:
            from diffusers import AutoencoderKLWan
            if "vae" not in eval_state:
                eval_state["vae"] = AutoencoderKLWan.from_pretrained(cfg.model_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
            vae = eval_state["vae"]
            n = int(cfg.training.get("eval_clips", 2))
            files = val_ds.files[:: max(1, len(val_ds.files) // n)][:n]
            ev_sched = FlowMatchScheduler(shift=float(cfg.training.flow_shift), sigma_min=0.0, extra_one_step=True)
            ev_sched.set_timesteps(int(cfg.training.get("eval_steps", 30)))
            gscale = float(cfg.training.get("eval_cfg", 2.0))
            system.eval()
            dec = lambda z: vae.decode((z.float() * lat_std + lat_mean)).sample[0].float().clamp(-1, 1).permute(1, 0, 2, 3).cpu().numpy()
            for fi, f in enumerate(files):
                item = val_ds[val_ds.files.index(f)]
                for d in [x for x in ("rgb2thermal", "thermal2rgb") if float(cfg.directions.get(x, 0)) > 0]:
                    cm, tm = DIRECTIONS[d]
                    batch = {k: ([item[k]] if not torch.is_tensor(item[k]) else item[k][None]) for k in item}
                    text = text_for(batch, d, train=False)
                    cond = norm(batch[cm]).to(weight_dtype)
                    g = torch.Generator(device=device).manual_seed(1234 + fi)
                    x = torch.randn(cond.shape, generator=g, device=device, dtype=torch.float32)
                    for t in ev_sched.timesteps:
                        tt = t.view(1).to(device)
                        with torch.autocast("cuda", dtype=weight_dtype, enabled=use_cuda):
                            vc = system(x.to(weight_dtype), cond, tt, text, cm, tm).float()
                            vu = system(x.to(weight_dtype), torch.zeros_like(cond), tt, text, cm, tm).float() if gscale != 1.0 else vc
                        x = ev_sched.step(vu + gscale * (vc - vu), t, x)
                    log_eval_clip(os.path.join(out_dir, "eval"), at_step, f"{d}/clip{fi}",
                                  dec(norm(batch[cm])), dec(x), dec(norm(batch[tm])), run=wb,
                                  extra={"file": os.path.basename(f)}, thermal_target=(tm == "thermal"))
            system.train()
            torch.cuda.empty_cache()
        dist.barrier()

    def save_ckpt(tag_step):
        if is_main:
            ck = {"step": tag_step, "trainable": trainable_state(system, cfg), "optimizer": opt.state_dict(), "config": OmegaConf.to_container(cfg)}
            if master is not None: ck["master"] = [m_.detach().cpu() for m_ in master]
            tmp = os.path.join(ckpt_dir, f"step_{tag_step}.pt.tmp"); torch.save(ck, tmp); os.replace(tmp, tmp[:-4])
            olds = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
            keep_every = int(cfg.training.get("keep_milestone_every", 1000))  # milestone ckpts are never pruned (evals reference them)
            for o in olds[: -int(cfg.training.keep_ckpts)]:
                if keep_every and int(o.split("_")[-1][:-3]) % keep_every == 0:
                    continue
                os.remove(o)
            log(f"saved {tmp[:-4]}")
        dist.barrier()

    # ---- train loop
    ddp.train()
    accum = int(cfg.training.accumulate)
    t_start = time.time(); ema_loss = None
    epoch = step * accum // max(len(loader), 1)
    done = False
    while not done:
        sampler.set_epoch(epoch); epoch += 1
        it = iter(loader)
        while True:
            try:
                micro = [next(it) for _ in range(accum)]
            except StopIteration:
                break
            for g in opt.param_groups: g["lr"] = lr_at(step)
            # one direction per optimizer step, broadcast from rank 0 (same sequence layout on all ranks)
            sel = torch.tensor([random.choices(range(len(dir_names)), weights=dir_w)[0]], device=device)
            dist.broadcast(sel, 0); direction = dir_names[int(sel)]
            loss_acc = 0.0
            for mi, batch in enumerate(micro):
                noisy, cond, t, text, cm, tm, v, w = make_batch(batch, direction)
                with torch.autocast("cuda", dtype=weight_dtype, enabled=use_cuda):
                    pred = ddp(noisy, cond, t, text, cm, tm)
                loss = (torch.nn.functional.mse_loss(pred.float(), v.float(), reduction="none").mean(dim=(1, 2, 3, 4)) * w).mean() / accum
                loss.backward(); loss_acc += loss.item()
            if master is not None:
                for m_, p_ in zip(master, params):
                    m_.grad = p_.grad.float() if p_.grad is not None else None
                gn = torch.nn.utils.clip_grad_norm_(master, float(cfg.training.grad_clip))
                opt.step(); opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for m_, p_ in zip(master, params): p_.data.copy_(m_.data)
                for p_ in params: p_.grad = None
            else:
                gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.training.grad_clip))
                opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            lt = torch.tensor([loss_acc], device=device); dist.all_reduce(lt); loss_acc = lt.item() / world
            ema_loss = loss_acc if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_acc
            if step % int(cfg.training.log_every) == 0 or step == 1:
                el = time.time() - t_start
                log(f"step {step} {direction} loss {loss_acc:.4f} ema {ema_loss:.4f} gn {float(gn):.2f} lr {lr_at(step - 1):.2e} "
                    f"{el / max(step, 1):.1f}s/step mem {(torch.cuda.max_memory_allocated() / 2**30) if use_cuda else 0:.1f}G",
                    step=step, direction=direction, loss=loss_acc, ema=ema_loss, grad_norm=float(gn), lr=lr_at(step - 1), elapsed=el)
            if step % int(cfg.training.val_every) == 0:
                vl = validate(); log(f"val @ {step}: {vl}", step=step, val=vl)
            if eval_timer.due():
                sample_eval(step)
            if step % int(cfg.training.save_every) == 0 or step >= total:
                save_ckpt(step)
            if step >= total:
                done = True; break
            if out_of_time():
                if step % int(cfg.training.save_every) != 0:
                    save_ckpt(step)
                if is_main and marker:
                    open(marker, "w").write(f"step {step} of {total}\n")
                log(f"ticket ending: saved at step {step}/{total}, exiting for requeue")
                done = True; break
    log("DONE"); dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
