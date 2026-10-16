"""Inference / evaluation for the ICL V2V model: condition video -> target video, with CFG on the condition.

Reads a training checkpoint (trainable-only state) + the config saved next to it, runs Euler flow-matching
sampling over pre-encoded val latents (or raw clips), decodes with the Wan VAE, writes PNG grids / mp4 and
paired metrics (PSNR / SSIM / LPIPS in the normalised target space, plus a flicker index).

  PYTHONPATH=. python src/infer_icl.py --run_dir <output_dir of training> [--ckpt step_N.pt] \
      --direction rgb2thermal --num_clips 8 --steps 30 --cfg 2.0 --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from models.icl.transformer import WanIclV2VTransformer3DModel
from models.icl.gated_lora import inject_gated_lora
from src.train_icl import DIRECTIONS, IclSystem, MODALITIES
from tools.flow_schedule import FlowMatchScheduler


def to_u8(x):  # [3,H,W] in [-1,1]
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default=None, help="file name under run_dir/ckpt (default: latest)")
    ap.add_argument("--direction", default="rgb2thermal", choices=list(DIRECTIONS))
    ap.add_argument("--latent_dir", default=None, help="default: cfg.data.val_latent_dirs[0]")
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=2.0, help="guidance on the condition video (1.0 = off)")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--save_tensors", action="store_true", help="save decoded pred/gt (fp16) per clip for offline re-scoring")
    ap.add_argument("--fv_teacher", default=None, help="FastVideo mode: merged teacher dir (.../Wan2.2-TI2V-5B-Diffusers)")
    ap.add_argument("--fv_student", default=None, help="FastVideo mode: student LoRA safetensors (generator_inference_transformer "
                    "or ema/generator_ema.safetensors); omit to evaluate the merged teacher itself")
    ap.add_argument("--fv_steps", default="1000,757,522", help="FastVideo DMD denoising steps (sigma*1000)")
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(a.run_dir, "config.yaml"))
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.enable_cudnn_sdp(False)
    ckpts = sorted(glob.glob(os.path.join(a.run_dir, "ckpt", "step_*.pt")) + glob.glob(os.path.join(a.run_dir, "ckpt", "gstep_*.pt")),
                   key=lambda p: int(p.split("_")[-1][:-3]))
    if a.fv_teacher:
        ckpt_path, ck = None, None
    else:
        ckpt_path = os.path.join(a.run_dir, "ckpt", a.ckpt) if a.ckpt else ckpts[-1]
        ck = torch.load(ckpt_path, map_location="cpu")
    os.makedirs(a.out, exist_ok=True)

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(cfg.model_dir, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    lat_mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    lat_std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)

    fv_mode = a.fv_teacher is not None
    if fv_mode:  # FastVideo-distilled student: merged teacher weights + student LoRA merged in memory, no LoRA layers
        cfg.model.trainable = "full"
        transformer = WanIclV2VTransformer3DModel.from_pretrained(a.fv_teacher, subfolder="transformer", torch_dtype=torch.float32)
        from safetensors import safe_open as _so
        with _so(os.path.join(a.fv_teacher, "transformer", "diffusion_pytorch_model.safetensors"), "pt") as tf:
            emb = {"rgb": tf.get_tensor("cond_mod_emb"), "thermal": tf.get_tensor("tgt_mod_emb")}
        if a.fv_student:
            import re
            lora = load_file(a.fv_student)
            sd = transformer.state_dict(); n = 0
            def fv2diff(name):  # FastVideo -> diffusers linear names
                name = re.sub(r"^blocks\.(\d+)\.to_(q|k|v)$", r"blocks.\1.attn1.to_\2", name)
                name = re.sub(r"^blocks\.(\d+)\.to_out$", r"blocks.\1.attn1.to_out.0", name)
                name = re.sub(r"^blocks\.(\d+)\.attn2\.to_out$", r"blocks.\1.attn2.to_out.0", name)
                name = re.sub(r"^blocks\.(\d+)\.ffn\.fc_in$", r"blocks.\1.ffn.net.0.proj", name)
                name = re.sub(r"^blocks\.(\d+)\.ffn\.fc_out$", r"blocks.\1.ffn.net.2", name)
                return name
            for k in lora:
                if not k.endswith(".lora_A"):
                    continue
                base = fv2diff(k[: -len(".lora_A")])
                A, B = lora[k].float(), lora[k[: -len("A")] + "B"].float()
                wk = base + ".weight"; assert wk in sd, (k, wk)
                sd[wk] = sd[wk] + (B @ A)  # FastVideo LoRA alpha == rank -> scale 1
                n += 1
            transformer.load_state_dict(sd)
            print(f"FastVideo student: merged {n} LoRA linears from {a.fv_student}")
        transformer = transformer.to(dtype)
    else:
        transformer = WanIclV2VTransformer3DModel.from_pretrained(cfg.model_dir, subfolder="transformer", torch_dtype=dtype)
    is_dmd = bool(ck.get("dmd", False)) if ck is not None else False
    mods = tuple(MODALITIES) + (("thermal@student", "thermal@fake") if is_dmd else ())
    if cfg.model.trainable == "lora":
        inject_gated_lora(transformer, mods, rank=int(cfg.model.lora.rank), alpha=float(cfg.model.lora.alpha),
                          target_modules=list(cfg.model.lora.target_modules), mode=str(cfg.model.lora.gating))
    system = IclSystem(transformer, cfg)
    if fv_mode:
        with torch.no_grad():
            system.modality_embed["rgb"].copy_(emb["rgb"]); system.modality_embed["thermal"].copy_(emb["thermal"])
    if is_dmd:  # teacher weights (embeddings + base-modality LoRAs) first, then the student/fake LoRAs
        tck = torch.load(ck["teacher_ckpt"], map_location="cpu")
        _, unexpected = system.load_state_dict(tck["trainable"], strict=False); assert not unexpected, unexpected[:5]
        print(f"DMD student; teacher {ck['teacher_ckpt']}")
    if not fv_mode:
        missing, unexpected = system.load_state_dict(ck["trainable"], strict=False)
        assert not unexpected, unexpected[:5]
    else:
        missing = []
    system.to(device).eval()
    print(f"loaded {'FastVideo ' + str(a.fv_student or a.fv_teacher) if fv_mode else ckpt_path}; missing={len(missing)}")

    prompt_embeds = torch.load(os.path.join(a.run_dir, "prompt_embeds.pt"))
    structured = str(cfg.get("captions", {}).get("mode", "fixed")) == "structured"
    text = None if structured else prompt_embeds[cfg.prompts[a.direction]].to(device).unsqueeze(0)
    cond_mod, target_mod = DIRECTIONS[a.direction]
    key_cond, key_target = ("cond", "target") if cond_mod == "rgb" else ("target", "cond")

    lat_dir = a.latent_dir or list(cfg.data.val_latent_dirs)[0]
    if not isinstance(lat_dir, str):
        lat_dir = str(lat_dir["dir"])
    files = sorted(glob.glob(os.path.join(lat_dir, "*.safetensors")))
    files = files[:: max(1, len(files) // a.num_clips)][: a.num_clips]

    sched = FlowMatchScheduler(shift=a.shift, sigma_min=0.0, extra_one_step=True)
    sched.set_timesteps(a.steps)
    lp = None
    if a.lpips:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device).eval()

    results = []
    for fi, f in enumerate(files):
        d = load_file(f)
        from safetensors import safe_open
        with safe_open(f, "pt") as sf:
            sf_meta = sf.metadata() or {}
            meta = json.loads(sf_meta.get("meta", "{}"))
        if structured:  # per-clip caption from the clip's sensor / media metadata (datasets/captions.py)
            from datasets.captions import caption
            text = prompt_embeds[caption(a.direction, meta.get("sensor", "radiometric"), meta.get("media", "video"))]
            text = text.to(device).unsqueeze(0)
        cond = ((d[key_cond].float().to(device).unsqueeze(0) - lat_mean) / lat_std).to(dtype)
        gt_lat = d[key_target].float().to(device).unsqueeze(0)
        g = torch.Generator(device=device).manual_seed(a.seed + fi)
        x = torch.randn(gt_lat.shape, generator=g, device=device, dtype=torch.float32)
        null_cond = torch.zeros_like(cond)
        torch.cuda.synchronize(); t_samp = time.time(); nfe = 0
        if is_dmd or (fv_mode and a.fv_student):  # DMD2 multi-step sampler: predict x0 at each sigma, re-noise to the next
            sig = [float(v) / 1000.0 for v in (a.fv_steps.split(",") if fv_mode else cfg.dmd.student_timesteps)]
            role = target_mod if fv_mode else "thermal@student"
            for k, sgm in enumerate(sig):
                with torch.autocast("cuda", dtype=dtype):
                    v = system(x.to(dtype), cond, torch.full((1,), sgm * 1000.0, device=device), text, cond_mod, role).float()
                nfe += 1
                x0 = x - sgm * v
                x = x0 if k == len(sig) - 1 else (1 - sig[k + 1]) * x0 + sig[k + 1] * torch.randn(x0.shape, generator=g, device=device)
        else:
            for t in sched.timesteps:
                tt = t.view(1).to(device)
                with torch.autocast("cuda", dtype=dtype):
                    v_c = system(x.to(dtype), cond, tt, text, cond_mod, target_mod).float(); nfe += 1
                    if a.cfg != 1.0:
                        v_u = system(x.to(dtype), null_cond, tt, text, cond_mod, target_mod).float(); nfe += 1
                        v = v_u + a.cfg * (v_c - v_u)
                    else:
                        v = v_c
                x = sched.step(v, t, x)
        torch.cuda.synchronize(); sample_sec = time.time() - t_samp
        pred_lat = x * lat_std + lat_mean
        pred = vae.decode(pred_lat.float()).sample.clamp(-1, 1)
        gt = vae.decode(gt_lat).sample.clamp(-1, 1)          # VAE-roundtrip GT (what the model can at best reproduce)
        cond_px = vae.decode(d[key_cond].float().to(device).unsqueeze(0)).sample.clamp(-1, 1)
        # --- common "clip space" for thermal targets: undo the training normalisation to 16-bit counts, then
        # re-normalise BOTH pred and gt with the GT clip's own percentiles (dataset clip mode: p0.5/p99.5, 5% margin).
        # Makes clip-norm and global-norm runs comparable (global-norm frames occupy ~5% of [-1,1] -> inflated raw PSNR).
        norm_used = sf_meta.get("norm", "clip")
        if target_mod == "thermal" and meta.get("lo") is not None:
            lo_f, hi_f = float(meta["lo"]), float(meta["hi"])
            to_counts = lambda x: (x.clamp(-1, 1) + 1) / 2 * (hi_f - lo_f) + lo_f
            gt_cnt, pred_cnt = to_counts(gt[:, :1]), to_counts(pred[:, :1])
            qlo, qhi = torch.quantile(gt_cnt.flatten()[::37].float(), torch.tensor([0.005, 0.995], device=device))
            span = max(float(qhi - qlo), 1.0); lo_c, hi_c = float(qlo) - 0.05 * span, float(qhi) + 0.05 * span
            to_clip = lambda c: ((c - lo_c) / max(hi_c - lo_c, 1.0)).clamp(0, 1) * 2 - 1
            gt_cs, pred_cs = to_clip(gt_cnt).expand(-1, 3, -1, -1, -1), to_clip(pred_cnt).expand(-1, 3, -1, -1, -1)
        else:
            gt_cs, pred_cs = gt, pred
        mse_cs = ((pred_cs - gt_cs) ** 2).mean().item(); psnr_cs = 10 * np.log10(4.0 / max(mse_cs, 1e-10))
        T = pred.shape[2]
        from skimage.metrics import structural_similarity
        ssim_cs = float(np.mean([structural_similarity(gt_cs[0, 0, k].cpu().numpy(), pred_cs[0, 0, k].cpu().numpy(), data_range=2.0) for k in range(0, T, max(1, T // 4))]))
        # metrics (target space, [-1,1]); thermal: use channel 0
        mse = ((pred - gt) ** 2).mean().item()
        psnr = 10 * np.log10(4.0 / max(mse, 1e-10))
        ssim = float(np.mean([structural_similarity(gt[0, 0, k].cpu().numpy(), pred[0, 0, k].cpu().numpy(), data_range=2.0) for k in range(0, T, max(1, T // 4))]))
        # affine-invariant (scale/shift-aligned) metrics: fit pred*a+b ~ gt per clip (least squares) to separate the
        # unknowable per-clip temperature range from structural error (depth-estimation style)
        pf, gf = pred[0, 0].flatten().double(), gt[0, 0].flatten().double()
        # closed-form 1-D least squares (cuSOLVER lstsq fails on ~10M-row matrices, e.g. 33 frames at 640x512)
        pm, gm = pf.mean(), gf.mean()
        a_ = ((pf - pm) * (gf - gm)).sum() / ((pf - pm) ** 2).sum().clamp_min(1e-12)
        sol = (a_, gm - a_ * pm)
        pred_al = (pred[0, 0].double() * sol[0] + sol[1]).clamp(-1, 1).float()
        mse_al = ((pred_al - gt[0, 0]) ** 2).mean().item(); psnr_al = 10 * np.log10(4.0 / max(mse_al, 1e-10))
        ssim_al = float(np.mean([structural_similarity(gt[0, 0, k].cpu().numpy(), pred_al[k].cpu().numpy(), data_range=2.0) for k in range(0, T, max(1, T // 4))]))
        flick_pred = pred[0, 0].mean(dim=(1, 2)).diff().abs().mean().item()  # mean |Δ global intensity| per frame
        flick_gt = gt[0, 0].mean(dim=(1, 2)).diff().abs().mean().item()
        r = {"file": os.path.basename(f), "sample_sec": sample_sec, "nfe": float(nfe), "psnr": psnr, "ssim": ssim, "psnr_clipspace": psnr_cs, "ssim_clipspace": ssim_cs,
             "norm": norm_used, "psnr_affine": psnr_al, "ssim_affine": ssim_al,
             "affine_a": float(sol[0]), "affine_b": float(sol[1]), "flicker_pred": flick_pred, "flicker_gt": flick_gt,
             **{k: meta.get(k) for k in ("seq", "start", "lo", "hi")}}
        if lp is not None:
            r["lpips"] = float(np.mean([lp(pred[0, :, k][None], gt[0, :, k][None]).item() for k in range(0, T, max(1, T // 4))]))
        results.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        # grid: rows = frames (0, mid, last); cols = cond | pred | gt | |err|
        rows = []
        for k in (0, T // 2, T - 1):
            err = ((pred_cs[0, :, k] - gt_cs[0, :, k]).abs() * 4 - 1).clamp(-1, 1)
            rows.append(np.concatenate([to_u8(cond_px[0, :, k]), to_u8(pred_cs[0, :, k]), to_u8(gt_cs[0, :, k]), to_u8(err)], 1))
        if a.save_tensors:
            torch.save({"pred": pred[0].half().cpu(), "gt": gt[0].half().cpu(), "meta": meta, "norm": norm_used},
                       os.path.join(a.out, r["file"].replace(".safetensors", ".pt")))
        from PIL import Image
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, r["file"].replace(".safetensors", ".png")))
        try:
            import imageio
            frames = [np.concatenate([to_u8(cond_px[0, :, k]), to_u8(pred_cs[0, :, k]), to_u8(gt_cs[0, :, k])], 1) for k in range(T)]
            imageio.mimwrite(os.path.join(a.out, r["file"].replace(".safetensors", ".mp4")), frames, fps=10, quality=7)
        except Exception as e:  # noqa: BLE001
            print("mp4 skipped:", e)
    agg = {k: float(np.mean([r[k] for r in results])) for k in results[0] if isinstance(results[0][k], float)}
    json.dump({"ckpt": ckpt_path or a.fv_student or a.fv_teacher, "direction": a.direction, "cfg": a.cfg, "steps": a.steps, "per_clip": results, "mean": agg},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print("MEAN", json.dumps({k: round(v, 4) for k, v in agg.items()}))


if __name__ == "__main__":
    main()
