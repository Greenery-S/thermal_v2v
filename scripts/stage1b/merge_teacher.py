"""Stage 1 -> stage 1b hand-off: merge a stage-1 checkpoint (plain LoRA, gating=all) into a full diffusers
Wan2.2-TI2V-5B directory for FastVideo DMD (rgb2thermal view).

Exact only for gating=all (the LoRA acts on every token, so W' = W + (alpha/r) B A reproduces it). In our trainer the
'all' mode uses the FIRST modality's adapter (MODALITIES[0] = "rgb"). The two modality embeddings become top-level
params `cond_mod_emb` (rgb, added to condition tokens) and `tgt_mod_emb` (thermal, added to target tokens) that the
patched FastVideo WanTransformer3DModel defines.

Output dir is named .../Wan2.2-TI2V-5B-Diffusers (FastVideo picks its pipeline config by the basename) and gets
symlinks to vae/text_encoder/tokenizer/scheduler + model_index.json from the base directory.

  python scripts/stage1b/merge_teacher.py --run_dir $THERMAL_RUNS/stage1_bidir_33f --ckpt step_3000.pt \
      --out $THERMAL_RUNS/stage1b_teacher/Wan2.2-TI2V-5B-Diffusers
"""
import argparse, glob, json, os, shutil

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cond_mod", default="rgb"); ap.add_argument("--target_mod", default="thermal")
a = ap.parse_args()

cfg = OmegaConf.load(os.path.join(a.run_dir, "config.yaml"))
assert cfg.model.trainable == "lora" and cfg.model.lora.gating == "all", "exact merge needs plain LoRA (gating=all)"
adapter = "rgb"  # GatedLoRALinear mode='all' uses the first registered modality's adapter
scale = float(cfg.model.lora.alpha) / float(cfg.model.lora.rank)
base = str(cfg.model_dir)
ck = torch.load(os.path.join(a.run_dir, "ckpt", a.ckpt), map_location="cpu")["trainable"]

sd = {}
for f in sorted(glob.glob(os.path.join(base, "transformer", "*.safetensors"))):
    sd.update(load_file(f))
n = 0
for k in list(ck):
    if k.endswith(f".lora_A.{adapter}"):
        mod = k[len("transformer."):-len(f".lora_A.{adapter}")]          # e.g. blocks.0.attn1.to_q
        A = ck[k].float(); B = ck[k.replace("lora_A", "lora_B")].float()
        wk = f"{mod}.weight"
        assert wk in sd, wk
        sd[wk] = (sd[wk].float() + scale * (B @ A)).to(sd[wk].dtype); n += 1
sd["cond_mod_emb"] = ck[f"modality_embed.{a.cond_mod}"].float().to(torch.float32)
sd["tgt_mod_emb"] = ck[f"modality_embed.{a.target_mod}"].float().to(torch.float32)
print(f"merged {n} LoRA linears (adapter '{adapter}', scale {scale}); cond_mod_emb |.|={sd['cond_mod_emb'].norm():.3f} tgt_mod_emb |.|={sd['tgt_mod_emb'].norm():.3f}")

out = a.out.rstrip("/")
assert os.path.basename(out) == "Wan2.2-TI2V-5B-Diffusers", "basename must stay Wan2.2-TI2V-5B-Diffusers for FastVideo's config registry"
os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
shutil.copy(os.path.join(base, "transformer", "config.json"), os.path.join(out, "transformer", "config.json"))
save_file(sd, os.path.join(out, "transformer", "diffusion_pytorch_model.safetensors"))
for comp in ("vae", "text_encoder", "tokenizer", "scheduler"):
    dst = os.path.join(out, comp)
    if not os.path.exists(dst):
        os.symlink(os.path.join(base, comp), dst)
shutil.copy(os.path.join(base, "model_index.json"), os.path.join(out, "model_index.json"))
json.dump({"source_run": a.run_dir, "ckpt": a.ckpt, "adapter": adapter, "scale": scale, "n_merged": n},
          open(os.path.join(out, "merge_info.json"), "w"), indent=1)
print("wrote", out)
