"""Stage 1 -> stage 2 hand-off: merge a stage-1 checkpoint (plain LoRA, gating=all) into Wan2.2-TI2V-5B and write it
in LongLive's original-Wan layout, ready for AR training (LLV2_MODEL_DIR).

  python scripts/stage2/export_teacher.py --run_dir $THERMAL_RUNS/stage1_bidir_33f --ckpt step_3000.pt \
      --out $THERMAL_RUNS/stage2_teacher

Writes <out>/{config.json, diffusion_pytorch_model.safetensors, prompt_embeds.pt, export_info.json}.
- The merge is exact for gating=all: the LoRA acts on every token, so W' = W + (alpha/r) B A. (gating=gated keeps a
  separate LoRA per generated modality and cannot be folded into one weight; it is rejected.)
- The rgb2thermal view is exported: modality embedding "rgb" -> cond_mod_emb (RGB frames), "thermal" ->
  tgt_mod_emb (thermal frames), the roles LongLive's interleaved model adds them in.
- prompt_embeds.pt is the stage-1 UMT5 cache of every structured caption (LLV2_PROMPT_EMBEDS for stage 2).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_teacher_to_wan import convert_key  # noqa: E402

WAN_CONFIG = {"_class_name": "WanModel", "_diffusers_version": "0.33.0", "dim": 3072, "eps": 1e-06, "ffn_dim": 14336,
              "freq_dim": 256, "in_dim": 48, "model_type": "ti2v", "num_heads": 24, "num_layers": 30, "out_dim": 48,
              "text_len": 512}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", required=True, help="file name under <run_dir>/ckpt, e.g. step_3000.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(a.run_dir, "config.yaml"))
    if not (cfg.model.trainable == "lora" and cfg.model.lora.gating == "all"):
        sys.exit("export needs a plain-LoRA stage-1 run (model.trainable=lora, model.lora.gating=all)")
    scale = float(cfg.model.lora.alpha) / float(cfg.model.lora.rank)
    adapter = "rgb"  # GatedLoRALinear mode='all' uses the first registered modality's adapter for every token
    ck = torch.load(os.path.join(a.run_dir, "ckpt", a.ckpt), map_location="cpu")["trainable"]

    sd = {}
    for f in sorted(glob.glob(os.path.join(str(cfg.model_dir), "transformer", "*.safetensors"))):
        sd.update(load_file(f))
    n = 0
    for k in list(ck):
        if k.endswith(f".lora_A.{adapter}"):
            mod = k[len("transformer."):-len(f".lora_A.{adapter}")]
            A, B = ck[k].float(), ck[k.replace("lora_A", "lora_B")].float()
            w = f"{mod}.weight"
            assert w in sd, w
            sd[w] = (sd[w].float() + scale * (B @ A)).to(sd[w].dtype)
            n += 1
    sd["cond_mod_emb"] = ck["modality_embed.rgb"].float()
    sd["tgt_mod_emb"] = ck["modality_embed.thermal"].float()

    out_sd = {convert_key(k): v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    os.makedirs(a.out, exist_ok=True)
    save_file(out_sd, os.path.join(a.out, "diffusion_pytorch_model.safetensors"), metadata={"format": "original-wan"})
    json.dump(WAN_CONFIG, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    shutil.copy(os.path.join(a.run_dir, "prompt_embeds.pt"), os.path.join(a.out, "prompt_embeds.pt"))
    json.dump({"run_dir": os.path.abspath(a.run_dir), "ckpt": a.ckpt, "lora_scale": scale, "merged_linears": n,
               "tensors": len(out_sd)}, open(os.path.join(a.out, "export_info.json"), "w"), indent=1)
    print(f"merged {n} LoRA linears (scale {scale}), wrote {len(out_sd)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
