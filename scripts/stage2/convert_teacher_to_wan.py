"""Convert our LoRA-merged diffusers Wan2.2-TI2V-5B teacher into original-Wan key names for NVlabs/LongLive 2.0.

LongLive's CausalWanModel (LongLive2.0/wan_5b/modules/causal_model.py) uses the original Wan layout
(blocks.N.self_attn.q, ffn.0, modulation, head.head, ...), not diffusers names. Our ICL modality embeddings
(cond_mod_emb / tgt_mod_emb) have no counterpart there; they are kept under the same names so the port can add
them as extra parameters.

  python scripts/longlive/convert_teacher_to_wan.py \
      --src <merged teacher>/Wan2.2-TI2V-5B-Diffusers/transformer --out <dir>/teacher_wan.safetensors
(scripts/stage2/export_teacher.py does merge + convert in one step and is what the pipeline uses.)
"""
import argparse, glob, os, re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK = {
    "attn1.to_q": "self_attn.q", "attn1.to_k": "self_attn.k", "attn1.to_v": "self_attn.v", "attn1.to_out.0": "self_attn.o",
    "attn1.norm_q": "self_attn.norm_q", "attn1.norm_k": "self_attn.norm_k",
    "attn2.to_q": "cross_attn.q", "attn2.to_k": "cross_attn.k", "attn2.to_v": "cross_attn.v", "attn2.to_out.0": "cross_attn.o",
    "attn2.norm_q": "cross_attn.norm_q", "attn2.norm_k": "cross_attn.norm_k",
    "norm2": "norm3", "ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2",
}
TOP = {
    "condition_embedder.text_embedder.linear_1": "text_embedding.0", "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0", "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1", "proj_out": "head.head", "patch_embedding": "patch_embedding",
}
KEEP = {"cond_mod_emb", "tgt_mod_emb"}


def convert_key(k):
    if k in KEEP:
        return k
    if k == "scale_shift_table":
        return "head.modulation"
    m = re.match(r"blocks\.(\d+)\.(.+)\.(weight|bias)$", k)
    if m:
        return f"blocks.{m.group(1)}.{BLOCK[m.group(2)]}.{m.group(3)}"
    m = re.match(r"blocks\.(\d+)\.scale_shift_table$", k)
    if m:
        return f"blocks.{m.group(1)}.modulation"
    m = re.match(r"(.+)\.(weight|bias)$", k)
    if m and m.group(1) in TOP:
        return f"{TOP[m.group(1)]}.{m.group(2)}"
    raise KeyError(f"no mapping for {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="diffusers transformer dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    sd = {}
    for f in sorted(glob.glob(f"{a.src}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                nk = convert_key(k)
                assert nk not in sd, f"duplicate target {nk}"
                sd[nk] = s.get_tensor(k).to(dt).contiguous()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_file(sd, a.out, metadata={"source": os.path.abspath(a.src), "format": "original-wan"})
    print(f"wrote {len(sd)} tensors -> {a.out}")


if __name__ == "__main__":
    main()
