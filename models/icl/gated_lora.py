"""Token-group gated LoRA (UniVidX "Decoupled Gated LoRA" recipe, adapted to token-concat ICL).

The ICL sequence is [cond tokens | target tokens]. Each modality owns a LoRA; a LoRA is applied only
to the tokens of its modality AND only when that modality is the noisy generation target. Clean
condition tokens pass through the frozen base weights (gate = 0), as in UniVidX.

Usage:
    inject_gated_lora(transformer, modalities=["rgb", "thermal"], rank=64, alpha=64,
                      target_modules=("to_q","to_k","to_v","to_out.0","ffn.net.0.proj","ffn.net.2"))
    set_token_groups(transformer, groups=[("rgb", n_cond, False), ("thermal", n_target, True)])
    out = transformer(...)          # every GatedLoRALinear reads the groups from the shared context
    set_token_groups(transformer, None)

`groups` is an ordered list of (modality, num_tokens, is_target) matching the sequence layout.
Cross-attention K/V projections (operating on text tokens) are skipped automatically: a layer only
applies the gated path when the incoming sequence length equals the sum of the group sizes; other
inputs (text context) use the base weight. `mode="all"` applies the single adapter to every token
(= plain LoRA, WorldWander-style), useful as an ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class TokenGroups:
    groups: List[Tuple[str, int, bool]]  # (modality, num_tokens, is_target)

    @property
    def total(self) -> int:
        return sum(n for _, n, _ in self.groups)


class _GroupContext:
    """Shared, mutable pointer so every GatedLoRALinear sees the same groups without plumbing kwargs."""

    def __init__(self):
        self.groups: Optional[TokenGroups] = None


class GatedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, modalities: Sequence[str], rank: int, alpha: float, ctx: _GroupContext,
                 mode: str = "gated", dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.ctx = ctx
        self.mode = mode
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.ParameterDict({m: nn.Parameter(torch.empty(rank, in_f)) for m in modalities})
        self.lora_B = nn.ParameterDict({m: nn.Parameter(torch.zeros(out_f, rank)) for m in modalities})
        for m in modalities:
            nn.init.kaiming_uniform_(self.lora_A[m], a=math.sqrt(5))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def _delta(self, x: torch.Tensor, m: str) -> torch.Tensor:
        A, B = self.lora_A[m], self.lora_B[m]
        return (self.dropout(x) @ A.t().to(x.dtype)) @ B.t().to(x.dtype) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        g = self.ctx.groups
        if g is None or x.ndim != 3 or x.shape[1] != g.total:
            return y  # text tokens / no context set -> frozen base only
        if self.mode == "all":
            m = next(iter(self.lora_A.keys()))
            return y + self._delta(x, m)
        # out-of-place (in-place slice writes break activation-checkpoint recomputation bookkeeping)
        parts, off, any_lora = [], 0, False
        for m, n, is_target in g.groups:
            xs = x[:, off:off + n]
            if is_target and m in self.lora_A:
                parts.append(self._delta(xs, m)); any_lora = True
            else:
                parts.append(xs.new_zeros(xs.shape[0], n, y.shape[-1]))
            off += n
        if not any_lora:
            return y
        return y + torch.cat(parts, dim=1)


def _get_parent(root: nn.Module, name: str):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_gated_lora(
    transformer: nn.Module,
    modalities: Sequence[str],
    rank: int = 64,
    alpha: Optional[float] = None,
    target_modules: Iterable[str] = ("to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"),
    mode: str = "gated",
    only_self_attn: bool = False,
    dropout: float = 0.0,
) -> _GroupContext:
    """Replace matching nn.Linear layers inside `transformer.blocks` with GatedLoRALinear."""
    alpha = rank if alpha is None else alpha
    ctx = _GroupContext()
    targets = tuple(target_modules)
    n = 0
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear) or not name.startswith("blocks."):
            continue
        if only_self_attn and ".attn2." in name:
            continue
        if not any(name.endswith(t) for t in targets):
            continue
        parent, attr = _get_parent(transformer, name)
        setattr(parent, attr, GatedLoRALinear(module, modalities, rank, alpha, ctx, mode=mode, dropout=dropout))
        n += 1
    if n == 0:
        raise RuntimeError("inject_gated_lora: no target modules matched")
    transformer._gated_lora_ctx = ctx
    return ctx


def set_token_groups(transformer: nn.Module, groups: Optional[List[Tuple[str, int, bool]]]):
    ctx: _GroupContext = transformer._gated_lora_ctx
    ctx.groups = TokenGroups(list(groups)) if groups is not None else None


def lora_state_dict(transformer: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in transformer.state_dict().items() if ".lora_A." in k or ".lora_B." in k}


def lora_parameters(transformer: nn.Module) -> List[nn.Parameter]:
    return [p for n, p in transformer.named_parameters() if ".lora_A." in n or ".lora_B." in n]
