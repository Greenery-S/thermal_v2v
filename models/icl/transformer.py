"""Wan2.2 transformer with in-context (token-concatenation) V2V conditioning.

Sequence layout fed to every self-attention layer:

    [ cond tokens (clean, t=0) | target tokens (noisy, t) ]

* cond and target are patchified separately with the SAME patch embedding and
  concatenated along the token axis (no ControlNet / VACE branch, no channel concat).
* RoPE: cond tokens reuse the target's (f, h, w) position ids ("shared positions",
  WorldWander collaborative PE / ICDepth / Cosmos3-Transfer). If the cond is at a lower
  spatial resolution, its ids are rescaled into target coordinates (OminiControl / LTX
  IC-LoRA style), so pixel alignment is preserved.
* Timestep: Wan2.2-TI2V-5B natively supports per-token timesteps; cond tokens get t=0.
* Attention: "full" (bidirectional) or "sand" (cond->cond only, target->all).
* Optional additive per-stream token biases (e.g. a learned modality / direction
  embedding) are passed in from outside, so the base checkpoint loads unchanged.
* Only target-token predictions are returned.

Adapted from WorldWander/SkyWander models/uav/transformer_wan_icl.py with the UAV pose,
reference-frame and causal-AR paths removed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from models.icl.wan_embedder import WanTimeTextImageEmbedding, WanRotaryPosEmbed
from models.icl.wan_block import IclAttnSpec, WanTransformerBlock
from models.icl.position_utils import (
    concat_rotary_emb,
    make_corrected_condition_position_ids,
    make_grid_position_ids,
    patch_grid_from_latents,
    token_count,
)

logger = logging.get_logger(__name__)


class WanIclV2VTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------ helpers
    @property
    def inner_dim(self) -> int:
        return self.config.num_attention_heads * self.config.attention_head_dim

    def _patchify(self, latents: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        grid = patch_grid_from_latents(latents, self.config.patch_size)
        tokens = self.patch_embedding(latents).flatten(2).transpose(1, 2).contiguous()
        if tokens.shape[1] != token_count(grid):
            raise RuntimeError(f"patch token count mismatch: {tokens.shape[1]} vs {token_count(grid)}")
        return tokens, grid

    def _unpatchify(self, tokens: torch.Tensor, batch_size: int, grid: Tuple[int, int, int]) -> torch.Tensor:
        p_t, p_h, p_w = self.config.patch_size
        f, h, w = grid
        tokens = tokens.reshape(batch_size, f, h, w, p_t, p_h, p_w, -1)
        tokens = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return tokens.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _rotary(self, target_grid, cond_grid, device, shared_positions: bool):
        target_ids = make_grid_position_ids(target_grid, device=device, dtype=torch.float32)
        parts = []
        if cond_grid is not None:
            if shared_positions:
                # identical ids when grids match; OminiControl-scaled ids for a low-res cond
                cond_ids = make_corrected_condition_position_ids(
                    condition_grid=cond_grid, target_grid=target_grid, device=device, dtype=torch.float32
                )
            else:
                # ablation: cond placed *before* the target in time (frame-concat style offset)
                cond_ids = make_grid_position_ids(cond_grid, device=device, dtype=torch.float32)
                target_ids = target_ids.clone()
                target_ids[..., 0] += float(cond_grid[0])
            parts.append(self.rope.forward_from_position_ids(cond_ids))
        parts.append(self.rope.forward_from_position_ids(target_ids))
        return concat_rotary_emb(parts)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cond_latents: Optional[torch.Tensor] = None,
        cond_token_bias: Optional[torch.Tensor] = None,
        target_token_bias: Optional[torch.Tensor] = None,
        attn_mode: str = "full",
        shared_positions: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        hidden_states:   noisy target latents [B, C, F, H, W]
        timestep:        [B] (target timestep; cond gets 0) or explicit per-token [B, S_total]
        cond_latents:    clean condition latents [B, C, F, Hc, Wc] (F must equal target F)
        *_token_bias:    [B, D] or [B, S_stream, D] additive embeddings for each stream
        """
        lora_scale = 1.0
        if attention_kwargs is not None:
            lora_scale = attention_kwargs.get("scale", 1.0)
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        bsz = hidden_states.shape[0]
        target_tokens, target_grid = self._patchify(hidden_states)
        if target_token_bias is not None:
            target_tokens = target_tokens + _expand_bias(target_token_bias, target_tokens)
        n_target = target_tokens.shape[1]

        cond_grid = None
        n_cond = 0
        parts = []
        if cond_latents is not None:
            cond_tokens, cond_grid = self._patchify(cond_latents)
            if cond_grid[0] != target_grid[0]:
                raise ValueError(f"cond/target latent frames differ: cond={cond_grid} target={target_grid}")
            if cond_token_bias is not None:
                cond_tokens = cond_tokens + _expand_bias(cond_token_bias, cond_tokens)
            parts.append(cond_tokens)
            n_cond = cond_tokens.shape[1]
        parts.append(target_tokens)
        tokens = torch.cat(parts, dim=1)
        seq = tokens.shape[1]

        rotary_emb = self._rotary(target_grid, cond_grid, tokens.device, shared_positions)

        if timestep.ndim == 1:
            t_all = torch.cat(
                [
                    torch.zeros(bsz, n_cond, device=tokens.device, dtype=tokens.dtype),
                    timestep.to(tokens.device, tokens.dtype).view(bsz, 1).expand(bsz, n_target),
                ],
                dim=1,
            )
        elif timestep.shape == (bsz, seq):
            t_all = timestep.to(tokens.device, tokens.dtype)
        else:
            raise ValueError(f"timestep must be [B] or [B,{seq}], got {tuple(timestep.shape)}")

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            t_all.flatten(), encoder_hidden_states, None, timestep_seq_len=seq
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        spec = IclAttnSpec(num_prefix_tokens=n_cond, mode=attn_mode)
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                tokens = self._gradient_checkpointing_func(block, tokens, encoder_hidden_states, timestep_proj, rotary_emb, spec)
        else:
            for block in self.blocks:
                tokens = block(tokens, encoder_hidden_states, timestep_proj, rotary_emb, spec)

        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2).to(tokens.device)
        scale = scale.squeeze(2).to(tokens.device)
        tokens = (self.norm_out(tokens.float()) * (1 + scale) + shift).type_as(tokens)
        tokens = self.proj_out(tokens)

        out = self._unpatchify(tokens[:, -n_target:], bsz, target_grid)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (out,)
        return Transformer2DModelOutput(sample=out)


def _expand_bias(bias: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    if bias.ndim == 2:
        bias = bias[:, None, :]
    return bias.to(dtype=tokens.dtype, device=tokens.device)
