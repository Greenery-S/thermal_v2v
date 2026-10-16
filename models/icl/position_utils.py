"""Position/shape utilities for UAV Wan in-context latent tokens.

This version keeps the original integer-grid helpers, but adds an
OminiControl-style corrected position-id path for lower-resolution conditions.

Key idea:
  condition grid tokens keep their own token count, but their spatial RoPE
  positions are placed in the target grid coordinate system:

    y' = y * (target_h / condition_h) + (target_h / condition_h - 1) / 2
    x' = x * (target_w / condition_w) + (target_w / condition_w - 1) / 2

This matches the OminiControl compact-position correction style
``ids[:, 1:] = ids[:, 1:] * position_scale + (position_scale - 1) / 2``.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch


def patch_grid_from_latents(latents: torch.Tensor, patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if latents.ndim != 5:
        raise ValueError(f"latents must be [B,C,F,H,W], got {tuple(latents.shape)}")
    _, _, frames, height, width = latents.shape
    p_t, p_h, p_w = patch_size
    if frames % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"latent shape {(frames, height, width)} is not divisible by patch_size={patch_size}"
        )
    return frames // p_t, height // p_h, width // p_w


def token_count(grid: Tuple[int, int, int]) -> int:
    f, h, w = [int(v) for v in grid]
    return f * h * w


def make_grid_position_ids(
    grid: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return regular flattened ``[t, y, x]`` position ids for a 3D token grid.

    Args:
        grid: ``(F, H, W)`` after latent patchification.

    Returns:
        ``[F*H*W, 3]`` float tensor in standard flatten order: t outer, y, x.
    """

    f, h, w = [int(v) for v in grid]
    if f <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"Invalid grid: {grid}")

    t = torch.arange(f, device=device, dtype=dtype)
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack([tt, yy, xx], dim=-1).reshape(-1, 3)


def omini_spatial_scale_and_bias(
    condition_grid: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
) -> Tuple[float, float, float, float]:
    """Return OminiControl-style spatial scales and center biases.

    OminiControl compact position correction uses:
        pos' = pos * scale + (scale - 1) / 2

    We use separate y/x scales because the condition and target grids may not
    have the same aspect-ratio after VAE + patchification.
    """

    _, ch, cw = [int(v) for v in condition_grid]
    _, th, tw = [int(v) for v in target_grid]
    if ch <= 0 or cw <= 0 or th <= 0 or tw <= 0:
        raise ValueError(f"Invalid grids: condition={condition_grid}, target={target_grid}")

    scale_y = float(th) / float(ch)
    scale_x = float(tw) / float(cw)
    bias_y = (scale_y - 1.0) / 2.0
    bias_x = (scale_x - 1.0) / 2.0
    return scale_y, scale_x, bias_y, bias_x


def make_corrected_condition_position_ids(
    condition_grid: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return OminiControl-style corrected condition position ids.

    The condition token count remains ``condition_grid``. Only its spatial
    coordinates are moved into the target coordinate system. Temporal positions
    are kept aligned frame-by-frame.

    Returns:
        ``[condition_F*condition_H*condition_W, 3]`` float tensor ``[t, y', x']``.
    """

    cf, ch, cw = [int(v) for v in condition_grid]
    tf, _, _ = [int(v) for v in target_grid]
    if cf != tf:
        raise ValueError(
            f"condition and target latent frames must match for depth control: {cf} vs {tf}"
        )

    pos = make_grid_position_ids(condition_grid, device=device, dtype=dtype)
    scale_y, scale_x, bias_y, bias_x = omini_spatial_scale_and_bias(condition_grid, target_grid)
    pos[:, 1] = pos[:, 1] * scale_y + bias_y
    pos[:, 2] = pos[:, 2] * scale_x + bias_x
    return pos


# Kept for backwards-compatible debugging and tests. The main transformer path
# now uses make_corrected_condition_position_ids(), not nearest integer indices.
def make_nearest_target_indices(
    condition_grid: Tuple[int, int, int],
    target_grid: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Map condition-grid tokens to nearest target-grid token indices.

    This is the earlier integer-compatible approximation. It is still useful for
    quick sanity checks, but the OminiControl-style path uses float position ids.
    """

    cf, ch, cw = [int(v) for v in condition_grid]
    tf, th, tw = [int(v) for v in target_grid]
    if cf != tf:
        raise ValueError(f"condition and target latent frames must match for depth control: {cf} vs {tf}")
    if ch <= 0 or cw <= 0 or th <= 0 or tw <= 0:
        raise ValueError(f"Invalid grids: condition={condition_grid}, target={target_grid}")

    t = torch.arange(cf, device=device)
    y = torch.arange(ch, device=device)
    x = torch.arange(cw, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")

    if ch == 1:
        yy_t = torch.zeros_like(yy)
    else:
        yy_t = torch.round(yy.float() * (th - 1) / (ch - 1)).long()
    if cw == 1:
        xx_t = torch.zeros_like(xx)
    else:
        xx_t = torch.round(xx.float() * (tw - 1) / (cw - 1)).long()

    indices = tt * (th * tw) + yy_t.clamp_(0, th - 1) * tw + xx_t.clamp_(0, tw - 1)
    return indices.reshape(-1)


def index_rotary_emb(rotary_emb: Tuple[torch.Tensor, ...], indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Index a Wan rotary embedding tuple along its sequence dimension (-2).

    Kept for compatibility with the earlier nearest-index implementation.
    """

    return tuple(emb.index_select(dim=-2, index=indices.to(device=emb.device)) for emb in rotary_emb)


def concat_rotary_emb(parts: Iterable[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    parts = list(parts)
    if len(parts) == 0:
        raise ValueError("No rotary parts to concatenate")
    n = len(parts[0])
    if any(len(p) != n for p in parts):
        raise ValueError("All rotary tuples must have the same length")
    return tuple(torch.cat([p[i] for p in parts], dim=-2) for i in range(n))
