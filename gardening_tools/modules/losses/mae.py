"""Patch-level MAE utilities for 3D volumes.

`patchify` / `unpatchify` are strict inverses. The grid flatten order mirrors the
rearrange convention used by the encoder in PrimusCLSREG
(`b c w h d -> b (h w d) c`, see asparagus/modules/networks/primus.py), i.e. the
*second* spatial dim is the outer index. This matters ONLY for the patch-level
loss: the order produced here must equal the token order produced by Eva, so the
predicted token for patch k is compared against the spatially-correct target
patch k. VERIFY this against your installed Eva before trusting the patch-level
path. The voxel-reconstruction path (used for the running baseline) is robust to
this because the decoder head learns the mapping and unpatchify is self-inverse.
"""

import torch
from einops import rearrange
from typing import Optional, Sequence


def patchify(imgs: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    """[B, C, S0, S1, S2] -> [B, num_patches, C * prod(patch_size)].

    Token order is (g1 g0 g2) to mirror the encoder's `(h w d)` flatten.
    """
    p0, p1, p2 = patch_size
    return rearrange(
        imgs,
        "b c (g0 p0) (g1 p1) (g2 p2) -> b (g1 g0 g2) (c p0 p1 p2)",
        p0=p0, p1=p1, p2=p2,
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: Sequence[int],
    grid: Sequence[int],
    channels: int = 1,
) -> torch.Tensor:
    """[B, num_patches, C * prod(patch_size)] -> [B, C, S0, S1, S2].

    Strict inverse of `patchify`. `grid` is (g0, g1, g2) = spatial // patch.
    """
    p0, p1, p2 = patch_size
    g0, g1, g2 = grid
    return rearrange(
        patches,
        "b (g1 g0 g2) (c p0 p1 p2) -> b c (g0 p0) (g1 p1) (g2 p2)",
        g0=g0, g1=g1, g2=g2, c=channels, p0=p0, p1=p1, p2=p2,
    )


def masked_patch_loss(
    pred_patches: torch.Tensor,
    imgs: torch.Tensor,
    patch_mask: Optional[torch.Tensor],
    patch_size: Sequence[int],
    norm_pix: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Patch-level MAE reconstruction loss.

    Args:
        pred_patches: [B, num_patches, patch_dim] decoder output.
        imgs:         [B, C, S0, S1, S2] reconstruction target (the input image).
        patch_mask:   [B, num_patches] bool, True for KEPT/visible patches
                      (repo convention). Loss is computed on masked (~mask) patches.
                      If None, loss is averaged over all patches.
        norm_pix:     per-patch normalization of the target (standard MAE).
    """
    target = patchify(imgs, patch_size)  # [B, num_patches, patch_dim]

    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + eps)

    per_patch = ((pred_patches - target) ** 2).mean(dim=-1)  # [B, num_patches]

    if patch_mask is None:
        return per_patch.mean()

    masked = ~patch_mask  # True where the patch was masked / to be reconstructed
    denom = masked.float().sum().clamp(min=1.0)
    return (per_patch * masked.float()).sum() / denom