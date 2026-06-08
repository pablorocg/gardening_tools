import torch
from einops import rearrange
from typing import Tuple


def patchify(imgs: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    [B, C, X, Y, Z] -> [B, N, C*p1*p2*p3]

    Token order MUST match Primus/primeta: (h w d), with X->w, Y->h, Z->d.
    This is the ordering produced by `rearrange("b c w h d -> b (h w d) c")`
    after the patch-embed conv, so prediction token i lines up with target token i.
    """
    p1, p2, p3 = patch_size
    return rearrange(
        imgs,
        "b c (w p1) (h p2) (d p3) -> b (h w d) (c p1 p2 p3)",
        p1=p1, p2=p2, p3=p3,
    )


def unpatchify(
    patches: torch.Tensor,
    patch_size: Tuple[int, int, int],
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    [B, N, C*p1*p2*p3] -> [B, C, X, Y, Z]. grid_shape is (W, H, D) = (X/p, Y/p, Z/p).
    Only needed for visualization, not for the loss.
    """
    p1, p2, p3 = patch_size
    w, h, d = grid_shape
    return rearrange(
        patches,
        "b (h w d) (c p1 p2 p3) -> b c (w p1) (h p2) (d p3)",
        h=h, w=w, d=d, p1=p1, p2=p2, p3=p3,
    )


def masked_patch_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Patch-level MAE loss on masked tokens only.

    pred:   [B, N, patch_dim]   per-token reconstruction from the decoder head
    target: [B, N, patch_dim]   patchify(input image)
    mask:   [B, N] bool         True = masked (loss is computed here)
    """
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + eps)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [B, N]

    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom