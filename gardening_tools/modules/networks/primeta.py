import torch
from einops import rearrange
from timm.layers import RotaryEmbeddingCat
from torch import nn
from typing import Tuple

from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.eva import Eva
from gardening_tools.modules.networks.components.transformer import (
    PatchEmbed,
    MAEDecoder,
)
from gardening_tools.modules.networks.components.weight_init import InitWeights_He


class Primeta(BaseNet):
    """
    Primus backbone adapted for metadata-conditioned MAE pretraining.

    - PatchEmbed -> tokens
    - metadata tokens (encoded externally, dim == embed_dim) prepended as prefix
    - Eva encoder masks vision tokens only (patch_drop_rate), metadata never dropped
    - MAEDecoder restores full vision sequence and predicts patch_dim per token
    - patch-level loss computed in the trainer via gardening_tools.modules.losses.mae
    """

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...],
        num_metadata_tokens: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_numheads: int = 8,
        patch_drop_rate: float = 0.6,
        mlp_ratio: float = 4 * 2 / 3,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner: bool = False,
    ):
        assert input_shape is not None and len(input_shape) == 3
        assert all(j % i == 0 for i, j in zip(patch_embed_size, input_shape))
        super().__init__()

        self.num_metadata_tokens = num_metadata_tokens
        self.patch_embed_size = patch_embed_size
        self.embed_dim = embed_dim

        grid = tuple(i // ds for i, ds in zip(input_shape, patch_embed_size))
        self.num_patches = int(grid[0] * grid[1] * grid[2])
        patch_dim = input_channels * int(
            patch_embed_size[0] * patch_embed_size[1] * patch_embed_size[2]
        )

        self.encoder = PatchEmbed(patch_embed_size, input_channels, embed_dim)

        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=grid,
            num_reg_tokens=num_metadata_tokens,  # protects metadata from drop + rope
            use_rot_pos_emb=True,
            use_abs_pos_emb=False,               # no abs pos embed -> none on metadata
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        self.decoder = MAEDecoder(
            encoder_dim=embed_dim,
            patch_dim=patch_dim,
            num_patches=self.num_patches,
            decoder_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_numheads,
        )

        self.encoder.apply(InitWeights_He(1e-2))

    def _build_mask(self, keep_indices, B, device):
        """[B, N] bool, True = masked (loss positions)."""
        if keep_indices is None:
            return torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        mask = torch.ones(B, self.num_patches, dtype=torch.bool, device=device)
        for i in range(B):
            mask[i, keep_indices[i]] = False
        return mask

    def forward(self, x, metadata_tokens):
        """
        x:               [B, C, X, Y, Z]
        metadata_tokens: [B, M, embed_dim]   one token per column, M == num_metadata_tokens

        returns:
            pred: [B, num_patches, patch_dim]
            mask: [B, num_patches] bool, True = masked
        """
        assert metadata_tokens.shape[1] == self.num_metadata_tokens
        assert metadata_tokens.shape[-1] == self.embed_dim

        B = x.shape[0]
        x = self.encoder(x)                      # [B, embed, W, H, D]
        x = rearrange(x, "b c w h d -> b (h w d) c")

        M = metadata_tokens.shape[1]
        x = torch.cat([metadata_tokens, x], dim=1)  # prepend metadata
        x, keep_indices = self.eva(x)

        meta_enc = x[:, :M]
        vis = x[:, M:]

        pred = self.decoder(vis, meta_enc, keep_indices)
        mask = self._build_mask(keep_indices, B, x.device)
        return pred, mask

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")