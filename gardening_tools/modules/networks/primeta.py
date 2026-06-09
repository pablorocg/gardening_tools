"""Primeta: an MAE pretraining backbone separated out from Primus.

Mirrors the Eva / PatchEmbed / mask-token mechanics of PrimusCLSREG
(asparagus/modules/networks/primus.py), but instead of a classification head it
attaches an MAEDecoder (transformer + linear-to-patch). It does NOT use the
transposed-convolution PatchDecode of Primus.

Metadata is optional. When `n_metadata_tokens > 0`, a [B, n_metadata_tokens,
embed_dim] tensor (produced by the Asparagus-side MetadataEncoder) is prepended
as PREFIX tokens, exactly like register tokens: they survive Eva's patch drop and
are stripped before decoding. Prefix count is folded into Eva's `num_reg_tokens`
so the absolute positions of patch tokens stay constant regardless of mask ratio.

NOTE: `Eva`, `PatchEmbed`, `InitWeights_He` and the existing `transformer.py`
live in gardening_tools and are not visible here; the contract used below is the
one observable in PrimusCLSREG. Verify with a smoke test (see notes in chat).
"""

import torch
from einops import rearrange
from math import prod
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.eva import Eva
from gardening_tools.modules.networks.components.transformer import PatchEmbed, MAEDecoder
from gardening_tools.modules.networks.components.weight_init import InitWeights_He
from gardening_tools.modules.losses.mae import unpatchify
from typing import Optional, Tuple

try:
    from timm.layers import RotaryEmbeddingCat
except ImportError:
    RotaryEmbeddingCat = None


class Primeta(BaseNet):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        eva_depth: int,
        eva_numheads: int,
        input_shape: Tuple[int, ...],
        num_register_tokens: int = 0,
        n_metadata_tokens: int = 0,
        patch_drop_rate: float = 0.0,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_numheads: int = 8,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio: float = 4 * 2 / 3,
        drop_path_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner: bool = False,
        norm_pix_loss: bool = True,
    ):
        assert input_shape is not None and len(input_shape) == 3, "Only 3D is supported"
        assert all(j % i == 0 for i, j in zip(patch_embed_size, input_shape))
        super().__init__()

        self.input_channels = input_channels
        self.patch_embed_size = tuple(patch_embed_size)
        self.n_metadata_tokens = n_metadata_tokens
        self.norm_pix_loss = norm_pix_loss
        # Used by BaseModule.load_state_dict for stem-weight repetition.
        self.stem_weight_name = "encoder.proj.weight"

        grid = tuple(i // p for i, p in zip(input_shape, patch_embed_size))
        self.grid_size = grid
        self.num_patches = prod(grid)
        patch_dim = input_channels * prod(self.patch_embed_size)  # 1 * 8^3 = 512

        self.encoder = PatchEmbed(self.patch_embed_size, input_channels, embed_dim)

        total_prefix = num_register_tokens + n_metadata_tokens
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=grid,
            num_reg_tokens=total_prefix,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
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

        if num_register_tokens > 0:
            self.register_tokens = torch.nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            torch.nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.decoder = MAEDecoder(
            embed_dim=embed_dim,
            num_patches=self.num_patches,
            patch_dim=patch_dim,
            decoder_embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_numheads,
        )

        self.encoder.apply(InitWeights_He(1e-2))

    # ------------------------------------------------------------------ encode
    def _encode(self, x, metadata_tokens: Optional[torch.Tensor] = None):
        x = self.encoder(x)                     # [B, C, w, h, d]
        B = x.shape[0]
        x = rearrange(x, "b c w h d -> b (h w d) c")  # same order as PrimusCLSREG

        prefixes = []
        if self.n_metadata_tokens > 0:
            assert metadata_tokens is not None, "n_metadata_tokens>0 but no metadata passed"
            assert metadata_tokens.shape[1] == self.n_metadata_tokens
            prefixes.append(metadata_tokens)
        if self.register_tokens is not None:
            prefixes.append(self.register_tokens.expand(B, -1, -1))

        n_prefix = sum(p.shape[1] for p in prefixes)
        if prefixes:
            x = torch.cat(prefixes + [x], dim=1)

        x, keep_indices = self.eva(x)           # patch drop happens inside Eva
        if n_prefix:
            x = x[:, n_prefix:]                  # strip metadata + register tokens
        return x, keep_indices

    def _keep_mask(self, keep_indices, B, device):
        if keep_indices is None:
            return None
        m = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        for i in range(B):
            m[i, keep_indices[i]] = True
        return m

    # --------------------------------------------------------------- forwards
    def forward_with_features(self, x, metadata_tokens: Optional[torch.Tensor] = None):
        """Drop-in for SelfSupervisedModule: returns (voxel_pred, encoder_features).

        Used by the *pixel-level* baseline (MSELoss in the existing lightning
        module). encoder_features are the visible patch tokens [B, n_kept, embed].
        """
        visible, keep_indices = self._encode(x, metadata_tokens)
        pred_patches = self.decoder(visible, keep_indices, self.num_patches)
        pred = unpatchify(pred_patches, self.patch_embed_size, self.grid_size, channels=self.input_channels)
        return pred, visible

    def forward_mae(self, x, metadata_tokens: Optional[torch.Tensor] = None):
        """Patch-level path: returns (pred_patches, patch_mask) to feed
        gardening_tools.modules.losses.mae.masked_patch_loss(...). Requires a
        small change in the lightning module to use that loss instead of MSE.
        """
        visible, keep_indices = self._encode(x, metadata_tokens)
        pred_patches = self.decoder(visible, keep_indices, self.num_patches)
        patch_mask = self._keep_mask(keep_indices, x.shape[0], pred_patches.device)
        return pred_patches, patch_mask

    def forward(self, x, metadata_tokens: Optional[torch.Tensor] = None):
        pred, _ = self.forward_with_features(x, metadata_tokens)
        return pred