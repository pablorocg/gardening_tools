import numpy as np
import torch
from gardening_tools.modules.networks.components.layers import LayerNormNd
from gardening_tools.modules.networks.components.utils import convert_dim_to_conv_op
from torch import nn
from typing import Tuple
from timm.models.eva import EvaBlock
from timm.layers import trunc_normal_

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        self.num_classes = out_channels

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [
            [2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size]
            for n in range(num_stages)
        ][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [
            _round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)
        ]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        channels[s],
                        channels[s + 1],
                        kernel_size=strides[s],
                        stride=strides[s],
                    ),
                    norm(channels[s + 1]),
                    activation(),
                )
            )
        stages.append(
            nn.ConvTranspose3d(
                channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]
            )
        )
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)
    


# =============================================================================
# ADD THIS CLASS to: gardening_tools/modules/networks/components/transformer.py
# (append it; do not overwrite the file. Keep the existing PatchEmbed etc.)
# =============================================================================
import torch
import torch.nn as nn


class MAEDecoder(nn.Module):
    """Lightweight transformer MAE decoder.

    This is the piece that is missing from the Primus PatchDecode (which is just
    transposed convolutions and does NOT mix information across tokens). It:

      1. projects encoder tokens to a (smaller) decoder dim,
      2. reinserts learnable mask tokens at the positions Eva dropped,
      3. adds a learnable decoder positional embedding,
      4. runs a small transformer (token mixing),
      5. projects each patch token to patch space via a single linear head
         (default 8**3 = 512 for single-channel 3D patches).

    Use the linear head for the original-MAE patch-level objective. Transposed
    convolution is intentionally not used here.
    """

    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        patch_dim: int,
        decoder_embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, patch_dim)

        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def _restore(self, visible, keep_indices):
        """Restore the full sequence by filling blanks with mask tokens."""
        if keep_indices is None:
            # No tokens were dropped, return full sequence with empty mask
            return visible, None  # ← CAMBIO: devolver tupla, no solo tensor
        
        B, num_kept, C = visible.shape
        device = visible.device
        dtype = visible.dtype
        
        num_masked = self.num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1).to(dtype)
        
        # Prepare tensor for restored sequence - MATCH DTYPE
        full = torch.zeros(B, self.num_patches, C, device=device, dtype=dtype)
        restored_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        
        # Assign tokens in correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            all_indices = torch.arange(self.num_patches, device=device)
            mask = torch.ones(self.num_patches, device=device, dtype=torch.bool)
            mask[kept_pos] = False
            masked_pos = all_indices[mask]
            
            full[i, kept_pos] = visible[i]
            full[i, masked_pos] = mask_tokens[i, : len(masked_pos)]
            restored_mask[i, kept_pos] = True
        
        return full, restored_mask

    def forward(self, visible_tokens, keep_indices, num_patches):
        """
        Args:
            visible_tokens: [B, num_kept, embed_dim] - visible patch tokens from encoder
            keep_indices: [B, num_kept] or None - indices of visible patches
            num_patches: int - total number of patches
        Returns:
            pred_patches: [B, num_patches, patch_dim] - predictions for all patches
        """
        # 1. Project visible tokens to decoder dim FIRST (embed_dim -> decoder_embed_dim)
        visible = self.decoder_embed(visible_tokens)  # [B, num_kept, decoder_embed_dim]
        
        # 2. Restore full sequence with mask tokens (now all in decoder_embed_dim)
        full, restored_mask = self._restore(visible, keep_indices)  # [B, num_patches, decoder_embed_dim]
        
        # 3. Add positional embeddings
        x = full + self.decoder_pos_embed
        
        # 4. Apply transformer decoder blocks (token mixing)
        x = self.blocks(x)
        x = self.norm(x)
        
        # 5. Project each token to patch space
        x = self.head(x)  # [B, num_patches, patch_dim]
        
        return x