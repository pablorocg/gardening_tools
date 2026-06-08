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
    




class MAEDecoder(nn.Module):
    """
    Lightweight transformer decoder for MAE pretraining.

    Restores the full vision sequence (encoded visible tokens + learned mask
    tokens), processes it with a few transformer blocks (so token information is
    actually mixed, unlike PatchDecode), and projects each token to patch_dim.

    Metadata tokens are prepended and carry no positional embedding.
    """

    def __init__(
        self,
        encoder_dim: int,
        patch_dim: int,
        num_patches: int,
        decoder_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        # positional embedding for vision tokens only; metadata gets none
        self.vis_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        self.blocks = nn.ModuleList(
            [
                EvaBlock(
                    dim=decoder_dim,
                    num_heads=num_heads,
                    qkv_bias=True,
                    qkv_fused=False,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=False,
                    scale_mlp=False,
                    num_prefix_tokens=0,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_dim, bias=True)

        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.vis_pos_embed, std=0.02)

    def _restore(self, vis, keep_indices):
        """Place projected visible tokens at kept positions, mask token elsewhere."""
        B = vis.shape[0]
        if keep_indices is None:
            return vis  # no masking, sequence is already full
        restored = self.mask_token.repeat(B, self.num_patches, 1).clone()
        for i in range(B):
            restored[i, keep_indices[i]] = vis[i]
        return restored

    def forward(self, vis, meta, keep_indices):
        """
        vis:  [B, num_kept, encoder_dim]  encoded visible vision tokens
        meta: [B, M, encoder_dim]         encoded metadata tokens
        returns: [B, num_patches, patch_dim]
        """
        vis = self.decoder_embed(vis)
        meta = self.decoder_embed(meta)

        vis = self._restore(vis, keep_indices)
        vis = vis + self.vis_pos_embed

        M = meta.shape[1]
        x = torch.cat([meta, vis], dim=1)
        for blk in self.blocks:
            x = blk(x, rope=None)
        x = self.norm(x)

        x = x[:, M:]  # drop metadata, keep vision tokens
        return self.head(x)
