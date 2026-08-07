"""
PyTorch Vision Encoder module for LawSLM Multimodal Vision-Language Architecture.
Converts 3-channel image inputs [B, 3, H, W] into token projection sequence [B, N_patches, d_model].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from slm.utils.logger import get_logger

logger = get_logger("slm.vision.encoder")


class VisionPatchEmbedding(nn.Module):
    """
    Splits image tensor into non-overlapping patches and projects them to target d_model dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 128
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch Convolution Projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor of shape [B, 3, H, W]

        Returns:
            Sequence embedding tensor of shape [B, N_patches + 1, d_model]
        """
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        x = self.proj(x)  # [B, embed_dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N_patches + 1, embed_dim]
        x = x + self.pos_embedding
        return x


class VisionEncoder(nn.Module):
    """
    Lightweight Vision Encoder Transformer Block for image feature extraction.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4
    ) -> None:
        super().__init__()
        self.patch_embed = VisionPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Processes normalized image pixels into sequence embeddings.

        Args:
            pixel_values: Image tensor [B, 3, H, W]

        Returns:
            Image token sequence tensor [B, N_patches + 1, d_model]
        """
        tokens = self.patch_embed(pixel_values)
        out = self.transformer(tokens)
        return self.ln_post(out)
