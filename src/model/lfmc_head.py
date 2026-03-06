"""
LFMC Regression Head for Galileo

This module adds a regression head on top of Galileo's encoder to predict
LFMC values. The encoder produces learned embeddings for each spatial
location, and the regression head maps those embeddings to a single
LFMC percentage value.

Architecture:
    Galileo Encoder --> [B, num_tokens, embed_dim]
    --> Global Average Pool --> [B, embed_dim]
    --> MLP Head --> [B, 1] (LFMC prediction)
"""

import logging
import time

import torch
import torch.nn as nn

logger = logging.getLogger("lfmc_head")


class LFMCRegressionHead(nn.Module):
    """
    MLP regression head that maps Galileo embeddings to LFMC values.

    Why MLP and not just linear?
    - LFMC has complex nonlinear relationships with spectral indices
    - A small MLP (2-3 layers) captures these without overfitting
    - Dropout prevents reliance on any single spectral feature
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        """
        Args:
            embed_dim: Dimension of Galileo encoder output (256 for Tiny)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate for regularization
            num_layers: Number of hidden layers (2 works well)
        """
        super().__init__()

        layers = []
        in_dim = embed_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),  # Smooth activation, works well with transformers
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection to scalar LFMC value
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"STATUS | LFMCRegressionHead: {param_count:,} parameters")

    def _init_weights(self):
        """Initialize with small weights to start predictions near zero."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Galileo encoder output [B, num_tokens, embed_dim]
                       or already pooled [B, embed_dim]

        Returns:
            LFMC predictions [B, 1]
        """
        # If we get token-level embeddings, pool them first
        if embeddings.dim() == 3:
            # Global average pooling across spatial/temporal tokens
            embeddings = embeddings.mean(dim=1)  # [B, embed_dim]

        return self.mlp(embeddings)


class GalileoLFMC(nn.Module):
    """
    Complete model: Galileo encoder + LFMC regression head.

    This wraps the pretrained Galileo encoder with a task-specific
    regression head for LFMC prediction.
    """

    def __init__(
        self,
        encoder,  # Galileo Encoder instance
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        """
        Args:
            encoder: Pretrained Galileo Encoder
            embed_dim: Encoder output dimension (256 for Galileo-Tiny)
            hidden_dim: Regression head hidden dimension
            dropout: Dropout rate
            freeze_encoder: If True, only train the regression head
                          (useful for limited data like Hawaii)
        """
        super().__init__()
        self.encoder = encoder
        self.head = LFMCRegressionHead(embed_dim, hidden_dim, dropout)
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("STATUS | Encoder frozen, only training regression head")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"STATUS | Total params: {total:,}, Trainable: {trainable:,}")

    def forward(self, masked_output) -> torch.Tensor:
        """
        Forward pass through encoder + regression head.

        Args:
            masked_output: Galileo MaskedOutput namedtuple

        Returns:
            LFMC predictions [B, 1]
        """
        start = time.time()

        # Get encoder embeddings
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.encoder(masked_output)
        else:
            embeddings = self.encoder(masked_output)

        # Predict LFMC
        predictions = self.head(embeddings)

        elapsed = time.time() - start
        if elapsed > 0.5:
            logger.debug(f"TIMING | Forward pass: {elapsed:.3f}s")

        return predictions

    def unfreeze_encoder(self, unfreeze_last_n: int = 2):
        """
        Gradually unfreeze encoder layers for fine-tuning.

        Strategy: Start with frozen encoder (train head only),
        then progressively unfreeze deeper layers.
        This prevents catastrophic forgetting of pretrained features.

        Args:
            unfreeze_last_n: Number of transformer blocks to unfreeze
        """
        # Get all transformer blocks
        blocks = list(self.encoder.children())
        total_blocks = len(blocks)

        # Unfreeze the last N blocks
        for i, block in enumerate(blocks):
            if i >= total_blocks - unfreeze_last_n:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"STATUS | Unfroze last {unfreeze_last_n} blocks. "
                    f"Trainable params: {trainable:,}")
