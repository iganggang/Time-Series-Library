"""Task-specific heads for the PatchTST SSL backbone."""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


def _flatten_latent(latent: Tensor) -> Tensor:
    if latent.dim() == 4:
        b, n_vars, d_model, n_patches = latent.shape
        return latent.permute(0, 1, 3, 2).reshape(b, n_vars, d_model * n_patches)
    raise ValueError("Expected latent tensor with shape [B, n_vars, d_model, n_patches]")


class ForecastHead(nn.Module):
    """Maps PatchTST latent sequence to future values for multivariate forecasting."""

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        pred_len: int,
        n_vars: int,
        head_dropout: float = 0.0,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.dropout = nn.Dropout(head_dropout)
        head_dim = d_model * n_patches
        if individual:
            self.proj = nn.ModuleList([nn.Linear(head_dim, pred_len) for _ in range(n_vars)])
        else:
            self.proj = nn.Linear(head_dim, pred_len)

    def forward(self, latent: Tensor) -> Tensor:
        features = _flatten_latent(latent)
        if self.individual:
            outs = []
            for idx in range(self.n_vars):
                outs.append(self.proj[idx](self.dropout(features[:, idx, :])))
            out = torch.stack(outs, dim=1)
        else:
            out = self.proj(self.dropout(features))
        return out.transpose(1, 2)


class ClassificationHead(nn.Module):
    """Global pooling + linear classifier on top of PatchTST latent sequence."""

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        num_classes: int,
        pooling: Literal["mean", "max", "cls"] = "mean",
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(head_dropout)
        self.projection = nn.Linear(d_model, num_classes)
        self.n_patches = n_patches

    def forward(self, latent: Tensor) -> Tensor:
        tokens = self._flatten_tokens(latent)
        if self.pooling == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pooling == "max":
            pooled, _ = tokens.max(dim=1)
        elif self.pooling == "cls":
            pooled = tokens[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
        return self.projection(self.dropout(pooled))

    def _flatten_tokens(self, latent: Tensor) -> Tensor:
        if latent.dim() == 4:
            b, n_vars, d_model, n_patches = latent.shape
            return latent.permute(0, 1, 3, 2).reshape(b, n_vars * n_patches, d_model)
        if latent.dim() == 3:
            return latent
        raise ValueError("Unsupported latent shape for classification head")


class ImputationHead(nn.Module):
    """Reconstructs the full sequence from PatchTST latent features."""

    def __init__(
        self,
        d_model: int,
        n_patches: int,
        seq_len: int,
        n_vars: int,
        head_dropout: float = 0.0,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.dropout = nn.Dropout(head_dropout)
        head_dim = d_model * n_patches
        if individual:
            self.proj = nn.ModuleList([nn.Linear(head_dim, seq_len) for _ in range(n_vars)])
        else:
            self.proj = nn.Linear(head_dim, seq_len)

    def forward(self, latent: Tensor) -> Tensor:
        features = _flatten_latent(latent)
        if self.individual:
            outs = []
            for idx in range(self.n_vars):
                outs.append(self.proj[idx](self.dropout(features[:, idx, :])))
            out = torch.stack(outs, dim=1)
        else:
            out = self.proj(self.dropout(features))
        return out.transpose(1, 2)


__all__ = ["ForecastHead", "ClassificationHead", "ImputationHead"]
