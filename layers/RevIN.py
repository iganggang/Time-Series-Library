from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalisation for time-series data.

    This implementation follows the formulation proposed in the PatchTST paper. The module keeps the
    per-sample statistics observed during the forward ``'norm'`` pass so that a subsequent ``'denorm'``
    call can recover the original scale of the data.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._last: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._compute_stats(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown RevIN mode: {mode}")
        return x

    def _compute_stats(self, x: torch.Tensor) -> None:
        dims = tuple(range(1, x.ndim - 1)) if x.ndim > 2 else (0,)
        if self.subtract_last:
            self._last = x[:, -1:, :]
        else:
            self._mean = x.mean(dim=dims, keepdim=True).detach()
        self._std = torch.sqrt(torch.var(x, dim=dims, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.subtract_last and self._last is not None:
            x = x - self._last
        elif self._mean is not None:
            x = x - self._mean

        if self._std is not None:
            x = x / self._std

        if self.affine:
            x = x * self.scale + self.bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.bias) / (self.scale + self.eps * self.eps)

        if self._std is not None:
            x = x * self._std

        if self.subtract_last and self._last is not None:
            x = x + self._last
        elif self._mean is not None:
            x = x + self._mean
        return x


__all__ = ["RevIN"]
