import math

import torch
from torch import nn


class Transpose(nn.Module):
    """Module wrapper for :meth:`torch.Tensor.transpose` supporting ``nn.Sequential`` pipelines."""

    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(*self.dims)
        if self.contiguous:
            x = x.contiguous()
        return x


def get_activation_fn(name: str) -> nn.Module:
    """Return an activation layer given its ``name``.

    Supported names are ``relu``, ``gelu``, ``silu``/``swish`` and ``tanh``. The function defaults to
    the GELU activation when an unknown name is provided to maintain backward compatibility with the
    original PatchTST implementation."""

    name = (name or "gelu").lower()
    if name in {"relu"}:
        return nn.ReLU()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"tanh"}:
        return nn.Tanh()
    return nn.GELU()


def _sinusoidal_encoding(q_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(0, q_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(q_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def positional_encoding(pe: str, learn_pe: bool, q_len: int, d_model: int) -> nn.Parameter:
    """Create positional encodings used by :class:`PatchTST_backbone`.

    Parameters
    ----------
    pe: str
        Type of positional encoding. ``"zeros"`` yields a learnable (or fixed) zero initialised encoding,
        while ``"sinusoidal"`` provides the classic sine/cosine positional embedding. ``"learnable"``
        behaves the same as ``"zeros"`` but always enables gradients.
    learn_pe: bool
        Whether the positional encoding should be optimised during training when ``pe`` is ``"zeros"``
        or ``"sinusoidal"``.
    q_len: int
        Sequence length of the encoder queries.
    d_model: int
        Hidden dimension of the model.
    """

    pe = (pe or "zeros").lower()
    if pe in {"learnable", "zeros"}:
        param = torch.zeros(1, q_len, d_model)
        requires_grad = True if pe == "learnable" else learn_pe
        return nn.Parameter(param, requires_grad=requires_grad)
    if pe == "sinusoidal":
        param = _sinusoidal_encoding(q_len, d_model)
        return nn.Parameter(param, requires_grad=learn_pe)
    raise ValueError(f"Unsupported positional encoding type: {pe}")


__all__ = ["Transpose", "get_activation_fn", "positional_encoding"]
