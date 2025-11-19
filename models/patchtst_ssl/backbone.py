"""Reusable PatchTST encoder backbone shared across SSL and downstream tasks."""
from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from layers.RevIN import RevIN


class PatchTSTBackbone(nn.Module):
    """PatchTST encoder reused for SSL pretraining and downstream tasks.

    Responsibilities
    ----------------
    * patch the raw input sequence (``[B, seq_len, n_vars]``)
    * embed patches with shared or per-channel projections
    * add positional encoding
    * run stacked transformer encoder blocks (with optional Switch-MoE MLPs)
    * (optional) apply RevIN normalization
    * keep track of the mixture-of-experts auxiliary loss
    * expose hooks for SSL/imputation mask support

    The encoder follows the architecture used in the original self-supervised
    PatchTST implementation but provides a clean PyTorch module that can be
    reused by any downstream head.
    """

    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        norm: str = "BatchNorm",
        shared_embedding: bool = True,
        res_attention: bool = False,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        n_experts: int = 4,
        learnable_mask_token: bool = False,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        base = max(seq_len, patch_len)
        self.num_patches = (base - patch_len) // stride + 1
        self.target_length = patch_len + stride * (self.num_patches - 1)
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.revin = revin
        self._aux_loss: Tensor | float | None = None

        if self.revin:
            self.revin_layer = RevIN(n_vars, affine=affine, subtract_last=subtract_last)
        else:
            self.revin_layer = None

        if not shared_embedding:
            self.value_embedding = nn.ModuleList(
                [SwitchLinear(patch_len, d_model, n_experts=n_experts) for _ in range(n_vars)]
            )
        else:
            self.value_embedding = SwitchLinear(patch_len, d_model, n_experts=n_experts)

        self.positional_encoding = positional_encoding(pe, learn_pe, self.num_patches, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            pre_norm=pre_norm,
            store_attn=store_attn,
            n_experts=n_experts,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, n_vars, patch_len)) if learnable_mask_token else None

    @property
    def aux_loss(self) -> Tensor | float:
        """Return the most recent auxiliary MoE loss."""

        if self._aux_loss is None:
            return torch.tensor(0.0, device=self.positional_encoding.device)
        return self._aux_loss

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Encode a time-series batch into latent patch tokens.

        Parameters
        ----------
        x:
            Either ``[B, seq_len, n_vars]`` raw sequences or precomputed patches
            ``[B, num_patches, n_vars, patch_len]``. The backbone performs
            patching automatically when a sequence is provided.
        mask:
            Optional mask for SSL or imputation. When the mask has the same
            shape as ``x`` (sequence mode), values set to ``0`` are ignored.
            When a patch-level mask ``[B, num_patches, n_vars]`` or
            ``[B, num_patches, n_vars, patch_len]`` is provided, masked patches
            are replaced with a learnable token (if enabled) or zeros.

        Returns
        -------
        Tensor
            Latent representation with shape ``[B, n_vars, d_model, num_patches]``.
        """

        seq_mask: Optional[Tensor] = None
        patch_mask: Optional[Tensor] = None
        if mask is not None:
            if mask.dim() == 3 and x.dim() == 3 and mask.shape == x.shape:
                seq_mask = mask
            elif mask.dim() in (3, 4):
                patch_mask = mask
            elif mask.shape == x.shape:
                seq_mask = mask

        if x.dim() == 3:
            x = self._maybe_apply_revin(x)
            if seq_mask is not None:
                x = x.masked_fill(seq_mask == 0, 0)
            patches = self._patchify(x)
        elif x.dim() == 4:
            patches = x
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        if patch_mask is not None:
            patches = self._apply_patch_mask(patches, patch_mask)

        latent = self._encode_patches(patches)
        return latent

    def _maybe_apply_revin(self, x: Tensor) -> Tensor:
        if not self.revin:
            return x
        return self.revin_layer(x, 'norm')

    def _patchify(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        num_patch = (max(seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        tgt_len = self.patch_len + self.stride * (num_patch - 1)
        s_begin = seq_len - tgt_len
        if s_begin < 0:
            pad = x.new_zeros(x.shape[0], -s_begin, x.shape[2])
            x = torch.cat([pad, x], dim=1)
            s_begin = 0
        x = x[:, s_begin:, :]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        return patches

    def _apply_patch_mask(self, patches: Tensor, mask: Tensor) -> Tensor:
        if mask.dim() == 3:
            mask = mask.unsqueeze(-1)
        mask = mask.to(device=patches.device, dtype=patches.dtype)
        if mask.shape[-1] == 1 and patches.shape[-1] > 1:
            mask = mask.expand(-1, -1, -1, patches.shape[-1])
        if mask.shape != patches.shape:
            raise ValueError(f"Mask shape {mask.shape} is incompatible with patches {patches.shape}")
        if self.mask_token is not None:
            patches = patches * (1 - mask) + self.mask_token * mask
        else:
            patches = patches.masked_fill(mask.bool(), 0.0)
        return patches

    def _encode_patches(self, patches: Tensor) -> Tensor:
        bs, num_patch, n_vars, patch_len = patches.shape
        aux_total = 0.0
        if not self.shared_embedding:
            embedded = []
            for i in range(n_vars):
                z, aux = self.value_embedding[i](patches[:, :, i, :])
                embedded.append(z)
                aux_total = aux_total + aux
            x = torch.stack(embedded, dim=2)
        else:
            x, aux_linear = self.value_embedding(patches)
            aux_total = aux_total + aux_linear
        x = x.transpose(1, 2)
        u = x.reshape(bs * n_vars, num_patch, self.d_model)
        pe = self.positional_encoding
        if pe.shape[0] != num_patch:
            pe = pe[:num_patch]
        u = self.dropout(u + pe)
        z, aux_encoder = self.encoder(u)
        aux_total = aux_total + aux_encoder
        z = z.reshape(bs, n_vars, num_patch, self.d_model).permute(0, 1, 3, 2)
        self._aux_loss = aux_total
        return z


class SwitchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_experts: int = 4) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_experts)])
        self.gate = nn.Linear(in_features, n_experts)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)
        top1 = gate.argmax(dim=-1)
        one_hot = F.one_hot(top1, self.n_experts).to(x.dtype)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        out = (expert_outs * one_hot.unsqueeze(-2)).sum(-1)
        mean_gate = gate.mean(dim=tuple(range(gate.dim() - 1)))
        aux_loss = (mean_gate * self.n_experts).pow(2).mean()
        return out, aux_loss


class SwitchFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(n_experts)
            ]
        )
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)
        top1 = gate.argmax(dim=-1)
        one_hot = F.one_hot(top1, self.n_experts).to(x.dtype)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        out = (expert_outs * one_hot.unsqueeze(-2)).sum(-1)
        mean_gate = gate.mean(dim=tuple(range(gate.dim() - 1)))
        aux_loss = (mean_gate * self.n_experts).pow(2).mean()
        return out, aux_loss


class TSTEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        n_layers: int = 1,
        pre_norm: bool = False,
        store_attn: bool = False,
        n_experts: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                    n_experts=n_experts,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        output = src
        scores = None
        aux_loss = 0.0
        if self.res_attention:
            for mod in self.layers:
                output, scores, aux = mod(output, prev=scores)
                aux_loss = aux_loss + aux
        else:
            for mod in self.layers:
                output, aux = mod(output)
                aux_loss = aux_loss + aux
        return output, aux_loss


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
        store_attn: bool = False,
        n_experts: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model,
            n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
        self.ff = SwitchFeedForward(d_model, d_ff, n_experts=n_experts, dropout=dropout, activation=activation)
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src: Tensor, prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)
        if self.pre_norm:
            src = self.norm_ffn(src)
        src2, aux_loss = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        if self.res_attention:
            return src, scores, aux_loss
        return src, aux_loss


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        lsa: bool = False,
    ) -> None:
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=res_attention, lsa=lsa)
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, res_attention: bool = False, lsa: bool = False) -> None:
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        if self.res_attention:
            return output, attn_weights, attn_scores
        return output, attn_weights


class Transpose(nn.Module):
    def __init__(self, *dims: int, contiguous: bool = False) -> None:
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def positional_encoding(pe: Optional[str], learn_pe: bool, q_len: int, d_model: int) -> nn.Parameter:
    if pe is None:
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe in {"normal", "gauss"}:
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "sincos":
        w_pos = _positional_encoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"Unknown positional encoding: {pe}")
    return nn.Parameter(w_pos, requires_grad=learn_pe)


def _positional_encoding(q_len: int, d_model: int, normalize: bool = True) -> torch.Tensor:
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def get_activation_fn(activation: str) -> nn.Module:
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


__all__ = [
    "PatchTSTBackbone",
]
