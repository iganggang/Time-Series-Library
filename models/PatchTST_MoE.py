from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from layers.PatchTST_layers import Transpose, get_activation_fn, positional_encoding
from layers.RevIN import RevIN


class PatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: Union[bool, str] = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch: Optional[str] = None,
        pretrain_head: bool = False,
        head_type: str = 'flatten',
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        n_experts: int = 4,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            n_experts=n_experts,
            **kwargs,
        )

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout)
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        else:
            self.head = nn.Identity()

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        z, aux_loss = self.backbone(z)
        z = self.head(z)

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z, aux_loss

    def create_pretrain_head(self, head_nf: int, vars: int, dropout: float) -> nn.Module:
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(nn.Module):
    def __init__(self, individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0) -> None:
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.flattens = nn.ModuleList()
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


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
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                get_activation_fn(activation),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(n_experts)
        ])
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


class TSTiEncoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        max_seq_len: int = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        key_padding_mask: Union[bool, str] = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        verbose: bool = False,
        n_experts: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        q_len = patch_num
        self.W_P = SwitchLinear(patch_len, d_model, n_experts=n_experts)
        self.seq_len = q_len
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
            n_experts=n_experts,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x, aux_linear = self.W_P(x)

        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)

        z, aux_encoder = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)

        return z, aux_linear + aux_encoder


class TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        activation: str = 'gelu',
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
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
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

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        output = src
        scores = None
        aux_loss = torch.zeros((), device=src.device, dtype=src.dtype)
        if self.res_attention:
            for mod in self.layers:
                output, scores, l_aux = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                aux_loss = aux_loss + l_aux
            return output, aux_loss
        else:
            for mod in self.layers:
                output, l_aux = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
                aux_loss = aux_loss + l_aux
            return output, aux_loss


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
        n_experts: int = 4,
    ) -> None:
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

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

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        if self.pre_norm:
            src = self.norm_attn(src)
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
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
        else:
            return src, aux_loss


class _MultiheadAttention(nn.Module):
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
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=res_attention, lsa=lsa)

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(
        self,
        Q: Tensor,
        K: Optional[Tensor] = None,
        V: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
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
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
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
        else:
            return output, attn_weights


class Model(nn.Module):
    """PatchTST with a Mixture-of-Experts feed-forward network."""

    def __init__(self, configs) -> None:
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.num_class = getattr(configs, 'num_class', 0)
        self.enc_in = configs.enc_in

        patch_len = getattr(configs, 'patch_len', 16)
        stride = getattr(configs, 'stride', patch_len // 2)
        n_experts = getattr(configs, 'moe_experts', 4)
        attn_dropout = getattr(configs, 'attn_dropout', configs.dropout)
        head_dropout = getattr(configs, 'head_dropout', configs.dropout)

        target_window = self.pred_len if self.task_name in {'long_term_forecast', 'short_term_forecast'} else self.seq_len

        self.backbone = PatchTST_backbone(
            c_in=self.enc_in,
            context_window=self.seq_len,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=self.seq_len,
            n_layers=configs.e_layers,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            d_ff=configs.d_ff,
            norm='BatchNorm',
            attn_dropout=attn_dropout,
            dropout=configs.dropout,
            act=configs.activation,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe='zeros',
            learn_pe=True,
            fc_dropout=configs.dropout,
            head_dropout=head_dropout,
            padding_patch=None,
            pretrain_head=False,
            head_type='flatten',
            individual=False,
            revin=True,
            affine=True,
            subtract_last=False,
            n_experts=n_experts,
        )

        if self.task_name == 'classification':
            self.cls_dropout = nn.Dropout(configs.dropout)
            self.cls_projection = nn.Linear(self.enc_in * target_window, self.num_class)
        else:
            self.cls_dropout = None
            self.cls_projection = None

        self.last_aux_loss: Optional[Tensor] = None

    def get_aux_loss(self) -> Tensor:
        if self.last_aux_loss is None:
            param = next(self.parameters(), None)
            if param is None:
                return torch.zeros(())
            return param.new_zeros(())
        return self.last_aux_loss

    def forecast(self, x_enc: Tensor, *args) -> Tensor:
        x_enc = x_enc.permute(0, 2, 1)
        dec_out, aux_loss = self.backbone(x_enc)
        self.last_aux_loss = aux_loss
        dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def classification(self, x_enc: Tensor, *args) -> Tensor:
        x_enc = x_enc.permute(0, 2, 1)
        dec_out, aux_loss = self.backbone(x_enc)
        self.last_aux_loss = aux_loss
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out.reshape(dec_out.shape[0], -1)
        dec_out = self.cls_dropout(dec_out)
        dec_out = self.cls_projection(dec_out)
        return dec_out

    def forward(self, x_enc: Tensor, x_mark_enc: Optional[Tensor], x_dec: Optional[Tensor], x_mark_dec: Optional[Tensor], mask: Optional[Tensor] = None) -> Tensor:
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'classification':
            return self.classification(x_enc)
        raise NotImplementedError(f"Task {self.task_name} is not supported by PatchTST_MoE.")


__all__ = ['Model']
