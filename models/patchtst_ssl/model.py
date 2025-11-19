"""Unified downstream model built on top of the SSL PatchTST backbone."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .backbone import PatchTSTBackbone
from .heads import ClassificationHead, ForecastHead, ImputationHead


class PatchTST_SSL(nn.Module):
    """PatchTST backbone plus a task-specific head.

    Examples
    --------
    Long-term forecasting::

        python run.py --task_name long_term_forecast --model PatchTST_SSL \
            --pretrained_model path/to/ssl_backbone.pth

    Classification::

        python run.py --task_name classification --model PatchTST_SSL \
            --pretrained_model path/to/ssl_backbone.pth

    Imputation::

        python run.py --task_name imputation --model PatchTST_SSL \
            --pretrained_model path/to/ssl_backbone.pth
    """

    def __init__(
        self,
        task: str,
        seq_len: int,
        pred_len: Optional[int] = None,
        n_vars: Optional[int] = None,
        num_classes: Optional[int] = None,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        use_series_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_vars is None:
            raise ValueError("n_vars must be provided for PatchTST_SSL")
        if task not in {"forecast", "classification", "imputation"}:
            raise ValueError(f"Unsupported task '{task}'")
        self.task = task
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.num_classes = num_classes
        self.use_series_norm = use_series_norm
        backbone_kwargs = dict(backbone_kwargs or {})
        backbone_kwargs.setdefault("n_vars", n_vars)
        backbone_kwargs.setdefault("seq_len", seq_len)
        self.backbone = PatchTSTBackbone(**backbone_kwargs)
        head_kwargs = dict(head_kwargs or {})
        head_dropout = head_kwargs.get("head_dropout", 0.0)
        individual = head_kwargs.get("individual", False)
        pooling = head_kwargs.get("pooling", "mean")
        if task == "forecast":
            if pred_len is None:
                raise ValueError("pred_len is required for forecasting")
            self.head = ForecastHead(
                d_model=self.backbone.d_model,
                n_patches=self.backbone.num_patches,
                pred_len=pred_len,
                n_vars=n_vars,
                head_dropout=head_dropout,
                individual=individual,
            )
        elif task == "classification":
            if num_classes is None:
                raise ValueError("num_classes is required for classification")
            self.head = ClassificationHead(
                d_model=self.backbone.d_model,
                n_patches=self.backbone.num_patches,
                num_classes=num_classes,
                pooling=pooling,
                head_dropout=head_dropout,
            )
        else:
            self.head = ImputationHead(
                d_model=self.backbone.d_model,
                n_patches=self.backbone.num_patches,
                seq_len=seq_len,
                n_vars=n_vars,
                head_dropout=head_dropout,
                individual=individual,
            )

        if pretrained_path:
            self.load_pretrained(pretrained_path)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Optional[Tensor] = None,
        x_dec: Optional[Tensor] = None,
        x_mark_dec: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.task == "forecast":
            return self._forward_forecast(x_enc, mask)
        if self.task == "classification":
            return self._forward_classification(x_enc)
        return self._forward_imputation(x_enc, mask)

    def _forward_forecast(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        x_norm, means, stdev = self._series_norm(x)
        latent = self.backbone(x_norm, mask=mask)
        out = self.head(latent)
        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return out

    def _forward_classification(self, x: Tensor) -> Tensor:
        x_norm, _, _ = self._series_norm(x)
        latent = self.backbone(x_norm)
        return self.head(latent)

    def _forward_imputation(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        if mask is None:
            raise ValueError("Imputation requires a mask tensor")
        x_norm, means, stdev = self._imputation_norm(x, mask)
        latent = self.backbone(x_norm, mask=mask)
        out = self.head(latent)
        out = out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        out = out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return out

    def _series_norm(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.use_series_norm:
            dummy = torch.zeros_like(x[:, :1, :])
            return x, dummy, torch.ones_like(dummy)
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = (x - means) / stdev
        return x_norm, means, stdev

    def _imputation_norm(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.use_series_norm:
            dummy = torch.zeros_like(x[:, :1, :])
            return x, dummy, torch.ones_like(dummy)
        mask = mask.float()
        means = torch.sum(x, dim=1, keepdim=True) / torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
        x_centered = x - means
        x_centered = x_centered.masked_fill(mask == 0, 0.0)
        stdev = torch.sqrt(
            torch.sum(x_centered * x_centered, dim=1, keepdim=True) / torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0) + 1e-5
        )
        x_norm = x_centered / stdev
        return x_norm, means.detach(), stdev.detach()

    def load_pretrained(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location="cpu")
        if "backbone_state_dict" in state:
            backbone_state = state["backbone_state_dict"]
        elif "state_dict" in state:
            backbone_state = {k.replace("backbone.", ""): v for k, v in state["state_dict"].items() if k.startswith("backbone.")}
        else:
            backbone_state = state
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=strict)
        if missing:
            print(f"PatchTST_SSL: missing backbone keys: {missing}")
        if unexpected:
            print(f"PatchTST_SSL: unexpected backbone keys: {unexpected}")


__all__ = ["PatchTST_SSL"]
