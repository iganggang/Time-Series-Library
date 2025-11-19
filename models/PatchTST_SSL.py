"""Entry point so the new PatchTST SSL model can be referenced as --model PatchTST_SSL."""
from __future__ import annotations

from .patchtst_ssl import PatchTST_SSL as _PatchTST_SSL


class Model(_PatchTST_SSL):
    """Adapts CLI/experiment arguments to the PatchTST_SSL module."""

    def __init__(self, configs) -> None:
        task_name = configs.task_name
        if task_name in {"long_term_forecast", "short_term_forecast"}:
            task = "forecast"
        elif task_name == "classification":
            task = "classification"
        elif task_name == "imputation":
            task = "imputation"
        else:
            raise ValueError(f"PatchTST_SSL does not support task '{task_name}'")

        stride = getattr(configs, "patch_stride", None)
        if stride is None:
            stride = getattr(configs, "stride", None)
        if stride is None:
            stride = max(1, getattr(configs, "patch_len", 16) // 2)

        backbone_kwargs = dict(
            patch_len=getattr(configs, "patch_len", 16),
            stride=stride,
            n_layers=getattr(configs, "e_layers", 3),
            d_model=getattr(configs, "d_model", 128),
            n_heads=getattr(configs, "n_heads", 16),
            d_ff=getattr(configs, "d_ff", 256),
            dropout=getattr(configs, "dropout", 0.1),
            attn_dropout=getattr(configs, "attn_dropout", getattr(configs, "dropout", 0.1)),
            act=getattr(configs, "activation", "gelu"),
            revin=bool(getattr(configs, "revin", 1)),
        )
        head_kwargs = dict(
            head_dropout=getattr(configs, "head_dropout", getattr(configs, "dropout", 0.1)),
            individual=bool(getattr(configs, "individual", 0)),
            pooling=getattr(configs, "pooling", "mean"),
        )
        super().__init__(
            task=task,
            seq_len=getattr(configs, "seq_len", 96),
            pred_len=getattr(configs, "pred_len", None),
            n_vars=getattr(configs, "enc_in", None),
            num_classes=getattr(configs, "num_class", None),
            backbone_kwargs=backbone_kwargs,
            head_kwargs=head_kwargs,
            pretrained_path=getattr(configs, "pretrained_model", None),
            freeze_backbone=bool(getattr(configs, "freeze_backbone", False)),
            use_series_norm=bool(getattr(configs, "use_norm", 1)),
        )

