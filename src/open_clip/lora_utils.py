"""
LoRA utilities for pathology CLIP fine-tuning.

Supports:
  - Vision: any timm ViT (bioptimus/H0-mini, bioptimus/H-optimus-0, owkin/aquavit …)
  - Text:   HuggingFace BERT-style encoders (BiomedBERT-large, BioBERT, …)

LoRA is injected via PEFT (pip install peft).  Only LoRA parameters are
trainable after wrapping; base weights stay frozen.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    raise ImportError("peft is required: pip install peft") from e


# ---------------------------------------------------------------------------
# Per-model timm kwargs
# ---------------------------------------------------------------------------

def _timm_extra_kwargs(model_name: str) -> dict:
    """Return timm.create_model kwargs specific to each model family."""
    name_lower = model_name.lower()
    if "h-optimus" in name_lower or "h0" in name_lower:
        return {"init_values": 1e-5, "img_size": 224, "dynamic_img_size": False}
    # AquaViT and SwiGLU-based ViTs
    return {
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
    }


def _resolve_embed_dim(model: nn.Module) -> int:
    for attr in ("num_features", "embed_dim"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot determine embed_dim for {type(model).__name__}. "
        "Set embed_dim explicitly."
    )


# ---------------------------------------------------------------------------
# Vision backbone (timm ViT) with LoRA
# ---------------------------------------------------------------------------

def _load_timm_base(
    model_name: str,
    model_local_dir: Optional[str] = None,
) -> nn.Module:
    """Load a timm model online or from a local directory."""
    extra_kwargs = _timm_extra_kwargs(model_name)

    if model_local_dir is None:
        return timm.create_model(model_name, pretrained=True, **extra_kwargs)

    local_dir = Path(model_local_dir)

    # HF cache snapshot layout (.../snapshots/<hash>/)
    if "/snapshots/" in str(local_dir):
        hub_cache = local_dir.parent.parent.parent
        old_cache   = os.environ.get("HF_HUB_CACHE")
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_CACHE"]   = str(hub_cache)
            os.environ["HF_HUB_OFFLINE"] = "1"
            return timm.create_model(model_name, pretrained=True, **extra_kwargs)
        finally:
            if old_cache is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = old_cache
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

    # Flat download layout (config.json + *.safetensors / *.bin)
    import json as _json
    from timm.models import load_checkpoint as timm_load_checkpoint
    cfg_path = local_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    with open(cfg_path) as f:
        timm_cfg = _json.load(f)
    architecture = timm_cfg.get("architecture")
    if not architecture:
        raise KeyError(f"'architecture' key missing in {cfg_path}")

    weights_file = (
        next(local_dir.glob("*.safetensors"), None)
        or next(local_dir.glob("*.bin"), None)
    )
    if weights_file is None:
        raise FileNotFoundError(f"No weight file (*.safetensors / *.bin) in {local_dir}")

    model = timm.create_model(
        architecture,
        pretrained=False,
        **{**extra_kwargs, **timm_cfg.get("model_args", {})},
    )
    timm_load_checkpoint(model, str(weights_file), strict=True)
    return model


class VisionEncoderWithLoRA(nn.Module):
    """timm ViT backbone with frozen base weights and trainable LoRA adapters.

    Parameters
    ----------
    model_name : str
        timm model identifier, e.g. ``"hf-hub:bioptimus/H0-mini"``.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor (keep lora_alpha / lora_r ≈ 2).
    lora_dropout : float
        Dropout applied inside LoRA adapters.
    target_modules : list[str]
        timm module names to adapt.  Default ``["qkv"]`` targets the fused
        attention QKV projection in standard timm ViT blocks.
    model_local_dir : str | None
        If set, load weights from this local directory instead of HuggingFace.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:bioptimus/H0-mini",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        model_local_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if target_modules is None:
            target_modules = ["qkv"]

        base = _load_timm_base(model_name, model_local_dir)
        self._embed_dim = _resolve_embed_dim(base)

        # Freeze all base weights
        for p in base.parameters():
            p.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.backbone = get_peft_model(base, lora_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        # Some timm models return (B, n_tokens, D); take CLS at index 0.
        # Models with global_pool="token" return (B, D) directly.
        return out[:, 0] if out.ndim == 3 else out

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def print_trainable_summary(self) -> None:
        self.backbone.print_trainable_parameters()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Text encoder (HuggingFace BERT-style) with LoRA
# ---------------------------------------------------------------------------

class TextEncoderWithLoRA(nn.Module):
    """HuggingFace BERT-style encoder with frozen base weights and LoRA adapters.

    Uses mean-pooling over non-padding tokens by default.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
        Default: ``"microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext"``.
    lora_r : int
        LoRA rank.
    lora_alpha : int
        LoRA scaling factor.
    lora_dropout : float
        Dropout applied inside LoRA adapters.
    target_modules : list[str]
        HF module names to adapt.  Default ``["query", "key", "value"]``
        targets the BERT self-attention projections.
    max_length : int
        Maximum tokenizer sequence length.
    """

    def __init__(
        self,
        model_name_or_path: str = (
            "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract-fulltext"
        ),
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        if target_modules is None:
            target_modules = ["query", "key", "value"]

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required: pip install transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        base = AutoModel.from_pretrained(model_name_or_path)
        self.max_length = max_length

        # Determine text embedding dim from model config
        self._embed_dim: int = base.config.hidden_size

        # Freeze all base weights
        for p in base.parameters():
            p.requires_grad_(False)

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_cfg)

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        """Tokenise a list of strings and return tensors on `device`."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over non-padding tokens
        hidden = out.last_hidden_state                            # (B, L, D)
        mask   = attention_mask.unsqueeze(-1).float()            # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)   # (B, D)
        return pooled

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def print_trainable_summary(self) -> None:
        self.encoder.print_trainable_parameters()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Frozen teacher (for knowledge distillation)
# ---------------------------------------------------------------------------

class FrozenVisionTeacher(nn.Module):
    """Frozen copy of the vision backbone (no LoRA) for distillation."""

    def __init__(self, model_name: str, model_local_dir: Optional[str] = None) -> None:
        super().__init__()
        base = _load_timm_base(model_name, model_local_dir)
        for p in base.parameters():
            p.requires_grad_(False)
        self.backbone = base
        self._embed_dim = _resolve_embed_dim(base)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return out[:, 0] if out.ndim == 3 else out

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_lora_checkpoint(
    path: str | Path,
    vision_backbone: VisionEncoderWithLoRA,
    text_encoder: TextEncoderWithLoRA,
    vision_proj: nn.Module,
    text_proj: nn.Module,
    logit_scale: float,
    epoch: int,
    step: int,
    cfg: dict,
) -> None:
    """Save only the trainable weights + projections to `path`."""
    torch.save(
        {
            "vision_backbone": vision_backbone.state_dict(),
            "text_encoder":    text_encoder.state_dict(),
            "vision_proj":     vision_proj.state_dict(),
            "text_proj":       text_proj.state_dict(),
            "logit_scale":     logit_scale,
            "epoch":           epoch,
            "step":            step,
            "cfg":             cfg,
        },
        path,
    )


def load_vision_backbone_from_checkpoint(
    checkpoint_path: str | Path,
    model_name: str = "hf-hub:bioptimus/H0-mini",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    model_local_dir: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> VisionEncoderWithLoRA:
    """Reconstruct a :class:`VisionEncoderWithLoRA` from a training checkpoint.

    The checkpoint must have been saved by :func:`save_lora_checkpoint`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Allow cfg stored in checkpoint to override defaults
    saved_cfg = ckpt.get("cfg", {})
    model_name     = saved_cfg.get("vision_model_name", model_name)
    lora_r         = saved_cfg.get("lora_r",            lora_r)
    lora_alpha     = saved_cfg.get("lora_alpha",         lora_alpha)
    lora_dropout   = saved_cfg.get("lora_dropout",       lora_dropout)
    target_modules = saved_cfg.get("vision_target_modules", target_modules)
    model_local_dir = saved_cfg.get("vision_model_local_dir", model_local_dir)

    backbone = VisionEncoderWithLoRA(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        model_local_dir=model_local_dir,
    )
    backbone.load_state_dict(ckpt["vision_backbone"])
    return backbone.to(device)


# ---------------------------------------------------------------------------
# open_clip integration helpers
# ---------------------------------------------------------------------------

def apply_lora_to_clip(model: nn.Module, args) -> nn.Module:
    """Apply LoRA adapters to an open_clip ``CustomTextCLIP`` model.

    Called by ``open_clip_train.main`` after ``create_model_and_transforms``.
    Freezes all base weights, wraps ``model.visual.trunk`` (timm ViT) and
    ``model.text.transformer`` (HF BERT) with peft LoRA, and keeps the
    projection heads and ``logit_scale`` trainable.

    Parameters
    ----------
    model : nn.Module
        A ``CustomTextCLIP`` instance (output of ``create_model_and_transforms``).
    args : argparse.Namespace
        Must contain ``lora_r``, ``lora_alpha``, ``lora_dropout``,
        ``lora_vision_target_modules``, and ``lora_text_target_modules``.

    Returns
    -------
    nn.Module
        The same model, modified in-place with LoRA adapters.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("peft is required for LoRA: pip install peft") from exc

    # 1. Freeze every parameter
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. LoRA on vision trunk (timm ViT inside TimmModel)
    vision_lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_vision_target_modules),
        bias="none",
    )
    model.visual.trunk = get_peft_model(model.visual.trunk, vision_lora_cfg)

    # 3. LoRA on text transformer (HF BERT inside HFTextEncoder)
    text_lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_text_target_modules),
        bias="none",
    )
    model.text.transformer = get_peft_model(model.text.transformer, text_lora_cfg)

    # 4. Keep vision projection head trainable
    if hasattr(model.visual, "head") and model.visual.head is not None:
        for p in model.visual.head.parameters():
            p.requires_grad_(True)

    # 5. Keep text projection trainable
    if hasattr(model.text, "proj") and model.text.proj is not None:
        if isinstance(model.text.proj, nn.Parameter):
            model.text.proj.requires_grad_(True)
        else:
            for p in model.text.proj.parameters():
                p.requires_grad_(True)

    # 6. Keep logit_scale (and logit_bias for SigLIP) trainable
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad_(True)
    if hasattr(model, "logit_bias") and model.logit_bias is not None:
        model.logit_bias.requires_grad_(True)

    return model


def load_visual_from_open_clip_checkpoint(
    checkpoint_path: Optional[str | Path],
    model_name: str = "H0-mini-BiomedBERT",
    apply_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_vision_target_modules: list[str] | None = None,
    lora_text_target_modules: list[str] | None = None,
    device: str | torch.device = "cpu",
):
    """Load the vision backbone from an open_clip training checkpoint.

    Creates the model with open_clip's factory (using the registered JSON
    config), optionally applies LoRA, loads the state dict, and returns
    ``(model.visual, preprocess_val)``.

    Parameters
    ----------
    checkpoint_path : str | Path | None
        Path to an open_clip checkpoint (``epoch_*.pt``).  Pass ``None`` to
        get the untrained / pre-trained-only backbone.
    model_name : str
        Registered open_clip model name (default ``"H0-mini-BiomedBERT"``).
    apply_lora : bool
        Whether to apply LoRA before loading the checkpoint weights.
    device : str or torch.device

    Returns
    -------
    (nn.Module, callable)
        ``model.visual`` (TimmModel) and ``preprocess_val`` transform.
    """
    import types
    try:
        from open_clip import create_model_and_transforms
    except ImportError as exc:
        raise ImportError("open_clip is required: install from the repo root") from exc

    model, _, preprocess_val = create_model_and_transforms(
        model_name,
        pretrained="",  # weights loaded from checkpoint below
        device=device,
    )

    if apply_lora:
        lora_args = types.SimpleNamespace(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_vision_target_modules=lora_vision_target_modules or ["qkv"],
            lora_text_target_modules=lora_text_target_modules or ["query", "key", "value"],
        )
        model = apply_lora_to_clip(model, lora_args)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        # Strip 'module.' prefix (DDP checkpoints)
        if sd and next(iter(sd)).startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            import warnings
            warnings.warn(f"Missing keys when loading checkpoint: {missing[:5]}…")

    return model.visual.to(device), preprocess_val
