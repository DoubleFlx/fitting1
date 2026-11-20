"""Mask target classification loss for images without pseudo boxes."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterator
import torch
from torch import nn


def pseudo_image_keep_mask(batch_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
    """True iff the image has at least one retained pseudo box (not necessarily a TAL positive)."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if batch_idx.ndim != 1:
        raise ValueError("batch_idx must have shape [number_of_pseudo_boxes]")
    if batch_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch_idx must contain integer image indices")
    if batch_idx.numel() and bool(((batch_idx < 0) | (batch_idx >= batch_size)).any()):
        raise ValueError("pseudo-box image index outside the original batch")
    keep = torch.zeros(batch_size, dtype=torch.bool, device=batch_idx.device)
    if batch_idx.numel():
        keep[batch_idx.to(torch.int64)] = True
    return keep


class _ImageMaskedBCE(nn.Module):
    def __init__(self, base: nn.Module, keep: torch.Tensor) -> None:
        super().__init__()
        if getattr(base, "reduction", None) != "none":
            raise ValueError("native BCE must have reduction='none'")
        self.base = base
        self.register_buffer("keep", keep, persistent=False)
        self.reduction = "none"

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.base(logits, labels)
        if loss.ndim != 3 or loss.shape[0] != self.keep.numel():
            raise RuntimeError("Expected elementwise BCE of shape [B, anchors, classes]")
        if loss.device != self.keep.device:
            raise RuntimeError("Pseudo image mask and loss must be on the same device")
        # Keep nonfinite losses visible.
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("Nonfinite BCE before pseudo-image masking")
        return torch.where(self.keep[:, None, None], loss, torch.zeros_like(loss))


@contextmanager
def ignore_empty_pseudo_images(
    criterion, batch_idx: torch.Tensor, batch_size: int
) -> Iterator[torch.Tensor]:
    """Temporarily mask target BCE and restore it on exit."""
    keep = pseudo_image_keep_mask(batch_idx, batch_size)
    base = criterion.bce
    if isinstance(base, _ImageMaskedBCE):
        raise RuntimeError("Nested pseudo-image masking is not supported")
    all_kept = bool(keep.all())
    if not all_kept:
        criterion.bce = _ImageMaskedBCE(base, keep)
    try:
        yield keep
    finally:
        if not all_kept:
            criterion.bce = base


def clip_xyxy_inplace(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Clamp to actual canvas; column slicing writes back to the original tensor."""
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not boxes.is_floating_point():
        raise ValueError("boxes must be a floating-point [N,4] xyxy tensor")
    if width <= 0 or height <= 0:
        raise ValueError("positive canvas dimensions required")
    if boxes.requires_grad:
        raise ValueError("Pseudo boxes must be detached teacher outputs")
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError("Nonfinite pseudo box before clipping")
    boxes[:, 0].clamp_(0, width)
    boxes[:, 2].clamp_(0, width)
    boxes[:, 1].clamp_(0, height)
    boxes[:, 3].clamp_(0, height)
    return boxes
