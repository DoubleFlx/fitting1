from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel


class FreezeBatchNormStats(AbstractContextManager):
    """Freeze BN running statistics during the Student Target forward only."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.saved: list[tuple[nn.Module, bool, float | None]] = []

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                self.saved.append((module, module.training, module.momentum))
                module.eval()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, training, momentum in self.saved:
            module.train(training)
            module.momentum = momentum
        self.saved.clear()
        return False


class ProtocolBTSDetectionModel(DetectionModel):
    """DetectionModel that adds a high-confidence pseudo-label loss after Source loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.protocol_b_last_diagnostics: dict[str, Any] = {}

    def loss(self, batch: dict[str, Any], preds=None):
        target_batch = batch.get("_protocol_b_target_batch")
        lambda_u = float(batch.get("_protocol_b_lambda_u", 0.0))

        # Source is always first and follows the same DetectionModel loss path as B1.
        source_loss, source_items = super().loss(batch, preds)

        target_loss = torch.zeros_like(source_loss)
        target_items = torch.zeros_like(source_items)
        pseudo_count = 0
        target_active = False

        if target_batch is not None:
            pseudo_count = int(target_batch.get("_protocol_b_pseudo_count", 0))
            if lambda_u > 0.0 and pseudo_count > 0:
                with FreezeBatchNormStats(self):
                    target_loss, target_items = super().loss(target_batch, preds=None)
                target_active = True

        combined_loss = source_loss + lambda_u * target_loss
        combined_items = source_items + lambda_u * target_items

        tensors_to_check = {
            "source_loss": source_loss,
            "source_items": source_items,
            "target_loss": target_loss,
            "target_items": target_items,
            "combined_loss": combined_loss,
            "combined_items": combined_items,
        }
        for name, value in tensors_to_check.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite {name} in Protocol B B2")

        source_total = source_loss.detach().sum()
        target_total = target_loss.detach().sum()
        weighted_target = target_total * lambda_u
        combined_total = combined_loss.detach().sum()
        share = (
            weighted_target / combined_total.clamp(min=torch.finfo(combined_total.dtype).eps)
            if combined_total.numel()
            else combined_total.new_tensor(0.0)
        )

        self.protocol_b_last_diagnostics = {
            "lambda_u": lambda_u,
            "pseudo_count": pseudo_count,
            "target_active": target_active,
            "source_items": source_items.detach().float().cpu().tolist(),
            "target_items": target_items.detach().float().cpu().tolist(),
            "combined_items": combined_items.detach().float().cpu().tolist(),
            "source_loss_total": float(source_total.float().cpu()),
            "target_loss_total": float(target_total.float().cpu()),
            "weighted_target_loss": float(weighted_target.float().cpu()),
            "combined_loss_total": float(combined_total.float().cpu()),
            "weighted_target_loss_share": float(share.float().cpu()),
        }
        return combined_loss, combined_items
