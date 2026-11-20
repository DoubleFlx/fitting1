from __future__ import annotations

"""Original B2 loss with read-only M1 mechanism diagnostics."""

from typing import Any

import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

from protocol_b_b2_model import FreezeBatchNormStats
from protocol_b_m1_common import (
    bn_running_state_clone,
    state_max_abs_delta,
    tensor_state_hash,
)


class M1AuditedV8DetectionLoss(v8DetectionLoss):
    """Observe the native TAL foreground count without altering its output."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.m1_last_positive_assignments = 0

    def get_assigned_targets_and_loss(self, preds, batch):
        output = super().get_assigned_targets_and_loss(preds, batch)
        self.m1_last_positive_assignments = int(output[0][0].sum().detach().cpu())
        return output


class ProtocolBM1DetectionModel(DetectionModel):
    """Mathematically identical B2 loss plus read-only TAL and BN evidence."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.protocol_b_last_diagnostics: dict[str, Any] = {}

    def init_criterion(self):
        if getattr(self, "end2end", False):
            raise RuntimeError("M1 is frozen to the ordinary non-end-to-end YOLO11 detector")
        return M1AuditedV8DetectionLoss(self)

    def loss(self, batch: dict[str, Any], preds=None):
        target_batch = batch.get("_protocol_b_target_batch")
        lambda_u = float(batch.get("_protocol_b_lambda_u", 0.0))

        source_loss, source_items = DetectionModel.loss(self, batch, preds)
        source_positive_assignments = int(
            self.criterion.m1_last_positive_assignments
        )
        bn_after_source = bn_running_state_clone(self)
        bn_after_source_sha = tensor_state_hash(bn_after_source)

        target_loss = torch.zeros_like(source_loss)
        target_items = torch.zeros_like(source_items)
        target_positive_assignments = 0
        pseudo_count = 0
        target_active = False
        if target_batch is not None:
            pseudo_count = int(target_batch.get("_protocol_b_pseudo_count", 0))
            if lambda_u > 0.0 and pseudo_count > 0:
                with FreezeBatchNormStats(self):
                    target_loss, target_items = DetectionModel.loss(
                        self, target_batch, preds=None
                    )
                target_positive_assignments = int(
                    self.criterion.m1_last_positive_assignments
                )
                target_active = True

        bn_after_target = bn_running_state_clone(self)
        bn_after_target_sha = tensor_state_hash(bn_after_target)
        bn_float_delta, bn_nonfloat_mismatches = state_max_abs_delta(
            bn_after_source, bn_after_target
        )
        if bn_float_delta != 0.0 or bn_nonfloat_mismatches:
            raise RuntimeError(
                "STOP_TARGET_BN_BEHAVIOR_MISMATCH: "
                f"max_abs_delta={bn_float_delta}, nonfloat={bn_nonfloat_mismatches}"
            )

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
                raise FloatingPointError(f"Non-finite {name} in M1")

        source_total = source_loss.detach().sum()
        target_total = target_loss.detach().sum()
        weighted_target = target_total * lambda_u
        combined_total = combined_loss.detach().sum()
        share = weighted_target / combined_total.clamp(
            min=torch.finfo(combined_total.dtype).eps
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
            "source_tal_positive_assignments": source_positive_assignments,
            "target_tal_positive_assignments": target_positive_assignments,
            "bn_after_source_sha": bn_after_source_sha,
            "bn_after_target_sha": bn_after_target_sha,
            "target_forward_bn_buffer_max_abs_delta": bn_float_delta,
            "target_forward_bn_nonfloat_mismatch_count": len(
                bn_nonfloat_mismatches
            ),
        }
        return combined_loss, combined_items

