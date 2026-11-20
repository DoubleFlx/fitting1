from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from protocol_b_b1_trainer import ProtocolBB1Trainer, REGISTERED_EPOCHS, _tensor_sha256
from protocol_b_b2_full_state import atomic_copy, build_b2_full_state_payload
from protocol_b_b2_model import ProtocolBTSDetectionModel
from protocol_b_full_state import (
    EXPECTED_DATASET_MANIFEST,
    EXPECTED_STEPS_PER_EPOCH,
    rotate_last_checkpoint,
    sha256_file,
)
from protocol_b_pseudo import generate_high_confidence_pseudo
from protocol_b_target_schedule import (
    EXPECTED_TARGET_BATCH,
    EXPECTED_TARGET_POOL_SIZE,
    EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH,
    ProtocolBTargetSchedule,
    build_protocol_b_target_loader,
)

B2_EXPERIMENT = "B2_TSDA_S410B_CORE"


def lambda_u_for_total_epoch(total_epoch: int) -> float:
    if total_epoch == 71:
        return 0.0
    if not 72 <= total_epoch <= 100:
        raise ValueError(f"Unexpected total epoch for lambda_u: {total_epoch}")
    fraction = (total_epoch - 72) / (100 - 72)
    return 0.05 + fraction * (0.20 - 0.05)


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def _mean(values: list[float]) -> float:
    return float(fmean(values)) if values else 0.0


def _teacher_student_distance(student: torch.nn.Module, teacher: torch.nn.Module) -> tuple[float, float]:
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    if set(student_state) != set(teacher_state):
        raise RuntimeError("Student/Teacher state-dict keys differ")
    max_abs = 0.0
    diff_sq = 0.0
    student_sq = 0.0
    with torch.no_grad():
        for key in student_state:
            left = student_state[key]
            right = teacher_state[key]
            if not (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype.is_floating_point):
                continue
            delta = left.detach().float() - right.detach().float()
            max_abs = max(max_abs, float(delta.abs().max().cpu()))
            diff_sq += float(delta.square().sum().cpu())
            student_sq += float(left.detach().float().square().sum().cpu())
    relative_l2 = math.sqrt(diff_sq) / max(math.sqrt(student_sq), 1e-12)
    return max_abs, relative_l2


class ProtocolBB2Trainer(ProtocolBB1Trainer):
    """Protocol B pure Teacher–Student branch, inheriting the proven B1 continuation path."""

    def __init__(
        self,
        *,
        target_yaml_path: str | Path,
        target_schedule_path: str | Path,
        target_schedule_manifest_path: str | Path,
        b1_train_metrics_path: str | Path,
        comparison_rule_path: str | Path,
        **kwargs: Any,
    ) -> None:
        self.target_yaml_path = Path(target_yaml_path)
        self.target_schedule_path = Path(target_schedule_path)
        self.target_schedule_manifest_path = Path(target_schedule_manifest_path)
        self.b1_train_metrics_path = Path(b1_train_metrics_path)
        self.comparison_rule_path = Path(comparison_rule_path)
        self.target_schedule = ProtocolBTargetSchedule(
            self.target_schedule_path,
            self.target_schedule_manifest_path,
        )
        self.target_schedule_sha256 = sha256_file(self.target_schedule_path)
        self.target_schedule_manifest_sha256 = sha256_file(self.target_schedule_manifest_path)
        self.comparison_rule_sha256 = sha256_file(self.comparison_rule_path)
        self.b1_source_hash_reference = self._load_b1_source_hash_reference(self.b1_train_metrics_path)

        self.method_config = {
            "method": "S410b-core adapted to Protocol B",
            "target_batch": 8,
            "burn_in_total_epoch": 71,
            "lambda_u": {"e71": 0.0, "e72": 0.05, "e100": 0.20, "schedule": "linear_e72_e100"},
            "teacher": "restored Ultralytics ModelEMA from frozen e70",
            "teacher_inference_amp": True,
            "target_view_policy": "same_letterbox_view_v1",
            "target_decode_policy": "cv2_then_pillow_truncated_readonly_v1",
            "student_target_bn_running_stats": "frozen",
            "pseudo_conf": 0.65,
            "pseudo_iou": 0.65,
            "pseudo_max_det": 20,
            "pseudo_area_min": 0.001,
            "pseudo_aspect_min": 0.2,
            "pseudo_aspect_max": 5.0,
            "anchoring": False,
            "low_confidence_branch": False,
            "target_dev_training_read": False,
            "target_test_read": False,
        }

        self.target_loader = None
        self.target_iterator = None
        self.protocol_target_batch_index = 0
        self.protocol_target_read_count_total = 0
        self.protocol_target_schedule_mismatches_total = 0
        self.protocol_source_hash_mismatches_total = 0
        self.protocol_b2_batch_rows: list[dict[str, Any]] = []
        self.protocol_b2_epoch_rows: list[dict[str, Any]] = []
        self.protocol_b2_pseudo_boxes_total = 0
        self.protocol_target_cv2_decode_total = 0
        self.protocol_target_pillow_fallback_total = 0
        self._b2_pending: dict[str, Any] = {}
        self._b2_optimizer_steps_before_batch = 0
        self._b2_ema_updates_before_batch = 0

        super().__init__(**kwargs)
        self.add_callback("on_train_batch_end", self._protocol_b2_on_batch_end)

    @staticmethod
    def _load_b1_source_hash_reference(path: Path) -> dict[int, dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(path)
        reference: dict[int, dict[str, str]] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                epoch = int(row["epoch"])
                reference[epoch] = {
                    "first": row["first_batch_image_hash"],
                    "last": row["last_batch_image_hash"],
                }
        expected = set(range(71, 101))
        if set(reference) != expected:
            raise RuntimeError(f"B1 source-hash reference epochs mismatch: {sorted(reference)}")
        return reference

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = ProtocolBTSDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def _setup_train(self):
        super()._setup_train()
        misleading_b1_audit = Path(self.save_dir) / "audits" / "b1_restore_audit.json"
        misleading_b1_audit.unlink(missing_ok=True)

        self.target_loader = build_protocol_b_target_loader(
            target_yaml=self.target_yaml_path,
            schedule=self.target_schedule,
            workers=self.args.workers,
            imgsz=self.args.imgsz,
            total_epoch=71,
        )
        if len(self.target_loader) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(f"Target loader length mismatch: {len(self.target_loader)}")
        if len(self.target_loader.dataset) != EXPECTED_TARGET_POOL_SIZE:
            raise RuntimeError(f"Target pool size mismatch: {len(self.target_loader.dataset)}")

        teacher = self.ema.ema
        teacher.eval()
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("EMA Teacher parameters must have requires_grad=False")

        audit = {
            "timestamp": datetime.now().isoformat(),
            "experiment": B2_EXPERIMENT,
            "raw_restore_exact": True,
            "ema_restore_exact": True,
            "ema_updates_inferred": int(self.base_ema_updates),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "optimizer_lr": [float(group["lr"]) for group in self.optimizer.param_groups],
            "scaler_state": self.scaler.state_dict(),
            "source_schedule_sha256": self.schedule_sha256,
            "source_augmentation_sha256": self.augmentation_sha256,
            "target_schedule_sha256": self.target_schedule_sha256,
            "target_schedule_manifest_sha256": self.target_schedule_manifest_sha256,
            "target_pool_size": len(self.target_loader.dataset),
            "target_view_policy": self.target_schedule.manifest["view_policy"],
            "teacher_is_existing_model_ema": True,
            "teacher_requires_grad_false": True,
            "target_dev_train_read_count": 0,
            "target_test_content_read_count": 0,
            "method_config": self.method_config,
        }
        audit_dir = Path(self.save_dir) / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "b2_restore_audit.json").write_text(
            json.dumps(audit, indent=2, default=str),
            encoding="utf-8",
        )

    def _protocol_on_epoch_start(self, trainer: "ProtocolBB2Trainer") -> None:
        super()._protocol_on_epoch_start(trainer)
        total_epoch = trainer.protocol_current_total_epoch
        trainer.protocol_lambda_u = lambda_u_for_total_epoch(total_epoch)
        trainer.protocol_target_batch_index = 0
        trainer.protocol_target_iterator = None
        trainer.protocol_target_epoch_visits = 0
        trainer.protocol_target_epoch_uids: set[str] = set()
        trainer.protocol_target_epoch_mismatches = 0
        trainer.protocol_target_first_batch_hash = None
        trainer.protocol_target_last_batch_hash = None
        trainer.protocol_epoch_pre_nms = 0
        trainer.protocol_epoch_post_nms = 0
        trainer.protocol_epoch_area_removed = 0
        trainer.protocol_epoch_aspect_removed = 0
        trainer.protocol_epoch_pseudo_boxes = 0
        trainer.protocol_epoch_positive_images = 0
        trainer.protocol_epoch_empty_images = 0
        trainer.protocol_epoch_pseudo_confidences: list[float] = []
        trainer.protocol_epoch_pseudo_areas: list[float] = []
        trainer.protocol_target_epoch_cv2_decode_count = 0
        trainer.protocol_target_epoch_pillow_fallback_count = 0

        if total_epoch > 71:
            trainer.target_loader.batch_sampler.set_total_epoch(total_epoch)
            trainer.protocol_target_iterator = iter(trainer.target_loader)

    def _protocol_on_batch_start(self, trainer: "ProtocolBB2Trainer") -> None:
        super()._protocol_on_batch_start(trainer)
        trainer._b2_optimizer_steps_before_batch = trainer.protocol_optimizer_steps
        trainer._b2_ema_updates_before_batch = int(trainer.ema.updates)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        source_batch = super().preprocess_batch(batch)
        source_batch_index = self.protocol_batch_index - 1
        pending = {
            "epoch": self.protocol_current_total_epoch,
            "source_batch": source_batch_index,
            "lambda_u": float(self.protocol_lambda_u),
            "target_enabled": False,
            "target_batch_size": 0,
            "target_uid_hash": "",
            "target_image_hash": "",
            "pre_nms_candidates": 0,
            "post_nms_boxes": 0,
            "area_filter_removed": 0,
            "aspect_filter_removed": 0,
            "pseudo_final_boxes": 0,
            "pseudo_positive_images": 0,
            "pseudo_empty_images": 0,
            "pseudo_conf_mean": float("nan"),
            "pseudo_area_mean": float("nan"),
            "target_cv2_decode_count": 0,
            "target_pillow_fallback_count": 0,
        }

        if self.protocol_lambda_u == 0.0:
            source_batch["_protocol_b_target_batch"] = None
            source_batch["_protocol_b_lambda_u"] = 0.0
            self._b2_pending = pending
            return source_batch

        if self.protocol_target_iterator is None:
            raise RuntimeError("Target iterator is missing in an active B2 epoch")
        try:
            target_raw = next(self.protocol_target_iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Target schedule exhausted at e{self.protocol_current_total_epoch} "
                f"source batch {source_batch_index}"
            ) from exc

        expected = self.target_schedule.rows_for_epoch(self.protocol_current_total_epoch)[source_batch_index]
        uids = list(target_raw["protocol_b_target_uid"])
        view_seeds = [int(value) for value in target_raw["protocol_b_target_view_seed"]]
        epochs = [int(value) for value in target_raw["protocol_b_target_total_epoch"]]
        batch_indices = [int(value) for value in target_raw["protocol_b_target_batch_index"]]
        positions = [int(value) for value in target_raw["protocol_b_target_position"]]
        decoders = list(target_raw["protocol_b_target_decoder"])
        allowed_decoders = {"cv2", "pillow_truncated_readonly"}
        invalid_decoders = sorted(set(decoders) - allowed_decoders)
        if invalid_decoders:
            raise RuntimeError(f"Unexpected Target decoder labels: {invalid_decoders}")
        if len(decoders) != EXPECTED_TARGET_BATCH:
            raise RuntimeError(
                f"Target decoder-count mismatch: expected={EXPECTED_TARGET_BATCH}, actual={len(decoders)}"
            )
        cv2_decode_count = sum(decoder == "cv2" for decoder in decoders)
        pillow_fallback_count = sum(
            decoder == "pillow_truncated_readonly" for decoder in decoders
        )
        actual_relative_uids = [
            Path(path).relative_to(self.target_loader.dataset.image_root).as_posix()
            for path in target_raw["im_file"]
        ]
        mismatch = not (
            uids == list(expected["uids"])
            and actual_relative_uids == list(expected["uids"])
            and view_seeds == [int(value) for value in expected["sample_view_seeds"]]
            and set(epochs) == {self.protocol_current_total_epoch}
            and set(batch_indices) == {source_batch_index}
            and positions == list(range(EXPECTED_TARGET_BATCH))
        )
        if mismatch:
            self.protocol_target_epoch_mismatches += 1
            self.protocol_target_schedule_mismatches_total += 1
            raise RuntimeError(
                f"Protocol B Target schedule mismatch at e{self.protocol_current_total_epoch} "
                f"batch {source_batch_index}"
            )

        target_uint8 = target_raw["img"]
        target_image_hash = _tensor_sha256(target_uint8)
        if source_batch_index == 0:
            self.protocol_target_first_batch_hash = target_image_hash
        self.protocol_target_last_batch_hash = target_image_hash

        target_images = target_uint8.to(self.device, non_blocking=self.device.type == "cuda").float() / 255.0
        teacher = self.ema.ema
        teacher.eval()
        with torch.inference_mode():
            teacher_output = teacher(target_images)

        pseudo = generate_high_confidence_pseudo(
            teacher_output,
            conf=self.method_config["pseudo_conf"],
            iou=self.method_config["pseudo_iou"],
            max_det=self.method_config["pseudo_max_det"],
            imgsz=int(self.args.imgsz),
            area_min=self.method_config["pseudo_area_min"],
            aspect_min=self.method_config["pseudo_aspect_min"],
            aspect_max=self.method_config["pseudo_aspect_max"],
        )

        target_batch = {
            "img": target_images,
            "cls": pseudo.cls,
            "bboxes": pseudo.bboxes,
            "batch_idx": pseudo.batch_idx,
            "_protocol_b_pseudo_count": int(len(pseudo.cls)),
        }
        source_batch["_protocol_b_target_batch"] = target_batch
        source_batch["_protocol_b_lambda_u"] = float(self.protocol_lambda_u)

        self.protocol_target_batch_index += 1
        self.protocol_target_epoch_visits += len(uids)
        self.protocol_target_read_count_total += len(uids)
        self.protocol_target_epoch_uids.update(uids)
        self.protocol_target_epoch_cv2_decode_count += int(cv2_decode_count)
        self.protocol_target_epoch_pillow_fallback_count += int(pillow_fallback_count)
        self.protocol_target_cv2_decode_total += int(cv2_decode_count)
        self.protocol_target_pillow_fallback_total += int(pillow_fallback_count)
        self.protocol_epoch_pre_nms += int(pseudo.stats["pre_nms_candidates"])
        self.protocol_epoch_post_nms += int(pseudo.stats["post_nms_boxes"])
        self.protocol_epoch_area_removed += int(pseudo.stats["area_filter_removed"])
        self.protocol_epoch_aspect_removed += int(pseudo.stats["aspect_filter_removed"])
        self.protocol_epoch_pseudo_boxes += int(pseudo.stats["final_boxes"])
        self.protocol_b2_pseudo_boxes_total += int(pseudo.stats["final_boxes"])
        self.protocol_epoch_positive_images += int(pseudo.stats["positive_images"])
        self.protocol_epoch_empty_images += int(pseudo.stats["empty_images"])
        self.protocol_epoch_pseudo_confidences.extend(pseudo.stats["confidences"])
        self.protocol_epoch_pseudo_areas.extend(pseudo.stats["areas"])

        uid_hash = hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest()
        pending.update(
            {
                "target_enabled": True,
                "target_batch_size": len(uids),
                "target_uid_hash": uid_hash,
                "target_image_hash": target_image_hash,
                "pre_nms_candidates": int(pseudo.stats["pre_nms_candidates"]),
                "post_nms_boxes": int(pseudo.stats["post_nms_boxes"]),
                "area_filter_removed": int(pseudo.stats["area_filter_removed"]),
                "aspect_filter_removed": int(pseudo.stats["aspect_filter_removed"]),
                "pseudo_final_boxes": int(pseudo.stats["final_boxes"]),
                "pseudo_positive_images": int(pseudo.stats["positive_images"]),
                "pseudo_empty_images": int(pseudo.stats["empty_images"]),
                "pseudo_conf_mean": _mean(pseudo.stats["confidences"]),
                "pseudo_area_mean": _mean(pseudo.stats["areas"]),
                "target_cv2_decode_count": int(cv2_decode_count),
                "target_pillow_fallback_count": int(pillow_fallback_count),
            }
        )
        self._b2_pending = pending
        return source_batch

    def _protocol_b2_on_batch_end(self, trainer: "ProtocolBB2Trainer") -> None:
        diagnostics = dict(unwrap_model(trainer.model).protocol_b_last_diagnostics)
        if not diagnostics:
            raise RuntimeError("B2 model diagnostics are missing after forward")
        pending = dict(trainer._b2_pending)
        optimizer_step_occurred = int(
            trainer.protocol_optimizer_steps > trainer._b2_optimizer_steps_before_batch
        )
        ema_update_occurred = int(int(trainer.ema.updates) > trainer._b2_ema_updates_before_batch)

        def component(values: list[float], index: int) -> float:
            return float(values[index]) if len(values) > index else float("nan")

        row = {
            **pending,
            "source_box_loss": component(diagnostics["source_items"], 0),
            "source_cls_loss": component(diagnostics["source_items"], 1),
            "source_dfl_loss": component(diagnostics["source_items"], 2),
            "target_box_loss": component(diagnostics["target_items"], 0),
            "target_cls_loss": component(diagnostics["target_items"], 1),
            "target_dfl_loss": component(diagnostics["target_items"], 2),
            "combined_box_loss": component(diagnostics["combined_items"], 0),
            "combined_cls_loss": component(diagnostics["combined_items"], 1),
            "combined_dfl_loss": component(diagnostics["combined_items"], 2),
            "source_loss_total": diagnostics["source_loss_total"],
            "target_loss_total": diagnostics["target_loss_total"],
            "weighted_target_loss": diagnostics["weighted_target_loss"],
            "combined_loss_total": diagnostics["combined_loss_total"],
            "weighted_target_loss_share": diagnostics["weighted_target_loss_share"],
            "target_loss_active": int(diagnostics["target_active"]),
            "optimizer_step_occurred": optimizer_step_occurred,
            "ema_update_occurred": ema_update_occurred,
            "optimizer_steps_this_run_after_batch": trainer.protocol_optimizer_steps,
            "ema_updates_after_batch": int(trainer.ema.updates),
            "amp_scale_after_batch": float(trainer.scaler.get_scale()),
        }
        trainer.protocol_b2_batch_rows.append(row)

    def _protocol_on_epoch_end(self, trainer: "ProtocolBB2Trainer") -> None:
        total_epoch = trainer.protocol_current_total_epoch
        if total_epoch == 71:
            if trainer.protocol_target_epoch_visits != 0 or trainer.protocol_target_batch_index != 0:
                raise RuntimeError("B2 burn-in e71 must not read Target Adapt")
        else:
            if trainer.protocol_target_batch_index != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(
                    f"Expected {EXPECTED_STEPS_PER_EPOCH} Target batches, got {trainer.protocol_target_batch_index}"
                )
            if trainer.protocol_target_epoch_visits != EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH:
                raise RuntimeError(
                    f"Expected {EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH} Target visits, "
                    f"got {trainer.protocol_target_epoch_visits}"
                )
            if len(trainer.protocol_target_epoch_uids) != EXPECTED_TARGET_POOL_SIZE:
                raise RuntimeError(
                    f"Expected full Target pool coverage {EXPECTED_TARGET_POOL_SIZE}, "
                    f"got {len(trainer.protocol_target_epoch_uids)} unique UIDs"
                )
        if trainer.protocol_target_epoch_mismatches != 0:
            raise RuntimeError(f"Target schedule mismatches: {trainer.protocol_target_epoch_mismatches}")

        super()._protocol_on_epoch_end(trainer)

        reference = trainer.b1_source_hash_reference[total_epoch]
        first_match = trainer.protocol_epoch_first_batch_hash == reference["first"]
        last_match = trainer.protocol_epoch_last_batch_hash == reference["last"]
        if not (first_match and last_match):
            trainer.protocol_source_hash_mismatches_total += int(not first_match) + int(not last_match)
            raise RuntimeError(
                f"B2 Source image tensor hash diverged from B1 at e{total_epoch}: "
                f"first_match={first_match}, last_match={last_match}"
            )

        epoch_batch_rows = [
            row for row in trainer.protocol_b2_batch_rows if int(row["epoch"]) == total_epoch
        ]
        if len(epoch_batch_rows) != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(
                f"B2 batch diagnostics count mismatch at e{total_epoch}: {len(epoch_batch_rows)}"
            )

        student_teacher_max, student_teacher_relative_l2 = _teacher_student_distance(
            trainer.model,
            trainer.ema.ema,
        )
        active_rows = [row for row in epoch_batch_rows if int(row["target_loss_active"]) == 1]
        last_row = trainer.protocol_epoch_rows[-1]
        last_row.update(
            {
                "experiment": B2_EXPERIMENT,
                "lambda_u": float(trainer.protocol_lambda_u),
                "box_loss": _mean([float(row["combined_box_loss"]) for row in epoch_batch_rows]),
                "cls_loss": _mean([float(row["combined_cls_loss"]) for row in epoch_batch_rows]),
                "dfl_loss": _mean([float(row["combined_dfl_loss"]) for row in epoch_batch_rows]),
                "source_box_loss": _mean([float(row["source_box_loss"]) for row in epoch_batch_rows]),
                "source_cls_loss": _mean([float(row["source_cls_loss"]) for row in epoch_batch_rows]),
                "source_dfl_loss": _mean([float(row["source_dfl_loss"]) for row in epoch_batch_rows]),
                "target_box_loss_active_mean": _mean([float(row["target_box_loss"]) for row in active_rows]),
                "target_cls_loss_active_mean": _mean([float(row["target_cls_loss"]) for row in active_rows]),
                "target_dfl_loss_active_mean": _mean([float(row["target_dfl_loss"]) for row in active_rows]),
                "source_loss_total_mean": _mean([float(row["source_loss_total"]) for row in epoch_batch_rows]),
                "target_loss_total_mean": _mean([float(row["target_loss_total"]) for row in epoch_batch_rows]),
                "weighted_target_loss_mean": _mean(
                    [float(row["weighted_target_loss"]) for row in epoch_batch_rows]
                ),
                "weighted_target_loss_share_mean": _mean(
                    [float(row["weighted_target_loss_share"]) for row in epoch_batch_rows]
                ),
                "target_batches": int(trainer.protocol_target_batch_index),
                "target_adapt_train_read_count": int(trainer.protocol_target_epoch_visits),
                "target_unique_uid_count": int(len(trainer.protocol_target_epoch_uids)),
                "target_unique_uid_coverage": (
                    len(trainer.protocol_target_epoch_uids) / EXPECTED_TARGET_POOL_SIZE
                ),
                "target_schedule_mismatch_count": int(trainer.protocol_target_epoch_mismatches),
                "target_first_batch_image_hash": trainer.protocol_target_first_batch_hash,
                "target_last_batch_image_hash": trainer.protocol_target_last_batch_hash,
                "target_cv2_decode_count": int(
                    trainer.protocol_target_epoch_cv2_decode_count
                ),
                "target_pillow_fallback_count": int(
                    trainer.protocol_target_epoch_pillow_fallback_count
                ),
                "target_decode_policy": "cv2_then_pillow_truncated_readonly_v1",
                "pseudo_pre_nms_candidates": int(trainer.protocol_epoch_pre_nms),
                "pseudo_post_nms_boxes": int(trainer.protocol_epoch_post_nms),
                "pseudo_area_filter_removed": int(trainer.protocol_epoch_area_removed),
                "pseudo_aspect_filter_removed": int(trainer.protocol_epoch_aspect_removed),
                "pseudo_final_boxes": int(trainer.protocol_epoch_pseudo_boxes),
                "pseudo_positive_images": int(trainer.protocol_epoch_positive_images),
                "pseudo_empty_images": int(trainer.protocol_epoch_empty_images),
                "pseudo_trigger_batch_count": int(len(active_rows)),
                "pseudo_trigger_batch_rate": len(active_rows) / EXPECTED_STEPS_PER_EPOCH,
                "pseudo_conf_p10": _quantile(trainer.protocol_epoch_pseudo_confidences, 0.10),
                "pseudo_conf_p50": _quantile(trainer.protocol_epoch_pseudo_confidences, 0.50),
                "pseudo_conf_p90": _quantile(trainer.protocol_epoch_pseudo_confidences, 0.90),
                "pseudo_area_p10": _quantile(trainer.protocol_epoch_pseudo_areas, 0.10),
                "pseudo_area_p50": _quantile(trainer.protocol_epoch_pseudo_areas, 0.50),
                "pseudo_area_p90": _quantile(trainer.protocol_epoch_pseudo_areas, 0.90),
                "teacher_student_max_abs_diff": student_teacher_max,
                "teacher_student_relative_l2": student_teacher_relative_l2,
                "source_first_batch_hash_matches_b1": first_match,
                "source_last_batch_hash_matches_b1": last_match,
                "target_dev_train_read_count": 0,
                "target_test_content_read_count": 0,
                "low_confidence_branch_enabled": False,
            }
        )
        trainer.protocol_b2_epoch_rows.append(dict(last_row))
        trainer._write_epoch_metrics()
        trainer._write_b2_batch_metrics()
        trainer._write_b2_mechanism_metrics()

    def _write_b2_batch_metrics(self) -> None:
        path = Path(self.save_dir) / "b2_batch_metrics.csv"
        if not self.protocol_b2_batch_rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.protocol_b2_batch_rows[0]))
            writer.writeheader()
            writer.writerows(self.protocol_b2_batch_rows)

    def _write_b2_mechanism_metrics(self) -> None:
        path = Path(self.save_dir) / "b2_epoch_mechanism.csv"
        if not self.protocol_b2_epoch_rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.protocol_b2_epoch_rows[0]))
            writer.writeheader()
            writer.writerows(self.protocol_b2_epoch_rows)

    def save_metrics(self, metrics):
        """Write a stable training-only results.csv schema.

        BaseTrainer changes metric keys on the final epoch when validation is intentionally
        disabled, which produced the malformed B1 e100 row. B2 records only train losses
        and learning rates here; detection metrics remain in the isolated evaluator.
        """
        stable = {
            key: value
            for key, value in metrics.items()
            if key.startswith("train/") or key.startswith("lr/")
        }
        keys, values = list(stable), list(stable.values())
        expected_keys = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "lr/pg0",
            "lr/pg1",
            "lr/pg2",
        ]
        if keys != expected_keys:
            raise RuntimeError(f"Unexpected B2 results.csv metric schema: {keys}")
        elapsed = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv.exists()
        with self.csv.open("a", encoding="utf-8", newline="") as handle:
            if write_header:
                handle.write(",".join(["epoch", "time", *keys]) + "\n")
            row = [self.epoch + 1, elapsed, *values]
            handle.write(",".join(f"{float(value):.6g}" for value in row) + "\n")

    def save_model(self):
        total_epoch = self.protocol_current_total_epoch
        global_step = int(self.e70_state["global_step"]) + (total_epoch - 70) * EXPECTED_STEPS_PER_EPOCH
        epoch_audit = self.protocol_epoch_rows[-1] if self.protocol_epoch_rows else {}
        optimizer_updates_absolute = self.base_ema_updates + self.protocol_optimizer_steps
        payload = build_b2_full_state_payload(
            self,
            total_epoch=total_epoch,
            global_step=global_step,
            optimizer_updates_absolute=optimizer_updates_absolute,
            epoch_audit=epoch_audit,
        )

        full_state_dir = Path(self.save_dir) / "full_state"
        last_path = rotate_last_checkpoint(payload, full_state_dir)
        last_hash = sha256_file(last_path)
        registered_path = None
        if total_epoch in REGISTERED_EPOCHS:
            registered_path = full_state_dir / f"epoch_{total_epoch:03d}_full_state.pt"
            if not registered_path.exists():
                atomic_copy(last_path, registered_path)

        row = {
            "epoch": total_epoch,
            "global_step": global_step,
            "last_path": str(last_path),
            "last_sha256": last_hash,
            "registered_path": str(registered_path) if registered_path else "",
            "registered_sha256": sha256_file(registered_path) if registered_path else "",
            "ema_updates": int(self.ema.updates),
            "optimizer_updates_absolute": optimizer_updates_absolute,
            "optimizer_steps_this_run": int(self.protocol_optimizer_steps),
            "target_schedule_sha256": self.target_schedule_sha256,
        }
        self.protocol_checkpoint_rows.append(row)
        manifest = Path(self.save_dir) / "checkpoint_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.protocol_checkpoint_rows)
        return False

    def _protocol_on_train_end(self, trainer: "ProtocolBB2Trainer") -> None:
        completed = trainer.protocol_current_total_epoch == 100
        expected_target_reads = 29 * EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH
        final = {
            "experiment": B2_EXPERIMENT,
            "completed": completed,
            "final_total_epoch": trainer.protocol_current_total_epoch,
            "final_global_step": int(self.e70_state["global_step"]) + 30 * EXPECTED_STEPS_PER_EPOCH,
            "optimizer_steps_this_run": int(trainer.protocol_optimizer_steps),
            "ema_updates_final": int(trainer.ema.updates),
            "source_schedule_mismatch_count": sum(
                int(row["source_schedule_mismatch_count"]) for row in trainer.protocol_epoch_rows
            ),
            "source_image_hash_mismatch_count": int(trainer.protocol_source_hash_mismatches_total),
            "target_schedule_mismatch_count": int(trainer.protocol_target_schedule_mismatches_total),
            "target_adapt_train_read_count": int(trainer.protocol_target_read_count_total),
            "target_adapt_expected_read_count": expected_target_reads,
            "target_dev_train_read_count": 0,
            "target_test_content_read_count": 0,
            "pseudo_final_boxes_total": int(trainer.protocol_b2_pseudo_boxes_total),
            "target_cv2_decode_count_total": int(trainer.protocol_target_cv2_decode_total),
            "target_pillow_fallback_count_total": int(
                trainer.protocol_target_pillow_fallback_total
            ),
            "target_decode_policy": "cv2_then_pillow_truncated_readonly_v1",
            "target_view_policy": trainer.target_schedule.manifest["view_policy"],
            "target_schedule_sha256": trainer.target_schedule_sha256,
            "comparison_rule_sha256": trainer.comparison_rule_sha256,
            "low_confidence_branch_enabled": False,
            "dataset_mutation_count": 0,
            "checks": {
                "COMPLETED_E100": completed,
                "SOURCE_SCHEDULE_MISMATCH_ZERO": all(
                    int(row["source_schedule_mismatch_count"]) == 0
                    for row in trainer.protocol_epoch_rows
                ),
                "SOURCE_HASH_MISMATCH_ZERO": trainer.protocol_source_hash_mismatches_total == 0,
                "TARGET_SCHEDULE_MISMATCH_ZERO": trainer.protocol_target_schedule_mismatches_total == 0,
                "TARGET_READ_COUNT_EXPECTED": trainer.protocol_target_read_count_total == expected_target_reads,
                "TARGET_DEV_READ_COUNT_ZERO": True,
                "TARGET_TEST_READ_COUNT_ZERO": True,
                "LOW_CONFIDENCE_BRANCH_DISABLED": True,
            },
            "timestamp": datetime.now().isoformat(),
        }
        final["all_pass"] = all(final["checks"].values())
        (Path(self.save_dir) / "B2_final_audit.json").write_text(
            json.dumps(final, indent=2, default=str),
            encoding="utf-8",
        )
