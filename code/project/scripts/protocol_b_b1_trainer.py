from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model

from protocol_b_full_state import (
    EXPECTED_DATASET_MANIFEST,
    EXPECTED_FROZEN_E70_SHA256,
    EXPECTED_STEPS_PER_EPOCH,
    build_full_state_payload,
    infer_ema_updates,
    restore_rng_state,
    rotate_last_checkpoint,
    sha256_file,
    state_dict_max_abs_diff,
    validate_e70_state,
)
from protocol_b_schedule import ProtocolBSchedule, build_protocol_b_source_loader

REGISTERED_EPOCHS = {75, 80, 85, 90, 94, 98, 100}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class ProtocolBB1Trainer(DetectionTrainer):
    """Source-only Protocol B continuation from the frozen e70 full state."""

    def __init__(
        self,
        *,
        full_state_path: str | Path,
        source_schedule_path: str | Path,
        source_augmentation_path: str | Path,
        contract_path: str | Path,
        code_snapshot_sha256: str | None = None,
        environment_sha256: str | None = None,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, list] | None = None,
    ) -> None:
        self.full_state_path = Path(full_state_path)
        self.source_schedule_path = Path(source_schedule_path)
        self.source_augmentation_path = Path(source_augmentation_path)
        self.contract_path = Path(contract_path)
        self.e70_state = validate_e70_state(self.full_state_path)
        self.base_checkpoint_sha256 = sha256_file(self.full_state_path)
        self.schedule_sha256 = sha256_file(self.source_schedule_path)
        self.augmentation_sha256 = sha256_file(self.source_augmentation_path)
        self.code_snapshot_sha256 = code_snapshot_sha256
        self.environment_sha256 = environment_sha256
        self.protocol_schedule = ProtocolBSchedule(self.source_schedule_path, self.source_augmentation_path)
        self.base_ema_updates = infer_ema_updates(self.e70_state["global_step"])
        self.protocol_optimizer_steps = 0
        self.protocol_last_grad_norm = float("nan")
        self.protocol_batch_index = 0
        self.protocol_epoch_image_visits = 0
        self.protocol_epoch_mismatches = 0
        self.protocol_epoch_first_batch_hash: str | None = None
        self.protocol_epoch_last_batch_hash: str | None = None
        self.protocol_epoch_rows: list[dict[str, Any]] = []
        self.protocol_checkpoint_rows: list[dict[str, Any]] = []
        self.protocol_current_total_epoch = 70
        super().__init__(cfg=DEFAULT_CFG if cfg is None else cfg, overrides=overrides or {}, _callbacks=_callbacks)
        self.add_callback("on_train_epoch_start", self._protocol_on_epoch_start)
        self.add_callback("on_train_batch_start", self._protocol_on_batch_start)
        self.add_callback("on_train_epoch_end", self._protocol_on_epoch_end)
        self.add_callback("on_train_end", self._protocol_on_train_end)

    def setup_model(self):
        # Build nc=1 architecture from YAML and load the raw student exactly.
        model = self.get_model(cfg=self.args.model, weights=None, verbose=RANK in {-1, 0})
        result = model.load_state_dict(self.e70_state["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"Raw model strict-load failed: {result}")
        self.model = model

        ema_model = deepcopy(model).float().cpu().eval()
        result = ema_model.load_state_dict(self.e70_state["ema_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"EMA strict-load failed: {result}")

        # Use BaseTrainer's real resume state injection, but preserve raw and EMA separately.
        self.resume = True
        return {
            "epoch": 69,  # Ultralytics stores zero-based epoch; e70 completed -> 69.
            "best_fitness": 0.0,
            "model": model,
            "ema": ema_model,
            "updates": self.base_ema_updates,
            "optimizer": self.e70_state["optimizer_state_dict"],
            "scaler": self.e70_state["scaler_state_dict"],
            "train_args": dict(self.e70_state["train_args"]),
        }

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)
        if rank not in {-1, 0}:
            raise RuntimeError("Protocol B scheduled source loader currently supports one formal GPU only")
        dataset = self.build_dataset(dataset_path, mode="train", batch=batch_size)
        return build_protocol_b_source_loader(
            dataset,
            self.protocol_schedule,
            workers=self.args.workers,
            total_epoch=71,
        )

    def _setup_train(self):
        super()._setup_train()
        if self.start_epoch != 70:
            raise RuntimeError(f"Expected zero-based start_epoch=70, got {self.start_epoch}")
        self.scheduler.load_state_dict(self.e70_state["scheduler_state_dict"])
        if self.scheduler.last_epoch != 69:
            raise RuntimeError(f"Scheduler restore failed: last_epoch={self.scheduler.last_epoch}")
        restore_rng_state(self.e70_state)

        raw_diff = state_dict_max_abs_diff(self.model.state_dict(), self.e70_state["model_state_dict"])
        ema_diff = state_dict_max_abs_diff(self.ema.ema.state_dict(), self.e70_state["ema_state_dict"])
        if raw_diff != 0.0 or ema_diff != 0.0:
            raise RuntimeError(f"Restore tensor mismatch: raw={raw_diff}, ema={ema_diff}")
        if self.ema.updates != self.base_ema_updates:
            raise RuntimeError(f"EMA updates mismatch: {self.ema.updates} != {self.base_ema_updates}")

        audit = {
            "timestamp": datetime.now().isoformat(),
            "raw_max_abs_diff": raw_diff,
            "ema_max_abs_diff": ema_diff,
            "raw_key_count": len(self.e70_state["model_state_dict"]),
            "ema_key_count": len(self.e70_state["ema_state_dict"]),
            "ema_updates_stored_in_e70": False,
            "ema_updates_inference_method": "Ultralytics optimizer-step cadence replay",
            "ema_updates_inferred": self.base_ema_updates,
            "global_batches_at_e70": int(self.e70_state["global_step"]),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "optimizer_lr": [float(group["lr"]) for group in self.optimizer.param_groups],
            "scaler_state": self.scaler.state_dict(),
            "B1_SOURCE_ONLY": True,
            "B1_TEACHER_PATH_ENABLED": False,
            "B1_TARGET_ADAPT_TRAIN_READ_COUNT": 0,
            "B1_PSEUDO_LABEL_ENABLED": False,
            "B1_UNSUPERVISED_LOSS_ENABLED": False,
        }
        audit_dir = Path(self.save_dir) / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "b1_restore_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    def _protocol_on_epoch_start(self, trainer: "ProtocolBB1Trainer") -> None:
        total_epoch = int(trainer.epoch) + 1
        if total_epoch not in range(71, 101):
            raise RuntimeError(f"B1 attempted unexpected total epoch {total_epoch}")
        trainer.protocol_current_total_epoch = total_epoch
        trainer.train_loader.batch_sampler.set_total_epoch(total_epoch)
        trainer.protocol_batch_index = 0
        trainer.protocol_epoch_image_visits = 0
        trainer.protocol_epoch_mismatches = 0
        trainer.protocol_epoch_first_batch_hash = None
        trainer.protocol_epoch_last_batch_hash = None


    def _protocol_on_batch_start(self, trainer: "ProtocolBB1Trainer") -> None:
        # BaseTrainer resets its local last_opt_step to -1 on resume. The e70 checkpoint ended
        # immediately after an optimizer step (global batch 38639), so e71 must accumulate
        # two fresh source batches before the first continuation step. Holding accumulate above
        # the absolute ni only for the first batch reproduces that boundary without rewriting
        # the upstream training loop.
        if (
            trainer.protocol_current_total_epoch == 71
            and trainer.protocol_batch_index == 0
            and trainer.protocol_optimizer_steps == 0
        ):
            trainer.accumulate = int(self.e70_state["global_step"]) + 2
        else:
            trainer.accumulate = max(round(trainer.args.nbs / trainer.batch_size), 1)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        uids = list(batch.pop("protocol_b_uid"))
        aug_seeds = [int(v) for v in batch.pop("protocol_b_aug_seed")]
        total_epochs = [int(v) for v in batch.pop("protocol_b_total_epoch")]
        batch_indices = [int(v) for v in batch.pop("protocol_b_batch_index")]
        positions = [int(v) for v in batch.pop("protocol_b_position")]

        expected_row = self.protocol_schedule.rows_for_epoch(self.protocol_current_total_epoch)[self.protocol_batch_index]
        expected_uids = list(expected_row["uids"])
        expected_seeds = list(map(int, expected_row["sample_aug_seeds"]))
        actual_file_uids = [Path(path).stem for path in batch["im_file"]]
        mismatch = not (
            uids == expected_uids
            and actual_file_uids == expected_uids
            and aug_seeds == expected_seeds
            and set(total_epochs) == {self.protocol_current_total_epoch}
            and set(batch_indices) == {self.protocol_batch_index}
            and positions == list(range(len(uids)))
        )
        if mismatch:
            self.protocol_epoch_mismatches += 1
            raise RuntimeError(
                f"Protocol B source schedule mismatch at e{self.protocol_current_total_epoch} "
                f"batch {self.protocol_batch_index}"
            )

        img_hash = _tensor_sha256(batch["img"])
        if self.protocol_batch_index == 0:
            self.protocol_epoch_first_batch_hash = img_hash
        self.protocol_epoch_last_batch_hash = img_hash
        self.protocol_epoch_image_visits += len(uids)
        self.protocol_batch_index += 1
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.protocol_last_grad_norm = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.protocol_optimizer_steps += 1

    def _protocol_on_epoch_end(self, trainer: "ProtocolBB1Trainer") -> None:
        if trainer.protocol_batch_index != EXPECTED_STEPS_PER_EPOCH:
            raise RuntimeError(
                f"Expected {EXPECTED_STEPS_PER_EPOCH} batches, observed {trainer.protocol_batch_index}"
            )
        if trainer.protocol_epoch_image_visits != 17_637:
            raise RuntimeError(f"Expected 17637 source image visits, observed {trainer.protocol_epoch_image_visits}")
        if trainer.protocol_epoch_mismatches != 0:
            raise RuntimeError(f"Source schedule mismatches: {trainer.protocol_epoch_mismatches}")

        loss_values = [float(v) for v in trainer.tloss.detach().cpu().flatten()] if trainer.tloss is not None else []
        while len(loss_values) < 3:
            loss_values.append(float("nan"))
        row = {
            "epoch": trainer.protocol_current_total_epoch,
            "global_step": int(self.e70_state["global_step"]) + (trainer.protocol_current_total_epoch - 70) * 552,
            "lr_pg0": float(trainer.optimizer.param_groups[0]["lr"]),
            "box_loss": loss_values[0],
            "cls_loss": loss_values[1],
            "dfl_loss": loss_values[2],
            "grad_norm_last_step": trainer.protocol_last_grad_norm,
            "optimizer_steps_this_run": trainer.protocol_optimizer_steps,
            "ema_updates": int(trainer.ema.updates),
            "amp_scale": float(trainer.scaler.get_scale()),
            "source_batches": trainer.protocol_batch_index,
            "source_image_visits": trainer.protocol_epoch_image_visits,
            "source_schedule_mismatch_count": trainer.protocol_epoch_mismatches,
            "first_batch_image_hash": trainer.protocol_epoch_first_batch_hash,
            "last_batch_image_hash": trainer.protocol_epoch_last_batch_hash,
            "target_adapt_train_read_count": 0,
            "target_dev_train_read_count": 0,
            "target_test_content_read_count": 0,
        }
        trainer.protocol_epoch_rows.append(row)
        trainer._write_epoch_metrics()

    def _write_epoch_metrics(self) -> None:
        path = Path(self.save_dir) / "train_epoch_metrics.csv"
        if not self.protocol_epoch_rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.protocol_epoch_rows[0]))
            writer.writeheader()
            writer.writerows(self.protocol_epoch_rows)

    def save_model(self):
        total_epoch = self.protocol_current_total_epoch
        global_step = int(self.e70_state["global_step"]) + (total_epoch - 70) * EXPECTED_STEPS_PER_EPOCH
        epoch_audit = self.protocol_epoch_rows[-1] if self.protocol_epoch_rows else {}
        payload = build_full_state_payload(
            self,
            total_epoch=total_epoch,
            global_step=global_step,
            base_checkpoint_sha256=self.base_checkpoint_sha256,
            dataset_manifest=EXPECTED_DATASET_MANIFEST,
            schedule_sha256=self.schedule_sha256,
            augmentation_sha256=self.augmentation_sha256,
            code_snapshot_sha256=self.code_snapshot_sha256,
            environment_sha256=self.environment_sha256,
            optimizer_step_count=self.base_ema_updates + self.protocol_optimizer_steps,
            epoch_audit=epoch_audit,
        )
        full_state_dir = Path(self.save_dir) / "full_state"
        last_path = rotate_last_checkpoint(payload, full_state_dir)
        last_hash = sha256_file(last_path)
        registered_path = None
        if total_epoch in REGISTERED_EPOCHS:
            registered_path = full_state_dir / f"epoch_{total_epoch:03d}_full_state.pt"
            if not registered_path.exists():
                # Copying byte-for-byte preserves the hash of last_full_state.pt.
                registered_path.write_bytes(last_path.read_bytes())
        row = {
            "epoch": total_epoch,
            "global_step": global_step,
            "last_path": str(last_path),
            "last_sha256": last_hash,
            "registered_path": str(registered_path) if registered_path else "",
            "registered_sha256": sha256_file(registered_path) if registered_path else "",
            "ema_updates": int(self.ema.updates),
            "optimizer_step_count": self.base_ema_updates + self.protocol_optimizer_steps,
        }
        self.protocol_checkpoint_rows.append(row)
        manifest = Path(self.save_dir) / "checkpoint_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.protocol_checkpoint_rows)
        return False  # suppress native on_model_save callbacks; custom full-state is already durable.

    def validate(self):
        # Evaluation is intentionally isolated into a separate process/script to avoid perturbing training RNG.
        return {}, 0.0

    def final_eval(self):
        return None

    def _protocol_on_train_end(self, trainer: "ProtocolBB1Trainer") -> None:
        final = {
            "experiment": "B1_SOURCE_ONLY",
            "completed": trainer.protocol_current_total_epoch == 100,
            "final_total_epoch": trainer.protocol_current_total_epoch,
            "final_global_step": int(self.e70_state["global_step"]) + 30 * EXPECTED_STEPS_PER_EPOCH,
            "optimizer_steps_this_run": trainer.protocol_optimizer_steps,
            "ema_updates_final": int(trainer.ema.updates),
            "source_schedule_mismatch_count": sum(int(row["source_schedule_mismatch_count"]) for row in trainer.protocol_epoch_rows),
            "target_adapt_train_read_count": 0,
            "target_dev_train_read_count": 0,
            "target_test_content_read_count": 0,
            "dataset_mutation_count": 0,
            "timestamp": datetime.now().isoformat(),
        }
        (Path(self.save_dir) / "B1_final_audit.json").write_text(
            json.dumps(final, indent=2, default=str), encoding="utf-8"
        )
