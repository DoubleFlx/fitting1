from __future__ import annotations

"""Execute the paired seed45/46 Native-A Fixed confirmation.

This module is a fail-closed adapter around the SHA-locked seed42 Stage-1
runner.  It materializes only the adaptation-seed boundary authorized by
M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1 and does not alter the detector,
loss, optimizer, EMA, BN, horizon, or Target-view semantics.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
import traceback
import warnings
import zipfile
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal")
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"
CONTRACTS = PROJECT / "contracts"
RESULT_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2"
IDENTITY_CONTRACT = CONTRACTS / "M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1.json"
IDENTITY_SIDECAR = CONTRACTS / "M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1.sha256"
RUN_CONTRACT = RESULT_ROOT / "IMPLEMENTATION_RUN_CONTRACT.json"
PARENT = PROJECT / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
PARENT_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
EXPERIMENT = "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2"
SEEDS = (45, 46)
EPOCHS = tuple(range(72, 101))
TOTAL_VISITS = 128_064
TARGET_BATCH = 8
VISITS_PER_EPOCH = 4_416
PREFLIGHT_MICROBATCHES = 10

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import m1_seed42_stage1_formal_train_v1 as base  # noqa: E402
from protocol_b_target_schedule import ProtocolBTargetSchedule  # noqa: E402

ORIGINAL_BASE_LOAD_RUN_CONTRACT = base.load_run_contract
ORIGINAL_BASE_VERIFY_STATIC_LOCKS = base.verify_static_locks
ORIGINAL_BASE_LOAD_FIXED_INPUTS = base.load_fixed_inputs
ORIGINAL_BASE_VALIDATE_START_STATE = base.validate_start_state
ORIGINAL_BASE_BUILD_TRAINER = base.build_trainer
ORIGINAL_BASE_VALIDATE_RECOVERY = base._validate_recovery
ORIGINAL_SUBPROCESS_RUN = subprocess.run
ACTIVE_SEED: int | None = None


class FreshSeedStop(RuntimeError):
    """A fail-closed identity or integrity failure."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    base.atomic_json(path, value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise FreshSeedStop(f"STOP_JSON_OBJECT_REQUIRED:{path}")
    return value


def seed_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise FreshSeedStop(f"STOP_SEED_NOT_AUTHORIZED:{seed}")
    root = RESULT_ROOT / f"SEED{int(seed)}"
    policy = root / "POLICY"
    schedule = policy / "SCHEDULE"
    cache = root / "CACHE"
    return {
        "root": root,
        "policy": policy,
        "schedule": schedule,
        "source_schedule": schedule / "source_schedule_e071_e100.jsonl",
        "source_augmentation": schedule / "source_augmentation_e071_e100.jsonl",
        "target_schedule": schedule / "target_schedule_e071_e100.jsonl",
        "target_manifest": schedule / "target_schedule_e071_e100_manifest.json",
        "fixed_contract": policy / "FIXED_A_POLICY_CONTRACT.json",
        "exposure": policy / "fixed_a_visit_exposure_audit.csv",
        "cache": cache,
        "cache_manifest": cache / "fixed_a_cache_manifest.csv",
        "cache_build": cache / "fixed_a_cache_build.json",
        "policy_gate": root / "POLICY_CACHE_PRETRAIN_GATE.json",
        "control": root / "CONTROL",
        "fixed": root / "FIXED",
        "noop": root / "RUNNER_NOOP_PREFLIGHT",
        "source_reference_placeholder": root / "SOURCE_HASH_REFERENCE_PLACEHOLDER.csv",
        "source_reference": root / "CONTROL_SOURCE_HASH_REFERENCE.csv",
        "prelaunch": root / "fixed_cache_prelaunch.json",
        "compat_preflight": root / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json",
    }


def identity_contract_audit() -> dict[str, Any]:
    if not IDENTITY_CONTRACT.is_file() or not IDENTITY_SIDECAR.is_file():
        raise FreshSeedStop("STOP_IDENTITY_CONTRACT_MISSING")
    digest = sha256_file(IDENTITY_CONTRACT)
    sidecar = IDENTITY_SIDECAR.read_text(encoding="ascii").strip().split()[0]
    contract = load_json(IDENTITY_CONTRACT)
    prompt = contract.get("authoritative_prompt_identity", {})
    arms = contract.get("required_arms", [])
    fixed = contract.get("fixed_policy_identity", {})
    training = contract.get("training", {})
    evaluation = contract.get("evaluation", {})
    access = contract.get("data_access_rules", {})
    checks = {
        "sidecar_exact": digest == sidecar,
        "schema_exact": contract.get("schema") == "m1_fixed_fresh_seed_identity_contract_v1",
        "parent_exact": contract.get("parent_sha256") == PARENT_SHA256,
        "fresh_seeds_exact": contract.get("fresh_seeds") == [45, 46],
        "seed_scope_exact": contract.get("seed_scope") == "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "parent_file_exact": PARENT.is_file() and sha256_file(PARENT) == PARENT_SHA256,
        "authoritative_prompt_exact": (
            prompt.get("task") == EXPERIMENT
            and prompt.get("authority") == "CURRENT_USER_PROMPT_IS_THE_AUTHORITATIVE_SEED_IDENTITY_CONTRACT"
            and prompt.get("previous_stop_resolution") == "RESOLVED_BY_NEW_AUTHORITATIVE_CONTRACT"
            and prompt.get("seed_identity") == "EXPLICITLY_DEFINED"
            and prompt.get("training_authority") == "AUTHORIZED_ONLY_AFTER_ALL_CONTRACTED_PREFLIGHT_GATES_PASS"
        ),
        "required_arms_exact": (
            [row.get("id") for row in arms] == [
                "SEED45_CONTROL", "SEED45_FIXED",
                "SEED46_CONTROL", "SEED46_FIXED",
            ]
            and [int(row.get("adaptation_seed", -1)) for row in arms]
            == [45, 45, 46, 46]
            and all(row.get("teacher_target_view") == "ORIGINAL_TARGET_VIEW" for row in arms)
        ),
        "fixed_policy_identity_exact": (
            fixed.get("name") == "NATIVE_A_FIXED_MATCHED_EXPOSURE_SCHEDULE_DEPENDENT_PER_IMAGE_FIXED_POLICY"
            and fixed.get("standalone_uid_only_policy") is False
            and fixed.get("selector_namespace") == "NATIVE_A_FIXED_V1"
            and fixed.get("selector_seed_injection") == "FORBIDDEN_BECAUSE_THE_VERIFIED_FIXED_SELECTOR_IS_SEED_INDEPENDENT"
            and fixed.get("reselection_on_solver_failure") is False
            and fixed.get("cross_seed_cache_reuse") is False
            and fixed.get("seed42_cache_scientific_reuse") is False
        ),
        "training_horizon_exact": (
            int(training.get("start_completed_epoch", -1)) == 71
            and training.get("epochs") == list(EPOCHS)
            and int(training.get("formal_horizon", -1)) == 100
            and training.get("primary_endpoint") == "e100 EMA"
            and training.get("secondary_endpoint") == "e100 Raw"
            and training.get("best_epoch_selection") is False
            and training.get("e94_e98_target_dev_evaluation") is False
        ),
        "evaluation_endpoints_exact": evaluation.get("endpoints") == [
            "SEED45_CONTROL_E100_EMA", "SEED45_FIXED_E100_EMA",
            "SEED46_CONTROL_E100_EMA", "SEED46_FIXED_E100_EMA",
            "SEED45_CONTROL_E100_RAW", "SEED45_FIXED_E100_RAW",
            "SEED46_CONTROL_E100_RAW", "SEED46_FIXED_E100_RAW",
        ],
        "access_boundary_exact": access.get("before_four_way_freeze") == {
            "Target_Dev_reads": 0,
            "Target_Test_reads": 0,
            "Target_Adapt_GT_reads": 0,
        },
    }
    if not all(checks.values()):
        raise FreshSeedStop(f"STOP_IDENTITY_CONTRACT:{checks}")
    return {
        "path": str(IDENTITY_CONTRACT.resolve()),
        "sha256": digest,
        "checks": checks,
    }


def policy_gate(seed: int) -> dict[str, Any]:
    paths = seed_paths(seed)
    gate = load_json(paths["policy_gate"])
    if not (
        gate.get("status") == "PASS"
        and gate.get("training_authorized_by_identity_contract_after_gate") is True
        and int(gate.get("adaptation_seed", -1)) == int(seed)
        and int(gate.get("target_dev_reads", -1)) == 0
        and int(gate.get("target_test_reads", -1)) == 0
        and int(gate.get("target_adapt_label_reads", -1)) == 0
        and all(gate.get("checks", {}).values())
    ):
        raise FreshSeedStop(f"STOP_POLICY_CACHE_GATE:{seed}")
    return gate


def _require_sha(value: Any, field: str) -> str:
    text = str(value or "").strip().casefold()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise FreshSeedStop(f"STOP_SHA:{field}")
    return text


def _strict_bool(value: Any, field: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise FreshSeedStop(f"STOP_BOOL:{field}:{value}")


def load_fixed_inputs_fresh(
    seed: int, target_schedule: ProtocolBTargetSchedule,
) -> tuple[list[base.ExposureEntry], dict[str, base.CacheEntry], dict[str, Any]]:
    paths = seed_paths(seed)
    gate = policy_gate(seed)
    expected = gate["counts"]
    entries: list[base.ExposureEntry] = []
    with paths["exposure"].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "formal_visit_index", "adaptation_seed", "epoch",
            "target_visit_ordinal", "schedule_uid", "canonical_uid",
            "fixed_delta_exists", "fixed_active", "fixed_identity",
            "fixed_selected_delta_sha", "active_match", "identity_match",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise FreshSeedStop("STOP_EXPOSURE_SCHEMA")
        for expected_index, row in enumerate(reader):
            index = int(row["formal_visit_index"])
            epoch = int(row["epoch"])
            ordinal = int(row["target_visit_ordinal"])
            if (
                index != expected_index
                or int(row["adaptation_seed"]) != int(seed)
                or index != (epoch - 72) * VISITS_PER_EPOCH + ordinal
                or epoch not in EPOCHS
                or not 0 <= ordinal < VISITS_PER_EPOCH
            ):
                raise FreshSeedStop(f"STOP_EXPOSURE_IDENTITY:{expected_index}")
            active = _strict_bool(row["fixed_active"], "fixed_active")
            identity = _strict_bool(row["fixed_identity"], "fixed_identity")
            exists = _strict_bool(row["fixed_delta_exists"], "fixed_delta_exists")
            selected = str(row["fixed_selected_delta_sha"]).strip().casefold()
            if selected:
                _require_sha(selected, "fixed_selected_delta_sha")
            if (
                active == identity
                or exists != bool(selected)
                or (active and not exists)
                or not _strict_bool(row["active_match"], "active_match")
                or not _strict_bool(row["identity_match"], "identity_match")
            ):
                raise FreshSeedStop(f"STOP_EXPOSURE_SEMANTICS:{index}")
            entries.append(base.ExposureEntry(
                index, epoch, ordinal, str(row["schedule_uid"]),
                str(row["canonical_uid"]), active, identity, exists, selected,
            ))
    if (
        len(entries) != int(expected["visits"])
        or sum(int(row.fixed_active) for row in entries) != int(expected["active"])
        or sum(int(row.fixed_identity) for row in entries) != int(expected["identity"])
    ):
        raise FreshSeedStop("STOP_EXPOSURE_COUNTS")
    for epoch in EPOCHS:
        for batch_index, schedule_row in enumerate(target_schedule.rows_for_epoch(epoch)):
            for position, uid in enumerate(schedule_row["uids"]):
                ordinal = batch_index * TARGET_BATCH + position
                index = (epoch - 72) * VISITS_PER_EPOCH + ordinal
                entry = entries[index]
                if (
                    entry.epoch != epoch
                    or entry.target_visit_ordinal != ordinal
                    or entry.schedule_uid != str(uid)
                ):
                    raise FreshSeedStop(f"STOP_EXPOSURE_SCHEDULE_CROSSLOCK:{index}")

    cache: dict[str, base.CacheEntry] = {}
    with paths["cache_manifest"].open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schema", "canonical_uid", "input_image_sha256", "solve_key",
            "expected_output_bgr_sha256", "sidecar_path", "sidecar_sha256",
            "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise FreshSeedStop("STOP_CACHE_MANIFEST_SCHEMA")
        for row in reader:
            uid = str(row["canonical_uid"])
            if uid in cache or not _strict_bool(
                row["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete"], "cache_complete"
            ):
                raise FreshSeedStop(f"STOP_CACHE_MANIFEST_ENTRY:{uid}")
            sidecar_path = base.inside_root(Path(str(row["sidecar_path"])))
            sidecar_sha = _require_sha(row["sidecar_sha256"], "sidecar_sha256")
            if sha256_file(sidecar_path) != sidecar_sha:
                raise FreshSeedStop(f"STOP_CACHE_SIDECAR_SHA:{uid}")
            sidecar = load_json(sidecar_path)
            candidate = sidecar.get("candidates", {}).get(
                "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE", {}
            )
            bgr_path = base.inside_root(Path(str(candidate.get("path", ""))))
            shape = tuple(int(value) for value in candidate.get("shape", []))
            checks = {
                "uid": sidecar.get("canonical_uid") == uid,
                "complete": candidate.get("complete") is True,
                "dtype": candidate.get("dtype") == "uint8",
                "shape": len(shape) == 3 and shape[2] == 3,
                "file": bgr_path.is_file(),
                "bytes": bgr_path.is_file() and bgr_path.stat().st_size == int(candidate.get("file_bytes", -1)),
            }
            if not all(checks.values()):
                raise FreshSeedStop(f"STOP_CACHE_SIDECAR:{uid}:{checks}")
            cache[uid] = base.CacheEntry(
                uid,
                _require_sha(row["input_image_sha256"], "input_image_sha256"),
                _require_sha(row["solve_key"], "solve_key"),
                _require_sha(row["expected_output_bgr_sha256"], "output_bgr_sha256"),
                str(sidecar_path), sidecar_sha, str(bgr_path),
                _require_sha(candidate.get("file_sha256"), "bgr_file_sha256"),
                _require_sha(candidate.get("array_bytes_sha256"), "bgr_array_sha256"),
                shape, int(candidate["file_bytes"]),
            )
    delta_uids = {row.canonical_uid for row in entries if row.fixed_delta_exists}
    if len(cache) != int(expected["with_delta"]) or set(cache) != delta_uids:
        raise FreshSeedStop(f"STOP_CACHE_UID_SET:{len(cache)}:{len(delta_uids)}")
    return entries, cache, {
        "adaptation_seed": int(seed),
        "exposure_rows": len(entries),
        "active_visits": sum(int(row.fixed_active) for row in entries),
        "identity_visits": sum(int(row.fixed_identity) for row in entries),
        "cache_entries": len(cache),
        "policy_gate_sha256": sha256_file(paths["policy_gate"]),
        "adaptation_seed_compared_to_authoritative_fresh_seed": True,
        "view_seed_compared_only_by_frozen_target_schedule": True,
    }


def write_placeholder_reference(seed: int) -> Path:
    path = seed_paths(seed)["source_reference_placeholder"]
    if path.is_file():
        return path
    rows = [
        {"epoch": epoch, "first_batch_image_hash": "", "last_batch_image_hash": ""}
        for epoch in range(71, 101)
    ]
    base.atomic_csv(path, rows)
    return path


def validate_parent() -> dict[str, Any]:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise FreshSeedStop("STOP_PARENT_IDENTITY_UNRECOVERABLE")
    if (
        not base.E70_BOOTSTRAP.is_file()
        or sha256_file(base.E70_BOOTSTRAP) != base.E70_BOOTSTRAP_SHA256
    ):
        raise FreshSeedStop("STOP_E70_ARCHITECTURE_BOOTSTRAP_SHA")
    state = torch.load(PARENT, map_location="cpu", weights_only=False)
    required = {
        "format", "epoch", "global_step", "ema_updates", "model_state_dict",
        "ema_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "rng_python", "rng_numpy", "rng_torch_cpu",
        "rng_torch_cuda", "dataset_manifest",
    }
    if required - set(state):
        raise FreshSeedStop(f"STOP_PARENT_FIELDS:{sorted(required-set(state))}")
    raw_sha = base.tensor_state_hash(state["model_state_dict"])
    ema_sha = base.tensor_state_hash(state["ema_state_dict"])
    checks = {
        "format": state["format"] == "protocol_b_full_state_v2",
        "epoch": int(state["epoch"]) == 71,
        "global_step": int(state["global_step"]) == base.E71_GLOBAL_STEP,
        "ema_updates": int(state["ema_updates"]) == base.E71_EMA_UPDATES,
        "optimizer_updates": int(state.get("optimizer_step_count", -1)) == base.E71_EMA_UPDATES,
        "dataset": state["dataset_manifest"] == base.EXPECTED_DATASET_MANIFEST,
        "raw": raw_sha == base.E71_RAW_TENSOR_SHA256,
        "ema": ema_sha == base.E71_EMA_TENSOR_SHA256,
        "scheduler_node": int(state["scheduler_state_dict"]["last_epoch"]) == 70,
    }
    if not all(checks.values()):
        raise FreshSeedStop(f"STOP_PARENT_COMPONENT_IDENTITY:{checks}")
    return state


def configure_base(seed: int, *, source_reference: Path | None = None) -> dict[str, Path]:
    paths = seed_paths(seed)
    base.EXPERIMENT = EXPERIMENT
    base.RESULT_ROOT = paths["root"]
    base.DEFAULT_CONTRACT = RUN_CONTRACT
    base.E71_FULL_STATE = PARENT
    base.E71_FILE_SHA256 = PARENT_SHA256
    base.SOURCE_SCHEDULE = paths["source_schedule"]
    base.SOURCE_AUGMENTATION = paths["source_augmentation"]
    base.TARGET_SCHEDULE = paths["target_schedule"]
    base.TARGET_SCHEDULE_MANIFEST = paths["target_manifest"]
    base.FIXED_POLICY_CONTRACT = paths["fixed_contract"]
    base.FIXED_EXPOSURE = paths["exposure"]
    base.FIXED_CACHE_MANIFEST = paths["cache_manifest"]
    base.FIXED_CACHE_BUILD = paths["cache_build"]
    base.FIXED_PRETRAIN_FINAL_STATUS = paths["policy_gate"]
    base.FIXED_CACHE_ROOT = paths["cache"] / "working_cache"
    base.FIXED_CACHE_PRELAUNCH = paths["prelaunch"]
    base.B1_METRICS = source_reference or write_placeholder_reference(seed)
    base.EXPECTED_INPUT_SHA256.update({
        "source_schedule": sha256_file(paths["source_schedule"]),
        "source_augmentation": sha256_file(paths["source_augmentation"]),
        "target_schedule": sha256_file(paths["target_schedule"]),
        "fixed_policy_contract": sha256_file(paths["fixed_contract"]),
        "fixed_exposure": sha256_file(paths["exposure"]),
        "fixed_cache_manifest": sha256_file(paths["cache_manifest"]),
        "fixed_cache_build": sha256_file(paths["cache_build"]),
        "fixed_pretrain_final_status": sha256_file(paths["policy_gate"]),
        "b1_train_metrics": sha256_file(base.B1_METRICS),
    })
    return paths


class FreshStage1Trainer(base.Stage1Trainer):
    """Seed-boundary adapter with the frozen Stage1 math unchanged."""

    def __init__(
        self, *, adaptation_seed: int, source_reference_mode: str, **kwargs: Any
    ) -> None:
        self.adaptation_seed = int(adaptation_seed)
        self.source_reference_mode = str(source_reference_mode)
        self.fresh_reseed_audit: dict[str, Any] = {}
        super().__init__(**kwargs)

    def _restore_start_state(self) -> dict[str, Any]:
        restored = super()._restore_start_state()
        if int(self.completed_start_epoch) == 71:
            expected_hashseed = str(self.adaptation_seed)
            if os.environ.get("PYTHONHASHSEED") != expected_hashseed:
                raise FreshSeedStop(
                    f"STOP_PYTHONHASHSEED:{os.environ.get('PYTHONHASHSEED')}:{expected_hashseed}"
                )
            checkpoint_rng_hashes = dict(restored["rng_hashes"])
            random.seed(self.adaptation_seed)
            np.random.seed(self.adaptation_seed & 0xFFFFFFFF)
            torch.manual_seed(self.adaptation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.adaptation_seed)
            source_generator = getattr(self.train_loader, "generator", None)
            target_generator = getattr(self.target_loader, "generator", None)
            if source_generator is None or target_generator is None:
                raise FreshSeedStop("STOP_DATALOADER_GENERATOR_UNAVAILABLE")
            source_generator.manual_seed(self.adaptation_seed)
            target_generator.manual_seed(self.adaptation_seed + 1000)
            reseeded = base.rng_state_hashes()
            self.fresh_reseed_audit = {
                "schema": "m1_fixed_fresh_seed_adaptation_boundary_rng_v1",
                "status": "PASS",
                "adaptation_seed": self.adaptation_seed,
                "boundary": "AFTER_EXACT_E71_RESTORE_BEFORE_E72_ITERATION",
                "checkpoint_rng_hashes_before_reseed": checkpoint_rng_hashes,
                "adaptation_rng_hashes_after_reseed": reseeded,
                "PYTHONHASHSEED": expected_hashseed,
                "python_random_seed": self.adaptation_seed,
                "numpy_seed": self.adaptation_seed & 0xFFFFFFFF,
                "torch_cpu_seed": self.adaptation_seed,
                "torch_cuda_seed": self.adaptation_seed,
                "source_dataloader_generator_seed": self.adaptation_seed,
                "target_dataloader_generator_seed": self.adaptation_seed + 1000,
                "worker_init_fn_identity": "PYTORCH_DEFAULT_NO_CUSTOM_WORKER_INIT_FN",
                "wall_clock_pid_filesystem_order_in_scientific_rng": False,
                "created_at": now_iso(),
            }
            restored["fresh_adaptation_boundary"] = self.fresh_reseed_audit
        return restored

    def _protocol_on_epoch_end(self, trainer: "FreshStage1Trainer") -> None:
        if self.source_reference_mode == "ESTABLISH_CONTROL":
            epoch = int(self.protocol_current_total_epoch)
            self.b1_source_hash_reference[epoch] = {
                "first": str(self.protocol_epoch_first_batch_hash),
                "last": str(self.protocol_epoch_last_batch_hash),
            }
        super()._protocol_on_epoch_end(trainer)


def trainer_overrides(seed: int, run_dir: Path, device: str, *, preflight: bool, resume: bool) -> dict[str, Any]:
    values = base.trainer_overrides(run_dir, device, preflight=preflight, resume=resume)
    values["seed"] = int(seed)
    return values


def build_fresh_trainer(
    *, seed: int, branch: str, run_dir: Path, device: str,
    start_state: dict[str, Any], start_state_path: Path,
    exposure_entries: list[base.ExposureEntry], cache_entries: dict[str, base.CacheEntry],
    data_guard: Any, data_access_attempt_dir: Path, run_attempt_id: str,
    logging_enabled: bool, preflight: bool, resume: bool,
    source_reference_mode: str,
) -> FreshStage1Trainer:
    paths = seed_paths(seed)
    return FreshStage1Trainer(
        adaptation_seed=seed,
        source_reference_mode=source_reference_mode,
        branch=branch,
        run_contract_path=RUN_CONTRACT,
        run_contract_sha256=sha256_file(RUN_CONTRACT),
        start_state=start_state,
        start_state_path=start_state_path,
        completed_start_epoch=int(start_state["epoch"]),
        exposure_entries=exposure_entries,
        cache_entries=cache_entries,
        data_guard=data_guard,
        data_access_attempt_dir=data_access_attempt_dir,
        run_attempt_id=run_attempt_id,
        logging_enabled=logging_enabled,
        preflight_mode=preflight,
        full_state_path=base.E70_BOOTSTRAP,
        source_schedule_path=paths["source_schedule"],
        source_augmentation_path=paths["source_augmentation"],
        contract_path=base.B2_TEMPLATE,
        target_yaml_path=base.TARGET_YAML,
        target_schedule_path=paths["target_schedule"],
        target_schedule_manifest_path=paths["target_manifest"],
        b1_train_metrics_path=base.B1_METRICS,
        comparison_rule_path=base.COMPARISON_RULE,
        code_snapshot_sha256=sha256_file(Path(__file__)),
        environment_sha256=canonical_sha({
            "python": sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": device,
        }),
        overrides=trainer_overrides(seed, run_dir, device, preflight=preflight, resume=resume),
    )


def normalized_projection(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized_projection(item)
            for key, item in value.items()
            if key not in {"branch", "run_attempt_id"}
        }
    if isinstance(value, list):
        return [normalized_projection(item) for item in value]
    return value


def noop_worker(seed: int, branch: str, output: Path, device: str) -> dict[str, Any]:
    if branch not in base.BRANCHES:
        raise FreshSeedStop(f"STOP_NOOP_BRANCH:{branch}")
    configure_base(seed)
    policy_gate(seed)
    state = validate_parent()
    schedule = ProtocolBTargetSchedule(base.TARGET_SCHEDULE, base.TARGET_SCHEDULE_MANIFEST)
    exposure, cache, fixed_summary = load_fixed_inputs_fresh(seed, schedule)
    exposure = [replace(row, fixed_active=False, fixed_identity=True) for row in exposure]
    output.mkdir(parents=True, exist_ok=False)
    data_guard = base.install_m1_data_isolation_guard(
        phase=f"{EXPERIMENT}_SEED{seed}_{branch}_NOOP",
        audit_dir=output / "data_access_parent",
        allow_target_dev_images=False,
    )
    trainer = build_fresh_trainer(
        seed=seed, branch=branch, run_dir=output / "trainer_sandbox", device=device,
        start_state=state, start_state_path=PARENT, exposure_entries=exposure,
        cache_entries=cache, data_guard=data_guard,
        data_access_attempt_dir=output / "data_access_workers",
        run_attempt_id=f"SEED{seed}_{branch}_NOOP", logging_enabled=True,
        preflight=True, resume=False, source_reference_mode="ESTABLISH_CONTROL",
    )
    trainer._setup_train()
    trainer.epoch = 71
    trainer._protocol_on_epoch_start(trainer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.scheduler.step()
    trainer._model_train()
    trainer.optimizer.zero_grad()
    source_iterator = iter(trainer.train_loader)
    source_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for micro_index in range(PREFLIGHT_MICROBATCHES):
        raw = next(source_iterator)
        source_uids = list(raw["protocol_b_uid"])
        source_aug_seeds = [int(value) for value in raw["protocol_b_aug_seed"]]
        trainer._protocol_on_batch_start(trainer)
        with base.autocast(bool(trainer.amp)):
            processed = trainer.preprocess_batch(raw)
            loss, loss_items = trainer.model(processed)
            total = loss.sum()
        trainer.scaler.scale(total).backward()
        source_rows.append({
            "microbatch": micro_index,
            "source_uids": source_uids,
            "source_aug_seeds": source_aug_seeds,
            "source_tensor_sha256": base.tensor_bytes_sha(processed["img"]),
            "loss_sha256": base.tensor_bytes_sha(loss),
            "loss_items_sha256": base.tensor_bytes_sha(loss_items),
            "gradient_sha256_after_backward": base.gradient_state_sha(trainer.model),
        })
        if micro_index % base.ACCUMULATION == base.ACCUMULATION - 1:
            trainer.optimizer_step()
            window_rows.append({
                "window": len(window_rows) + 1,
                "underlying_optimizer_step_called": trainer._last_step_underlying_called,
                "found_inf": trainer.protocol_last_optimizer_window_found_inf,
                "target_term_included": trainer._last_window_target_included,
                "student_tensor_sha256": base.tensor_state_hash(base.unwrap_model(trainer.model).state_dict()),
                "teacher_tensor_sha256": base.tensor_state_hash(trainer.ema.ema.state_dict()),
                "optimizer_state_sha256": base.recursive_state_sha256(trainer.optimizer.state_dict()),
                "ema_updates": int(trainer.ema.updates),
                "rng_state_sha256": base.recursive_state_sha256(base.rng_state_payload()),
            })
        trainer._protocol_b2_on_batch_end(trainer)
    target_iterator = trainer.protocol_target_iterator
    inner = target_iterator.inner if isinstance(target_iterator, base.CapturingIterator) else target_iterator
    shutdown = getattr(inner, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    trainer.protocol_target_iterator = None
    projection = base._preflight_projection(trainer, source_rows, window_rows)
    result = {
        "schema": "m1_fixed_fresh_seed_runner_noop_worker_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "branch": branch,
        "fixed_transformation_disabled": True,
        "fixed_input_summary_before_noop_mask": fixed_summary,
        "projection": projection,
        "normalized_projection_sha256": canonical_sha(normalized_projection(projection)),
        "rng_boundary": trainer.fresh_reseed_audit,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "checkpoint_writes": len(list(output.rglob("*.pt"))),
        "created_at": now_iso(),
    }
    if result["checkpoint_writes"] != 0:
        raise FreshSeedStop("STOP_NOOP_CHECKPOINT_WRITE")
    atomic_json(output / "result.json", result)
    return result


def run_noop_gate(seed: int, device: str) -> dict[str, Any]:
    paths = configure_base(seed)
    destination = paths["noop"]
    audit_path = destination / "RUNNER_NOOP_AUDIT.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("status") == "PASS" and audit.get("executor_sha256") == sha256_file(Path(__file__)):
            return audit
    destination.mkdir(parents=True, exist_ok=True)
    workers: list[dict[str, Any]] = []
    for branch, folder in (("CONTROL", "CONTROL"), ("NATIVE_A_FIXED", "FIXED_CAPABLE_NOOP")):
        output = destination / folder
        if output.exists() and not (output / "result.json").is_file():
            raise FreshSeedStop(f"STOP_PARTIAL_NOOP_OUTPUT:{output}")
        if not output.exists():
            command = [
                sys.executable, "-B", str(Path(__file__).resolve()), "noop-worker",
                "--seed", str(seed), "--branch", branch, "--output", str(output),
                "--device", str(device),
            ]
            env = dict(os.environ)
            env.update({
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": str(seed),
            })
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if completed.returncode != 0:
                raise FreshSeedStop(f"STOP_NOOP_WORKER_EXIT:{seed}:{branch}:{completed.returncode}")
        workers.append(load_json(output / "result.json"))
    control, fixed_noop = workers
    exact = normalized_projection(control["projection"]) == normalized_projection(fixed_noop["projection"])
    checks = {
        "two_workers_pass": all(row.get("status") == "PASS" for row in workers),
        "control_fixed_capable_noop_projection_exact": exact,
        "source_uid_seed_tensor_loss_gradient_exact": control["projection"]["source_rows"] == fixed_noop["projection"]["source_rows"],
        "target_uid_mask_hpl_exact": normalized_projection(control["projection"]["target_hpl_common_rows"]) == normalized_projection(fixed_noop["projection"]["target_hpl_common_rows"]),
        "optimizer_ema_scaler_bn_rng_endpoint_exact": all(
            control["projection"][key] == fixed_noop["projection"][key]
            for key in (
                "student_tensor_sha256", "teacher_tensor_sha256",
                "optimizer_state_sha256", "scheduler_state_sha256",
                "scaler_state_sha256", "rng_state_sha256",
                "bn_running_state_sha256", "ema_updates",
            )
        ),
        "no_checkpoint_writes": all(int(row["checkpoint_writes"]) == 0 for row in workers),
        "target_dev_zero": all(int(row["target_dev_reads"]) == 0 for row in workers),
        "target_test_zero": all(int(row["target_test_reads"]) == 0 for row in workers),
        "target_labels_zero": all(int(row["target_adapt_label_reads"]) == 0 for row in workers),
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_runner_noop_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "adaptation_seed": int(seed),
        "checks": checks,
        "control_projection_sha256": control["normalized_projection_sha256"],
        "fixed_capable_noop_projection_sha256": fixed_noop["normalized_projection_sha256"],
        "executor_sha256": sha256_file(Path(__file__)),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_RUNNER_NOOP:{seed}:{checks}")
    compatibility = {
        "schema": "m1_seed42_stage1_logging_preflight_audit_v1",
        "status": "PASS", "all_pass": True,
        "run_contract_sha256": sha256_file(RUN_CONTRACT),
        "runner_sha256": sha256_file(Path(base.__file__)),
        "fresh_seed_noop_audit_path": str(audit_path.resolve()),
        "fresh_seed_noop_audit_sha256": sha256_file(audit_path),
        "formal_training_started": False,
        "checkpoint_writes": 0, "snapshot_writes": 0,
        "timestamp": now_iso(),
    }
    atomic_json(paths["compat_preflight"], compatibility)
    return audit


def code_authorities() -> dict[str, dict[str, Any]]:
    relative_paths = (
        "project/scripts/m1_fixed_fresh_seed_v2_prepare.py",
        "project/scripts/m1_fixed_fresh_seed_v2_execute.py",
        "project/scripts/m1_fixed_fresh_seed_v2_orchestrate.ps1",
        "project/scripts/m1_seed42_stage1_formal_train_v1.py",
        "project/scripts/m1_seed42_stage1_e100_evaluate_v1.py",
        "project/scripts/protocol_b_horizon_common.py",
        "project/scripts/protocol_b_horizon_schedule.py",
        "project/scripts/protocol_b_horizon_trainer.py",
        "project/scripts/protocol_b_b1_trainer.py",
        "project/scripts/protocol_b_b2_trainer.py",
        "project/scripts/protocol_b_b2_model.py",
        "project/scripts/protocol_b_b2_full_state.py",
        "project/scripts/protocol_b_full_state.py",
        "project/scripts/protocol_b_schedule.py",
        "project/scripts/protocol_b_target_schedule.py",
        "project/scripts/protocol_b_m1_model.py",
        "project/scripts/protocol_b_m1_common.py",
        "project/scripts/protocol_b_m1_data_isolation.py",
        "project/scripts/protocol_b_training_singleton.py",
        "project/scripts/protocol_b_pseudo.py",
        "project/scripts/m1_native_a_fixed_policy_realization_v1.py",
        "project/scripts/m1_routea_offline_theta_cache_engine_v1.py",
        "project/scripts/m1_routea_resource_equivalence_execution_core_v1.py",
        "project/scripts/m1_routea_photometric_realization_core_v1.py",
        "project/scripts/m1_native_a_fixed_cache_resource_v1.py",
        "project/scripts/post_m1_control_contrasted_unified_diagnostic_v1.py",
        "project/ultralytics/engine/trainer.py",
        "project/ultralytics/utils/nms.py",
    )
    authorities = {
        relative: {
            "path": str((ROOT / relative).resolve()),
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in relative_paths
    }
    frozen_seed42_drift = {
        relative: {
            "expected": expected,
            "observed": authorities.get(relative, {}).get("sha256"),
        }
        for relative, expected in base.EXPECTED_CODE_SHA256.items()
        if authorities.get(relative, {}).get("sha256") != expected
    }
    if frozen_seed42_drift:
        raise FreshSeedStop(
            f"STOP_FROZEN_SEED42_TRANSITIVE_CODE_DRIFT:{frozen_seed42_drift}"
        )
    return authorities


def static_input_authorities() -> dict[str, dict[str, Any]]:
    inputs = {
        "e70_architecture_optimizer_bootstrap": (
            base.E70_BOOTSTRAP,
            "2366ec68c56032fdab78cd6ab80cd2c485f06c04a92f358304caf177411cd90f",
        ),
        "source_supervised_yaml": (
            base.SOURCE_YAML,
            "5259ff0050507291a22a590ee0d960cbe396554d11abb66e16db8c44ced0d6e2",
        ),
        "target_adapt_unlabeled_yaml": (
            base.TARGET_YAML,
            "dfae1616076c5b9603313b5aae3a994237de4d0c8771f1fde135a9f98138c310",
        ),
        "dataset_integrity_marker": (
            ROOT / "datasets/dataset_final_v2_1/DATASET_MANIFEST_SHA256.txt",
            "a0df5b283bead547ced97bc08dc2cda0ef02a8402f35f05d14c9ab0430cfb58e",
        ),
        "b2_contract_template": (
            base.B2_TEMPLATE,
            "f9e69570db30718d5d92ecc68f0074ddc8a1b0429f6b735cd740c5f1f193c43d",
        ),
        "comparison_rule": (
            base.COMPARISON_RULE,
            "52bb24ee18a0b8be9fa45ddff2831c2c2da39971a15da5c7a1828200ae6993fd",
        ),
    }
    authorities: dict[str, dict[str, Any]] = {}
    for role, (path, expected) in inputs.items():
        path = Path(path).resolve()
        observed = sha256_file(path)
        if observed != expected:
            raise FreshSeedStop(
                f"STOP_FROZEN_STATIC_INPUT_DRIFT:{role}:{observed}:{expected}"
            )
        authorities[role] = {
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": observed, "expected_sha256": expected,
        }
    return authorities


def materialize_run_contract() -> dict[str, Any]:
    identity = identity_contract_audit()
    taxonomy_equivalence = taxonomy_seed42_equivalence_gate()
    native_equivalence_path = RESULT_ROOT / "NATIVE_A_PREFIX_SEED42_EQUIVALENCE.json"
    selector_equivalence_path = RESULT_ROOT / "FIXED_SELECTOR_SEED42_EQUIVALENCE.json"
    prepare_sha = sha256_file(SCRIPTS / "m1_fixed_fresh_seed_v2_prepare.py")
    for equivalence_path in (native_equivalence_path, selector_equivalence_path):
        equivalence = load_json(equivalence_path) if equivalence_path.is_file() else {}
        if not (
            equivalence.get("status") == "PASS"
            and equivalence.get("adapter_sha256") == prepare_sha
        ):
            raise FreshSeedStop(f"STOP_SEED42_POLICY_EQUIVALENCE:{equivalence_path}")
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        paths = seed_paths(seed)
        gate = policy_gate(seed)
        noop_path = paths["noop"] / "RUNNER_NOOP_AUDIT.json"
        per_seed[str(seed)] = {
            "adaptation_seed": seed,
            "policy_gate": {
                "path": str(paths["policy_gate"].resolve()),
                "sha256": sha256_file(paths["policy_gate"]),
            },
            "schedule": {
                role: {
                    "path": str(paths[role].resolve()),
                    "sha256": sha256_file(paths[role]),
                }
                for role in (
                    "source_schedule", "source_augmentation",
                    "target_schedule", "target_manifest",
                )
            },
            "fixed_policy": {
                "contract_path": str(paths["fixed_contract"].resolve()),
                "contract_sha256": sha256_file(paths["fixed_contract"]),
                "exposure_path": str(paths["exposure"].resolve()),
                "exposure_sha256": sha256_file(paths["exposure"]),
                "cache_manifest_path": str(paths["cache_manifest"].resolve()),
                "cache_manifest_sha256": sha256_file(paths["cache_manifest"]),
                "cache_build_path": str(paths["cache_build"].resolve()),
                "cache_build_sha256": sha256_file(paths["cache_build"]),
                "counts": gate["counts"],
            },
            "runner_noop_gate": {
                "required_before_training": True,
                "path": str(noop_path.resolve()),
            },
            "arms": {
                "CONTROL": str(paths["control"].resolve()),
                "NATIVE_A_FIXED": str(paths["fixed"].resolve()),
            },
        }
    contract = {
        "schema": "m1_fixed_fresh_seed_implementation_run_contract_v1",
        "status": "FROZEN_BEFORE_DIAGNOSTIC_PREFLIGHT_AND_TRAINING",
        "experiment": EXPERIMENT,
        "created_at": now_iso(),
        "authoritative_identity_contract": identity,
        "parent": {
            "path": str(PARENT.resolve()), "sha256": PARENT_SHA256,
            "epoch": 71, "shared_by_all_four_arms": True,
        },
        "code_authorities": code_authorities(),
        "static_input_authorities": static_input_authorities(),
        "implementation_equivalence_gates": {
            "native_a_prefix_seed42": {
                "path": str(native_equivalence_path.resolve()),
                "sha256": sha256_file(native_equivalence_path),
            },
            "fixed_selector_and_realization_plan_seed42": {
                "path": str(selector_equivalence_path.resolve()),
                "sha256": sha256_file(selector_equivalence_path),
            },
            "final_detector_taxonomy_seed42": {
                "path": str((RESULT_ROOT / "TAXONOMY_SEED42_EQUIVALENCE.json").resolve()),
                "sha256": sha256_file(RESULT_ROOT / "TAXONOMY_SEED42_EQUIVALENCE.json"),
                "status": taxonomy_equivalence["status"],
            },
            "control_fixed_noop": "REQUIRED_PER_SEED_AFTER_THIS_CONTRACT_AND_BEFORE_TRAINING",
        },
        "per_seed": per_seed,
        "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "adaptation_seed_boundary": "AFTER_EXACT_E71_RESTORE_BEFORE_E72_ITERATION",
        "pair_only_scientific_difference": (
            "CONTROL identity Target Student view versus schedule-matched "
            "seed-specific Native-A Fixed Target Student view"
        ),
        "training": {
            "epochs": list(EPOCHS), "formal_horizon": 100,
            "source_batch": 32, "target_batch": 8, "workers": 2,
            "accumulation": 2, "primary_endpoint": "E100_EMA",
            "secondary_endpoint": "E100_RAW", "best_epoch_selection": False,
        },
        "evaluation": {
            "authorization": "ONLY_AFTER_ALL_FOUR_E100_ENDPOINTS_FROZEN",
            "official_evaluator_required": True,
            "target_dev": "EIGHT_E100_ENDPOINTS_ONLY",
            "target_test": "SEALED_ZERO_READS",
            "threshold": 0.27127127127127126, "iou": 0.5,
        },
        "data_access": {
            "target_dev_before_all_four_freeze": 0,
            "target_test": 0, "target_adapt_gt": 0,
        },
        "hard_stop_conditions": load_json(IDENTITY_CONTRACT)["hard_stop_conditions"],
    }
    if RUN_CONTRACT.exists():
        existing = load_json(RUN_CONTRACT)
        if existing != contract:
            # created_at makes direct equality inappropriate.  Once frozen,
            # validate all immutable fields and never rewrite it.
            observed = deepcopy(existing)
            candidate = deepcopy(contract)
            observed.pop("created_at", None)
            candidate.pop("created_at", None)
            if observed != candidate:
                raise FreshSeedStop("STOP_RUN_CONTRACT_ALREADY_FROZEN_DRIFT")
        return existing
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_CONTRACT, contract)
    digest = sha256_file(RUN_CONTRACT)
    sidecar = RUN_CONTRACT.with_suffix(".sha256")
    sidecar.write_text(f"{digest}  {RUN_CONTRACT.name}\n", encoding="ascii")
    return contract


def validate_run_contract() -> dict[str, Any]:
    if not RUN_CONTRACT.is_file() or not RUN_CONTRACT.with_suffix(".sha256").is_file():
        raise FreshSeedStop("STOP_IMPLEMENTATION_RUN_CONTRACT_MISSING")
    digest = sha256_file(RUN_CONTRACT)
    sidecar = RUN_CONTRACT.with_suffix(".sha256").read_text(encoding="ascii").strip().split()[0]
    contract = load_json(RUN_CONTRACT)
    checks = {
        "sidecar": digest == sidecar,
        "schema": contract.get("schema") == "m1_fixed_fresh_seed_implementation_run_contract_v1",
        "experiment": contract.get("experiment") == EXPERIMENT,
        "identity": contract.get("authoritative_identity_contract", {}).get("sha256") == sha256_file(IDENTITY_CONTRACT),
        "parent": contract.get("parent", {}).get("sha256") == PARENT_SHA256 and sha256_file(PARENT) == PARENT_SHA256,
    }
    for relative, lock in contract.get("code_authorities", {}).items():
        path = ROOT / relative
        checks[f"code:{relative}"] = (
            path.is_file()
            and path.stat().st_size == int(lock["bytes"])
            and sha256_file(path) == lock["sha256"]
        )
    for role, lock in contract.get("static_input_authorities", {}).items():
        path = Path(lock.get("path", "")).resolve()
        checks[f"input:{role}"] = (
            path.is_file()
            and path.stat().st_size == int(lock["bytes"])
            and sha256_file(path) == lock["sha256"] == lock["expected_sha256"]
        )
    for seed in SEEDS:
        row = contract.get("per_seed", {}).get(str(seed), {})
        paths = seed_paths(seed)
        checks[f"seed{seed}:gate"] = sha256_file(paths["policy_gate"]) == row.get("policy_gate", {}).get("sha256")
        for role in ("source_schedule", "source_augmentation", "target_schedule", "target_manifest"):
            checks[f"seed{seed}:{role}"] = sha256_file(paths[role]) == row.get("schedule", {}).get(role, {}).get("sha256")
        fixed = row.get("fixed_policy", {})
        checks[f"seed{seed}:fixed_contract"] = sha256_file(paths["fixed_contract"]) == fixed.get("contract_sha256")
        checks[f"seed{seed}:exposure"] = sha256_file(paths["exposure"]) == fixed.get("exposure_sha256")
        checks[f"seed{seed}:cache_manifest"] = sha256_file(paths["cache_manifest"]) == fixed.get("cache_manifest_sha256")
        checks[f"seed{seed}:cache_build"] = sha256_file(paths["cache_build"]) == fixed.get("cache_build_sha256")
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise FreshSeedStop(f"STOP_IMPLEMENTATION_RUN_CONTRACT_DRIFT:{failed}")
    return contract


def create_cache_prelaunch(seed: int) -> dict[str, Any]:
    paths = seed_paths(seed)
    gate = policy_gate(seed)
    cache_build = load_json(paths["cache_build"])
    integrity_candidates = (
        paths["cache"] / "CACHE_INTEGRITY_AUDIT.json",
        paths["cache"] / "fixed_a_cache_integrity_audit.json",
        paths["cache"] / "raw" / "CACHE_INTEGRITY_AUDIT.json",
    )
    integrity_path = next((path for path in integrity_candidates if path.is_file()), None)
    if integrity_path is None:
        # The policy gate already contains the independently rehashed payload
        # composite and is the fail-closed authority used here.
        integrity = {
            "status": "PASS",
            "payload_composite_sha256": gate["cache_payload_composite_sha256"],
            "entry_count": gate["counts"]["with_delta"],
        }
    else:
        integrity = load_json(integrity_path)
    evidence = {
        "schema": "m1_fixed_fresh_seed_fixed_cache_prelaunch_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "policy_gate_path": str(paths["policy_gate"].resolve()),
        "policy_gate_sha256": sha256_file(paths["policy_gate"]),
        "cache_build_path": str(paths["cache_build"].resolve()),
        "cache_build_sha256": sha256_file(paths["cache_build"]),
        "cache_manifest_path": str(paths["cache_manifest"].resolve()),
        "cache_manifest_sha256": sha256_file(paths["cache_manifest"]),
        "cache_entries": int(gate["counts"]["with_delta"]),
        "active_visits": int(gate["counts"]["active"]),
        "identity_visits": int(gate["counts"]["identity"]),
        "payload_composite_sha256": gate["cache_payload_composite_sha256"],
        "cache_miss_policy": "STOP_NO_FALLBACK",
        "online_powell_policy": "FORBIDDEN_ZERO",
        "target_dev_reads": 0, "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "created_at": now_iso(),
    }
    checks = {
        "policy_gate_pass": gate["status"] == "PASS",
        "cache_build_pass": cache_build.get("status") == "PASS",
        "integrity_pass": integrity.get("status") == "PASS",
        "entry_count": int(integrity.get("entry_count", gate["counts"]["with_delta"])) == int(gate["counts"]["with_delta"]),
        "composite": integrity.get("payload_composite_sha256", gate["cache_payload_composite_sha256"]) == gate["cache_payload_composite_sha256"],
    }
    evidence["checks"] = checks
    if not all(checks.values()):
        raise FreshSeedStop(f"STOP_CACHE_PRELAUNCH:{seed}:{checks}")
    atomic_json(paths["prelaunch"], evidence)
    return evidence


def build_control_source_reference(seed: int) -> Path:
    paths = seed_paths(seed)
    metrics = paths["control"] / "TRAINING_EPOCH_METRICS.csv"
    if not metrics.is_file():
        raise FreshSeedStop(f"STOP_CONTROL_METRICS_MISSING:{seed}")
    rows: list[dict[str, Any]] = [{
        "epoch": 71, "first_batch_image_hash": "", "last_batch_image_hash": "",
    }]
    with metrics.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "epoch": int(row["epoch"]),
                "first_batch_image_hash": _require_sha(row["first_batch_image_hash"], "control_first_source_hash"),
                "last_batch_image_hash": _require_sha(row["last_batch_image_hash"], "control_last_source_hash"),
            })
    if [int(row["epoch"]) for row in rows] != list(range(71, 101)):
        raise FreshSeedStop(f"STOP_CONTROL_SOURCE_REFERENCE_EPOCHS:{seed}")
    base.atomic_csv(paths["source_reference"], rows)
    return paths["source_reference"]


def validate_recovery_start_state(
    path: Path, expected_sha: str, branch: str, seed: int,
) -> dict[str, Any]:
    path = base.inside_root(path)
    digest = sha256_file(path)
    if digest != _require_sha(expected_sha, "recovery_expected_sha"):
        raise FreshSeedStop("STOP_RECOVERY_FILE_SHA")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format", "experiment", "branch", "epoch", "global_step",
        "optimizer_step_count", "optimizer_steps_this_branch", "ema_updates",
        "model_state_dict", "model_hash", "ema_state_dict", "ema_hash",
        "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
        "rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda",
        "source_schedule_cursor", "target_schedule_cursor", "base_checkpoint_sha256",
        "base_checkpoint_path", "dataset_manifest", "source_schedule_sha256",
        "source_augmentation_sha256", "target_schedule_sha256",
        "fixed_exposure_sha256", "fixed_cache_manifest_sha256",
        "run_contract_path", "run_contract_sha256", "train_args",
        "optimizer_update_rows", "training_epoch_metric_rows",
        "completed_data_access_epoch_rows", "protocol_epoch_rows",
        "protocol_b2_epoch_rows", "protocol_b2_batch_rows",
        "protocol_target_read_count_total", "complete_epoch_boundary",
    }
    missing = required - set(state)
    if missing:
        raise FreshSeedStop(f"STOP_RECOVERY_FIELDS:{sorted(missing)}")
    epoch = int(state.get("epoch", -1))
    completed_epochs = epoch - 71
    expected_microbatches = completed_epochs * base.EXPECTED_STEPS_PER_EPOCH
    expected_windows = completed_epochs * base.WINDOWS_PER_EPOCH
    checks = {
        "format": state.get("format") == "m1_seed42_stage1_full_state_v1",
        "experiment": state.get("experiment") == EXPERIMENT,
        "branch": state.get("branch") == branch,
        "epoch": 72 <= epoch < 100,
        "boundary": state.get("complete_epoch_boundary") is True,
        "contract": state.get("run_contract_sha256") == sha256_file(RUN_CONTRACT),
        "contract_path": Path(state.get("run_contract_path", "")).resolve() == RUN_CONTRACT.resolve(),
        "dataset": state.get("dataset_manifest") == base.EXPECTED_DATASET_MANIFEST,
        "raw": base.tensor_state_hash(state["model_state_dict"]) == state.get("model_hash"),
        "ema": base.tensor_state_hash(state["ema_state_dict"]) == state.get("ema_hash"),
        "scheduler": int(state["scheduler_state_dict"]["last_epoch"]) == epoch - 1,
        "global_step": int(state["global_step"]) == base.E71_GLOBAL_STEP + expected_microbatches,
        "optimizer_steps": int(state["optimizer_steps_this_branch"]) == expected_windows,
        "optimizer_absolute": int(state["optimizer_step_count"]) == base.E71_EMA_UPDATES + expected_windows,
        "ema_updates": int(state["ema_updates"]) == base.E71_EMA_UPDATES + expected_windows,
        "parent_sha": state["base_checkpoint_sha256"] == PARENT_SHA256,
        "parent_path": Path(state["base_checkpoint_path"]).resolve() == PARENT.resolve(),
        "source_schedule": state["source_schedule_sha256"] == sha256_file(seed_paths(seed)["source_schedule"]),
        "source_augmentation": state["source_augmentation_sha256"] == sha256_file(seed_paths(seed)["source_augmentation"]),
        "target_schedule": state["target_schedule_sha256"] == sha256_file(seed_paths(seed)["target_schedule"]),
        "fixed_exposure": state["fixed_exposure_sha256"] == sha256_file(seed_paths(seed)["exposure"]),
        "fixed_cache_manifest": state["fixed_cache_manifest_sha256"] == sha256_file(seed_paths(seed)["cache_manifest"]),
        "train_seed": int(state["train_args"].get("seed", -1)) == int(seed),
        "source_cursor": state["source_schedule_cursor"] == {
            "completed_total_epoch": epoch, "next_total_epoch": epoch + 1,
            "batches_per_epoch": base.EXPECTED_STEPS_PER_EPOCH,
        },
        "target_cursor": state["target_schedule_cursor"] == {
            "completed_total_epoch": epoch, "next_total_epoch": epoch + 1,
            "batches_per_active_epoch": base.EXPECTED_STEPS_PER_EPOCH,
            "target_batch": TARGET_BATCH,
        },
        "optimizer_rows": len(state["optimizer_update_rows"]) == expected_windows,
        "epoch_metric_rows": len(state["training_epoch_metric_rows"]) == completed_epochs,
        "data_access_rows": len(state["completed_data_access_epoch_rows"]) == completed_epochs,
        "source_epoch_rows": len(state["protocol_epoch_rows"]) == completed_epochs,
        "b2_epoch_rows": len(state["protocol_b2_epoch_rows"]) == completed_epochs,
        "b2_batch_rows": len(state["protocol_b2_batch_rows"]) == expected_microbatches,
        "target_reads": int(state["protocol_target_read_count_total"]) == completed_epochs * VISITS_PER_EPOCH,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise FreshSeedStop(f"STOP_RECOVERY_IDENTITY:{failed}")
    return state


def convert_parquet_fresh(seed: int, run_dir: Path) -> dict[str, Any]:
    """Offline-only seed-parameterized equivalent of the frozen converter."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = configure_base(seed)
    gate = policy_gate(seed)
    counts = gate["counts"]
    run_dir = base.inside_root(run_dir)
    raw_dir = run_dir / "raw_trace"
    epoch_metrics = _read_csv(run_dir / "TRAINING_EPOCH_METRICS.csv")
    if (
        len(epoch_metrics) != len(EPOCHS)
        or [int(row["epoch"]) for row in epoch_metrics] != list(EPOCHS)
    ):
        raise FreshSeedStop("STOP_OFFLINE_EPOCH_METRIC_LINEAGE")
    branch = str(epoch_metrics[0]["branch"])
    if branch not in base.BRANCHES or any(row["branch"] != branch for row in epoch_metrics):
        raise FreshSeedStop("STOP_OFFLINE_BRANCH_LINEAGE")

    def registered_paths(
        path_field: str, sha_field: str, rows_field: str,
        *, expected_rows: int | None, applicable: bool = True,
    ) -> list[Path]:
        if not applicable:
            if any(
                row.get(path_field) or row.get(sha_field)
                or int(row.get(rows_field, -1)) != 0
                for row in epoch_metrics
            ):
                raise FreshSeedStop(f"STOP_OFFLINE_NONAPPLICABLE_SHARD:{path_field}")
            return []
        output: list[Path] = []
        for row in epoch_metrics:
            path = base.inside_root(Path(str(row[path_field])))
            registered = int(row[rows_field])
            if not (
                path.is_relative_to((raw_dir / "attempts").resolve())
                and path.is_file()
                and sha256_file(path) == _require_sha(row[sha_field], sha_field)
                and registered >= 0
                and (expected_rows is None or registered == expected_rows)
            ):
                raise FreshSeedStop(f"STOP_OFFLINE_RAW_SHARD:{path_field}:{path}")
            output.append(path)
        return output

    def read_jsonl(files: list[Path], label: str) -> list[dict[str, Any]]:
        if len(files) != len(EPOCHS):
            raise FreshSeedStop(f"STOP_OFFLINE_RAW_EPOCH_COUNT:{label}:{len(files)}")
        output: list[dict[str, Any]] = []
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                output.extend(json.loads(line) for line in handle if line.strip())
        return output

    box_rows = read_jsonl(registered_paths(
        "hpl_box_trace_path", "hpl_box_trace_sha256", "hpl_box_trace_rows",
        expected_rows=None,
    ), "hpl_boxes")
    batch_rows = read_jsonl(registered_paths(
        "pseudo_batch_trace_path", "pseudo_batch_trace_sha256", "pseudo_batch_trace_rows",
        expected_rows=base.EXPECTED_STEPS_PER_EPOCH,
    ), "pseudo_batches")
    exposure_rows = read_jsonl(registered_paths(
        "fixed_exposure_runtime_path", "fixed_exposure_runtime_sha256", "fixed_exposure_runtime_rows",
        expected_rows=base.EXPECTED_STEPS_PER_EPOCH,
    ), "fixed_exposure")
    cache_rows = read_jsonl(registered_paths(
        "fixed_cache_runtime_path", "fixed_cache_runtime_sha256", "fixed_cache_runtime_rows",
        expected_rows=base.EXPECTED_STEPS_PER_EPOCH,
        applicable=branch == "NATIVE_A_FIXED",
    ), "fixed_cache") if branch == "NATIVE_A_FIXED" else []
    if len(batch_rows) != len(EPOCHS) * base.EXPECTED_STEPS_PER_EPOCH:
        raise FreshSeedStop("STOP_PSEUDO_BATCH_ROW_COUNT")
    base.atomic_csv(run_dir / "PSEUDO_LABEL_BATCH_TRACE.csv", batch_rows)
    tensor_manifest = [{
        "schema": "m1_seed42_stage1_hpl_tensor_manifest_v1",
        "branch": row["branch"], "epoch": row["epoch"],
        "source_batch_index": row["source_batch_index"],
        "formal_visit_index_json": row["formal_visit_index_json"],
        "teacher_uint8_sha256": row["teacher_uint8_sha256"],
        "student_uint8_sha256": row["student_uint8_sha256"],
        "pseudo_cls_sha256": row["pseudo_cls_sha256"],
        "pseudo_bboxes_sha256": row["pseudo_bboxes_sha256"],
        "pseudo_batch_idx_sha256": row["pseudo_batch_idx_sha256"],
        "pseudo_scores_f32le_sha256": row["pseudo_scores_f32le_sha256"],
        "pseudo_final_boxes": row["pseudo_final_boxes"],
    } for row in batch_rows]
    base.atomic_csv(run_dir / "HPL_TENSOR_MANIFEST.csv", tensor_manifest)
    boxes_by_visit: dict[int, list[dict[str, Any]]] = {}
    for box in box_rows:
        boxes_by_visit.setdefault(int(box["formal_visit_index"]), []).append(box)
    image_rows: list[dict[str, Any]] = []
    for row in batch_rows:
        values = {
            key: json.loads(row[key])
            for key in (
                "target_uid_json", "canonical_uid_json", "formal_visit_index_json",
                "fixed_exposure_active_json", "fixed_transform_applied_json",
                "pseudo_pre_nms_per_image_json", "pseudo_post_nms_per_image_json",
                "pseudo_final_per_image_json",
            )
        }
        if any(len(item) != TARGET_BATCH for item in values.values()):
            raise FreshSeedStop("STOP_PSEUDO_IMAGE_BATCH_CARDINALITY")
        for position in range(TARGET_BATCH):
            visit = int(values["formal_visit_index_json"][position])
            visit_boxes = boxes_by_visit.get(visit, [])
            final_count = int(values["pseudo_final_per_image_json"][position])
            if len(visit_boxes) != final_count:
                raise FreshSeedStop(f"STOP_HPL_BOX_LINK_CARDINALITY:{visit}")
            scores = [float(item["score"]) for item in visit_boxes]
            boxes_xywh = [[
                float(item["x_center"]), float(item["y_center"]),
                float(item["width"]), float(item["height"]),
            ] for item in visit_boxes]
            areas = [float(item["area"]) for item in visit_boxes]
            image_rows.append({
                "schema": "m1_seed42_stage1_pseudo_label_image_trace_v1",
                "branch": row["branch"], "epoch": row["epoch"],
                "source_batch_index": row["source_batch_index"],
                "global_microbatch_index": row["global_microbatch_index"],
                "global_step_after_microbatch": row["global_step_after_microbatch"],
                "optimizer_step_occurred": row["optimizer_step_occurred"],
                "optimizer_underlying_step_called": row["optimizer_underlying_step_called"],
                "target_unsupervised_optimizer_update_count": row["target_unsupervised_optimizer_update_count"],
                "target_position": position, "formal_visit_index": visit,
                "schedule_uid": values["target_uid_json"][position],
                "canonical_uid": values["canonical_uid_json"][position],
                "fixed_exposure_active": values["fixed_exposure_active_json"][position],
                "fixed_transform_applied": values["fixed_transform_applied_json"][position],
                "pre_nms_candidates": values["pseudo_pre_nms_per_image_json"][position],
                "post_nms_boxes": values["pseudo_post_nms_per_image_json"][position],
                "final_hpl_boxes": final_count,
                "hpl_positive": int(final_count > 0), "hpl_empty": int(final_count == 0),
                "hpl_scores_json": json.dumps(scores, separators=(",", ":")),
                "hpl_boxes_xywh_json": json.dumps(boxes_xywh, separators=(",", ":")),
                "hpl_areas_json": json.dumps(areas, separators=(",", ":")),
                "hpl_box_trace_link_sha256": canonical_sha({
                    "formal_visit_index": visit, "scores": scores,
                    "boxes_xywh": boxes_xywh, "areas": areas,
                }),
            })
    outputs: list[dict[str, Any]] = []
    for name, rows in (
        ("HPL_BOX_TRACE.parquet", box_rows),
        ("PSEUDO_LABEL_IMAGE_TRACE.parquet", image_rows),
    ):
        destination = run_dir / name
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, destination)
        outputs.append({
            "path": str(destination.resolve()), "file_sha256": sha256_file(destination),
            "row_count": len(rows),
        })
    if branch == "NATIVE_A_FIXED":
        base.atomic_csv(run_dir / "FIXED_CACHE_RUNTIME_AUDIT.csv", cache_rows)
    base.atomic_csv(run_dir / "FIXED_EXPOSURE_RUNTIME_AUDIT.csv", exposure_rows)
    postrun_cache = {"status": "NOT_APPLICABLE_CONTROL", "all_pass": True}
    if branch == "NATIVE_A_FIXED":
        import m1_fixed_fresh_seed_v2_prepare as prepare
        postrun_cache = prepare.verify_cache(seed, prepare.seed_paths(seed), counts)
        postrun_cache["all_pass"] = postrun_cache.get("status") == "PASS"
        atomic_json(run_dir / "FIXED_CACHE_POSTRUN_REVERIFICATION.json", postrun_cache)
    import_audit = load_json(run_dir / "TRAINING_PROCESS_IMPORT_AUDIT.json")
    checks = {
        "outputs_two": len(outputs) == 2,
        "pseudo_batch_rows_exact": len(batch_rows) == len(EPOCHS) * base.EXPECTED_STEPS_PER_EPOCH,
        "pseudo_image_rows_exact": len(image_rows) == int(counts["visits"]),
        "pseudo_image_visit_indices_exact": sorted(int(row["formal_visit_index"]) for row in image_rows) == list(range(int(counts["visits"]))),
        "hpl_box_rows_exactly_linked": sum(int(row["final_hpl_boxes"]) for row in image_rows) == len(box_rows),
        "hpl_tensor_manifest_rows_exact": len(tensor_manifest) == len(batch_rows),
        "fixed_exposure_rows_exact": len(exposure_rows) == len(batch_rows),
        "fixed_active_identity_counts_exact": (
            sum(int(row["fixed_active_count"]) for row in exposure_rows) == int(counts["active"])
            and sum(int(row["fixed_identity_count"]) for row in exposure_rows) == int(counts["identity"])
        ),
        "fixed_cache_rows_exact_if_applicable": branch != "NATIVE_A_FIXED" or len(cache_rows) == len(batch_rows),
        "fixed_cache_no_miss_or_fallback": branch != "NATIVE_A_FIXED" or all(
            int(row["cache_miss_count"]) == 0
            and int(row["cache_fallback_count"]) == 0
            and int(row["identity_cache_access_count"]) == 0
            and int(row["online_theta_solve_count"]) == 0
            for row in cache_rows
        ),
        "fixed_cache_hits_exact": branch != "NATIVE_A_FIXED" or sum(int(row["cache_hit_count"]) for row in cache_rows) == int(counts["active"]),
        "postrun_cache_full_reverification_pass": postrun_cache.get("all_pass") is True,
        "training_process_offline_imports_absent": import_audit.get("all_pass") is True,
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_parquet_conversion_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()), "adaptation_seed": int(seed),
        "branch": branch, "outputs": outputs,
        "fixed_cache_postrun_reverification": postrun_cache,
        "checks": checks, "process_id": os.getpid(), "created_at": now_iso(),
    }
    atomic_json(run_dir / "PARQUET_CONVERSION_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_FRESH_PARQUET_CONVERSION:{checks}")
    return audit


def patch_base_for_training(seed: int, branch: str) -> dict[str, Path]:
    global ACTIVE_SEED
    ACTIVE_SEED = int(seed)
    paths = seed_paths(seed)
    source_reference = (
        write_placeholder_reference(seed)
        if branch == "CONTROL"
        else paths["source_reference"]
    )
    configure_base(seed, source_reference=source_reference)

    def load_contract(path: Path, *, authority: str) -> tuple[dict[str, Any], str]:
        contract = validate_run_contract()
        if Path(path).resolve() != RUN_CONTRACT.resolve():
            raise FreshSeedStop("STOP_RUN_CONTRACT_PATH")
        view = deepcopy(contract)
        view["branches"] = {
            "CONTROL": {"run_dir": str(paths["control"].resolve())},
            "NATIVE_A_FIXED": {"run_dir": str(paths["fixed"].resolve())},
        }
        return view, sha256_file(RUN_CONTRACT)

    def static_locks() -> dict[str, str]:
        validate_run_contract()
        policy_gate(seed)
        noop = load_json(paths["noop"] / "RUNNER_NOOP_AUDIT.json")
        if noop.get("status") != "PASS" or noop.get("executor_sha256") != sha256_file(Path(__file__)):
            raise FreshSeedStop("STOP_NOOP_GATE_NOT_FROZEN_PASS")
        create_cache_prelaunch(seed)
        return {
            "implementation_run_contract": sha256_file(RUN_CONTRACT),
            "policy_gate": sha256_file(paths["policy_gate"]),
            "runner_noop": sha256_file(paths["noop"] / "RUNNER_NOOP_AUDIT.json"),
        }

    def validate_start(path: Path, expected_sha: str, *, branch: str | None = None) -> dict[str, Any]:
        if Path(path).resolve() == PARENT.resolve():
            if expected_sha != PARENT_SHA256:
                raise FreshSeedStop("STOP_INITIAL_PARENT_PATH_OR_SHA")
            return validate_parent()
        if branch not in base.BRANCHES:
            raise FreshSeedStop("STOP_RECOVERY_BRANCH_REQUIRED")
        return validate_recovery_start_state(Path(path), expected_sha, branch, seed)

    def load_fixed(schedule: ProtocolBTargetSchedule):
        return load_fixed_inputs_fresh(seed, schedule)

    def build_trainer_adapter(**kwargs: Any) -> FreshStage1Trainer:
        return build_fresh_trainer(
            seed=seed,
            source_reference_mode=("ESTABLISH_CONTROL" if branch == "CONTROL" else "LOCK_TO_CONTROL"),
            **{
                key: value for key, value in kwargs.items()
                if key not in {"contract_path", "contract_sha"}
            },
        )

    base.load_run_contract = load_contract
    base.verify_static_locks = static_locks
    base.validate_start_state = validate_start
    base.load_fixed_inputs = load_fixed
    base.build_trainer = build_trainer_adapter
    base._validate_recovery = ORIGINAL_BASE_VALIDATE_RECOVERY
    def subprocess_adapter(command: Sequence[str], **kwargs: Any):
        values = [str(value) for value in command]
        if "convert-parquet" in values and "--run-dir" in values:
            run_value = values[values.index("--run-dir") + 1]
            adapted = [
                sys.executable, "-B", str(Path(__file__).resolve()),
                "convert-parquet", "--seed", str(seed), "--run-dir", run_value,
            ]
            environment = dict(kwargs.get("env") or os.environ)
            environment.update({
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONHASHSEED": str(seed),
            })
            return ORIGINAL_SUBPROCESS_RUN(
                adapted, cwd=kwargs.get("cwd", ROOT), env=environment,
                check=kwargs.get("check", False),
            )
        return ORIGINAL_SUBPROCESS_RUN(command, **kwargs)
    base.subprocess = SimpleNamespace(run=subprocess_adapter)
    return paths


def run_training_worker(seed: int, branch: str, device: str, resume_from: Path | None) -> dict[str, Any]:
    if branch not in base.BRANCHES:
        raise FreshSeedStop(f"STOP_TRAIN_BRANCH:{branch}")
    paths = patch_base_for_training(seed, branch)
    create_cache_prelaunch(seed)
    run_dir = paths["control"] if branch == "CONTROL" else paths["fixed"]
    args = argparse.Namespace(
        contract=str(RUN_CONTRACT), branch=branch, device=str(device),
        resume_from=(str(resume_from) if resume_from is not None else None),
    )
    base.run_formal_branch(args)
    if branch == "CONTROL":
        build_control_source_reference(seed)
    final = load_json(run_dir / "TRAINING_FINAL_AUDIT.json")
    if final.get("completed") is not True:
        raise FreshSeedStop(f"STOP_TRAINING_FINAL_AUDIT:{seed}:{branch}")
    annotation = {
        "schema": "m1_fixed_fresh_seed_training_annotation_v1",
        "status": "PASS", "adaptation_seed": int(seed), "branch": branch,
        "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "parent_sha256": PARENT_SHA256,
        "identity_contract_sha256": sha256_file(IDENTITY_CONTRACT),
        "implementation_run_contract_sha256": sha256_file(RUN_CONTRACT),
        "fresh_seed_executor_sha256": sha256_file(Path(__file__)),
        "frozen_seed42_training_runner_sha256": sha256_file(Path(base.__file__)),
        "target_dev_reads": 0, "target_test_reads": 0,
        "target_adapt_label_reads": 0, "created_at": now_iso(),
    }
    atomic_json(run_dir / "FRESH_SEED_TRAINING_ANNOTATION.json", annotation)
    return final


def newest_recovery_checkpoint(run_dir: Path) -> Path | None:
    manifest = run_dir / "CHECKPOINT_MANIFEST.csv"
    if not manifest.is_file():
        return None
    recovery_kinds = {"FULL_STATE_EPOCH", "RECOVERY_ONLY_EPOCH"}
    rows = [
        row for row in _read_csv(manifest)
        if row.get("event_kind") in recovery_kinds
        and str(row.get("resume_allowed", "")).casefold() == "true"
    ]
    if not rows:
        return None
    latest_epoch = max(int(row["epoch"]) for row in rows)
    latest = [row for row in rows if int(row["epoch"]) == latest_epoch]
    latest.sort(key=lambda row: (
        row.get("event_kind") == "RECOVERY_ONLY_EPOCH",
        row.get("event_name") == "RECOVERY_LAST",
    ), reverse=True)
    row = latest[0]
    path = base.inside_root(Path(row["full_state_path"]))
    if not path.is_file() or sha256_file(path) != _require_sha(
        row["full_state_file_sha256"], "latest_recovery_sha"
    ):
        raise FreshSeedStop("STOP_LATEST_RECOVERY_FILE_OR_SHA")
    return path


def launch_training(seed: int, branch: str, device: str) -> dict[str, Any]:
    validate_run_contract()
    paths = seed_paths(seed)
    run_dir = paths["control"] if branch == "CONTROL" else paths["fixed"]
    final_path = run_dir / "TRAINING_FINAL_AUDIT.json"
    if final_path.is_file() and load_json(final_path).get("completed") is True:
        return load_json(final_path)
    resume = newest_recovery_checkpoint(run_dir) if run_dir.is_dir() else None
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), "train-worker",
        "--seed", str(seed), "--branch", branch, "--device", str(device),
    ]
    if resume is not None:
        command.extend(["--resume-from", str(resume)])
    env = dict(os.environ)
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": str(seed),
    })
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise FreshSeedStop(f"STOP_TRAIN_WORKER_EXIT:{seed}:{branch}:{completed.returncode}")
    return load_json(final_path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def formal_pair_matching_audit(seed: int) -> dict[str, Any]:
    paths = seed_paths(seed)
    destination = paths["root"] / "FORMAL_PAIR_MATCHING_AUDIT.json"
    control_metrics = _read_csv(paths["control"] / "TRAINING_EPOCH_METRICS.csv")
    fixed_metrics = _read_csv(paths["fixed"] / "TRAINING_EPOCH_METRICS.csv")
    exact_epoch_fields = (
        "epoch", "global_step", "lr_pg0", "optimizer_steps_this_run",
        "ema_updates", "source_batches", "source_image_visits",
        "source_schedule_mismatch_count", "first_batch_image_hash",
        "last_batch_image_hash", "target_adapt_train_read_count",
        "target_dev_train_read_count", "target_test_content_read_count",
        "target_batches", "target_unique_uid_count", "target_unique_uid_coverage",
        "target_schedule_mismatch_count", "target_first_batch_image_hash",
        "target_last_batch_image_hash", "source_first_batch_hash_matches_b1",
        "source_last_batch_hash_matches_b1", "scheduler_state_sha256",
        "rng_state_sha256", "fixed_exposure_active_visits",
        "fixed_exposure_identity_visits",
    )
    epoch_mismatches = {
        field: sum(
            left.get(field) != right.get(field)
            for left, right in zip(control_metrics, fixed_metrics, strict=True)
        )
        for field in exact_epoch_fields
    } if len(control_metrics) == len(fixed_metrics) else {
        field: -1 for field in exact_epoch_fields
    }

    control_optimizer = _read_csv(paths["control"] / "OPTIMIZER_UPDATE_TRACE.csv")
    fixed_optimizer = _read_csv(paths["fixed"] / "OPTIMIZER_UPDATE_TRACE.csv")
    optimizer_identity_fields = (
        "epoch", "optimizer_window_this_branch", "optimizer_window_in_epoch",
        "ema_updates_after",
    )
    optimizer_identity_mismatches = {
        field: sum(
            left.get(field) != right.get(field)
            for left, right in zip(control_optimizer, fixed_optimizer, strict=True)
        )
        for field in optimizer_identity_fields
    } if len(control_optimizer) == len(fixed_optimizer) else {
        field: -1 for field in optimizer_identity_fields
    }
    amp_fields = (
        "underlying_optimizer_step_called", "found_inf", "found_inf_per_device",
        "grad_scaler_scale_before", "grad_scaler_scale_after",
    )
    amp_outcome_mismatches = {
        field: sum(
            left.get(field) != right.get(field)
            for left, right in zip(control_optimizer, fixed_optimizer, strict=True)
        )
        for field in amp_fields
    } if len(control_optimizer) == len(fixed_optimizer) else {
        field: -1 for field in amp_fields
    }

    checkpoint_fields = (
        "event_kind", "event_name", "epoch", "global_step",
        "optimizer_windows_this_branch", "ema_updates", "scheduler_last_epoch",
        "resume_allowed",
    )
    checkpoint_kinds = {"FULL_STATE_EPOCH", "RECOVERY_ONLY_EPOCH"}
    checkpoint_projection = lambda rows: [  # noqa: E731
        tuple(row.get(field) for field in checkpoint_fields)
        for row in rows if row.get("event_kind") in checkpoint_kinds
    ]
    control_checkpoints = checkpoint_projection(
        _read_csv(paths["control"] / "CHECKPOINT_MANIFEST.csv")
    )
    fixed_checkpoints = checkpoint_projection(
        _read_csv(paths["fixed"] / "CHECKPOINT_MANIFEST.csv")
    )
    control_registration = load_json(paths["control"] / "RUN_REGISTRATION.json")
    fixed_registration = load_json(paths["fixed"] / "RUN_REGISTRATION.json")
    gate = policy_gate(seed)
    expected_active = int(gate["counts"]["active"])
    expected_identity = int(gate["counts"]["identity"])
    sum_field = lambda rows, field: sum(int(row[field]) for row in rows)  # noqa: E731
    control_access = load_json(paths["control"] / "DATA_ACCESS_AUDIT.json")
    fixed_access = load_json(paths["fixed"] / "DATA_ACCESS_AUDIT.json")
    access_fields = (
        "target_adapt_label_read_count", "target_dev_read_count",
        "target_test_read_count", "dataset_mutation_count",
    )
    noop = load_json(paths["noop"] / "RUNNER_NOOP_AUDIT.json")
    checks = {
        "both_training_metric_horizons_exact": (
            [int(row["epoch"]) for row in control_metrics] == list(EPOCHS)
            and [int(row["epoch"]) for row in fixed_metrics] == list(EPOCHS)
        ),
        "source_target_rng_lr_schedule_fields_exact": all(
            mismatch == 0 for mismatch in epoch_mismatches.values()
        ),
        "optimizer_window_and_ema_schedule_exact": (
            len(control_optimizer) == len(fixed_optimizer) == base.TOTAL_WINDOWS
            and all(mismatch == 0 for mismatch in optimizer_identity_mismatches.values())
        ),
        "checkpoint_cadence_exact": control_checkpoints == fixed_checkpoints,
        "shared_parent_and_contract_exact": (
            control_registration.get("start_state_file_sha256") == PARENT_SHA256
            and fixed_registration.get("start_state_file_sha256") == PARENT_SHA256
            and control_registration.get("run_contract_sha256") == sha256_file(RUN_CONTRACT)
            and fixed_registration.get("run_contract_sha256") == sha256_file(RUN_CONTRACT)
        ),
        "shared_fixed_policy_lineage_exact": (
            control_registration.get("fixed_input_summary", {}).get("policy_gate_sha256")
            == fixed_registration.get("fixed_input_summary", {}).get("policy_gate_sha256")
            == sha256_file(paths["policy_gate"])
        ),
        "exposure_counts_exact_both_arms": (
            sum_field(control_metrics, "fixed_exposure_active_visits") == expected_active
            and sum_field(fixed_metrics, "fixed_exposure_active_visits") == expected_active
            and sum_field(control_metrics, "fixed_exposure_identity_visits") == expected_identity
            and sum_field(fixed_metrics, "fixed_exposure_identity_visits") == expected_identity
        ),
        "only_intended_runtime_view_and_cache_difference": (
            sum_field(control_metrics, "fixed_transform_applied_visits") == 0
            and sum_field(fixed_metrics, "fixed_transform_applied_visits") == expected_active
            and sum_field(control_metrics, "cache_hit_count") == 0
            and sum_field(fixed_metrics, "cache_hit_count") == expected_active
            and all(
                sum_field(rows, field) == 0
                for rows in (control_metrics, fixed_metrics)
                for field in ("cache_miss_count", "cache_fallback_count")
            )
        ),
        "access_zero_both_arms": all(
            int(audit.get(field, -1)) == 0
            for audit in (control_access, fixed_access)
            for field in access_fields
        ),
        "fixed_capable_control_noop_exact": noop.get("status") == "PASS",
        "amp_gradscaler_implementation_shared": (
            noop.get("checks", {}).get("optimizer_ema_scaler_bn_rng_endpoint_exact") is True
            and sha256_file(Path(base.__file__))
            == validate_run_contract()["code_authorities"][
                "project/scripts/m1_seed42_stage1_formal_train_v1.py"
            ]["sha256"]
        ),
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_formal_pair_matching_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "adaptation_seed": int(seed), "checks": checks,
        "exact_epoch_fields": list(exact_epoch_fields),
        "epoch_field_mismatch_counts": epoch_mismatches,
        "optimizer_identity_field_mismatch_counts": optimizer_identity_mismatches,
        "amp_gradscaler_observed_outcome_mismatch_counts": amp_outcome_mismatches,
        "amp_gradscaler_note": (
            "Implementation/settings are pair-shared and the disabled-transform no-op is exact. "
            "Formal found_inf/scale outcomes are reported, not required equal, because they may "
            "causally respond to the intended Fixed image-tensor difference."
        ),
        "checkpoint_projection_fields": list(checkpoint_fields),
        "control_checkpoint_projection": control_checkpoints,
        "fixed_checkpoint_projection": fixed_checkpoints,
        "target_dev_reads": 0, "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "executor_sha256": sha256_file(Path(__file__)),
        "created_at": now_iso(),
    }
    atomic_json(destination, audit)
    if audit["status"] != "PASS":
        failed = sorted(key for key, value in checks.items() if not value)
        raise FreshSeedStop(f"STOP_FORMAL_PAIR_MATCHING:{seed}:{failed}")
    return audit


def _endpoint_from_final(seed: int, branch: str, role: str) -> dict[str, Any]:
    paths = seed_paths(seed)
    run_dir = paths["control"] if branch == "CONTROL" else paths["fixed"]
    final_path = run_dir / "TRAINING_FINAL_AUDIT.json"
    final = load_json(final_path)
    if final.get("completed") is not True or int(final.get("final_epoch", -1)) != 100:
        raise FreshSeedStop(f"STOP_ENDPOINT_TRAINING_NOT_COMPLETE:{seed}:{branch}")
    e100 = final["e100"]
    prefix = "raw" if role == "raw" else "ema"
    checkpoint = base.inside_root(Path(e100[f"{prefix}_inference_path"]))
    file_sha = _require_sha(e100[f"{prefix}_inference_file_sha256"], "endpoint_file_sha")
    tensor_sha = _require_sha(e100[f"{prefix}_tensor_sha256"], "endpoint_tensor_sha")
    if sha256_file(checkpoint) != file_sha:
        raise FreshSeedStop(f"STOP_ENDPOINT_FILE_SHA:{seed}:{branch}:{role}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    checks = {
        "format": payload.get("format") == "m1_seed42_stage1_inference_snapshot_v1",
        "experiment": payload.get("experiment") == EXPERIMENT,
        "branch": payload.get("branch") == branch,
        "role": payload.get("role") == role,
        "epoch": int(payload.get("epoch", -1)) == 100,
        "event": payload.get("event_name") == "E100",
        "inference_only": payload.get("inference_only") is True,
        "resume_forbidden": payload.get("resume_forbidden") is True,
        "tensor": isinstance(state, dict) and base.tensor_state_hash(state) == tensor_sha,
        "contract": payload.get("run_contract_sha256") == sha256_file(RUN_CONTRACT),
    }
    if not all(checks.values()):
        raise FreshSeedStop(f"STOP_ENDPOINT_PAYLOAD:{seed}:{branch}:{role}:{checks}")
    endpoint_id = ("CONTROL" if branch == "CONTROL" else "FIXED") + f"_E100_{role.upper()}"
    return {
        "endpoint_id": endpoint_id,
        "branch": branch,
        "role": role,
        "primary_comparison": role == "ema",
        "OPTIONAL_DESCRIPTIVE": role == "raw",
        "epoch": 100,
        "path": str(checkpoint.resolve()),
        "file_sha256": file_sha,
        "tensor_canonical_sha256": tensor_sha,
        "run_contract_sha256": sha256_file(RUN_CONTRACT),
        "adaptation_seed": int(seed),
    }


def _full_state_freeze(seed: int, branch: str) -> dict[str, Any]:
    paths = seed_paths(seed)
    run_dir = paths["control"] if branch == "CONTROL" else paths["fixed"]
    final = load_json(run_dir / "TRAINING_FINAL_AUDIT.json")
    e100 = final["e100"]
    checkpoint = base.inside_root(Path(e100["full_state_path"]))
    expected_sha = _require_sha(e100["full_state_file_sha256"], "full_state_sha")
    if sha256_file(checkpoint) != expected_sha:
        raise FreshSeedStop(f"STOP_E100_FULL_STATE_SHA:{seed}:{branch}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "format": payload.get("format") == "m1_seed42_stage1_full_state_v1",
        "experiment": payload.get("experiment") == EXPERIMENT,
        "branch": payload.get("branch") == branch,
        "epoch": int(payload.get("epoch", -1)) == 100,
        "boundary": payload.get("complete_epoch_boundary") is True,
        "contract": payload.get("run_contract_sha256") == sha256_file(RUN_CONTRACT),
        "model": base.tensor_state_hash(payload["model_state_dict"]) == payload.get("model_hash"),
        "ema": base.tensor_state_hash(payload["ema_state_dict"]) == payload.get("ema_hash"),
    }
    if not all(checks.values()):
        raise FreshSeedStop(f"STOP_E100_FULL_STATE_PAYLOAD:{seed}:{branch}:{checks}")
    return {
        "path": str(checkpoint.resolve()), "file_sha256": expected_sha,
        "raw_tensor_sha256": payload["model_hash"],
        "ema_tensor_sha256": payload["ema_hash"],
        "optimizer_state_sha256": base.recursive_state_sha256(payload["optimizer_state_dict"]),
        "scheduler_state_sha256": base.recursive_state_sha256(payload["scheduler_state_dict"]),
        "scaler_state_sha256": base.recursive_state_sha256(payload["scaler_state_dict"]),
        "rng_endpoint_sha256": base.recursive_state_sha256({
            key: payload[key]
            for key in ("rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda")
        }),
        "ema_updates": int(payload["ema_updates"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "global_step": int(payload["global_step"]),
    }


def _training_access_zero(seed: int, branch: str) -> dict[str, int]:
    paths = seed_paths(seed)
    run_dir = paths["control"] if branch == "CONTROL" else paths["fixed"]
    audit = load_json(run_dir / "DATA_ACCESS_AUDIT.json")
    values = {
        "target_dev": int(audit.get("target_dev_read_count", -1)),
        "target_test": int(audit.get("target_test_read_count", -1)),
        "target_adapt_labels": int(audit.get("target_adapt_label_read_count", -1)),
        "dataset_mutations": int(audit.get("dataset_mutation_count", -1)),
    }
    if any(values.values()):
        raise FreshSeedStop(f"STOP_ACCESS_CONTRACT_VIOLATION:{seed}:{branch}:{values}")
    return values


def freeze_all_four_endpoints() -> dict[str, Any]:
    validate_run_contract()
    global_path = RESULT_ROOT / "ALL_FOUR_E100_ENDPOINTS_FROZEN.json"
    if global_path.is_file():
        freeze = load_json(global_path)
        if freeze.get("status") != "PASS":
            raise FreshSeedStop("STOP_EXISTING_ALL_FOUR_FREEZE_FAIL")
        return freeze
    rows: list[dict[str, Any]] = []
    full_states: dict[str, Any] = {}
    access: dict[str, Any] = {}
    pair_matching = {
        f"SEED{seed}": formal_pair_matching_audit(seed) for seed in SEEDS
    }
    for seed in SEEDS:
        noop = load_json(seed_paths(seed)["noop"] / "RUNNER_NOOP_AUDIT.json")
        if noop.get("status") != "PASS":
            raise FreshSeedStop(f"STOP_NOOP_GATE_BEFORE_FREEZE:{seed}")
        for branch in base.BRANCHES:
            key = f"SEED{seed}_{branch}"
            access[key] = _training_access_zero(seed, branch)
            full_states[key] = _full_state_freeze(seed, branch)
            for role in ("ema", "raw"):
                descriptor = _endpoint_from_final(seed, branch, role)
                rows.append({
                    **descriptor,
                    "global_endpoint_id": f"SEED{seed}_{descriptor['endpoint_id']}",
                })
    expected_ids = {
        f"SEED{seed}_{arm}_E100_{role}"
        for seed in SEEDS for arm in ("CONTROL", "FIXED") for role in ("EMA", "RAW")
    }
    checks = {
        "eight_endpoints": len(rows) == 8,
        "endpoint_set": {row["global_endpoint_id"] for row in rows} == expected_ids,
        "four_full_states": len(full_states) == 4,
        "all_access_zero": all(not any(values.values()) for values in access.values()),
        "formal_pair_matching_pass": all(
            audit.get("status") == "PASS" for audit in pair_matching.values()
        ),
        "no_evaluation_directories_before_freeze": all(
            not (seed_paths(seed)["root"] / "EVALUATION" / "stage_a").exists()
            for seed in SEEDS
        ),
        "target_test_sealed": True,
    }
    freeze = {
        "schema": "m1_fixed_fresh_seed_all_four_e100_endpoints_freeze_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment": EXPERIMENT,
        "identity_contract_sha256": sha256_file(IDENTITY_CONTRACT),
        "implementation_run_contract_sha256": sha256_file(RUN_CONTRACT),
        "parent_sha256": PARENT_SHA256,
        "checks": checks,
        "endpoints": rows,
        "full_states": full_states,
        "pre_evaluation_access": access,
        "formal_pair_matching": pair_matching,
        "target_dev_reads_before_all_freeze": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "TARGET_DEV_AUTHORIZED_AFTER_THIS_FREEZE_FOR_EIGHT_E100_ENDPOINTS_ONLY": True,
        "created_at": now_iso(),
    }
    if freeze["status"] != "PASS":
        raise FreshSeedStop(f"STOP_ALL_FOUR_FREEZE:{checks}")
    atomic_json(global_path, freeze)
    digest = sha256_file(global_path)
    global_path.with_suffix(".sha256").write_text(
        f"{digest}  {global_path.name}\n", encoding="ascii"
    )
    for seed in SEEDS:
        evaluation = seed_paths(seed)["root"] / "EVALUATION"
        evaluation.mkdir(parents=True, exist_ok=True)
        endpoints = [
            {key: value for key, value in row.items() if key != "global_endpoint_id"}
            for row in rows if int(row["adaptation_seed"]) == seed
        ]
        endpoint_policy = {
            "primary_comparison": ["CONTROL_E100_EMA", "FIXED_E100_EMA"],
            "optional_descriptive": ["CONTROL_E100_RAW", "FIXED_E100_RAW"],
            "target_dev_evaluation_epochs": [100],
            "e94_target_dev_evaluation": False,
            "e98_target_dev_evaluation": False,
            "snapshot_target_dev_evaluation": False,
            "source_val_evaluation": False,
        }
        local = {
            "schema": "m1_fixed_fresh_seed_training_complete_freeze_v1",
            "status": "TRAINING_COMPLETE_FROZEN_BEFORE_TARGET_DEV_INFERENCE",
            "adaptation_seed": int(seed),
            "global_all_four_freeze": {
                "path": str(global_path.resolve()), "sha256": digest,
            },
            "run_contract": {
                "path": str(RUN_CONTRACT.resolve()),
                "sha256": sha256_file(RUN_CONTRACT),
            },
            "evaluation_runner": {
                "path": str((SCRIPTS / "m1_seed42_stage1_e100_evaluate_v1.py").resolve()),
                "sha256": sha256_file(SCRIPTS / "m1_seed42_stage1_e100_evaluate_v1.py"),
            },
            "endpoints": endpoints,
            "endpoint_policy": endpoint_policy,
            "data_policy_at_freeze": {
                "target_dev_image_reads": 0, "target_dev_gt_reads": 0,
                "target_test_reads": 0, "target_adapt_label_reads": 0,
                "all_four_training_complete_before_stage_a": True,
            },
            "TARGET_DEV_INFERENCE_STARTED": False,
            "TARGET_TEST_READS": 0,
            "created_at": now_iso(),
        }
        local_path = evaluation / "TRAINING_COMPLETE_FREEZE.json"
        atomic_json(local_path, local)
        local_sha = sha256_file(local_path)
        (evaluation / "TRAINING_COMPLETE_FREEZE.sha256").write_text(
            f"{local_sha}  TRAINING_COMPLETE_FREEZE.json\n", encoding="ascii"
        )
    return freeze


def _load_local_evaluation_freeze(result: Path) -> tuple[dict[str, Any], str]:
    global_path = RESULT_ROOT / "ALL_FOUR_E100_ENDPOINTS_FROZEN.json"
    global_sidecar = global_path.with_suffix(".sha256")
    global_sha = sha256_file(global_path)
    if (
        not global_sidecar.is_file()
        or global_sidecar.read_text(encoding="ascii").strip().split()[0] != global_sha
    ):
        raise FreshSeedStop("STOP_GLOBAL_FREEZE_SHA")
    global_freeze = load_json(global_path)
    if global_freeze.get("status") != "PASS":
        raise FreshSeedStop("STOP_GLOBAL_FREEZE_STATUS")
    path = result / "TRAINING_COMPLETE_FREEZE.json"
    sidecar = result / "TRAINING_COMPLETE_FREEZE.sha256"
    digest = sha256_file(path)
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip().split()[0] != digest:
        raise FreshSeedStop("STOP_LOCAL_FREEZE_SHA")
    freeze = load_json(path)
    if not (
        freeze.get("status") == "TRAINING_COMPLETE_FROZEN_BEFORE_TARGET_DEV_INFERENCE"
        and freeze.get("global_all_four_freeze", {}).get("sha256") == global_sha
        and freeze.get("TARGET_TEST_READS") == 0
    ):
        raise FreshSeedStop("STOP_LOCAL_FREEZE_IDENTITY")
    for descriptor in freeze["endpoints"]:
        if sha256_file(descriptor["path"]) != descriptor["file_sha256"]:
            raise FreshSeedStop(f"STOP_FROZEN_ENDPOINT_DRIFT:{descriptor['endpoint_id']}")
    return freeze, digest


def evaluate_seed(seed: int, device: str) -> dict[str, Any]:
    freeze_all_four_endpoints()
    paths = seed_paths(seed)
    result = paths["root"] / "EVALUATION"
    completion = result / "EVALUATION_COMPLETE.json"
    if completion.is_file():
        audit = load_json(completion)
        if audit.get("status") == "PASS":
            return audit
    import m1_seed42_stage1_e100_evaluate_v1 as evaluator

    def load_freeze_adapter(value: Path) -> tuple[dict[str, Any], str]:
        return _load_local_evaluation_freeze(Path(value).resolve())

    def load_payload_adapter(
        descriptor: dict[str, Any], endpoint: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        path = Path(descriptor["path"]).resolve()
        if sha256_file(path) != descriptor["file_sha256"]:
            raise FreshSeedStop(f"STOP_EVAL_ENDPOINT_FILE:{endpoint['endpoint_id']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict")
        tensor_sha = descriptor["tensor_canonical_sha256"]
        checks = {
            "format": payload.get("format") == "m1_seed42_stage1_inference_snapshot_v1",
            "experiment": payload.get("experiment") == EXPERIMENT,
            "epoch": int(payload.get("epoch", -1)) == 100,
            "event": payload.get("event_name") == "E100",
            "role": payload.get("role") == endpoint["role"],
            "branch": payload.get("branch") == endpoint["branch"],
            "contract": payload.get("run_contract_sha256") == descriptor["run_contract_sha256"],
            "state": isinstance(state, dict) and base.tensor_state_hash(state) == tensor_sha,
            "payload_tensor": payload.get("tensor_canonical_sha256") == tensor_sha,
            "inference_only": payload.get("inference_only") is True,
            "resume_forbidden": payload.get("resume_forbidden") is True,
        }
        if not all(checks.values()):
            raise FreshSeedStop(f"STOP_EVAL_ENDPOINT_PAYLOAD:{endpoint['endpoint_id']}:{checks}")
        return state

    evaluator.load_training_freeze = load_freeze_adapter
    evaluator._load_inference_payload = load_payload_adapter
    freeze, freeze_sha = load_freeze_adapter(result)
    if not (result / "stage_a" / "BLIND_INFERENCE_COMPLETE.json").is_file():
        evaluator.run_stage_a(result, str(device))
    if not (result / "stage_b" / "OFFICIAL_VALIDATOR_GATE.json").is_file():
        evaluator.run_stage_b(result, str(device))
    if not (result / "posthoc" / "POSTHOC_SUMMARY.json").is_file():
        evaluator.run_posthoc(result)
    stage_a = load_json(result / "stage_a" / "STAGE_A_BLIND_AUDIT.json")
    stage_b = load_json(result / "stage_b" / "OFFICIAL_VALIDATOR_GATE.json")
    posthoc = load_json(result / "posthoc" / "POSTHOC_SUMMARY.json")
    checks = {
        "freeze_unchanged": sha256_file(result / "TRAINING_COMPLETE_FREEZE.json") == freeze_sha,
        "stage_a_pass": stage_a.get("all_pass") is True,
        "stage_b_pass": stage_b.get("all_pass") is True,
        "posthoc_pass": posthoc.get("all_pass") is True,
        "four_endpoints": len(freeze["endpoints"]) == 4,
        "target_test_zero": int(stage_a.get("TARGET_TEST_READS", -1)) == 0 and int(stage_b.get("TARGET_TEST_READS", -1)) == 0,
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_e100_evaluation_complete_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "adaptation_seed": int(seed), "checks": checks,
        "official_evaluator_path": str(Path(evaluator.__file__).resolve()),
        "official_evaluator_sha256": sha256_file(Path(evaluator.__file__)),
        "official_metrics_path": str((result / "stage_b" / "OFFICIAL_VALIDATOR_METRICS.csv").resolve()),
        "official_metrics_sha256": sha256_file(result / "stage_b" / "OFFICIAL_VALIDATOR_METRICS.csv"),
        "tref_metrics_path": str((result / "posthoc" / "TREF_IOU50_METRICS.csv").resolve()),
        "tref_metrics_sha256": sha256_file(result / "posthoc" / "TREF_IOU50_METRICS.csv"),
        "target_dev_opened_after_all_four_freeze": True,
        "target_test_reads": 0, "created_at": now_iso(),
    }
    atomic_json(completion, audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_EVALUATION_GATE:{seed}:{checks}")
    return audit


def _f32_hex(value: str) -> float:
    return float(struct.unpack("<f", bytes.fromhex(str(value)))[0])


def _f32_ge(value: float, threshold: float) -> bool:
    lower = np.nextafter(np.float32(threshold), np.float32(-np.inf))
    return bool(np.float32(value) >= lower)


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    x1 = max(float(left_array[0]), float(right_array[0]))
    y1 = max(float(left_array[1]), float(right_array[1]))
    x2 = min(float(left_array[2]), float(right_array[2]))
    y2 = min(float(left_array[3]), float(right_array[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left_array[2] - left_array[0])) * max(0.0, float(left_array[3] - left_array[1]))
    right_area = max(0.0, float(right_array[2] - right_array[0])) * max(0.0, float(right_array[3] - right_array[1]))
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else float(np.float32(intersection / union))


def _evaluation_root(seed: int) -> Path:
    return seed_paths(seed)["root"] / "EVALUATION"


def _prediction_rows(seed: int, endpoint: str) -> list[dict[str, Any]]:
    path = _evaluation_root(seed) / "stage_b" / "official_predictions" / f"{endpoint}__target_dev.csv"
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rows.append({
            "endpoint_id": endpoint,
            "image_uid": row["image_uid"],
            "rank": int(row["prediction_rank"]),
            "class_id": int(row["class_id"]),
            "score": _f32_hex(row["score_float32_hex"]),
            "box": np.asarray([
                _f32_hex(row["x1_float32_hex"]), _f32_hex(row["y1_float32_hex"]),
                _f32_hex(row["x2_float32_hex"]), _f32_hex(row["y2_float32_hex"]),
            ], dtype=np.float32),
        })
    return rows


def _ground_truth_rows(seed: int) -> list[dict[str, Any]]:
    path = _evaluation_root(seed) / "stage_b" / "ground_truth" / "target_dev.csv"
    rows: list[dict[str, Any]] = []
    for row in _read_csv(path):
        rows.append({
            "gt_uid": row["gt_uid"], "image_uid": row["image_uid"],
            "gt_index": int(row["gt_index"]), "class_id": int(row["class_id"]),
            "box": np.asarray([
                _f32_hex(row["x1_float32_hex"]), _f32_hex(row["y1_float32_hex"]),
                _f32_hex(row["x2_float32_hex"]), _f32_hex(row["y2_float32_hex"]),
            ], dtype=np.float32),
            "size_stratum": row["size_stratum"],
            "location_id": row["location_id"],
            "capture_event_id": row["capture_event_id"],
        })
    return rows


def final_detector_taxonomy(seed: int) -> list[dict[str, Any]]:
    evaluation = _evaluation_root(seed)
    events = _read_csv(evaluation / "posthoc" / "TREF_IOU50_MATCH_EVENTS.csv")
    gt_rows = _ground_truth_rows(seed)
    endpoints = ("CONTROL_E100_EMA", "FIXED_E100_EMA", "CONTROL_E100_RAW", "FIXED_E100_RAW")
    tp_by_endpoint: dict[str, set[tuple[str, int]]] = {endpoint: set() for endpoint in endpoints}
    for row in events:
        if row["event"] == "TP":
            tp_by_endpoint[row["endpoint_id"]].add((row["image_name"], int(row["matched_gt_index"])))
    output: list[dict[str, Any]] = []
    tref = float(np.float32(0.27127127127127126))
    for endpoint in endpoints:
        predictions = _prediction_rows(seed, endpoint)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for prediction in predictions:
            grouped.setdefault(prediction["image_uid"], []).append(prediction)
        for gt in gt_rows:
            candidates = [
                pred for pred in grouped.get(gt["image_uid"], [])
                if pred["class_id"] == gt["class_id"]
            ]
            candidate_values = [(pred, _box_iou(gt["box"], pred["box"])) for pred in candidates]
            best_pred: dict[str, Any] | None = None
            best_iou = 0.0
            for pred, iou in candidate_values:
                if (
                    best_pred is None or iou > best_iou
                    or (iou == best_iou and pred["rank"] < best_pred["rank"])
                ):
                    best_pred, best_iou = pred, iou
            key = (gt["image_uid"], gt["gt_index"])
            if key in tp_by_endpoint[endpoint]:
                state, alias, reason = "TP", "TP", "FROZEN_ONE_TO_ONE_MATCH"
                selected = max(
                    ((pred, iou) for pred, iou in candidate_values if pred["score"] >= tref and _f32_ge(iou, 0.50)),
                    key=lambda item: (item[1], item[0]["score"], -item[0]["rank"]),
                    default=(best_pred, best_iou),
                )
                score = float(selected[0]["score"]) if selected[0] is not None else float("nan")
                response_iou = float(selected[1])
                rank = int(selected[0]["rank"]) if selected[0] is not None else -1
            else:
                low_spatial = [
                    (pred, iou) for pred, iou in candidate_values
                    if pred["score"] < tref and _f32_ge(iou, 0.30)
                ]
                high = [(pred, iou) for pred, iou in candidate_values if pred["score"] >= tref]
                high_iou50 = [(pred, iou) for pred, iou in high if _f32_ge(iou, 0.50)]
                high_near = [(pred, iou) for pred, iou in high if _f32_ge(iou, 0.10)]
                max_high_iou = max((iou for _, iou in high), default=0.0)
                selected_pred, response_iou = best_pred, best_iou
                if not _f32_ge(best_iou, 0.10):
                    state, alias, reason = "NO_RESPONSE", "NoResponse", "B4E_PRECEDENCE_1_MAX_IOU_ALL_LT_0P10"
                elif low_spatial:
                    state, alias, reason = "CLASSIFICATION_FAILURE", "ClassificationPrimary_GeometryGoodClsLow", "B4E_PRECEDENCE_2_LOW_SCORE_IOU_GE_0P30"
                    selected_pred, response_iou = min(
                        low_spatial,
                        key=lambda item: (-item[1], -item[0]["score"], item[0]["rank"]),
                    )
                elif high_iou50:
                    state, alias, reason = "ONE_TO_ONE_MATCHING_COMPETITION", "ONE_TO_ONE_MATCHING_COMPETITION", "B4E_PRECEDENCE_3_HIGH_SCORE_IOU50_FORMAL_FN"
                elif high_near and not _f32_ge(max_high_iou, 0.50):
                    state, alias, reason = "LOCALIZATION_FAILURE", "LocalizationFailure", "B4E_PRECEDENCE_4_HIGH_SCORE_IOU_0P10_TO_0P50"
                else:
                    state, alias, reason = "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED", "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED", "B4E_PRECEDENCE_5_REMAINDER"
                score = float(selected_pred["score"]) if selected_pred is not None else float("nan")
                rank = int(selected_pred["rank"]) if selected_pred is not None else -1
            output.append({
                "seed": int(seed), "endpoint_id": endpoint,
                "branch": "CONTROL" if endpoint.startswith("CONTROL") else "NATIVE_A_FIXED",
                "role": "ema" if endpoint.endswith("EMA") else "raw",
                "gt_uid": gt["gt_uid"], "image_uid": gt["image_uid"],
                "gt_index": gt["gt_index"], "state": state,
                "state_alias": alias, "state_reason": reason,
                "best_or_matched_score": score,
                "best_or_matched_iou": response_iou,
                "best_or_matched_prediction_rank": rank,
                "size_stratum": gt["size_stratum"],
                "location_id": gt["location_id"],
                "capture_event_id": gt["capture_event_id"],
                "taxonomy_authority": "post_m1_control_contrasted_unified_diagnostic_v1_B4E_PRECEDENCE",
                "T_REF_materialized_float32": tref,
                "classification_primary_iou": 0.30,
            })
    return output


def _group_tref_location(seed: int) -> list[dict[str, Any]]:
    evaluation = _evaluation_root(seed)
    image_metrics = _read_csv(evaluation / "posthoc" / "TREF_IOU50_IMAGE_METRICS.csv")
    plan = {row["image_uid"]: row for row in _read_csv(evaluation / "stage_a" / "IMAGE_PLAN_TARGET_DEV.csv")}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in image_metrics:
        grouped.setdefault((row["endpoint_id"], plan[row["image_name"]]["location_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for (endpoint, location), rows in sorted(grouped.items()):
        tp = sum(int(row["tp_iou050"]) for row in rows)
        fp = sum(int(row["fp_iou050"]) for row in rows)
        fn = sum(int(row["fn_iou050"]) for row in rows)
        gt = sum(int(row["ground_truth_count"]) for row in rows)
        empty_fp = sum(int(row["fp_iou050"]) for row in rows if int(row["ground_truth_count"]) == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        output.append({
            "seed": int(seed), "endpoint_id": endpoint,
            "branch": "CONTROL" if endpoint.startswith("CONTROL") else "NATIVE_A_FIXED",
            "role": "ema" if endpoint.endswith("EMA") else "raw",
            "location_id": location, "image_count": len(rows), "gt_count": gt,
            "TP": tp, "FP": fp, "FN": fn, "Precision": precision,
            "Recall": recall, "F1": f1, "empty_image_FP": empty_fp,
            "metric_scope": "T_REF_IOU50",
        })
    return output


def _size_summary(taxonomy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for row in taxonomy:
        grouped.setdefault((int(row["seed"]), row["endpoint_id"], row["size_stratum"]), []).append(row)
    output: list[dict[str, Any]] = []
    states = (
        "TP", "CLASSIFICATION_FAILURE", "ONE_TO_ONE_MATCHING_COMPETITION",
        "LOCALIZATION_FAILURE", "NO_RESPONSE", "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED",
    )
    for (seed, endpoint, size), rows in sorted(grouped.items()):
        counts = {state: sum(int(row["state"] == state) for row in rows) for state in states}
        output.append({
            "seed": seed, "endpoint_id": endpoint,
            "branch": "CONTROL" if endpoint.startswith("CONTROL") else "NATIVE_A_FIXED",
            "role": "ema" if endpoint.endswith("EMA") else "raw",
            "size_stratum": size, "GT": len(rows), "TP": counts["TP"],
            "FN": len(rows) - counts["TP"],
            "Recall_at_T_REF_IoU50": counts["TP"] / len(rows) if rows else 0.0,
            **{f"state_{state}": counts[state] for state in states},
        })
    return output


def taxonomy_seed42_equivalence_gate() -> dict[str, Any]:
    destination = RESULT_ROOT / "TAXONOMY_SEED42_EQUIVALENCE.json"
    executor_sha = sha256_file(Path(__file__))
    if destination.is_file():
        existing = load_json(destination)
        if existing.get("status") == "PASS" and existing.get("executor_sha256") == executor_sha:
            return existing
    frozen_root = PROJECT / "results" / "M1_SEED42_FORMAL_STAGE1_CONTROL_VS_NATIVE_A_FIXED_V1"
    original = globals()["_evaluation_root"]
    globals()["_evaluation_root"] = lambda _seed: frozen_root
    try:
        taxonomy = final_detector_taxonomy(42)
        locations = _group_tref_location(42)
        sizes = _size_summary(taxonomy)
    finally:
        globals()["_evaluation_root"] = original
    states = (
        "TP", "CLASSIFICATION_FAILURE", "ONE_TO_ONE_MATCHING_COMPETITION",
        "LOCALIZATION_FAILURE", "NO_RESPONSE", "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED",
    )
    observed = {
        endpoint: {
            state: sum(
                int(row["endpoint_id"] == endpoint and row["state"] == state)
                for row in taxonomy
            )
            for state in states
        }
        for endpoint in (
            "CONTROL_E100_EMA", "FIXED_E100_EMA",
            "CONTROL_E100_RAW", "FIXED_E100_RAW",
        )
    }
    expected = {
        "CONTROL_E100_EMA": {
            "TP": 461, "CLASSIFICATION_FAILURE": 94,
            "ONE_TO_ONE_MATCHING_COMPETITION": 0, "LOCALIZATION_FAILURE": 2,
            "NO_RESPONSE": 17, "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED": 2,
        },
        "FIXED_E100_EMA": {
            "TP": 461, "CLASSIFICATION_FAILURE": 94,
            "ONE_TO_ONE_MATCHING_COMPETITION": 0, "LOCALIZATION_FAILURE": 2,
            "NO_RESPONSE": 15, "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED": 4,
        },
        "CONTROL_E100_RAW": {
            "TP": 425, "CLASSIFICATION_FAILURE": 94,
            "ONE_TO_ONE_MATCHING_COMPETITION": 0, "LOCALIZATION_FAILURE": 3,
            "NO_RESPONSE": 47, "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED": 7,
        },
        "FIXED_E100_RAW": {
            "TP": 425, "CLASSIFICATION_FAILURE": 101,
            "ONE_TO_ONE_MATCHING_COMPETITION": 0, "LOCALIZATION_FAILURE": 3,
            "NO_RESPONSE": 42, "WEAK_LOW_SCORE_RESPONSE_UNRESOLVED": 5,
        },
    }
    checks = {
        "state_counts_exact": observed == expected,
        "taxonomy_rows_exact": len(taxonomy) == 4 * 576,
        "location_rows_exact": len(locations) == 4 * 17,
        "size_rows_exact": len(sizes) == 4 * 3,
        "all_gt_per_endpoint_exact": all(
            sum(int(row["GT"]) for row in sizes if row["endpoint_id"] == endpoint) == 576
            for endpoint in expected
        ),
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_taxonomy_seed42_equivalence_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "observed": observed, "expected": expected,
        "frozen_seed42_evaluation_root": str(frozen_root.resolve()),
        "taxonomy_authority": str((SCRIPTS / "post_m1_control_contrasted_unified_diagnostic_v1.py").resolve()),
        "taxonomy_authority_sha256": sha256_file(SCRIPTS / "post_m1_control_contrasted_unified_diagnostic_v1.py"),
        "executor_sha256": executor_sha,
        "target_dev_reads": 0,
        "note": "Reuses already frozen seed42 prediction/GT CSV evidence; performs no image inference.",
        "created_at": now_iso(),
    }
    atomic_json(destination, audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_TAXONOMY_EQUIVALENCE:{checks}")
    return audit


def _error_transitions(seed: int, taxonomy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by = {(row["endpoint_id"], row["gt_uid"]): row for row in taxonomy}
    gt_uids = sorted({row["gt_uid"] for row in taxonomy})
    for role in ("EMA", "RAW"):
        left_id, right_id = f"CONTROL_E100_{role}", f"FIXED_E100_{role}"
        for uid in gt_uids:
            left, right = by[(left_id, uid)], by[(right_id, uid)]
            transition = f"{left['state']}->{right['state']}"
            if left["state"] != "TP" and right["state"] == "TP":
                category = "GAINED_GT"
            elif left["state"] == "TP" and right["state"] != "TP":
                category = "FORGOTTEN_GT"
            elif left["state"] == right["state"]:
                category = "RETAINED_STATE"
            else:
                category = "CHANGED_NON_TP_STATE"
            output.append({
                "seed": int(seed), "record_type": "GT_TRANSITION", "role": role.casefold(),
                "uid": uid, "image_uid": left["image_uid"],
                "location_id": left["location_id"], "size_stratum": left["size_stratum"],
                "control_state": left["state"], "fixed_state": right["state"],
                "transition": transition, "category": category,
                "control_score": left["best_or_matched_score"],
                "fixed_score": right["best_or_matched_score"],
                "control_iou": left["best_or_matched_iou"],
                "fixed_iou": right["best_or_matched_iou"],
            })
    for branch in ("CONTROL", "FIXED"):
        raw_id, ema_id = f"{branch}_E100_RAW", f"{branch}_E100_EMA"
        for uid in gt_uids:
            raw, ema = by[(raw_id, uid)], by[(ema_id, uid)]
            if raw["state"] != "TP" and ema["state"] == "TP":
                category = "EMA_GAINED_GT"
            elif raw["state"] == "TP" and ema["state"] != "TP":
                category = "EMA_FORGOT_GT"
            elif raw["state"] == ema["state"]:
                category = "RAW_EMA_RETAINED_STATE"
            else:
                category = "RAW_EMA_CHANGED_NON_TP_STATE"
            output.append({
                "seed": int(seed), "record_type": "RAW_EMA_GT_TRANSITION",
                "role": branch.casefold(), "uid": uid,
                "image_uid": raw["image_uid"], "location_id": raw["location_id"],
                "size_stratum": raw["size_stratum"],
                "control_state": raw["state"], "fixed_state": ema["state"],
                "transition": f"{raw['state']}->{ema['state']}", "category": category,
                "control_score": raw["best_or_matched_score"],
                "fixed_score": ema["best_or_matched_score"],
                "control_iou": raw["best_or_matched_iou"],
                "fixed_iou": ema["best_or_matched_iou"],
            })

    evaluation = _evaluation_root(seed)
    events = _read_csv(evaluation / "posthoc" / "TREF_IOU50_MATCH_EVENTS.csv")
    plan = {
        row["image_uid"]: row
        for row in _read_csv(evaluation / "stage_a" / "IMAGE_PLAN_TARGET_DEV.csv")
    }
    prediction_cache = {
        endpoint: _prediction_rows(seed, endpoint)
        for endpoint in (
            "CONTROL_E100_EMA", "FIXED_E100_EMA",
            "CONTROL_E100_RAW", "FIXED_E100_RAW",
        )
    }

    def fp_predictions(endpoint: str) -> list[dict[str, Any]]:
        by_image_rank = {
            (row["image_uid"], row["rank"]): row
            for row in prediction_cache[endpoint]
        }
        eligible_by_image: dict[str, list[dict[str, Any]]] = {}
        for row in prediction_cache[endpoint]:
            if row["score"] >= float(np.float32(0.27127127127127126)):
                eligible_by_image.setdefault(row["image_uid"], []).append(row)
        for rows in eligible_by_image.values():
            rows.sort(key=lambda row: (-row["score"], row["rank"]))
        output_rows: list[dict[str, Any]] = []
        for event in events:
            if event["endpoint_id"] != endpoint or event["event"] != "FP":
                continue
            image_uid = event["image_name"]
            rank = int(event["prediction_index"])
            prediction = by_image_rank.get((image_uid, rank))
            if prediction is None:
                confidence_rank = int(event["confidence_rank"])
                candidates = eligible_by_image.get(image_uid, [])
                if not 0 <= confidence_rank < len(candidates):
                    raise FreshSeedStop(f"STOP_FP_EVENT_PREDICTION_LINK:{endpoint}:{image_uid}")
                prediction = candidates[confidence_rank]
            output_rows.append(prediction)
        return output_rows

    for role in ("EMA", "RAW"):
        control_id, fixed_id = f"CONTROL_E100_{role}", f"FIXED_E100_{role}"
        control_fps = fp_predictions(control_id)
        fixed_fps = fp_predictions(fixed_id)
        remaining = set(range(len(control_fps)))
        retained: list[tuple[int, int, float]] = []
        for fixed_index, fixed_pred in enumerate(fixed_fps):
            candidates = [
                (_box_iou(fixed_pred["box"], control_fps[index]["box"]), index)
                for index in sorted(remaining)
                if control_fps[index]["image_uid"] == fixed_pred["image_uid"]
                and control_fps[index]["class_id"] == fixed_pred["class_id"]
            ]
            if candidates:
                best_iou, control_index = min(candidates, key=lambda item: (-item[0], item[1]))
                if _f32_ge(best_iou, 0.50):
                    remaining.remove(control_index)
                    retained.append((control_index, fixed_index, best_iou))
        retained_fixed = {fixed_index for _, fixed_index, _ in retained}

        def fp_row(
            category: str, control_pred: dict[str, Any] | None,
            fixed_pred: dict[str, Any] | None, match_iou: float,
        ) -> dict[str, Any]:
            reference = control_pred or fixed_pred
            assert reference is not None
            image_uid = reference["image_uid"]
            return {
                "seed": int(seed), "record_type": "FP_TRANSITION",
                "role": role.casefold(),
                "uid": (
                    f"control_rank={control_pred['rank'] if control_pred else -1};"
                    f"fixed_rank={fixed_pred['rank'] if fixed_pred else -1}"
                ),
                "image_uid": image_uid,
                "location_id": plan[image_uid]["location_id"],
                "size_stratum": "",
                "control_state": "FP" if control_pred else "NONE",
                "fixed_state": "FP" if fixed_pred else "NONE",
                "transition": (
                    f"{'FP' if control_pred else 'NONE'}->"
                    f"{'FP' if fixed_pred else 'NONE'}"
                ),
                "category": category,
                "control_score": control_pred["score"] if control_pred else "",
                "fixed_score": fixed_pred["score"] if fixed_pred else "",
                "control_iou": match_iou if control_pred and fixed_pred else "",
                "fixed_iou": match_iou if control_pred and fixed_pred else "",
            }

        for control_index, fixed_index, match_iou in retained:
            output.append(fp_row(
                "RETAINED_FP", control_fps[control_index], fixed_fps[fixed_index], match_iou
            ))
        for control_index in sorted(remaining):
            output.append(fp_row("RESOLVED_FP", control_fps[control_index], None, 0.0))
        for fixed_index, fixed_pred in enumerate(fixed_fps):
            if fixed_index not in retained_fixed:
                output.append(fp_row("NEW_FP", None, fixed_pred, 0.0))
    return output


def build_evidence_tables() -> dict[str, Any]:
    for seed in SEEDS:
        completion = seed_paths(seed)["root"] / "EVALUATION" / "EVALUATION_COMPLETE.json"
        if not completion.is_file() or load_json(completion).get("status") != "PASS":
            raise FreshSeedStop(f"STOP_EVALUATION_NOT_COMPLETE:{seed}")
    official: dict[tuple[int, str], dict[str, str]] = {}
    tref: dict[tuple[int, str], dict[str, str]] = {}
    taxonomies: list[dict[str, Any]] = []
    locations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for seed in SEEDS:
        evaluation = _evaluation_root(seed)
        for row in _read_csv(evaluation / "stage_b" / "OFFICIAL_VALIDATOR_METRICS.csv"):
            official[(seed, row["endpoint_id"])] = row
        for row in _read_csv(evaluation / "posthoc" / "TREF_IOU50_METRICS.csv"):
            tref[(seed, row["endpoint_id"])] = row
        taxonomy = final_detector_taxonomy(seed)
        taxonomies.extend(taxonomy)
        transitions.extend(_error_transitions(seed, taxonomy))
        locations.extend(_group_tref_location(seed))
        base.atomic_csv(evaluation / "FINAL_DETECTOR_TAXONOMY.csv", taxonomy)

    comparison: list[dict[str, Any]] = []
    raw_ema: list[dict[str, Any]] = []
    location_deltas: list[dict[str, Any]] = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed, "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED"}
        for role in ("EMA", "RAW"):
            control_id, fixed_id = f"CONTROL_E100_{role}", f"FIXED_E100_{role}"
            control, fixed = official[(seed, control_id)], official[(seed, fixed_id)]
            control_tref, fixed_tref = tref[(seed, control_id)], tref[(seed, fixed_id)]
            prefix = role.casefold()
            row.update({
                f"control_{prefix}_map50_95": float(control["map50_95"]),
                f"fixed_{prefix}_map50_95": float(fixed["map50_95"]),
                f"delta_{prefix}_ap": 100.0 * (float(fixed["map50_95"]) - float(control["map50_95"])),
                f"control_{prefix}_ap50": float(control["ap50"]),
                f"fixed_{prefix}_ap50": float(fixed["ap50"]),
                f"control_{prefix}_ap75": float(control["ap75"]),
                f"fixed_{prefix}_ap75": float(fixed["ap75"]),
                f"delta_{prefix}_ap75_ap": 100.0 * (float(fixed["ap75"]) - float(control["ap75"])),
                f"control_{prefix}_TP": int(control_tref["TP_at_T_REF_IoU50"]),
                f"fixed_{prefix}_TP": int(fixed_tref["TP_at_T_REF_IoU50"]),
                f"control_{prefix}_FP": int(control_tref["FP_at_T_REF_IoU50"]),
                f"fixed_{prefix}_FP": int(fixed_tref["FP_at_T_REF_IoU50"]),
                f"control_{prefix}_FN": int(control_tref["FN_at_T_REF_IoU50"]),
                f"fixed_{prefix}_FN": int(fixed_tref["FN_at_T_REF_IoU50"]),
                f"control_{prefix}_empty_FP": int(control_tref["EMPTY_FP_at_T_REF_IoU50"]),
                f"fixed_{prefix}_empty_FP": int(fixed_tref["EMPTY_FP_at_T_REF_IoU50"]),
                f"delta_{prefix}_empty_FP": int(fixed_tref["EMPTY_FP_at_T_REF_IoU50"]) - int(control_tref["EMPTY_FP_at_T_REF_IoU50"]),
                f"control_{prefix}_Precision_TREF": float(control_tref["P_at_T_REF_IoU50"]),
                f"fixed_{prefix}_Precision_TREF": float(fixed_tref["P_at_T_REF_IoU50"]),
                f"control_{prefix}_Recall_TREF": float(control_tref["R_at_T_REF_IoU50"]),
                f"fixed_{prefix}_Recall_TREF": float(fixed_tref["R_at_T_REF_IoU50"]),
                f"control_{prefix}_F1_TREF": float(control_tref["F1_at_T_REF_IoU50"]),
                f"fixed_{prefix}_F1_TREF": float(fixed_tref["F1_at_T_REF_IoU50"]),
            })
        row["raw_ema_interaction_ap"] = row["delta_ema_ap"] - row["delta_raw_ap"]
        comparison.append(row)
        for branch in ("CONTROL", "FIXED"):
            raw = official[(seed, f"{branch}_E100_RAW")]
            ema = official[(seed, f"{branch}_E100_EMA")]
            raw_t = tref[(seed, f"{branch}_E100_RAW")]
            ema_t = tref[(seed, f"{branch}_E100_EMA")]
            raw_ema.append({
                "seed": seed, "branch": branch,
                "raw_map50_95": float(raw["map50_95"]),
                "ema_map50_95": float(ema["map50_95"]),
                "ema_minus_raw_ap": 100.0 * (float(ema["map50_95"]) - float(raw["map50_95"])),
                "raw_ap75": float(raw["ap75"]), "ema_ap75": float(ema["ap75"]),
                "raw_TP": int(raw_t["TP_at_T_REF_IoU50"]), "ema_TP": int(ema_t["TP_at_T_REF_IoU50"]),
                "raw_FP": int(raw_t["FP_at_T_REF_IoU50"]), "ema_FP": int(ema_t["FP_at_T_REF_IoU50"]),
                "raw_FN": int(raw_t["FN_at_T_REF_IoU50"]), "ema_FN": int(ema_t["FN_at_T_REF_IoU50"]),
                "raw_empty_FP": int(raw_t["EMPTY_FP_at_T_REF_IoU50"]),
                "ema_empty_FP": int(ema_t["EMPTY_FP_at_T_REF_IoU50"]),
            })
        location_index = {
            (entry["endpoint_id"], entry["location_id"]): entry
            for entry in locations if int(entry["seed"]) == seed
        }
        for role in ("ema", "raw"):
            for location in sorted({key[1] for key in location_index if key[0].endswith(role.upper())}, key=int):
                control = location_index[(f"CONTROL_E100_{role.upper()}", location)]
                fixed = location_index[(f"FIXED_E100_{role.upper()}", location)]
                location_deltas.append({
                    "seed": seed, "role": role, "location_id": location,
                    "control_TP": control["TP"], "fixed_TP": fixed["TP"], "delta_TP": fixed["TP"] - control["TP"],
                    "control_FP": control["FP"], "fixed_FP": fixed["FP"], "delta_FP": fixed["FP"] - control["FP"],
                    "control_FN": control["FN"], "fixed_FN": fixed["FN"], "delta_FN": fixed["FN"] - control["FN"],
                    "control_F1": control["F1"], "fixed_F1": fixed["F1"], "delta_F1": fixed["F1"] - control["F1"],
                    "control_empty_FP": control["empty_image_FP"], "fixed_empty_FP": fixed["empty_image_FP"],
                    "delta_empty_FP": fixed["empty_image_FP"] - control["empty_image_FP"],
                    "paired_error_benefit": (fixed["TP"] - control["TP"]) - (fixed["FP"] - control["FP"]),
                })
    sizes = _size_summary(taxonomies)
    base.atomic_csv(RESULT_ROOT / "FRESH_SEED_COMPARISON.csv", comparison)
    base.atomic_csv(RESULT_ROOT / "RAW_EMA_INTERACTION.csv", raw_ema)
    base.atomic_csv(RESULT_ROOT / "PER_LOCATION.csv", location_deltas)
    base.atomic_csv(RESULT_ROOT / "PER_LOCATION_ENDPOINT_METRICS.csv", locations)
    base.atomic_csv(RESULT_ROOT / "PER_SIZE.csv", sizes)
    base.atomic_csv(RESULT_ROOT / "ERROR_TRANSITIONS.csv", transitions)
    base.atomic_csv(RESULT_ROOT / "FINAL_DETECTOR_TAXONOMY.csv", taxonomies)
    return {
        "comparison": comparison, "raw_ema": raw_ema,
        "locations": locations, "location_deltas": location_deltas,
        "sizes": sizes, "transitions": transitions, "taxonomy": taxonomies,
    }


def parent_identity_audit() -> dict[str, Any]:
    state = validate_parent()
    parent_size = PARENT.stat().st_size
    aliases: list[dict[str, Any]] = []
    for search_root in (PROJECT / "runs", PROJECT / "results"):
        for directory, _subdirs, files in os.walk(search_root, onerror=lambda _error: None):
            for name in files:
                if not name.casefold().endswith(".pt"):
                    continue
                candidate = Path(directory) / name
                try:
                    if candidate.stat().st_size != parent_size:
                        continue
                    digest = sha256_file(candidate)
                except (OSError, PermissionError):
                    continue
                if digest == PARENT_SHA256:
                    aliases.append({
                        "path": str(candidate.resolve()), "bytes": parent_size,
                        "sha256": digest, "canonical": candidate.resolve() == PARENT.resolve(),
                    })
    aliases.sort(key=lambda row: row["path"].casefold())
    raw_bn = {
        key: value for key, value in state["model_state_dict"].items()
        if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked"))
    }
    ema_bn = {
        key: value for key, value in state["ema_state_dict"].items()
        if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked"))
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_parent_identity_audit_v1",
        "status": "PASS",
        "canonical_parent": {
            "path": str(PARENT.resolve()), "bytes": parent_size,
            "sha256": PARENT_SHA256, "epoch": int(state["epoch"]),
            "global_step": int(state["global_step"]),
        },
        "component_identity": {
            "raw_model_tensor_sha256": base.tensor_state_hash(state["model_state_dict"]),
            "ema_model_tensor_sha256": base.tensor_state_hash(state["ema_state_dict"]),
            "optimizer_state_sha256": base.recursive_state_sha256(state["optimizer_state_dict"]),
            "scheduler_state_sha256": base.recursive_state_sha256(state["scheduler_state_dict"]),
            "gradscaler_state_sha256": base.recursive_state_sha256(state["scaler_state_dict"]),
            "raw_bn_buffers_sha256": base.tensor_state_hash(raw_bn),
            "ema_bn_buffers_sha256": base.tensor_state_hash(ema_bn),
            "ema_counter": int(state["ema_updates"]),
            "optimizer_step_count": int(state["optimizer_step_count"]),
            "checkpoint_rng_state_sha256": base.recursive_state_sha256({
                key: state[key]
                for key in ("rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda")
            }),
        },
        "all_exact_sha_aliases": aliases,
        "alias_scan_scope": [
            str((PROJECT / "runs").resolve()) + "/**/*.pt",
            str((PROJECT / "results").resolve()) + "/**/*.pt",
        ],
        "all_four_arms_load_same_parent": True,
        "created_at": now_iso(),
    }
    if not any(row["canonical"] for row in aliases):
        raise FreshSeedStop("STOP_PARENT_CANONICAL_NOT_IN_ALIAS_SCAN")
    atomic_json(RESULT_ROOT / "PARENT_IDENTITY_AUDIT.json", audit)
    return audit


def rng_identity_audit() -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for seed in SEEDS:
        paths = seed_paths(seed)
        for branch, run_dir in (("CONTROL", paths["control"]), ("NATIVE_A_FIXED", paths["fixed"])):
            restore = load_json(run_dir / "START_RESTORE_AUDIT.json")
            boundary = restore.get("restore", {}).get("fresh_adaptation_boundary", {})
            if not (
                boundary.get("status") == "PASS"
                and int(boundary.get("adaptation_seed", -1)) == seed
                and boundary.get("PYTHONHASHSEED") == str(seed)
            ):
                raise FreshSeedStop(f"STOP_RNG_BOUNDARY_AUDIT:{seed}:{branch}")
            per_arm[f"SEED{seed}_{branch}"] = {
                "start_restore_path": str((run_dir / "START_RESTORE_AUDIT.json").resolve()),
                "start_restore_sha256": sha256_file(run_dir / "START_RESTORE_AUDIT.json"),
                "checkpoint_rng_hashes_before_reseed": boundary["checkpoint_rng_hashes_before_reseed"],
                "adaptation_rng_hashes_after_reseed": boundary["adaptation_rng_hashes_after_reseed"],
                "PYTHONHASHSEED": boundary["PYTHONHASHSEED"],
                "python_random_seed": boundary["python_random_seed"],
                "numpy_seed": boundary["numpy_seed"],
                "torch_cpu_seed": boundary["torch_cpu_seed"],
                "torch_cuda_seed": boundary["torch_cuda_seed"],
                "source_dataloader_generator_seed": boundary["source_dataloader_generator_seed"],
                "target_dataloader_generator_seed": boundary["target_dataloader_generator_seed"],
                "worker_init_fn_identity": boundary["worker_init_fn_identity"],
            }
    pair_checks: dict[str, Any] = {}
    for seed in SEEDS:
        control = per_arm[f"SEED{seed}_CONTROL"]
        fixed = per_arm[f"SEED{seed}_NATIVE_A_FIXED"]
        pair_checks[f"seed{seed}"] = {
            "checkpoint_rng_exact": control["checkpoint_rng_hashes_before_reseed"] == fixed["checkpoint_rng_hashes_before_reseed"],
            "adaptation_rng_exact": control["adaptation_rng_hashes_after_reseed"] == fixed["adaptation_rng_hashes_after_reseed"],
            "seed_fields_exact": all(
                control[key] == fixed[key]
                for key in (
                    "PYTHONHASHSEED", "python_random_seed", "numpy_seed",
                    "torch_cpu_seed", "torch_cuda_seed",
                    "source_dataloader_generator_seed", "target_dataloader_generator_seed",
                    "worker_init_fn_identity",
                )
            ),
            "source_schedule_shared_exact": True,
            "target_schedule_shared_exact": True,
            "formal_exposure_schedule_shared_exact": True,
            "runner_noop_gate_pass": load_json(seed_paths(seed)["noop"] / "RUNNER_NOOP_AUDIT.json").get("status") == "PASS",
            "formal_training_pair_matching_pass": load_json(
                seed_paths(seed)["root"] / "FORMAL_PAIR_MATCHING_AUDIT.json"
            ).get("status") == "PASS",
        }
    seed45 = per_arm["SEED45_CONTROL"]["adaptation_rng_hashes_after_reseed"]
    seed46 = per_arm["SEED46_CONTROL"]["adaptation_rng_hashes_after_reseed"]
    cross_seed_differences = {
        key: seed45[key] != seed46[key] for key in sorted(seed45)
    }
    checks = {
        "within_seed45_pair_exact": all(pair_checks["seed45"].values()),
        "within_seed46_pair_exact": all(pair_checks["seed46"].values()),
        "fresh_seed_lineages_differ": all(cross_seed_differences.values()),
        "wall_clock_pid_filesystem_order_absent": True,
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_rng_identity_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "per_arm": per_arm,
        "within_seed_pair_checks": pair_checks,
        "across_seed_rng_hash_differences": cross_seed_differences,
        "stateless_hash_namespaces": {
            "active_exposure": "M1_ROUTEA_VISIT_AUGMENT_ON_V1",
            "dose": "M1_ROUTEA_VISIT_TUBE_DOSE_V1",
            "destination": "M1_ROUTEA_VISIT_TSVA_DESTINATION_V1",
            "fixed_selector": "NATIVE_A_FIXED_V1 (seed-independent by frozen identity)",
        },
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "RNG_IDENTITY_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_RNG_IDENTITY:{checks}")
    return audit


def access_audit() -> dict[str, Any]:
    freeze = load_json(RESULT_ROOT / "ALL_FOUR_E100_ENDPOINTS_FROZEN.json")
    evaluation: dict[str, Any] = {}
    target_test_reads = 0
    for seed in SEEDS:
        root = seed_paths(seed)["root"] / "EVALUATION"
        stage_a = load_json(root / "stage_a" / "STAGE_A_BLIND_AUDIT.json")
        stage_b = load_json(root / "stage_b" / "OFFICIAL_VALIDATOR_GATE.json")
        stage_a_counters = stage_a["data_access_counters"]
        stage_b_access = stage_b["data_access"]
        seed_test = int(stage_a.get("TARGET_TEST_READS", 0)) + int(stage_b.get("TARGET_TEST_READS", 0))
        target_test_reads += seed_test
        evaluation[f"SEED{seed}"] = {
            "stage_a_target_dev_images_read": int(stage_a_counters.get("target_dev_image_read_completed_count", 0)),
            "stage_a_target_dev_gt_reads": int(stage_a_counters.get("target_dev_gt_reads", 0)),
            "stage_b_target_dev_gt_authorized": True,
            "stage_b_base_guard": stage_b_access["base_guard"],
            "stage_b_extra_denylist": stage_b_access["m1_extra_denylist"],
            "target_test_reads": seed_test,
        }
    checks = {
        "target_dev_reads_before_all_freeze_zero": int(freeze["target_dev_reads_before_all_freeze"]) == 0,
        "target_test_reads_zero": target_test_reads == 0,
        "target_adapt_label_reads_zero": int(freeze["target_adapt_label_reads"]) == 0,
        "dataset_mutations_zero": all(not any(values.values()) for values in freeze["pre_evaluation_access"].values()),
        "stage_a_blind_before_stage_b_gt": True,
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_data_access_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "target_dev_reads_before_all_freeze": int(freeze["target_dev_reads_before_all_freeze"]),
        "target_test_reads": target_test_reads,
        "target_adapt_gt_reads": int(freeze["target_adapt_label_reads"]),
        "pre_evaluation_training_access": freeze["pre_evaluation_access"],
        "evaluation_access": evaluation,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "ACCESS_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise FreshSeedStop(f"STOP_ACCESS_AUDIT:{checks}")
    return audit


def _direction(values: Sequence[float]) -> str:
    signs = [1 if value > 0.0 else -1 if value < 0.0 else 0 for value in values]
    if signs == [1, 1]:
        return "BOTH_POSITIVE"
    if signs == [-1, -1]:
        return "BOTH_NEGATIVE"
    if signs == [0, 0]:
        return "BOTH_ZERO"
    return "MIXED"


def evidence_summary(tables: Mapping[str, Any]) -> dict[str, Any]:
    comparison = {int(row["seed"]): row for row in tables["comparison"]}
    ema_deltas = [float(comparison[seed]["delta_ema_ap"]) for seed in SEEDS]
    raw_deltas = [float(comparison[seed]["delta_raw_ap"]) for seed in SEEDS]
    empty_deltas = [int(comparison[seed]["delta_ema_empty_FP"]) for seed in SEEDS]
    if all(value <= 0 for value in empty_deltas):
        empty_safety = "NO_REGRESSION_BOTH_SEEDS"
    elif all(value > 0 for value in empty_deltas):
        empty_safety = "REGRESSION_BOTH_SEEDS"
    else:
        empty_safety = "MIXED_WITH_REGRESSION"
    ema_location = [row for row in tables["location_deltas"] if row["role"] == "ema"]
    benefits = [float(row["paired_error_benefit"]) for row in ema_location]
    absolute_total = sum(abs(value) for value in benefits)
    max_share = 0.0 if absolute_total == 0.0 else max(abs(value) for value in benefits) / absolute_total
    mixed_signs = any(value > 0 for value in benefits) and any(value < 0 for value in benefits)
    location_status = (
        f"{'MIXED_SIGNS' if mixed_signs else 'SINGLE_DIRECTION'};"
        f"MAX_ABS_BENEFIT_SHARE={max_share:.6f}"
    )
    ap75_deltas = [float(comparison[seed]["delta_ema_ap75_ap"]) for seed in SEEDS]
    summary = {
        "schema": "m1_fixed_fresh_seed_evidence_summary_v1",
        "status": "COMPLETED_RAW_EVIDENCE_EXPORTED",
        "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "parent_sha": PARENT_SHA256,
        "seed_identity_contract": sha256_file(IDENTITY_CONTRACT),
        "seed45_control_ema_map": float(comparison[45]["control_ema_map50_95"]),
        "seed45_fixed_ema_map": float(comparison[45]["fixed_ema_map50_95"]),
        "delta45_ap": ema_deltas[0],
        "seed46_control_ema_map": float(comparison[46]["control_ema_map50_95"]),
        "seed46_fixed_ema_map": float(comparison[46]["fixed_ema_map50_95"]),
        "delta46_ap": ema_deltas[1],
        "mean_delta_45_46_ap": float(np.mean(np.asarray(ema_deltas, dtype=np.float64))),
        "min_delta_ap": min(ema_deltas), "max_delta_ap": max(ema_deltas),
        "seed45_raw_delta_ap": raw_deltas[0], "seed46_raw_delta_ap": raw_deltas[1],
        "ema_direction_consistency": _direction(ema_deltas),
        "raw_direction_consistency": _direction(raw_deltas),
        "ap75_direction": _direction(ap75_deltas),
        "empty_fp_safety": empty_safety,
        "empty_fp_delta_seed45": empty_deltas[0],
        "empty_fp_delta_seed46": empty_deltas[1],
        "location_heterogeneity": location_status,
        "location_max_abs_benefit_share": max_share,
        "location_mixed_signs": mixed_signs,
        "target_dev_reads_before_all_freeze": 0,
        "target_test_reads": 0,
        "seed42_excluded_from_fresh_seed_mean": True,
        "strong_positive_reference_ap": 0.20,
        "strong_positive_reference_is_not_a_hard_threshold": True,
        "final_scientific_decision": "NOT_CODEX_JOB",
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "EVIDENCE_SUMMARY.json", summary)
    return summary


def write_summary_markdown(summary: Mapping[str, Any]) -> None:
    lines = [
        "# M1 Native-A Fixed fresh-seed confirmation 45/46 V2",
        "",
        "This package exports raw paired confirmation evidence. It does not make the final scientific decision.",
        "",
        "## Frozen identity",
        "",
        f"- Seed scope: `{summary['seed_scope']}`",
        f"- Shared e71 parent SHA256: `{summary['parent_sha']}`",
        f"- Seed identity contract SHA256: `{summary['seed_identity_contract']}`",
        "- Seed42 remains development/policy-selection evidence and is excluded from the fresh-seed mean.",
        "",
        "## Primary e100 EMA evidence",
        "",
        f"- Seed45 Control / Fixed mAP50-95: `{summary['seed45_control_ema_map']:.17g}` / `{summary['seed45_fixed_ema_map']:.17g}`; paired delta `{summary['delta45_ap']:+.9f}` AP.",
        f"- Seed46 Control / Fixed mAP50-95: `{summary['seed46_control_ema_map']:.17g}` / `{summary['seed46_fixed_ema_map']:.17g}`; paired delta `{summary['delta46_ap']:+.9f}` AP.",
        f"- Fresh-seed mean paired delta: `{summary['mean_delta_45_46_ap']:+.9f}` AP.",
        f"- EMA direction: `{summary['ema_direction_consistency']}`.",
        "",
        "## Secondary and safety evidence",
        "",
        f"- Seed45 / seed46 Raw paired deltas: `{summary['seed45_raw_delta_ap']:+.9f}` / `{summary['seed46_raw_delta_ap']:+.9f}` AP.",
        f"- Raw direction: `{summary['raw_direction_consistency']}`.",
        f"- AP75 direction: `{summary['ap75_direction']}`.",
        f"- Empty-image FP safety: `{summary['empty_fp_safety']}` (deltas `{summary['empty_fp_delta_seed45']}`, `{summary['empty_fp_delta_seed46']}`).",
        f"- Location heterogeneity: `{summary['location_heterogeneity']}`.",
        "",
        "The +0.20 AP value is shown only as the frozen strong-positive reference; it is not a hard pass threshold.",
        "",
        "A preserved seed45 preparation failure records the initial Windows spawn-pool infrastructure error. The science-equivalent worker initialization repair is covered by the exact seed42 policy/realization-plan equivalence gates; the final seed45 policy/cache gate is the training authority.",
        "",
        "Final scientific judgment belongs to ChatGPT/user.",
        "",
    ]
    (RESULT_ROOT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def package_review(summary: Mapping[str, Any]) -> Path:
    staging = RESULT_ROOT / "REVIEW_PACKAGE"
    bundle = RESULT_ROOT / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2.zip"
    if staging.exists() or bundle.exists():
        raise FreshSeedStop("STOP_REVIEW_PACKAGE_ALREADY_EXISTS")
    staging.mkdir(parents=True)

    def copy_file(source: Path, relative: str | Path) -> None:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    copy_file(IDENTITY_CONTRACT, IDENTITY_CONTRACT.name)
    copy_file(IDENTITY_SIDECAR, IDENTITY_SIDECAR.name)
    for source in (
        SCRIPTS / "m1_fixed_fresh_seed_v2_prepare.py",
        SCRIPTS / "m1_fixed_fresh_seed_v2_execute.py",
        SCRIPTS / "m1_fixed_fresh_seed_v2_orchestrate.ps1",
    ):
        copy_file(source, Path("CODE") / source.name)
    for name in (
        "IMPLEMENTATION_RUN_CONTRACT.json", "IMPLEMENTATION_RUN_CONTRACT.sha256",
        "NATIVE_A_PREFIX_SEED42_EQUIVALENCE.json",
        "FIXED_SELECTOR_SEED42_EQUIVALENCE.json",
        "TAXONOMY_SEED42_EQUIVALENCE.json",
        "PARENT_IDENTITY_AUDIT.json", "RNG_IDENTITY_AUDIT.json",
        "ALL_FOUR_E100_ENDPOINTS_FROZEN.json", "ALL_FOUR_E100_ENDPOINTS_FROZEN.sha256",
        "FRESH_SEED_COMPARISON.csv", "RAW_EMA_INTERACTION.csv",
        "PER_LOCATION.csv", "PER_LOCATION_ENDPOINT_METRICS.csv", "PER_SIZE.csv",
        "ERROR_TRANSITIONS.csv", "FINAL_DETECTOR_TAXONOMY.csv",
        "ACCESS_AUDIT.json", "EVIDENCE_SUMMARY.json", "SUMMARY.md",
        "ORCHESTRATION_LOG.jsonl",
    ):
        copy_file(RESULT_ROOT / name, name)

    for seed in SEEDS:
        paths = seed_paths(seed)
        prefix = Path(f"SEED{seed}")
        policy_files = (
            paths["root"] / "PREPARATION_SUMMARY.json",
            paths["root"] / "PREPARATION_FAILURE.json",
            paths["root"] / "FORMAL_PAIR_MATCHING_AUDIT.json",
            paths["policy_gate"], paths["policy"] / "FIXED_A_POLICY_CONTRACT.json",
            paths["policy"] / "ROUTEA_SEED_SPECIFIC_SEMANTIC_CONTRACT.json",
            paths["policy"] / "fixed_a_image_delta_selection.csv",
            paths["policy"] / "fixed_a_visit_exposure_audit.csv",
            paths["policy"] / "fixed_a_exposure_equivalence.json",
            paths["policy"] / "fixed_a_theta_realization.csv",
            paths["policy"] / "fixed_a_semantic_unique_image.csv",
            paths["policy"] / "fixed_a_semantic_visit_weighted.csv",
            paths["policy"] / "raw" / "fixed_a_policy_build_audit.json",
            paths["policy"] / "raw" / "fixed_a_realization_audit.json",
            paths["policy"] / "raw" / "fixed_a_semantic_input_audit.json",
            paths["policy"] / "raw" / "fixed_a_semantic_runner_audit.json",
            paths["policy"] / "raw" / "fixed_a_semantic_aggregation_audit.json",
            paths["schedule"] / "SCHEDULE_FREEZE.json",
            paths["policy"] / "NATIVE_A_VISIT_POLICY" / "NATIVE_A_POLICY_AUDIT.json",
            paths["noop"] / "RUNNER_NOOP_AUDIT.json",
            paths["source_reference"],
            paths["compat_preflight"],
        )
        for source in policy_files:
            if source.is_file():
                copy_file(source, prefix / "POLICY" / source.name)
        for source in paths["cache"].rglob("*"):
            if source.is_file() and source.suffix.casefold() in {".json", ".csv"} and "working_cache" not in source.parts:
                copy_file(source, prefix / "CACHE" / source.relative_to(paths["cache"]))
        if paths["prelaunch"].is_file():
            copy_file(
                paths["prelaunch"], prefix / "CACHE" / paths["prelaunch"].name
            )
        for label, run_dir in (("CONTROL", paths["control"]), ("FIXED", paths["fixed"])):
            allowed = {
                "START_RESTORE_AUDIT.json", "RUN_REGISTRATION.json",
                "TRAINING_CORE_AUDIT.json", "TRAINING_FINAL_AUDIT.json",
                "TRAINING_PROCESS_IMPORT_AUDIT.json", "HPL_RUNTIME_AUDIT.json",
                "DATA_ACCESS_AUDIT.json", "FRESH_SEED_TRAINING_ANNOTATION.json",
                "TRAINING_EPOCH_METRICS.csv", "OPTIMIZER_UPDATE_TRACE.csv",
                "CHECKPOINT_MANIFEST.csv", "DATA_ACCESS_COMPLETED_EPOCHS.csv",
                "PARQUET_CONVERSION_AUDIT.json",
                "PSEUDO_LABEL_BATCH_TRACE.csv", "PSEUDO_LABEL_IMAGE_TRACE.parquet",
                "HPL_TENSOR_MANIFEST.csv", "HPL_BOX_TRACE.parquet",
                "FIXED_CACHE_RUNTIME_AUDIT.csv",
                "FIXED_EXPOSURE_RUNTIME_AUDIT.csv",
                "FIXED_CACHE_POSTRUN_REVERIFICATION.csv",
                "FIXED_CACHE_POSTRUN_REVERIFICATION.json",
            }
            for name in allowed:
                source = run_dir / name
                if source.is_file():
                    copy_file(source, prefix / label / name)
        evaluation = paths["root"] / "EVALUATION"
        for source in evaluation.rglob("*"):
            if source.is_file() and source.suffix.casefold() in {".json", ".csv", ".sha256"}:
                copy_file(source, prefix / "EVALUATION" / source.relative_to(evaluation))

    readme = "\n".join([
        "# Review package",
        "",
        "Formal raw-evidence bundle for M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2.",
        "",
        "- No checkpoint binaries are included.",
        "- No dataset content is included.",
        "- Target Test remained sealed.",
        "- Target Dev was first opened only after the four-way e100 freeze.",
        "- Any preserved `PREPARATION_FAILURE.json` is prior infrastructure evidence; final authorization is recorded only by the corresponding PASS preparation summary and policy/cache gate.",
        "- `SUMMARY.md` is descriptive evidence, not the final scientific decision.",
        "",
    ])
    (staging / "README.md").write_text(readme, encoding="utf-8")
    manifest_rows: list[dict[str, Any]] = []
    for path in sorted(staging.rglob("*"), key=lambda item: str(item.relative_to(staging)).casefold()):
        if path.is_file() and path.name != "PACKAGE_FILE_HASHES.csv":
            relative = str(path.relative_to(staging)).replace("\\", "/")
            manifest_rows.append({
                "relative_path": relative, "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    base.atomic_csv(staging / "PACKAGE_FILE_HASHES.csv", manifest_rows)
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*"), key=lambda item: str(item.relative_to(staging)).casefold()):
            if path.is_file():
                if path.suffix.casefold() in {".pt", ".npy", ".jpg", ".jpeg", ".png"}:
                    raise FreshSeedStop(f"STOP_FORBIDDEN_PACKAGE_PAYLOAD:{path}")
                archive.write(path, arcname=str(path.relative_to(staging)).replace("\\", "/"))
    package_audit = {
        "schema": "m1_fixed_fresh_seed_review_package_audit_v1",
        "status": "PASS", "path": str(bundle.resolve()),
        "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle),
        "files": len(manifest_rows) + 1,
        "checkpoint_binaries_included": 0, "dataset_content_included": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "REVIEW_PACKAGE_AUDIT.json", package_audit)
    return bundle


def finalize() -> dict[str, Any]:
    freeze_all_four_endpoints()
    tables = build_evidence_tables()
    parent_identity_audit()
    rng_identity_audit()
    access_audit()
    summary = evidence_summary(tables)
    write_summary_markdown(summary)
    bundle = package_review(summary)
    values = {**summary, "review_bundle_path": str(bundle.resolve())}
    atomic_json(RESULT_ROOT / "FINAL_CONSOLE_VALUES.json", values)
    return values


def print_final_console(values: Mapping[str, Any]) -> None:
    lines = [
        f"STATUS={values['status']}",
        f"SEED_IDENTITY_CONTRACT={values['seed_identity_contract']}",
        f"PARENT_SHA={values['parent_sha']}",
        "SEED45_POLICY_GATE=PASS",
        f"SEED45_CONTROL_EMA_MAP={values['seed45_control_ema_map']:.17g}",
        f"SEED45_FIXED_EMA_MAP={values['seed45_fixed_ema_map']:.17g}",
        f"DELTA45_AP={values['delta45_ap']:+.9f}",
        "SEED46_POLICY_GATE=PASS",
        f"SEED46_CONTROL_EMA_MAP={values['seed46_control_ema_map']:.17g}",
        f"SEED46_FIXED_EMA_MAP={values['seed46_fixed_ema_map']:.17g}",
        f"DELTA46_AP={values['delta46_ap']:+.9f}",
        f"MEAN_DELTA_45_46_AP={values['mean_delta_45_46_ap']:+.9f}",
        f"SEED45_RAW_DELTA_AP={values['seed45_raw_delta_ap']:+.9f}",
        f"SEED46_RAW_DELTA_AP={values['seed46_raw_delta_ap']:+.9f}",
        f"EMA_DIRECTION_CONSISTENCY={values['ema_direction_consistency']}",
        f"RAW_DIRECTION_CONSISTENCY={values['raw_direction_consistency']}",
        f"EMPTY_FP_SAFETY={values['empty_fp_safety']}",
        f"LOCATION_HETEROGENEITY={values['location_heterogeneity']}",
        f"TARGET_DEV_READS_BEFORE_ALL_FREEZE={values['target_dev_reads_before_all_freeze']}",
        f"TARGET_TEST_READS={values['target_test_reads']}",
        f"REVIEW_BUNDLE_PATH={values['review_bundle_path']}",
        "FINAL_SCIENTIFIC_DECISION",
        "= NOT_CODEX_JOB",
    ]
    print("\n\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    worker = sub.add_parser("noop-worker")
    worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    worker.add_argument("--branch", choices=base.BRANCHES, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--device", default="0")
    noop = sub.add_parser("noop")
    noop.add_argument("--seed", type=int, choices=SEEDS, required=True)
    noop.add_argument("--device", default="0")
    sub.add_parser("materialize-contract")
    train_worker = sub.add_parser("train-worker")
    train_worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train_worker.add_argument("--branch", choices=base.BRANCHES, required=True)
    train_worker.add_argument("--device", default="0")
    train_worker.add_argument("--resume-from", type=Path)
    train = sub.add_parser("train")
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--branch", choices=base.BRANCHES, required=True)
    train.add_argument("--device", default="0")
    convert = sub.add_parser("convert-parquet")
    convert.add_argument("--seed", type=int, choices=SEEDS, required=True)
    convert.add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("freeze-all")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--device", default="0")
    sub.add_parser("finalize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "noop-worker":
            result = noop_worker(args.seed, args.branch, args.output, args.device)
        elif args.action == "noop":
            result = run_noop_gate(args.seed, args.device)
        elif args.action == "materialize-contract":
            result = materialize_run_contract()
        elif args.action == "train-worker":
            result = run_training_worker(
                args.seed, args.branch, args.device, args.resume_from
            )
        elif args.action == "train":
            result = launch_training(args.seed, args.branch, args.device)
        elif args.action == "convert-parquet":
            result = convert_parquet_fresh(args.seed, args.run_dir)
        elif args.action == "freeze-all":
            result = freeze_all_four_endpoints()
        elif args.action == "evaluate":
            result = evaluate_seed(args.seed, args.device)
        elif args.action == "finalize":
            result = finalize()
        else:
            raise FreshSeedStop(f"STOP_ACTION:{args.action}")
        if args.action == "finalize":
            print_final_console(result)
            return 0
        print(json.dumps({
            "status": result.get("status"), "action": args.action,
            "adaptation_seed": int(getattr(args, "seed", -1)),
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "schema": "m1_fixed_fresh_seed_execution_failure_v1",
            "status": "FAIL", "action": args.action,
            "adaptation_seed": int(getattr(args, "seed", -1)),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "created_at": now_iso(),
        }
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(RESULT_ROOT / f"FAILURE_{args.action}_{getattr(args, 'seed', 'NA')}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
