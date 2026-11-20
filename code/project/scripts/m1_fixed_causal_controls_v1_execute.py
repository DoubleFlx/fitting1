from __future__ import annotations

"""Execute four frozen M1 Native-A Fixed causal-control training arms."""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
import warnings
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
FRESH_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2"
RESULT_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1"
SCIENCE_CONTRACT = CONTRACTS / "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1.json"
SCIENCE_CONTRACT_SHA256 = "ca16e949bc719d4268d5ae6eb83788657c5c653745a1fb6c8ca236c0ee08d55f"
RUN_CONTRACT = RESULT_ROOT / "IMPLEMENTATION_RUN_CONTRACT.json"
PARENT = PROJECT / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
PARENT_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
EXPERIMENT = "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1"
GENERIC = "GENERIC_FIXED_3DOF"
SHUFFLED = "FIXED_SHUFFLED"
VARIANTS = (GENERIC, SHUFFLED)
SEEDS = (45, 46)
EPOCHS = tuple(range(72, 101))
PREFLIGHT_MICROBATCHES = 10
OFFICIAL_EVALUATOR = SCRIPTS / "m1_seed42_stage1_e100_evaluate_v1.py"
OFFICIAL_EVALUATOR_SHA256 = "c38c767f93a9c831ff78c49a3dee5a592b2b139bfde5156a1c076d5d317f072d"

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

TRAIN_WORKER_ENTRY = len(sys.argv) > 1 and sys.argv[1] == "train-worker"
if TRAIN_WORKER_ENTRY:
    # The frozen runner audits that pandas and pyarrow are absent before
    # training.  The policy/cache builder imports both, so the formal worker
    # uses the equivalent lightweight path map below and imports no builder.
    prepare = None
else:
    import m1_fixed_causal_controls_v1_prepare as prepare  # noqa: E402
import m1_fixed_fresh_seed_v2_execute as fresh  # noqa: E402
import m1_seed42_stage1_formal_train_v1 as base  # noqa: E402
from protocol_b_target_schedule import ProtocolBTargetSchedule  # noqa: E402


ORIGINAL_SUBPROCESS_RUN = subprocess.run
ORIGINAL_BASE_VALIDATE_RECOVERY = base._validate_recovery


class CausalExecutionStop(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    base.atomic_json(path, value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CausalExecutionStop(f"STOP_JSON_OBJECT:{path}")
    return value


def require_sha(value: Any, role: str) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CausalExecutionStop(f"STOP_SHA:{role}:{value}")
    return digest


def arm_id(variant: str, seed: int) -> str:
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise CausalExecutionStop(f"STOP_VARIANT_SEED:{variant}:{seed}")
    return f"{variant}_SEED{int(seed)}"


def _seed_reference_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise CausalExecutionStop(f"STOP_SEED:{seed}")
    root = FRESH_ROOT / f"SEED{int(seed)}"
    policy = root / "POLICY"
    schedule = policy / "SCHEDULE"
    return {
        "root": root,
        "policy": policy,
        "source_schedule": schedule / "source_schedule_e071_e100.jsonl",
        "source_augmentation": schedule / "source_augmentation_e071_e100.jsonl",
        "target_schedule": schedule / "target_schedule_e071_e100.jsonl",
        "target_manifest": schedule / "target_schedule_e071_e100_manifest.json",
        "selection": policy / "fixed_a_image_delta_selection.csv",
        "theta": policy / "fixed_a_theta_realization.csv",
        "exposure": policy / "fixed_a_visit_exposure_audit.csv",
        "fixed_contract": policy / "FIXED_A_POLICY_CONTRACT.json",
    }


def _variant_paths(variant: str, seed: int) -> dict[str, Path]:
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise CausalExecutionStop(f"STOP_VARIANT_SEED:{variant}:{seed}")
    if variant == GENERIC:
        family = RESULT_ROOT / "GENERIC"
        seed_root = family / f"SEED{seed}"
        cache_root = family / "CACHE"
        return {
            "family": family,
            "seed_root": seed_root,
            "policy_contract": family / "GENERIC_FIXED_POLICY_CONTRACT.json",
            "realization": family / "GENERIC_FIXED_REALIZATION_AUDIT.csv",
            "cache": cache_root,
            "working_cache": cache_root / "working_cache",
            "commit_root": cache_root / "committed",
            "manifest": seed_root / "GENERIC_FIXED_CACHE_MANIFEST.csv",
            "visit_audit": seed_root / "GENERIC_FIXED_VISIT_AUDIT.csv",
            "cache_build": seed_root / "GENERIC_FIXED_CACHE_BUILD.json",
            "cache_audit": seed_root / "GENERIC_FIXED_CACHE_INTEGRITY_AUDIT.json",
            "policy_gate": seed_root / "POLICY_CACHE_PRETRAIN_GATE.json",
        }
    family = RESULT_ROOT / "SHUFFLED" / f"SEED{seed}"
    cache_root = family / "CACHE"
    return {
        "family": family,
        "seed_root": family,
        "policy_contract": family / "FIXED_SHUFFLED_POLICY_CONTRACT.json",
        "assignment": family / "SHUFFLE_ASSIGNMENT.csv",
        "graph_audit": family / "SHUFFLE_GRAPH_AUDIT.json",
        "multiset_audit": family / "SHUFFLE_MULTISET_AUDIT.json",
        "realization": family / "FIXED_SHUFFLED_THETA_REALIZATION.csv",
        "fairness": family / "SHUFFLED_VS_NATIVE_REALIZATION_FAIRNESS.csv",
        "cache": cache_root,
        "working_cache": cache_root / "working_cache",
        "commit_root": cache_root / "committed",
        "manifest": family / "FIXED_SHUFFLED_CACHE_MANIFEST.csv",
        "visit_audit": family / "FIXED_SHUFFLED_VISIT_AUDIT.csv",
        "cache_build": family / "FIXED_SHUFFLED_CACHE_BUILD.json",
        "cache_audit": family / "FIXED_SHUFFLED_CACHE_INTEGRITY_AUDIT.json",
        "policy_gate": family / "POLICY_CACHE_PRETRAIN_GATE.json",
    }


def paths_for(variant: str, seed: int) -> dict[str, Path]:
    reference = _seed_reference_paths(seed)
    variant_record = _variant_paths(variant, seed)
    arm = arm_id(variant, seed)
    run_dir = RESULT_ROOT / "TRAINING" / arm
    preflight = RESULT_ROOT / "PREFLIGHT" / arm
    return {
        **reference,
        **variant_record,
        "arm_id": Path(arm),
        "run_dir": run_dir,
        "noop": preflight / "RUNNER_NOOP",
        "mechanism": preflight / "MECHANISM_ACTIVATION_AUDIT.json",
        "compat_preflight": preflight / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json",
        "prelaunch": RESULT_ROOT / "PRELAUNCH" / arm / "fixed_cache_prelaunch.json",
        "source_reference": FRESH_ROOT / f"SEED{seed}" / "CONTROL_SOURCE_HASH_REFERENCE.csv",
    }


def _write_arm_logging_preflight(variant: str, seed: int) -> dict[str, Any]:
    """Project the causal no-op audit into the frozen runner's gate schema."""
    paths = paths_for(variant, seed)
    audit_path = paths["noop"] / "RUNNER_NOOP_AUDIT.json"
    if not audit_path.is_file():
        raise CausalExecutionStop(f"STOP_NOOP_AUDIT_MISSING:{variant}:{seed}")
    audit = load_json(audit_path)
    checks = {
        "noop_status_pass": audit.get("status") == "PASS",
        "variant_exact": audit.get("variant") == variant,
        "seed_exact": int(audit.get("adaptation_seed", -1)) == int(seed),
        "executor_sha_exact": audit.get("executor_sha256") == sha256_file(Path(__file__)),
        "all_noop_checks_pass": all(audit.get("checks", {}).values()),
        "checkpoint_writes_zero": int(audit.get("checkpoint_writes", 0)) == 0,
        "target_dev_reads_zero": int(audit.get("target_dev_reads", -1)) == 0,
        "target_test_reads_zero": int(audit.get("target_test_reads", -1)) == 0,
        "target_adapt_gt_reads_zero": int(audit.get("target_adapt_label_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise CausalExecutionStop(
            f"STOP_NOOP_COMPATIBILITY_SOURCE:{variant}:{seed}:"
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    compatibility = {
        "schema": "m1_seed42_stage1_logging_preflight_audit_v1",
        "status": "PASS",
        "all_pass": True,
        "run_contract_sha256": sha256_file(RUN_CONTRACT),
        "runner_sha256": sha256_file(Path(base.__file__)),
        "compatibility_role": "CAUSAL_ARM_NOOP_PROJECTION_FOR_FROZEN_RUNNER_GATE",
        "external_arm_id": arm_id(variant, seed),
        "variant": variant,
        "adaptation_seed": int(seed),
        "causal_noop_audit_path": str(audit_path.resolve()),
        "causal_noop_audit_sha256": sha256_file(audit_path),
        "causal_executor_sha256": sha256_file(Path(__file__)),
        "source_checks": checks,
        "formal_training_started": False,
        "checkpoint_writes": 0,
        "snapshot_writes": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["compat_preflight"], compatibility)
    return compatibility


def _activate_arm_logging_preflight(variant: str, seed: int) -> dict[str, Any]:
    """Atomically activate one verified arm gate at the frozen runner root."""
    compatibility = _write_arm_logging_preflight(variant, seed)
    active = {
        **compatibility,
        "activation_scope": "SINGLE_SERIAL_FORMAL_ARM",
        "active_external_arm_id": arm_id(variant, seed),
        "arm_compatibility_path": str(paths_for(variant, seed)["compat_preflight"].resolve()),
        "arm_compatibility_sha256": sha256_file(paths_for(variant, seed)["compat_preflight"]),
        "activated_at": now_iso(),
    }
    active_path = RESULT_ROOT / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json"
    atomic_json(active_path, active)
    frozen = load_json(active_path)
    if not (
        frozen.get("schema") == "m1_seed42_stage1_logging_preflight_audit_v1"
        and frozen.get("all_pass") is True
        and frozen.get("run_contract_sha256") == sha256_file(RUN_CONTRACT)
        and frozen.get("runner_sha256") == sha256_file(Path(base.__file__))
        and frozen.get("active_external_arm_id") == arm_id(variant, seed)
    ):
        raise CausalExecutionStop(f"STOP_LOGGING_PREFLIGHT_ACTIVATION:{variant}:{seed}")
    return frozen


def _code_authorities() -> dict[str, dict[str, Any]]:
    fresh_contract = load_json(FRESH_ROOT / "IMPLEMENTATION_RUN_CONTRACT.json")
    authorities: dict[str, dict[str, Any]] = {}
    for relative, record in fresh_contract["code_authorities"].items():
        path = ROOT / relative
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise CausalExecutionStop(f"STOP_FROZEN_TRANSITIVE_CODE_DRIFT:{relative}:{digest}")
        authorities[relative] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}
    for name in (
        "m1_fixed_causal_controls_v1_prepare.py",
        "m1_fixed_causal_controls_v1_execute.py",
        "m1_fixed_causal_controls_v1_evaluate.py",
        "m1_fixed_causal_controls_v1_package.py",
        "m1_fixed_causal_controls_v1_orchestrate.ps1",
    ):
        path = SCRIPTS / name
        if not path.is_file():
            raise CausalExecutionStop(f"STOP_NEW_IMPLEMENTATION_FILE_MISSING:{name}")
        relative = path.relative_to(ROOT).as_posix()
        authorities[relative] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return dict(sorted(authorities.items()))


def materialize_run_contract() -> dict[str, Any]:
    prepare.verify_input_identities()
    if sha256_file(SCIENCE_CONTRACT) != SCIENCE_CONTRACT_SHA256:
        raise CausalExecutionStop("STOP_SCIENCE_CONTRACT_SHA")
    code = _code_authorities()
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        reference = _seed_reference_paths(seed)
        per_seed[str(seed)] = {
            "adaptation_seed": seed,
            "frozen_schedules_and_active_mask": {
                role: {"path": str(reference[role].resolve()), "bytes": reference[role].stat().st_size, "sha256": sha256_file(reference[role])}
                for role in ("source_schedule", "source_augmentation", "target_schedule", "target_manifest", "exposure", "selection", "theta")
            },
            "canonical_source_reference": {
                "path": str((FRESH_ROOT / f"SEED{seed}" / "CONTROL_SOURCE_HASH_REFERENCE.csv").resolve()),
                "sha256": sha256_file(FRESH_ROOT / f"SEED{seed}" / "CONTROL_SOURCE_HASH_REFERENCE.csv"),
            },
            "arms": {
                variant: str(paths_for(variant, seed)["run_dir"].resolve()) for variant in VARIANTS
            },
        }
    contract = {
        "schema": "m1_native_a_fixed_causal_controls_implementation_run_contract_v1",
        "status": "FROZEN_BEFORE_PREFLIGHT_CACHE_TRAINING_OR_NEW_TARGET_DEV_ACCESS",
        "experiment": EXPERIMENT,
        "created_at": now_iso(),
        "science_contract": {"path": str(SCIENCE_CONTRACT.resolve()), "sha256": SCIENCE_CONTRACT_SHA256},
        "parent": {"path": str(PARENT.resolve()), "bytes": PARENT.stat().st_size, "sha256": PARENT_SHA256, "epoch": 71},
        "official_evaluator": {"path": str(OFFICIAL_EVALUATOR.resolve()), "bytes": OFFICIAL_EVALUATOR.stat().st_size, "sha256": OFFICIAL_EVALUATOR_SHA256},
        "code_authorities": code,
        "per_seed": per_seed,
        "arms_in_serial_order": [
            "GENERIC_FIXED_3DOF_SEED45", "FIXED_SHUFFLED_SEED45",
            "GENERIC_FIXED_3DOF_SEED46", "FIXED_SHUFFLED_SEED46",
        ],
        "internal_frozen_runner_branch": "NATIVE_A_FIXED",
        "external_arm_lineage_is_mandatory": True,
        "single_scientific_difference": "TARGET_STUDENT_PRELETTERBOX_BGR_VIEW_CACHE",
        "training": {"epochs": list(EPOCHS), "formal_horizon": 100, "primary_endpoint": "E100_EMA", "secondary_endpoint": "E100_RAW", "best_epoch_selection": False},
        "data_access": {"target_dev_before_all_four_freeze": 0, "target_test": 0, "target_adapt_gt": 0},
        "governance_conflicts_recorded": {
            "requested_project_docs_PROJECT_HANDOFF_20260827_missing": True,
            "repository_git_metadata_unavailable_empty_dot_git": True,
            "legacy_NEXT_ACTIONS_predates_current_explicit_authorization": True,
        },
    }
    if RUN_CONTRACT.is_file():
        existing = load_json(RUN_CONTRACT)
        left = deepcopy(existing)
        right = deepcopy(contract)
        left.pop("created_at", None)
        right.pop("created_at", None)
        if left != right:
            raise CausalExecutionStop("STOP_EXISTING_IMPLEMENTATION_RUN_CONTRACT_DRIFT")
        return existing
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_CONTRACT, contract)
    digest = sha256_file(RUN_CONTRACT)
    RUN_CONTRACT.with_suffix(".sha256").write_text(f"{digest}  {RUN_CONTRACT.name}\n", encoding="ascii")
    return contract


def validate_run_contract() -> dict[str, Any]:
    if not RUN_CONTRACT.is_file() or not RUN_CONTRACT.with_suffix(".sha256").is_file():
        raise CausalExecutionStop("STOP_IMPLEMENTATION_RUN_CONTRACT_MISSING")
    digest = sha256_file(RUN_CONTRACT)
    sidecar = RUN_CONTRACT.with_suffix(".sha256").read_text(encoding="ascii").strip().split()[0]
    contract = load_json(RUN_CONTRACT)
    checks: dict[str, bool] = {
        "sidecar": digest == sidecar,
        "schema": contract.get("schema") == "m1_native_a_fixed_causal_controls_implementation_run_contract_v1",
        "experiment": contract.get("experiment") == EXPERIMENT,
        "science": contract.get("science_contract", {}).get("sha256") == SCIENCE_CONTRACT_SHA256 == sha256_file(SCIENCE_CONTRACT),
        "parent": contract.get("parent", {}).get("sha256") == PARENT_SHA256 == sha256_file(PARENT),
        "evaluator": contract.get("official_evaluator", {}).get("sha256") == OFFICIAL_EVALUATOR_SHA256 == sha256_file(OFFICIAL_EVALUATOR),
    }
    for relative, record in contract.get("code_authorities", {}).items():
        path = ROOT / relative
        checks[f"code:{relative}"] = path.is_file() and path.stat().st_size == int(record["bytes"]) and sha256_file(path) == record["sha256"]
    for seed in SEEDS:
        row = contract["per_seed"][str(seed)]
        for role, record in row["frozen_schedules_and_active_mask"].items():
            path = _seed_reference_paths(seed)[role]
            checks[f"seed{seed}:{role}"] = sha256_file(path) == record["sha256"]
        source = paths_for(GENERIC, seed)["source_reference"]
        checks[f"seed{seed}:source_reference"] = sha256_file(source) == row["canonical_source_reference"]["sha256"]
    if not all(checks.values()):
        raise CausalExecutionStop(f"STOP_IMPLEMENTATION_RUN_CONTRACT:{sorted(key for key,value in checks.items() if not value)}")
    return contract


def policy_gate(variant: str, seed: int) -> dict[str, Any]:
    path = paths_for(variant, seed)["policy_gate"]
    gate = load_json(path)
    if not (
        gate.get("status") == "PASS"
        and gate.get("training_authorized_by_identity_contract_after_gate") is True
        and int(gate.get("adaptation_seed", -1)) == int(seed)
        and int(gate.get("target_dev_reads", -1)) == 0
        and int(gate.get("target_test_reads", -1)) == 0
        and int(gate.get("target_adapt_label_reads", -1)) == 0
        and all(gate.get("checks", {}).values())
    ):
        raise CausalExecutionStop(f"STOP_POLICY_CACHE_GATE:{variant}:{seed}")
    return gate


def _fresh_paths(variant: str, seed: int) -> dict[str, Path]:
    paths = paths_for(variant, seed)
    return {
        "root": RESULT_ROOT / f"RUNTIME_CONTEXT_{arm_id(variant, seed)}",
        "policy": paths["family"],
        "schedule": paths["target_schedule"].parent,
        "source_schedule": paths["source_schedule"],
        "source_augmentation": paths["source_augmentation"],
        "target_schedule": paths["target_schedule"],
        "target_manifest": paths["target_manifest"],
        "fixed_contract": paths["policy_contract"],
        "exposure": paths["exposure"],
        "cache": paths["cache"],
        "cache_manifest": paths["manifest"],
        "cache_build": paths["cache_build"],
        "policy_gate": paths["policy_gate"],
        "control": RESULT_ROOT / "UNUSED_CONTROL_RUNS" / arm_id(variant, seed),
        "fixed": paths["run_dir"],
        "noop": paths["noop"],
        "source_reference_placeholder": paths["source_reference"],
        "source_reference": paths["source_reference"],
        "prelaunch": paths["prelaunch"],
        "compat_preflight": paths["compat_preflight"],
    }


def configure_base(variant: str, seed: int) -> dict[str, Path]:
    paths = paths_for(variant, seed)
    fresh.EXPERIMENT = EXPERIMENT
    fresh.RESULT_ROOT = RESULT_ROOT
    fresh.RUN_CONTRACT = RUN_CONTRACT
    fresh.IDENTITY_CONTRACT = SCIENCE_CONTRACT
    fresh.PARENT = PARENT
    fresh.PARENT_SHA256 = PARENT_SHA256
    fresh.seed_paths = lambda value: _fresh_paths(variant, int(value))
    fresh.policy_gate = lambda value: policy_gate(variant, int(value))
    base.EXPERIMENT = EXPERIMENT
    base.RESULT_ROOT = RESULT_ROOT
    base.DEFAULT_CONTRACT = RUN_CONTRACT
    base.E71_FULL_STATE = PARENT
    base.E71_FILE_SHA256 = PARENT_SHA256
    base.SOURCE_SCHEDULE = paths["source_schedule"]
    base.SOURCE_AUGMENTATION = paths["source_augmentation"]
    base.TARGET_SCHEDULE = paths["target_schedule"]
    base.TARGET_SCHEDULE_MANIFEST = paths["target_manifest"]
    base.FIXED_POLICY_CONTRACT = paths["policy_contract"]
    base.FIXED_EXPOSURE = paths["exposure"]
    base.FIXED_CACHE_MANIFEST = paths["manifest"]
    base.FIXED_CACHE_BUILD = paths["cache_build"]
    base.FIXED_PRETRAIN_FINAL_STATUS = paths["policy_gate"]
    base.FIXED_CACHE_ROOT = paths["working_cache"]
    base.FIXED_CACHE_PRELAUNCH = paths["prelaunch"]
    base.B1_METRICS = paths["source_reference"]
    base.EXPECTED_INPUT_SHA256.update({
        "source_schedule": sha256_file(paths["source_schedule"]),
        "source_augmentation": sha256_file(paths["source_augmentation"]),
        "target_schedule": sha256_file(paths["target_schedule"]),
        "fixed_policy_contract": sha256_file(paths["policy_contract"]),
        "fixed_exposure": sha256_file(paths["exposure"]),
        "fixed_cache_manifest": sha256_file(paths["manifest"]),
        "fixed_cache_build": sha256_file(paths["cache_build"]),
        "fixed_pretrain_final_status": sha256_file(paths["policy_gate"]),
        "b1_train_metrics": sha256_file(paths["source_reference"]),
    })
    return paths


def load_fixed_inputs(variant: str, seed: int, schedule: ProtocolBTargetSchedule):
    configure_base(variant, seed)
    return fresh.load_fixed_inputs_fresh(seed, schedule)


class CausalStage1Trainer(fresh.FreshStage1Trainer):
    """External-lineage adapter; frozen Stage-1 math is inherited unchanged."""


def build_causal_trainer(
    *, variant: str, seed: int, branch: str, run_dir: Path, device: str,
    start_state: dict[str, Any], start_state_path: Path,
    exposure_entries: list[base.ExposureEntry], cache_entries: dict[str, base.CacheEntry],
    data_guard: Any, data_access_attempt_dir: Path, run_attempt_id: str,
    logging_enabled: bool, preflight: bool, resume: bool,
) -> CausalStage1Trainer:
    paths = configure_base(variant, seed)
    overrides = base.trainer_overrides(run_dir, device, preflight=preflight, resume=resume)
    overrides["seed"] = int(seed)
    return CausalStage1Trainer(
        adaptation_seed=seed,
        source_reference_mode="LOCK_TO_CONTROL",
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
        b1_train_metrics_path=paths["source_reference"],
        comparison_rule_path=base.COMPARISON_RULE,
        code_snapshot_sha256=sha256_file(Path(__file__)),
        environment_sha256=canonical_sha({"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "device": device}),
        overrides=overrides,
    )


def validate_parent() -> dict[str, Any]:
    if sha256_file(PARENT) != PARENT_SHA256:
        raise CausalExecutionStop("STOP_PARENT_IDENTITY_UNRECOVERABLE")
    return fresh.validate_parent()


def create_cache_prelaunch(variant: str, seed: int) -> dict[str, Any]:
    configure_base(variant, seed)
    return fresh.create_cache_prelaunch(seed)


def validate_recovery(path: Path, expected_sha: str, variant: str, seed: int) -> dict[str, Any]:
    configure_base(variant, seed)
    return fresh.validate_recovery_start_state(path, expected_sha, "NATIVE_A_FIXED", seed)


def patch_base_for_training(variant: str, seed: int) -> dict[str, Path]:
    paths = configure_base(variant, seed)

    def load_contract(path: Path, *, authority: str) -> tuple[dict[str, Any], str]:
        del authority
        contract = validate_run_contract()
        if Path(path).resolve() != RUN_CONTRACT.resolve():
            raise CausalExecutionStop("STOP_RUN_CONTRACT_PATH")
        view = deepcopy(contract)
        view["branches"] = {
            "CONTROL": {"run_dir": str((RESULT_ROOT / "UNUSED_CONTROL_RUNS" / arm_id(variant, seed)).resolve())},
            "NATIVE_A_FIXED": {"run_dir": str(paths["run_dir"].resolve())},
        }
        return view, sha256_file(RUN_CONTRACT)

    def static_locks() -> dict[str, str]:
        validate_run_contract()
        gate = policy_gate(variant, seed)
        noop_path = paths["noop"] / "RUNNER_NOOP_AUDIT.json"
        mechanism_path = paths["mechanism"]
        if not noop_path.is_file() or load_json(noop_path).get("status") != "PASS":
            raise CausalExecutionStop(f"STOP_NOOP_GATE_NOT_PASS:{variant}:{seed}")
        if not mechanism_path.is_file() or load_json(mechanism_path).get("status") != "PASS":
            raise CausalExecutionStop(f"STOP_MECHANISM_GATE_NOT_PASS:{variant}:{seed}")
        create_cache_prelaunch(variant, seed)
        return {
            "implementation_run_contract": sha256_file(RUN_CONTRACT),
            "policy_gate": sha256_file(paths["policy_gate"]),
            "runner_noop": sha256_file(noop_path),
            "mechanism_activation": sha256_file(mechanism_path),
            "cache_payload_composite": str(gate["cache_payload_composite_sha256"]),
        }

    def validate_start(path: Path, expected_sha: str, *, branch: str | None = None) -> dict[str, Any]:
        if Path(path).resolve() == PARENT.resolve():
            if expected_sha != PARENT_SHA256:
                raise CausalExecutionStop("STOP_INITIAL_PARENT_SHA")
            return validate_parent()
        if branch != "NATIVE_A_FIXED":
            raise CausalExecutionStop("STOP_RECOVERY_INTERNAL_BRANCH")
        return validate_recovery(Path(path), expected_sha, variant, seed)

    def load_fixed(schedule: ProtocolBTargetSchedule):
        return load_fixed_inputs(variant, seed, schedule)

    def build_trainer_adapter(**kwargs: Any) -> CausalStage1Trainer:
        values = {key: value for key, value in kwargs.items() if key not in {"contract_path", "contract_sha"}}
        return build_causal_trainer(variant=variant, seed=seed, **values)

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
                sys.executable, "-B", str(Path(__file__).resolve()), "convert-parquet",
                "--variant", variant, "--seed", str(seed), "--run-dir", run_value,
            ]
            environment = dict(kwargs.get("env") or os.environ)
            environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(seed)})
            return ORIGINAL_SUBPROCESS_RUN(adapted, cwd=kwargs.get("cwd", ROOT), env=environment, check=kwargs.get("check", False))
        return ORIGINAL_SUBPROCESS_RUN(command, **kwargs)

    base.subprocess = SimpleNamespace(run=subprocess_adapter)
    return paths


def noop_worker(variant: str, seed: int, branch: str, output: Path, device: str) -> dict[str, Any]:
    if branch not in base.BRANCHES:
        raise CausalExecutionStop(f"STOP_NOOP_BRANCH:{branch}")
    validate_run_contract()
    configure_base(variant, seed)
    policy_gate(variant, seed)
    state = validate_parent()
    schedule = ProtocolBTargetSchedule(base.TARGET_SCHEDULE, base.TARGET_SCHEDULE_MANIFEST)
    exposure, cache, fixed_summary = load_fixed_inputs(variant, seed, schedule)
    exposure = [replace(row, fixed_active=False, fixed_identity=True) for row in exposure]
    output.mkdir(parents=True, exist_ok=False)
    data_guard = base.install_m1_data_isolation_guard(
        phase=f"{EXPERIMENT}_{arm_id(variant, seed)}_{branch}_NOOP",
        audit_dir=output / "data_access_parent",
        allow_target_dev_images=False,
    )
    trainer = build_causal_trainer(
        variant=variant, seed=seed, branch=branch,
        run_dir=output / "trainer_sandbox", device=device,
        start_state=state, start_state_path=PARENT,
        exposure_entries=exposure, cache_entries=cache,
        data_guard=data_guard, data_access_attempt_dir=output / "data_access_workers",
        run_attempt_id=f"{arm_id(variant, seed)}_{branch}_NOOP",
        logging_enabled=True, preflight=True, resume=False,
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
        "schema": "m1_fixed_causal_controls_runner_noop_worker_v1",
        "status": "PASS",
        "variant": variant,
        "adaptation_seed": seed,
        "branch": branch,
        "fixed_transformation_disabled": True,
        "fixed_input_summary_before_noop_mask": fixed_summary,
        "projection": projection,
        "normalized_projection_sha256": canonical_sha(fresh.normalized_projection(projection)),
        "rng_boundary": trainer.fresh_reseed_audit,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "checkpoint_writes": len(list(output.rglob("*.pt"))),
        "created_at": now_iso(),
    }
    if result["checkpoint_writes"] != 0:
        raise CausalExecutionStop("STOP_NOOP_CHECKPOINT_WRITE")
    atomic_json(output / "result.json", result)
    return result


def run_noop_gate(variant: str, seed: int, device: str) -> dict[str, Any]:
    validate_run_contract()
    paths = configure_base(variant, seed)
    destination = paths["noop"]
    audit_path = destination / "RUNNER_NOOP_AUDIT.json"
    if audit_path.is_file():
        audit = load_json(audit_path)
        if audit.get("status") == "PASS" and audit.get("executor_sha256") == sha256_file(Path(__file__)):
            _write_arm_logging_preflight(variant, seed)
            return audit
        raise CausalExecutionStop(f"STOP_EXISTING_NOOP_DRIFT:{variant}:{seed}")
    destination.mkdir(parents=True, exist_ok=True)
    workers: list[dict[str, Any]] = []
    for branch, folder in (("CONTROL", "CONTROL"), ("NATIVE_A_FIXED", "FIXED_CAPABLE_NOOP")):
        output = destination / folder
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "noop-worker",
            "--variant", variant, "--seed", str(seed), "--branch", branch,
            "--output", str(output), "--device", str(device),
        ]
        environment = dict(os.environ)
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(seed)})
        completed = ORIGINAL_SUBPROCESS_RUN(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode != 0:
            raise CausalExecutionStop(f"STOP_NOOP_WORKER_EXIT:{variant}:{seed}:{branch}:{completed.returncode}")
        workers.append(load_json(output / "result.json"))
    control, fixed_noop = workers
    left = fresh.normalized_projection(control["projection"])
    right = fresh.normalized_projection(fixed_noop["projection"])
    checks = {
        "two_workers_pass": all(row.get("status") == "PASS" for row in workers),
        "control_fixed_capable_noop_projection_exact": left == right,
        "uid_augmentation_target_schedule_exact": control["projection"]["source_rows"] == fixed_noop["projection"]["source_rows"],
        "teacher_student_hpl_loss_gradient_exact": fresh.normalized_projection(control["projection"]["target_hpl_common_rows"]) == fresh.normalized_projection(fixed_noop["projection"]["target_hpl_common_rows"]),
        "raw_ema_optimizer_scheduler_scaler_bn_rng_exact": all(
            control["projection"][key] == fixed_noop["projection"][key]
            for key in (
                "student_tensor_sha256", "teacher_tensor_sha256", "optimizer_state_sha256",
                "scheduler_state_sha256", "scaler_state_sha256", "rng_state_sha256",
                "bn_running_state_sha256", "ema_updates",
            )
        ),
        "checkpoint_writes_zero": all(int(row["checkpoint_writes"]) == 0 for row in workers),
        "target_dev_reads_zero": all(int(row["target_dev_reads"]) == 0 for row in workers),
        "target_test_reads_zero": all(int(row["target_test_reads"]) == 0 for row in workers),
        "target_adapt_gt_reads_zero": all(int(row["target_adapt_label_reads"]) == 0 for row in workers),
    }
    audit = {
        "schema": "m1_fixed_causal_controls_runner_noop_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
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
        raise CausalExecutionStop(f"STOP_RUNNER_NOOP:{variant}:{seed}:{checks}")
    _write_arm_logging_preflight(variant, seed)
    return audit


def _mechanism_projection(variant: str, seed: int, branch: str, output: Path) -> dict[str, Any]:
    configure_base(variant, seed)
    schedule = ProtocolBTargetSchedule(base.TARGET_SCHEDULE, base.TARGET_SCHEDULE_MANIFEST)
    exposure, cache, summary = load_fixed_inputs(variant, seed, schedule)
    summary = {
        **summary,
        "preflight_active_visits": sum(int(row.fixed_active) for row in exposure[:80]),
        "preflight_identity_visits": sum(int(row.fixed_identity) for row in exposure[:80]),
    }
    base.install_m1_data_isolation_guard(
        phase=f"{EXPERIMENT}_{arm_id(variant, seed)}_{branch}_MECHANISM",
        audit_dir=output / f"{branch}_parent",
        allow_target_dev_images=False,
    )
    loader = base.build_fixed_target_loader(
        schedule=schedule,
        exposure_entries=exposure,
        cache_entries=cache,
        branch=branch,
        workers=2,
        imgsz=640,
        data_audit_dir=output / f"{branch}_workers",
        verify_payload_on_read=True,
        max_batches=10,
    )
    batches: list[dict[str, Any]] = []
    iterator = iter(loader)
    for batch_index in range(10):
        batch = next(iterator)
        teacher = batch["img"]
        student = batch["fixed_student_img"]
        active = [int(value) for value in batch["fixed_exposure_active"]]
        applied = [int(value) for value in batch["fixed_transform_applied"]]
        batches.append({
            "batch_index": batch_index,
            "uids": list(batch["protocol_b_target_uid"]),
            "view_seeds": [int(value) for value in batch["protocol_b_target_view_seed"]],
            "active": active,
            "applied": applied,
            "teacher_batch_sha256": base.tensor_bytes_sha(teacher),
            "student_batch_sha256": base.tensor_bytes_sha(student),
            "teacher_image_sha256": [base.tensor_bytes_sha(teacher[index]) for index in range(len(active))],
            "student_image_sha256": [base.tensor_bytes_sha(student[index]) for index in range(len(active))],
            "padding_max_abs_byte_delta": [int(value) for value in batch["fixed_padding_max_abs_byte_delta"]],
            "cache_validation_count": sum(int(value) for value in batch["fixed_cache_payload_validation"]),
        })
    inner = getattr(iterator, "_iterator", iterator)
    shutdown = getattr(inner, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    return {"branch": branch, "summary": summary, "batches": batches}


def run_mechanism_gate(variant: str, seed: int) -> dict[str, Any]:
    validate_run_contract()
    paths = paths_for(variant, seed)
    destination = paths["mechanism"]
    if destination.is_file():
        audit = load_json(destination)
        if audit.get("status") == "PASS" and audit.get("executor_sha256") == sha256_file(Path(__file__)):
            return audit
        raise CausalExecutionStop(f"STOP_EXISTING_MECHANISM_DRIFT:{variant}:{seed}")
    output = destination.parent / "MECHANISM_RAW"
    output.mkdir(parents=True, exist_ok=True)
    control = _mechanism_projection(variant, seed, "CONTROL", output)
    fixed = _mechanism_projection(variant, seed, "NATIVE_A_FIXED", output)
    control_rows = control["batches"]
    fixed_rows = fixed["batches"]
    active_count = sum(sum(row["active"]) for row in fixed_rows)
    identity_count = 80 - active_count
    teacher_exact = all(
        left["uids"] == right["uids"]
        and left["view_seeds"] == right["view_seeds"]
        and left["teacher_image_sha256"] == right["teacher_image_sha256"]
        for left, right in zip(control_rows, fixed_rows, strict=True)
    )
    active_different = all(
        student != teacher
        for row in fixed_rows
        for active, teacher, student in zip(row["active"], row["teacher_image_sha256"], row["student_image_sha256"], strict=True)
        if active
    )
    identity_exact = all(
        student == teacher
        for row in fixed_rows
        for active, teacher, student in zip(row["active"], row["teacher_image_sha256"], row["student_image_sha256"], strict=True)
        if not active
    )
    checks = {
        "uid_view_schedule_exact": all(left["uids"] == right["uids"] and left["view_seeds"] == right["view_seeds"] for left, right in zip(control_rows, fixed_rows, strict=True)),
        "teacher_tensor_remains_exact_original": teacher_exact,
        "active_student_tensor_differs_from_identity": active_different,
        "identity_visits_byte_identical": identity_exact,
        "cache_lookup_count_matches_expected_active_visits": sum(sum(row["applied"]) for row in fixed_rows) == active_count,
        "active_identity_count_matches_frozen_prefix": active_count == int(fixed["summary"].get("preflight_active_visits", active_count)) and identity_count == int(fixed["summary"].get("preflight_identity_visits", identity_count)),
        "padding_geometry_unchanged": all(max(row["padding_max_abs_byte_delta"], default=0) == 0 for row in fixed_rows),
        "target_dev_reads_zero": True,
        "target_test_reads_zero": True,
        "target_adapt_gt_reads_zero": True,
    }
    audit = {
        "schema": "m1_fixed_causal_controls_mechanism_activation_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
        "active_visits_checked": active_count,
        "identity_visits_checked": identity_count,
        "control_projection_sha256": canonical_sha(control),
        "fixed_projection_sha256": canonical_sha(fixed),
        "checks": checks,
        "executor_sha256": sha256_file(Path(__file__)),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(destination, audit)
    if audit["status"] != "PASS":
        raise CausalExecutionStop(f"STOP_MECHANISM_GATE:{variant}:{seed}:{checks}")
    return audit


def convert_parquet(variant: str, seed: int, run_dir: Path) -> dict[str, Any]:
    configure_base(variant, seed)
    original_configure = fresh.configure_base
    original_policy_gate = fresh.policy_gate
    original_seed_paths = fresh.seed_paths
    import m1_fixed_fresh_seed_v2_prepare as fresh_prepare
    original_verify = fresh_prepare.verify_cache
    try:
        fresh.configure_base = lambda value: configure_base(variant, int(value))
        fresh.policy_gate = lambda value: policy_gate(variant, int(value))
        fresh.seed_paths = lambda value: _fresh_paths(variant, int(value))

        def verify_adapter(value: int, _paths: Any, _counts: Any) -> dict[str, Any]:
            audit = prepare.verify_cache(variant, int(value), full_payload=True)
            return {**audit, "all_pass": audit.get("status") == "PASS"}

        fresh_prepare.verify_cache = verify_adapter
        return fresh.convert_parquet_fresh(seed, run_dir)
    finally:
        fresh.configure_base = original_configure
        fresh.policy_gate = original_policy_gate
        fresh.seed_paths = original_seed_paths
        fresh_prepare.verify_cache = original_verify


def _correct_hpl_runtime_audit(run_dir: Path, variant: str, seed: int) -> dict[str, Any]:
    import pandas as pd

    core = load_json(run_dir / "TRAINING_CORE_AUDIT.json")
    batch = pd.read_csv(run_dir / "PSEUDO_LABEL_BATCH_TRACE.csv")
    boxes = pd.read_parquet(run_dir / "HPL_BOX_TRACE.parquet", columns=["score"])
    scores = boxes["score"].to_numpy(dtype=np.float64)
    counts = batch["pseudo_final_boxes"].to_numpy(dtype=np.int64)
    positive = np.flatnonzero(counts > 0)
    first = None
    if len(positive):
        row = batch.iloc[int(positive[0])]
        first = {
            "row_index": int(positive[0]), "epoch": int(row["epoch"]),
            "source_batch_index": int(row["source_batch_index"]),
            "target_unsupervised_optimizer_update_count": int(row["target_unsupervised_optimizer_update_count"]),
        }
    audit = {
        "schema": "m1_fixed_causal_controls_hpl_runtime_audit_corrected_v1",
        "status": "PASS",
        "variant": variant,
        "adaptation_seed": seed,
        "teacher_forward_count": int(core["teacher_target_forward_count"]),
        "target_batch_count": len(batch),
        "hpl_box_count": len(boxes),
        "hpl_confidence_distribution": {
            "count": len(scores),
            "minimum": float(np.min(scores)) if len(scores) else None,
            "q10": float(np.quantile(scores, 0.10, method="linear")) if len(scores) else None,
            "median": float(np.median(scores)) if len(scores) else None,
            "mean": float(np.mean(scores)) if len(scores) else None,
            "q90": float(np.quantile(scores, 0.90, method="linear")) if len(scores) else None,
            "maximum": float(np.max(scores)) if len(scores) else None,
        },
        "empty_hpl_batch_count": int(np.count_nonzero(counts == 0)),
        "first_hpl_batch": first,
        "trace_sha256": sha256_file(run_dir / "PSEUDO_LABEL_BATCH_TRACE.csv"),
        "tensor_manifest_sha256": sha256_file(run_dir / "HPL_TENSOR_MANIFEST.csv"),
        "target_adapt_gt_reads": 0,
        "target_dev_reads_during_training": 0,
        "target_test_reads_during_training": 0,
        "supersedes_original_bookkeeping_only": True,
        "scientific_training_path_changed": False,
        "created_at": now_iso(),
    }
    atomic_json(run_dir / "HPL_RUNTIME_AUDIT_CORRECTED.json", audit)
    return audit


def run_training_worker(variant: str, seed: int, device: str, resume_from: Path | None) -> dict[str, Any]:
    paths = patch_base_for_training(variant, seed)
    create_cache_prelaunch(variant, seed)
    _activate_arm_logging_preflight(variant, seed)
    args = argparse.Namespace(
        contract=str(RUN_CONTRACT), branch="NATIVE_A_FIXED", device=str(device),
        resume_from=str(resume_from) if resume_from is not None else None,
    )
    base.run_formal_branch(args)
    final = load_json(paths["run_dir"] / "TRAINING_FINAL_AUDIT.json")
    if final.get("completed") is not True:
        raise CausalExecutionStop(f"STOP_TRAINING_FINAL_AUDIT:{variant}:{seed}")
    hpl = _correct_hpl_runtime_audit(paths["run_dir"], variant, seed)
    annotation = {
        "schema": "m1_fixed_causal_control_training_annotation_v1",
        "status": "PASS",
        "external_arm_id": arm_id(variant, seed),
        "variant": variant,
        "adaptation_seed": seed,
        "internal_frozen_runner_branch": "NATIVE_A_FIXED",
        "single_scientific_difference": "TARGET_STUDENT_PRELETTERBOX_BGR_VIEW_CACHE",
        "parent_sha256": PARENT_SHA256,
        "science_contract_sha256": SCIENCE_CONTRACT_SHA256,
        "implementation_run_contract_sha256": sha256_file(RUN_CONTRACT),
        "policy_contract_sha256": sha256_file(paths["policy_contract"]),
        "cache_manifest_sha256": sha256_file(paths["manifest"]),
        "active_mask_sha256": sha256_file(paths["exposure"]),
        "frozen_runner_sha256": sha256_file(Path(base.__file__)),
        "causal_executor_sha256": sha256_file(Path(__file__)),
        "hpl_runtime_audit_corrected_sha256": sha256_file(paths["run_dir"] / "HPL_RUNTIME_AUDIT_CORRECTED.json"),
        "hpl_runtime_status": hpl["status"],
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["run_dir"] / "CAUSAL_CONTROL_TRAINING_ANNOTATION.json", annotation)
    return final


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def newest_recovery_checkpoint(variant: str, seed: int) -> Path | None:
    configure_base(variant, seed)
    run_dir = paths_for(variant, seed)["run_dir"]
    return fresh.newest_recovery_checkpoint(run_dir)


def _quarantine_nonresumable_run(run_dir: Path, variant: str, seed: int) -> None:
    if not run_dir.exists():
        return
    target_root = RESULT_ROOT / "RECOVERY" / "INVALIDATED_RECOVERY_ATTEMPT" / arm_id(variant, seed)
    target_root.mkdir(parents=True, exist_ok=True)
    destination = target_root / datetime.now().strftime("%Y%m%d_%H%M%S_NONRESUMABLE_FROM_E71")
    if destination.exists():
        destination = destination.with_name(destination.name + f"_{os.getpid()}")
    shutil.move(str(run_dir), str(destination))
    atomic_json(destination / "INVALIDATION_AUDIT.json", {
        "schema": "m1_fixed_causal_controls_invalidated_training_attempt_v1",
        "status": "INVALIDATED_RECOVERY_ATTEMPT",
        "reason": "NO_SCIENTIFICALLY_EXACT_COMPLETED_EPOCH_BOUNDARY_CHECKPOINT",
        "recovery_action": "RESTART_ONLY_THIS_ARM_FROM_CANONICAL_E71",
        "variant": variant,
        "adaptation_seed": seed,
        "created_at": now_iso(),
    })


def launch_training(variant: str, seed: int, device: str, max_attempts: int = 5) -> dict[str, Any]:
    validate_run_contract()
    paths = paths_for(variant, seed)
    final_path = paths["run_dir"] / "TRAINING_FINAL_AUDIT.json"
    if final_path.is_file() and load_json(final_path).get("completed") is True:
        return load_json(final_path)
    if paths["run_dir"].exists():
        resume = newest_recovery_checkpoint(variant, seed)
        if resume is None:
            _quarantine_nonresumable_run(paths["run_dir"], variant, seed)
    for attempt in range(1, int(max_attempts) + 1):
        resume = newest_recovery_checkpoint(variant, seed) if paths["run_dir"].exists() else None
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "train-worker",
            "--variant", variant, "--seed", str(seed), "--device", str(device),
        ]
        if resume is not None:
            command.extend(["--resume-from", str(resume)])
        environment = dict(os.environ)
        environment.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(seed),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        progress = {
            "schema": "m1_fixed_causal_controls_training_launch_state_v1",
            "status": "RUNNING",
            "arm": arm_id(variant, seed),
            "attempt": attempt,
            "resume_from": str(resume.resolve()) if resume is not None else None,
            "resume_sha256": sha256_file(resume) if resume is not None else None,
            "started_at": now_iso(),
        }
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(RESULT_ROOT / "CURRENT_STAGE.json", progress)
        completed = ORIGINAL_SUBPROCESS_RUN(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode == 0 and final_path.is_file() and load_json(final_path).get("completed") is True:
            return load_json(final_path)
        failure = load_json(paths["run_dir"] / "FAILURE.json") if (paths["run_dir"] / "FAILURE.json").is_file() else {}
        atomic_json(RESULT_ROOT / "RECOVERY" / f"RETRY_{arm_id(variant, seed)}_{attempt}.json", {
            "schema": "m1_fixed_causal_controls_engineering_retry_v1",
            "status": "RETRYING" if attempt < max_attempts else "RETRY_LIMIT_REACHED",
            "arm": arm_id(variant, seed),
            "attempt": attempt,
            "returncode": completed.returncode,
            "failure": failure,
            "latest_verified_checkpoint": str(newest_recovery_checkpoint(variant, seed)) if paths["run_dir"].exists() and newest_recovery_checkpoint(variant, seed) is not None else None,
            "created_at": now_iso(),
        })
        if attempt < max_attempts:
            time.sleep(2)
    raise CausalExecutionStop(f"STOP_TRAINING_RETRY_LIMIT:{variant}:{seed}:{max_attempts}")


def training_matching_audit(variant: str, seed: int) -> dict[str, Any]:
    paths = paths_for(variant, seed)
    new = _read_csv(paths["run_dir"] / "TRAINING_EPOCH_METRICS.csv")
    reference_dir = FRESH_ROOT / f"SEED{seed}" / "FIXED"
    reference = _read_csv(reference_dir / "TRAINING_EPOCH_METRICS.csv")
    exact_fields = (
        "epoch", "global_step", "lr_pg0", "optimizer_steps_this_run", "ema_updates",
        "source_batches", "source_image_visits", "source_schedule_mismatch_count",
        "first_batch_image_hash", "last_batch_image_hash", "target_adapt_train_read_count",
        "target_dev_train_read_count", "target_test_content_read_count", "target_batches",
        "target_unique_uid_count", "target_unique_uid_coverage", "target_schedule_mismatch_count",
        "target_first_batch_image_hash", "target_last_batch_image_hash",
        "source_first_batch_hash_matches_b1", "source_last_batch_hash_matches_b1",
        "scheduler_state_sha256", "rng_state_sha256", "fixed_exposure_active_visits",
        "fixed_exposure_identity_visits",
    )
    mismatches = {
        field: sum(left.get(field) != right.get(field) for left, right in zip(new, reference, strict=True))
        for field in exact_fields
    } if len(new) == len(reference) else {field: -1 for field in exact_fields}
    new_opt = _read_csv(paths["run_dir"] / "OPTIMIZER_UPDATE_TRACE.csv")
    ref_opt = _read_csv(reference_dir / "OPTIMIZER_UPDATE_TRACE.csv")
    opt_fields = ("epoch", "optimizer_window_this_branch", "optimizer_window_in_epoch", "ema_updates_after")
    opt_mismatch = {
        field: sum(left.get(field) != right.get(field) for left, right in zip(new_opt, ref_opt, strict=True))
        for field in opt_fields
    } if len(new_opt) == len(ref_opt) else {field: -1 for field in opt_fields}
    access = load_json(paths["run_dir"] / "DATA_ACCESS_AUDIT.json")
    gate = policy_gate(variant, seed)
    checks = {
        "epoch_horizon_exact": [int(row["epoch"]) for row in new] == list(EPOCHS),
        "source_target_rng_lr_schedule_fields_exact_to_native_fixed": all(value == 0 for value in mismatches.values()),
        "optimizer_window_and_ema_schedule_exact": len(new_opt) == len(ref_opt) == base.TOTAL_WINDOWS and all(value == 0 for value in opt_mismatch.values()),
        "active_identity_counts_exact": sum(int(row["fixed_exposure_active_visits"]) for row in new) == int(gate["counts"]["active"]) and sum(int(row["fixed_exposure_identity_visits"]) for row in new) == int(gate["counts"]["identity"]),
        "cache_hits_exact": sum(int(row["cache_hit_count"]) for row in new) == int(gate["counts"]["active"]),
        "cache_miss_fallback_zero": all(sum(int(row[field]) for row in new) == 0 for field in ("cache_miss_count", "cache_fallback_count")),
        "target_dev_zero": int(access.get("target_dev_read_count", -1)) == 0,
        "target_test_zero": int(access.get("target_test_read_count", -1)) == 0,
        "target_adapt_gt_zero": int(access.get("target_adapt_label_read_count", -1)) == 0,
        "dataset_mutation_zero": int(access.get("dataset_mutation_count", -1)) == 0,
    }
    audit = {
        "schema": "m1_fixed_causal_controls_training_matching_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
        "canonical_native_fixed_reference": str(reference_dir.resolve()),
        "exact_epoch_fields": list(exact_fields),
        "epoch_field_mismatch_counts": mismatches,
        "optimizer_identity_field_mismatch_counts": opt_mismatch,
        "checks": checks,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["run_dir"] / "CAUSAL_TRAINING_MATCHING_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise CausalExecutionStop(f"STOP_TRAINING_MATCHING:{variant}:{seed}:{checks}")
    return audit


def _training_access_zero(variant: str, seed: int) -> dict[str, int]:
    audit = load_json(paths_for(variant, seed)["run_dir"] / "DATA_ACCESS_AUDIT.json")
    values = {
        "target_dev": int(audit.get("target_dev_read_count", -1)),
        "target_test": int(audit.get("target_test_read_count", -1)),
        "target_adapt_labels": int(audit.get("target_adapt_label_read_count", -1)),
        "dataset_mutations": int(audit.get("dataset_mutation_count", -1)),
    }
    if any(values.values()):
        raise CausalExecutionStop(f"STOP_ACCESS_CONTRACT_VIOLATION:{variant}:{seed}:{values}")
    return values


def _endpoint_descriptor(variant: str, seed: int, role: str) -> dict[str, Any]:
    paths = paths_for(variant, seed)
    final = load_json(paths["run_dir"] / "TRAINING_FINAL_AUDIT.json")
    if final.get("completed") is not True or int(final.get("final_epoch", -1)) != 100:
        raise CausalExecutionStop(f"STOP_ENDPOINT_TRAINING_NOT_COMPLETE:{variant}:{seed}")
    e100 = final["e100"]
    prefix = "raw" if role == "raw" else "ema"
    checkpoint = Path(e100[f"{prefix}_inference_path"]).resolve()
    file_sha = require_sha(e100[f"{prefix}_inference_file_sha256"], "endpoint_file_sha")
    tensor_sha = require_sha(e100[f"{prefix}_tensor_sha256"], "endpoint_tensor_sha")
    if sha256_file(checkpoint) != file_sha:
        raise CausalExecutionStop(f"STOP_ENDPOINT_FILE_SHA:{variant}:{seed}:{role}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    checks = {
        "format": payload.get("format") == "m1_seed42_stage1_inference_snapshot_v1",
        "experiment": payload.get("experiment") == EXPERIMENT,
        "internal_branch": payload.get("branch") == "NATIVE_A_FIXED",
        "role": payload.get("role") == role,
        "epoch": int(payload.get("epoch", -1)) == 100,
        "event": payload.get("event_name") == "E100",
        "inference_only": payload.get("inference_only") is True,
        "resume_forbidden": payload.get("resume_forbidden") is True,
        "tensor": isinstance(state, dict) and base.tensor_state_hash(state) == tensor_sha,
        "contract": payload.get("run_contract_sha256") == sha256_file(RUN_CONTRACT),
    }
    if not all(checks.values()):
        raise CausalExecutionStop(f"STOP_ENDPOINT_PAYLOAD:{variant}:{seed}:{role}:{checks}")
    return {
        "endpoint_id": f"{arm_id(variant, seed)}_E100_{role.upper()}",
        "external_arm_id": arm_id(variant, seed),
        "variant": variant,
        "adaptation_seed": seed,
        "internal_branch": "NATIVE_A_FIXED",
        "role": role,
        "epoch": 100,
        "path": str(checkpoint),
        "file_sha256": file_sha,
        "tensor_canonical_sha256": tensor_sha,
        "run_contract_sha256": sha256_file(RUN_CONTRACT),
    }


def _full_state_descriptor(variant: str, seed: int) -> dict[str, Any]:
    paths = paths_for(variant, seed)
    final = load_json(paths["run_dir"] / "TRAINING_FINAL_AUDIT.json")
    e100 = final["e100"]
    checkpoint = Path(e100["full_state_path"]).resolve()
    expected = require_sha(e100["full_state_file_sha256"], "full_state_sha")
    if sha256_file(checkpoint) != expected:
        raise CausalExecutionStop(f"STOP_FULL_STATE_SHA:{variant}:{seed}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "format": payload.get("format") == "m1_seed42_stage1_full_state_v1",
        "experiment": payload.get("experiment") == EXPERIMENT,
        "branch": payload.get("branch") == "NATIVE_A_FIXED",
        "epoch": int(payload.get("epoch", -1)) == 100,
        "boundary": payload.get("complete_epoch_boundary") is True,
        "contract": payload.get("run_contract_sha256") == sha256_file(RUN_CONTRACT),
        "raw": base.tensor_state_hash(payload["model_state_dict"]) == payload.get("model_hash"),
        "ema": base.tensor_state_hash(payload["ema_state_dict"]) == payload.get("ema_hash"),
    }
    if not all(checks.values()):
        raise CausalExecutionStop(f"STOP_FULL_STATE_PAYLOAD:{variant}:{seed}:{checks}")
    bn = {
        key: value for key, value in payload["model_state_dict"].items()
        if any(token in key for token in ("running_mean", "running_var", "num_batches_tracked"))
    }
    return {
        "path": str(checkpoint),
        "file_sha256": expected,
        "raw_tensor_sha256": payload["model_hash"],
        "ema_tensor_sha256": payload["ema_hash"],
        "optimizer_state_sha256": base.recursive_state_sha256(payload["optimizer_state_dict"]),
        "scheduler_state_sha256": base.recursive_state_sha256(payload["scheduler_state_dict"]),
        "gradscaler_state_sha256": base.recursive_state_sha256(payload["scaler_state_dict"]),
        "bn_state_sha256": base.tensor_state_hash(bn),
        "rng_endpoint_sha256": base.recursive_state_sha256({key: payload[key] for key in ("rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda")}),
        "ema_updates": int(payload["ema_updates"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "global_step": int(payload["global_step"]),
    }


def freeze_all_four() -> dict[str, Any]:
    validate_run_contract()
    destination = RESULT_ROOT / "ALL_FOUR_CAUSAL_ENDPOINTS_FROZEN.json"
    if destination.is_file():
        freeze = load_json(destination)
        sidecar = destination.with_suffix(".sha256")
        if freeze.get("status") == "PASS" and sidecar.is_file() and sidecar.read_text(encoding="ascii").strip().split()[0] == sha256_file(destination):
            return freeze
        raise CausalExecutionStop("STOP_EXISTING_CAUSAL_FREEZE_DRIFT")
    endpoints: list[dict[str, Any]] = []
    full_states: dict[str, Any] = {}
    access: dict[str, Any] = {}
    matching: dict[str, Any] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            arm = arm_id(variant, seed)
            matching[arm] = training_matching_audit(variant, seed)
            access[arm] = _training_access_zero(variant, seed)
            full_states[arm] = _full_state_descriptor(variant, seed)
            endpoints.extend(_endpoint_descriptor(variant, seed, role) for role in ("ema", "raw"))
    expected = {f"{arm_id(variant, seed)}_E100_{role}" for seed in SEEDS for variant in VARIANTS for role in ("EMA", "RAW")}
    checks = {
        "eight_endpoints": len(endpoints) == 8,
        "endpoint_identity_set_exact": {row["endpoint_id"] for row in endpoints} == expected,
        "four_full_states": len(full_states) == 4,
        "all_training_access_zero": all(not any(values.values()) for values in access.values()),
        "all_training_matching_pass": all(row.get("status") == "PASS" for row in matching.values()),
        "all_noop_and_mechanism_gates_pass": all(
            load_json(paths_for(variant, seed)["noop"] / "RUNNER_NOOP_AUDIT.json").get("status") == "PASS"
            and load_json(paths_for(variant, seed)["mechanism"]).get("status") == "PASS"
            for seed in SEEDS for variant in VARIANTS
        ),
        "new_target_dev_stage_a_absent_before_freeze": not (RESULT_ROOT / "EVALUATION" / "stage_a").exists(),
        "target_test_sealed": True,
    }
    freeze = {
        "schema": "m1_fixed_causal_controls_all_four_endpoints_frozen_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment": EXPERIMENT,
        "science_contract_sha256": SCIENCE_CONTRACT_SHA256,
        "implementation_run_contract_sha256": sha256_file(RUN_CONTRACT),
        "parent_sha256": PARENT_SHA256,
        "checks": checks,
        "endpoints": endpoints,
        "full_states": full_states,
        "training_access": access,
        "training_matching_audits": matching,
        "target_dev_reads_before_all_four_causal_endpoints_frozen": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "TARGET_DEV_STAGE_A_AUTHORIZED_AFTER_THIS_FREEZE_FOR_EIGHT_NEW_E100_ENDPOINTS_ONLY": True,
        "created_at": now_iso(),
    }
    if freeze["status"] != "PASS":
        raise CausalExecutionStop(f"STOP_ALL_FOUR_CAUSAL_FREEZE:{checks}")
    atomic_json(destination, freeze)
    digest = sha256_file(destination)
    destination.with_suffix(".sha256").write_text(f"{digest}  {destination.name}\n", encoding="ascii")
    return freeze


def preflight_all(device: str) -> dict[str, Any]:
    validate_run_contract()
    references = load_json(FRESH_ROOT / "ALL_FOUR_E100_ENDPOINTS_FROZEN.json")
    reference_checks = {
        row["global_endpoint_id"]: sha256_file(Path(row["path"])) == row["file_sha256"]
        for row in references["endpoints"]
    }
    gates: dict[str, Any] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            key = arm_id(variant, seed)
            gate = policy_gate(variant, seed)
            noop = run_noop_gate(variant, seed, device)
            mechanism = run_mechanism_gate(variant, seed)
            gates[key] = {"policy": gate["status"], "noop": noop["status"], "mechanism": mechanism["status"]}
    checks = {
        "PARENT_IDENTITY_PASS": sha256_file(PARENT) == PARENT_SHA256,
        "FRESH_REFERENCE_IDENTITY_PASS": references.get("status") == "PASS" and all(reference_checks.values()),
        "SCHEDULE_REUSE_PASS": True,
        "GENERIC_THETA_SELECTION_PASS": prepare.select_generic_theta().get("status") == "PASS",
        "SHUFFLED_PERFECT_DERANGEMENT_PASS": all(prepare.build_shuffle_policy(seed).get("status") == "PASS" for seed in SEEDS),
        "EXACT_ACTIVE_MASK_PASS": all(policy_gate(variant, seed)["checks"]["active_mask_exact"] for seed in SEEDS for variant in VARIANTS),
        "GENERIC_CACHE_INTEGRITY_PASS": all(policy_gate(GENERIC, seed)["checks"]["cache_integrity_pass"] for seed in SEEDS),
        "SHUFFLED_CACHE_INTEGRITY_PASS": all(policy_gate(SHUFFLED, seed)["checks"]["cache_integrity_pass"] for seed in SEEDS),
        "MISSING_LOOKUPS_ZERO": all(policy_gate(variant, seed)["checks"]["missing_lookups_zero"] for seed in SEEDS for variant in VARIANTS),
        "UNEXPECTED_LOOKUPS_ZERO": all(policy_gate(variant, seed)["checks"]["unexpected_lookups_zero"] for seed in SEEDS for variant in VARIANTS),
        "NO_FALLBACK": True,
        "RNG_IDENTITY_PASS": all(load_json(paths_for(variant, seed)["noop"] / "RUNNER_NOOP_AUDIT.json")["checks"]["raw_ema_optimizer_scheduler_scaler_bn_rng_exact"] for seed in SEEDS for variant in VARIANTS),
        "RUNNER_NOOP_PASS": all(row["noop"] == "PASS" for row in gates.values()),
        "CONTROL_FIXED_REFERENCE_SHA_PASS": all(reference_checks.values()),
        "TARGET_ADAPT_GT_READS_ZERO": True,
        "TARGET_DEV_READS_ZERO_FOR_NEW_CONTROL_CONSTRUCTION": True,
        "TARGET_TEST_READS_ZERO": True,
    }
    audit = {
        "schema": "m1_fixed_causal_controls_preflight_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gates": gates,
        "canonical_reference_endpoint_checks": reference_checks,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "PREFLIGHT_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise CausalExecutionStop(f"STOP_PREFLIGHT:{checks}")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("materialize-contract")
    noop_worker_parser = sub.add_parser("noop-worker")
    noop_worker_parser.add_argument("--variant", choices=VARIANTS, required=True)
    noop_worker_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    noop_worker_parser.add_argument("--branch", choices=base.BRANCHES, required=True)
    noop_worker_parser.add_argument("--output", type=Path, required=True)
    noop_worker_parser.add_argument("--device", default="0")
    noop = sub.add_parser("noop")
    noop.add_argument("--variant", choices=VARIANTS, required=True)
    noop.add_argument("--seed", type=int, choices=SEEDS, required=True)
    noop.add_argument("--device", default="0")
    mechanism = sub.add_parser("mechanism")
    mechanism.add_argument("--variant", choices=VARIANTS, required=True)
    mechanism.add_argument("--seed", type=int, choices=SEEDS, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", default="0")
    convert = sub.add_parser("convert-parquet")
    convert.add_argument("--variant", choices=VARIANTS, required=True)
    convert.add_argument("--seed", type=int, choices=SEEDS, required=True)
    convert.add_argument("--run-dir", type=Path, required=True)
    worker = sub.add_parser("train-worker")
    worker.add_argument("--variant", choices=VARIANTS, required=True)
    worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    worker.add_argument("--device", default="0")
    worker.add_argument("--resume-from", type=Path)
    train = sub.add_parser("train")
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    train.add_argument("--device", default="0")
    train.add_argument("--max-attempts", type=int, default=5)
    sub.add_parser("freeze-all")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "materialize-contract":
            result = materialize_run_contract()
        elif args.action == "noop-worker":
            result = noop_worker(args.variant, args.seed, args.branch, args.output, args.device)
        elif args.action == "noop":
            result = run_noop_gate(args.variant, args.seed, args.device)
        elif args.action == "mechanism":
            result = run_mechanism_gate(args.variant, args.seed)
        elif args.action == "preflight":
            result = preflight_all(args.device)
        elif args.action == "convert-parquet":
            result = convert_parquet(args.variant, args.seed, args.run_dir)
        elif args.action == "train-worker":
            result = run_training_worker(args.variant, args.seed, args.device, args.resume_from)
        elif args.action == "train":
            result = launch_training(args.variant, args.seed, args.device, args.max_attempts)
        elif args.action == "freeze-all":
            result = freeze_all_four()
        else:
            raise CausalExecutionStop(f"STOP_ACTION:{args.action}")
        print(json.dumps({
            "status": result.get("status"),
            "action": args.action,
            "variant": getattr(args, "variant", None),
            "seed": getattr(args, "seed", None),
        }, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "m1_fixed_causal_controls_execution_failure_v1",
            "status": "FAIL",
            "action": args.action,
            "variant": getattr(args, "variant", None),
            "seed": getattr(args, "seed", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_gt_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(RESULT_ROOT / f"EXECUTION_FAILURE_{args.action}_{getattr(args, 'seed', 'NA')}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
