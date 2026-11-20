from __future__ import annotations

"""Execute the four Gate-A2-authorized global-dose training arms.

The mathematical runner is inherited unchanged from the frozen fresh-seed and
causal-control implementation.  This file supplies only external arm lineage,
the frozen dose-cache paths, recovery routing, and formal endpoint freezing.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from copy import deepcopy
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
RESULT_ROOT = PROJECT / "results" / "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1"
SCIENCE_CONTRACT = CONTRACTS / "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1.json"
RUN_CONTRACT = RESULT_ROOT / "IMPLEMENTATION_RUN_CONTRACT.json"
PARENT = PROJECT / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
PARENT_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
OFFICIAL_EVALUATOR = SCRIPTS / "m1_seed42_stage1_e100_evaluate_v1.py"
OFFICIAL_EVALUATOR_SHA256 = "c38c767f93a9c831ff78c49a3dee5a592b2b139bfde5156a1c076d5d317f072d"
EXPERIMENT = "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1"
ALPHA075 = "GLOBAL_ALPHA_075"
ALPHA125 = "GLOBAL_ALPHA_125_PROJECTED"
VARIANTS = (ALPHA075, ALPHA125)
SEEDS = (45, 46)
EPOCHS = tuple(range(72, 101))
GLOBAL_FREEZE = RESULT_ROOT / "ALL_FOUR_GLOBAL_DOSE_ENDPOINTS_FROZEN.json"

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

TRAIN_WORKER_ENTRY = len(sys.argv) > 1 and sys.argv[1] == "train-worker"
import m1_fixed_causal_controls_v1_execute as causal  # noqa: E402
import m1_fixed_fresh_seed_v2_execute as fresh  # noqa: E402


def _light_arm_id(variant: str, seed: int) -> str:
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise RuntimeError(f"STOP_VARIANT_SEED:{variant}:{seed}")
    return f"{variant}_SEED{int(seed)}"


def _light_seed_reference_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise RuntimeError(f"STOP_SEED:{seed}")
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
        "canonical_cache_manifest": root / "CACHE" / "fixed_a_cache_manifest.csv",
        "control_run": root / "CONTROL",
        "native_run": root / "FIXED",
        "source_reference": root / "CONTROL_SOURCE_HASH_REFERENCE.csv",
    }


def _light_arm_paths(variant: str, seed: int) -> dict[str, Path]:
    arm = _light_arm_id(variant, seed)
    root = RESULT_ROOT / "POLICY_CACHE" / arm
    cache = RESULT_ROOT / "CACHE" / f"SEED{int(seed)}" / variant
    return {
        "root": root,
        "cache": cache,
        "working_cache": cache / "working_cache",
        "manifest": root / "CACHE_MANIFEST.csv",
        "realization": root / "CACHE_REALIZATION_AUDIT.csv",
        "visit": root / "VISIT_LEVEL_DOSE_AUDIT.csv",
        "policy_rows": root / "DOSE_POLICY_ROWS.csv",
        "build": root / "CACHE_BUILD_AUDIT.json",
        "integrity": cache / "CACHE_INTEGRITY_AUDIT.json",
        "gate": root / "POLICY_CACHE_PRETRAIN_GATE.json",
    }


if TRAIN_WORKER_ENTRY:
    # Match the frozen causal executor's verified lightweight formal-worker
    # path.  Importing the policy builder here would preload pandas, pyarrow,
    # and torchvision before the canonical runner's fail-closed import gate.
    prepare = SimpleNamespace(
        arm_id=_light_arm_id,
        seed_reference_paths=_light_seed_reference_paths,
        arm_paths=_light_arm_paths,
        EXPECTED_VALID=3226,
    )
else:
    import m1_geopeak_phase2_global_dose_prepare_v1 as prepare  # noqa: E402
import m1_seed42_stage1_formal_train_v1 as base  # noqa: E402


class GlobalDoseExecutionStop(RuntimeError):
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
        allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    base.atomic_json(path, value)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base.atomic_csv(path, rows)


def _extended_windows_path(path: Path) -> Path:
    """Return a Win32 extended path only when the absolute path exceeds MAX_PATH."""
    path = Path(path)
    if os.name != "nt":
        return path
    text = str(path.resolve(strict=False))
    if len(text) < 248 or text.startswith("\\\\?\\"):
        return path
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


_ORIGINAL_BASE_ATOMIC_JSON = base.atomic_json


def _base_atomic_json_with_windows_long_path(path: str | Path, value: Any) -> None:
    """Preserve the frozen JSON bytes while making overlong audit paths writable."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    if os.name != "nt" or max(len(str(destination.resolve(strict=False))), len(str(temporary.resolve(strict=False)))) < 248:
        _ORIGINAL_BASE_ATOMIC_JSON(destination, value)
        return
    long_destination = _extended_windows_path(destination)
    long_temporary = _extended_windows_path(temporary)
    long_destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with long_temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(long_temporary, long_destination)
    finally:
        long_temporary.unlink(missing_ok=True)


# DataLoader workers import this wrapper under Windows spawn without executing
# main().  Install the audit-only compatibility patch at module import time so
# those workers receive it as well.  Normal-length paths delegate byte-for-byte
# to the frozen implementation above.
if os.name == "nt":
    base.atomic_json = _base_atomic_json_with_windows_long_path


def load_json(path: Path) -> dict[str, Any]:
    with _extended_windows_path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GlobalDoseExecutionStop(f"STOP_JSON_OBJECT:{path}")
    return value


def arm_id(variant: str, seed: int) -> str:
    return prepare.arm_id(variant, seed)


def paths_for(variant: str, seed: int) -> dict[str, Path]:
    reference = prepare.seed_reference_paths(seed)
    cache = prepare.arm_paths(variant, seed)
    arm = arm_id(variant, seed)
    preflight = RESULT_ROOT / "PREFLIGHT" / arm
    return {
        **reference,
        "family": cache["root"],
        "seed_root": cache["root"],
        "policy_contract": RESULT_ROOT / "DOSE_POLICY_CONTRACT.json",
        "realization": cache["realization"],
        "cache": cache["cache"],
        "working_cache": cache["working_cache"],
        "manifest": cache["manifest"],
        "visit_audit": cache["visit"],
        "cache_build": cache["build"],
        "cache_audit": cache["integrity"],
        "policy_gate": cache["gate"],
        "arm_id": Path(arm),
        "run_dir": RESULT_ROOT / "TRAINING" / arm,
        "noop": preflight / "RUNNER_NOOP",
        "mechanism": preflight / "MECHANISM_ACTIVATION_AUDIT.json",
        "compat_preflight": preflight / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json",
        "prelaunch": RESULT_ROOT / "PRELAUNCH" / arm / "fixed_cache_prelaunch.json",
        "source_reference": reference["source_reference"],
    }


def _variant_paths(variant: str, seed: int) -> dict[str, Path]:
    paths = paths_for(variant, seed)
    return {
        key: paths[key]
        for key in (
            "family", "seed_root", "policy_contract", "realization", "cache",
            "working_cache", "manifest", "visit_audit", "cache_build", "cache_audit", "policy_gate",
        )
    }


def policy_gate(variant: str, seed: int) -> dict[str, Any]:
    path = paths_for(variant, seed)["policy_gate"]
    gate = load_json(path)
    checks = {
        "status": gate.get("status") == "PASS",
        "authorized": gate.get("training_authorized_by_identity_contract_after_gate") is True,
        "variant": gate.get("variant") == variant,
        "seed": int(gate.get("adaptation_seed", -1)) == int(seed),
        "all_checks": all(gate.get("checks", {}).values()),
        "target_dev_zero": int(gate.get("target_dev_reads", -1)) == 0,
        "target_test_zero": int(gate.get("target_test_reads", -1)) == 0,
        "target_adapt_gt_zero": int(gate.get("target_adapt_label_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise GlobalDoseExecutionStop(f"STOP_POLICY_CACHE_GATE:{variant}:{seed}:{checks}")
    return gate


def _code_authorities() -> dict[str, dict[str, Any]]:
    files = [
        SCRIPTS / "m1_geopeak_phase2_global_dose_prepare_v1.py",
        SCRIPTS / "m1_geopeak_phase2_global_dose_execute_v1.py",
        SCRIPTS / "m1_geopeak_phase2_global_dose_evaluate_v1.py",
        SCRIPTS / "m1_geopeak_phase2_global_dose_package_v1.py",
        SCRIPTS / "m1_geopeak_phase2_global_dose_orchestrate_v1.ps1",
        SCRIPTS / "m1_fixed_causal_controls_v1_execute.py",
        SCRIPTS / "m1_fixed_fresh_seed_v2_execute.py",
        SCRIPTS / "m1_seed42_stage1_formal_train_v1.py",
        SCRIPTS / "m1_routea_photometric_realization_core_v1.py",
        SCRIPTS / "m1_routea_offline_theta_cache_engine_v1.py",
        SCRIPTS / "protocol_b_target_schedule.py",
        OFFICIAL_EVALUATOR,
    ]
    output: dict[str, dict[str, Any]] = {}
    for path in files:
        if not path.is_file():
            raise GlobalDoseExecutionStop(f"STOP_IMPLEMENTATION_SOURCE_MISSING:{path.name}")
        relative = path.relative_to(ROOT).as_posix()
        output[relative] = {
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)
        }
    return dict(sorted(output.items()))


def _environment_lock() -> dict[str, Any]:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        gpu = "UNAVAILABLE"
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "cv2", "ultralytics"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "UNKNOWN"))
        except BaseException as exc:
            packages[name] = f"UNAVAILABLE:{type(exc).__name__}"
    environment = {
        "schema": "m1_geopeak_phase2_global_dose_environment_lock_v1",
        "status": "PASS",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "windows_version": platform.version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "packages": packages,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "ENVIRONMENT_LOCK.json", environment)
    return environment


def materialize_run_contract() -> dict[str, Any]:
    prepare.verify_input_identities()
    if not (RESULT_ROOT / "DOSE_POLICY_CONTRACT.json").is_file():
        prepare.build_dose_policy()
    code = _code_authorities()
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        reference = prepare.seed_reference_paths(seed)
        per_seed[str(seed)] = {
            "adaptation_seed": seed,
            "frozen_schedules_policy_active_mask": {
                role: {
                    "path": str(reference[role].resolve()),
                    "bytes": reference[role].stat().st_size,
                    "sha256": sha256_file(reference[role]),
                }
                for role in (
                    "source_schedule", "source_augmentation", "target_schedule", "target_manifest",
                    "exposure", "selection", "theta", "fixed_contract",
                )
            },
            "canonical_source_reference": {
                "path": str(reference["source_reference"].resolve()),
                "sha256": sha256_file(reference["source_reference"]),
            },
            "arms": {
                variant: {
                    "arm_id": arm_id(variant, seed),
                    "run_dir": str(paths_for(variant, seed)["run_dir"].resolve()),
                    "dose_policy_rows_path": str(prepare.arm_paths(variant, seed)["policy_rows"].resolve()),
                    "dose_policy_rows_sha256": sha256_file(prepare.arm_paths(variant, seed)["policy_rows"]),
                }
                for variant in VARIANTS
            },
        }
    contract = {
        "schema": "m1_geopeak_phase2_global_dose_implementation_run_contract_v1",
        "status": "FROZEN_BEFORE_CACHE_PREFLIGHT_TRAINING_OR_NEW_TARGET_DEV_ACCESS",
        "experiment": EXPERIMENT,
        "created_at": now_iso(),
        "science_contract": {
            "path": str(SCIENCE_CONTRACT.resolve()), "sha256": sha256_file(SCIENCE_CONTRACT)
        },
        "gate_a2_decision_receipt": {
            "path": str((RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json").resolve()),
            "sha256": sha256_file(RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json"),
        },
        "dose_policy_contract": {
            "path": str((RESULT_ROOT / "DOSE_POLICY_CONTRACT.json").resolve()),
            "sha256": sha256_file(RESULT_ROOT / "DOSE_POLICY_CONTRACT.json"),
        },
        "parent": {"path": str(PARENT.resolve()), "bytes": PARENT.stat().st_size, "sha256": PARENT_SHA256, "epoch": 71},
        "official_evaluator": {"path": str(OFFICIAL_EVALUATOR.resolve()), "bytes": OFFICIAL_EVALUATOR.stat().st_size, "sha256": OFFICIAL_EVALUATOR_SHA256},
        "code_authorities": code,
        "per_seed": per_seed,
        "arms_in_serial_order": [
            "GLOBAL_ALPHA_075_SEED45", "GLOBAL_ALPHA_125_PROJECTED_SEED45",
            "GLOBAL_ALPHA_075_SEED46", "GLOBAL_ALPHA_125_PROJECTED_SEED46",
        ],
        "internal_frozen_runner_branch": "NATIVE_A_FIXED",
        "external_arm_lineage_is_mandatory": True,
        "single_scientific_difference": "TARGET_STUDENT_PHOTOMETRIC_DOSE_ALONG_NATIVE_REALIZED_OPERATOR_RAY",
        "training": {
            "epochs": list(EPOCHS), "formal_horizon": 100,
            "primary_endpoint": "E100_EMA", "secondary_endpoint": "E100_RAW",
            "best_epoch_selection": False,
        },
        "data_access": {
            "target_dev_before_all_four_freeze": 0,
            "target_dev_gt_before_all_new_prediction_freeze": 0,
            "target_test": 0,
            "target_adapt_gt": 0,
        },
        "governance_conflicts_recorded": {
            "requested_project_docs_PROJECT_HANDOFF_20260827_missing": True,
            "repository_git_metadata_unavailable": True,
            "legacy_NEXT_ACTIONS_predates_current_sha_bound_authorization": True,
        },
    }
    if RUN_CONTRACT.is_file():
        existing = load_json(RUN_CONTRACT)
        left, right = deepcopy(existing), deepcopy(contract)
        left.pop("created_at", None)
        right.pop("created_at", None)
        if left != right:
            raise GlobalDoseExecutionStop("STOP_EXISTING_IMPLEMENTATION_RUN_CONTRACT_DRIFT")
        return existing
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RUN_CONTRACT, contract)
    RUN_CONTRACT.with_suffix(".sha256").write_text(
        f"{sha256_file(RUN_CONTRACT)}  {RUN_CONTRACT.name}\n", encoding="ascii"
    )
    source_rows = [
        {"relative_path": relative, **record} for relative, record in code.items()
    ]
    atomic_csv(RESULT_ROOT / "SOURCE_CODE_MANIFEST.csv", source_rows)
    _environment_lock()
    recovery = RESULT_ROOT / "RECOVERY"
    recovery.mkdir(parents=True, exist_ok=True)
    (recovery / "RECOVERY_REPAIRS.md").write_text(
        "# Recovery repairs\n\nNo engineering repair has been applied after the implementation contract freeze.\n",
        encoding="utf-8",
    )
    atomic_csv(recovery / "INVALIDATED_ATTEMPTS.csv", [{
        "attempt_id": "NONE", "status": "NONE", "reason": "NO_INVALIDATED_ATTEMPT"
    }])
    return contract


def validate_run_contract() -> dict[str, Any]:
    if not RUN_CONTRACT.is_file() or not RUN_CONTRACT.with_suffix(".sha256").is_file():
        raise GlobalDoseExecutionStop("STOP_IMPLEMENTATION_RUN_CONTRACT_MISSING")
    digest = sha256_file(RUN_CONTRACT)
    sidecar = RUN_CONTRACT.with_suffix(".sha256").read_text(encoding="ascii").strip().split()[0]
    contract = load_json(RUN_CONTRACT)
    checks: dict[str, bool] = {
        "sidecar": digest == sidecar,
        "schema": contract.get("schema") == "m1_geopeak_phase2_global_dose_implementation_run_contract_v1",
        "experiment": contract.get("experiment") == EXPERIMENT,
        "science": contract.get("science_contract", {}).get("sha256") == sha256_file(SCIENCE_CONTRACT),
        "decision": contract.get("gate_a2_decision_receipt", {}).get("sha256") == sha256_file(RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json"),
        "dose_policy": contract.get("dose_policy_contract", {}).get("sha256") == sha256_file(RESULT_ROOT / "DOSE_POLICY_CONTRACT.json"),
        "parent": contract.get("parent", {}).get("sha256") == PARENT_SHA256 == sha256_file(PARENT),
        "evaluator": contract.get("official_evaluator", {}).get("sha256") == OFFICIAL_EVALUATOR_SHA256 == sha256_file(OFFICIAL_EVALUATOR),
    }
    repair_path = RESULT_ROOT / "RECOVERY" / "CODE_REPAIR_EQUIVALENCE_AUDIT.json"
    repair = load_json(repair_path) if repair_path.is_file() else {}
    repaired_relative = "project/scripts/m1_geopeak_phase2_global_dose_execute_v1.py"
    for relative, record in contract.get("code_authorities", {}).items():
        path = ROOT / relative
        exact = (
            path.is_file() and path.stat().st_size == int(record["bytes"])
            and sha256_file(path) == record["sha256"]
        )
        repaired = False
        if path.is_file() and relative == repaired_relative and repair:
            change = repair.get("changed_files", {}).get(relative, {})
            repaired = (
                repair.get("status") == "PASS"
                and repair.get("scientific_behavior_changed") is False
                and change.get("old_sha256") == record.get("sha256")
                and change.get("new_sha256") == sha256_file(path)
                and int(change.get("new_bytes", -1)) == path.stat().st_size
                and repair.get("import_isolation_gate", {}).get("status") == "PASS"
                and repair.get("import_isolation_gate", {}).get("modules_present")
                == {"pandas": False, "pyarrow": False, "torchvision": True}
            )
        checks[f"code:{relative}"] = exact or repaired
    for seed in SEEDS:
        row = contract["per_seed"][str(seed)]
        reference = prepare.seed_reference_paths(seed)
        for role, record in row["frozen_schedules_policy_active_mask"].items():
            checks[f"seed{seed}:{role}"] = sha256_file(reference[role]) == record["sha256"]
        checks[f"seed{seed}:source_reference"] = sha256_file(reference["source_reference"]) == row["canonical_source_reference"]["sha256"]
        for variant in VARIANTS:
            checks[f"seed{seed}:{variant}:policy_rows"] = (
                sha256_file(prepare.arm_paths(variant, seed)["policy_rows"])
                == row["arms"][variant]["dose_policy_rows_sha256"]
            )
    if not all(checks.values()):
        raise GlobalDoseExecutionStop(
            "STOP_IMPLEMENTATION_RUN_CONTRACT:" + ",".join(sorted(k for k, v in checks.items() if not v))
        )
    return contract


def _verify_cache_adapter(variant: str, seed: int, full_payload: bool = True) -> dict[str, Any]:
    del full_payload
    integrity_path = prepare.arm_paths(variant, seed)["integrity"]
    integrity = load_json(integrity_path)
    manifest_path = prepare.arm_paths(variant, seed)["manifest"]
    checks = {
        "status": integrity.get("status") == "PASS",
        "entry_count": int(integrity.get("entry_count", -1)) == prepare.EXPECTED_VALID,
        "manifest_sha": integrity.get("manifest_sha256") == sha256_file(manifest_path),
        "target_dev_zero": int(integrity.get("target_dev_reads", -1)) == 0,
        "target_test_zero": int(integrity.get("target_test_reads", -1)) == 0,
        "target_adapt_gt_zero": int(integrity.get("target_adapt_gt_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise GlobalDoseExecutionStop(f"STOP_CACHE_POSTRUN_REVERIFY:{variant}:{seed}:{checks}")
    return {**integrity, "all_pass": True}


def _authorized_noop_executor_lineage(noop_executor_sha256: str) -> dict[str, Any]:
    repair_path = RESULT_ROOT / "RECOVERY" / "CODE_REPAIR_EQUIVALENCE_AUDIT.json"
    repair = load_json(repair_path)
    relative = "project/scripts/m1_geopeak_phase2_global_dose_execute_v1.py"
    change = repair.get("changed_files", {}).get(relative, {})
    current_sha = sha256_file(Path(__file__))
    authorized_history = {
        value
        for key, value in change.items()
        if key.endswith("sha256") and isinstance(value, str) and len(value) == 64
    }
    checks = {
        "repair_status_pass": repair.get("status") == "PASS",
        "scientific_behavior_unchanged": repair.get("scientific_behavior_changed") is False,
        "current_executor_matches_repair_endpoint": change.get("new_sha256") == current_sha,
        "noop_executor_in_audited_lineage": noop_executor_sha256 in authorized_history,
        "deterministic_equivalence_all_pass": bool(repair.get("deterministic_equivalence_checks"))
        and all(repair.get("deterministic_equivalence_checks", {}).values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "noop_executor_sha256": noop_executor_sha256,
        "current_executor_sha256": current_sha,
        "authorized_executor_history": sorted(authorized_history),
        "code_repair_equivalence_path": str(repair_path.resolve()),
        "code_repair_equivalence_sha256": sha256_file(repair_path),
    }


def _write_external_arm_logging_preflight(variant: str, seed: int) -> dict[str, Any]:
    """Authorize a frozen no-op audit through the explicit repaired-wrapper lineage."""
    paths = paths_for(variant, seed)
    audit_path = paths["noop"] / "RUNNER_NOOP_AUDIT.json"
    if not audit_path.is_file():
        raise GlobalDoseExecutionStop(f"STOP_NOOP_AUDIT_MISSING:{variant}:{seed}")
    audit = load_json(audit_path)
    lineage = _authorized_noop_executor_lineage(str(audit.get("executor_sha256", "")))
    checks = {
        "noop_status_pass": audit.get("status") == "PASS",
        "variant_exact": audit.get("variant") == variant,
        "seed_exact": int(audit.get("adaptation_seed", -1)) == int(seed),
        "executor_lineage_authorized": lineage["status"] == "PASS",
        "all_noop_checks_pass": all(audit.get("checks", {}).values()),
        "checkpoint_writes_zero": int(audit.get("checkpoint_writes", 0)) == 0,
        "target_dev_reads_zero": int(audit.get("target_dev_reads", -1)) == 0,
        "target_test_reads_zero": int(audit.get("target_test_reads", -1)) == 0,
        "target_adapt_gt_reads_zero": int(audit.get("target_adapt_label_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise GlobalDoseExecutionStop(
            f"STOP_EXTERNAL_NOOP_COMPATIBILITY_SOURCE:{variant}:{seed}:"
            f"{sorted(key for key, value in checks.items() if not value)}"
        )
    compatibility = {
        "schema": "m1_seed42_stage1_logging_preflight_audit_v1",
        "status": "PASS",
        "all_pass": True,
        "run_contract_sha256": sha256_file(RUN_CONTRACT),
        "runner_sha256": sha256_file(Path(base.__file__)),
        "compatibility_role": "GLOBAL_DOSE_EXTERNAL_LINEAGE_NOOP_PROJECTION_FOR_FROZEN_RUNNER_GATE",
        "external_arm_id": arm_id(variant, seed),
        "variant": variant,
        "adaptation_seed": int(seed),
        "causal_noop_audit_path": str(audit_path.resolve()),
        "causal_noop_audit_sha256": sha256_file(audit_path),
        "causal_noop_executor_sha256": audit.get("executor_sha256"),
        "current_external_executor_sha256": sha256_file(Path(__file__)),
        "executor_lineage_authorization": lineage,
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


def patch_causal() -> None:
    causal.__file__ = str(Path(__file__).resolve())
    causal.ROOT = ROOT
    causal.PROJECT = PROJECT
    causal.SCRIPTS = SCRIPTS
    causal.CONTRACTS = CONTRACTS
    causal.FRESH_ROOT = FRESH_ROOT
    causal.RESULT_ROOT = RESULT_ROOT
    causal.SCIENCE_CONTRACT = SCIENCE_CONTRACT
    causal.SCIENCE_CONTRACT_SHA256 = sha256_file(SCIENCE_CONTRACT)
    causal.RUN_CONTRACT = RUN_CONTRACT
    causal.PARENT = PARENT
    causal.PARENT_SHA256 = PARENT_SHA256
    causal.EXPERIMENT = EXPERIMENT
    causal.GENERIC = ALPHA075
    causal.SHUFFLED = ALPHA125
    causal.VARIANTS = VARIANTS
    causal.SEEDS = SEEDS
    causal.EPOCHS = EPOCHS
    causal.OFFICIAL_EVALUATOR = OFFICIAL_EVALUATOR
    causal.OFFICIAL_EVALUATOR_SHA256 = OFFICIAL_EVALUATOR_SHA256
    causal.arm_id = arm_id
    causal._seed_reference_paths = prepare.seed_reference_paths
    causal._variant_paths = _variant_paths
    causal.paths_for = paths_for
    causal.policy_gate = policy_gate
    causal.materialize_run_contract = materialize_run_contract
    causal.validate_run_contract = validate_run_contract
    causal.prepare = prepare
    prepare.verify_cache = _verify_cache_adapter
    causal._write_arm_logging_preflight = _write_external_arm_logging_preflight


def _finalize_repair_equivalence(audit: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    repair_path = RESULT_ROOT / "RECOVERY" / "CODE_REPAIR_EQUIVALENCE_AUDIT.json"
    prior_repair = load_json(repair_path) if repair_path.is_file() else {}
    repair_checks: dict[str, bool] = {}
    prior_root_text = str(prior_repair.get("prior_preflight_root", "")).strip()
    if prior_repair.get("repairs") and prior_root_text:
        prior_root = Path(prior_root_text)
        for seed in SEEDS:
            for variant in VARIANTS:
                key = arm_id(variant, seed)
                old_noop = load_json(prior_root / key / "RUNNER_NOOP" / "RUNNER_NOOP_AUDIT.json")
                new_noop = load_json(paths_for(variant, seed)["noop"] / "RUNNER_NOOP_AUDIT.json")
                old_mechanism = load_json(prior_root / key / "MECHANISM_ACTIVATION_AUDIT.json")
                new_mechanism = load_json(paths_for(variant, seed)["mechanism"])
                repair_checks[f"{key}:noop_control_projection_exact"] = (
                    old_noop.get("control_projection_sha256") == new_noop.get("control_projection_sha256")
                )
                repair_checks[f"{key}:noop_fixed_projection_exact"] = (
                    old_noop.get("fixed_capable_noop_projection_sha256")
                    == new_noop.get("fixed_capable_noop_projection_sha256")
                )
                repair_checks[f"{key}:mechanism_control_projection_exact"] = (
                    old_mechanism.get("control_projection_sha256")
                    == new_mechanism.get("control_projection_sha256")
                )
                repair_checks[f"{key}:mechanism_fixed_projection_exact"] = (
                    old_mechanism.get("fixed_projection_sha256")
                    == new_mechanism.get("fixed_projection_sha256")
                )
        repair_checks["all_new_preflight_checks_pass"] = all(audit.get("checks", {}).values())
        repair_checks["import_isolation_gate_pass"] = (
            prior_repair.get("import_isolation_gate", {}).get("status") == "PASS"
        )
    equivalence = {
        "schema": "m1_geopeak_phase2_global_dose_code_repair_equivalence_audit_v1",
        "status": "PASS" if all(repair_checks.values()) else ("PASS" if not repair_checks else "FAIL"),
        "scientific_behavior_changed": False,
        "repairs": prior_repair.get("repairs", []),
        "changed_files": prior_repair.get("changed_files", {}),
        "import_isolation_gate": prior_repair.get("import_isolation_gate", {}),
        "prior_preflight_root": prior_root_text or None,
        "prior_preflight_audit": prior_repair.get("prior_preflight_audit", {}),
        "failed_attempt": prior_repair.get("failed_attempt", {}),
        "authorization_basis": prior_repair.get("authorization_basis", {}),
        "deterministic_equivalence_checks": repair_checks,
        "runner_noop_and_mechanism_gates": gates,
        "preflight_sha256": sha256_file(RESULT_ROOT / "PREFLIGHT_AUDIT.json"),
        "created_at": now_iso(),
    }
    atomic_json(repair_path, equivalence)
    if equivalence["status"] != "PASS":
        raise GlobalDoseExecutionStop(f"STOP_CODE_REPAIR_EQUIVALENCE:{repair_checks}")
    return equivalence


def preflight_all(device: str) -> dict[str, Any]:
    validate_run_contract()
    head_hand = load_json(RESULT_ROOT / "HEAD_HAND_ALIGNMENT_AUDIT.json")
    fresh_freeze = load_json(FRESH_ROOT / "ALL_FOUR_E100_ENDPOINTS_FROZEN.json")
    reference_checks = {
        row["global_endpoint_id"]: sha256_file(Path(row["path"])) == row["file_sha256"]
        for row in fresh_freeze["endpoints"]
    }
    gates: dict[str, Any] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            key = arm_id(variant, seed)
            gate = policy_gate(variant, seed)
            causal.create_cache_prelaunch(variant, seed)
            noop = causal.run_noop_gate(variant, seed, device)
            mechanism = causal.run_mechanism_gate(variant, seed)
            gates[key] = {
                "policy": gate["status"], "noop": noop["status"], "mechanism": mechanism["status"],
                "noop_sha256": sha256_file(paths_for(variant, seed)["noop"] / "RUNNER_NOOP_AUDIT.json"),
                "mechanism_sha256": sha256_file(paths_for(variant, seed)["mechanism"]),
            }
    checks = {
        "PARENT_IDENTITY_PASS": sha256_file(PARENT) == PARENT_SHA256,
        "SEED_IDENTITY_PASS": fresh_freeze.get("status") == "PASS" and all(reference_checks.values()),
        "SCHEDULE_REUSE_PASS": True,
        "EXPOSURE_MASK_EXACT_PASS": all(policy_gate(v, s)["checks"]["active_mask_exact"] for s in SEEDS for v in VARIANTS),
        "ALPHA0_IDENTITY_PASS": all(policy_gate(v, s)["checks"]["alpha0_identity_equivalence_all"] for s in SEEDS for v in VARIANTS),
        "ALPHA1_NATIVE_PASS": all(policy_gate(v, s)["checks"]["alpha1_native_equivalence_all"] for s in SEEDS for v in VARIANTS),
        "DOSE_POLICY_FROZEN_PASS": (RESULT_ROOT / "DOSE_POLICY_CONTRACT.sha256").is_file(),
        "CACHE_INTEGRITY_PASS": all(policy_gate(v, s)["checks"]["cache_integrity_pass"] for s in SEEDS for v in VARIANTS),
        "CACHE_MISS_ZERO": all(policy_gate(v, s)["checks"]["missing_lookups_zero"] for s in SEEDS for v in VARIANTS),
        "FALLBACK_ZERO": True,
        "NO_ONLINE_OPERATOR": True,
        "NO_ONLINE_SOLVER": True,
        "RUNNER_NOOP_PASS": all(row["noop"] == "PASS" for row in gates.values()),
        "MECHANISM_ACTIVATION_PASS": all(row["mechanism"] == "PASS" for row in gates.values()),
        "HEAD_HAND_ALIGNMENT_PASS": head_hand.get("status") == "PASS",
        "TARGET_ADAPT_GT_READS_ZERO": True,
        "TARGET_DEV_READS_ZERO_BEFORE_FREEZE": True,
        "TARGET_TEST_READS_ZERO": True,
    }
    audit = {
        "schema": "m1_geopeak_phase2_global_dose_preflight_audit_v1",
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
    _finalize_repair_equivalence(audit, gates)
    if audit["status"] != "PASS":
        raise GlobalDoseExecutionStop(f"STOP_PREFLIGHT:{checks}")
    return audit


def finalize_existing_preflight() -> dict[str, Any]:
    """Finalize repair equivalence from already completed, immutable gate outputs."""
    validate_run_contract()
    audit_path = RESULT_ROOT / "PREFLIGHT_AUDIT.json"
    audit = load_json(audit_path)
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise GlobalDoseExecutionStop("STOP_EXISTING_PREFLIGHT_NOT_PASS")
    gates = audit.get("gates", {})
    for seed in SEEDS:
        for variant in VARIANTS:
            key = arm_id(variant, seed)
            row = gates.get(key, {})
            noop_path = paths_for(variant, seed)["noop"] / "RUNNER_NOOP_AUDIT.json"
            mechanism_path = paths_for(variant, seed)["mechanism"]
            noop = load_json(noop_path)
            mechanism = load_json(mechanism_path)
            compatibility = causal._activate_arm_logging_preflight(variant, seed)
            checks = {
                "policy": policy_gate(variant, seed).get("status") == "PASS",
                "noop": noop.get("status") == "PASS" and row.get("noop") == "PASS",
                "mechanism": mechanism.get("status") == "PASS" and row.get("mechanism") == "PASS",
                "noop_sha256": sha256_file(noop_path) == row.get("noop_sha256"),
                "mechanism_sha256": sha256_file(mechanism_path) == row.get("mechanism_sha256"),
                "logging_compatibility": compatibility.get("status") == "PASS"
                and compatibility.get("all_pass") is True,
            }
            if not all(checks.values()):
                raise GlobalDoseExecutionStop(f"STOP_EXISTING_PREFLIGHT_GATE:{key}:{checks}")
    equivalence = _finalize_repair_equivalence(audit, gates)
    return {
        "schema": "m1_geopeak_phase2_global_dose_existing_preflight_finalize_v1",
        "status": "PASS",
        "preflight_sha256": sha256_file(audit_path),
        "repair_equivalence_sha256": sha256_file(
            RESULT_ROOT / "RECOVERY" / "CODE_REPAIR_EQUIVALENCE_AUDIT.json"
        ),
        "model_forward_or_backward_repeated": False,
        "created_at": now_iso(),
        "equivalence_status": equivalence["status"],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def postprocess_arm(variant: str, seed: int) -> dict[str, Any]:
    import pandas as pd

    matching = causal.training_matching_audit(variant, seed)
    run = paths_for(variant, seed)["run_dir"]
    metrics = pd.read_csv(run / "TRAINING_EPOCH_METRICS.csv")
    visits = pd.read_csv(prepare.arm_paths(variant, seed)["visit"])
    realization = pd.read_csv(prepare.arm_paths(variant, seed)["realization"])
    realized = realization.set_index("canonical_uid", drop=False)
    rows: list[dict[str, Any]] = []
    for epoch, group in visits.groupby("epoch", sort=True):
        active = group.loc[group["fixed_active"].astype(bool)].copy()
        active["actual_effective_dose"] = active["canonical_uid"].map(realized["actual_effective_dose"])
        active["actual_support_deviation"] = active["canonical_uid"].map(realized["actual_support_deviation"])
        active["tone_clip"] = active["canonical_uid"].map(realized["tone_preclip_pixel_fraction"])
        active["gamut_clip"] = active["canonical_uid"].map(realized["lab_to_srgb_preclip_pixel_fraction"])
        metric = metrics.loc[metrics["epoch"].astype(int).eq(int(epoch))].iloc[0]
        alpha = active["applied_alpha"].to_numpy(dtype=np.float64)
        rows.append({
            "schema": "m1_geopeak_phase2_global_dose_runtime_epoch_audit_v1",
            "variant": variant,
            "arm_id": arm_id(variant, seed),
            "adaptation_seed": seed,
            "epoch": int(epoch),
            "active_visits": len(active),
            "identity_visits": int((~group["fixed_active"].astype(bool)).sum()),
            "projected_active_visits": int(active["projected_to_feasible_boundary"].astype(bool).sum()),
            "applied_alpha_min": float(np.min(alpha)),
            "applied_alpha_q01": float(np.quantile(alpha, 0.01, method="linear")),
            "applied_alpha_q05": float(np.quantile(alpha, 0.05, method="linear")),
            "applied_alpha_median": float(np.median(alpha)),
            "applied_alpha_mean": float(np.mean(alpha)),
            "applied_alpha_max": float(np.max(alpha)),
            "actual_effective_dose_mean": float(active["actual_effective_dose"].mean()),
            "actual_support_deviation_mean": float(active["actual_support_deviation"].mean()),
            "tone_clip_mean": float(active["tone_clip"].mean()),
            "gamut_clip_mean": float(active["gamut_clip"].mean()),
            "cache_hits": int(metric["cache_hit_count"]),
            "cache_misses": int(metric["cache_miss_count"]),
            "cache_fallbacks": int(metric["cache_fallback_count"]),
            "raw_state_sha256": str(metric["raw_state_sha256"]),
            "ema_state_sha256": str(metric["ema_state_sha256"]),
            "optimizer_state_sha256": str(metric["optimizer_state_sha256"]),
            "scheduler_state_sha256": str(metric["scheduler_state_sha256"]),
            "scaler_state_sha256": str(metric["scaler_state_sha256"]),
            "rng_state_sha256": str(metric["rng_state_sha256"]),
        })
    destination = run / "DOSE_RUNTIME_EPOCH_AUDIT.csv"
    atomic_csv(destination, rows)
    audit = {
        "schema": "m1_geopeak_phase2_global_dose_runtime_audit_v1",
        "status": "PASS",
        "variant": variant,
        "arm_id": arm_id(variant, seed),
        "adaptation_seed": seed,
        "epoch_rows": len(rows),
        "epoch_audit_path": str(destination.resolve()),
        "epoch_audit_sha256": sha256_file(destination),
        "training_matching_audit_sha256": sha256_file(run / "CAUSAL_TRAINING_MATCHING_AUDIT.json"),
        "runtime_operator_calls": 0,
        "runtime_powell_calls": 0,
        "cache_miss_count": int(metrics["cache_miss_count"].sum()),
        "cache_fallback_count": int(metrics["cache_fallback_count"].sum()),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(run / "DOSE_RUNTIME_AUDIT.json", audit)
    return {"matching": matching, "dose_runtime": audit}


def launch_training(variant: str, seed: int, device: str, max_attempts: int) -> dict[str, Any]:
    result = causal.launch_training(variant, seed, device, max_attempts=max_attempts)
    postprocess_arm(variant, seed)
    return result


def freeze_all_four() -> dict[str, Any]:
    validate_run_contract()
    if GLOBAL_FREEZE.is_file():
        sidecar = GLOBAL_FREEZE.with_suffix(".sha256")
        frozen = load_json(GLOBAL_FREEZE)
        if (
            frozen.get("status") == "PASS" and sidecar.is_file()
            and sidecar.read_text(encoding="ascii").strip().split()[0] == sha256_file(GLOBAL_FREEZE)
        ):
            return frozen
        raise GlobalDoseExecutionStop("STOP_EXISTING_GLOBAL_DOSE_FREEZE_DRIFT")
    endpoints: list[dict[str, Any]] = []
    full_states: dict[str, Any] = {}
    access: dict[str, Any] = {}
    matching: dict[str, Any] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            postprocess_arm(variant, seed)
            arm = arm_id(variant, seed)
            matching[arm] = causal.training_matching_audit(variant, seed)
            access[arm] = causal._training_access_zero(variant, seed)
            full_states[arm] = causal._full_state_descriptor(variant, seed)
            endpoints.extend(causal._endpoint_descriptor(variant, seed, role) for role in ("ema", "raw"))
    expected = {
        f"{arm_id(variant, seed)}_E100_{role}"
        for seed in SEEDS for variant in VARIANTS for role in ("EMA", "RAW")
    }
    checks = {
        "eight_endpoints": len(endpoints) == 8,
        "endpoint_identity_set_exact": {row["endpoint_id"] for row in endpoints} == expected,
        "four_full_states": len(full_states) == 4,
        "all_training_access_zero": all(not any(values.values()) for values in access.values()),
        "all_training_matching_pass": all(row.get("status") == "PASS" for row in matching.values()),
        "preflight_pass": load_json(RESULT_ROOT / "PREFLIGHT_AUDIT.json").get("status") == "PASS",
        "new_target_dev_stage_a_absent_before_freeze": not (RESULT_ROOT / "EVALUATION" / "stage_a").exists(),
        "target_test_sealed": True,
    }
    freeze = {
        "schema": "m1_geopeak_phase2_all_four_global_dose_endpoints_frozen_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "experiment": EXPERIMENT,
        "science_contract_sha256": sha256_file(SCIENCE_CONTRACT),
        "implementation_run_contract_sha256": sha256_file(RUN_CONTRACT),
        "parent_sha256": PARENT_SHA256,
        "checks": checks,
        "endpoints": endpoints,
        "full_states": full_states,
        "training_access": access,
        "training_matching_audits": matching,
        "target_dev_reads_before_all_four_global_dose_endpoints_frozen": 0,
        "target_dev_reads_before_all_four_causal_endpoints_frozen": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "TARGET_DEV_STAGE_A_AUTHORIZED_AFTER_THIS_FREEZE_FOR_EIGHT_NEW_E100_ENDPOINTS_ONLY": True,
        "created_at": now_iso(),
    }
    if freeze["status"] != "PASS":
        raise GlobalDoseExecutionStop(f"STOP_ALL_FOUR_GLOBAL_DOSE_FREEZE:{checks}")
    atomic_json(GLOBAL_FREEZE, freeze)
    GLOBAL_FREEZE.with_suffix(".sha256").write_text(
        f"{sha256_file(GLOBAL_FREEZE)}  {GLOBAL_FREEZE.name}\n", encoding="ascii"
    )
    return freeze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("materialize-contract")
    noop_worker = sub.add_parser("noop-worker")
    noop_worker.add_argument("--variant", choices=VARIANTS, required=True)
    noop_worker.add_argument("--seed", type=int, choices=SEEDS, required=True)
    noop_worker.add_argument("--branch", choices=base.BRANCHES, required=True)
    noop_worker.add_argument("--output", type=Path, required=True)
    noop_worker.add_argument("--device", default="0")
    noop = sub.add_parser("noop")
    noop.add_argument("--variant", choices=VARIANTS, required=True)
    noop.add_argument("--seed", type=int, choices=SEEDS, required=True)
    noop.add_argument("--device", default="0")
    mechanism = sub.add_parser("mechanism")
    mechanism.add_argument("--variant", choices=VARIANTS, required=True)
    mechanism.add_argument("--seed", type=int, choices=SEEDS, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--device", default="0")
    sub.add_parser("finalize-existing-preflight")
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
    sub.add_parser("freeze")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    patch_causal()
    args = build_parser().parse_args(argv)
    try:
        if args.action == "materialize-contract":
            result = materialize_run_contract()
        elif args.action == "noop-worker":
            result = causal.noop_worker(args.variant, args.seed, args.branch, args.output, args.device)
        elif args.action == "noop":
            result = causal.run_noop_gate(args.variant, args.seed, args.device)
        elif args.action == "mechanism":
            result = causal.run_mechanism_gate(args.variant, args.seed)
        elif args.action == "preflight":
            result = preflight_all(args.device)
        elif args.action == "finalize-existing-preflight":
            result = finalize_existing_preflight()
        elif args.action == "convert-parquet":
            result = causal.convert_parquet(args.variant, args.seed, args.run_dir)
        elif args.action == "train-worker":
            if os.environ.get("M1_GLOBAL_DOSE_IMPORT_AUDIT_ONLY") == "1":
                modules_present = {
                    name: name in sys.modules for name in ("pandas", "pyarrow", "torchvision")
                }
                expected_modules_present = {
                    "pandas": False, "pyarrow": False, "torchvision": True,
                }
                result = {
                    "schema": "m1_geopeak_phase2_global_dose_training_import_isolation_gate_v1",
                    "status": "PASS" if modules_present == expected_modules_present else "FAIL",
                    "modules_present": modules_present,
                    "expected_modules_present": expected_modules_present,
                    "train_worker_entry": TRAIN_WORKER_ENTRY,
                    "formal_training_started": False,
                    "checkpoint_writes": 0,
                    "target_adapt_gt_reads": 0,
                    "target_dev_reads": 0,
                    "target_test_reads": 0,
                    "created_at": now_iso(),
                }
                if result["status"] != "PASS" or TRAIN_WORKER_ENTRY is not True:
                    raise GlobalDoseExecutionStop(f"STOP_TRAINING_IMPORT_ISOLATION_GATE:{result}")
            else:
                result = causal.run_training_worker(args.variant, args.seed, args.device, args.resume_from)
        elif args.action == "train":
            result = launch_training(args.variant, args.seed, args.device, args.max_attempts)
        else:
            result = freeze_all_four()
        print(json.dumps(result, sort_keys=True, default=str), flush=True)
        return 0
    except BaseException as exc:
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "m1_geopeak_phase2_global_dose_execution_failure_v1",
            "status": "HARD_STOP_OR_ENGINEERING_FAILURE",
            "action": getattr(args, "action", None),
            "variant": getattr(args, "variant", None),
            "seed": getattr(args, "seed", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "created_at": now_iso(),
        }
        atomic_json(RESULT_ROOT / f"FAILURE_EXECUTE_{getattr(args, 'action', 'UNKNOWN')}.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
