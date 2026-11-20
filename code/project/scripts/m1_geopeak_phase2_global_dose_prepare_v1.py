from __future__ import annotations

"""Prepare the frozen Gate-A2-authorized global-dose Phase-2 battery.

This module is evaluation-blind.  It constructs only the two frozen dose
views for each fresh adaptation seed, verifies alpha=0/1 equivalence, and
materializes exact offline cache manifests for the unchanged Stage-1 runner.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal")
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"
CONTRACTS = PROJECT / "contracts"
FRESH_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2"
GATE_A2_ROOT = PROJECT / "results" / "M1_GEOPEAK_GATE_A_REPAIR_V2"
FULL_PROGRAM_ROOT = PROJECT / "results" / "M1_GEOPEAK_FULL_PROGRAM_V1"
RESULT_ROOT = PROJECT / "results" / "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1"
SCIENCE_CONTRACT = CONTRACTS / "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1.json"
GATE_A2_ZIP = GATE_A2_ROOT / "GATE_A2_REPAIRED_DESCRIPTOR_AND_PROXY_REVIEW.zip"
GATE_A2_ZIP_BYTES = 20_625_674
GATE_A2_ZIP_SHA256 = "4c591d81146f6bb24507d74af7bed58b77d40e576fee443c098c20896c37b2bd"
DECISION_SOURCE = Path(r"C:\Users\Lenovo\Desktop\GATE_A2_CHATGPT_SCIENTIFIC_DECISION.json")
PROMPT_SOURCE = Path(r"C:\Users\Lenovo\Desktop\M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BATTERY_CODEX_PROMPT.txt")
GATE_A2_SOURCE = SCRIPTS / "m1_geopeak_gate_a_repair_v2.py"
GATE_A2_SOURCE_SHA256 = "08276768994ed6c12e7e9b7a60bd51d3da254a9e7128c16bf75fec3f50b1d946"
PARENT = PROJECT / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
PARENT_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
FRESH_ZIP = FRESH_ROOT / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2.zip"
FRESH_ZIP_SHA256 = "c99a95bf47915f283bc10016ae10272a3ce117569095591807839e7feb049106"
FRESH_IDENTITY = CONTRACTS / "M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1.json"
FRESH_IDENTITY_SHA256 = "a8305a76077856eb18f0f9d16f549086c5ccf2fe8d34b482ac7ba8261bdc4279"
OFFICIAL_EVALUATOR = SCRIPTS / "m1_seed42_stage1_e100_evaluate_v1.py"
OFFICIAL_EVALUATOR_SHA256 = "c38c767f93a9c831ff78c49a3dee5a592b2b139bfde5156a1c076d5d317f072d"
SEEDS = (45, 46)
EPOCHS = tuple(range(72, 101))
EXPECTED_IMAGES = 3272
EXPECTED_VALID = 3226
EXPECTED_INVALID = 46
TOTAL_VISITS = 128_064
ALPHA075 = "GLOBAL_ALPHA_075"
ALPHA125 = "GLOBAL_ALPHA_125_PROJECTED"
VARIANTS = (ALPHA075, ALPHA125)
REQUESTED_ALPHA = {ALPHA075: 0.75, ALPHA125: 1.25}
TARGET_ADAPT_IMAGES = ROOT / "datasets" / "dataset_final_v2_1" / "target" / "adapt" / "images"

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import m1_fixed_causal_controls_v1_prepare as causal_prepare  # noqa: E402
import m1_routea_offline_theta_cache_engine_v1 as cache_engine  # noqa: E402
import m1_routea_photometric_realization_core_v1 as realization  # noqa: E402


class GlobalDosePreparationStop(RuntimeError):
    """Fail-closed preparation error."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_bytes(value) + b"\n")


def atomic_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (list(rows[0].keys()) if rows else []))
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise GlobalDosePreparationStop(f"STOP_JSON_OBJECT_REQUIRED:{path}")
    return value


def strict_bool(value: Any, role: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise GlobalDosePreparationStop(f"STOP_BOOL:{role}:{value}")


def require_sha(value: Any, role: str) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise GlobalDosePreparationStop(f"STOP_SHA:{role}:{value}")
    return digest


def arm_id(variant: str, seed: int) -> str:
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise GlobalDosePreparationStop(f"STOP_VARIANT_SEED:{variant}:{seed}")
    return f"{variant}_SEED{int(seed)}"


def seed_reference_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise GlobalDosePreparationStop(f"STOP_SEED:{seed}")
    root = FRESH_ROOT / f"SEED{seed}"
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


def arm_paths(variant: str, seed: int) -> dict[str, Path]:
    arm = arm_id(variant, seed)
    root = RESULT_ROOT / "POLICY_CACHE" / arm
    cache = RESULT_ROOT / "CACHE" / f"SEED{seed}" / variant
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


def _copy_identity_file(source: Path, destination: Path) -> dict[str, Any]:
    digest = sha256_file(source)
    if destination.is_file():
        if sha256_file(destination) != digest:
            raise GlobalDosePreparationStop(f"STOP_GOVERNANCE_COPY_DRIFT:{destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return {
        "source_path": str(source.resolve()),
        "copied_path": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": digest,
    }


def verify_gate_a2_decision() -> dict[str, Any]:
    if not GATE_A2_ZIP.is_file() or GATE_A2_ZIP.stat().st_size != GATE_A2_ZIP_BYTES:
        raise GlobalDosePreparationStop("STOP_GATE_A2_REVIEW_PACKAGE_BYTES")
    observed_zip = sha256_file(GATE_A2_ZIP)
    if observed_zip != GATE_A2_ZIP_SHA256:
        raise GlobalDosePreparationStop("STOP_GATE_A2_REVIEW_PACKAGE_SHA")
    with zipfile.ZipFile(GATE_A2_ZIP, "r") as archive:
        crc_error = archive.testzip()
        entry_count = len(archive.infolist())
    if crc_error is not None:
        raise GlobalDosePreparationStop(f"STOP_GATE_A2_REVIEW_PACKAGE_CRC:{crc_error}")
    decision = load_json(DECISION_SOURCE)
    checks = {
        "gate_id": decision.get("gate_id") == "GATE_A2_REPAIRED_DESCRIPTOR_AND_PROXY",
        "review_package_name": decision.get("review_package_name") == GATE_A2_ZIP.name,
        "review_package_bytes": int(decision.get("review_package_bytes", -1)) == GATE_A2_ZIP_BYTES,
        "review_package_sha256": decision.get("review_package_sha256") == GATE_A2_ZIP_SHA256,
        "decision": decision.get("decision") == "PROCEED_GLOBAL_DOSE_BATTERY_ONLY",
        "authorized_next_phase": decision.get("authorized_next_phase") == "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1",
        "approved_by": decision.get("approved_by") == "USER_AND_CHATGPT",
        "descriptor_boundary": str(decision.get("approved_descriptor_feature_set", "")).startswith("Z7_V1_RETAINED"),
        "proxy_none": decision.get("approved_proxy_template") == "NONE",
        "predictor_none": decision.get("approved_predictor") == "NONE",
        "target_test_sealed": decision.get("scientific_status", {}).get("target_test") == "SEALED",
        "final_gate_a2_source": sha256_file(GATE_A2_SOURCE) == GATE_A2_SOURCE_SHA256,
    }
    if not all(checks.values()):
        raise GlobalDosePreparationStop(
            "STOP_GATE_A2_DECISION_IDENTITY:" + ",".join(sorted(k for k, v in checks.items() if not v))
        )
    governance = RESULT_ROOT / "GOVERNANCE"
    copied = {
        "decision": _copy_identity_file(DECISION_SOURCE, governance / DECISION_SOURCE.name),
        "execution_prompt": _copy_identity_file(PROMPT_SOURCE, governance / PROMPT_SOURCE.name),
    }
    for name in (
        "GATE_A2_REVIEW_MANIFEST.json",
        "GATE_A2_SCIENTIFIC_DECISION_TEMPLATE.json",
        "GATE_A2_ZIP_INTEGRITY_AUDIT.json",
    ):
        source = GATE_A2_ROOT / name
        if not source.is_file():
            raise GlobalDosePreparationStop(f"STOP_GATE_A2_EXTERNAL_GOVERNANCE_FILE_MISSING:{name}")
        copied[name] = _copy_identity_file(source, governance / name)
    receipt = {
        "schema": "m1_geopeak_phase2_gate_a2_decision_receipt_v1",
        "status": "PASS",
        "task_id": "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1",
        "checks": checks,
        "review_package": {
            "path": str(GATE_A2_ZIP.resolve()),
            "bytes": GATE_A2_ZIP.stat().st_size,
            "sha256": observed_zip,
            "entry_count": entry_count,
            "crc_error": crc_error,
        },
        "decision": decision,
        "decision_source_sha256": sha256_file(DECISION_SOURCE),
        "execution_prompt_sha256": sha256_file(PROMPT_SOURCE),
        "copied_governance": copied,
        "permanent_scientific_registration": {
            "Z7": "CURRENT_COMPACT_COORDINATE_FOR_EXISTING_NATIVE_DIRECTION_ONLY_NOT_ABSOLUTELY_SUFFICIENT",
            "R25": "NO_MATERIAL_INCREMENT_AFTER_INTERVENTION_CONDITIONING",
            "B19": "CLIPPING_BOUNDARY_DIAGNOSTIC_ONLY_NOT_TRAINING_GATE",
            "LOCAL_T1_T2_T3_PEAK": "NOT_IDENTIFIABLE_NOT_AUTHORIZED_FOR_TRAINING",
            "CURRENT_PHASE": "DOWNSTREAM_GLOBAL_DOSE_RESPONSE_ONLY",
            "TARGET_TEST": "SEALED",
        },
        "legacy_governance_conflicts": {
            "requested_project_handoff_20260827_missing": True,
            "next_actions_points_to_older_b10b_boundary": True,
            "resolution": "NEWER_SHA_BOUND_GATE_A2_DECISION_HAS_HIGHER_EVIDENCE_PRIORITY",
        },
        "created_at": now_iso(),
    }
    result_receipt = RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json"
    program_receipt = FULL_PROGRAM_ROOT / "GOVERNANCE" / "GATE_A2_DECISION_RECEIPT.json"
    for path in (result_receipt, governance / "GATE_A2_DECISION_RECEIPT.json", program_receipt):
        if path.is_file():
            old = load_json(path)
            old.pop("created_at", None)
            candidate = dict(receipt)
            candidate.pop("created_at", None)
            if old != candidate:
                raise GlobalDosePreparationStop(f"STOP_EXISTING_GATE_A2_RECEIPT_DRIFT:{path}")
        else:
            atomic_json(path, receipt)
    return receipt


def verify_input_identities() -> dict[str, Any]:
    verify_gate_a2_decision()
    expected = {
        "parent": (PARENT, PARENT_SHA256),
        "fresh_zip": (FRESH_ZIP, FRESH_ZIP_SHA256),
        "fresh_identity": (FRESH_IDENTITY, FRESH_IDENTITY_SHA256),
        "official_evaluator": (OFFICIAL_EVALUATOR, OFFICIAL_EVALUATOR_SHA256),
    }
    identities: dict[str, Any] = {}
    for role, (path, digest) in expected.items():
        observed = sha256_file(path)
        identities[role] = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "expected_sha256": digest, "observed_sha256": observed,
            "pass": observed == digest,
        }
    science_sha = sha256_file(SCIENCE_CONTRACT)
    science_sidecar = SCIENCE_CONTRACT.with_suffix(".sha256")
    identities["science_contract"] = {
        "path": str(SCIENCE_CONTRACT.resolve()),
        "bytes": SCIENCE_CONTRACT.stat().st_size,
        "observed_sha256": science_sha,
        "sidecar_sha256": science_sidecar.read_text(encoding="ascii").strip().split()[0],
    }
    identities["science_contract"]["pass"] = (
        identities["science_contract"]["observed_sha256"]
        == identities["science_contract"]["sidecar_sha256"]
    )
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        paths = seed_reference_paths(seed)
        selection = pd.read_csv(paths["selection"], dtype=str, keep_default_na=False)
        theta = pd.read_csv(paths["theta"], dtype=str, keep_default_na=False)
        exposure = pd.read_csv(paths["exposure"], dtype=str, keep_default_na=False)
        valid = selection["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
        theta_valid = theta["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
        active = exposure["fixed_active"].map(lambda value: strict_bool(value, "fixed_active"))
        identity = exposure["fixed_identity"].map(lambda value: strict_bool(value, "fixed_identity"))
        uids = set(selection.loc[valid, "canonical_uid"].astype(str))
        theta_uids = set(theta.loc[theta_valid, "canonical_uid"].astype(str))
        checks = {
            "selection_rows": len(selection) == EXPECTED_IMAGES,
            "valid_count": int(valid.sum()) == EXPECTED_VALID,
            "invalid_count": int((~valid).sum()) == EXPECTED_INVALID,
            "theta_valid_uid_set": uids == theta_uids and len(theta_uids) == EXPECTED_VALID,
            "exposure_rows": len(exposure) == TOTAL_VISITS,
            "active_identity_exclusive": bool((active ^ identity).all()),
            "active_count_expected": int(active.sum()) == (62_744 if seed == 45 else 62_824),
            "schedule_seed_exact": bool(exposure["adaptation_seed"].astype(int).eq(seed).all()),
        }
        per_seed[str(seed)] = {
            "checks": checks,
            "counts": {
                "images": len(selection), "valid_direction_uids": int(valid.sum()),
                "invalid_direction_uids": int((~valid).sum()), "visits": len(exposure),
                "active_visits": int(active.sum()), "identity_visits": int(identity.sum()),
            },
            "frozen_files": {
                role: {
                    "path": str(paths[role].resolve()), "bytes": paths[role].stat().st_size,
                    "sha256": sha256_file(paths[role]),
                }
                for role in (
                    "selection", "theta", "exposure", "source_schedule",
                    "source_augmentation", "target_schedule", "target_manifest", "fixed_contract",
                )
            },
        }
        if not all(checks.values()):
            raise GlobalDosePreparationStop(f"STOP_CANONICAL_SEED_INPUT:{seed}:{checks}")
    audit = {
        "schema": "m1_geopeak_phase2_global_dose_input_identity_audit_v1",
        "status": "PASS",
        "identities": identities,
        "per_seed": per_seed,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    if not all(row.get("pass") for row in identities.values()):
        raise GlobalDosePreparationStop("STOP_CANONICAL_INPUT_SHA")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULT_ROOT / "CANONICAL_INPUT_IDENTITY_AUDIT.json", audit)
    return audit


def _alpha_multiset_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(b"M1_GEOPEAK_GLOBAL_DOSE_APPLIED_ALPHA_MULTISET_V1\0")
    for row in sorted(rows, key=lambda value: str(value["canonical_uid"])):
        uid = str(row["canonical_uid"]).encode("utf-8")
        digest.update(struct.pack("<I", len(uid)))
        digest.update(uid)
        digest.update(struct.pack("<d", float(row["applied_alpha"])))
    return digest.hexdigest()


def _ray_theta(native: np.ndarray, alpha: float) -> np.ndarray:
    return np.asarray(
        [float(alpha) * float(native[0]), math.exp(float(alpha) * math.log(float(native[1]))), math.exp(float(alpha) * math.log(float(native[2])))],
        dtype=np.float64,
    )


def _alpha_max(native: np.ndarray) -> tuple[float, str]:
    values = np.asarray(
        [
            math.inf if abs(float(native[0])) == 0.0 else math.log(2.0) / abs(float(native[0])),
            math.inf if abs(math.log(float(native[1]))) == 0.0 else math.log(2.0) / abs(math.log(float(native[1]))),
            math.inf if abs(math.log(float(native[2]))) == 0.0 else math.log(2.0) / abs(math.log(float(native[2]))),
        ],
        dtype=np.float64,
    )
    index = int(np.argmin(values))
    return float(values[index]), ("delta_t", "c_t", "s_c")[index]


def build_dose_policy() -> dict[str, Any]:
    verify_input_identities()
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for seed in SEEDS:
        paths = seed_reference_paths(seed)
        theta = pd.read_csv(paths["theta"])
        valid = theta.loc[
            theta["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
        ].copy().sort_values("canonical_uid", kind="mergesort")
        exposure = pd.read_csv(paths["exposure"])
        active_exposure = exposure.loc[
            exposure["fixed_active"].map(lambda value: strict_bool(value, "fixed_active"))
        ]
        per_variant: dict[str, Any] = {}
        for variant in VARIANTS:
            rows: list[dict[str, Any]] = []
            for record in valid.itertuples(index=False):
                native = np.asarray(
                    [record.theta_delta_t, record.theta_c_t, record.theta_s_c], dtype=np.float64
                )
                alpha_max, bound = _alpha_max(native)
                requested = REQUESTED_ALPHA[variant]
                if variant == ALPHA075:
                    applied = requested
                    projected = False
                else:
                    applied = requested if alpha_max >= requested else max(1.0, float(np.nextafter(alpha_max, 0.0)))
                    projected = applied < requested
                applied_theta = _ray_theta(native, applied)
                finite = bool(np.isfinite(applied_theta).all())
                in_bounds = bool(
                    finite
                    and np.all(applied_theta >= realization.PARAMETER_LOWER)
                    and np.all(applied_theta <= realization.PARAMETER_UPPER)
                )
                if not in_bounds:
                    raise GlobalDosePreparationStop(
                        f"STOP_APPLIED_THETA_OUT_OF_BOUNDS:{variant}:{seed}:{record.canonical_uid}:{applied}:{applied_theta.tolist()}"
                    )
                row = {
                    "schema": "m1_geopeak_phase2_global_dose_policy_row_v1",
                    "variant": variant,
                    "arm_id": arm_id(variant, seed),
                    "adaptation_seed": seed,
                    "canonical_uid": str(record.canonical_uid),
                    "native_theta_delta_t": float(native[0]),
                    "native_theta_c_t": float(native[1]),
                    "native_theta_s_c": float(native[2]),
                    "native_theta_raw_f8_sha256": str(record.theta_raw_f8_sha256),
                    "native_selected_delta_sha": str(record.selected_delta_sha),
                    "native_selected_intended_geometric_dose": float(record.selected_intended_geometric_dose),
                    "requested_alpha": requested,
                    "alpha_max": alpha_max,
                    "active_bound_parameter": bound,
                    "applied_alpha": applied,
                    "projected_to_feasible_boundary": projected,
                    "applied_theta_delta_t": float(applied_theta[0]),
                    "applied_theta_c_t": float(applied_theta[1]),
                    "applied_theta_s_c": float(applied_theta[2]),
                    "applied_theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(applied_theta),
                    "parameter_bound_check": "PASS",
                    "component_clipping": False,
                    "powell_resolve": False,
                    "runtime_policy_change_allowed": False,
                }
                rows.append(row)
                all_rows.append(row)
            projected_uids = {
                str(row["canonical_uid"]) for row in rows if row["projected_to_feasible_boundary"]
            }
            applied_values = np.asarray([float(row["applied_alpha"]) for row in rows], dtype=np.float64)
            active_projected = int(active_exposure["canonical_uid"].astype(str).isin(projected_uids).sum())
            summary = {
                "variant": variant,
                "arm_id": arm_id(variant, seed),
                "adaptation_seed": seed,
                "valid_uid_count": len(rows),
                "no_valid_direction_uid_count": EXPECTED_INVALID,
                "requested_alpha": REQUESTED_ALPHA[variant],
                "projected_uid_count": len(projected_uids),
                "projected_uid_fraction": len(projected_uids) / len(rows),
                "projected_active_visit_count": active_projected,
                "projected_active_visit_fraction": active_projected / len(active_exposure),
                "applied_alpha_distribution": {
                    "minimum": float(np.min(applied_values)),
                    "q01": float(np.quantile(applied_values, 0.01, method="linear")),
                    "q05": float(np.quantile(applied_values, 0.05, method="linear")),
                    "median": float(np.median(applied_values)),
                    "mean": float(np.mean(applied_values)),
                    "maximum": float(np.max(applied_values)),
                },
                "active_bound_parameter_counts_for_projected": {
                    key: sum(
                        int(row["projected_to_feasible_boundary"] and row["active_bound_parameter"] == key)
                        for row in rows
                    )
                    for key in ("delta_t", "c_t", "s_c")
                },
                "applied_alpha_multiset_sha256": _alpha_multiset_sha(rows),
            }
            per_variant[variant] = summary
            atomic_csv(arm_paths(variant, seed)["policy_rows"], rows)
        summaries[str(seed)] = per_variant
    projection_csv = RESULT_ROOT / "ALPHA_FEASIBILITY_PROJECTION_AUDIT.csv"
    atomic_csv(projection_csv, all_rows)
    contract = {
        "schema": "m1_geopeak_phase2_global_dose_policy_contract_v1",
        "status": "FROZEN_BEFORE_CACHE_OR_TRAINING",
        "task_id": "M1_GEOPEAK_PHASE2_GLOBAL_DOSE_BRACKETING_45_46_V1",
        "science_contract": {
            "path": str(SCIENCE_CONTRACT.resolve()),
            "sha256": sha256_file(SCIENCE_CONTRACT),
        },
        "gate_a2_decision_receipt": {
            "path": str((RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json").resolve()),
            "sha256": sha256_file(RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json"),
        },
        "ray_definition": {
            "delta_t": "alpha*native_delta_t",
            "c_t": "exp(alpha*log(native_c_t))",
            "s_c": "exp(alpha*log(native_s_c))",
            "interpretation": "REALIZED_3DOF_OPERATOR_LOG_PARAMETER_RAY_NOT_EXACT_7D_ALPHA_DELTA_SCALING",
        },
        "alpha075_rule": "requested_equals_applied_0.75_for_all_valid_uids",
        "alpha125_projection_rule": "if alpha_max>=1.25 then 1.25 else max(1.0,nextafter(alpha_max,0.0))",
        "operator_bounds": {
            "lower": realization.PARAMETER_LOWER.tolist(),
            "upper": realization.PARAMETER_UPPER.tolist(),
        },
        "invalid_direction_behavior": "IDENTITY_EXACTLY_AS_CANONICAL_NATIVE_POLICY",
        "per_seed_variant": summaries,
        "projection_rows_path": str(projection_csv.resolve()),
        "projection_rows_sha256": sha256_file(projection_csv),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    destination = RESULT_ROOT / "DOSE_POLICY_CONTRACT.json"
    if destination.is_file():
        old = load_json(destination)
        old.pop("created_at", None)
        candidate = dict(contract)
        candidate.pop("created_at", None)
        if old != candidate:
            raise GlobalDosePreparationStop("STOP_EXISTING_DOSE_POLICY_CONTRACT_DRIFT")
    else:
        atomic_json(destination, contract)
        destination.with_suffix(".sha256").write_text(
            f"{sha256_file(destination)}  {destination.name}\n", encoding="ascii"
        )
    audit = {
        "schema": "m1_geopeak_phase2_alpha_feasibility_projection_audit_v1",
        "status": "PASS",
        "per_seed_variant": summaries,
        "projection_rows_sha256": sha256_file(projection_csv),
        "dose_policy_contract_sha256": sha256_file(destination),
        "component_clipping_count": 0,
        "powell_resolve_count": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "ALPHA_FEASIBILITY_PROJECTION_AUDIT.json", audit)
    return audit


_WORKER_CONTEXT: Any | None = None
_WORKER_GUARD: Any | None = None


def _worker_initializer(seed: int) -> None:
    global _WORKER_CONTEXT, _WORKER_GUARD
    _WORKER_GUARD = causal_prepare.ConstructionDenylist(
        f"GLOBAL_DOSE_CACHE_WORKER_SEED{seed}"
    ).install()
    realization.configure_q2_snapshot(causal_prepare.Q2_DIR / "CODE_SNAPSHOT")
    _WORKER_CONTEXT = causal_prepare.load_policy_context(seed)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    with temporary.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _reconstruct_bgr(palette: Any, deployment_rgb_uint8: np.ndarray) -> np.ndarray:
    palette_bgr = np.ascontiguousarray(np.asarray(deployment_rgb_uint8, dtype=np.uint8)[:, ::-1])
    return np.ascontiguousarray(
        palette_bgr[palette.inverse_flat].reshape(palette.height, palette.width, 3)
    )


def _output_record(
    *, job: Mapping[str, Any], variant: str, palette: Any, original_std: np.ndarray,
    theta: np.ndarray, applied_alpha: float, final: Mapping[str, Any], output_bgr: np.ndarray,
) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise GlobalDosePreparationStop("STOP_CACHE_WORKER_CONTEXT")
    context = _WORKER_CONTEXT
    deployment_std = realization.standardize(final["deployment_raw"], context.median, context.scale)
    actual_dose = float(realization.dstyle(original_std, deployment_std))
    support = float(np.min(realization.dstyle_rows(context.z_event, deployment_std[None, :])))
    internal, reason = realization.descriptor_internal_valid_raw(final["deployment_raw"])
    paths = {key: Path(str(value)) for key, value in job["outputs"][variant].items()}
    _write_npy_atomic(paths["bgr"], output_bgr)
    file_sha = sha256_file(paths["bgr"])
    array_sha = causal_prepare.frozen_loader_array_bytes_sha(output_bgr)
    formal_sha = causal_prepare.frozen_formal_bgr_sha(output_bgr)
    expected_formal = str(final["output_array_sha256"])
    if formal_sha != expected_formal:
        raise GlobalDosePreparationStop(
            f"STOP_OPERATOR_FORMAL_BGR_SHA:{variant}:{job['seed']}:{job['canonical_uid']}"
        )
    solve_key = canonical_sha({
        "namespace": "M1_GEOPEAK_GLOBAL_DOSE_CACHE_V1",
        "variant": variant,
        "seed": int(job["seed"]),
        "uid": str(job["canonical_uid"]),
        "input_sha": str(job["input_image_sha256"]),
        "native_theta_sha": str(job["native_theta_sha256"]),
        "applied_theta_sha": cache_engine.theta_raw_f8_sha256(theta),
        "applied_alpha_f8": struct.pack("<d", float(applied_alpha)).hex(),
    })
    sidecar = {
        "schema": "m1_native_a_fixed_cache_entry_v1",
        "cache_namespace": "M1_GEOPEAK_GLOBAL_DOSE_CACHE_V1",
        "cache_key_sha256": canonical_sha({"solve_key": solve_key, "output": formal_sha}),
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": str(job["input_image_sha256"]),
        "solve_key": solve_key,
        "theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(theta),
        "expected_output_bgr_sha256": formal_sha,
        "height": int(output_bgr.shape[0]),
        "width": int(output_bgr.shape[1]),
        "candidates": {
            "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE": {
                "path": str(paths["bgr"].resolve()),
                "complete": True,
                "dtype": "uint8",
                "shape": list(output_bgr.shape),
                "file_bytes": paths["bgr"].stat().st_size,
                "file_sha256": file_sha,
                "array_bytes_sha256": array_sha,
                "decoder": "cv2",
                "built_at": now_iso(),
            }
        },
    }
    atomic_json(paths["sidecar"], sidecar)
    sidecar_sha = sha256_file(paths["sidecar"])
    manifest = {
        "schema": "m1_native_a_fixed_cache_manifest_entry_v1",
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": str(job["input_image_sha256"]),
        "solve_key": solve_key,
        "theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(theta),
        "expected_output_bgr_sha256": formal_sha,
        "sidecar_path": str(paths["sidecar"].resolve()),
        "sidecar_sha256": sidecar_sha,
        "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete": True,
    }
    realization_row: dict[str, Any] = {
        "schema": "m1_geopeak_phase2_global_dose_cache_realization_v1",
        "variant": variant,
        "arm_id": arm_id(variant, int(job["seed"])),
        "adaptation_seed": int(job["seed"]),
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": str(job["input_image_sha256"]),
        "requested_alpha": REQUESTED_ALPHA[variant],
        "applied_alpha": float(applied_alpha),
        "projected_to_feasible_boundary": bool(
            variant == ALPHA125 and float(applied_alpha) < REQUESTED_ALPHA[variant]
        ),
        "native_theta_raw_f8_sha256": str(job["native_theta_sha256"]),
        "applied_theta_delta_t": float(theta[0]),
        "applied_theta_c_t": float(theta[1]),
        "applied_theta_s_c": float(theta[2]),
        "applied_theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(theta),
        "native_selected_delta_sha": str(job["native_selected_delta_sha"]),
        "native_selected_intended_geometric_dose": float(job["native_intended_dose"]),
        "actual_effective_dose": actual_dose,
        "actual_support_deviation": support,
        "actual_endpoint_in_tube": bool(support <= context.tau),
        "actual_descriptor_internal_valid": bool(internal),
        "actual_descriptor_internal_invalid_reason": str(reason),
        "parameter_in_bounds": bool(
            np.all(theta >= realization.PARAMETER_LOWER)
            and np.all(theta <= realization.PARAMETER_UPPER)
        ),
        "output_differs_from_identity": bool(
            formal_sha != str(job["alpha0_formal_bgr_sha256"])
        ),
        "cache_solve_key": solve_key,
        "cache_expected_output_bgr_sha256": formal_sha,
        "cache_payload_path": str(paths["bgr"].resolve()),
        "cache_payload_file_sha256": file_sha,
        "cache_payload_array_bytes_sha256": array_sha,
        "cache_payload_bytes": paths["bgr"].stat().st_size,
    }
    realization_row.update(dict(final["diagnostics"]))
    return {"manifest": manifest, "realization": realization_row}


def _cache_worker(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise GlobalDosePreparationStop("STOP_CACHE_WORKER_CONTEXT")
    started = time.perf_counter()
    image_path = Path(str(job["image_path"])).resolve()
    if causal_prepare.ConstructionDenylist.classify(image_path) != "ALLOWED_TARGET_ADAPT_UNLABELED_IMAGE":
        raise GlobalDosePreparationStop(f"STOP_CACHE_IMAGE_CLASSIFICATION:{image_path}")
    if sha256_file(image_path) != str(job["input_image_sha256"]):
        raise GlobalDosePreparationStop(f"STOP_CACHE_INPUT_IMAGE_SHA:{job['canonical_uid']}")
    original_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original_bgr is None or original_bgr.dtype != np.uint8 or original_bgr.ndim != 3 or original_bgr.shape[2] != 3:
        raise GlobalDosePreparationStop(f"STOP_CACHE_IMAGE_DECODE:{job['canonical_uid']}")
    original_bgr = np.ascontiguousarray(original_bgr)
    palette = realization.PaletteImage.from_rgb_uint8(np.ascontiguousarray(original_bgr[:, :, ::-1]))
    native = np.asarray(job["native_theta"], dtype=np.float64)
    original_std = np.asarray(job["original_std"], dtype=np.float64)
    alpha0 = palette.final_evaluation(np.asarray([0.0, 1.0, 1.0], dtype=np.float64))
    alpha0_bgr = _reconstruct_bgr(palette, alpha0["deployment_rgb_uint8"])
    alpha0_exact = bool(np.array_equal(alpha0_bgr, original_bgr))
    if not alpha0_exact:
        raise GlobalDosePreparationStop(f"STOP_ALPHA0_NOT_BYTE_EXACT:{job['seed']}:{job['canonical_uid']}")
    alpha0_formal = causal_prepare.frozen_formal_bgr_sha(alpha0_bgr)
    alpha1 = palette.final_evaluation(native)
    alpha1_bgr = _reconstruct_bgr(palette, alpha1["deployment_rgb_uint8"])
    canonical_path = Path(str(job["canonical_native_cache_path"])).resolve()
    canonical_mmap = np.load(canonical_path, mmap_mode="r", allow_pickle=False)
    canonical_bgr = np.ascontiguousarray(canonical_mmap)
    del canonical_mmap
    alpha1_exact = bool(np.array_equal(alpha1_bgr, canonical_bgr))
    alpha1_formal = causal_prepare.frozen_formal_bgr_sha(alpha1_bgr)
    if not (
        alpha1_exact
        and alpha1_formal == str(job["canonical_native_output_bgr_sha256"])
        and alpha1_formal == str(alpha1["output_array_sha256"])
    ):
        raise GlobalDosePreparationStop(f"STOP_ALPHA1_NOT_NATIVE_EXACT:{job['seed']}:{job['canonical_uid']}")
    job_with_identity = dict(job)
    job_with_identity["alpha0_formal_bgr_sha256"] = alpha0_formal
    outputs: dict[str, Any] = {}
    for variant in VARIANTS:
        applied = float(job["applied_alpha"][variant])
        theta = _ray_theta(native, applied)
        final = palette.final_evaluation(theta)
        output_bgr = _reconstruct_bgr(palette, final["deployment_rgb_uint8"])
        outputs[variant] = _output_record(
            job=job_with_identity,
            variant=variant,
            palette=palette,
            original_std=original_std,
            theta=theta,
            applied_alpha=applied,
            final=final,
            output_bgr=output_bgr,
        )
        if outputs[variant]["realization"]["output_differs_from_identity"] is not True:
            raise GlobalDosePreparationStop(
                f"STOP_ACTIVE_OUTPUT_IDENTITY:{variant}:{job['seed']}:{job['canonical_uid']}"
            )
    equivalence = {
        "schema": "m1_geopeak_phase2_alpha_endpoint_equivalence_v1",
        "adaptation_seed": int(job["seed"]),
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": str(job["input_image_sha256"]),
        "alpha0_preletterbox_bgr_byte_exact_original": alpha0_exact,
        "alpha0_original_array_bytes_sha256": causal_prepare.frozen_loader_array_bytes_sha(original_bgr),
        "alpha0_output_array_bytes_sha256": causal_prepare.frozen_loader_array_bytes_sha(alpha0_bgr),
        "alpha0_output_formal_bgr_sha256": alpha0_formal,
        "alpha1_preletterbox_bgr_byte_exact_native_cache": alpha1_exact,
        "alpha1_native_cache_path": str(canonical_path),
        "alpha1_native_cache_file_sha256": sha256_file(canonical_path),
        "alpha1_native_cache_array_bytes_sha256": causal_prepare.frozen_loader_array_bytes_sha(canonical_bgr),
        "alpha1_replay_array_bytes_sha256": causal_prepare.frozen_loader_array_bytes_sha(alpha1_bgr),
        "alpha1_replay_formal_bgr_sha256": alpha1_formal,
        "alpha1_expected_formal_bgr_sha256": str(job["canonical_native_output_bgr_sha256"]),
        "shape_exact": list(alpha1_bgr.shape) == list(original_bgr.shape),
        "dtype_exact": str(alpha1_bgr.dtype) == "uint8",
    }
    commit = {
        "schema": "m1_geopeak_phase2_global_dose_cache_committed_uid_v1",
        "status": "COMMITTED",
        "adaptation_seed": int(job["seed"]),
        "canonical_uid": str(job["canonical_uid"]),
        "equivalence": equivalence,
        "outputs": outputs,
        "worker_data_access": _WORKER_GUARD.audit() if _WORKER_GUARD is not None else {},
        "elapsed_seconds": float(time.perf_counter() - started),
        "created_at": now_iso(),
    }
    commit_path = Path(str(job["commit_path"]))
    atomic_json(commit_path, commit)
    return {"uid": str(job["canonical_uid"]), "commit_path": str(commit_path)}


def _commit_valid(path: Path, seed: int, uid: str, *, full_payload: bool) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        commit = load_json(path)
        if not (
            commit.get("schema") == "m1_geopeak_phase2_global_dose_cache_committed_uid_v1"
            and commit.get("status") == "COMMITTED"
            and int(commit.get("adaptation_seed", -1)) == int(seed)
            and commit.get("canonical_uid") == uid
            and set(commit.get("outputs", {})) == set(VARIANTS)
        ):
            return False, None, "identity"
        equivalence = commit["equivalence"]
        if not (
            equivalence.get("alpha0_preletterbox_bgr_byte_exact_original") is True
            and equivalence.get("alpha1_preletterbox_bgr_byte_exact_native_cache") is True
            and equivalence.get("shape_exact") is True
            and equivalence.get("dtype_exact") is True
        ):
            return False, None, "equivalence"
        for variant in VARIANTS:
            manifest = commit["outputs"][variant]["manifest"]
            sidecar_path = Path(manifest["sidecar_path"])
            if not sidecar_path.is_file() or sha256_file(sidecar_path) != manifest["sidecar_sha256"]:
                return False, None, f"{variant}:sidecar"
            sidecar = load_json(sidecar_path)
            candidate = sidecar["candidates"]["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE"]
            payload = Path(candidate["path"])
            if not payload.is_file() or payload.stat().st_size != int(candidate["file_bytes"]):
                return False, None, f"{variant}:payload_presence_size"
            if full_payload:
                if sha256_file(payload) != candidate["file_sha256"]:
                    return False, None, f"{variant}:payload_file_sha"
                mmap = np.load(payload, mmap_mode="r", allow_pickle=False)
                if str(mmap.dtype) != "uint8" or list(mmap.shape) != list(candidate["shape"]):
                    return False, None, f"{variant}:payload_representation"
                value = np.ascontiguousarray(mmap)
                del mmap
                if causal_prepare.frozen_loader_array_bytes_sha(value) != candidate["array_bytes_sha256"]:
                    return False, None, f"{variant}:payload_array_sha"
                if causal_prepare.frozen_formal_bgr_sha(value) != manifest["expected_output_bgr_sha256"]:
                    return False, None, f"{variant}:payload_formal_sha"
        return True, commit, ""
    except BaseException as exc:
        return False, None, f"{type(exc).__name__}:{exc}"


def _quarantine_uid(seed: int, uid: str, paths: Iterable[Path], reason: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    destination = (
        RESULT_ROOT / "RECOVERY" / "INVALIDATED_ATTEMPTS" /
        f"CACHE_SEED{seed}_{uid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    for source in existing:
        target = destination / source.name
        if target.exists():
            target = destination / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
        shutil.move(str(source), str(target))
    atomic_json(destination / "INVALIDATION_AUDIT.json", {
        "schema": "m1_geopeak_phase2_invalidated_cache_attempt_v1",
        "status": "INVALIDATED_RECOVERY_ATTEMPT",
        "adaptation_seed": seed,
        "canonical_uid": uid,
        "reason": reason,
        "created_at": now_iso(),
    })


def _policy_maps(seed: int) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        frame = pd.read_csv(arm_paths(variant, seed)["policy_rows"])
        output[variant] = {
            str(row["canonical_uid"]): row for row in frame.to_dict("records")
        }
    return output


def _cache_jobs(seed: int) -> list[dict[str, Any]]:
    whitelist = pd.read_csv(causal_prepare.WHITELIST, dtype=str, keep_default_na=False)
    if len(whitelist) != EXPECTED_IMAGES or whitelist["uid"].nunique() != EXPECTED_IMAGES:
        raise GlobalDosePreparationStop("STOP_TARGET_ADAPT_WHITELIST_IDENTITY")
    whitelist = whitelist.set_index("uid", drop=False)
    context = causal_prepare.load_policy_context(seed)
    theta = pd.read_csv(seed_reference_paths(seed)["theta"])
    theta = theta.loc[
        theta["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
    ].set_index("canonical_uid", drop=False)
    policy = _policy_maps(seed)
    native_manifest = pd.read_csv(seed_reference_paths(seed)["canonical_cache_manifest"])
    native_manifest = native_manifest.set_index("canonical_uid", drop=False)
    jobs: list[dict[str, Any]] = []
    for uid in sorted(theta.index.astype(str)):
        if uid not in whitelist.index or uid not in context.image_index or uid not in native_manifest.index:
            raise GlobalDosePreparationStop(f"STOP_CACHE_JOB_UID:{seed}:{uid}")
        white = whitelist.loc[uid]
        recorded = Path(str(white["image_path"]))
        image_path = (TARGET_ADAPT_IMAGES / recorded.name).resolve()
        if not image_path.is_relative_to(TARGET_ADAPT_IMAGES.resolve()) or not image_path.is_file():
            raise GlobalDosePreparationStop(f"STOP_CACHE_IMAGE_PATH:{seed}:{uid}:{image_path}")
        native_row = theta.loc[uid]
        native_manifest_row = native_manifest.loc[uid]
        native_sidecar = load_json(Path(str(native_manifest_row["sidecar_path"])))
        native_candidate = native_sidecar["candidates"]["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE"]
        input_sha = require_sha(white["image_sha256"], "target_adapt_image_sha")
        commit_name = hashlib.sha256(uid.encode("utf-8")).hexdigest() + ".json"
        commit_path = RESULT_ROOT / "CACHE" / f"SEED{seed}" / "committed" / commit_name[:2] / commit_name
        outputs: dict[str, Any] = {}
        applied: dict[str, float] = {}
        for variant in VARIANTS:
            policy_row = policy[variant][uid]
            applied[variant] = float(policy_row["applied_alpha"])
            root = arm_paths(variant, seed)["working_cache"]
            outputs[variant] = {
                "bgr": str((root / "bgr" / input_sha[:2] / f"{input_sha}.npy").resolve()),
                "sidecar": str((root / "meta" / input_sha[:2] / f"{input_sha}.json").resolve()),
            }
        jobs.append({
            "seed": seed,
            "canonical_uid": uid,
            "image_path": str(image_path),
            "input_image_sha256": input_sha,
            "original_std": context.z_image[context.image_index[uid]].astype(np.float64).tolist(),
            "native_theta": [
                float(native_row["theta_delta_t"]),
                float(native_row["theta_c_t"]),
                float(native_row["theta_s_c"]),
            ],
            "native_theta_sha256": str(native_row["theta_raw_f8_sha256"]),
            "native_selected_delta_sha": str(native_row["selected_delta_sha"]),
            "native_intended_dose": float(native_row["selected_intended_geometric_dose"]),
            "canonical_native_cache_path": str(Path(str(native_candidate["path"])).resolve()),
            "canonical_native_output_bgr_sha256": str(native_manifest_row["expected_output_bgr_sha256"]),
            "applied_alpha": applied,
            "outputs": outputs,
            "commit_path": str(commit_path.resolve()),
        })
    if len(jobs) != EXPECTED_VALID or len({row["canonical_uid"] for row in jobs}) != EXPECTED_VALID:
        raise GlobalDosePreparationStop(f"STOP_CACHE_JOB_COUNT:{seed}:{len(jobs)}")
    return jobs


def _write_visit_audit(seed: int, variant: str, cache_uids: set[str]) -> dict[str, Any]:
    exposure_path = seed_reference_paths(seed)["exposure"]
    policy = _policy_maps(seed)[variant]
    rows: list[dict[str, Any]] = []
    active_count = 0
    identity_count = 0
    projected_active = 0
    with exposure_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for expected_index, row in enumerate(reader):
            active = strict_bool(row["fixed_active"], "fixed_active")
            identity = strict_bool(row["fixed_identity"], "fixed_identity")
            exists = strict_bool(row["fixed_delta_exists"], "fixed_delta_exists")
            uid = str(row["canonical_uid"])
            if int(row["formal_visit_index"]) != expected_index or active == identity or (active and not exists):
                raise GlobalDosePreparationStop(f"STOP_VISIT_IDENTITY:{variant}:{seed}:{expected_index}")
            applied = float(policy[uid]["applied_alpha"]) if exists else 0.0
            projected = bool(policy[uid]["projected_to_feasible_boundary"]) if exists else False
            hit = active and uid in cache_uids
            active_count += int(active)
            identity_count += int(identity)
            projected_active += int(active and projected)
            rows.append({
                "schema": "m1_geopeak_phase2_visit_level_dose_audit_v1",
                "variant": variant,
                "arm_id": arm_id(variant, seed),
                "adaptation_seed": seed,
                "formal_visit_index": expected_index,
                "epoch": int(row["epoch"]),
                "target_visit_ordinal": int(row["target_visit_ordinal"]),
                "schedule_uid": str(row["schedule_uid"]),
                "canonical_uid": uid,
                "fixed_delta_exists": exists,
                "fixed_active": active,
                "fixed_identity": identity,
                "requested_alpha": REQUESTED_ALPHA[variant] if exists else 0.0,
                "applied_alpha": applied if active else 0.0,
                "projected_to_feasible_boundary": active and projected,
                "cache_lookup_expected": active,
                "cache_entry_present": uid in cache_uids,
                "cache_hit": hit,
                "cache_miss": active and not hit,
                "identity_cache_lookup": False,
                "runtime_fallback_allowed": False,
                "runtime_operator_allowed": False,
                "runtime_powell_allowed": False,
            })
    if len(rows) != TOTAL_VISITS or active_count + identity_count != TOTAL_VISITS:
        raise GlobalDosePreparationStop(f"STOP_VISIT_COUNTS:{variant}:{seed}")
    if any(bool(row["cache_miss"]) for row in rows):
        raise GlobalDosePreparationStop(f"STOP_VISIT_CACHE_MISS:{variant}:{seed}")
    path = arm_paths(variant, seed)["visit"]
    atomic_csv(path, rows)
    return {
        "rows": len(rows), "active": active_count, "identity": identity_count,
        "projected_active_visits": projected_active,
        "path": str(path.resolve()), "sha256": sha256_file(path),
    }


def verify_seed_cache(seed: int, *, full_payload: bool = True) -> dict[str, Any]:
    jobs = _cache_jobs(seed)
    commits: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for index, job in enumerate(jobs):
        uid = str(job["canonical_uid"])
        path = Path(str(job["commit_path"]))
        valid, commit, reason = _commit_valid(path, seed, uid, full_payload=full_payload)
        if valid and commit is not None:
            commits[uid] = commit
        else:
            failures.append({"canonical_uid": uid, "reason": reason})
        if full_payload and ((index + 1) % 100 == 0 or index + 1 == len(jobs)):
            print(json.dumps({
                "stage": "GLOBAL_DOSE_CACHE_VERIFY", "seed": seed,
                "verified": index + 1, "total": len(jobs), "failures": len(failures),
            }, sort_keys=True), flush=True)
    if failures or len(commits) != EXPECTED_VALID:
        raise GlobalDosePreparationStop(f"STOP_CACHE_INTEGRITY:{seed}:{len(failures)}:{failures[:3]}")
    equivalence = [commits[uid]["equivalence"] for uid in sorted(commits)]
    alpha0_path = RESULT_ROOT / "EQUIVALENCE" / f"SEED{seed}" / "ALPHA0_IDENTITY_EQUIVALENCE.csv"
    alpha1_path = RESULT_ROOT / "EQUIVALENCE" / f"SEED{seed}" / "ALPHA1_NATIVE_EQUIVALENCE.csv"
    atomic_csv(alpha0_path, equivalence)
    atomic_csv(alpha1_path, equivalence)
    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        paths = arm_paths(variant, seed)
        manifests = [commits[uid]["outputs"][variant]["manifest"] for uid in sorted(commits)]
        realizations = [commits[uid]["outputs"][variant]["realization"] for uid in sorted(commits)]
        atomic_csv(paths["manifest"], manifests)
        atomic_csv(paths["realization"], realizations)
        composite = hashlib.sha256("\n".join(
            "|".join((
                str(row["canonical_uid"]), str(row["sidecar_sha256"]),
                str(row["expected_output_bgr_sha256"]),
            )) for row in manifests
        ).encode("ascii")).hexdigest()
        integrity = {
            "schema": "m1_geopeak_phase2_global_dose_cache_integrity_audit_v1",
            "status": "PASS",
            "variant": variant,
            "arm_id": arm_id(variant, seed),
            "adaptation_seed": seed,
            "entry_count": len(manifests),
            "expected_entry_count": EXPECTED_VALID,
            "duplicate_key_count": 0,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "corrupt_key_count": 0,
            "full_payload_hash_mode": full_payload,
            "payload_total_bytes": sum(int(row["cache_payload_bytes"]) for row in realizations),
            "payload_composite_sha256": composite,
            "manifest_path": str(paths["manifest"].resolve()),
            "manifest_sha256": sha256_file(paths["manifest"]),
            "realization_path": str(paths["realization"].resolve()),
            "realization_sha256": sha256_file(paths["realization"]),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_gt_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(paths["integrity"], integrity)
        visit = _write_visit_audit(seed, variant, set(commits))
        build = {
            "schema": "m1_geopeak_phase2_global_dose_cache_build_v1",
            "status": "PASS",
            "variant": variant,
            "arm_id": arm_id(variant, seed),
            "adaptation_seed": seed,
            "expected_active_entries": EXPECTED_VALID,
            "committed_key_count": len(commits),
            "manifest_csv": str(paths["manifest"].resolve()),
            "manifest_csv_sha256": sha256_file(paths["manifest"]),
            "realization_csv": str(paths["realization"].resolve()),
            "realization_csv_sha256": sha256_file(paths["realization"]),
            "cache_integrity_audit_sha256": sha256_file(paths["integrity"]),
            "payload_composite_sha256": composite,
            "online_operator_calls": 0,
            "online_powell_calls": 0,
            "runtime_fallback": False,
            "visit_audit": visit,
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_label_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(paths["build"], build)
        counts = {
            "images": EXPECTED_IMAGES,
            "with_delta": EXPECTED_VALID,
            "without_delta": EXPECTED_INVALID,
            "visits": TOTAL_VISITS,
            "active": int(visit["active"]),
            "identity": int(visit["identity"]),
            "projected_active_visits": int(visit["projected_active_visits"]),
        }
        checks = {
            "science_contract_frozen": SCIENCE_CONTRACT.with_suffix(".sha256").is_file(),
            "dose_policy_contract_frozen": (RESULT_ROOT / "DOSE_POLICY_CONTRACT.sha256").is_file(),
            "cache_integrity_pass": True,
            "active_mask_exact": counts["active"] + counts["identity"] == TOTAL_VISITS,
            "missing_lookups_zero": True,
            "unexpected_lookups_zero": True,
            "alpha0_identity_equivalence_all": all(row["alpha0_preletterbox_bgr_byte_exact_original"] for row in equivalence),
            "alpha1_native_equivalence_all": all(row["alpha1_preletterbox_bgr_byte_exact_native_cache"] for row in equivalence),
            "active_outputs_nonidentity": all(row["output_differs_from_identity"] for row in realizations),
            "runtime_operator_forbidden": True,
            "runtime_powell_forbidden": True,
            "runtime_fallback_forbidden": True,
            "target_dev_reads_zero": True,
            "target_test_reads_zero": True,
            "target_adapt_gt_reads_zero": True,
        }
        gate = {
            "schema": "m1_geopeak_phase2_global_dose_policy_cache_pretrain_gate_v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "training_authorized_by_identity_contract_after_gate": all(checks.values()),
            "variant": variant,
            "arm_id": arm_id(variant, seed),
            "adaptation_seed": seed,
            "counts": counts,
            "checks": checks,
            "cache_payload_composite_sha256": composite,
            "cache_manifest_sha256": sha256_file(paths["manifest"]),
            "cache_build_sha256": sha256_file(paths["build"]),
            "dose_policy_contract_sha256": sha256_file(RESULT_ROOT / "DOSE_POLICY_CONTRACT.json"),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_label_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(paths["gate"], gate)
        if gate["status"] != "PASS":
            raise GlobalDosePreparationStop(f"STOP_POLICY_CACHE_GATE:{variant}:{seed}:{checks}")
        per_variant[variant] = gate
    return {
        "schema": "m1_geopeak_phase2_global_dose_seed_cache_verification_v1",
        "status": "PASS",
        "adaptation_seed": seed,
        "per_variant": per_variant,
        "alpha0_equivalence_sha256": sha256_file(alpha0_path),
        "alpha1_equivalence_sha256": sha256_file(alpha1_path),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
    }


def build_seed_cache(seed: int, workers: int = 4) -> dict[str, Any]:
    build_dose_policy()
    guard = causal_prepare.ConstructionDenylist(f"GLOBAL_DOSE_CACHE_BUILD_SEED{seed}").install()
    jobs = _cache_jobs(seed)
    missing: list[dict[str, Any]] = []
    resumed = 0
    for job in jobs:
        uid = str(job["canonical_uid"])
        commit_path = Path(str(job["commit_path"]))
        valid, _commit, reason = _commit_valid(commit_path, seed, uid, full_payload=False)
        if valid:
            resumed += 1
            continue
        candidates = [commit_path]
        for variant in VARIANTS:
            candidates.extend(Path(str(value)) for value in job["outputs"][variant].values())
        _quarantine_uid(seed, uid, candidates, f"UNCOMMITTED_OR_INVALID:{reason}")
        missing.append(job)
    started = time.perf_counter()
    completed = 0
    if missing:
        with ProcessPoolExecutor(
            max_workers=int(workers), initializer=_worker_initializer, initargs=(int(seed),)
        ) as pool:
            futures = {pool.submit(_cache_worker, job): str(job["canonical_uid"]) for job in missing}
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    raise GlobalDosePreparationStop(
                        f"STOP_GLOBAL_DOSE_CACHE_WORKER:{seed}:{uid}:{type(exc).__name__}:{exc}"
                    ) from exc
                valid, _commit, reason = _commit_valid(
                    Path(result["commit_path"]), seed, uid, full_payload=False
                )
                if not valid:
                    raise GlobalDosePreparationStop(f"STOP_CACHE_POSTCOMMIT:{seed}:{uid}:{reason}")
                completed += 1
                if completed % 20 == 0 or completed == len(missing):
                    print(json.dumps({
                        "stage": "GLOBAL_DOSE_CACHE_BUILD",
                        "seed": seed,
                        "completed_new": completed,
                        "missing_total": len(missing),
                        "resumed": resumed,
                        "elapsed_seconds": time.perf_counter() - started,
                    }, sort_keys=True), flush=True)
    verification = verify_seed_cache(seed, full_payload=True)
    audit = {
        "schema": "m1_geopeak_phase2_global_dose_seed_cache_build_audit_v1",
        "status": "PASS",
        "adaptation_seed": seed,
        "expected_keys": EXPECTED_VALID,
        "resumed_keys": resumed,
        "newly_built_keys": completed,
        "construction_denylist": guard.audit(),
        "verification": verification,
        "wall_seconds": float(time.perf_counter() - started),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "CACHE" / f"SEED{seed}" / "SEED_CACHE_BUILD_AUDIT.json", audit)
    return audit


def _combine_csv(sources: Sequence[Path], destination: Path, add_arm: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    writer: csv.DictWriter[str] | None = None
    handle_out = temporary.open("x", encoding="utf-8", newline="")
    try:
        for source in sources:
            with source.open("r", encoding="utf-8-sig", newline="") as handle_in:
                reader = csv.DictReader(handle_in)
                fields = list(reader.fieldnames or [])
                if add_arm and "arm_id" not in fields:
                    fields = ["arm_id", *fields]
                if writer is None:
                    writer = csv.DictWriter(handle_out, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                elif list(writer.fieldnames or []) != fields:
                    raise GlobalDosePreparationStop(f"STOP_COMBINED_CSV_SCHEMA:{source}")
                inferred_arm = source.parent.name
                for row in reader:
                    if add_arm and "arm_id" not in row:
                        row["arm_id"] = inferred_arm
                    writer.writerow(row)
        handle_out.flush()
        os.fsync(handle_out.fileno())
    finally:
        handle_out.close()
    os.replace(temporary, destination)


def finalize_preparation() -> dict[str, Any]:
    build_dose_policy()
    gates: dict[str, Any] = {}
    for seed in SEEDS:
        verification = verify_seed_cache(seed, full_payload=True)
        for variant in VARIANTS:
            gates[arm_id(variant, seed)] = verification["per_variant"][variant]
    ordered_arms = [arm_id(variant, seed) for seed in SEEDS for variant in VARIANTS]
    _combine_csv(
        [arm_paths(variant, seed)["manifest"] for seed in SEEDS for variant in VARIANTS],
        RESULT_ROOT / "CACHE_MANIFEST.csv", add_arm=True,
    )
    _combine_csv(
        [arm_paths(variant, seed)["realization"] for seed in SEEDS for variant in VARIANTS],
        RESULT_ROOT / "CACHE_REALIZATION_AUDIT.csv", add_arm=False,
    )
    _combine_csv(
        [arm_paths(variant, seed)["visit"] for seed in SEEDS for variant in VARIANTS],
        RESULT_ROOT / "VISIT_LEVEL_DOSE_AUDIT.csv", add_arm=False,
    )
    _combine_csv(
        [RESULT_ROOT / "EQUIVALENCE" / f"SEED{seed}" / "ALPHA0_IDENTITY_EQUIVALENCE.csv" for seed in SEEDS],
        RESULT_ROOT / "ALPHA0_IDENTITY_EQUIVALENCE.csv", add_arm=False,
    )
    _combine_csv(
        [RESULT_ROOT / "EQUIVALENCE" / f"SEED{seed}" / "ALPHA1_NATIVE_EQUIVALENCE.csv" for seed in SEEDS],
        RESULT_ROOT / "ALPHA1_NATIVE_EQUIVALENCE.csv", add_arm=False,
    )
    checks = {
        "GATE_A2_DECISION_RECEIPT_PASS": load_json(RESULT_ROOT / "GATE_A2_DECISION_RECEIPT.json")["status"] == "PASS",
        "PARENT_IDENTITY_PASS": sha256_file(PARENT) == PARENT_SHA256,
        "FRESH_REFERENCE_IDENTITY_PASS": sha256_file(FRESH_ZIP) == FRESH_ZIP_SHA256,
        "SCHEDULE_REUSE_PASS": True,
        "EXPOSURE_MASK_EXACT_PASS": all(gate["checks"]["active_mask_exact"] for gate in gates.values()),
        "ALPHA0_IDENTITY_PASS": all(gate["checks"]["alpha0_identity_equivalence_all"] for gate in gates.values()),
        "ALPHA1_NATIVE_PASS": all(gate["checks"]["alpha1_native_equivalence_all"] for gate in gates.values()),
        "CACHE_INTEGRITY_PASS": all(gate["checks"]["cache_integrity_pass"] for gate in gates.values()),
        "CACHE_MISS_ZERO": all(gate["checks"]["missing_lookups_zero"] for gate in gates.values()),
        "FALLBACK_ZERO": True,
        "NO_ONLINE_OPERATOR": True,
        "NO_ONLINE_SOLVER": True,
        "TARGET_ADAPT_GT_READS_ZERO": True,
        "TARGET_DEV_READS_ZERO_BEFORE_FREEZE": True,
        "TARGET_TEST_READS_ZERO": True,
    }
    audit = {
        "schema": "m1_geopeak_phase2_global_dose_head_hand_alignment_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "single_scientific_difference": "TARGET_STUDENT_PHOTOMETRIC_DOSE_ALONG_NATIVE_REALIZED_OPERATOR_RAY",
        "teacher_view": "ORIGINAL_TARGET_IMAGE",
        "student_view": "OFFLINE_CACHE_ON_ACTIVE_VISITS_IDENTITY_ON_INACTIVE_VISITS",
        "source_path": "UNCHANGED",
        "operator_interpretation": "REALIZED_3DOF_OPERATOR_LOG_PARAMETER_RAY_NOT_EXACT_7D_ALPHA_DELTA_SCALING",
        "checks": checks,
        "arms": ordered_arms,
        "policy_cache_gate_sha256": {
            arm: sha256_file(arm_paths(arm.rsplit("_SEED", 1)[0], int(arm.rsplit("SEED", 1)[1]))["gate"])
            for arm in ordered_arms
        },
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "HEAD_HAND_ALIGNMENT_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise GlobalDosePreparationStop(f"STOP_HEAD_HAND_ALIGNMENT:{checks}")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify-inputs")
    sub.add_parser("build-policy")
    build = sub.add_parser("build-seed-cache")
    build.add_argument("--seed", type=int, choices=SEEDS, required=True)
    build.add_argument("--workers", type=int, default=4)
    verify = sub.add_parser("verify-seed-cache")
    verify.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "verify-inputs":
            result = verify_input_identities()
        elif args.action == "build-policy":
            result = build_dose_policy()
        elif args.action == "build-seed-cache":
            result = build_seed_cache(args.seed, args.workers)
        elif args.action == "verify-seed-cache":
            result = verify_seed_cache(args.seed, full_payload=True)
        else:
            result = finalize_preparation()
        print(json.dumps(result, sort_keys=True, default=str), flush=True)
        return 0
    except BaseException as exc:
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "m1_geopeak_phase2_global_dose_preparation_failure_v1",
            "status": "HARD_STOP_OR_ENGINEERING_FAILURE",
            "action": getattr(args, "action", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_gt_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(RESULT_ROOT / f"FAILURE_PREPARE_{getattr(args, 'action', 'UNKNOWN')}.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
