from __future__ import annotations

"""Prepare the frozen M1 Native-A Fixed causal controls.

This module is intentionally evaluation-blind.  It may read only frozen
Target-Adapt descriptors/images, fresh policy/schedule metadata, and frozen
scientific implementation authorities.  Evaluation identity verification is
performed by the separate package/evaluation processes.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from collections import Counter, deque
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
RESULT_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1"
SCIENCE_CONTRACT = CONTRACTS / "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1.json"
SCIENCE_CONTRACT_SHA256 = "ca16e949bc719d4268d5ae6eb83788657c5c653745a1fb6c8ca236c0ee08d55f"
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
VISITS_PER_EPOCH = 4416
TOTAL_VISITS = 128064
TARGET_BATCH = 8
EXPECTED_IMAGES = 3272
EXPECTED_WITH_DELTA = 3226
EXPECTED_WITHOUT_DELTA = 46
GENERIC = "GENERIC_FIXED_3DOF"
SHUFFLED = "FIXED_SHUFFLED"
VARIANTS = (GENERIC, SHUFFLED)
GENERIC_FORMAL = "POOLED_GLOBAL_MEDOID_GENERIC_FIXED_3DOF_V1"
SHUFFLED_FORMAL = "FIXED_SHUFFLED_EXACT_DELTA_MULTISET_V1"
Q2_DIR = PROJECT / "audits" / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
WHITELIST = Q2_DIR / "TARGET_ADAPT_STYLE_AUDIT_WHITELIST.csv"
TARGET_ADAPT_IMAGES = ROOT / "datasets" / "dataset_final_v2_1" / "target" / "adapt" / "images"

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import m1_routea_offline_theta_cache_engine_v1 as cache_engine  # noqa: E402
import m1_routea_photometric_realization_core_v1 as realization  # noqa: E402
import m1_routea_semantic_resource_policy_v1 as route_policy  # noqa: E402


class CausalPreparationStop(RuntimeError):
    """Fail-closed preparation, policy, or cache error."""


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


def frozen_loader_array_bytes_sha(array: np.ndarray) -> str:
    """Match the frozen Stage-1 loader's dtype+shape+payload hash exactly."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def frozen_formal_bgr_sha(array: np.ndarray) -> str:
    """Match the frozen realization/operator formal BGR hash exactly."""
    value = np.ascontiguousarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise CausalPreparationStop("STOP_CACHE_BGR_REPRESENTATION")
    digest = hashlib.sha256()
    digest.update(realization.ARRAY_HASH_PREFIX)
    digest.update(b"uint8")
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8") + b"\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CausalPreparationStop(f"STOP_JSON_OBJECT_REQUIRED:{path}")
    return value


def strict_bool(value: Any, role: str) -> bool:
    text = str(value).strip().casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise CausalPreparationStop(f"STOP_BOOL:{role}:{value}")


def require_sha(value: Any, role: str) -> str:
    digest = str(value or "").strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CausalPreparationStop(f"STOP_SHA:{role}:{value}")
    return digest


class ConstructionDenylist:
    """Process-wide audit-hook guard for policy/cache construction."""

    def __init__(self, phase: str) -> None:
        self.phase = str(phase)
        self.denied: list[dict[str, str]] = []
        self.sensitive_allowed: Counter[str] = Counter()
        self._installed = False

    @staticmethod
    def classify(path: str | Path) -> str:
        try:
            text = str(Path(path).resolve()).replace("/", "\\").casefold()
        except (OSError, ValueError):
            text = str(path).replace("/", "\\").casefold()
        if any(
            token in text
            for token in (
                "\\seed45\\evaluation\\",
                "\\seed46\\evaluation\\",
                "\\target\\dev\\",
                "\\target\\test\\",
                "\\target\\adapt\\labels\\",
            )
        ) or text.endswith("\\target_dev.yaml"):
            return "DENIED_SCIENTIFIC_DATA_OR_RESULT"
        if "\\target\\adapt\\images\\" in text:
            return "ALLOWED_TARGET_ADAPT_UNLABELED_IMAGE"
        if "fixed_a_" in text or "\\policy\\schedule\\" in text:
            return "ALLOWED_FROZEN_POLICY_OR_SCHEDULE"
        if "style_" in text or "descriptor" in text or "geometry" in text:
            return "ALLOWED_FROZEN_DESCRIPTOR_OR_GEOMETRY"
        return "OTHER"

    def _hook(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        classification = self.classify(os.fsdecode(candidate))
        if classification.startswith("DENIED"):
            row = {"classification": classification, "path": os.fsdecode(candidate)}
            self.denied.append(row)
            raise CausalPreparationStop(
                f"STOP_CONSTRUCTION_DENYLIST:{classification}:{row['path']}"
            )
        if classification.startswith("ALLOWED"):
            self.sensitive_allowed[classification] += 1

    def install(self) -> "ConstructionDenylist":
        if not self._installed:
            sys.addaudithook(self._hook)
            self._installed = True
        return self

    def audit(self) -> dict[str, Any]:
        return {
            "schema": "m1_fixed_causal_controls_construction_denylist_audit_v1",
            "phase": self.phase,
            "status": "PASS" if not self.denied else "FAIL",
            "denylist_installed": self._installed,
            "denied_attempt_count": len(self.denied),
            "denied_attempts": self.denied,
            "sensitive_allowed_open_event_counts": dict(sorted(self.sensitive_allowed.items())),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "target_adapt_gt_reads": 0,
        }


def verify_science_contract() -> dict[str, Any]:
    observed = sha256_file(SCIENCE_CONTRACT)
    sidecar = SCIENCE_CONTRACT.with_suffix(".sha256").read_text(encoding="ascii").strip().split()[0]
    contract = load_json(SCIENCE_CONTRACT)
    checks = {
        "sha": observed == SCIENCE_CONTRACT_SHA256 == sidecar,
        "schema": contract.get("schema") == "m1_native_a_fixed_causal_controls_45_46_v1_contract_v1",
        "status": contract.get("status") == "FROZEN_BEFORE_IMPLEMENTATION_CACHE_TRAINING_OR_NEW_TARGET_DEV_ACCESS",
        "parent": contract.get("canonical_inputs", {}).get("canonical_e71_parent", {}).get("sha256") == PARENT_SHA256,
        "target_test_sealed": contract.get("data_access_contract", {}).get("target_test_state") == "SEALED",
    }
    if not all(checks.values()):
        raise CausalPreparationStop(f"STOP_SCIENCE_CONTRACT:{checks}")
    return {"path": str(SCIENCE_CONTRACT.resolve()), "sha256": observed, "checks": checks}


def seed_reference_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise CausalPreparationStop(f"STOP_SEED:{seed}")
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


def variant_paths(variant: str, seed: int) -> dict[str, Path]:
    if variant not in VARIANTS or int(seed) not in SEEDS:
        raise CausalPreparationStop(f"STOP_VARIANT_SEED:{variant}:{seed}")
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


def _schedule_uids(path: Path) -> list[str]:
    output: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            epoch = int(row["epoch"])
            if epoch in EPOCHS:
                output.extend(str(uid) for uid in row["uids"])
    if len(output) != TOTAL_VISITS:
        raise CausalPreparationStop(f"STOP_TARGET_SCHEDULE_VISITS:{path}:{len(output)}")
    return output


def load_policy_context(seed: int) -> Any:
    schedule = seed_reference_paths(seed)["target_schedule"]
    expected = route_policy.EXPECTED_HASHES["target_schedule"]
    route_policy.EXPECTED_HASHES["target_schedule"] = sha256_file(schedule)
    try:
        return route_policy.load_policy_context(
            ROOT / route_policy.DEFAULT_GEOMETRY_REL,
            ROOT / route_policy.DEFAULT_Q2_REL,
            schedule,
            _schedule_uids(schedule),
        )
    finally:
        route_policy.EXPECTED_HASHES["target_schedule"] = expected


def load_active_selection(seed: int) -> pd.DataFrame:
    frame = pd.read_csv(seed_reference_paths(seed)["selection"])
    exists = frame["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
    active = frame.loc[exists].copy()
    active["adaptation_seed"] = int(seed)
    active = active.sort_values("canonical_uid", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_IMAGES or len(active) != EXPECTED_WITH_DELTA:
        raise CausalPreparationStop(f"STOP_SELECTION_COUNTS:{seed}:{len(frame)}:{len(active)}")
    if active["canonical_uid"].nunique() != len(active):
        raise CausalPreparationStop(f"STOP_SELECTION_UID_DUPLICATE:{seed}")
    delta_columns = [f"selected_delta_{index}" for index in range(7)]
    values = active[delta_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise CausalPreparationStop(f"STOP_SELECTION_DELTA_NONFINITE:{seed}")
    active["selected_delta_csv_f8_sha256"] = [
        hashlib.sha256(b"M1_FIXED_SHUFFLED_CSV_F8_DELTA_V1\0" + np.asarray(row, dtype="<f8").tobytes()).hexdigest()
        for row in values
    ]
    for digest in active["selected_delta_sha"].astype(str):
        require_sha(digest, "selected_delta_sha")
    return active


def load_whitelist() -> pd.DataFrame:
    frame = pd.read_csv(WHITELIST, dtype=str, keep_default_na=False)
    required = {"uid", "image_path", "image_sha256", "capture_event_id", "location_id"}
    if not required.issubset(frame.columns) or len(frame) != EXPECTED_IMAGES:
        raise CausalPreparationStop("STOP_WHITELIST_SCHEMA_OR_COUNT")
    frame = frame.sort_values("uid", kind="mergesort").reset_index(drop=True)
    if frame["uid"].nunique() != len(frame):
        raise CausalPreparationStop("STOP_WHITELIST_UID_DUPLICATE")
    canonical_paths: list[str] = []
    canonical_names: set[str] = set()
    for row in frame.itertuples(index=False):
        recorded = Path(str(row.image_path))
        if recorded.name in canonical_names:
            raise CausalPreparationStop(f"STOP_WHITELIST_IMAGE_BASENAME_DUPLICATE:{recorded.name}")
        canonical_names.add(recorded.name)
        path = (TARGET_ADAPT_IMAGES / recorded.name).resolve()
        if not path.is_relative_to(TARGET_ADAPT_IMAGES.resolve()):
            raise CausalPreparationStop(f"STOP_WHITELIST_IMAGE_PATH:{path}")
        expected_sha = require_sha(row.image_sha256, "whitelist_image_sha")
        if not path.is_file():
            raise CausalPreparationStop(f"STOP_WHITELIST_IMAGE_MISSING:{path}")
        if sha256_file(path) != expected_sha:
            raise CausalPreparationStop(f"STOP_WHITELIST_IMAGE_SHA:{row.uid}:{path}")
        canonical_paths.append(str(path))
    frame["recorded_image_path"] = frame["image_path"]
    frame["image_path"] = canonical_paths
    return frame


def select_generic_theta() -> dict[str, Any]:
    verify_science_contract()
    guard = ConstructionDenylist("GENERIC_THETA_SELECTION").install()
    family = RESULT_ROOT / "GENERIC"
    audit_path = family / "GENERIC_THETA_SELECTION_AUDIT.json"
    candidates_path = family / "GENERIC_THETA_CANDIDATES.csv"
    contract_path = family / "GENERIC_FIXED_POLICY_CONTRACT.json"
    if audit_path.is_file() and candidates_path.is_file() and contract_path.is_file():
        audit = load_json(audit_path)
        locks = audit.get("outputs", {})
        if (
            audit.get("status") == "PASS"
            and sha256_file(candidates_path) == locks.get("candidates_sha256")
            and sha256_file(contract_path) == locks.get("contract_sha256")
        ):
            return audit
        raise CausalPreparationStop("STOP_EXISTING_GENERIC_SELECTION_DRIFT")
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    for seed in SEEDS:
        path = seed_reference_paths(seed)["theta"]
        frame = pd.read_csv(path)
        banned = [
            column for column in frame.columns
            if any(token in column.casefold() for token in ("map50", "target_dev", "evaluation_result", "location_improvement"))
        ]
        if banned:
            raise CausalPreparationStop(f"STOP_GENERIC_EVALUATION_FIELD:{banned}")
        exists = frame["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
        solved = frame["realization_status"].astype(str).eq("SOLVED_RETAINED_NO_RESELECTION")
        frame = frame.loc[exists & solved].copy()
        frame["adaptation_seed"] = int(seed)
        theta = frame[["theta_delta_t", "theta_c_t", "theta_s_c"]].to_numpy(dtype=np.float64)
        in_bounds = (
            np.isfinite(theta).all(axis=1)
            & (theta >= realization.PARAMETER_LOWER[None, :]).all(axis=1)
            & (theta <= realization.PARAMETER_UPPER[None, :]).all(axis=1)
        )
        if len(frame) != EXPECTED_WITH_DELTA or not in_bounds.all():
            raise CausalPreparationStop(f"STOP_GENERIC_CANDIDATE_FILTER:{seed}:{len(frame)}")
        frames.append(frame)
        inputs.append({"adaptation_seed": seed, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    bank = pd.concat(frames, ignore_index=True)
    dose = bank["selected_intended_geometric_dose"].to_numpy(dtype=np.float64)
    z = np.column_stack((
        bank["theta_delta_t"].to_numpy(dtype=np.float64),
        np.log(bank["theta_c_t"].to_numpy(dtype=np.float64)),
        np.log(bank["theta_s_c"].to_numpy(dtype=np.float64)),
    ))
    if len(bank) != 2 * EXPECTED_WITH_DELTA or not np.isfinite(dose).all() or not np.isfinite(z).all():
        raise CausalPreparationStop("STOP_GENERIC_BANK_IDENTITY")
    m_d = float(np.median(dose))
    q25, q75 = np.quantile(dose, [0.25, 0.75], method="linear")
    s_d = max(float(q75 - q25), 1e-12)
    m_k = np.median(z, axis=0)
    mad = np.median(np.abs(z - m_k[None, :]), axis=0)
    s_k = np.maximum(1.4826 * mad, 1e-12)
    score = np.abs(dose - m_d) / s_d + np.sum(np.abs(z - m_k[None, :]) / s_k[None, :], axis=1)
    bank["generic_score"] = score
    bank["generic_dose_center"] = m_d
    bank["generic_dose_scale_iqr"] = s_d
    for index in range(3):
        bank[f"generic_z{index}_center"] = float(m_k[index])
        bank[f"generic_z{index}_scale_1p4826_mad"] = float(s_k[index])
    bank = bank.sort_values(
        ["generic_score", "theta_raw_f8_sha256", "adaptation_seed", "canonical_uid"],
        kind="mergesort",
    ).reset_index(drop=True)
    selected = bank.iloc[0]
    theta = np.asarray(
        [selected["theta_delta_t"], selected["theta_c_t"], selected["theta_s_c"]],
        dtype="<f8",
    )
    declared_theta_sha = require_sha(
        selected["theta_raw_f8_sha256"], "selected_declared_theta_raw_f8_sha256"
    )
    materialized_theta_sha = cache_engine.theta_raw_f8_sha256(theta)
    candidate_columns = [
        "adaptation_seed", "canonical_uid", "selected_intended_geometric_dose",
        "theta_delta_t", "theta_c_t", "theta_s_c", "theta_raw_f8_sha256",
        "generic_score", "generic_dose_center", "generic_dose_scale_iqr",
        "generic_z0_center", "generic_z0_scale_1p4826_mad",
        "generic_z1_center", "generic_z1_scale_1p4826_mad",
        "generic_z2_center", "generic_z2_scale_1p4826_mad",
    ]
    atomic_csv(candidates_path, bank[candidate_columns].to_dict("records"), candidate_columns)
    policy_contract = {
        "schema": "m1_generic_fixed_policy_contract_v1",
        "status": "FROZEN_BEFORE_CACHE_OR_TRAINING",
        "task_id": "M1_NATIVE_A_FIXED_CAUSAL_CONTROLS_45_46_V1",
        "formal_name": GENERIC_FORMAL,
        "science_contract": {"path": str(SCIENCE_CONTRACT.resolve()), "sha256": SCIENCE_CONTRACT_SHA256},
        "selection_inputs": inputs,
        "pooled_row_count": len(bank),
        "quantile_method": "numpy.quantile_method_linear",
        "robust_statistics": {
            "median_d": m_d,
            "q25_d": float(q25),
            "q75_d": float(q75),
            "iqr_d_scale": s_d,
            "median_z": [float(value) for value in m_k],
            "mad_z": [float(value) for value in mad],
            "scale_1p4826_mad_z": [float(value) for value in s_k],
        },
        "selected": {
            "source_adaptation_seed": int(selected["adaptation_seed"]),
            "source_canonical_uid": str(selected["canonical_uid"]),
            "theta": [float(value) for value in theta],
            "theta_raw_f8_sha256": declared_theta_sha,
            "declared_pre_csv_theta_raw_f8_sha256": declared_theta_sha,
            "materialized_csv_theta_raw_f8_sha256": materialized_theta_sha,
            "theta_sha_representation_boundary": (
                "The historical declared SHA locks the pre-CSV theta lineage; "
                "the materialized SHA locks the exact float64 values deployed from the canonical CSV."
            ),
            "selected_intended_geometric_dose": float(selected["selected_intended_geometric_dose"]),
            "generic_score": float(selected["generic_score"]),
        },
        "tie_break": ["generic_score", "theta_raw_f8_sha256", "adaptation_seed", "canonical_uid"],
        "same_theta_across_seed45_and_seed46": True,
        "active_behavior": "APPLY_SHARED_THETA_TO_RECIPIENT_ORIGINAL_PRELETTERBOX_BGR",
        "identity_behavior": "BYTE_IDENTICAL_PASSTHROUGH",
        "per_image_reselection_resolve_or_fallback": False,
        "no_evaluation_fields_accessed": True,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
    }
    atomic_json(contract_path, policy_contract)
    audit = {
        **policy_contract,
        "schema": "m1_generic_theta_selection_audit_v1",
        "status": "PASS",
        "construction_denylist": guard.audit(),
        "outputs": {
            "candidates_path": str(candidates_path.resolve()),
            "candidates_sha256": sha256_file(candidates_path),
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": sha256_file(contract_path),
        },
        "created_at": now_iso(),
    }
    atomic_json(audit_path, audit)
    return audit


def _hopcroft_karp(adjacency: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(adjacency)
    pair_u = np.full(n, -1, dtype=np.int32)
    pair_v = np.full(n, -1, dtype=np.int32)
    distance = np.zeros(n, dtype=np.int32)
    infinity = n + 1

    def bfs() -> bool:
        queue: deque[int] = deque()
        shortest = infinity
        for u in range(n):
            if pair_u[u] < 0:
                distance[u] = 0
                queue.append(u)
            else:
                distance[u] = infinity
        while queue:
            u = queue.popleft()
            if distance[u] >= shortest:
                continue
            for value in adjacency[u]:
                v = int(value)
                paired = int(pair_v[v])
                if paired < 0:
                    shortest = distance[u] + 1
                elif distance[paired] == infinity:
                    distance[paired] = distance[u] + 1
                    queue.append(paired)
        return shortest != infinity

    def dfs(u: int) -> bool:
        for value in adjacency[u]:
            v = int(value)
            paired = int(pair_v[v])
            if paired < 0 or (distance[paired] == distance[u] + 1 and dfs(paired)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        distance[u] = infinity
        return False

    while bfs():
        for u in range(n):
            if pair_u[u] < 0:
                dfs(u)
    return pair_u, pair_v


def build_shuffle_policy(seed: int) -> dict[str, Any]:
    verify_science_contract()
    guard = ConstructionDenylist(f"SHUFFLE_POLICY_SEED{seed}").install()
    paths = variant_paths(SHUFFLED, seed)
    if paths["graph_audit"].is_file() and paths["assignment"].is_file() and paths["multiset_audit"].is_file() and paths["policy_contract"].is_file():
        audit = load_json(paths["graph_audit"])
        if audit.get("status") == "PASS" and sha256_file(paths["assignment"]) == audit.get("assignment_sha256"):
            return audit
        raise CausalPreparationStop(f"STOP_EXISTING_SHUFFLE_POLICY_DRIFT:{seed}")
    source = load_active_selection(seed)
    context = load_policy_context(seed)
    uids = source["canonical_uid"].astype(str).tolist()
    if set(uids) - set(context.image_index):
        raise CausalPreparationStop(f"STOP_SHUFFLE_CONTEXT_UID_SET:{seed}")
    delta_columns = [f"selected_delta_{index}" for index in range(7)]
    deltas = source[delta_columns].to_numpy(dtype=np.float64)
    recipient_indices = np.asarray([context.image_index[uid] for uid in uids], dtype=np.int32)
    anchors = context.z_image[recipient_indices]
    declared_shas = source["selected_delta_sha"].astype(str).tolist()
    n = len(uids)
    adjacency: list[np.ndarray] = []
    degree: list[int] = []
    adjacency_hashes: list[str] = []
    edge_count = 0
    for recipient in range(n):
        targets = anchors[recipient][None, :] + deltas
        internal, _reasons = route_policy.validity_arrays(targets, context.median, context.scale)
        distance, _nearest = context.event_tree.query(
            targets * route_policy.WEIGHTS,
            k=1,
            p=1,
            eps=0,
            workers=1,
        )
        legal = internal & (np.asarray(distance, dtype=np.float64) <= context.tau)
        legal[recipient] = False
        donors = np.flatnonzero(legal).tolist()
        ranked = sorted(
            donors,
            key=lambda donor: hashlib.sha256(
                f"M1_FIXED_SHUFFLED_V1|{seed}|{uids[recipient]}|{uids[donor]}|{declared_shas[donor]}".encode("utf-8")
            ).digest(),
        )
        row = np.asarray(ranked, dtype=np.int32)
        adjacency.append(row)
        degree.append(len(row))
        edge_count += len(row)
        adjacency_hashes.append(hashlib.sha256(row.astype("<i4", copy=False).tobytes()).hexdigest())
    if min(degree, default=0) <= 0:
        raise CausalPreparationStop(f"STOP_SHUFFLE_ZERO_DEGREE:{seed}")
    pair_u, pair_v = _hopcroft_karp(adjacency)
    matched = int(np.count_nonzero(pair_u >= 0))
    if matched != n:
        hard_stop = {
            "schema": "m1_fixed_shuffled_hard_stop_v1",
            "status": "HARD_STOP",
            "reason": "SHUFFLED_SUPPORT_COMPATIBLE_PERFECT_DERANGEMENT_NOT_CONSTRUCTIBLE",
            "adaptation_seed": seed,
            "nodes": n,
            "matched": matched,
            "edge_count": edge_count,
            "min_degree": min(degree),
            "created_at": now_iso(),
        }
        atomic_json(paths["family"] / "HARD_STOP_AUDIT.json", hard_stop)
        raise CausalPreparationStop(f"STOP_SHUFFLE_NO_PERFECT_DERANGEMENT:{seed}:{matched}:{n}")
    rows: list[dict[str, Any]] = []
    for recipient, donor_value in enumerate(pair_u.tolist()):
        donor = int(donor_value)
        if donor == recipient:
            raise CausalPreparationStop(f"STOP_SHUFFLE_SELF_ASSIGNMENT:{seed}:{recipient}")
        target = anchors[recipient] + deltas[donor]
        internal, reasons = route_policy.validity_arrays(target[None, :], context.median, context.scale)
        distance, nearest = context.event_tree.query(target * route_policy.WEIGHTS, k=1, p=1, eps=0, workers=1)
        edge_material = f"M1_FIXED_SHUFFLED_V1|{seed}|{uids[recipient]}|{uids[donor]}|{declared_shas[donor]}"
        row: dict[str, Any] = {
            "schema": "m1_fixed_shuffled_assignment_v1",
            "adaptation_seed": seed,
            "recipient_uid": uids[recipient],
            "donor_uid": uids[donor],
            "recipient_index": recipient,
            "donor_index": donor,
            "donor_delta_sha": declared_shas[donor],
            "donor_delta_csv_f8_sha256": str(source.iloc[donor]["selected_delta_csv_f8_sha256"]),
            "donor_selected_intended_geometric_dose": float(source.iloc[donor]["selected_intended_geometric_dose"]),
            "edge_rank_sha256": hashlib.sha256(edge_material.encode("utf-8")).hexdigest(),
            "recipient_internal_valid": bool(internal[0]),
            "recipient_internal_invalid_reason": str(reasons[0]),
            "recipient_support_deviation": float(distance),
            "recipient_endpoint_in_tube": bool(float(distance) <= context.tau),
            "nearest_event_id": str(context.event_ids[int(nearest)]),
        }
        for index in range(7):
            row[f"donor_delta_{index}"] = float(deltas[donor, index])
            row[f"recipient_anchor_std_{index}"] = float(anchors[recipient, index])
            row[f"recipient_target_std_{index}"] = float(target[index])
        rows.append(row)
    atomic_csv(paths["assignment"], rows)
    donor_indices = [int(row["donor_index"]) for row in rows]
    declared_native = sorted(declared_shas)
    declared_shuffle = sorted(str(row["donor_delta_sha"]) for row in rows)
    csv_native = sorted(source["selected_delta_csv_f8_sha256"].astype(str).tolist())
    csv_shuffle = sorted(str(row["donor_delta_csv_f8_sha256"]) for row in rows)
    native_dose_bytes = sorted(np.asarray([float(value)], dtype="<f8").tobytes() for value in source["selected_intended_geometric_dose"])
    shuffle_dose_bytes = sorted(np.asarray([float(row["donor_selected_intended_geometric_dose"])], dtype="<f8").tobytes() for row in rows)
    multiset_checks = {
        "donor_delta_multiset_exact_equal_to_native": csv_native == csv_shuffle,
        "donor_delta_sha_multiset_exact_equal": declared_native == declared_shuffle,
        "dose_multiset_exact_equal": native_dose_bytes == shuffle_dose_bytes,
        "self_assignment_count_zero": all(int(row["recipient_index"]) != int(row["donor_index"]) for row in rows),
        "missing_donor_count_zero": set(donor_indices) == set(range(n)),
        "duplicate_donor_count_zero": len(donor_indices) == len(set(donor_indices)),
        "all_edges_internal_valid": all(bool(row["recipient_internal_valid"]) for row in rows),
        "all_edges_in_frozen_tube": all(bool(row["recipient_endpoint_in_tube"]) for row in rows),
    }
    multiset = {
        "schema": "m1_fixed_shuffled_multiset_audit_v1",
        "status": "PASS" if all(multiset_checks.values()) else "FAIL",
        "adaptation_seed": seed,
        "node_count": n,
        "checks": multiset_checks,
        "native_declared_delta_sha_multiset_sha256": hashlib.sha256("\n".join(declared_native).encode("ascii")).hexdigest(),
        "shuffled_declared_delta_sha_multiset_sha256": hashlib.sha256("\n".join(declared_shuffle).encode("ascii")).hexdigest(),
        "native_csv_f8_delta_multiset_sha256": hashlib.sha256("\n".join(csv_native).encode("ascii")).hexdigest(),
        "shuffled_csv_f8_delta_multiset_sha256": hashlib.sha256("\n".join(csv_shuffle).encode("ascii")).hexdigest(),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
    }
    atomic_json(paths["multiset_audit"], multiset)
    if multiset["status"] != "PASS":
        raise CausalPreparationStop(f"STOP_SHUFFLE_MULTISET:{seed}:{multiset_checks}")
    policy = {
        "schema": "m1_fixed_shuffled_policy_contract_v1",
        "status": "FROZEN_BEFORE_REALIZATION_CACHE_OR_TRAINING",
        "formal_name": SHUFFLED_FORMAL,
        "adaptation_seed": seed,
        "science_contract_sha256": SCIENCE_CONTRACT_SHA256,
        "selection_path": str(seed_reference_paths(seed)["selection"].resolve()),
        "selection_sha256": sha256_file(seed_reference_paths(seed)["selection"]),
        "assignment_path": str(paths["assignment"].resolve()),
        "assignment_sha256": sha256_file(paths["assignment"]),
        "node_count": n,
        "legal_edge_rule": "RECIPIENT_ANCHOR_PLUS_DONOR_SELECTED_DELTA_INTERNAL_VALID_AND_EMPIRICAL_TARGET_SUPPORT_TUBE_AND_NOT_SELF",
        "edge_rank_material": "M1_FIXED_SHUFFLED_V1|adaptation_seed|recipient_uid|donor_uid|donor_delta_sha",
        "matching_algorithm": "DETERMINISTIC_HASH_RANKED_HOPCROFT_KARP_V1",
        "operator_solver": {
            "operator": "FROZEN_NATIVE_A_3DOF_RGB",
            "method": "Powell",
            "single_start": True,
            "maxfev": 400,
            "solver_options": dict(realization.SOLVER_OPTIONS),
            "parameter_lower": realization.PARAMETER_LOWER.tolist(),
            "parameter_upper": realization.PARAMETER_UPPER.tolist(),
            "retain_first_finite_result_even_if_solver_success_false": True,
            "result_driven_reselection": False,
        },
        "interpretation_boundary": "Exact Delta multiset and support-valid assignment can be matched. Actual support proximity and realizability need not be exactly matched.",
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
    }
    atomic_json(paths["policy_contract"], policy)
    graph = {
        "schema": "m1_fixed_shuffled_graph_audit_v1",
        "status": "PASS",
        "adaptation_seed": seed,
        "nodes": n,
        "legal_edge_count": edge_count,
        "min_degree": int(min(degree)),
        "median_degree": float(np.median(degree)),
        "max_degree": int(max(degree)),
        "perfect_matching_cardinality": matched,
        "self_assignment_count": int(sum(index == donor for index, donor in enumerate(pair_u.tolist()))),
        "adjacency_rows_composite_sha256": hashlib.sha256("\n".join(adjacency_hashes).encode("ascii")).hexdigest(),
        "assignment_path": str(paths["assignment"].resolve()),
        "assignment_sha256": sha256_file(paths["assignment"]),
        "multiset_audit_sha256": sha256_file(paths["multiset_audit"]),
        "policy_contract_sha256": sha256_file(paths["policy_contract"]),
        "construction_denylist": guard.audit(),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["graph_audit"], graph)
    return graph


_WORKER_CONTEXT: Any | None = None
_WORKER_GUARD: ConstructionDenylist | None = None


def _cache_worker_initializer(seed: int) -> None:
    global _WORKER_CONTEXT, _WORKER_GUARD
    _WORKER_GUARD = ConstructionDenylist(f"CACHE_WORKER_SEED{seed}").install()
    realization.configure_q2_snapshot(Q2_DIR / "CODE_SNAPSHOT")
    _WORKER_CONTEXT = load_policy_context(seed)


def _reconstruct_bgr(palette: Any, deployment_rgb_uint8: np.ndarray) -> np.ndarray:
    palette_bgr = np.ascontiguousarray(np.asarray(deployment_rgb_uint8, dtype=np.uint8)[:, ::-1])
    return np.ascontiguousarray(
        palette_bgr[palette.inverse_flat].reshape(palette.height, palette.width, 3)
    )


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}" + path.suffix)
    with temporary.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cache_job(job: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise CausalPreparationStop("STOP_CACHE_WORKER_CONTEXT")
    context = _WORKER_CONTEXT
    started = time.perf_counter()
    image_path = Path(str(job["image_path"])).resolve()
    if ConstructionDenylist.classify(image_path) != "ALLOWED_TARGET_ADAPT_UNLABELED_IMAGE":
        raise CausalPreparationStop(f"STOP_CACHE_IMAGE_CLASSIFICATION:{image_path}")
    input_sha = require_sha(job["input_image_sha256"], "job_input_sha")
    if sha256_file(image_path) != input_sha:
        raise CausalPreparationStop(f"STOP_CACHE_INPUT_IMAGE_SHA:{job['canonical_uid']}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise CausalPreparationStop(f"STOP_CACHE_DECODE:{job['canonical_uid']}")
    palette = realization.PaletteImage.from_rgb_uint8(np.ascontiguousarray(bgr[:, :, ::-1]))
    original_std = np.asarray(job["original_std"], dtype=np.float64)
    target_std = np.asarray(job["target_std"], dtype=np.float64)
    if str(job["variant"]) == GENERIC:
        theta = np.asarray(job["theta"], dtype=np.float64)
        final = palette.final_evaluation(theta)
        original_raw = realization.inverse_standardize(original_std, context.median, context.scale)
        deployment_std = realization.standardize(final["deployment_raw"], context.median, context.scale)
        actual_dose = realization.dstyle(original_std, deployment_std)
        support = float(np.min(realization.dstyle_rows(context.z_event, deployment_std[None, :])))
        internal, reason = realization.descriptor_internal_valid_raw(final["deployment_raw"])
        lower_hits = np.abs(theta - realization.PARAMETER_LOWER) <= realization.BOUND_HIT_TOL
        upper_hits = np.abs(theta - realization.PARAMETER_UPPER) <= realization.BOUND_HIT_TOL
        row: dict[str, Any] = {
            "schema": "m1_generic_fixed_realization_audit_v1",
            "adaptation_seed_scope": "SHARED_SEED45_SEED46",
            "canonical_uid": str(job["canonical_uid"]),
            "input_image_sha256": input_sha,
            "theta_delta_t": float(theta[0]),
            "theta_c_t": float(theta[1]),
            "theta_s_c": float(theta[2]),
            "theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(theta),
            "selected_source_intended_dose": float(job["intended_dose"]),
            "actual_effective_dose": float(actual_dose),
            "actual_support_deviation": support,
            "actual_endpoint_in_tube": bool(support <= context.tau),
            "actual_descriptor_internal_valid": bool(internal),
            "actual_descriptor_internal_invalid_reason": str(reason),
            "any_parameter_bound_hit": bool(np.any(lower_hits | upper_hits)),
            "image_height": int(palette.height),
            "image_width": int(palette.width),
            "unique_rgb_count": int(len(palette.palette_uint8)),
            "operator_output_array_sha256": str(final["output_array_sha256"]),
        }
        row.update(final["diagnostics"])
        for index, name in enumerate(realization.BASE):
            row[f"original_raw__{name}"] = float(original_raw[index])
            row[f"deployment_raw__{name}"] = float(final["deployment_raw"][index])
            row[f"original_std__{name}"] = float(original_std[index])
            row[f"deployment_std__{name}"] = float(deployment_std[index])
    else:
        target_delta_sha = require_sha(job["donor_delta_sha"], "donor_delta_sha")
        row = realization.realization_case(
            palette=palette,
            uid=str(job["canonical_uid"]),
            event_id=str(job["capture_event_id"]),
            case_type=SHUFFLED_FORMAL,
            input_image_sha256=input_sha,
            decoder="cv2.IMREAD_COLOR_BGR_TO_RGB",
            original_std=original_std,
            target_std=target_std,
            intended_dose=float(job["intended_dose"]),
            target_delta_sha256=target_delta_sha,
            donor_uid=str(job["donor_uid"]),
            donor_event_id=str(job["donor_event_id"]),
            median=context.median,
            scale=context.scale,
            event_std=context.z_event,
        )
        theta = np.asarray([row["delta_t"], row["c_t"], row["s_c"]], dtype=np.float64)
        final = palette.final_evaluation(theta)
        if str(final["output_array_sha256"]) != str(row["output_array_sha256"]):
            raise CausalPreparationStop(f"STOP_SHUFFLED_OUTPUT_REPLAY:{job['canonical_uid']}")
        row.update({
            "schema": "m1_fixed_shuffled_theta_realization_v1",
            "adaptation_seed": int(job["seed"]),
            "recipient_uid": str(job["canonical_uid"]),
            "donor_uid": str(job["donor_uid"]),
            "donor_delta_sha": target_delta_sha,
            "donor_delta_csv_f8_sha256": str(job["donor_delta_csv_f8_sha256"]),
            "theta_delta_t": float(theta[0]),
            "theta_c_t": float(theta[1]),
            "theta_s_c": float(theta[2]),
            "theta_raw_f8_sha256": cache_engine.theta_raw_f8_sha256(theta),
            "realization_status": "SOLVED_RETAINED_NO_RESELECTION",
        })
    output_bgr = _reconstruct_bgr(palette, final["deployment_rgb_uint8"])
    output_array_sha = str(final["output_array_sha256"])
    bgr_path = Path(str(job["bgr_path"])).resolve()
    sidecar_path = Path(str(job["sidecar_path"])).resolve()
    commit_path = Path(str(job["commit_path"])).resolve()
    _write_npy_atomic(bgr_path, output_bgr)
    file_sha = sha256_file(bgr_path)
    array_sha = frozen_loader_array_bytes_sha(output_bgr)
    theta_sha = cache_engine.theta_raw_f8_sha256(theta)
    solve_key = canonical_sha({
        "namespace": "M1_FIXED_CAUSAL_CONTROL_CACHE_V1",
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "uid": str(job["canonical_uid"]),
        "input_sha": input_sha,
        "theta_sha": theta_sha,
        "donor_delta_sha": str(job.get("donor_delta_sha", "")),
    })
    sidecar = {
        "schema": "m1_native_a_fixed_cache_entry_v1",
        "cache_namespace": "M1_FIXED_CAUSAL_CONTROL_CACHE_V1",
        "cache_key_sha256": canonical_sha({"solve_key": solve_key, "output": output_array_sha}),
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": input_sha,
        "solve_key": solve_key,
        "theta_raw_f8_sha256": theta_sha,
        "expected_output_bgr_sha256": output_array_sha,
        "height": int(output_bgr.shape[0]),
        "width": int(output_bgr.shape[1]),
        "unique_rgb": int(len(palette.palette_uint8)),
        "candidates": {
            "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE": {
                "path": str(bgr_path),
                "complete": True,
                "dtype": "uint8",
                "shape": list(output_bgr.shape),
                "file_bytes": bgr_path.stat().st_size,
                "file_sha256": file_sha,
                "array_bytes_sha256": array_sha,
                "decoder": "cv2",
                "built_at": now_iso(),
                "transform_seconds_build": float(time.perf_counter() - started),
            }
        },
    }
    atomic_json(sidecar_path, sidecar)
    sidecar_sha = sha256_file(sidecar_path)
    manifest = {
        "schema": "m1_native_a_fixed_cache_manifest_entry_v1",
        "canonical_uid": str(job["canonical_uid"]),
        "input_image_sha256": input_sha,
        "solve_key": solve_key,
        "theta_raw_f8_sha256": theta_sha,
        "expected_output_bgr_sha256": output_array_sha,
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete": True,
    }
    row["cache_solve_key"] = solve_key
    row["cache_expected_output_bgr_sha256"] = output_array_sha
    row["cache_payload_path"] = str(bgr_path)
    row["cache_payload_file_sha256"] = file_sha
    row["cache_payload_array_bytes_sha256"] = array_sha
    row["cache_payload_bytes"] = bgr_path.stat().st_size
    row["elapsed_seconds_total"] = float(time.perf_counter() - started)
    commit = {
        "schema": "m1_fixed_causal_control_cache_committed_key_v1",
        "status": "COMMITTED",
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "canonical_uid": str(job["canonical_uid"]),
        "manifest": manifest,
        "realization": row,
        "construction_denylist": _WORKER_GUARD.audit() if _WORKER_GUARD is not None else {},
        "created_at": now_iso(),
    }
    atomic_json(commit_path, commit)
    return {"canonical_uid": str(job["canonical_uid"]), "commit_path": str(commit_path)}


def _commit_valid(path: Path, variant: str, seed: int, uid: str, *, full_payload: bool) -> tuple[bool, dict[str, Any] | None, str]:
    try:
        commit = load_json(path)
        if not (
            commit.get("schema") == "m1_fixed_causal_control_cache_committed_key_v1"
            and commit.get("status") == "COMMITTED"
            and commit.get("variant") == variant
            and int(commit.get("seed", -1)) == int(seed)
            and commit.get("canonical_uid") == uid
        ):
            return False, None, "commit_identity"
        manifest = commit["manifest"]
        sidecar_path = Path(manifest["sidecar_path"]).resolve()
        if not sidecar_path.is_file() or sha256_file(sidecar_path) != manifest["sidecar_sha256"]:
            return False, None, "sidecar_sha"
        sidecar = load_json(sidecar_path)
        candidate = sidecar["candidates"]["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE"]
        payload = Path(candidate["path"]).resolve()
        if not payload.is_file() or payload.stat().st_size != int(candidate["file_bytes"]):
            return False, None, "payload_presence_size"
        if full_payload and sha256_file(payload) != candidate["file_sha256"]:
            return False, None, "payload_file_sha"
        if full_payload:
            mmap = np.load(payload, mmap_mode="r", allow_pickle=False)
            if str(mmap.dtype) != "uint8" or list(mmap.shape) != list(candidate["shape"]):
                return False, None, "payload_representation"
            if frozen_loader_array_bytes_sha(np.asarray(mmap)) != candidate["array_bytes_sha256"]:
                return False, None, "payload_array_sha"
        return True, commit, ""
    except BaseException as exc:
        return False, None, f"{type(exc).__name__}:{exc}"


def _quarantine_uncommitted(paths: Mapping[str, Path], job: Mapping[str, Any], reason: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = paths["cache"] / "INVALIDATED_RECOVERY_ATTEMPT" / f"{stamp}_{job['canonical_uid']}"
    moved = False
    for key in ("commit_path", "sidecar_path", "bgr_path"):
        source = Path(str(job[key]))
        if source.exists():
            root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(root / source.name))
            moved = True
    if moved:
        atomic_json(root / "INVALIDATION.json", {
            "schema": "m1_fixed_causal_control_invalidated_cache_attempt_v1",
            "status": "INVALIDATED_RECOVERY_ATTEMPT",
            "reason": reason,
            "canonical_uid": str(job["canonical_uid"]),
            "created_at": now_iso(),
        })


def _jobs_for_variant(variant: str, seed: int) -> list[dict[str, Any]]:
    whitelist = load_whitelist().set_index("uid", drop=False)
    context = load_policy_context(seed)
    paths = variant_paths(variant, seed)
    if variant == GENERIC:
        selection45 = load_active_selection(45)
        selection46 = load_active_selection(46)
        uids45 = set(selection45["canonical_uid"].astype(str))
        uids46 = set(selection46["canonical_uid"].astype(str))
        if uids45 != uids46:
            raise CausalPreparationStop("STOP_GENERIC_ELIGIBLE_UID_SET_CROSS_SEED")
        selected = select_generic_theta()["selected"]
        theta = [float(value) for value in selected["theta"]]
        dose = float(selected["selected_intended_geometric_dose"])
        records = [{"recipient_uid": uid} for uid in sorted(uids45)]
        physical_seed = 0
    else:
        build_shuffle_policy(seed)
        records = pd.read_csv(paths["assignment"]).to_dict("records")
        physical_seed = int(seed)
        theta = []
        dose = math.nan
    jobs: list[dict[str, Any]] = []
    for assignment in records:
        uid = str(assignment["recipient_uid"])
        if uid not in whitelist.index or uid not in context.image_index:
            raise CausalPreparationStop(f"STOP_CACHE_JOB_UID:{variant}:{seed}:{uid}")
        whitelist_row = whitelist.loc[uid]
        original = context.z_image[context.image_index[uid]].astype(np.float64)
        if variant == GENERIC:
            target = original.copy()
            donor_uid = ""
            donor_event_id = ""
            donor_delta_sha = ""
            donor_delta_csv_sha = ""
            intended_dose = dose
        else:
            donor_uid = str(assignment["donor_uid"])
            donor_event_id = str(whitelist.loc[donor_uid]["capture_event_id"])
            delta = np.asarray([float(assignment[f"donor_delta_{index}"]) for index in range(7)], dtype=np.float64)
            target = original + delta
            donor_delta_sha = str(assignment["donor_delta_sha"])
            donor_delta_csv_sha = str(assignment["donor_delta_csv_f8_sha256"])
            intended_dose = float(assignment["donor_selected_intended_geometric_dose"])
        input_sha = str(whitelist_row["image_sha256"])
        bgr_path = paths["working_cache"] / "bgr" / input_sha[:2] / f"{input_sha}.npy"
        sidecar_path = paths["working_cache"] / "meta" / input_sha[:2] / f"{input_sha}.json"
        commit_name = hashlib.sha256(uid.encode("utf-8")).hexdigest() + ".json"
        commit_path = paths["commit_root"] / commit_name[:2] / commit_name
        jobs.append({
            "variant": variant,
            "seed": physical_seed,
            "canonical_uid": uid,
            "capture_event_id": str(whitelist_row["capture_event_id"]),
            "image_path": str(Path(str(whitelist_row["image_path"])).resolve()),
            "input_image_sha256": input_sha,
            "original_std": original.tolist(),
            "target_std": target.tolist(),
            "theta": theta,
            "intended_dose": intended_dose,
            "donor_uid": donor_uid,
            "donor_event_id": donor_event_id,
            "donor_delta_sha": donor_delta_sha,
            "donor_delta_csv_f8_sha256": donor_delta_csv_sha,
            "bgr_path": str(bgr_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
            "commit_path": str(commit_path.resolve()),
        })
    if len(jobs) != EXPECTED_WITH_DELTA or len({row["canonical_uid"] for row in jobs}) != len(jobs):
        raise CausalPreparationStop(f"STOP_CACHE_JOB_COUNT:{variant}:{seed}:{len(jobs)}")
    return jobs


def migrate_cache_array_hash_metadata(variant: str, seed: int) -> dict[str, Any]:
    """Repair only cache hash metadata; never rewrite a scientific payload."""
    verify_science_contract()
    guard = ConstructionDenylist(f"CACHE_ARRAY_HASH_MIGRATION_{variant}_SEED{seed}").install()
    paths = variant_paths(variant, seed)
    jobs = _jobs_for_variant(variant, seed)
    physical_seed = int(jobs[0]["seed"])
    rows: list[dict[str, Any]] = []
    migrated = 0
    payload_file_sha_unchanged = True
    for index, job in enumerate(sorted(jobs, key=lambda item: str(item["canonical_uid"]))):
        uid = str(job["canonical_uid"])
        commit_path = Path(str(job["commit_path"])).resolve()
        if not commit_path.is_file():
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_COMMIT_MISSING:{variant}:{seed}:{uid}")
        old_commit_sha = sha256_file(commit_path)
        commit = load_json(commit_path)
        if not (
            commit.get("schema") == "m1_fixed_causal_control_cache_committed_key_v1"
            and commit.get("status") == "COMMITTED"
            and commit.get("variant") == variant
            and int(commit.get("seed", -1)) == physical_seed
            and commit.get("canonical_uid") == uid
        ):
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_COMMIT_IDENTITY:{variant}:{seed}:{uid}")
        manifest = commit["manifest"]
        realization_row = commit["realization"]
        sidecar_path = Path(str(manifest["sidecar_path"])).resolve()
        if sidecar_path != Path(str(job["sidecar_path"])).resolve() or not sidecar_path.is_file():
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_SIDECAR_PATH:{variant}:{seed}:{uid}")
        old_sidecar_sha = sha256_file(sidecar_path)
        sidecar = load_json(sidecar_path)
        candidate = sidecar.get("candidates", {}).get("EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE", {})
        payload = Path(str(candidate.get("path", ""))).resolve()
        if payload != Path(str(job["bgr_path"])).resolve() or not payload.is_file():
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_PAYLOAD_PATH:{variant}:{seed}:{uid}")
        file_sha_before = sha256_file(payload)
        if (
            file_sha_before != require_sha(candidate.get("file_sha256"), "migration_payload_file_sha")
            or payload.stat().st_size != int(candidate.get("file_bytes", -1))
        ):
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_PAYLOAD_FILE:{variant}:{seed}:{uid}")
        mmap = np.load(payload, mmap_mode="r", allow_pickle=False)
        if str(mmap.dtype) != "uint8" or list(mmap.shape) != list(candidate.get("shape", [])):
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_REPRESENTATION:{variant}:{seed}:{uid}")
        value = np.ascontiguousarray(mmap)
        del mmap
        raw_payload_sha = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        loader_array_sha = frozen_loader_array_bytes_sha(value)
        formal_bgr_sha = frozen_formal_bgr_sha(value)
        expected_formal = require_sha(sidecar.get("expected_output_bgr_sha256"), "migration_formal_bgr_sha")
        formal_fields = {
            expected_formal,
            require_sha(manifest.get("expected_output_bgr_sha256"), "migration_manifest_formal_bgr_sha"),
            require_sha(realization_row.get("cache_expected_output_bgr_sha256"), "migration_realization_formal_bgr_sha"),
        }
        if formal_fields != {formal_bgr_sha}:
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_FORMAL_BGR:{variant}:{seed}:{uid}")
        recorded_array_sha = require_sha(candidate.get("array_bytes_sha256"), "migration_recorded_array_sha")
        realization_array_sha = require_sha(
            realization_row.get("cache_payload_array_bytes_sha256"),
            "migration_realization_array_sha",
        )
        if recorded_array_sha not in {raw_payload_sha, loader_array_sha} or realization_array_sha not in {
            raw_payload_sha, loader_array_sha
        }:
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_UNKNOWN_HASH:{variant}:{seed}:{uid}")
        recorded_sidecar_sha = require_sha(manifest.get("sidecar_sha256"), "migration_recorded_sidecar_sha")
        if recorded_array_sha == raw_payload_sha and recorded_sidecar_sha != old_sidecar_sha:
            raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_SIDECAR_SHA:{variant}:{seed}:{uid}")
        changed = (
            recorded_array_sha != loader_array_sha
            or realization_array_sha != loader_array_sha
            or recorded_sidecar_sha != old_sidecar_sha
        )
        candidate["array_bytes_sha256"] = loader_array_sha
        atomic_json(sidecar_path, sidecar)
        new_sidecar_sha = sha256_file(sidecar_path)
        manifest["sidecar_sha256"] = new_sidecar_sha
        realization_row["cache_payload_array_bytes_sha256"] = loader_array_sha
        atomic_json(commit_path, commit)
        new_commit_sha = sha256_file(commit_path)
        file_sha_after = sha256_file(payload)
        payload_file_sha_unchanged &= file_sha_before == file_sha_after
        migrated += int(changed)
        rows.append({
            "schema": "m1_fixed_causal_control_cache_array_hash_metadata_migration_entry_v1",
            "variant": variant,
            "adaptation_seed": seed,
            "physical_cache_seed_scope": physical_seed,
            "canonical_uid": uid,
            "payload_path": str(payload),
            "payload_bytes": payload.stat().st_size,
            "payload_file_sha256_before": file_sha_before,
            "payload_file_sha256_after": file_sha_after,
            "raw_pixel_bytes_sha256": raw_payload_sha,
            "frozen_loader_array_bytes_sha256": loader_array_sha,
            "formal_bgr_sha256": formal_bgr_sha,
            "expected_formal_bgr_sha256": expected_formal,
            "old_recorded_array_bytes_sha256": recorded_array_sha,
            "old_sidecar_sha256": old_sidecar_sha,
            "new_sidecar_sha256": new_sidecar_sha,
            "old_commit_sha256": old_commit_sha,
            "new_commit_sha256": new_commit_sha,
            "metadata_changed": changed,
            "scientific_payload_changed": False,
        })
        if (index + 1) % 250 == 0:
            print(json.dumps({
                "stage": "CACHE_ARRAY_HASH_METADATA_MIGRATION",
                "variant": variant,
                "seed": seed,
                "processed": index + 1,
                "total": len(jobs),
            }, sort_keys=True), flush=True)
    csv_path = paths["seed_root"] / "CACHE_ARRAY_HASH_METADATA_MIGRATION.csv"
    audit_path = paths["seed_root"] / "CACHE_ARRAY_HASH_METADATA_MIGRATION_AUDIT.json"
    atomic_csv(csv_path, rows)
    checks = {
        "row_count_exact": len(rows) == EXPECTED_WITH_DELTA,
        "uid_unique_exact": len({row["canonical_uid"] for row in rows}) == EXPECTED_WITH_DELTA,
        "all_payload_file_sha_unchanged": payload_file_sha_unchanged and all(
            row["payload_file_sha256_before"] == row["payload_file_sha256_after"] for row in rows
        ),
        "all_formal_bgr_sha_unchanged_and_exact": all(
            row["formal_bgr_sha256"] == row["expected_formal_bgr_sha256"]
            for row in rows
        ),
        "all_scientific_payload_changed_false": all(row["scientific_payload_changed"] is False for row in rows),
        "target_dev_reads_zero": int(guard.audit().get("target_dev_reads", -1)) == 0,
        "target_test_reads_zero": int(guard.audit().get("target_test_reads", -1)) == 0,
        "target_adapt_gt_reads_zero": int(guard.audit().get("target_adapt_gt_reads", -1)) == 0,
    }
    audit = {
        "schema": "m1_fixed_causal_control_cache_array_hash_metadata_migration_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
        "physical_cache_seed_scope": physical_seed,
        "checks": checks,
        "entry_count": len(rows),
        "metadata_changed_count": migrated,
        "old_hash_definition": "SHA256(C_CONTIGUOUS_UINT8_PIXEL_BYTES_ONLY)",
        "new_hash_definition": "SHA256(ASCII_DTYPE || ASCII_PYTHON_SHAPE_TUPLE || C_CONTIGUOUS_PIXEL_BYTES)",
        "frozen_loader_hash_equivalence": True,
        "payload_files_rewritten": 0,
        "migration_csv": str(csv_path.resolve()),
        "migration_csv_sha256": sha256_file(csv_path),
        "construction_denylist": guard.audit(),
        "followup_build_cache_metadata_rematerialization_required": True,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise CausalPreparationStop(f"STOP_ARRAY_HASH_MIGRATION_AUDIT:{variant}:{seed}:{checks}")
    return audit


def _materialize_visit_audit(variant: str, seed: int, cache_uids: set[str]) -> dict[str, Any]:
    paths = variant_paths(variant, seed)
    source = seed_reference_paths(seed)["exposure"]
    rows: list[dict[str, Any]] = []
    active_count = 0
    identity_count = 0
    missing = 0
    unexpected = 0
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for expected_index, row in enumerate(reader):
            index = int(row["formal_visit_index"])
            active = strict_bool(row["fixed_active"], "fixed_active")
            identity = strict_bool(row["fixed_identity"], "fixed_identity")
            exists = strict_bool(row["fixed_delta_exists"], "fixed_delta_exists")
            uid = str(row["canonical_uid"])
            if (
                index != expected_index
                or int(row["adaptation_seed"]) != int(seed)
                or active == identity
                or (active and not exists)
            ):
                raise CausalPreparationStop(f"STOP_VISIT_IDENTITY:{variant}:{seed}:{expected_index}")
            hit = active and uid in cache_uids
            missing += int(active and not hit)
            unexpected += int(identity and uid in cache_uids and False)
            active_count += int(active)
            identity_count += int(identity)
            rows.append({
                "schema": "m1_fixed_causal_control_visit_audit_v1",
                "variant": variant,
                "adaptation_seed": seed,
                "formal_visit_index": index,
                "epoch": int(row["epoch"]),
                "target_visit_ordinal": int(row["target_visit_ordinal"]),
                "schedule_uid": str(row["schedule_uid"]),
                "canonical_uid": uid,
                "fixed_delta_exists": exists,
                "fixed_active": active,
                "fixed_identity": identity,
                "cache_lookup_expected": active,
                "cache_entry_present": uid in cache_uids,
                "cache_hit": hit,
                "cache_miss": active and not hit,
                "identity_cache_lookup": False,
                "runtime_fallback_allowed": False,
            })
    if len(rows) != TOTAL_VISITS or active_count + identity_count != TOTAL_VISITS or missing or unexpected:
        raise CausalPreparationStop(
            f"STOP_VISIT_COUNTS:{variant}:{seed}:{len(rows)}:{active_count}:{identity_count}:{missing}:{unexpected}"
        )
    atomic_csv(paths["visit_audit"], rows)
    return {
        "visits": len(rows),
        "active": active_count,
        "identity": identity_count,
        "missing_lookups": missing,
        "unexpected_lookups": unexpected,
        "visit_audit_path": str(paths["visit_audit"].resolve()),
        "visit_audit_sha256": sha256_file(paths["visit_audit"]),
    }


def _write_shuffled_fairness(seed: int, realization_rows: Sequence[Mapping[str, Any]]) -> None:
    paths = variant_paths(SHUFFLED, seed)
    native = pd.read_csv(seed_reference_paths(seed)["theta"])
    native = native.loc[
        native["fixed_delta_exists"].map(lambda value: strict_bool(value, "fixed_delta_exists"))
    ].set_index("canonical_uid", drop=False)
    required = (
        "intended_dose", "target_descriptor_dose", "actual_effective_dose",
        "realization_residual", "normalized_realization_residual",
        "actual_support_deviation", "actual_endpoint_in_tube",
        "tone_preclip_pixel_fraction", "lab_to_srgb_preclip_pixel_fraction",
        "solver_success", "solver_nfev", "elapsed_seconds",
    )
    output: list[dict[str, Any]] = []
    for shuffled in realization_rows:
        uid = str(shuffled["recipient_uid"])
        old = native.loc[uid]
        row: dict[str, Any] = {
            "schema": "m1_shuffled_vs_native_realization_fairness_v1",
            "adaptation_seed": seed,
            "recipient_uid": uid,
            "shuffled_donor_uid": str(shuffled["donor_uid"]),
            "native_selected_delta_sha": str(old["selected_delta_sha"]),
            "shuffled_donor_delta_sha": str(shuffled["donor_delta_sha"]),
            "exact_delta_multiset_is_global_not_per_recipient": True,
        }
        for field in required:
            row[f"native_{field}"] = old[field]
            row[f"shuffled_{field}"] = shuffled[field]
        output.append(row)
    atomic_csv(paths["fairness"], output)


def build_cache(variant: str, seed: int, workers: int = 8) -> dict[str, Any]:
    verify_science_contract()
    if variant == GENERIC:
        select_generic_theta()
    else:
        build_shuffle_policy(seed)
    guard = ConstructionDenylist(f"CACHE_BUILD_{variant}_SEED{seed}").install()
    paths = variant_paths(variant, seed)
    jobs = _jobs_for_variant(variant, seed)
    physical_seed = int(jobs[0]["seed"])
    committed: dict[str, dict[str, Any]] = {}
    missing_jobs: list[dict[str, Any]] = []
    for job in jobs:
        uid = str(job["canonical_uid"])
        commit_path = Path(str(job["commit_path"]))
        if commit_path.is_file():
            valid, commit, reason = _commit_valid(commit_path, variant, physical_seed, uid, full_payload=False)
            if valid and commit is not None:
                committed[uid] = commit
                continue
            _quarantine_uncommitted(paths, job, f"INVALID_COMMIT:{reason}")
        else:
            orphaned = any(Path(str(job[key])).exists() for key in ("bgr_path", "sidecar_path"))
            if orphaned:
                _quarantine_uncommitted(paths, job, "UNCOMMITTED_ATOMIC_ORPHAN")
        missing_jobs.append(job)
    started = time.perf_counter()
    completed_new = 0
    if missing_jobs:
        init_seed = 45 if variant == GENERIC else seed
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_cache_worker_initializer,
            initargs=(int(init_seed),),
        ) as pool:
            futures = {pool.submit(_cache_job, job): str(job["canonical_uid"]) for job in missing_jobs}
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    raise CausalPreparationStop(
                        f"STOP_CACHE_WORKER:{variant}:{seed}:{uid}:{type(exc).__name__}:{exc}"
                    ) from exc
                valid, commit, reason = _commit_valid(
                    Path(result["commit_path"]), variant, physical_seed, uid, full_payload=False
                )
                if not valid or commit is None:
                    raise CausalPreparationStop(f"STOP_CACHE_POSTCOMMIT:{uid}:{reason}")
                committed[uid] = commit
                completed_new += 1
                if completed_new % 25 == 0 or completed_new == len(missing_jobs):
                    print(json.dumps({
                        "stage": "CACHE_BUILD",
                        "variant": variant,
                        "seed": seed,
                        "completed_new": completed_new,
                        "missing_total": len(missing_jobs),
                        "committed_total": len(committed),
                        "elapsed_seconds": time.perf_counter() - started,
                    }, sort_keys=True), flush=True)
    if len(committed) != EXPECTED_WITH_DELTA or set(committed) != {str(row["canonical_uid"]) for row in jobs}:
        raise CausalPreparationStop(f"STOP_CACHE_COMMITTED_SET:{variant}:{seed}:{len(committed)}")
    ordered = [committed[uid] for uid in sorted(committed)]
    manifest_rows = [row["manifest"] for row in ordered]
    realization_rows = [row["realization"] for row in ordered]
    atomic_csv(paths["manifest"], manifest_rows)
    atomic_csv(paths["realization"], realization_rows)
    if variant == GENERIC:
        atomic_csv(paths["family"] / "GENERIC_FIXED_CACHE_MANIFEST.csv", manifest_rows)
    else:
        _write_shuffled_fairness(seed, realization_rows)
    visit = _materialize_visit_audit(variant, seed, set(committed))
    build = {
        "schema": "m1_fixed_causal_control_cache_build_v1",
        "status": "PASS",
        "variant": variant,
        "formal_name": GENERIC_FORMAL if variant == GENERIC else SHUFFLED_FORMAL,
        "adaptation_seed": seed,
        "physical_cache_seed_scope": physical_seed,
        "expected_active_entries": EXPECTED_WITH_DELTA,
        "candidate_counts": {
            "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE": {
                "built": EXPECTED_WITH_DELTA,
                "complete": EXPECTED_WITH_DELTA,
                "resumed": EXPECTED_WITH_DELTA - completed_new,
                "newly_built": completed_new,
            }
        },
        "manifest_csv": str(paths["manifest"].resolve()),
        "manifest_csv_sha256": sha256_file(paths["manifest"]),
        "realization_csv": str(paths["realization"].resolve()),
        "realization_csv_sha256": sha256_file(paths["realization"]),
        "actual_npy_payload_bytes": sum(int(row["realization"]["cache_payload_bytes"]) for row in ordered),
        "committed_key_count": len(committed),
        "committed_key_set_sha256": hashlib.sha256("\n".join(sorted(committed)).encode("utf-8")).hexdigest(),
        "online_powell_calls": 0 if variant == GENERIC else EXPECTED_WITH_DELTA,
        "runtime_online_powell_calls": 0,
        "runtime_fallback": False,
        "visit_audit": visit,
        "construction_denylist": guard.audit(),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "wall_seconds_this_invocation": float(time.perf_counter() - started),
        "created_at": now_iso(),
    }
    atomic_json(paths["cache_build"], build)
    integrity = verify_cache(variant, seed, full_payload=True)
    gate = finalize_policy_cache_gate(variant, seed, visit, integrity)
    build["cache_integrity_audit_sha256"] = sha256_file(paths["cache_audit"])
    build["policy_cache_gate_sha256"] = sha256_file(paths["policy_gate"])
    atomic_json(paths["cache_build"], build)
    gate["cache_build_sha256"] = sha256_file(paths["cache_build"])
    atomic_json(paths["policy_gate"], gate)
    return build


def verify_cache(variant: str, seed: int, full_payload: bool = True) -> dict[str, Any]:
    paths = variant_paths(variant, seed)
    if not paths["manifest"].is_file():
        raise CausalPreparationStop(f"STOP_CACHE_MANIFEST_MISSING:{variant}:{seed}")
    rows = pd.read_csv(paths["manifest"], dtype=str, keep_default_na=False).to_dict("records")
    expected = set(load_active_selection(seed)["canonical_uid"].astype(str))
    observed = [str(row["canonical_uid"]) for row in rows]
    duplicate = len(observed) - len(set(observed))
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    failures: list[dict[str, str]] = []
    composite: list[str] = []
    total_bytes = 0
    for index, row in enumerate(sorted(rows, key=lambda item: item["canonical_uid"])):
        uid = str(row["canonical_uid"])
        try:
            sidecar_path = Path(row["sidecar_path"]).resolve()
            sidecar_sha = require_sha(row["sidecar_sha256"], "manifest_sidecar_sha")
            if sha256_file(sidecar_path) != sidecar_sha:
                raise CausalPreparationStop("sidecar_sha")
            sidecar = load_json(sidecar_path)
            candidate = sidecar["candidates"]["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE"]
            payload = Path(candidate["path"]).resolve()
            if not payload.is_file() or payload.stat().st_size != int(candidate["file_bytes"]):
                raise CausalPreparationStop("payload_presence_size")
            total_bytes += payload.stat().st_size
            if full_payload:
                file_sha = sha256_file(payload)
                if file_sha != candidate["file_sha256"]:
                    raise CausalPreparationStop("payload_file_sha")
                mmap = np.load(payload, mmap_mode="r", allow_pickle=False)
                if str(mmap.dtype) != "uint8" or list(mmap.shape) != list(candidate["shape"]):
                    raise CausalPreparationStop("payload_representation")
                array_sha = frozen_loader_array_bytes_sha(np.asarray(mmap))
                if array_sha != candidate["array_bytes_sha256"]:
                    raise CausalPreparationStop("payload_array_sha")
            composite.append("|".join((
                uid,
                sidecar_sha,
                str(candidate["file_sha256"]),
                str(candidate["array_bytes_sha256"]),
                str(row["expected_output_bgr_sha256"]),
            )))
            if (index + 1) % 250 == 0:
                print(json.dumps({"stage": "CACHE_VERIFY", "variant": variant, "seed": seed, "verified": index + 1, "total": len(rows)}, sort_keys=True), flush=True)
        except BaseException as exc:
            failures.append({"canonical_uid": uid, "reason": f"{type(exc).__name__}:{exc}"})
    checks = {
        "entry_count_exact": len(rows) == EXPECTED_WITH_DELTA,
        "expected_uid_set_exact": not missing and not unexpected,
        "duplicate_key_count_zero": duplicate == 0,
        "payload_integrity": not failures,
        "full_payload_hash_mode": bool(full_payload),
    }
    audit = {
        "schema": "m1_fixed_causal_control_cache_integrity_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
        "entry_count": len(rows),
        "expected_entry_count": EXPECTED_WITH_DELTA,
        "duplicate_key_count": duplicate,
        "missing_key_count": len(missing),
        "missing_keys": missing,
        "unexpected_key_count": len(unexpected),
        "unexpected_keys": unexpected,
        "corrupt_key_count": len(failures),
        "corrupt_keys": failures,
        "payload_total_bytes": total_bytes,
        "payload_composite_sha256": hashlib.sha256("\n".join(composite).encode("ascii")).hexdigest(),
        "manifest_path": str(paths["manifest"].resolve()),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "checks": checks,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["cache_audit"], audit)
    if audit["status"] != "PASS":
        raise CausalPreparationStop(f"STOP_CACHE_INTEGRITY:{variant}:{seed}:{checks}")
    return audit


def finalize_policy_cache_gate(
    variant: str,
    seed: int,
    visit: Mapping[str, Any] | None = None,
    integrity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    paths = variant_paths(variant, seed)
    if integrity is None:
        integrity = verify_cache(variant, seed, full_payload=True)
    if visit is None:
        manifest = pd.read_csv(paths["manifest"], dtype=str, keep_default_na=False)
        visit = _materialize_visit_audit(variant, seed, set(manifest["canonical_uid"].astype(str)))
    checks = {
        "science_contract_frozen": sha256_file(SCIENCE_CONTRACT) == SCIENCE_CONTRACT_SHA256,
        "policy_contract_frozen": paths["policy_contract"].is_file(),
        "cache_integrity_pass": integrity.get("status") == "PASS",
        "active_mask_exact": int(visit["active"]) + int(visit["identity"]) == TOTAL_VISITS,
        "missing_lookups_zero": int(visit["missing_lookups"]) == 0,
        "unexpected_lookups_zero": int(visit["unexpected_lookups"]) == 0,
        "runtime_fallback_forbidden": True,
        "target_dev_reads_zero": True,
        "target_test_reads_zero": True,
        "target_adapt_gt_reads_zero": True,
    }
    gate = {
        "schema": "m1_fixed_causal_control_policy_cache_pretrain_gate_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "variant": variant,
        "adaptation_seed": seed,
        "training_authorized_by_identity_contract_after_gate": all(checks.values()),
        "counts": {
            "visits": int(visit["visits"]),
            "active": int(visit["active"]),
            "identity": int(visit["identity"]),
            "with_delta": EXPECTED_WITH_DELTA,
            "without_delta": EXPECTED_WITHOUT_DELTA,
        },
        "policy_contract_path": str(paths["policy_contract"].resolve()),
        "policy_contract_sha256": sha256_file(paths["policy_contract"]),
        "cache_manifest_path": str(paths["manifest"].resolve()),
        "cache_manifest_sha256": sha256_file(paths["manifest"]),
        "cache_payload_composite_sha256": integrity["payload_composite_sha256"],
        "cache_integrity_audit_path": str(paths["cache_audit"].resolve()),
        "cache_integrity_audit_sha256": sha256_file(paths["cache_audit"]),
        "visit_audit_path": str(paths["visit_audit"].resolve()),
        "visit_audit_sha256": sha256_file(paths["visit_audit"]),
        "checks": checks,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["policy_gate"], gate)
    if gate["status"] != "PASS":
        raise CausalPreparationStop(f"STOP_POLICY_CACHE_GATE:{variant}:{seed}:{checks}")
    return gate


def verify_input_identities() -> dict[str, Any]:
    verify_science_contract()
    authorities = {
        "fresh_zip": (FRESH_ZIP, FRESH_ZIP_SHA256),
        "fresh_identity": (FRESH_IDENTITY, FRESH_IDENTITY_SHA256),
        "parent": (PARENT, PARENT_SHA256),
        "official_evaluator": (OFFICIAL_EVALUATOR, OFFICIAL_EVALUATOR_SHA256),
    }
    observed: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for role, (path, expected) in authorities.items():
        digest = sha256_file(path)
        checks[role] = digest == expected
        observed[role] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "expected_sha256": expected,
        }
    schedules: dict[str, Any] = {}
    for seed in SEEDS:
        paths = seed_reference_paths(seed)
        rows = _schedule_uids(paths["target_schedule"])
        exposure = pd.read_csv(paths["exposure"], usecols=["fixed_active", "fixed_identity"])
        active = int(exposure["fixed_active"].map(lambda value: strict_bool(value, "fixed_active")).sum())
        identity = int(exposure["fixed_identity"].map(lambda value: strict_bool(value, "fixed_identity")).sum())
        checks[f"seed{seed}_schedule_visits"] = len(rows) == TOTAL_VISITS
        checks[f"seed{seed}_active_mask"] = len(exposure) == TOTAL_VISITS and active + identity == TOTAL_VISITS
        schedules[str(seed)] = {
            role: {
                "path": str(paths[role].resolve()),
                "bytes": paths[role].stat().st_size,
                "sha256": sha256_file(paths[role]),
            }
            for role in (
                "source_schedule", "source_augmentation", "target_schedule",
                "target_manifest", "selection", "theta", "exposure", "fixed_contract",
            )
        }
        schedules[str(seed)]["recomputed_counts"] = {
            "visits": len(exposure), "active": active, "identity": identity,
        }
    audit = {
        "schema": "m1_fixed_causal_controls_input_identity_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "authorities": observed,
        "per_seed_frozen_inputs": schedules,
        "checks": checks,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(RESULT_ROOT / "INPUT_IDENTITY_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise CausalPreparationStop(f"STOP_INPUT_IDENTITY:{checks}")
    return audit


def preparation_summary() -> dict[str, Any]:
    inputs = verify_input_identities()
    generic = select_generic_theta()
    shuffled = {str(seed): build_shuffle_policy(seed) for seed in SEEDS}
    gates: dict[str, Any] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            paths = variant_paths(variant, seed)
            key = f"{variant}_SEED{seed}"
            gates[key] = load_json(paths["policy_gate"]) if paths["policy_gate"].is_file() else {"status": "NOT_BUILT"}
    summary = {
        "schema": "m1_fixed_causal_controls_preparation_summary_v1",
        "status": "PASS" if all(row.get("status") == "PASS" for row in gates.values()) else "POLICY_PASS_CACHE_PENDING",
        "input_identity": inputs,
        "generic_selection": generic,
        "shuffle_graphs": shuffled,
        "policy_cache_gates": gates,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_gt_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(RESULT_ROOT / "PREPARATION_SUMMARY.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("verify-inputs")
    sub.add_parser("select-generic")
    shuffle = sub.add_parser("build-shuffle-policy")
    shuffle.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cache = sub.add_parser("build-cache")
    cache.add_argument("--variant", choices=VARIANTS, required=True)
    cache.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cache.add_argument("--workers", type=int, default=8)
    verify = sub.add_parser("verify-cache")
    verify.add_argument("--variant", choices=VARIANTS, required=True)
    verify.add_argument("--seed", type=int, choices=SEEDS, required=True)
    migrate = sub.add_parser("migrate-array-hash")
    migrate.add_argument("--variant", choices=VARIANTS, required=True)
    migrate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--variant", choices=VARIANTS, required=True)
    gate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "verify-inputs":
            result = verify_input_identities()
        elif args.action == "select-generic":
            result = select_generic_theta()
        elif args.action == "build-shuffle-policy":
            result = build_shuffle_policy(args.seed)
        elif args.action == "build-cache":
            result = build_cache(args.variant, args.seed, args.workers)
        elif args.action == "verify-cache":
            result = verify_cache(args.variant, args.seed, full_payload=True)
        elif args.action == "migrate-array-hash":
            result = migrate_cache_array_hash_metadata(args.variant, args.seed)
        elif args.action == "gate":
            result = finalize_policy_cache_gate(args.variant, args.seed)
        elif args.action == "summary":
            result = preparation_summary()
        else:
            raise CausalPreparationStop(f"STOP_ACTION:{args.action}")
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
            "schema": "m1_fixed_causal_controls_preparation_failure_v1",
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
        atomic_json(RESULT_ROOT / f"PREPARATION_FAILURE_{args.action}_{getattr(args, 'seed', 'NA')}.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
