from __future__ import annotations

"""Fixed-A exact cache prototype, equivalence audit, and resource gate.

This is an execution/audit-only entry point for the frozen Fixed-A pretrain
gate.  It does not construct the Fixed-A selector, solve theta, alter the
formal training runner, or provide a formal-training entry point.

The three execution candidates are:

* ``THETA_ONLY_LOOKUP``: the protected theta plus frozen exact transform;
* ``EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE``: one exact BGR array/image;
* ``EXACT_FIXED_OUTPUT_PALETTE_REMAP``: one sorted source/deployment palette
  pair/image, remapped in bounded chunks before the unchanged letterbox.

Cache payloads live under ``working_cache`` and are intentionally excluded
from the eventual CORE ZIP.  Per-image sidecars and root manifests make the
3226-entry materialization atomic and resumable without weakening exact-key
or payload validation.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import psutil


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal").resolve()
SCRIPT_DIR = ROOT / "project" / "scripts"
RESULT_DIR = (
    ROOT
    / "project"
    / "results"
    / "M1_NATIVE_A_FIXED_COMPLEXITY_UTILITY_ABLATION_V1_PRETRAIN_GATE"
)
FIXED_CONTRACT = RESULT_DIR / "FIXED_A_POLICY_CONTRACT.json"
FIXED_CONTRACT_SHA256 = (
    "f26b856ab54cd3a55ff720a8215a3300fee16f82388fa045a1d05e840bb27d0c"
)

Q2_AUDIT = (
    ROOT
    / "project"
    / "audits"
    / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
)
Q2_WHITELIST = Q2_AUDIT / "TARGET_ADAPT_STYLE_AUDIT_WHITELIST.csv"
Q2_WHITELIST_SHA256 = (
    "24426506f2fba321ca1c98f593d459dca3e9e76d030c0303299eacd1ad76b29f"
)
FROZEN_TARGET_ADAPT_IMAGE_ROOT = (
    ROOT / "datasets" / "dataset_final_v2_1" / "target" / "adapt" / "images"
).resolve()
NATIVE_SCHEDULE = (
    ROOT
    / "project"
    / "audits"
    / "M1_ROUTE_A_SEMANTIC_RESOURCE_GATE_MEASUREMENT_V1_20260810_001748"
    / "raw"
    / "visit_policy_schedule.parquet"
)
NATIVE_SCHEDULE_SHA256 = (
    "1c5c10f3bc349e7aa62f1b27398d03c36048dcf61c155e62b6f57a0cd91457ee"
)
SEMANTIC_RESOURCE_CONTRACT = (
    ROOT / "project" / "contracts" / "m1_routea_semantic_resource_gate_measurement_v1.json"
)
SEMANTIC_RESOURCE_CONTRACT_SHA256 = (
    "d78175cd4d0ec78d803cf7003f36f75560d1afe93daf319f2cc3c12f3ada7b43"
)
EXACT_TRANSFORM_CORE = SCRIPT_DIR / "m1_routea_resource_equivalence_execution_core_v1.py"
EXACT_TRANSFORM_CORE_SHA256 = (
    "72ebbf342d50decc7f0a1053adf59c40a614f9b758ab64667aea9a146586d20b"
)
FROZEN_REALIZATION_CORE_SHA256 = (
    "45b62bf0a192e2d527f3171661d36b19d2dd7da21c63afb93140015251b449c1"
)
OFFLINE_CACHE_ENGINE = SCRIPT_DIR / "m1_routea_offline_theta_cache_engine_v1.py"
OFFLINE_CACHE_ENGINE_SHA256 = (
    "eede42866164fbbac4589c036277f4f2137a8e13d4435dfb85e8b433ea32c09f"
)
RESOURCE_WORKER = SCRIPT_DIR / "m1_routea_semantic_resource_worker_v1.py"
RESOURCE_WORKER_SHA256 = (
    "f1785c0ab81624694182f39ca5153e0b8666284e3811f0f8409bd873f125bd2c"
)
FORMAL_RESOURCE_RUNNER = SCRIPT_DIR / "m1_routea_resource_equivalence_training_window_runner_v1.py"
FORMAL_RESOURCE_RUNNER_SHA256 = (
    "5a1dabde99289a2310f5fcd0eb33008235c3b1f9b5c6a518914ad2f996ac9873"
)
RESAMPLED_RESOURCE_ROOT = (
    ROOT
    / "project"
    / "results"
    / "M1_ROUTE_A_RESOURCE_EQUIVALENCE_ENGINEERING_V1"
)
RESAMPLED_RESOURCE_COMPARISON = RESAMPLED_RESOURCE_ROOT / "training_window_before_after.csv"
RESAMPLED_RESOURCE_COMPARISON_SHA256 = (
    "32b465764dab48ec898c87c65bebe9e9f5a7da2499560604d94a2c7d6d06cbba"
)
RESAMPLED_RESOURCE_WINDOWS = (
    RESAMPLED_RESOURCE_ROOT / "raw" / "resource_per_optimizer_window_NATIVE_A.csv"
)
RESAMPLED_RESOURCE_WINDOWS_SHA256 = (
    "07cd7f3398f2999319689002ba70f5660b1c0010d0f6725d555359d423b5cf87"
)
RESAMPLED_RESOURCE_WORKER_AUDIT = (
    RESAMPLED_RESOURCE_ROOT / "raw" / "resource_worker_NATIVE_A.json"
)
RESAMPLED_RESOURCE_WORKER_AUDIT_SHA256 = (
    "6fd690eed195553adba685ea97b3c9c779b60d3750b95249a62bc4d619a8711c"
)
RESAMPLED_RESOURCE_RUNNER_AUDIT = (
    RESAMPLED_RESOURCE_ROOT / "raw" / "training_window_runner_audit.json"
)
RESAMPLED_RESOURCE_RUNNER_AUDIT_SHA256 = (
    "d771cc4b729fc2eea29f12dea326935f5db88450d8e890fc1d9dff93aea35cbf"
)
FIXED_POLICY_REALIZATION_RUNNER = (
    SCRIPT_DIR / "m1_native_a_fixed_policy_realization_v1.py"
)
FIXED_POLICY_REALIZATION_RUNNER_SHA256 = (
    "5219a0a0bb579f801ac5e58889701eb25f164738fbab2c1dd82d744210791279"
)

GOLDEN_CASES = (
    ROOT
    / "project"
    / "results"
    / "M1_ROUTE_A_RESOURCE_EQUIVALENCE_ENGINEERING_V1"
    / "golden_cases.csv"
)
GOLDEN_CASES_SHA256 = (
    "b6412030007820e360d17fc93c986231bad49438248289f6a1f3207debf78bfd"
)
GOLDEN_EQUIVALENCE = GOLDEN_CASES.parent / "golden_equivalence.csv"
GOLDEN_EQUIVALENCE_SHA256 = (
    "71f75b7e7fb6474c81bf8ecd34799c58d708ef0ef8472edc3d7998637cc621a6"
)

SELECTION_NAME = "fixed_a_image_delta_selection.csv"
EXPOSURE_NAME = "fixed_a_visit_exposure_audit.csv"
THETA_NAME = "fixed_a_theta_realization.csv"
EXPOSURE_AUDIT_NAME = "fixed_a_exposure_equivalence.json"
SELECTION_SHA256 = (
    "d0dfa20fd65b56e4d9ca0e75162fdf8e821df0c1cdcd51df365d5600fcb32790"
)
EXPOSURE_SHA256 = (
    "1be0e1b15051d9d68a115c7a6f223ded55bec9c6b32edfea253d29edde51b231"
)
SELECTION_SCHEMA = "m1_native_a_fixed_delta_selection_v1"
EXPOSURE_SCHEMA = "m1_native_a_fixed_visit_exposure_audit_v1"
THETA_SCHEMA = "m1_native_a_fixed_theta_realization_v1"
SOLVED_STATUS = "SOLVED_RETAINED_NO_RESELECTION"
IDENTITY_STATUS = "IDENTITY_NO_VALID_DELTA"

THETA_ONLY = "THETA_ONLY_LOOKUP"
BGR_CACHE = "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE"
PALETTE_CACHE = "EXACT_FIXED_OUTPUT_PALETTE_REMAP"
CANDIDATES = (THETA_ONLY, BGR_CACHE, PALETTE_CACHE)

EXPECTED_IMAGE_COUNT = 3272
EXPECTED_FIXED_SOLVES = 3226
EXPECTED_IDENTITY_IMAGES = 46
EXPECTED_VISITS = 128064
EXPECTED_ACTIVE_VISITS = 62679
EXPECTED_RESOURCE_VISITS = 80
EXPECTED_RESOURCE_TARGET_BATCH = 8
EXPECTED_FIXED_REPLAY_VISITS = 240
HIGH_PALETTE_COUNT = 16
DEFAULT_REPETITIONS = 3
PALETTE_REMAP_CHUNK_PIXELS = 262_144
COLD_OBSERVATION = "COLD_OBSERVATION_REPETITION_1"
WARM_OBSERVATION = "WARM_OBSERVATION_REPETITIONS_2_PLUS"
COLD_WARM_OS_CACHE_CAVEAT = (
    "deterministic sequential observation only: repetition 1 is called cold and "
    "repetitions 2..N warm; the runner does not evict or flush the OS page cache, "
    "so cold is not a physical cold-disk guarantee"
)

THETA_COLUMNS = ("theta_delta_t", "theta_c_t", "theta_s_c")
DELTA_COLUMNS = tuple(f"selected_delta_{index}" for index in range(7))
THETA_HASH_PREFIX = b"M1_ROUTEA_THETA_RAW_FLOAT64_LE_V1\0"
ARRAY_HASH_PREFIX = b"M1_ROUTEA_REALIZED_BGR_UINT8_HWC_V1\0"
HEX64 = set("0123456789abcdef")

SELECTION_REQUIRED = (
    "schema",
    "canonical_uid",
    "fixed_delta_exists",
    "selected_delta_sha",
    "selected_formal_visit_index",
    "selected_epoch",
    "selected_target_visit_ordinal",
    "selected_schedule_uid",
    "selected_intended_geometric_dose",
    *DELTA_COLUMNS,
    *(f"selected_target_std_{index}" for index in range(7)),
)
EXPOSURE_REQUIRED = (
    "schema",
    "formal_visit_index",
    "canonical_uid",
    "resample_active",
    "fixed_active",
    "fixed_identity",
    "fixed_selected_delta_sha",
    "fixed_delta_sha_used",
    "active_match",
    "identity_match",
)
THETA_REQUIRED_BASE = (
    "schema",
    "canonical_uid",
    "fixed_delta_exists",
    "realization_status",
    "input_image_sha256",
    "target_descriptor_sha256",
    "solve_key",
    "cache_namespace",
    "cache_key_sha256",
    *THETA_COLUMNS,
    "theta_raw_f8_sha256",
    "output_array_sha256",
    "actual_effective_dose",
    "tone_preclip_pixel_fraction",
    "lab_to_srgb_preclip_pixel_fraction",
    "science_cache_entry_sha256",
)


class FixedACacheStop(RuntimeError):
    """Fail-closed Fixed-A cache, equivalence, or resource-gate stop."""


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, Mapping):
            return {str(key): convert(inner) for key, inner in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(inner) for inner in item]
        return item

    return json.dumps(
        convert(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in HEX64 for character in text)


def _require_sha256(value: Any, role: str) -> str:
    text = str(value).strip().lower()
    if not _is_sha256(text):
        raise FixedACacheStop(f"STOP_FIXED_A_INVALID_SHA256:{role}:{text}")
    return text


def _safe_result_dir(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(RESULT_DIR):
        raise FixedACacheStop(f"STOP_FIXED_A_RESULT_OUTSIDE_FIXED_GATE:{path}")
    return path


_FROZEN_ADAPT_IMAGE_PATHS: frozenset[str] | None = None


def _frozen_adapt_image_paths() -> frozenset[str]:
    global _FROZEN_ADAPT_IMAGE_PATHS
    if _FROZEN_ADAPT_IMAGE_PATHS is None:
        if sha256_file(Q2_WHITELIST) != Q2_WHITELIST_SHA256:
            raise FixedACacheStop("STOP_FIXED_A_Q2_WHITELIST_SHA_MISMATCH")
        frame = pd.read_csv(Q2_WHITELIST, dtype=str)
        _require_columns(frame, ("uid", "image_path", "image_sha256"), "whitelist")
        if len(frame) != EXPECTED_IMAGE_COUNT or frame["uid"].duplicated().any():
            raise FixedACacheStop("STOP_FIXED_A_Q2_WHITELIST_SCHEMA_MISMATCH")
        whitelist_paths = [
            str(Path(value).resolve()).casefold() for value in frame["image_path"]
        ]
        frozen_dataset_paths = [
            str((FROZEN_TARGET_ADAPT_IMAGE_ROOT / Path(value).name).resolve()).casefold()
            for value in frame["image_path"]
        ]
        if (
            len(set(whitelist_paths)) != EXPECTED_IMAGE_COUNT
            or len(set(frozen_dataset_paths)) != EXPECTED_IMAGE_COUNT
        ):
            raise FixedACacheStop("STOP_FIXED_A_Q2_WHITELIST_PATH_DUPLICATE")
        _FROZEN_ADAPT_IMAGE_PATHS = frozenset(
            [*whitelist_paths, *frozen_dataset_paths]
        )
    return _FROZEN_ADAPT_IMAGE_PATHS


def _guard_adapt_image(value: str | Path) -> Path:
    path = Path(value).resolve()
    normalized = str(path).replace("\\", "/").casefold()
    if any(
        marker in normalized
        for marker in ("/target/dev/", "/target/test/", "/target_dev/", "/target_test/")
    ):
        raise FixedACacheStop(f"STOP_FIXED_A_FORBIDDEN_TARGET_SPLIT:{path}")
    if str(path).casefold() not in _frozen_adapt_image_paths():
        raise FixedACacheStop(f"STOP_FIXED_A_NON_ADAPT_IMAGE:{path}")
    if not path.is_file():
        raise FixedACacheStop(f"STOP_FIXED_A_IMAGE_MISSING:{path}")
    return path


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(value) + b"\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _strict_bool(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise FixedACacheStop(f"STOP_FIXED_A_BOOLEAN_REPRESENTATION:{role}:{value!r}")


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return str(value)


def _require_columns(frame: pd.DataFrame, required: Sequence[str], role: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise FixedACacheStop(f"STOP_FIXED_A_COLUMNS_MISSING:{role}:" + ",".join(missing))


def _resolve_column(frame: pd.DataFrame, choices: Sequence[str], role: str) -> str:
    present = [column for column in choices if column in frame.columns]
    if len(present) != 1:
        raise FixedACacheStop(
            f"STOP_FIXED_A_COLUMN_RESOLUTION:{role}:choices={','.join(choices)}:present={','.join(present)}"
        )
    return present[0]


def _verify_sha_locks() -> dict[str, dict[str, Any]]:
    locks = {
        "fixed_contract": (FIXED_CONTRACT, FIXED_CONTRACT_SHA256),
        "q2_whitelist": (Q2_WHITELIST, Q2_WHITELIST_SHA256),
        "native_schedule": (NATIVE_SCHEDULE, NATIVE_SCHEDULE_SHA256),
        "semantic_resource_contract": (
            SEMANTIC_RESOURCE_CONTRACT,
            SEMANTIC_RESOURCE_CONTRACT_SHA256,
        ),
        "exact_transform_core": (EXACT_TRANSFORM_CORE, EXACT_TRANSFORM_CORE_SHA256),
        "offline_cache_engine": (OFFLINE_CACHE_ENGINE, OFFLINE_CACHE_ENGINE_SHA256),
        "resource_worker": (RESOURCE_WORKER, RESOURCE_WORKER_SHA256),
        "formal_resource_runner": (FORMAL_RESOURCE_RUNNER, FORMAL_RESOURCE_RUNNER_SHA256),
        "resampled_resource_comparison": (
            RESAMPLED_RESOURCE_COMPARISON,
            RESAMPLED_RESOURCE_COMPARISON_SHA256,
        ),
        "resampled_resource_windows": (
            RESAMPLED_RESOURCE_WINDOWS,
            RESAMPLED_RESOURCE_WINDOWS_SHA256,
        ),
        "resampled_resource_worker_audit": (
            RESAMPLED_RESOURCE_WORKER_AUDIT,
            RESAMPLED_RESOURCE_WORKER_AUDIT_SHA256,
        ),
        "resampled_resource_runner_audit": (
            RESAMPLED_RESOURCE_RUNNER_AUDIT,
            RESAMPLED_RESOURCE_RUNNER_AUDIT_SHA256,
        ),
        "fixed_policy_realization_runner": (
            FIXED_POLICY_REALIZATION_RUNNER,
            FIXED_POLICY_REALIZATION_RUNNER_SHA256,
        ),
        "golden_cases": (GOLDEN_CASES, GOLDEN_CASES_SHA256),
        "golden_equivalence": (GOLDEN_EQUIVALENCE, GOLDEN_EQUIVALENCE_SHA256),
    }
    evidence: dict[str, dict[str, Any]] = {}
    for role, (path, expected) in locks.items():
        if not path.is_file():
            raise FixedACacheStop(f"STOP_FIXED_A_LOCKED_FILE_MISSING:{role}:{path}")
        actual = sha256_file(path)
        if actual != expected:
            raise FixedACacheStop(
                f"STOP_FIXED_A_LOCKED_FILE_SHA:{role}:{actual}!={expected}"
            )
        evidence[role] = {
            "path": str(path.resolve()),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    contract = json.loads(FIXED_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN_BEFORE_MEASUREMENT":
        raise FixedACacheStop("STOP_FIXED_A_CONTRACT_NOT_FROZEN")
    if contract.get("authorization", {}).get("training_authorized") is not False:
        raise FixedACacheStop("STOP_FIXED_A_TRAINING_AUTHORIZATION_NOT_FALSE")
    if contract.get("cache_gate", {}).get("production_training_path_modification_allowed") is not False:
        raise FixedACacheStop("STOP_FIXED_A_PRODUCTION_MODIFICATION_NOT_FALSE")
    if tuple(contract.get("cache_gate", {}).get("candidates", ())) != CANDIDATES:
        raise FixedACacheStop("STOP_FIXED_A_CANDIDATE_SET_DRIFT")
    return evidence


@dataclass(frozen=True)
class FixedInputs:
    directory: Path
    selection: pd.DataFrame
    exposure: pd.DataFrame
    theta: pd.DataFrame
    whitelist: pd.DataFrame
    height_column: str
    width_column: str
    unique_rgb_column: str
    file_sha256: Mapping[str, str]

    @property
    def theta_by_uid(self) -> pd.DataFrame:
        return self.theta.set_index("canonical_uid", verify_integrity=True)

    @property
    def selection_by_uid(self) -> pd.DataFrame:
        return self.selection.set_index("canonical_uid", verify_integrity=True)

    @property
    def whitelist_by_uid(self) -> pd.DataFrame:
        return self.whitelist.set_index("uid", verify_integrity=True)


def load_fixed_inputs(directory: str | Path) -> FixedInputs:
    root = _safe_result_dir(directory)
    paths = {
        "selection": root / SELECTION_NAME,
        "exposure": root / EXPOSURE_NAME,
        "theta": root / THETA_NAME,
        "exposure_audit": root / EXPOSURE_AUDIT_NAME,
    }
    for role, path in paths.items():
        if not path.is_file():
            raise FixedACacheStop(f"STOP_FIXED_A_INPUT_MISSING:{role}:{path}")
    frozen_input_hashes = {
        "selection": SELECTION_SHA256,
        "exposure": EXPOSURE_SHA256,
    }
    for role, expected_sha256 in frozen_input_hashes.items():
        actual_sha256 = sha256_file(paths[role])
        if actual_sha256 != expected_sha256:
            raise FixedACacheStop(
                f"STOP_FIXED_A_INPUT_SHA:{role}:{actual_sha256}:{expected_sha256}"
            )

    selection = pd.read_csv(paths["selection"], float_precision="round_trip")
    exposure = pd.read_csv(paths["exposure"], float_precision="round_trip")
    theta = pd.read_csv(paths["theta"], float_precision="round_trip")
    whitelist = pd.read_csv(Q2_WHITELIST, dtype=str)
    _require_columns(selection, SELECTION_REQUIRED, "selection")
    _require_columns(exposure, EXPOSURE_REQUIRED, "exposure")
    _require_columns(theta, THETA_REQUIRED_BASE, "theta")
    _require_columns(whitelist, ("uid", "image_path", "image_sha256"), "whitelist")
    schema_checks = (
        (selection, SELECTION_SCHEMA, "selection"),
        (exposure, EXPOSURE_SCHEMA, "exposure"),
        (theta, THETA_SCHEMA, "theta"),
    )
    for frame, expected_schema, role in schema_checks:
        observed = set(frame["schema"].astype(str))
        if observed != {expected_schema}:
            raise FixedACacheStop(
                f"STOP_FIXED_A_INPUT_SCHEMA:{role}:{sorted(observed)}"
            )
    height_column = _resolve_column(theta, ("image_height", "height"), "theta.height")
    width_column = _resolve_column(theta, ("image_width", "width"), "theta.width")
    unique_rgb_column = _resolve_column(
        theta, ("unique_rgb_count", "unique_rgb"), "theta.unique_rgb"
    )

    if len(selection) != EXPECTED_IMAGE_COUNT or selection["canonical_uid"].duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_SELECTION_CARDINALITY")
    if len(theta) != EXPECTED_IMAGE_COUNT or theta["canonical_uid"].duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_THETA_CARDINALITY")
    if len(whitelist) != EXPECTED_IMAGE_COUNT or whitelist["uid"].duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_WHITELIST_CARDINALITY")
    if len(exposure) != EXPECTED_VISITS or exposure["formal_visit_index"].duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_EXPOSURE_CARDINALITY")
    if sorted(exposure["formal_visit_index"].astype(int).tolist()) != list(range(EXPECTED_VISITS)):
        raise FixedACacheStop("STOP_FIXED_A_EXPOSURE_INDEX_SET")

    uid_sets = {
        "selection": set(selection["canonical_uid"].astype(str)),
        "theta": set(theta["canonical_uid"].astype(str)),
        "whitelist": set(whitelist["uid"].astype(str)),
        "exposure": set(exposure["canonical_uid"].astype(str)),
    }
    if len({frozenset(value) for value in uid_sets.values()}) != 1:
        raise FixedACacheStop("STOP_FIXED_A_UID_SET_MISMATCH")

    selection_exists = selection["fixed_delta_exists"].map(
        lambda value: _strict_bool(value, "selection.fixed_delta_exists")
    )
    theta_exists = theta["fixed_delta_exists"].map(
        lambda value: _strict_bool(value, "theta.fixed_delta_exists")
    )
    if int(selection_exists.sum()) != EXPECTED_FIXED_SOLVES:
        raise FixedACacheStop("STOP_FIXED_A_SELECTION_SOLVE_COUNT")
    if int(theta_exists.sum()) != EXPECTED_FIXED_SOLVES:
        raise FixedACacheStop("STOP_FIXED_A_THETA_SOLVE_COUNT")
    selected_exists_by_uid = dict(zip(selection["canonical_uid"].astype(str), selection_exists))
    if any(
        bool(selected_exists_by_uid[str(uid)]) != bool(exists)
        for uid, exists in zip(theta["canonical_uid"].astype(str), theta_exists)
    ):
        raise FixedACacheStop("STOP_FIXED_A_SELECTION_THETA_EXISTENCE_MISMATCH")
    expected_status = np.where(theta_exists.to_numpy(), SOLVED_STATUS, IDENTITY_STATUS)
    if not np.array_equal(theta["realization_status"].astype(str).to_numpy(), expected_status):
        raise FixedACacheStop("STOP_FIXED_A_REALIZATION_STATUS")
    active_input_sha = (
        theta.loc[theta_exists, "input_image_sha256"].astype(str).str.lower()
    )
    if active_input_sha.duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_ACTIVE_IMAGE_SHA_NOT_ONE_TO_ONE")

    active = exposure["fixed_active"].map(lambda value: _strict_bool(value, "fixed_active"))
    resampled = exposure["resample_active"].map(
        lambda value: _strict_bool(value, "resample_active")
    )
    identity = exposure["fixed_identity"].map(
        lambda value: _strict_bool(value, "fixed_identity")
    )
    active_match = exposure["active_match"].map(
        lambda value: _strict_bool(value, "active_match")
    )
    identity_match = exposure["identity_match"].map(
        lambda value: _strict_bool(value, "identity_match")
    )
    if int(active.sum()) != EXPECTED_ACTIVE_VISITS:
        raise FixedACacheStop("STOP_FIXED_A_ACTIVE_VISIT_COUNT")
    if not (active == resampled).all() or not (identity == ~active).all():
        raise FixedACacheStop("STOP_FIXED_A_EXPOSURE_MASK_MISMATCH")
    if not active_match.all() or not identity_match.all():
        raise FixedACacheStop("STOP_FIXED_A_EXPOSURE_AUDIT_FLAG_FALSE")

    whitelist_sha = dict(
        zip(whitelist["uid"].astype(str), whitelist["image_sha256"].astype(str).str.lower())
    )
    for row, exists in zip(theta.to_dict("records"), theta_exists.tolist()):
        uid = str(row["canonical_uid"])
        if exists:
            input_sha = _require_sha256(row["input_image_sha256"], f"theta.input:{uid}")
            if input_sha != whitelist_sha[uid]:
                raise FixedACacheStop(f"STOP_FIXED_A_THETA_WHITELIST_SHA:{uid}")
            height = int(row[height_column])
            width = int(row[width_column])
            unique_rgb = int(row[unique_rgb_column])
            if height <= 0 or width <= 0 or unique_rgb <= 0 or unique_rgb > height * width:
                raise FixedACacheStop(f"STOP_FIXED_A_IMAGE_GEOMETRY:{uid}")
            for field in (
                "target_descriptor_sha256",
                "solve_key",
                "cache_key_sha256",
                "theta_raw_f8_sha256",
                "output_array_sha256",
                "science_cache_entry_sha256",
            ):
                _require_sha256(row[field], f"theta.{field}:{uid}")
            theta_value = np.asarray([row[column] for column in THETA_COLUMNS], dtype="<f8")
            if theta_value.shape != (3,) or not np.isfinite(theta_value).all():
                raise FixedACacheStop(f"STOP_FIXED_A_THETA_NUMERIC:{uid}")
            theta_sha = hashlib.sha256(
                THETA_HASH_PREFIX + theta_value.tobytes(order="C")
            ).hexdigest()
            if theta_sha != str(row["theta_raw_f8_sha256"]).lower():
                raise FixedACacheStop(f"STOP_FIXED_A_THETA_RAW_SHA:{uid}")
        else:
            optional_input_sha = _optional_text(row["input_image_sha256"]).lower()
            if optional_input_sha and optional_input_sha != whitelist_sha[uid]:
                raise FixedACacheStop(f"STOP_FIXED_A_IDENTITY_WHITELIST_SHA:{uid}")
            for field in (
                *THETA_COLUMNS,
                "output_array_sha256",
                height_column,
                width_column,
                unique_rgb_column,
            ):
                if not pd.isna(row[field]) and _optional_text(row[field]):
                    raise FixedACacheStop(f"STOP_FIXED_A_IDENTITY_REALIZATION_VALUE:{uid}:{field}")

    selection_sha_by_uid = dict(
        zip(selection["canonical_uid"].astype(str), selection["selected_delta_sha"].map(_optional_text))
    )
    for row, is_active in zip(exposure.to_dict("records"), active.tolist()):
        uid = str(row["canonical_uid"])
        selected_sha = _optional_text(row["fixed_selected_delta_sha"])
        used_sha = _optional_text(row["fixed_delta_sha_used"])
        if selected_sha != selection_sha_by_uid[uid]:
            raise FixedACacheStop(f"STOP_FIXED_A_EXPOSURE_SELECTION_SHA:{uid}")
        if is_active and used_sha != selected_sha:
            raise FixedACacheStop(f"STOP_FIXED_A_ACTIVE_DELTA_USE:{uid}")
        if not is_active and used_sha:
            raise FixedACacheStop(f"STOP_FIXED_A_IDENTITY_DELTA_USE:{uid}")

    exposure_audit = json.loads(paths["exposure_audit"].read_text(encoding="utf-8"))
    if exposure_audit.get("pass") is not True and exposure_audit.get("status") not in {
        "PASS",
        "COMPLETE_PASS",
    }:
        raise FixedACacheStop("STOP_FIXED_A_EXPOSURE_JSON_NOT_PASS")
    return FixedInputs(
        directory=root,
        selection=selection,
        exposure=exposure,
        theta=theta,
        whitelist=whitelist,
        height_column=height_column,
        width_column=width_column,
        unique_rgb_column=unique_rgb_column,
        file_sha256={role: sha256_file(path) for role, path in paths.items()},
    )


def _load_execution_core() -> tuple[Any, Any]:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import m1_routea_resource_equivalence_execution_core_v1 as execution_core

    if sha256_file(Path(execution_core.__file__)) != EXACT_TRANSFORM_CORE_SHA256:
        raise FixedACacheStop("STOP_FIXED_A_IMPORTED_EXECUTION_CORE_SHA")
    frozen = execution_core.load_frozen_realization_core()
    return execution_core, frozen


def _decode_rgb(path: Path, expected_sha256: str) -> tuple[np.ndarray, str, float]:
    started = time.perf_counter()
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise FixedACacheStop(f"STOP_FIXED_A_IMAGE_SHA:{path}:{actual_sha}")
    _, frozen = _load_execution_core()
    rgb, decoder = frozen.decode_q2_rgb(path)
    elapsed = time.perf_counter() - started
    value = np.ascontiguousarray(rgb)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise FixedACacheStop("STOP_FIXED_A_DECODE_REPRESENTATION")
    return value, str(decoder), elapsed


def _formal_bgr_sha_from_rgb(rgb_uint8: np.ndarray) -> str:
    rgb = np.asarray(rgb_uint8)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise FixedACacheStop("STOP_FIXED_A_FORMAL_HASH_REPRESENTATION")
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    digest = hashlib.sha256()
    digest.update(ARRAY_HASH_PREFIX)
    digest.update(b"uint8")
    digest.update(str(tuple(int(value) for value in bgr.shape)).encode("ascii"))
    digest.update(bgr.tobytes(order="C"))
    return digest.hexdigest()


def _array_bytes_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _packed_palette(rgb_palette: np.ndarray) -> np.ndarray:
    palette = np.asarray(rgb_palette)
    if palette.dtype != np.uint8 or palette.ndim != 2 or palette.shape[1] != 3:
        raise FixedACacheStop("STOP_FIXED_A_SOURCE_PALETTE_REPRESENTATION")
    return (
        (palette[:, 0].astype(np.uint32) << np.uint32(16))
        | (palette[:, 1].astype(np.uint32) << np.uint32(8))
        | palette[:, 2].astype(np.uint32)
    )


@dataclass(frozen=True)
class ExactRendered:
    rgb: np.ndarray
    bgr_sha256: str
    source_palette: np.ndarray
    deployment_palette: np.ndarray
    unique_rgb: int


def _render_theta_exact(rgb: np.ndarray, theta: Sequence[float]) -> ExactRendered:
    _, frozen = _load_execution_core()
    value = np.ascontiguousarray(rgb)
    palette = frozen.PaletteImage.from_rgb_uint8(value)
    theta_value = np.asarray(theta, dtype=np.float64)
    continuous, _, _, _ = palette.continuous_output(theta_value)
    if continuous.dtype != np.float64 or not np.isfinite(continuous).all():
        raise FixedACacheStop("STOP_FIXED_A_CONTINUOUS_OUTPUT")
    deployment = np.ascontiguousarray(np.rint(continuous * 255.0).astype(np.uint8))
    output = np.ascontiguousarray(
        deployment[palette.inverse_flat].reshape(palette.height, palette.width, 3)
    )
    reported = str(
        frozen._array_hash_from_palette(
            deployment, palette.inverse_flat, palette.height, palette.width
        )
    )
    direct = _formal_bgr_sha_from_rgb(output)
    if reported != direct:
        raise FixedACacheStop("STOP_FIXED_A_FROZEN_DIRECT_BGR_HASH_MISMATCH")
    source = np.ascontiguousarray(palette.palette_uint8)
    packed = _packed_palette(source)
    if len(packed) > 1 and not bool(np.all(packed[1:] > packed[:-1])):
        raise FixedACacheStop("STOP_FIXED_A_SOURCE_PALETTE_NOT_STRICT_SORTED")
    return ExactRendered(
        rgb=output,
        bgr_sha256=reported,
        source_palette=source,
        deployment_palette=deployment,
        unique_rgb=len(source),
    )


def palette_remap_exact(
    rgb_uint8: np.ndarray,
    source_palette_uint8: np.ndarray,
    deployment_palette_uint8: np.ndarray,
    *,
    chunk_pixels: int = PALETTE_REMAP_CHUNK_PIXELS,
) -> np.ndarray:
    """Remap exact source colors in bounded chunks using frozen palette order."""

    image = np.asarray(rgb_uint8)
    source = np.asarray(source_palette_uint8)
    deployment = np.asarray(deployment_palette_uint8)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise FixedACacheStop("STOP_FIXED_A_PALETTE_INPUT_REPRESENTATION")
    if (
        source.dtype != np.uint8
        or deployment.dtype != np.uint8
        or source.shape != deployment.shape
        or source.ndim != 2
        or source.shape[1] != 3
    ):
        raise FixedACacheStop("STOP_FIXED_A_PALETTE_PAIR_REPRESENTATION")
    if int(chunk_pixels) <= 0:
        raise FixedACacheStop("STOP_FIXED_A_PALETTE_CHUNK_INVALID")
    source_packed = _packed_palette(source)
    if len(source_packed) > 1 and not bool(np.all(source_packed[1:] > source_packed[:-1])):
        raise FixedACacheStop("STOP_FIXED_A_PALETTE_SOURCE_ORDER")
    flat = image.reshape(-1, 3)
    output = np.empty_like(flat, dtype=np.uint8)
    for start in range(0, len(flat), int(chunk_pixels)):
        block = flat[start : start + int(chunk_pixels)]
        packed = (
            (block[:, 0].astype(np.uint32) << np.uint32(16))
            | (block[:, 1].astype(np.uint32) << np.uint32(8))
            | block[:, 2].astype(np.uint32)
        )
        indices = np.searchsorted(source_packed, packed, side="left")
        if np.any(indices >= len(source_packed)):
            raise FixedACacheStop("STOP_FIXED_A_PALETTE_COLOR_MISSING_HIGH")
        if not np.array_equal(source_packed[indices], packed):
            raise FixedACacheStop("STOP_FIXED_A_PALETTE_COLOR_MISSING")
        output[start : start + len(block)] = deployment[indices]
    rendered = np.ascontiguousarray(output.reshape(image.shape))
    if rendered.dtype != np.uint8 or not rendered.flags.c_contiguous:
        raise FixedACacheStop("STOP_FIXED_A_PALETTE_OUTPUT_REPRESENTATION")
    return rendered


def _payload_paths(cache_root: Path, image_sha256: str) -> dict[str, Path]:
    digest = _require_sha256(image_sha256, "payload.image")
    prefix = digest[:2]
    return {
        "bgr": cache_root / "bgr" / prefix / f"{digest}.npy",
        "palette": cache_root / "palette" / prefix / f"{digest}.npy",
        "meta": cache_root / "meta" / prefix / f"{digest}.json",
    }


def _golden_audit_payload_paths(cache_root: Path, case: "AuditCase") -> dict[str, Path]:
    """Namespace historical Golden theta away from the one-output Fixed-A cache."""

    if case.theta is None:
        raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_THETA_MISSING:{case.case_id}")
    theta = np.asarray(case.theta, dtype="<f8")
    if theta.shape != (3,) or not np.isfinite(theta).all():
        raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_THETA_REPRESENTATION:{case.case_id}")
    digest = hashlib.sha256()
    digest.update(b"M1_NATIVE_A_FIXED_GOLDEN_AUDIT_PAYLOAD_V1\0")
    digest.update(_require_sha256(case.image_sha256, "golden_payload.image").encode("ascii"))
    digest.update(theta.tobytes(order="C"))
    digest.update(
        _require_sha256(case.expected_bgr_sha256, "golden_payload.output").encode("ascii")
    )
    key = digest.hexdigest()
    root = cache_root / "golden_audit_only" / key[:2] / key
    return {
        "bgr": root / "preletterbox_bgr.npy",
        "palette": root / "source_deployment_palette.npy",
        "meta": root / "audit_payload.json",
    }


def _prepare_golden_audit_payloads(
    cache_root: Path, case: "AuditCase", rendered: ExactRendered
) -> dict[str, Any]:
    """Atomically materialize audit-only payloads for the historical solve theta.

    Golden Native/SMG theta generally differs from Fixed-A's selected theta for
    the same image.  Consequently these files are exact-solve audit fixtures,
    never entries in or fallbacks for the per-image Fixed-A production cache.
    """

    paths = _golden_audit_payload_paths(cache_root, case)
    bgr = np.ascontiguousarray(rendered.rgb[:, :, ::-1])
    palette = np.ascontiguousarray(
        np.concatenate([rendered.source_palette, rendered.deployment_palette], axis=1)
    )
    expected = {
        "schema": "m1_native_a_fixed_golden_cache_audit_payload_v1",
        "purpose": "HISTORICAL_GOLDEN_EXACT_SOLVE_AUDIT_ONLY_NOT_FIXED_A_CACHE_REUSE",
        "case_id": case.case_id,
        "canonical_uid": case.canonical_uid,
        "input_image_sha256": case.image_sha256,
        "theta_raw_f8_sha256": hashlib.sha256(
            THETA_HASH_PREFIX + np.asarray(case.theta, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "expected_output_bgr_sha256": case.expected_bgr_sha256,
        "bgr_shape": [int(value) for value in bgr.shape],
        "palette_shape": [int(value) for value in palette.shape],
        "bgr_array_bytes_sha256": _array_bytes_sha(bgr),
        "palette_array_bytes_sha256": _array_bytes_sha(palette),
        "cache_fallback_allowed": False,
    }
    exists = {role: path.is_file() for role, path in paths.items()}
    if any(exists.values()) and not all(exists.values()):
        raise FixedACacheStop(
            f"STOP_FIXED_A_GOLDEN_AUDIT_PARTIAL_PAYLOAD:{case.case_id}:{exists}"
        )
    if all(exists.values()):
        observed = json.loads(paths["meta"].read_text(encoding="utf-8"))
        for field, value in expected.items():
            if observed.get(field) != value:
                raise FixedACacheStop(
                    f"STOP_FIXED_A_GOLDEN_AUDIT_LINEAGE:{case.case_id}:{field}"
                )
        observed_bgr = np.load(paths["bgr"], mmap_mode="r", allow_pickle=False)
        observed_palette = np.load(paths["palette"], mmap_mode="r", allow_pickle=False)
        valid = (
            observed_bgr.dtype == np.uint8
            and list(observed_bgr.shape) == expected["bgr_shape"]
            and _array_bytes_sha(observed_bgr) == expected["bgr_array_bytes_sha256"]
            and observed_palette.dtype == np.uint8
            and list(observed_palette.shape) == expected["palette_shape"]
            and _array_bytes_sha(observed_palette)
            == expected["palette_array_bytes_sha256"]
        )
        del observed_bgr, observed_palette
        if not valid:
            raise FixedACacheStop(
                f"STOP_FIXED_A_GOLDEN_AUDIT_PAYLOAD_BYTES:{case.case_id}"
            )
        return observed
    atomic_npy(paths["bgr"], bgr)
    atomic_npy(paths["palette"], palette)
    value = {
        **expected,
        "bgr_path": str(paths["bgr"].resolve()),
        "bgr_file_sha256": sha256_file(paths["bgr"]),
        "palette_path": str(paths["palette"].resolve()),
        "palette_file_sha256": sha256_file(paths["palette"]),
        "cache_payload_excluded_from_core_zip": True,
    }
    atomic_json(paths["meta"], value)
    return value


def _load_golden_bgr_candidate(cache_root: Path, case: "AuditCase") -> np.ndarray:
    path = _golden_audit_payload_paths(cache_root, case)["bgr"]
    if not path.is_file():
        raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_BGR_CACHE_MISS:{case.case_id}")
    bgr = np.load(path, mmap_mode="r", allow_pickle=False)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_BGR_REPRESENTATION:{case.case_id}")
    output = np.ascontiguousarray(bgr[:, :, ::-1])
    del bgr
    return output


def _load_golden_palette_candidate(
    cache_root: Path, case: "AuditCase", original_rgb: np.ndarray
) -> np.ndarray:
    path = _golden_audit_payload_paths(cache_root, case)["palette"]
    if not path.is_file():
        raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_PALETTE_CACHE_MISS:{case.case_id}")
    payload = np.load(path, mmap_mode="r", allow_pickle=False)
    if payload.dtype != np.uint8 or payload.ndim != 2 or payload.shape[1] != 6:
        raise FixedACacheStop(
            f"STOP_FIXED_A_GOLDEN_PALETTE_REPRESENTATION:{case.case_id}"
        )
    source = np.ascontiguousarray(payload[:, :3])
    deployment = np.ascontiguousarray(payload[:, 3:])
    del payload
    return palette_remap_exact(original_rgb, source, deployment)


def _load_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "m1_native_a_fixed_cache_entry_v1":
        raise FixedACacheStop(f"STOP_FIXED_A_SIDECAR_SCHEMA:{path}")
    return value


def _sidecar_base(row: Mapping[str, Any], inputs: FixedInputs) -> dict[str, Any]:
    uid = str(row["canonical_uid"])
    return {
        "schema": "m1_native_a_fixed_cache_entry_v1",
        "canonical_uid": uid,
        "input_image_sha256": str(row["input_image_sha256"]).lower(),
        "solve_key": str(row["solve_key"]).lower(),
        "cache_namespace": str(row["cache_namespace"]),
        "cache_key_sha256": str(row["cache_key_sha256"]).lower(),
        "theta_raw_f8_sha256": str(row["theta_raw_f8_sha256"]).lower(),
        "expected_output_bgr_sha256": str(row["output_array_sha256"]).lower(),
        "height": int(row[inputs.height_column]),
        "width": int(row[inputs.width_column]),
        "unique_rgb": int(row[inputs.unique_rgb_column]),
        "candidates": {},
    }


def _validate_sidecar_lineage(
    sidecar: Mapping[str, Any], row: Mapping[str, Any], inputs: FixedInputs
) -> None:
    expected = _sidecar_base(row, inputs)
    for field in (
        "canonical_uid",
        "input_image_sha256",
        "solve_key",
        "cache_namespace",
        "cache_key_sha256",
        "theta_raw_f8_sha256",
        "expected_output_bgr_sha256",
        "height",
        "width",
        "unique_rgb",
    ):
        if sidecar.get(field) != expected[field]:
            raise FixedACacheStop(
                f"STOP_FIXED_A_SIDECAR_LINEAGE:{expected['canonical_uid']}:{field}"
            )


def _validate_payload_record(
    candidate: str,
    path: Path,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    inputs: FixedInputs,
    *,
    verify_file_sha: bool,
) -> bool:
    if not path.is_file() or record.get("complete") is not True:
        return False
    if Path(str(record.get("path", ""))).resolve() != path.resolve():
        return False
    if int(record.get("file_bytes", -1)) != path.stat().st_size:
        return False
    if verify_file_sha and sha256_file(path) != str(record.get("file_sha256", "")):
        return False
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    height = int(row[inputs.height_column])
    width = int(row[inputs.width_column])
    unique_rgb = int(row[inputs.unique_rgb_column])
    if candidate == BGR_CACHE:
        valid = array.dtype == np.uint8 and array.shape == (height, width, 3)
    elif candidate == PALETTE_CACHE:
        valid = array.dtype == np.uint8 and array.shape == (unique_rgb, 6)
    else:
        raise FixedACacheStop(f"STOP_FIXED_A_PAYLOAD_CANDIDATE:{candidate}")
    del array
    return bool(valid)


def _estimated_cache_bytes(inputs: FixedInputs, candidates: Sequence[str]) -> dict[str, int]:
    active = inputs.theta.loc[
        inputs.theta["fixed_delta_exists"].map(
            lambda value: _strict_bool(value, "estimate.fixed_delta_exists")
        )
    ]
    estimates = {THETA_ONLY: 0, BGR_CACHE: 0, PALETTE_CACHE: 0}
    if BGR_CACHE in candidates:
        estimates[BGR_CACHE] = int(
            (
                active[inputs.height_column].astype("int64")
                * active[inputs.width_column].astype("int64")
                * 3
            ).sum()
        )
    if PALETTE_CACHE in candidates:
        estimates[PALETTE_CACHE] = int(
            (active[inputs.unique_rgb_column].astype("int64") * 6).sum()
        )
    return estimates


def build_cache(
    *,
    result_dir: Path,
    fixed_input_dir: Path,
    candidates: Sequence[str],
    verify_resume_file_sha: bool,
) -> dict[str, Any]:
    if not verify_resume_file_sha:
        raise FixedACacheStop("STOP_FIXED_A_RESUME_PAYLOAD_SHA_VERIFICATION_REQUIRED")
    requested = tuple(dict.fromkeys(candidates))
    if not requested or any(candidate not in CANDIDATES for candidate in requested):
        raise FixedACacheStop("STOP_FIXED_A_BUILD_CANDIDATES")
    locks = _verify_sha_locks()
    inputs = load_fixed_inputs(fixed_input_dir)
    output = _safe_result_dir(result_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_root = output / "working_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    estimates = _estimated_cache_bytes(inputs, requested)
    payload_estimate = sum(estimates[candidate] for candidate in requested)
    free_before = shutil.disk_usage(output).free
    if int(math.ceil(payload_estimate * 1.20)) > free_before:
        raise FixedACacheStop(
            f"STOP_FIXED_A_DISK_HEADROOM:{payload_estimate}:{free_before}"
        )

    theta = inputs.theta.sort_values("canonical_uid", kind="mergesort")
    whitelist = inputs.whitelist_by_uid
    active_rows = [
        row
        for row in theta.to_dict("records")
        if _strict_bool(row["fixed_delta_exists"], "build.fixed_delta_exists")
    ]
    if len(active_rows) != EXPECTED_FIXED_SOLVES:
        raise FixedACacheStop("STOP_FIXED_A_BUILD_ACTIVE_COUNT")

    started = time.perf_counter()
    candidate_counts = {
        candidate: {"built": 0, "resumed": 0, "complete": 0}
        for candidate in requested
    }
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(active_rows, start=1):
        uid = str(row["canonical_uid"])
        image_sha = str(row["input_image_sha256"]).lower()
        paths = _payload_paths(cache_root, image_sha)
        sidecar = _load_sidecar(paths["meta"])
        if sidecar:
            _validate_sidecar_lineage(sidecar, row, inputs)
        else:
            sidecar = _sidecar_base(row, inputs)
        missing: list[str] = []
        for candidate in requested:
            if candidate == THETA_ONLY:
                candidate_counts[candidate]["resumed"] += 1
                candidate_counts[candidate]["complete"] += 1
                continue
            payload_path = paths["bgr"] if candidate == BGR_CACHE else paths["palette"]
            record = sidecar.get("candidates", {}).get(candidate, {})
            payload_valid = _validate_payload_record(
                candidate,
                payload_path,
                record,
                row,
                inputs,
                verify_file_sha=True,
            )
            if payload_valid:
                candidate_counts[candidate]["resumed"] += 1
                candidate_counts[candidate]["complete"] += 1
            elif record.get("complete") is True:
                raise FixedACacheStop(
                    f"STOP_FIXED_A_RESUME_REGISTERED_PAYLOAD_CORRUPT:{uid}:{candidate}"
                )
            else:
                missing.append(candidate)
        if missing:
            whitelist_row = whitelist.loc[uid]
            image_path = _guard_adapt_image(str(whitelist_row["image_path"]))
            rgb, decoder, decode_seconds = _decode_rgb(image_path, image_sha)
            theta_value = np.asarray([row[column] for column in THETA_COLUMNS], dtype="<f8")
            transform_started = time.perf_counter()
            rendered = _render_theta_exact(rgb, theta_value)
            transform_seconds = time.perf_counter() - transform_started
            if rendered.bgr_sha256 != str(row["output_array_sha256"]).lower():
                raise FixedACacheStop(f"STOP_FIXED_A_BUILD_REFERENCE_BGR_SHA:{uid}")
            if rendered.rgb.shape != (
                int(row[inputs.height_column]),
                int(row[inputs.width_column]),
                3,
            ):
                raise FixedACacheStop(f"STOP_FIXED_A_BUILD_OUTPUT_SHAPE:{uid}")
            if rendered.unique_rgb != int(row[inputs.unique_rgb_column]):
                raise FixedACacheStop(f"STOP_FIXED_A_BUILD_UNIQUE_RGB:{uid}")
            for candidate in missing:
                if candidate == BGR_CACHE:
                    payload = np.ascontiguousarray(rendered.rgb[:, :, ::-1])
                    payload_path = paths["bgr"]
                elif candidate == PALETTE_CACHE:
                    payload = np.ascontiguousarray(
                        np.concatenate(
                            [rendered.source_palette, rendered.deployment_palette], axis=1
                        )
                    )
                    payload_path = paths["palette"]
                else:
                    continue
                atomic_npy(payload_path, payload)
                record = {
                    "complete": True,
                    "path": str(payload_path.resolve()),
                    "file_bytes": payload_path.stat().st_size,
                    "file_sha256": sha256_file(payload_path),
                    "array_bytes_sha256": _array_bytes_sha(payload),
                    "dtype": str(payload.dtype),
                    "shape": [int(value) for value in payload.shape],
                    "decoder": decoder,
                    "decode_seconds_build": decode_seconds,
                    "transform_seconds_build": transform_seconds,
                    "built_at": datetime.now().isoformat(),
                }
                sidecar.setdefault("candidates", {})[candidate] = record
                atomic_json(paths["meta"], sidecar)
                candidate_counts[candidate]["built"] += 1
                candidate_counts[candidate]["complete"] += 1
        manifest_rows.append(
            {
                "schema": "m1_native_a_fixed_cache_manifest_entry_v1",
                "canonical_uid": uid,
                "input_image_sha256": image_sha,
                "solve_key": str(row["solve_key"]),
                "theta_raw_f8_sha256": str(row["theta_raw_f8_sha256"]),
                "expected_output_bgr_sha256": str(row["output_array_sha256"]),
                "sidecar_path": str(paths["meta"].resolve()),
                "sidecar_sha256": sha256_file(paths["meta"]),
                **{
                    f"{candidate}_complete": (
                        True
                        if candidate == THETA_ONLY
                        else bool(sidecar.get("candidates", {}).get(candidate, {}).get("complete"))
                    )
                    for candidate in requested
                },
            }
        )
        if index % 16 == 0 or index == len(active_rows):
            atomic_json(
                output / "raw" / "fixed_a_cache_build_progress.json",
                {
                    "schema": "m1_native_a_fixed_cache_build_progress_v1",
                    "complete_entries": index,
                    "expected_entries": EXPECTED_FIXED_SOLVES,
                    "candidate_counts": candidate_counts,
                    "last_canonical_uid": uid,
                    "cache_payload_excluded_from_core_zip": True,
                },
            )

    manifest_frame = pd.DataFrame(manifest_rows)
    if any(
        int(candidate_counts[candidate]["complete"]) != EXPECTED_FIXED_SOLVES
        for candidate in requested
    ):
        raise FixedACacheStop("STOP_FIXED_A_BUILD_COMPLETE_COUNT")
    manifest_csv = output / "fixed_a_cache_manifest.csv"
    atomic_csv(manifest_csv, manifest_frame)
    actual_payload_bytes: dict[str, int] = {}
    for candidate in requested:
        if candidate == THETA_ONLY:
            actual_payload_bytes[candidate] = 0
            continue
        column = "bgr" if candidate == BGR_CACHE else "palette"
        actual_payload_bytes[candidate] = sum(
            _payload_paths(cache_root, str(row["input_image_sha256"]))[column].stat().st_size
            for row in active_rows
        )
    result = {
        "schema": "m1_native_a_fixed_cache_build_v1",
        "status": "PASS",
        "fixed_contract_sha256": FIXED_CONTRACT_SHA256,
        "sha_locks": locks,
        "fixed_input_sha256": dict(inputs.file_sha256),
        "requested_candidates": list(requested),
        "expected_active_entries": EXPECTED_FIXED_SOLVES,
        "candidate_counts": candidate_counts,
        "estimated_raw_payload_bytes": estimates,
        "actual_npy_payload_bytes": actual_payload_bytes,
        "disk_free_before_bytes": free_before,
        "disk_free_after_bytes": shutil.disk_usage(output).free,
        "wall_seconds": time.perf_counter() - started,
        "cache_root": str(cache_root.resolve()),
        "cache_payload_excluded_from_core_zip": True,
        "manifest_csv": str(manifest_csv.resolve()),
        "manifest_csv_sha256": sha256_file(manifest_csv),
        "online_powell_calls": 0,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "dataset_mutations": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "SCIENCE_CHANGE": False,
    }
    atomic_json(output / "fixed_a_cache_build.json", result)
    return result


@dataclass(frozen=True)
class AuditCase:
    suite: str
    case_id: str
    canonical_uid: str
    image_path: Path
    image_sha256: str
    active: bool
    theta: np.ndarray | None
    expected_bgr_sha256: str
    formal_visit_index: int | None
    unique_rgb: int | None


def _deterministic_replay_selection(exposure: pd.DataFrame) -> pd.DataFrame:
    frame = exposure.copy()
    frame["selection_sha256"] = [
        hashlib.sha256(
            (
                "M1_NATIVE_A_FIXED_CACHE_REPLAY_SELECTION_V1|"
                f"{int(index)}|{uid}"
            ).encode("utf-8")
        ).hexdigest()
        for index, uid in zip(
            frame["formal_visit_index"], frame["canonical_uid"].astype(str)
        )
    ]
    selected = frame.sort_values(
        ["selection_sha256", "formal_visit_index"], kind="mergesort"
    ).head(EXPECTED_FIXED_REPLAY_VISITS)
    if len(selected) != EXPECTED_FIXED_REPLAY_VISITS:
        raise FixedACacheStop("STOP_FIXED_A_REPLAY_SELECTION_COUNT")
    return selected.sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)


def _load_audit_cases(inputs: FixedInputs) -> tuple[list[AuditCase], pd.DataFrame]:
    cases: list[AuditCase] = []
    golden = pd.read_csv(GOLDEN_CASES, float_precision="round_trip")
    golden_eq = pd.read_csv(GOLDEN_EQUIVALENCE, float_precision="round_trip")
    if len(golden) != 6 or len(golden_eq) != 6:
        raise FixedACacheStop("STOP_FIXED_A_GOLDEN_COUNT")
    if "strict_pass" not in golden_eq or not golden_eq["strict_pass"].map(
        lambda value: _strict_bool(value, "golden_equivalence.strict_pass")
    ).all():
        raise FixedACacheStop("STOP_FIXED_A_GOLDEN_PRIOR_EQUIVALENCE")
    for row in golden.to_dict("records"):
        image_path = _guard_adapt_image(str(row["image_path"]))
        cases.append(
            AuditCase(
                suite="golden_6",
                case_id=str(row["golden_label"]),
                canonical_uid=str(row["uid"]),
                image_path=image_path,
                image_sha256=_require_sha256(
                    row["input_image_sha256"], f"golden.input:{row['golden_label']}"
                ),
                active=True,
                theta=np.asarray([row[column] for column in THETA_COLUMNS], dtype="<f8"),
                expected_bgr_sha256=_require_sha256(
                    row["output_uint8_BGR_SHA_old"], f"golden.output:{row['golden_label']}"
                ),
                formal_visit_index=None,
                unique_rgb=None,
            )
        )

    replay = _deterministic_replay_selection(inputs.exposure)
    replay.insert(0, "audit_selection_schema", "m1_native_a_fixed_cache_audit_selection_v1")
    theta_by_uid = inputs.theta_by_uid
    whitelist = inputs.whitelist_by_uid
    for row in replay.to_dict("records"):
        uid = str(row["canonical_uid"])
        theta_row = theta_by_uid.loc[uid]
        whitelist_row = whitelist.loc[uid]
        active = _strict_bool(row["fixed_active"], "audit.fixed_active")
        theta = (
            np.asarray([theta_row[column] for column in THETA_COLUMNS], dtype="<f8")
            if active
            else None
        )
        expected = str(theta_row["output_array_sha256"]).lower() if active else ""
        cases.append(
            AuditCase(
                suite="fixed_replay_240",
                case_id=f"visit_{int(row['formal_visit_index']):06d}",
                canonical_uid=uid,
                image_path=_guard_adapt_image(str(whitelist_row["image_path"])),
                image_sha256=str(whitelist_row["image_sha256"]).lower(),
                active=active,
                theta=theta,
                expected_bgr_sha256=expected,
                formal_visit_index=int(row["formal_visit_index"]),
                unique_rgb=(
                    int(theta_row[inputs.unique_rgb_column]) if active else None
                ),
            )
        )

    high = inputs.theta.loc[
        inputs.theta["fixed_delta_exists"].map(
            lambda value: _strict_bool(value, "high_palette.fixed_delta_exists")
        )
    ].sort_values(
        [inputs.unique_rgb_column, "canonical_uid"],
        ascending=[False, True],
        kind="mergesort",
    ).head(HIGH_PALETTE_COUNT)
    if len(high) != HIGH_PALETTE_COUNT:
        raise FixedACacheStop("STOP_FIXED_A_HIGH_PALETTE_COUNT")
    for rank, row in enumerate(high.to_dict("records"), start=1):
        uid = str(row["canonical_uid"])
        whitelist_row = whitelist.loc[uid]
        cases.append(
            AuditCase(
                suite="high_palette_16",
                case_id=f"high_palette_{rank:02d}_{uid}",
                canonical_uid=uid,
                image_path=_guard_adapt_image(str(whitelist_row["image_path"])),
                image_sha256=str(row["input_image_sha256"]).lower(),
                active=True,
                theta=np.asarray([row[column] for column in THETA_COLUMNS], dtype="<f8"),
                expected_bgr_sha256=str(row["output_array_sha256"]).lower(),
                formal_visit_index=None,
                unique_rgb=int(row[inputs.unique_rgb_column]),
            )
        )
    return cases, replay


def _load_bgr_candidate(cache_root: Path, case: AuditCase) -> np.ndarray:
    path = _payload_paths(cache_root, case.image_sha256)["bgr"]
    if not path.is_file():
        raise FixedACacheStop(f"STOP_FIXED_A_BGR_CACHE_MISS:{case.canonical_uid}")
    bgr = np.load(path, mmap_mode="r", allow_pickle=False)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise FixedACacheStop(f"STOP_FIXED_A_BGR_CACHE_REPRESENTATION:{case.canonical_uid}")
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    del bgr
    return rgb


def _load_palette_candidate(cache_root: Path, case: AuditCase, rgb: np.ndarray) -> np.ndarray:
    path = _payload_paths(cache_root, case.image_sha256)["palette"]
    if not path.is_file():
        raise FixedACacheStop(f"STOP_FIXED_A_PALETTE_CACHE_MISS:{case.canonical_uid}")
    payload = np.load(path, mmap_mode="r", allow_pickle=False)
    if payload.dtype != np.uint8 or payload.ndim != 2 or payload.shape[1] != 6:
        raise FixedACacheStop(
            f"STOP_FIXED_A_PALETTE_CACHE_REPRESENTATION:{case.canonical_uid}"
        )
    source = np.ascontiguousarray(payload[:, :3])
    deployment = np.ascontiguousarray(payload[:, 3:])
    del payload
    return palette_remap_exact(rgb, source, deployment)


def _letterbox_bgr_sha(bgr: np.ndarray) -> str:
    if str(ROOT / "project" / "ultralytics") not in sys.path:
        sys.path.insert(0, str(ROOT / "project" / "ultralytics"))
    import cv2
    from ultralytics.data.augment import LetterBox

    letterbox = LetterBox(
        new_shape=(640, 640),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        stride=32,
        padding_value=114,
        interpolation=cv2.INTER_LINEAR,
    )
    labels = {"img": np.ascontiguousarray(bgr)}
    params = letterbox.get_params(labels)
    output = np.ascontiguousarray(letterbox.apply_image(labels, params)["img"])
    return _array_bytes_sha(output)


def _prepare_equivalence_runtime_imports() -> None:
    """Resolve all lazy transform/letterbox imports before the RNG snapshot."""

    _load_execution_core()
    local_ultralytics = str(ROOT / "project" / "ultralytics")
    if local_ultralytics not in sys.path:
        sys.path.insert(0, local_ultralytics)
    import cv2  # noqa: F401
    from ultralytics.data.augment import LetterBox  # noqa: F401


def _rng_snapshot() -> dict[str, str]:
    python_sha = hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest()
    state = np.random.get_state()
    numpy_digest = hashlib.sha256()
    numpy_digest.update(str(state[0]).encode("ascii"))
    numpy_digest.update(np.ascontiguousarray(state[1]).tobytes(order="C"))
    numpy_digest.update(repr(state[2:]).encode("utf-8"))
    result = {"python": python_sha, "numpy": numpy_digest.hexdigest()}
    if "torch" in sys.modules:
        import torch

        result["torch_cpu"] = hashlib.sha256(
            torch.get_rng_state().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        if torch.cuda.is_available():
            cuda_digest = hashlib.sha256()
            for value in torch.cuda.get_rng_state_all():
                cuda_digest.update(value.cpu().contiguous().numpy().tobytes())
            result["torch_cuda"] = cuda_digest.hexdigest()
    return result


def _candidate_output(
    *,
    candidate: str,
    case: AuditCase,
    original_rgb: np.ndarray,
    cache_root: Path,
) -> np.ndarray:
    if not case.active:
        return np.ascontiguousarray(original_rgb.copy())
    if case.theta is None:
        raise FixedACacheStop(f"STOP_FIXED_A_ACTIVE_THETA_MISSING:{case.case_id}")
    if candidate == THETA_ONLY:
        return _render_theta_exact(original_rgb, case.theta).rgb
    if candidate == BGR_CACHE:
        if case.suite == "golden_6":
            return _load_golden_bgr_candidate(cache_root, case)
        return _load_bgr_candidate(cache_root, case)
    if candidate == PALETTE_CACHE:
        if case.suite == "golden_6":
            return _load_golden_palette_candidate(cache_root, case, original_rgb)
        return _load_palette_candidate(cache_root, case, original_rgb)
    raise FixedACacheStop(f"STOP_FIXED_A_UNKNOWN_CANDIDATE:{candidate}")


def run_equivalence_and_benchmark(
    *,
    result_dir: Path,
    fixed_input_dir: Path,
    candidates: Sequence[str],
    repetitions: int,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(candidates))
    if requested != CANDIDATES:
        raise FixedACacheStop("STOP_FIXED_A_AUDIT_REQUIRES_ALL_THREE_CANDIDATES")
    if int(repetitions) < DEFAULT_REPETITIONS:
        raise FixedACacheStop("STOP_FIXED_A_BENCHMARK_REPETITIONS_LT_3")
    locks = _verify_sha_locks()
    inputs = load_fixed_inputs(fixed_input_dir)
    output = _safe_result_dir(result_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cache_root = output / "working_cache"
    cases, replay = _load_audit_cases(inputs)
    selection_path = raw / "fixed_a_cache_audit_selection.csv"
    atomic_csv(selection_path, replay)
    _prepare_equivalence_runtime_imports()
    rng_before = _rng_snapshot()
    equivalence_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for ordinal, case in enumerate(cases):
        original_rgb, decoder, decode_seconds = _decode_rgb(
            case.image_path, case.image_sha256
        )
        if case.active:
            if case.theta is None:
                raise FixedACacheStop(f"STOP_FIXED_A_CASE_THETA:{case.case_id}")
            reference = _render_theta_exact(original_rgb, case.theta)
            reference_rgb = reference.rgb
            if reference.bgr_sha256 != case.expected_bgr_sha256:
                raise FixedACacheStop(f"STOP_FIXED_A_CASE_REFERENCE_SHA:{case.case_id}")
            if case.unique_rgb is not None and reference.unique_rgb != case.unique_rgb:
                raise FixedACacheStop(f"STOP_FIXED_A_CASE_PALETTE_COUNT:{case.case_id}")
        else:
            reference = None
            reference_rgb = np.ascontiguousarray(original_rgb.copy())
        if case.suite == "golden_6":
            if reference is None:
                raise FixedACacheStop(f"STOP_FIXED_A_GOLDEN_REFERENCE:{case.case_id}")
            _prepare_golden_audit_payloads(cache_root, case, reference)
        reference_bgr_sha = _formal_bgr_sha_from_rgb(reference_rgb)
        reference_rgb_sha = _array_bytes_sha(reference_rgb)
        reference_letterbox_sha = _letterbox_bgr_sha(
            np.ascontiguousarray(reference_rgb[:, :, ::-1])
        )
        signatures: dict[str, tuple[str, str, str, tuple[int, ...], str]] = {}
        for candidate in CANDIDATES:
            first_output: np.ndarray | None = None
            for repetition in range(1, int(repetitions) + 1):
                candidate_started = time.perf_counter()
                candidate_rgb = _candidate_output(
                    candidate=candidate,
                    case=case,
                    original_rgb=original_rgb,
                    cache_root=cache_root,
                )
                candidate_seconds = time.perf_counter() - candidate_started
                timing_rows.append(
                    {
                        "schema": "m1_native_a_fixed_cache_benchmark_call_v2",
                        "suite": case.suite,
                        "case_id": case.case_id,
                        "canonical_uid": case.canonical_uid,
                        "formal_visit_index": case.formal_visit_index,
                        "active": case.active,
                        "candidate": candidate,
                        "repetition": repetition,
                        "cold_warm_observation": (
                            COLD_OBSERVATION if repetition == 1 else WARM_OBSERVATION
                        ),
                        "decode_seconds_common": decode_seconds,
                        "mapping_or_transform_seconds": candidate_seconds,
                        "image_pixels": int(original_rgb.shape[0] * original_rgb.shape[1]),
                        "unique_rgb": case.unique_rgb,
                    }
                )
                if first_output is None:
                    first_output = candidate_rgb
            if first_output is None:
                raise FixedACacheStop("STOP_FIXED_A_NO_CANDIDATE_OUTPUT")
            candidate_bgr_sha = _formal_bgr_sha_from_rgb(first_output)
            candidate_letterbox_sha = _letterbox_bgr_sha(
                np.ascontiguousarray(first_output[:, :, ::-1])
            )
            signatures[candidate] = (
                _array_bytes_sha(first_output),
                candidate_bgr_sha,
                candidate_letterbox_sha,
                tuple(int(value) for value in first_output.shape),
                str(first_output.dtype),
            )
            if not np.array_equal(first_output, reference_rgb):
                mismatch_dir = raw / "fixed_a_cache_mismatch"
                mismatch_dir.mkdir(parents=True, exist_ok=True)
                safe = "".join(
                    character if character.isalnum() else "_" for character in case.case_id
                )
                atomic_npy(mismatch_dir / f"{safe}_REFERENCE.npy", reference_rgb)
                atomic_npy(mismatch_dir / f"{safe}_{candidate}.npy", first_output)
        strict = all(
            signature
            == (
                reference_rgb_sha,
                reference_bgr_sha,
                reference_letterbox_sha,
                tuple(int(value) for value in reference_rgb.shape),
                str(reference_rgb.dtype),
            )
            for signature in signatures.values()
        )
        equivalence_rows.append(
            {
                "schema": "m1_native_a_fixed_cache_equivalence_case_v1",
                "suite": case.suite,
                "case_id": case.case_id,
                "canonical_uid": case.canonical_uid,
                "formal_visit_index": case.formal_visit_index,
                "active": case.active,
                "decoder": decoder,
                "input_image_sha256": case.image_sha256,
                "unique_rgb": case.unique_rgb,
                "reference_rgb_array_sha256": reference_rgb_sha,
                "reference_formal_bgr_sha256": reference_bgr_sha,
                "reference_letterbox_bgr_array_sha256": reference_letterbox_sha,
                **{
                    f"{candidate}_rgb_array_sha256": signatures[candidate][0]
                    for candidate in CANDIDATES
                },
                **{
                    f"{candidate}_formal_bgr_sha256": signatures[candidate][1]
                    for candidate in CANDIDATES
                },
                **{
                    f"{candidate}_letterbox_bgr_array_sha256": signatures[candidate][2]
                    for candidate in CANDIDATES
                },
                "output_shape_exact": all(
                    signatures[candidate][3] == tuple(reference_rgb.shape)
                    for candidate in CANDIDATES
                ),
                "dtype_uint8_exact": all(
                    signatures[candidate][4] == "uint8" for candidate in CANDIDATES
                ),
                "pixel_array_exact": all(
                    signatures[candidate][0] == reference_rgb_sha for candidate in CANDIDATES
                ),
                "bgr_sha256_exact": all(
                    signatures[candidate][1] == reference_bgr_sha for candidate in CANDIDATES
                ),
                "letterbox_exact": all(
                    signatures[candidate][2] == reference_letterbox_sha
                    for candidate in CANDIDATES
                ),
                "strict_bit_equivalence": strict,
            }
        )
        if not strict:
            atomic_csv(output / "fixed_a_cache_equivalence.csv", pd.DataFrame(equivalence_rows))
            atomic_csv(raw / "fixed_a_cache_benchmark_per_call.csv", pd.DataFrame(timing_rows))
            raise FixedACacheStop(f"STOP_FIXED_A_BIT_EQUIVALENCE:{case.case_id}")
        if (ordinal + 1) % 16 == 0:
            atomic_csv(output / "fixed_a_cache_equivalence.csv", pd.DataFrame(equivalence_rows))
            atomic_csv(raw / "fixed_a_cache_benchmark_per_call.csv", pd.DataFrame(timing_rows))

    rng_after = _rng_snapshot()
    if rng_after != rng_before:
        raise FixedACacheStop("STOP_FIXED_A_EQUIVALENCE_RNG_CHANGED")
    equivalence = pd.DataFrame(equivalence_rows)
    timing = pd.DataFrame(timing_rows)
    if (
        len(equivalence.loc[equivalence["suite"].eq("golden_6")]) != 6
        or len(equivalence.loc[equivalence["suite"].eq("fixed_replay_240")])
        != EXPECTED_FIXED_REPLAY_VISITS
        or len(equivalence.loc[equivalence["suite"].eq("high_palette_16")])
        != HIGH_PALETTE_COUNT
        or not equivalence["strict_bit_equivalence"].all()
    ):
        raise FixedACacheStop("STOP_FIXED_A_EQUIVALENCE_FINAL_COUNTS")
    atomic_csv(output / "fixed_a_cache_equivalence.csv", equivalence)
    atomic_csv(raw / "fixed_a_cache_benchmark_per_call.csv", timing)

    aggregates: list[dict[str, Any]] = []
    for (suite, candidate), group in timing.groupby(["suite", "candidate"], sort=True):
        values = group["mapping_or_transform_seconds"].to_numpy(dtype=float)
        cold = group.loc[group["repetition"].astype(int).eq(1)]
        warm = group.loc[group["repetition"].astype(int).ge(2)]
        case_count = int(group["case_id"].nunique())
        if len(cold) != case_count or len(warm) != case_count * (int(repetitions) - 1):
            raise FixedACacheStop(
                f"STOP_FIXED_A_COLD_WARM_CARDINALITY:{suite}:{candidate}"
            )

        def throughput(frame: pd.DataFrame) -> dict[str, float | int]:
            wall = float(frame["mapping_or_transform_seconds"].sum())
            count = int(len(frame))
            pixels = int(frame["image_pixels"].sum())
            denominator = max(wall, np.finfo(float).tiny)
            return {
                "call_count": count,
                "wall_seconds": wall,
                "views_per_second": float(count / denominator),
                "pixels_per_second": float(pixels / denominator),
            }

        cold_all = throughput(cold)
        warm_all = throughput(warm)
        cold_active = throughput(cold.loc[cold["active"]])
        warm_active = throughput(warm.loc[warm["active"]])
        lookup_candidate = candidate in (BGR_CACHE, PALETTE_CACHE)
        aggregates.append(
            {
                "schema": "m1_native_a_fixed_cache_benchmark_aggregate_v2",
                "scope_suite": suite,
                "candidate": candidate,
                "lookup_candidate": lookup_candidate,
                "repetitions": int(repetitions),
                "case_count": case_count,
                "active_case_count": int(group.loc[group["active"], "case_id"].nunique()),
                "call_count": len(group),
                "decode_seconds_common_sum_once_per_case": float(
                    group.drop_duplicates("case_id")["decode_seconds_common"].sum()
                ),
                "mapping_or_transform_wall_seconds": float(values.sum()),
                "mean_seconds": float(np.mean(values)),
                "p50_seconds": float(np.quantile(values, 0.5)),
                "p90_seconds": float(np.quantile(values, 0.9)),
                "cold_call_count": cold_all["call_count"],
                "cold_wall_seconds": cold_all["wall_seconds"],
                "cold_views_per_second": cold_all["views_per_second"],
                "cold_pixels_per_second": cold_all["pixels_per_second"],
                "warm_call_count": warm_all["call_count"],
                "warm_wall_seconds": warm_all["wall_seconds"],
                "warm_views_per_second": warm_all["views_per_second"],
                "warm_pixels_per_second": warm_all["pixels_per_second"],
                "cold_active_path_call_count": cold_active["call_count"],
                "cold_active_path_wall_seconds": cold_active["wall_seconds"],
                "cold_active_path_views_per_second": cold_active["views_per_second"],
                "cold_active_path_pixels_per_second": cold_active["pixels_per_second"],
                "warm_active_path_call_count": warm_active["call_count"],
                "warm_active_path_wall_seconds": warm_active["wall_seconds"],
                "warm_active_path_views_per_second": warm_active["views_per_second"],
                "warm_active_path_pixels_per_second": warm_active["pixels_per_second"],
                "cold_active_lookup_call_count": (
                    cold_active["call_count"] if lookup_candidate else 0
                ),
                "cold_active_lookup_wall_seconds": (
                    cold_active["wall_seconds"] if lookup_candidate else 0.0
                ),
                "cold_active_lookup_views_per_second": (
                    cold_active["views_per_second"] if lookup_candidate else np.nan
                ),
                "cold_active_lookup_pixels_per_second": (
                    cold_active["pixels_per_second"] if lookup_candidate else np.nan
                ),
                "warm_active_lookup_call_count": (
                    warm_active["call_count"] if lookup_candidate else 0
                ),
                "warm_active_lookup_wall_seconds": (
                    warm_active["wall_seconds"] if lookup_candidate else 0.0
                ),
                "warm_active_lookup_views_per_second": (
                    warm_active["views_per_second"] if lookup_candidate else np.nan
                ),
                "warm_active_lookup_pixels_per_second": (
                    warm_active["pixels_per_second"] if lookup_candidate else np.nan
                ),
                "cold_warm_rule": "repetition 1 cold; repetitions 2..N warm per case/candidate",
                "os_page_cache_caveat": COLD_WARM_OS_CACHE_CAVEAT,
                "all_bit_equivalent": True,
            }
        )
    benchmark = pd.DataFrame(aggregates)
    theta_by_suite = benchmark.loc[benchmark["candidate"].eq(THETA_ONLY)].set_index(
        "scope_suite"
    )["mapping_or_transform_wall_seconds"]
    benchmark["speedup_vs_THETA_ONLY"] = [
        float(theta_by_suite.loc[row["scope_suite"]])
        / max(float(row["mapping_or_transform_wall_seconds"]), np.finfo(float).tiny)
        for row in benchmark.to_dict("records")
    ]
    atomic_csv(output / "fixed_a_cache_benchmark.csv", benchmark)
    storage_estimates = _estimated_cache_bytes(inputs, CANDIDATES)
    active_theta = inputs.theta.loc[
        inputs.theta["fixed_delta_exists"].map(
            lambda value: _strict_bool(value, "summary.fixed_delta_exists")
        )
    ]
    max_unique_rgb = int(active_theta[inputs.unique_rgb_column].astype("int64").max())
    palette_bounded_extra_bytes = int(
        max_unique_rgb * (4 + 6)
        + PALETTE_REMAP_CHUNK_PIXELS
        * (4 + np.dtype(np.intp).itemsize + 3)
    )
    equivalence_path = output / "fixed_a_cache_equivalence.csv"
    benchmark_path = output / "fixed_a_cache_benchmark.csv"
    per_call_path = raw / "fixed_a_cache_benchmark_per_call.csv"
    summary = {
        "schema": "m1_native_a_fixed_cache_equivalence_benchmark_v2",
        "status": "PASS",
        "fixed_contract_sha256": FIXED_CONTRACT_SHA256,
        "sha_locks": locks,
        "fixed_input_sha256": dict(inputs.file_sha256),
        "selection": {
            "rule": (
                "ascending SHA256(M1_NATIVE_A_FIXED_CACHE_REPLAY_SELECTION_V1|"
                "formal_visit_index|canonical_uid), first 240; label/performance independent"
            ),
            "fixed_replay_rows": EXPECTED_FIXED_REPLAY_VISITS,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "golden_rows": 6,
            "high_palette_rows": HIGH_PALETTE_COUNT,
            "high_palette_rule": "descending unique_rgb_count then canonical_uid",
        },
        "equivalence_rows": len(equivalence),
        "strict_pass_rows": int(equivalence["strict_bit_equivalence"].sum()),
        "benchmark_repetitions": int(repetitions),
        "artifact_schemas": {
            "fixed_a_cache_equivalence.csv": "m1_native_a_fixed_cache_equivalence_case_v1",
            "fixed_a_cache_benchmark.csv": "m1_native_a_fixed_cache_benchmark_aggregate_v2",
            "raw/fixed_a_cache_benchmark_per_call.csv": "m1_native_a_fixed_cache_benchmark_call_v2",
            "raw/fixed_a_cache_audit_selection.csv": "m1_native_a_fixed_cache_audit_selection_v1",
        },
        "storage_and_working_set": {
            "active_fixed_output_entries": EXPECTED_FIXED_SOLVES,
            "theta_only_payload_bytes": storage_estimates[THETA_ONLY],
            "preletterbox_bgr_raw_payload_bytes": storage_estimates[BGR_CACHE],
            "palette_source_plus_deployment_raw_payload_bytes": storage_estimates[
                PALETTE_CACHE
            ],
            "npy_header_and_sidecar_bytes_excluded_from_raw_estimates": True,
            "max_unique_rgb": max_unique_rgb,
            "palette_remap_bounded_extra_working_bytes_upper_estimate": (
                palette_bounded_extra_bytes
            ),
            "palette_remap_working_estimate_components": (
                "4*U source packed + 6*U source/deployment copies + chunk*(4 packed + "
                f"{np.dtype(np.intp).itemsize} search indices + 3 output)"
            ),
            "full_inverse_cached": False,
            "working_cache_excluded_from_core_zip": True,
        },
        "cold_warm_lookup_throughput_definition": {
            "frozen_before_measurement": True,
            "cold_observation": COLD_OBSERVATION,
            "cold_rule": "repetition == 1 independently for each case and candidate",
            "warm_observation": WARM_OBSERVATION,
            "warm_rule": "repetition >= 2 independently for each case and candidate",
            "reported_units": ["call_count", "wall_seconds", "views_per_second", "pixels_per_second"],
            "reported_scopes": ["all cases", "active transform/cache-lookup cases"],
            "lookup_candidates": [BGR_CACHE, PALETTE_CACHE],
            "theta_only_is_reference_transform_not_lookup": True,
            "os_page_cache_caveat": COLD_WARM_OS_CACHE_CAVEAT,
            "physical_cold_disk_claimed": False,
        },
        "palette_remap": {
            "source_order": "strict ascending packed 24-bit RGB inherited from np.unique",
            "duplicate_color_semantics": "all equal source colors map to one exact deployment row",
            "tie_semantics": "np.searchsorted(side=left) plus exact packed-value equality",
            "chunk_pixels": PALETTE_REMAP_CHUNK_PIXELS,
            "full_inverse_cached": False,
            "bounded_working_set": True,
        },
        "golden_payload_isolation": {
            "namespace": "working_cache/golden_audit_only/<exact image+theta+output key>",
            "purpose": "HISTORICAL_GOLDEN_EXACT_SOLVE_AUDIT_ONLY_NOT_FIXED_A_CACHE_REUSE",
            "fixed_a_per_image_cache_collision_possible": False,
            "payload_count_per_candidate": 6,
            "excluded_from_core_zip": True,
        },
        "rng_before": rng_before,
        "rng_after": rng_after,
        "rng_unchanged": True,
        "wall_seconds": time.perf_counter() - started,
        "outputs": {
            "equivalence": str(equivalence_path.resolve()),
            "equivalence_sha256": sha256_file(equivalence_path),
            "benchmark": str(benchmark_path.resolve()),
            "benchmark_sha256": sha256_file(benchmark_path),
            "raw_per_call": str(per_call_path.resolve()),
            "raw_per_call_sha256": sha256_file(per_call_path),
        },
        "cache_fallback_calls": 0,
        "online_powell_calls": 0,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "dataset_mutations": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "SCIENCE_CHANGE": False,
    }
    atomic_json(output / "fixed_a_cache_equivalence_summary.json", summary)
    return summary


def _resource_patch_audit() -> dict[str, Any]:
    if sha256_file(RESOURCE_WORKER) != RESOURCE_WORKER_SHA256:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_WORKER_SHA")
    source = RESOURCE_WORKER.read_text(encoding="utf-8")
    required_call_points = {
        "dataset_constructor_global": "dataset = RouteATargetDataset(",
        "policy_loader_global": "policy_cache = load_policy_cache(policy_cache_dir, branch)",
        "exact_transform_global": "apply_theta_training_rgb_uint8(original_rgb, entry.theta)",
    }
    missing = [name for name, text in required_call_points.items() if text not in source]
    if missing:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_PATCH_POINT:" + ",".join(missing))
    tree = ast.parse(source)
    return {
        "resource_worker_path": str(RESOURCE_WORKER.resolve()),
        "resource_worker_sha256": RESOURCE_WORKER_SHA256,
        "formal_resource_runner_path": str(FORMAL_RESOURCE_RUNNER.resolve()),
        "formal_resource_runner_sha256": FORMAL_RESOURCE_RUNNER_SHA256,
        "formal_resource_runner_modified": False,
        "patch_objects": ["RouteATargetDataset", "load_policy_cache"],
        "patch_scope": "fresh disposable resource-worker process only",
        "required_call_points": required_call_points,
        "source_ast_node_count": sum(1 for _ in ast.walk(tree)),
        "transform_dispatch": (
            "base RouteATargetDataset exact_fast call is intercepted only inside "
            "FixedATargetDataset.__getitem__; cache miss raises and never calls theta"
        ),
        "fallback_available": False,
    }


def materialize_resource_policy(
    *, output_dir: Path, inputs: FixedInputs
) -> dict[str, Any]:
    output = _safe_result_dir(output_dir)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    schedule_path = raw / "visit_policy_schedule.parquet"
    theta_path = raw / "visit_offline_theta_cache.parquet"
    audit_path = output / "fixed_a_resource_cache_materialization.json"
    for path in (schedule_path, theta_path, audit_path):
        if path.exists():
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_OUTPUT_EXISTS:{path}")

    native = pd.read_parquet(NATIVE_SCHEDULE)
    native = native.loc[native["branch"].astype(str).eq("NATIVE_A")].copy()
    if len(native) != EXPECTED_VISITS:
        raise FixedACacheStop("STOP_FIXED_A_NATIVE_SCHEDULE_COUNT")
    native["formal_visit_index"] = (
        (native["epoch"].astype(int) - 72) * 4416
        + native["target_visit_ordinal"].astype(int)
    )
    if native["formal_visit_index"].duplicated().any():
        raise FixedACacheStop("STOP_FIXED_A_NATIVE_SCHEDULE_INDEX_DUPLICATE")
    native_by_index = native.set_index("formal_visit_index", verify_integrity=True)
    exposure_by_index = inputs.exposure.set_index(
        "formal_visit_index", verify_integrity=True
    )
    selection_by_uid = inputs.selection_by_uid
    theta_by_uid = inputs.theta_by_uid

    schedule_rows: list[dict[str, Any]] = []
    theta_rows: list[dict[str, Any]] = []
    active_count = 0
    for formal_index in range(EXPECTED_RESOURCE_VISITS):
        native_row = native_by_index.loc[formal_index].to_dict()
        exposure_row = exposure_by_index.loc[formal_index]
        uid = str(exposure_row["canonical_uid"])
        if str(native_row["q2_uid"]) != uid:
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_Q2_UID:{formal_index}")
        if str(native_row["schedule_uid"]) != str(exposure_row["schedule_uid"]):
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_SCHEDULE_UID:{formal_index}")
        active = _strict_bool(exposure_row["fixed_active"], "resource.fixed_active")
        if active != _strict_bool(exposure_row["resample_active"], "resource.resample_active"):
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_EXPOSURE:{formal_index}")
        output_row = {
            column: native_row[column]
            for column in (
                "adaptation_seed",
                "epoch",
                "target_visit_ordinal",
                "schedule_uid",
                "q2_uid",
                "branch",
                "augment_on",
                "valid",
                "policy_flag",
                "failure_reason",
                "delta_sha256",
                *(f"delta_{index}" for index in range(7)),
                "donor_schedule_uid",
                "donor_q2_uid",
            )
        }
        if active:
            selected = selection_by_uid.loc[uid]
            selected_index = int(selected["selected_formal_visit_index"])
            selected_native = native_by_index.loc[selected_index]
            if not (
                _strict_bool(selected_native["augment_on"], "selected.augment_on")
                and _strict_bool(selected_native["valid"], "selected.valid")
                and str(selected_native["policy_flag"]) == "APPLY_OFFLINE_THETA"
            ):
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_SELECTED_STATE:{uid}")
            selected_sha = str(selected["selected_delta_sha"])
            if selected_sha != str(selected_native["delta_sha256"]):
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_SELECTED_DELTA_SHA:{uid}")
            output_row["delta_sha256"] = selected_sha
            for index in range(7):
                left = np.asarray([selected[f"selected_delta_{index}"]], dtype="<f8")
                right = np.asarray([selected_native[f"delta_{index}"]], dtype="<f8")
                if left.tobytes() != right.tobytes():
                    raise FixedACacheStop(
                        f"STOP_FIXED_A_RESOURCE_SELECTED_DELTA_VALUE:{uid}:{index}"
                    )
                output_row[f"delta_{index}"] = float(left[0])
            output_row["donor_schedule_uid"] = selected_native["donor_schedule_uid"]
            output_row["donor_q2_uid"] = selected_native["donor_q2_uid"]
            active_count += 1
        schedule_rows.append(output_row)

        realization = theta_by_uid.loc[uid]
        theta_row = dict(output_row)
        if active:
            if not _strict_bool(realization["fixed_delta_exists"], "resource.theta_exists"):
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_THETA_MISSING:{uid}")
            theta_row.update(
                {
                    column: float(realization[column]) for column in THETA_COLUMNS
                }
            )
            theta_row.update(
                {
                    "input_image_sha256": str(realization["input_image_sha256"]),
                    "output_array_sha256": str(realization["output_array_sha256"]),
                    "actual_effective_dose": float(realization["actual_effective_dose"]),
                    "tone_preclip_pixel_fraction": float(
                        realization["tone_preclip_pixel_fraction"]
                    ),
                    "lab_to_srgb_preclip_pixel_fraction": float(
                        realization["lab_to_srgb_preclip_pixel_fraction"]
                    ),
                }
            )
        else:
            theta_row.update(
                {
                    **{column: np.nan for column in THETA_COLUMNS},
                    "input_image_sha256": None,
                    "output_array_sha256": None,
                    "actual_effective_dose": np.nan,
                    "tone_preclip_pixel_fraction": np.nan,
                    "lab_to_srgb_preclip_pixel_fraction": np.nan,
                }
            )
        theta_rows.append(theta_row)

    schedule = pd.DataFrame(schedule_rows)
    theta = pd.DataFrame(theta_rows)
    if len(schedule) != EXPECTED_RESOURCE_VISITS or len(theta) != EXPECTED_RESOURCE_VISITS:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CACHE_COUNT")
    schedule.to_parquet(schedule_path, index=False)
    theta.to_parquet(theta_path, index=False)
    audit = {
        "schema": "m1_native_a_fixed_resource_cache_materialization_v1",
        "status": "PASS",
        "row_count": EXPECTED_RESOURCE_VISITS,
        "active_count": active_count,
        "identity_count": EXPECTED_RESOURCE_VISITS - active_count,
        "source_native_schedule_path": str(NATIVE_SCHEDULE.resolve()),
        "source_native_schedule_sha256": NATIVE_SCHEDULE_SHA256,
        "schedule_path": str(schedule_path.resolve()),
        "schedule_sha256": sha256_file(schedule_path),
        "cache_path": str(theta_path.resolve()),
        "cache_sha256": sha256_file(theta_path),
        "fixed_input_sha256": dict(inputs.file_sha256),
        "online_powell_calls": 0,
        "resource_subset_rows": EXPECTED_RESOURCE_VISITS,
        "cache_fallback_calls": 0,
    }
    atomic_json(audit_path, audit)
    return audit


def _custom_resource_policy_cache(theta_path: Path, branch: str) -> Any:
    # This function is called only inside the isolated resource-worker child.
    module = globals().get("_RESOURCE_MODULE")
    if module is None:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_MODULE_UNAVAILABLE")
    theta_path = Path(theta_path).resolve()
    schedule_path = theta_path.parent / "visit_policy_schedule.parquet"
    audit_path = theta_path.parent.parent / "fixed_a_resource_cache_materialization.json"
    if not theta_path.is_file() or not schedule_path.is_file() or not audit_path.is_file():
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CACHE_FILES_MISSING")
    schedule_sha = sha256_file(schedule_path)
    theta_sha = sha256_file(theta_path)
    audit_sha = sha256_file(audit_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status") != "PASS"
        or Path(str(audit.get("schedule_path"))).resolve() != schedule_path
        or Path(str(audit.get("cache_path"))).resolve() != theta_path
        or audit.get("schedule_sha256") != schedule_sha
        or audit.get("cache_sha256") != theta_sha
        or int(audit.get("row_count", -1)) != EXPECTED_RESOURCE_VISITS
        or int(audit.get("online_powell_calls", -1)) != 0
    ):
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CACHE_AUDIT")
    schedule = pd.read_parquet(schedule_path)
    theta = pd.read_parquet(theta_path)
    module._assert_columns(
        list(schedule.columns), module.POLICY_REQUIRED_COLUMNS, "fixed_a_resource_schedule"
    )
    module._assert_columns(
        list(theta.columns), module.THETA_REQUIRED_COLUMNS, "fixed_a_resource_theta"
    )
    if len(schedule) != EXPECTED_RESOURCE_VISITS or len(theta) != EXPECTED_RESOURCE_VISITS:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_POLICY_ROW_COUNT")
    if schedule["target_visit_ordinal"].astype(int).tolist() != list(
        range(EXPECTED_RESOURCE_VISITS)
    ):
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_POLICY_ORDINALS")
    if not schedule["branch"].astype(str).eq("NATIVE_A").all():
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_POLICY_BRANCH")

    whitelist = pd.read_csv(Q2_WHITELIST, dtype=str).set_index("uid", verify_integrity=True)
    theta_by_ordinal = theta.set_index(
        "target_visit_ordinal", drop=False, verify_integrity=True
    )
    entries: dict[int, Any] = {}
    crosslocks = 0
    active_inputs = 0
    for row in schedule.to_dict("records"):
        ordinal = int(row["target_visit_ordinal"])
        theta_row = theta_by_ordinal.loc[ordinal]
        uid = str(row["q2_uid"])
        expected_input_sha = str(whitelist.loc[uid, "image_sha256"]).lower()
        requested = _strict_bool(row["augment_on"], "resource.schedule.augment_on")
        valid = _strict_bool(row["valid"], "resource.schedule.valid")
        policy_flag = _optional_text(row["policy_flag"])
        failure_reason = _optional_text(row["failure_reason"])
        delta = tuple(float(row[f"delta_{index}"]) for index in range(7))
        delta_sha = _optional_text(row["delta_sha256"])
        donor_schedule = _optional_text(row["donor_schedule_uid"])
        donor_q2 = _optional_text(row["donor_q2_uid"])
        active = module._validate_policy_row_state(
            schedule_branch="NATIVE_A",
            requested=requested,
            valid=valid,
            policy_flag=policy_flag,
            failure_reason=failure_reason,
            delta=delta,
            delta_sha256=delta_sha,
            donor_schedule_uid=donor_schedule,
            donor_q2_uid=donor_q2,
        )
        for field in module.POLICY_REQUIRED_COLUMNS:
            left = row[field]
            right = theta_row[field]
            if isinstance(left, (float, np.floating)) or isinstance(
                right, (float, np.floating)
            ):
                left_array = np.asarray([left], dtype="<f8")
                right_array = np.asarray([right], dtype="<f8")
                if left_array.tobytes() != right_array.tobytes():
                    raise FixedACacheStop(
                        f"STOP_FIXED_A_RESOURCE_POLICY_CROSSLOCK:{ordinal}:{field}"
                    )
            elif _optional_text(left) != _optional_text(right):
                raise FixedACacheStop(
                    f"STOP_FIXED_A_RESOURCE_POLICY_CROSSLOCK:{ordinal}:{field}"
                )
        crosslocks += 1
        if branch == "CONTROL":
            entry = module.PolicyEntry(
                adaptation_seed=int(row["adaptation_seed"]),
                epoch=int(row["epoch"]),
                target_visit_ordinal=ordinal,
                schedule_uid=str(row["schedule_uid"]),
                q2_uid=uid,
                branch="CONTROL",
                augment_on=False,
                valid=False,
                policy_flag="CONTROL_EXACT_PASSTHROUGH",
                failure_reason="",
                delta_sha256=canonical_json_sha256([0.0] * 7),
                delta=(0.0,) * 7,
                donor_schedule_uid="",
                donor_q2_uid="",
                theta=None,
                input_image_sha256=expected_input_sha,
                output_array_sha256="",
                actual_effective_dose=0.0,
                tone_preclip_pixel_fraction=0.0,
                lab_to_srgb_preclip_pixel_fraction=0.0,
            )
        elif branch == "NATIVE_A":
            theta_value: tuple[float, float, float] | None = None
            output_sha = ""
            actual_dose = 0.0
            tone_clip = 0.0
            gamut_clip = 0.0
            if active:
                theta_value = tuple(float(theta_row[column]) for column in THETA_COLUMNS)
                if not all(math.isfinite(value) for value in theta_value):
                    raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_THETA:{ordinal}")
                if str(theta_row["input_image_sha256"]).lower() != expected_input_sha:
                    raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_INPUT_SHA:{ordinal}")
                output_sha = _require_sha256(
                    theta_row["output_array_sha256"], f"resource.output:{ordinal}"
                )
                actual_dose = float(theta_row["actual_effective_dose"])
                tone_clip = float(theta_row["tone_preclip_pixel_fraction"])
                gamut_clip = float(theta_row["lab_to_srgb_preclip_pixel_fraction"])
                active_inputs += 1
            else:
                for field in (*THETA_COLUMNS, "output_array_sha256"):
                    if not pd.isna(theta_row[field]) and _optional_text(theta_row[field]):
                        raise FixedACacheStop(
                            f"STOP_FIXED_A_RESOURCE_IDENTITY_VALUE:{ordinal}:{field}"
                        )
            entry = module.PolicyEntry(
                adaptation_seed=int(row["adaptation_seed"]),
                epoch=int(row["epoch"]),
                target_visit_ordinal=ordinal,
                schedule_uid=str(row["schedule_uid"]),
                q2_uid=uid,
                branch="NATIVE_A",
                augment_on=requested,
                valid=valid,
                policy_flag=policy_flag,
                failure_reason=failure_reason,
                delta_sha256=delta_sha,
                delta=delta,
                donor_schedule_uid=donor_schedule,
                donor_q2_uid=donor_q2,
                theta=theta_value,
                input_image_sha256=expected_input_sha,
                output_array_sha256=output_sha,
                actual_effective_dose=actual_dose,
                tone_preclip_pixel_fraction=tone_clip,
                lab_to_srgb_preclip_pixel_fraction=gamut_clip,
            )
        else:
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_BRANCH:{branch}")
        entries[ordinal] = entry
    if sha256_file(schedule_path) != schedule_sha or sha256_file(theta_path) != theta_sha:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CACHE_CHANGED_DURING_READ")
    return module.PolicyCache(
        directory=theta_path.parent,
        schedule_path=schedule_path,
        theta_path=theta_path,
        schedule_sha256=schedule_sha,
        theta_sha256=theta_sha,
        precompute_audit_path=audit_path,
        precompute_audit_sha256=audit_sha,
        q2_target_whitelist_sha256=Q2_WHITELIST_SHA256,
        theta_policy_crosslock_row_count=crosslocks,
        active_theta_input_sha_crosscheck_count=active_inputs,
        branch=branch,
        entries=entries,
    )


# The actual class is defined only in explicitly marked isolated resource
# subprocesses.  This keeps --help/self-check free of CUDA/training imports,
# while DataLoader spawn children import the same stable module/class name.
_RESOURCE_MODULE: Any | None = None
if os.environ.get("M1_FIXED_A_RESOURCE_CHILD") == "1":
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import m1_routea_semantic_resource_worker_v1 as _RESOURCE_MODULE

    class FixedATargetDataset(_RESOURCE_MODULE.RouteATargetDataset):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            candidate = os.environ.get("M1_FIXED_A_CANDIDATE", "")
            if candidate not in CANDIDATES:
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_CANDIDATE:{candidate}")
            self.fixed_a_candidate = candidate
            self.fixed_a_cache_root = Path(
                os.environ.get("M1_FIXED_A_CACHE_ROOT", "")
            ).resolve()

        def __getitem__(self, token: Any) -> dict[str, Any]:
            token_value = _RESOURCE_MODULE.TargetScheduleToken(*token)
            ordinal = (
                int(token_value.batch_index) * EXPECTED_RESOURCE_TARGET_BATCH
                + int(token_value.position)
            )
            try:
                entry = self.policy_entries[ordinal]
            except KeyError as exc:
                raise FixedACacheStop("STOP_FIXED_A_RESOURCE_ENTRY_MISSING") from exc
            if self.branch == "CONTROL" or self.fixed_a_candidate == THETA_ONLY:
                return super().__getitem__(token)
            if not entry.apply_transform:
                return super().__getitem__(token)
            if entry.theta is None:
                raise FixedACacheStop("STOP_FIXED_A_RESOURCE_ACTIVE_THETA_MISSING")

            paths = _payload_paths(self.fixed_a_cache_root, entry.input_image_sha256)
            sidecar = _load_sidecar(paths["meta"])
            theta_value = np.asarray(entry.theta, dtype="<f8")
            theta_sha = hashlib.sha256(
                THETA_HASH_PREFIX + theta_value.tobytes(order="C")
            ).hexdigest()
            for field, expected in (
                ("input_image_sha256", entry.input_image_sha256),
                ("theta_raw_f8_sha256", theta_sha),
                ("expected_output_bgr_sha256", entry.output_array_sha256),
            ):
                if str(sidecar.get(field, "")) != str(expected):
                    raise FixedACacheStop(
                        f"STOP_FIXED_A_RESOURCE_SIDECAR_CROSSLOCK:{ordinal}:{field}"
                    )

            original_function = _RESOURCE_MODULE.apply_theta_training_rgb_uint8

            def exact_cache_dispatch(
                original_rgb: np.ndarray, theta_argument: Sequence[float]
            ) -> tuple[np.ndarray, str]:
                argument = np.asarray(theta_argument, dtype="<f8")
                if argument.tobytes(order="C") != theta_value.tobytes(order="C"):
                    raise FixedACacheStop("STOP_FIXED_A_RESOURCE_THETA_ARGUMENT_DRIFT")
                if self.fixed_a_candidate == BGR_CACHE:
                    output_rgb = _load_bgr_candidate(
                        self.fixed_a_cache_root,
                        AuditCase(
                            suite="resource",
                            case_id=str(ordinal),
                            canonical_uid=entry.q2_uid,
                            image_path=Path("."),
                            image_sha256=entry.input_image_sha256,
                            active=True,
                            theta=argument,
                            expected_bgr_sha256=entry.output_array_sha256,
                            formal_visit_index=ordinal,
                            unique_rgb=None,
                        ),
                    )
                elif self.fixed_a_candidate == PALETTE_CACHE:
                    output_rgb = _load_palette_candidate(
                        self.fixed_a_cache_root,
                        AuditCase(
                            suite="resource",
                            case_id=str(ordinal),
                            canonical_uid=entry.q2_uid,
                            image_path=Path("."),
                            image_sha256=entry.input_image_sha256,
                            active=True,
                            theta=argument,
                            expected_bgr_sha256=entry.output_array_sha256,
                            formal_visit_index=ordinal,
                            unique_rgb=None,
                        ),
                        np.ascontiguousarray(original_rgb),
                    )
                else:
                    raise FixedACacheStop("STOP_FIXED_A_RESOURCE_DISPATCH_FALLBACK_FORBIDDEN")
                reported = _formal_bgr_sha_from_rgb(output_rgb)
                if reported != entry.output_array_sha256:
                    raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CACHE_OUTPUT_SHA")
                return output_rgb, reported

            # One DataLoader worker invokes one __getitem__ at a time.  The
            # global is restored in finally; no fallback is invoked for cache
            # candidates, and every child independently performs this patch.
            _RESOURCE_MODULE.apply_theta_training_rgb_uint8 = exact_cache_dispatch
            try:
                return super().__getitem__(token)
            finally:
                _RESOURCE_MODULE.apply_theta_training_rgb_uint8 = original_function

else:

    class FixedATargetDataset:  # pragma: no cover - inert self-check placeholder
        pass


def _run_resource_worker_child(args: argparse.Namespace) -> int:
    if os.environ.get("M1_FIXED_A_RESOURCE_CHILD") != "1" or _RESOURCE_MODULE is None:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CHILD_ENV")
    _resource_patch_audit()
    _RESOURCE_MODULE.RouteATargetDataset = FixedATargetDataset
    _RESOURCE_MODULE.load_policy_cache = _custom_resource_policy_cache
    branch = "CONTROL" if args.resource_role == "CONTROL" else "NATIVE_A"
    result = _RESOURCE_MODULE.run_worker(
        branch=branch,
        policy_cache_dir=Path(args.policy_cache),
        output_dir=Path(args.output_dir),
        contract_path=SEMANTIC_RESOURCE_CONTRACT,
        device=str(args.device),
        transform_impl="exact_fast",
    )
    if result.get("pass") is not True:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_WORKER_PASS_FALSE")
    return 0


def _launch_resource_subprocess(
    *,
    candidate: str,
    role: str,
    output_dir: Path,
    cache_root: Path,
    device: str,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "M1_FIXED_A_RESOURCE_CHILD": "1",
            "M1_FIXED_A_CANDIDATE": candidate,
            "M1_FIXED_A_CACHE_ROOT": str(cache_root.resolve()),
            "PYTHONHASHSEED": "42",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPT_DIR), str(ROOT / "project" / "ultralytics"), existing_pythonpath]
    ).rstrip(os.pathsep)
    code = (
        f"import sys;sys.path.insert(0,{str(SCRIPT_DIR)!r});"
        "import m1_native_a_fixed_cache_resource_v1 as m;"
        "raise SystemExit(m.main())"
    )
    command = [
        sys.executable,
        "-B",
        "-c",
        code,
        "resource-worker",
        "--candidate",
        candidate,
        "--resource-role",
        role,
        "--output-dir",
        str(output_dir),
        "--policy-cache",
        str(output_dir / "raw" / "visit_offline_theta_cache.parquet"),
        "--device",
        str(device),
    ]
    log_path = output_dir / "resource_subprocess.log"
    started = time.perf_counter()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        observed_peak_rss = 0
        observed_read_bytes = 0
        observed_write_bytes = 0
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                family = [monitor, *monitor.children(recursive=True)]
                observed_peak_rss = max(
                    observed_peak_rss,
                    sum(item.memory_info().rss for item in family if item.is_running()),
                )
                family_read_bytes = 0
                family_write_bytes = 0
                for item in family:
                    try:
                        io = item.io_counters()
                        family_read_bytes += int(io.read_bytes)
                        family_write_bytes += int(io.write_bytes)
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        continue
                observed_read_bytes = max(observed_read_bytes, family_read_bytes)
                observed_write_bytes = max(observed_write_bytes, family_write_bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.2)
        returncode = int(process.wait())
    record = {
        "schema": "m1_native_a_fixed_resource_subprocess_record_v1",
        "candidate": candidate,
        "role": role,
        "command": command,
        "returncode": returncode,
        "wall_seconds": time.perf_counter() - started,
        "observed_peak_process_tree_rss_bytes": observed_peak_rss,
        "observed_process_read_bytes": observed_read_bytes,
        "observed_process_write_bytes": observed_write_bytes,
        "observed_peak_process_tree_read_bytes": observed_read_bytes,
        "observed_peak_process_tree_write_bytes": observed_write_bytes,
        "log_path": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "oom": False,
    }
    text = log_path.read_text(encoding="utf-8", errors="replace").casefold()
    record["oom"] = any(
        marker in text
        for marker in ("cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed")
    )
    record["status"] = (
        "PASS" if returncode == 0 and not record["oom"] else "FAILED_CLOSED"
    )
    atomic_json(output_dir / "resource_subprocess_record.json", record)
    if returncode != 0 or record["oom"]:
        raise FixedACacheStop(
            f"STOP_FIXED_A_RESOURCE_SUBPROCESS:{candidate}:{role}:{returncode}:oom={record['oom']}"
        )
    return record


def _resource_worker_output(directory: Path, role: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    branch = "CONTROL" if role == "CONTROL" else "NATIVE_A"
    raw = directory / "raw"
    audit_path = raw / f"resource_worker_{branch}.json"
    micro_path = raw / f"resource_per_microbatch_{branch}.csv"
    window_path = raw / f"resource_per_optimizer_window_{branch}.csv"
    if not all(path.is_file() for path in (audit_path, micro_path, window_path)):
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_OUTPUT_MISSING:{directory}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    micro = pd.read_csv(micro_path, float_precision="round_trip")
    windows = pd.read_csv(window_path, float_precision="round_trip")
    if audit.get("pass") is not True or len(micro) != 10 or len(windows) != 5:
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_OUTPUT_SCHEMA:{directory}")
    if not all(bool(value) for value in audit.get("checks", {}).values()):
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_WORKER_CHECK:{directory}")
    if (
        int(audit["transform_execution"].get("online_powell_call_count", -1)) != 0
        or int(windows["found_inf"].sum()) != 0
        or int(windows["skipped_step"].sum()) != 0
        or audit.get("checkpoint_writes_zero") is not True
    ):
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_ZERO_FLAG:{directory}")
    data = audit.get("data_access", {})
    if any(
        int(data.get(field, -1)) != 0
        for field in ("target_adapt_label_reads", "target_dev_reads", "target_test_reads")
    ):
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_FORBIDDEN_READ:{directory}")
    return audit, micro, windows


def _frames_exact(
    left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str], role: str
) -> None:
    missing = [column for column in columns if column not in left or column not in right]
    if missing:
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_EXACT_COLUMNS:{role}:" + ",".join(missing))
    if len(left) != len(right):
        raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_EXACT_ROW_COUNT:{role}")
    for column in columns:
        a = left[column].to_numpy()
        b = right[column].to_numpy()
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            if not np.array_equal(a, b, equal_nan=True):
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_EXACT_NUMERIC:{role}:{column}")
        else:
            left_values = ["" if pd.isna(value) else str(value) for value in a]
            right_values = ["" if pd.isna(value) else str(value) for value in b]
            if left_values != right_values:
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_EXACT_TEXT:{role}:{column}")


def _resampled_a_frozen_resource_reference(output: Path | None) -> dict[str, Any]:
    """Read and SHA-crosslock the already measured Resampled Native-A window."""

    comparison = pd.read_csv(
        RESAMPLED_RESOURCE_COMPARISON, float_precision="round_trip"
    )
    selected = comparison.loc[
        comparison["stage"].astype(str).eq("after")
        & comparison["branch"].astype(str).eq("NATIVE_A")
        & comparison["transform_impl"].astype(str).eq("exact_fast")
    ]
    if len(selected) != 1:
        raise FixedACacheStop("STOP_FIXED_A_RESAMPLED_REFERENCE_ROW")
    row = selected.iloc[0]
    exact_expected = {
        "measured_windows": 3,
        "target_visits": 80,
        "transformed_visits": 41,
        "identity_visits": 39,
        "exact_fast_call_count": 41,
        "legacy_call_count": 0,
        "online_powell_call_count": 0,
        "found_inf_count": 0,
        "skipped_step_count": 0,
        "checkpoint_writes": 0,
    }
    for field, expected in exact_expected.items():
        if int(row[field]) != expected:
            raise FixedACacheStop(
                f"STOP_FIXED_A_RESAMPLED_REFERENCE_VALUE:{field}:{row[field]}:{expected}"
            )
    for field in (
        "oom",
        "training_authorized",
        "formal_training_started",
        "route_b_started",
        "science_change",
    ):
        if _strict_bool(row[field], f"resampled_reference.{field}"):
            raise FixedACacheStop(f"STOP_FIXED_A_RESAMPLED_REFERENCE_FLAG:{field}")

    windows = pd.read_csv(RESAMPLED_RESOURCE_WINDOWS, float_precision="round_trip")
    if (
        len(windows) != 5
        or not windows["branch"].astype(str).eq("NATIVE_A").all()
        or not windows["transform_impl"].astype(str).eq("exact_fast").all()
        or int(windows["measurement_phase"].astype(str).eq("measured").sum()) != 3
        or int(windows["found_inf"].sum()) != 0
        or int(windows["skipped_step"].sum()) != 0
        or int(windows["online_powell_call_count"].sum()) != 0
    ):
        raise FixedACacheStop("STOP_FIXED_A_RESAMPLED_REFERENCE_WINDOWS")
    worker = json.loads(RESAMPLED_RESOURCE_WORKER_AUDIT.read_text(encoding="utf-8"))
    runner = json.loads(RESAMPLED_RESOURCE_RUNNER_AUDIT.read_text(encoding="utf-8"))
    if (
        worker.get("pass") is not True
        or runner.get("status") != "PASS"
        or runner.get("all_hard_gates_pass") is not True
    ):
        raise FixedACacheStop("STOP_FIXED_A_RESAMPLED_REFERENCE_AUDIT_PASS")

    reference_row = {
        "schema": "m1_native_a_fixed_resampled_resource_reference_v1",
        "reference_kind": "FROZEN_RESAMPLED_NATIVE_A_EXISTING_EVIDENCE_NO_RERUN",
        "stage": "after",
        "branch": "NATIVE_A",
        "transform_impl": "exact_fast",
        "resource_protocol": str(row["resource_protocol"]),
        "measured_windows": int(row["measured_windows"]),
        "mean_measured_window_wall_seconds": float(
            row["mean_measured_window_wall_seconds"]
        ),
        "total_measured_window_wall_seconds": float(
            row["total_measured_window_wall_seconds"]
        ),
        "relative_slowdown_percent_vs_control": float(
            row["relative_slowdown_percent_vs_same_stage_control"]
        ),
        "relative_slowdown_ratio_vs_control": (
            1.0
            + float(row["relative_slowdown_percent_vs_same_stage_control"]) / 100.0
        ),
        "target_loader_wait_seconds_measured_total": float(
            row["target_loader_wait_seconds_measured_total"]
        ),
        "photometric_transform_seconds_measured_total": float(
            row["photometric_transform_seconds_measured_total"]
        ),
        "mean_cpu_utilization_percent": float(row["mean_cpu_utilization_percent"]),
        "peak_cpu_observed_rss_bytes": int(row["peak_cpu_observed_rss_bytes"]),
        "max_gpu_peak_allocated_bytes": int(row["max_gpu_peak_allocated_bytes"]),
        "max_gpu_peak_reserved_bytes": int(row["max_gpu_peak_reserved_bytes"]),
        **exact_expected,
        "oom": False,
        "training_authorized": False,
        "formal_training_started": False,
        "route_b_started": False,
        "science_change": False,
    }
    result = {
        **reference_row,
        "comparison_path": str(RESAMPLED_RESOURCE_COMPARISON.resolve()),
        "comparison_sha256": RESAMPLED_RESOURCE_COMPARISON_SHA256,
        "optimizer_windows_path": str(RESAMPLED_RESOURCE_WINDOWS.resolve()),
        "optimizer_windows_sha256": RESAMPLED_RESOURCE_WINDOWS_SHA256,
        "worker_audit_path": str(RESAMPLED_RESOURCE_WORKER_AUDIT.resolve()),
        "worker_audit_sha256": RESAMPLED_RESOURCE_WORKER_AUDIT_SHA256,
        "runner_audit_path": str(RESAMPLED_RESOURCE_RUNNER_AUDIT.resolve()),
        "runner_audit_sha256": RESAMPLED_RESOURCE_RUNNER_AUDIT_SHA256,
    }
    if output is None:
        result["reference_csv_written"] = False
        return result
    reference_csv = output / "fixed_a_resampled_resource_reference.csv"
    expected_reference = pd.DataFrame([reference_row])
    reference_reused = reference_csv.is_file()
    if reference_reused:
        observed_reference = pd.read_csv(reference_csv, float_precision="round_trip")
        if list(observed_reference.columns) != list(expected_reference.columns):
            raise FixedACacheStop("STOP_FIXED_A_RESAMPLED_REFERENCE_EXISTING_COLUMNS")
        _frames_exact(
            expected_reference,
            observed_reference,
            tuple(expected_reference.columns),
            "resampled_reference_existing_no_overwrite",
        )
    else:
        atomic_csv(reference_csv, expected_reference)
    result.update(
        {
            "reference_csv_available": True,
            "reference_csv_written": not reference_reused,
            "reference_csv_written_this_attempt": not reference_reused,
            "reference_csv_reused_without_overwrite": reference_reused,
            "reference_csv_path": str(reference_csv.resolve()),
            "reference_csv_sha256": sha256_file(reference_csv),
        }
    )
    return result


def _prior_resource_evidence_snapshot(
    output: Path, current_attempt: int
) -> dict[str, Any]:
    """Hash every file in earlier immutable resource-attempt namespaces."""

    if int(current_attempt) < 1:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_ATTEMPT_INVALID")
    prior_roots: list[Path] = []
    if int(current_attempt) > 1:
        default_root = output / "resource_windows"
        if default_root.exists():
            prior_roots.append(default_root)
        for attempt in range(2, int(current_attempt)):
            path = output / f"resource_windows_attempt{attempt}"
            if path.exists():
                prior_roots.append(path)
    roots: list[dict[str, Any]] = []
    all_files: list[dict[str, Any]] = []
    for root_path in prior_roots:
        if not root_path.is_dir():
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_PRIOR_ROOT_TYPE:{root_path}")
        root_files: list[dict[str, Any]] = []
        for file_path in sorted(
            (path for path in root_path.rglob("*") if path.is_file()),
            key=lambda path: str(path.relative_to(output)).casefold(),
        ):
            record = {
                "relative_path": str(file_path.relative_to(output)).replace("\\", "/"),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": sha256_file(file_path),
            }
            root_files.append(record)
            all_files.append(record)
        roots.append(
            {
                "relative_root": str(root_path.relative_to(output)).replace("\\", "/"),
                "file_count": len(root_files),
                "total_bytes": int(sum(item["size_bytes"] for item in root_files)),
                "files": root_files,
            }
        )
    payload: dict[str, Any] = {
        "schema": "m1_native_a_fixed_prior_resource_evidence_snapshot_v1",
        "current_attempt": int(current_attempt),
        "prior_root_count": len(roots),
        "prior_file_count": len(all_files),
        "prior_total_bytes": int(sum(item["size_bytes"] for item in all_files)),
        "roots": roots,
        "preservation_rule": "read-only SHA snapshot; no delete, move, overwrite, or reuse",
    }
    payload["snapshot_sha256"] = canonical_json_sha256(payload)
    return payload


def run_resource_gate(
    *,
    result_dir: Path,
    fixed_input_dir: Path,
    candidates: Sequence[str],
    device: str,
) -> dict[str, Any]:
    requested = tuple(dict.fromkeys(candidates))
    if not requested or any(candidate not in CANDIDATES for candidate in requested):
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_CANDIDATES")
    if THETA_ONLY not in requested and not (
        _safe_result_dir(result_dir) / "resource_windows" / THETA_ONLY
    ).is_dir():
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_THETA_BASELINE_REQUIRED")
    locks = _verify_sha_locks()
    patch_audit = _resource_patch_audit()
    inputs = load_fixed_inputs(fixed_input_dir)
    output = _safe_result_dir(result_dir)
    resampled_reference = _resampled_a_frozen_resource_reference(output)
    cache_root = output / "working_cache"
    resource_root = output / "resource_windows"
    resource_attempt = 1
    if any((resource_root / candidate).exists() for candidate in requested):
        for candidate_attempt in range(2, 1000):
            candidate_root = output / f"resource_windows_attempt{candidate_attempt}"
            if not candidate_root.exists():
                resource_root = candidate_root
                resource_attempt = candidate_attempt
                break
        else:
            raise FixedACacheStop("STOP_FIXED_A_RESOURCE_ATTEMPT_NAMESPACE_EXHAUSTED")
    prior_failed_evidence = _prior_resource_evidence_snapshot(
        output, resource_attempt
    )
    resource_root.mkdir(parents=True, exist_ok=True)
    subprocess_records: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for candidate in requested:
        candidate_root = resource_root / candidate
        if candidate_root.exists():
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_CANDIDATE_EXISTS:{candidate_root}")
        candidate_results: dict[str, Any] = {}
        for role in ("CONTROL", "FIXED_A"):
            branch_dir = candidate_root / role.casefold()
            branch_dir.mkdir(parents=True, exist_ok=False)
            materialization = materialize_resource_policy(
                output_dir=branch_dir, inputs=inputs
            )
            record = _launch_resource_subprocess(
                candidate=candidate,
                role=role,
                output_dir=branch_dir,
                cache_root=cache_root,
                device=device,
            )
            subprocess_records.append(record)
            audit, micro, windows = _resource_worker_output(branch_dir, role)
            active_calls = int(audit["transform_execution"]["transformed_visit_count"])
            if role == "FIXED_A" and active_calls != int(materialization["active_count"]):
                raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_ACTIVE_CALLS:{candidate}")
            candidate_results[role] = {
                "directory": str(branch_dir.resolve()),
                "audit": audit,
                "micro": micro,
                "windows": windows,
                "materialization": materialization,
                "cache_hit_count": active_calls,
                "cache_miss_count": 0,
                "cache_fallback_count": 0,
            }
        # Global RNG consumption must remain independent of the Student view.
        if (
            candidate_results["CONTROL"]["audit"]["final"]["rng_hashes"]
            != candidate_results["FIXED_A"]["audit"]["final"]["rng_hashes"]
        ):
            raise FixedACacheStop(f"STOP_FIXED_A_RESOURCE_CONTROL_FIXED_RNG:{candidate}")
        results[candidate] = candidate_results

    # Existing baseline outputs may be reused only when this invocation did not
    # request THETA_ONLY; they are read with the same strict validators.
    if THETA_ONLY not in results:
        theta_root = resource_root / THETA_ONLY
        results[THETA_ONLY] = {}
        for role in ("CONTROL", "FIXED_A"):
            audit, micro, windows = _resource_worker_output(
                theta_root / role.casefold(), role
            )
            results[THETA_ONLY][role] = {
                "directory": str((theta_root / role.casefold()).resolve()),
                "audit": audit,
                "micro": micro,
                "windows": windows,
            }

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import m1_routea_resource_equivalence_training_window_runner_v1 as frozen_runner

    if sha256_file(Path(frozen_runner.__file__)) != FORMAL_RESOURCE_RUNNER_SHA256:
        raise FixedACacheStop("STOP_FIXED_A_IMPORTED_RESOURCE_RUNNER_SHA")
    theta_reference = results[THETA_ONLY]
    state_fields = (
        "raw_state_sha256",
        "ema_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "scaler_state_sha256",
        "bn_running_state_sha256",
        "rng_hashes",
    )
    exact_evidence: dict[str, Any] = {}
    for candidate, candidate_results in results.items():
        if candidate == THETA_ONLY:
            continue
        exact_evidence[candidate] = {}
        for role in ("CONTROL", "FIXED_A"):
            baseline = theta_reference[role]
            current = candidate_results[role]
            for field in state_fields:
                if current["audit"]["final"][field] != baseline["audit"]["final"][field]:
                    raise FixedACacheStop(
                        f"STOP_FIXED_A_RESOURCE_FINAL_STATE:{candidate}:{role}:{field}"
                    )
            if current["audit"]["initial"] != baseline["audit"]["initial"]:
                raise FixedACacheStop(
                    f"STOP_FIXED_A_RESOURCE_INITIAL_STATE:{candidate}:{role}"
                )
            _frames_exact(
                baseline["micro"],
                current["micro"],
                frozen_runner.MICRO_EXACT_COLUMNS,
                f"{candidate}:{role}:micro",
            )
            _frames_exact(
                baseline["windows"],
                current["windows"],
                frozen_runner.WINDOW_EXACT_COLUMNS,
                f"{candidate}:{role}:window",
            )
            exact_evidence[candidate][role] = {
                "initial_state_exact": True,
                "final_model_optimizer_ema_amp_bn_rng_exact_vs_theta_only": True,
                "micro_science_columns_exact_vs_theta_only": True,
                "window_science_columns_exact_vs_theta_only": True,
            }

    rows: list[dict[str, Any]] = []
    record_by_pair = {
        (record["candidate"], record["role"]): record for record in subprocess_records
    }
    for candidate in requested:
        for role in ("CONTROL", "FIXED_A"):
            value = results[candidate][role]
            measured = value["windows"].loc[
                value["windows"]["measurement_phase"].astype(str).str.casefold().eq("measured")
            ]
            record = record_by_pair[(candidate, role)]
            rows.append(
                {
                    "candidate": candidate,
                    "role": role,
                    "measured_window_count": len(measured),
                    "mean_measured_optimizer_window_wall_seconds": float(
                        measured["total_optimizer_window_wall_time"].mean()
                    ),
                    "mean_measured_target_loader_wait_seconds": float(
                        measured["target_loader_wait_seconds"].mean()
                    ),
                    "photometric_transform_seconds_sum": float(
                        measured["photometric_transform_time"].sum()
                    ),
                    "subprocess_wall_seconds": record["wall_seconds"],
                    "peak_process_tree_rss_bytes": record[
                        "observed_peak_process_tree_rss_bytes"
                    ],
                    "process_read_bytes": record["observed_process_read_bytes"],
                    "process_write_bytes": record["observed_process_write_bytes"],
                    "gpu_max_allocated_bytes": int(measured["gpu_max_allocated_bytes"].max()),
                    "gpu_max_reserved_bytes": int(measured["gpu_max_reserved_bytes"].max()),
                    "found_inf": int(measured["found_inf"].sum()),
                    "skipped_step": int(measured["skipped_step"].sum()),
                    "oom": False,
                    "cache_hit_count": int(
                        value.get("cache_hit_count", value["audit"]["transform_execution"]["transformed_visit_count"])
                    ),
                    "cache_miss_count": 0,
                    "cache_fallback_count": 0,
                }
            )
    metrics = pd.DataFrame(rows)
    control_means = metrics.loc[metrics["role"].eq("CONTROL")].set_index("candidate")[
        "mean_measured_optimizer_window_wall_seconds"
    ]
    metrics["relative_slowdown_vs_same_candidate_control"] = [
        (
            float(row["mean_measured_optimizer_window_wall_seconds"])
            / float(control_means.loc[row["candidate"]])
            if row["role"] == "FIXED_A"
            else 1.0
        )
        for row in metrics.to_dict("records")
    ]
    metrics.insert(0, "schema", "m1_native_a_fixed_cache_resource_window_summary_v1")
    metrics_path = output / "fixed_a_cache_resource_windows.csv"
    atomic_csv(metrics_path, metrics)
    required_summary_path = output / "fixed_a_resource_summary.csv"
    atomic_csv(required_summary_path, metrics)
    fixed_means = metrics.loc[metrics["role"].eq("FIXED_A")].set_index("candidate")[
        "mean_measured_optimizer_window_wall_seconds"
    ]
    eligible = [
        candidate
        for candidate in requested
        if candidate in fixed_means.index
        and (candidate == THETA_ONLY or candidate in exact_evidence)
    ]
    if not eligible:
        raise FixedACacheStop("STOP_FIXED_A_RESOURCE_NO_ELIGIBLE_CANDIDATE")
    best_candidate = min(eligible, key=lambda candidate: float(fixed_means.loc[candidate]))
    prior_failed_evidence_after = _prior_resource_evidence_snapshot(
        output, resource_attempt
    )
    if prior_failed_evidence_after != prior_failed_evidence:
        raise FixedACacheStop("STOP_FIXED_A_PRIOR_RESOURCE_EVIDENCE_CHANGED")
    summary = {
        "schema": "m1_native_a_fixed_cache_resource_gate_v1",
        "status": "PASS",
        "scope": "DISPOSABLE_E72_FIVE_WINDOW_RESOURCE_ONLY_NOT_FORMAL_TRAINING",
        "fixed_contract_sha256": FIXED_CONTRACT_SHA256,
        "sha_locks": locks,
        "runtime_patch_audit": patch_audit,
        "resampled_a_frozen_resource_reference": resampled_reference,
        "resource_windows_root": str(resource_root.resolve()),
        "resource_attempt": resource_attempt,
        "prior_failed_resource_evidence_preserved": resource_attempt > 1,
        "prior_failed_evidence": prior_failed_evidence,
        "requested_candidates": list(requested),
        "best_candidate_rule": (
            "minimum measured optimizer-window mean among strictly bit-equivalent candidates"
        ),
        "best_candidate": best_candidate,
        "exact_evidence": exact_evidence,
        "subprocess_records": subprocess_records,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "required_resource_summary_path": str(required_summary_path.resolve()),
        "required_resource_summary_sha256": sha256_file(required_summary_path),
        "zero_flags": {
            "cache_fallback_calls": 0,
            "online_powell_calls": 0,
            "found_inf": 0,
            "skipped_step": 0,
            "oom": False,
            "checkpoint_writes": 0,
            "target_adapt_label_reads": 0,
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "dataset_mutations": 0,
        },
        "resource_protocol": {
            "epoch": 72,
            "optimizer_windows": 5,
            "warmup_windows": [1, 2],
            "measured_windows": [3, 4, 5],
            "source_batch": 32,
            "target_batch": 8,
            "workers": 2,
            "accumulation": 2,
            "control_vs_fixed_only": True,
            "fresh_process_per_role_and_candidate": True,
        },
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "SCIENCE_CHANGE": False,
    }
    atomic_json(output / "fixed_a_cache_resource.json", summary)
    return summary


def _parse_candidate_csv(value: str) -> tuple[str, ...]:
    text_value = str(value).strip()
    if text_value.casefold() == "all":
        return CANDIDATES
    requested = tuple(
        dict.fromkeys(part.strip() for part in text_value.split(",") if part.strip())
    )
    if not requested or any(candidate not in CANDIDATES for candidate in requested):
        raise argparse.ArgumentTypeError(
            "candidates must be ALL or a comma-separated subset of "
            + ",".join(CANDIDATES)
        )
    return requested


def _self_check() -> dict[str, Any]:
    """Pure static/synthetic check: no images, training, cache build, or CUDA."""

    locks = _verify_sha_locks()
    script_path = Path(__file__).resolve()
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    prohibited_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        else:
            continue
        if call_name in {
            "minimize",
            "fmin_powell",
            "differential_evolution",
            "least_squares",
        }:
            prohibited_calls.append(call_name)
    if prohibited_calls:
        raise FixedACacheStop(
            "STOP_FIXED_A_SELF_CHECK_SOLVER_CALL:" + ",".join(sorted(prohibited_calls))
        )

    source_palette = np.asarray(
        [[0, 0, 0], [0, 0, 1], [1, 2, 3], [255, 255, 255]], dtype=np.uint8
    )
    deployment_palette = np.asarray(
        [[9, 8, 7], [6, 5, 4], [3, 2, 1], [101, 102, 103]], dtype=np.uint8
    )
    image = np.asarray(
        [
            [[1, 2, 3], [0, 0, 0], [1, 2, 3], [255, 255, 255]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    expected_indices = np.asarray([[2, 0, 2, 3], [1, 1, 0, 3]], dtype=np.int64)
    expected = np.ascontiguousarray(deployment_palette[expected_indices])
    rng_before = _rng_snapshot()
    for chunk_pixels in (1, 2, 3, image.shape[0] * image.shape[1] + 1):
        observed = palette_remap_exact(
            image,
            source_palette,
            deployment_palette,
            chunk_pixels=chunk_pixels,
        )
        if not np.array_equal(observed, expected):
            raise FixedACacheStop(
                f"STOP_FIXED_A_SELF_CHECK_PALETTE_REMAP:{chunk_pixels}"
            )
    if _rng_snapshot() != rng_before:
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_RNG_CONSUMPTION")

    missing_color_rejected = False
    try:
        palette_remap_exact(
            np.asarray([[[0, 0, 2]]], dtype=np.uint8),
            source_palette,
            deployment_palette,
        )
    except FixedACacheStop:
        missing_color_rejected = True
    if not missing_color_rejected:
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_MISSING_COLOR_ACCEPTED")

    duplicate_source_rejected = False
    try:
        palette_remap_exact(
            image,
            np.asarray([[0, 0, 0], [0, 0, 0]], dtype=np.uint8),
            np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.uint8),
        )
    except FixedACacheStop:
        duplicate_source_rejected = True
    if not duplicate_source_rejected:
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_DUPLICATE_SOURCE_ACCEPTED")

    left = pd.DataFrame({"integer": [1, 2], "float": [1.0, np.nan], "text": ["a", "b"]})
    _frames_exact(left, left.copy(), tuple(left.columns), "self_check")
    formal_hash = _formal_bgr_sha_from_rgb(expected)
    if not _is_sha256(formal_hash) or formal_hash != _formal_bgr_sha_from_rgb(expected.copy()):
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_FORMAL_HASH")
    synthetic_case = AuditCase(
        suite="golden_6",
        case_id="self_check_golden",
        canonical_uid="synthetic_uid",
        image_path=Path("synthetic_not_read.png"),
        image_sha256="0" * 64,
        active=True,
        theta=np.asarray([0.125, 1.0, 1.0], dtype="<f8"),
        expected_bgr_sha256=formal_hash,
        formal_visit_index=None,
        unique_rgb=len(source_palette),
    )
    second_theta_case = AuditCase(
        **{
            **synthetic_case.__dict__,
            "case_id": "self_check_golden_second_theta",
            "theta": np.asarray([0.25, 1.0, 1.0], dtype="<f8"),
        }
    )
    synthetic_cache_root = Path(r"D:\synthetic_fixed_a_cache_not_written")
    if _golden_audit_payload_paths(synthetic_cache_root, synthetic_case)["meta"] == (
        _golden_audit_payload_paths(synthetic_cache_root, second_theta_case)["meta"]
    ):
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_GOLDEN_THETA_NAMESPACE")
    synthetic_temperature = [
        COLD_OBSERVATION if repetition == 1 else WARM_OBSERVATION
        for repetition in range(1, DEFAULT_REPETITIONS + 1)
    ]
    if synthetic_temperature != [
        COLD_OBSERVATION,
        *([WARM_OBSERVATION] * (DEFAULT_REPETITIONS - 1)),
    ]:
        raise FixedACacheStop("STOP_FIXED_A_SELF_CHECK_COLD_WARM_RULE")
    patch_audit = _resource_patch_audit()
    resampled_reference = _resampled_a_frozen_resource_reference(None)
    return {
        "schema": "m1_native_a_fixed_cache_resource_self_check_v1",
        "status": "PASS",
        "script_path": str(script_path),
        "script_ast_node_count": sum(1 for _ in ast.walk(tree)),
        "sha_locks": locks,
        "resource_patch_audit": patch_audit,
        "resampled_a_frozen_resource_reference": resampled_reference,
        "synthetic_palette": {
            "pixel_count": int(image.shape[0] * image.shape[1]),
            "unique_source_count": len(source_palette),
            "duplicate_pixels_preserved": True,
            "chunk_boundaries_exact": True,
            "missing_source_color_rejected": True,
            "duplicate_source_palette_rejected": True,
            "formal_bgr_sha256": formal_hash,
            "golden_disk_payload_IO_executed": False,
            "golden_disk_payload_reason": (
                "self-check is no-write; actual Golden audit hard-gates persisted payload bytes"
            ),
            "golden_same_image_distinct_theta_namespaced": True,
        },
        "cold_warm_lookup_throughput": {
            "frozen_rule_exact": True,
            "default_repetition_assignment": synthetic_temperature,
            "os_page_cache_caveat_present": bool(COLD_WARM_OS_CACHE_CAVEAT),
            "physical_cold_disk_claimed": False,
        },
        "online_powell_calls": 0,
        "image_reads": 0,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "training_steps": 0,
        "checkpoint_writes": 0,
        "dataset_mutations": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "SCIENCE_CHANGE": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-A exact cache prototype/equivalence/resource gate. "
            "This entry point cannot start formal training or solve theta."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "self-check",
        help="static locks plus synthetic palette/remap check; no image or training reads",
    )

    build = subparsers.add_parser(
        "build-cache", help="atomic/resumable exact cache materialization; never solves theta"
    )
    build.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    build.add_argument("--fixed-input-dir", type=Path, default=RESULT_DIR)
    build.add_argument("--candidates", type=_parse_candidate_csv, default=CANDIDATES)
    build.add_argument(
        "--verify-resume-file-sha",
        action="store_true",
        default=True,
        help=(
            "always enabled: rehash every resumed NPY payload in addition to "
            "lineage/size/shape checks"
        ),
    )

    audit = subparsers.add_parser(
        "audit-benchmark",
        help="6 Golden + deterministic 240 replay + high-palette exact audit/benchmark",
    )
    audit.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    audit.add_argument("--fixed-input-dir", type=Path, default=RESULT_DIR)
    audit.add_argument("--candidates", type=_parse_candidate_csv, default=CANDIDATES)
    audit.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)

    resource = subparsers.add_parser(
        "resource-window",
        help="disposable frozen five-window Control-vs-Fixed resource gate only",
    )
    resource.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    resource.add_argument("--fixed-input-dir", type=Path, default=RESULT_DIR)
    resource.add_argument("--candidates", type=_parse_candidate_csv, default=CANDIDATES)
    resource.add_argument("--device", default="cuda:0")

    child = subparsers.add_parser("resource-worker", help=argparse.SUPPRESS)
    child.add_argument("--candidate", choices=CANDIDATES, required=True)
    child.add_argument("--resource-role", choices=("CONTROL", "FIXED_A"), required=True)
    child.add_argument("--output-dir", type=Path, required=True)
    child.add_argument("--policy-cache", type=Path, required=True)
    child.add_argument("--device", required=True)
    return parser


def _failure_directory(args: argparse.Namespace) -> Path | None:
    if args.command == "self-check":
        return None
    candidate = getattr(args, "result_dir", None)
    if candidate is None:
        candidate = getattr(args, "output_dir", None)
    if candidate is None:
        return None
    try:
        return _safe_result_dir(Path(candidate)) / "raw"
    except Exception:
        return None


def _preserve_failure(args: argparse.Namespace, exc: BaseException) -> str | None:
    directory = _failure_directory(args)
    if directory is None:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"fixed_a_cache_failure_{timestamp}.json"
    payload = {
        "schema": "m1_native_a_fixed_cache_failure_v1",
        "status": "FAILED_CLOSED",
        "command": str(getattr(args, "command", "")),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "SCIENCE_CHANGE": False,
    }
    atomic_json(path, payload)
    return str(path.resolve())


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "self-check":
            result = _self_check()
        elif args.command == "build-cache":
            result = build_cache(
                result_dir=args.result_dir,
                fixed_input_dir=args.fixed_input_dir,
                candidates=args.candidates,
                verify_resume_file_sha=bool(args.verify_resume_file_sha),
            )
        elif args.command == "audit-benchmark":
            if int(args.repetitions) < DEFAULT_REPETITIONS:
                raise FixedACacheStop("STOP_FIXED_A_BENCHMARK_REPETITIONS_LT_3")
            result = run_equivalence_and_benchmark(
                result_dir=args.result_dir,
                fixed_input_dir=args.fixed_input_dir,
                candidates=args.candidates,
                repetitions=int(args.repetitions),
            )
        elif args.command == "resource-window":
            result = run_resource_gate(
                result_dir=args.result_dir,
                fixed_input_dir=args.fixed_input_dir,
                candidates=args.candidates,
                device=str(args.device),
            )
        elif args.command == "resource-worker":
            return _run_resource_worker_child(args)
        else:  # pragma: no cover - argparse owns the command domain.
            parser.error(f"unsupported command: {args.command}")
            return 2
    except Exception as exc:
        failure_path: str | None = None
        try:
            failure_path = _preserve_failure(args, exc)
        except Exception as failure_exc:
            print(
                "FIXED_A_FAILURE_EVIDENCE_WRITE_FAILED="
                f"{type(failure_exc).__name__}:{failure_exc}",
                file=sys.stderr,
            )
        traceback.print_exc()
        if failure_path:
            print(f"FIXED_A_FAILURE_EVIDENCE={failure_path}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
