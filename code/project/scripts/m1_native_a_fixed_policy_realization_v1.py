from __future__ import annotations

"""Construct and instantiate the frozen Fixed-A per-image policy.

This runner is intentionally limited to policy construction and offline
photometric realization.  It does not expose a training entry point.  Every
scientific solve is delegated to the SHA-locked Route-A offline engine and its
protected float64 single-start Powell realization core.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from m1_routea_offline_theta_cache_engine_v1 import (
    BASE,
    BRANCH_NATIVE,
    DELTA_HASH_PREFIX,
    ExecutionPlan,
    PersistentSpawnPool,
    ScientificConfigIdentity,
    VisitAlias,
    build_execution_plan,
    build_frozen_runtime_spec,
    canonical_json_sha256,
    implementation_unit_tests,
    make_explicit_target_solve_request,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "M1_NATIVE_A_FIXED_COMPLEXITY_UTILITY_ABLATION_V1_PRETRAIN_GATE"
CONTRACT_SCHEMA = (
    "m1_native_a_fixed_complexity_utility_ablation_v1_pretrain_gate_contract_v1"
)
CONTRACT_SHA256 = "f26b856ab54cd3a55ff720a8215a3300fee16f82388fa045a1d05e840bb27d0c"
SELECTION_SCHEMA = "m1_native_a_fixed_delta_selection_v1"
EXPOSURE_SCHEMA = "m1_native_a_fixed_visit_exposure_audit_v1"
THETA_SCHEMA = "m1_native_a_fixed_theta_realization_v1"
SELECTOR_NAMESPACE = "NATIVE_A_FIXED_V1"
FORMAL_EPOCH_START = 72
FORMAL_EPOCH_END = 100
VISITS_PER_EPOCH = 4416
EXPECTED_IMAGES = 3272
EXPECTED_WITH_DELTA = 3226
EXPECTED_WITHOUT_DELTA = 46
EXPECTED_CANDIDATES = 62679
EXPECTED_VISITS = 128064
EXPECTED_ACTIVE = 62679
EXPECTED_IDENTITY = 65385
FROZEN_WORKERS = 8
DEFAULT_GROUPS_PER_CHECKPOINT = 64
RAW_DIRNAME = "raw"

SEMANTIC_DIR = ROOT / "project" / "audits" / (
    "M1_ROUTE_A_SEMANTIC_RESOURCE_GATE_MEASUREMENT_V1_20260810_001748"
)
Q2_DIR = ROOT / "project" / "audits" / (
    "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
)
REALIZATION_DIR = ROOT / "project" / "audits" / (
    "M1_ROUTE_A_PHOTOMETRIC_REALIZATION_FEASIBILITY_MEASUREMENT_V1_20260809_131730"
)
RESOURCE_DIR = ROOT / "project" / "results" / (
    "M1_ROUTE_A_RESOURCE_EQUIVALENCE_ENGINEERING_V1"
)
CONTINUATION_DIR = ROOT / "project" / "results" / (
    "M1_ROUTE_A_RESOURCE_EQUIVALENCE_ENGINEERING_V1_CONTINUATION_AUDIT"
)

ENGINE = ROOT / "project" / "scripts" / "m1_routea_offline_theta_cache_engine_v1.py"
FROZEN_CORE = ROOT / "project" / "scripts" / "m1_routea_photometric_realization_core_v1.py"
EXACT_TRANSFORM_CORE = (
    ROOT / "project" / "scripts" / "m1_routea_resource_equivalence_execution_core_v1.py"
)
SEMANTIC_MODULE = ROOT / "project" / "scripts" / "m1_routea_semantic_resource_semantic_v1.py"
SEMANTIC_CORE = ROOT / "project" / "scripts" / "m1_routea_semantic_resource_core_v1.py"
SCHEDULE = SEMANTIC_DIR / RAW_DIRNAME / "visit_policy_schedule.parquet"
NATIVE_DELTAS = SEMANTIC_DIR / RAW_DIRNAME / "visit_native_deltas.parquet"
WHITELIST = Q2_DIR / "TARGET_ADAPT_STYLE_AUDIT_WHITELIST.csv"
IMAGE_DESCRIPTORS = Q2_DIR / "IMAGE_STYLE_DESCRIPTORS.parquet"
DESCRIPTOR_SCHEMA = Q2_DIR / "STYLE_DESCRIPTOR_SCHEMA.json"
EVENT_PROTOTYPES = Q2_DIR / "EVENT_STYLE_PROTOTYPES.csv"
Q2_SNAPSHOT = Q2_DIR / "CODE_SNAPSHOT"
TARGET_SCHEDULE = ROOT / "project" / "contracts" / "protocol_b_target_schedule_e71_e100.jsonl"
TARGET_SCHEDULE_MANIFEST = ROOT / "project" / "contracts" / "protocol_b_target_schedule_manifest.json"
GEOMETRY_CONTRACT = ROOT / "project" / "contracts" / "m1_routea_descriptor_geometry_feasibility_measurement_v1.json"
REALIZATION_CONTRACT = ROOT / "project" / "contracts" / "m1_routea_photometric_realization_feasibility_measurement_v1.json"
SEMANTIC_CONTRACT = ROOT / "project" / "contracts" / "m1_routea_semantic_resource_gate_measurement_v1.json"
SUBSET_AUDIT = RESOURCE_DIR / "solve_key_subset_audit.json"
RESOURCE_FINAL = RESOURCE_DIR / "FINAL_STATUS.json"
CONTINUATION_FINAL = CONTINUATION_DIR / "FINAL_STATUS.json"
CONTINUATION_ZIP = CONTINUATION_DIR / (
    "M1_ROUTE_A_RESOURCE_EQUIVALENCE_ENGINEERING_V1_CONTINUATION_AUDIT_CORE.zip"
)
INTERNAL_IMAGE_ROOT = (
    ROOT / "datasets" / "dataset_final_v2_1" / "target" / "adapt" / "images"
)

EXPECTED_HASHES: dict[Path, str] = {
    ENGINE: "eede42866164fbbac4589c036277f4f2137a8e13d4435dfb85e8b433ea32c09f",
    FROZEN_CORE: "45b62bf0a192e2d527f3171661d36b19d2dd7da21c63afb93140015251b449c1",
    EXACT_TRANSFORM_CORE: "72ebbf342d50decc7f0a1053adf59c40a614f9b758ab64667aea9a146586d20b",
    SEMANTIC_MODULE: "7f98f1ad75807a793e0b7c066897628f288e5be5d815d3e582e08f4457c46677",
    SEMANTIC_CORE: "7d2396fdb083e05bb9d0ca91bd48a0aadf05642555f8f5b1c0aebe377923d051",
    SCHEDULE: "1c5c10f3bc349e7aa62f1b27398d03c36048dcf61c155e62b6f57a0cd91457ee",
    NATIVE_DELTAS: "0231d7499390ae2d25521380aa8a707081d147fd9e0dbdfce76d74a2566b12d3",
    WHITELIST: "24426506f2fba321ca1c98f593d459dca3e9e76d030c0303299eacd1ad76b29f",
    IMAGE_DESCRIPTORS: "2281400f9601203437df6a8fefd06e8516ff760189faba54d7126abca62068cc",
    DESCRIPTOR_SCHEMA: "628a8b2da5927e816c4291d6622824076214d62486092ddfcabfc435c5f7f8f9",
    EVENT_PROTOTYPES: "0519d5d08b97647f9a84a95e1d1913e88d2c695d7b7876c19a8ff699f0479d02",
    Q2_SNAPSHOT / "q2_descriptors.py": "eb1da56bd60a13e7f5ad148300bf5e993c17414a8dc68d6065e720e59859a321",
    Q2_SNAPSHOT / "q2_frozen.py": "b4968174cc9d77717095282e24a49ad63aa200dc7a17e831274626d5eb09d2b3",
    Q2_SNAPSHOT / "q2_hpl.py": "11d306ded09da1b44b078bad5ed49872e141492dab90533d3454bccae111fa1c",
    Q2_SNAPSHOT / "q2_rng.py": "ce1f64f60c3eff1789d511c49902e13ba30f7404198a753fa6e37a146216aaef",
    TARGET_SCHEDULE: "d4dc5afea3adacec2f5562a10cf4f10226d751ada3b52f0c5a69aa1949d5b923",
    TARGET_SCHEDULE_MANIFEST: "7071514f5c6f7bb93db30f1786b061ece9d1a3d4b5e6f009b5ed0fb260500a9a",
    GEOMETRY_CONTRACT: "5e4bb9ab6560196e0fc37eed96d23403090901700d2d27a01217a4f526dcafb4",
    REALIZATION_CONTRACT: "c2e935adcfa217c3d0d83589ac38f819a5870d8d010a44129410b94311ff1326",
    SEMANTIC_CONTRACT: "d78175cd4d0ec78d803cf7003f36f75560d1afe93daf319f2cc3c12f3ada7b43",
    SUBSET_AUDIT: "1a837bacba34473168855ce0764d2f4eabd058ff6de397d81bb33a0d99bbf4a5",
    RESOURCE_FINAL: "7a6d551cfe806e0c8b55024967c1226a3cff2eec34665743762c1a193517c29f",
    CONTINUATION_FINAL: "ff7c73e671469312d629e9db9165b56918b5e418c56ef60dabcae82c8e3c0157",
    CONTINUATION_ZIP: "7b6297fe621a2184b89aa5592f52b9fa0034d9771e6942809a15a11626b87e9d",
}

EXPECTED_CONFIG_SHAS = {
    "solver_config_sha256": "b6d927dc89aa2210ccc999d38b2a9384fa64ddd2d6eeaa0f24d0215befd48e9b",
    "operator_config_sha256": "f263351a4ab9a4209c0426db736ec23432ed0700994d5e8eea241251c431aaaf",
    "descriptor_config_sha256": "97cee33877d757aac83ffb654103facf58cca6c6e503e4b9a9e38df919281d53",
    "realization_core_sha256": "45b62bf0a192e2d527f3171661d36b19d2dd7da21c63afb93140015251b449c1",
}

DELTA_COLUMNS = tuple(f"native_delta__{name}" for name in BASE)
TARGET_COLUMNS = tuple(f"native_target_std__{name}" for name in BASE)
ORIGINAL_COLUMNS = tuple(f"std__whole__{name}" for name in BASE)
SELECTION_COLUMNS = (
    "schema", "canonical_uid", "valid_delta_count", "fixed_delta_exists",
    "ordered_formal_visit_indices", "ordered_delta_sha256",
    "valid_delta_list_sha256", "hash_input_utf8", "hash_input_sha256",
    "hash_integer_decimal", "selected_index", "selected_formal_visit_index",
    "selected_epoch", "selected_target_visit_ordinal", "selected_schedule_uid",
    "selected_visit_key", "selected_delta_sha",
    "selected_intended_geometric_dose",
    *(f"selected_delta_{index}" for index in range(7)),
    *(f"selected_target_std_{index}" for index in range(7)),
)
EXPOSURE_COLUMNS = (
    "schema", "formal_visit_index", "adaptation_seed", "epoch",
    "target_visit_ordinal", "schedule_uid", "canonical_uid",
    "resample_augment_on", "resample_descriptor_policy_valid",
    "resample_policy_flag", "resample_failure_reason", "resample_active",
    "resample_identity", "fixed_delta_exists", "fixed_active", "fixed_identity",
    "fixed_selected_delta_sha", "fixed_delta_sha_used", "active_match",
    "identity_match",
)
THETA_SCIENCE_COLUMNS = (
    "input_image_sha256", "target_descriptor_sha256", "solve_key",
    "cache_namespace", "cache_key_sha256", "theta_delta_t", "theta_c_t",
    "theta_s_c", "theta_raw_f8_sha256", "solver_success", "solver_status",
    "solver_message", "solver_nfev", "solver_nit", "solver_fun",
    "elapsed_seconds", "output_array_sha256", "intended_dose",
    "target_descriptor_dose", "actual_effective_dose",
    "actual_to_target_dose_ratio", "absolute_dose_error",
    "continuous_realization_residual", "deployment_realization_residual",
    "quantization_residual_increment", "realization_residual",
    "normalized_realization_residual", "actual_support_deviation",
    "actual_endpoint_in_tube", "actual_descriptor_internal_valid",
    "tone_preclip_pixel_fraction", "tone_preclip_channel_fraction",
    "tone_clip_mean_excess", "tone_clip_max_excess",
    "lab_to_srgb_preclip_pixel_fraction",
    "lab_to_srgb_preclip_channel_fraction", "gamut_clip_mean_excess",
    "gamut_clip_max_excess", "final_channel_at_zero_fraction",
    "final_channel_at_one_fraction", "any_parameter_bound_hit",
    "image_height", "image_width", "unique_rgb_count",
    "science_cache_entry_sha256",
)
THETA_COLUMNS = (
    "schema", "canonical_uid", "fixed_delta_exists", "realization_status",
    "selected_index", "selected_formal_visit_index", "selected_epoch",
    "selected_target_visit_ordinal", "selected_schedule_uid",
    "selected_delta_sha", "selected_intended_geometric_dose",
    *THETA_SCIENCE_COLUMNS,
)


class FixedPolicyStop(RuntimeError):
    """Fail-closed contract, provenance, policy, or execution violation."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
            default=_json_default,
        )
    else:
        text = json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"), default=_json_default,
        )
    return (text + "\n").encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(_json_bytes(dict(row), pretty=False) for row in rows)
    _atomic_bytes(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(
        temporary, index=False, encoding="utf-8", lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    os.replace(temporary, path)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL,
    )
    return buffer.getvalue().encode("utf-8")


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def formal_visit_index(epoch: int, target_visit_ordinal: int) -> int:
    epoch_value = int(epoch)
    ordinal_value = int(target_visit_ordinal)
    if not FORMAL_EPOCH_START <= epoch_value <= FORMAL_EPOCH_END:
        raise FixedPolicyStop(f"STOP_FORMAL_EPOCH_OUT_OF_RANGE:{epoch_value}")
    if not 0 <= ordinal_value < VISITS_PER_EPOCH:
        raise FixedPolicyStop(f"STOP_FORMAL_VISIT_ORDINAL_OUT_OF_RANGE:{ordinal_value}")
    return (epoch_value - FORMAL_EPOCH_START) * VISITS_PER_EPOCH + ordinal_value


def selector_digest(canonical_uid: str) -> tuple[str, str, int]:
    uid = str(canonical_uid)
    if not uid or uid != uid.strip():
        raise FixedPolicyStop("STOP_CANONICAL_UID_EMPTY_OR_PADDED")
    hash_input = SELECTOR_NAMESPACE + "|" + uid
    digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    integer = int.from_bytes(bytes.fromhex(digest), "big", signed=False)
    return hash_input, digest, integer


def _strict_bool(series: pd.Series, role: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not set(normalized.unique()).issubset({"true", "false"}):
        raise FixedPolicyStop(f"STOP_BOOLEAN_SCHEMA:{role}")
    return normalized.map({"true": True, "false": False}).astype(bool)


def _verify_locked_inputs(contract_path: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    if not contract_path.is_file():
        raise FixedPolicyStop(f"STOP_CONTRACT_MISSING:{contract_path}")
    contract_digest = sha256_file(contract_path)
    if contract_digest != CONTRACT_SHA256:
        raise FixedPolicyStop(
            f"STOP_CONTRACT_SHA256:{contract_digest}:{CONTRACT_SHA256}"
        )
    observed[str(contract_path.resolve())] = contract_digest
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FixedPolicyStop(f"STOP_LOCKED_INPUT_MISSING:{path}")
        digest = sha256_file(path)
        if digest != expected:
            raise FixedPolicyStop(
                f"STOP_LOCKED_INPUT_SHA256:{path}:{digest}:{expected}"
            )
        observed[str(path.resolve())] = digest
    if not INTERNAL_IMAGE_ROOT.is_dir():
        raise FixedPolicyStop("STOP_TARGET_ADAPT_IMAGE_ROOT_MISSING")
    return observed


def _require_contract_value(value: Any, expected: Any, role: str) -> None:
    if value != expected:
        raise FixedPolicyStop(
            f"STOP_CONTRACT_VALUE:{role}:{value!r}:{expected!r}"
        )


def load_and_validate_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    _require_contract_value(contract.get("schema"), CONTRACT_SCHEMA, "schema")
    _require_contract_value(contract.get("experiment"), EXPERIMENT, "experiment")
    _require_contract_value(contract.get("status"), "FROZEN_BEFORE_MEASUREMENT", "status")
    authorization = contract.get("authorization", {})
    _require_contract_value(
        authorization.get("training_authorized"), False,
        "authorization.training_authorized",
    )
    _require_contract_value(
        authorization.get("formal_training_started"), False,
        "authorization.formal_training_started",
    )
    selector = contract.get("fixed_delta_selector", {})
    expected_selector = {
        "namespace": SELECTOR_NAMESPACE,
        "canonical_uid_field": (
            "Q2 canonical q2_uid, byte-for-byte equal to visit_native_deltas.target_uid"
        ),
        "hash_input_utf8": "NATIVE_A_FIXED_V1|<canonical_uid>",
        "digest": "SHA256(hash_input UTF-8 bytes)",
        "hash_integer": (
            "full 256-bit digest interpreted as unsigned big-endian integer"
        ),
        "selected_index": "hash_integer mod valid_delta_count",
        "retry": False,
        "ordering": "ascending formal_visit_index",
        "formal_visit_index_definition": (
            "(epoch - 72) * 4416 + target_visit_ordinal"
        ),
    }
    for key, expected in expected_selector.items():
        _require_contract_value(selector.get(key), expected, f"fixed_delta_selector.{key}")
    for key, expected in {
        "expected_target_unique_images": EXPECTED_IMAGES,
        "expected_images_with_valid_delta": EXPECTED_WITH_DELTA,
        "expected_images_without_valid_delta": EXPECTED_WITHOUT_DELTA,
        "expected_descriptor_valid_delta_rows": EXPECTED_CANDIDATES,
    }.items():
        _require_contract_value(selector.get(key), expected, f"fixed_delta_selector.{key}")
    exposure = contract.get("exposure_policy", {})
    _require_contract_value(exposure.get("resampled_branch"), BRANCH_NATIVE, "exposure.branch")
    _require_contract_value(
        exposure.get("resampled_active_definition"),
        "augment_on == true AND valid == true AND policy_flag == APPLY_OFFLINE_THETA",
        "exposure.active_definition",
    )
    for key, expected in {
        "expected_total_visits": EXPECTED_VISITS,
        "expected_active_visits": EXPECTED_ACTIVE,
        "expected_identity_visits": EXPECTED_IDENTITY,
        "required_mismatch_count": 0,
    }.items():
        _require_contract_value(exposure.get(key), expected, f"exposure_policy.{key}")
    realization = contract.get("fixed_realization", {})
    _require_contract_value(realization.get("expected_unique_solves"), EXPECTED_WITH_DELTA, "fixed_realization.expected_unique_solves")
    _require_contract_value(realization.get("default_workers"), FROZEN_WORKERS, "fixed_realization.default_workers")
    _require_contract_value(realization.get("solver_calls_per_selected_image"), 1, "fixed_realization.solver_calls_per_selected_image")
    locks = contract.get("upstream_locks", {})
    lock_map = {
        "descriptor_geometry_contract": (GEOMETRY_CONTRACT, EXPECTED_HASHES[GEOMETRY_CONTRACT]),
        "realization_contract": (REALIZATION_CONTRACT, EXPECTED_HASHES[REALIZATION_CONTRACT]),
        "semantic_resource_contract": (SEMANTIC_CONTRACT, EXPECTED_HASHES[SEMANTIC_CONTRACT]),
        "formal_target_schedule": (TARGET_SCHEDULE, EXPECTED_HASHES[TARGET_SCHEDULE]),
        "native_visit_policy_schedule": (SCHEDULE, EXPECTED_HASHES[SCHEDULE]),
        "native_descriptor_valid_deltas": (NATIVE_DELTAS, EXPECTED_HASHES[NATIVE_DELTAS]),
        "frozen_realization_core": (FROZEN_CORE, EXPECTED_HASHES[FROZEN_CORE]),
        "offline_cache_engine": (ENGINE, EXPECTED_HASHES[ENGINE]),
        "exact_transform_core": (EXACT_TRANSFORM_CORE, EXPECTED_HASHES[EXACT_TRANSFORM_CORE]),
        "semantic_module": (SEMANTIC_MODULE, EXPECTED_HASHES[SEMANTIC_MODULE]),
        "semantic_core": (SEMANTIC_CORE, EXPECTED_HASHES[SEMANTIC_CORE]),
    }
    for role, (expected_path, expected_sha) in lock_map.items():
        value = locks.get(role, {})
        declared_path = (ROOT / str(value.get("path", ""))).resolve()
        _require_contract_value(declared_path, expected_path.resolve(), f"upstream_locks.{role}.path")
        _require_contract_value(value.get("sha256"), expected_sha, f"upstream_locks.{role}.sha256")
    _require_contract_value(
        locks.get("q2_target_adapt_whitelist_sha256"), EXPECTED_HASHES[WHITELIST],
        "upstream_locks.q2_target_adapt_whitelist_sha256",
    )
    return contract


def _load_policy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whitelist = pd.read_csv(
        WHITELIST,
        dtype=str,
        usecols=["uid", "image_path", "image_sha256", "capture_event_id"],
    )
    if len(whitelist) != EXPECTED_IMAGES or whitelist["uid"].duplicated().any():
        raise FixedPolicyStop("STOP_WHITELIST_UID_CARDINALITY")
    schedule_columns = [
        "adaptation_seed", "epoch", "target_visit_ordinal", "schedule_uid",
        "q2_uid", "branch", "augment_on", "valid", "policy_flag",
        "failure_reason", "delta_sha256", *(f"delta_{i}" for i in range(7)),
    ]
    schedule = pd.read_parquet(SCHEDULE, columns=schedule_columns)
    schedule = schedule.loc[schedule["branch"].eq(BRANCH_NATIVE)].copy()
    if len(schedule) != EXPECTED_VISITS:
        raise FixedPolicyStop(f"STOP_NATIVE_SCHEDULE_COUNT:{len(schedule)}")
    schedule["augment_on"] = _strict_bool(schedule["augment_on"], "augment_on")
    schedule["valid"] = _strict_bool(schedule["valid"], "valid")
    schedule["formal_visit_index"] = [
        formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(schedule["epoch"], schedule["target_visit_ordinal"])
    ]
    schedule = schedule.sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)
    expected_indices = np.arange(EXPECTED_VISITS, dtype=np.int64)
    if not np.array_equal(schedule["formal_visit_index"].to_numpy(dtype=np.int64), expected_indices):
        raise FixedPolicyStop("STOP_FORMAL_VISIT_INDEX_NOT_COMPLETE_CONTIGUOUS")
    if int(schedule["adaptation_seed"].nunique()) != 1 or int(schedule["adaptation_seed"].iloc[0]) != 42:
        raise FixedPolicyStop("STOP_ADAPTATION_SEED_DRIFT")
    delta_columns = [
        "epoch", "target_visit_ordinal", "visit_key", "schedule_uid",
        "target_uid", "capture_event_id", "intended_geometric_dose",
        "native_delta_sha256", *DELTA_COLUMNS, *TARGET_COLUMNS,
    ]
    deltas = pd.read_parquet(NATIVE_DELTAS, columns=delta_columns)
    if len(deltas) != EXPECTED_CANDIDATES:
        raise FixedPolicyStop(f"STOP_NATIVE_DELTA_COUNT:{len(deltas)}")
    deltas["formal_visit_index"] = [
        formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(deltas["epoch"], deltas["target_visit_ordinal"])
    ]
    if deltas["formal_visit_index"].duplicated().any():
        raise FixedPolicyStop("STOP_NATIVE_DELTA_FORMAL_VISIT_DUPLICATE")
    descriptors = pd.read_parquet(
        IMAGE_DESCRIPTORS,
        columns=["domain", "uid", *ORIGINAL_COLUMNS],
        filters=[[('domain', '==', 'TARGET_ADAPT')]],
    )
    if len(descriptors) != EXPECTED_IMAGES or descriptors["uid"].duplicated().any():
        raise FixedPolicyStop("STOP_TARGET_ADAPT_DESCRIPTOR_CARDINALITY")
    return whitelist, schedule, deltas, descriptors


def _validate_candidate_lineage(
    whitelist: pd.DataFrame,
    schedule: pd.DataFrame,
    deltas: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> dict[str, Any]:
    uid_set = set(whitelist["uid"].astype(str))
    if set(schedule["q2_uid"].astype(str)) != uid_set:
        raise FixedPolicyStop("STOP_SCHEDULE_CANONICAL_UID_SET")
    if set(descriptors["uid"].astype(str)) != uid_set:
        raise FixedPolicyStop("STOP_DESCRIPTOR_CANONICAL_UID_SET")
    active = (
        schedule["augment_on"]
        & schedule["valid"]
        & schedule["policy_flag"].eq("APPLY_OFFLINE_THETA")
    )
    if int(active.sum()) != EXPECTED_ACTIVE:
        raise FixedPolicyStop(f"STOP_ACTIVE_COUNT:{int(active.sum())}")
    if not schedule.loc[active, "failure_reason"].eq("").all():
        raise FixedPolicyStop("STOP_ACTIVE_FAILURE_REASON_NONEMPTY")
    if not schedule.loc[~schedule["augment_on"], "policy_flag"].eq("IDENTITY_POLICY_OFF").all():
        raise FixedPolicyStop("STOP_POLICY_OFF_IDENTITY_DRIFT")
    if not schedule.loc[schedule["augment_on"] & ~schedule["valid"], "policy_flag"].eq("IDENTITY_FALLBACK").all():
        raise FixedPolicyStop("STOP_INVALID_FALLBACK_IDENTITY_DRIFT")
    left = schedule.loc[active].sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)
    right = deltas.sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(left["formal_visit_index"].to_numpy(), right["formal_visit_index"].to_numpy()):
        raise FixedPolicyStop("STOP_ACTIVE_DELTA_VISIT_SET_MISMATCH")
    if not left["schedule_uid"].astype(str).equals(right["schedule_uid"].astype(str)):
        raise FixedPolicyStop("STOP_ACTIVE_DELTA_SCHEDULE_UID_MISMATCH")
    if not left["q2_uid"].astype(str).equals(right["target_uid"].astype(str)):
        raise FixedPolicyStop("STOP_ACTIVE_DELTA_CANONICAL_UID_MISMATCH")
    schedule_delta = left[[f"delta_{i}" for i in range(7)]].to_numpy(dtype="<f8")
    native_delta = right[list(DELTA_COLUMNS)].to_numpy(dtype="<f8")
    if not np.array_equal(schedule_delta.view(np.uint64), native_delta.view(np.uint64)):
        raise FixedPolicyStop("STOP_ACTIVE_DELTA_FLOAT64_BITS_MISMATCH")
    if not left["delta_sha256"].astype(str).equals(right["native_delta_sha256"].astype(str)):
        raise FixedPolicyStop("STOP_ACTIVE_DELTA_SHA_MISMATCH")
    computed_hashes = [
        hashlib.sha256(
            DELTA_HASH_PREFIX + row.tobytes(order="C")
        ).hexdigest()
        for row in native_delta
    ]
    if computed_hashes != right["native_delta_sha256"].astype(str).tolist():
        raise FixedPolicyStop("STOP_NATIVE_DELTA_SHA_RECOMPUTE_MISMATCH")
    descriptor_index = descriptors.set_index("uid", verify_integrity=True)
    original = descriptor_index.loc[
        right["target_uid"].astype(str), list(ORIGINAL_COLUMNS)
    ].to_numpy(dtype="<f8")
    target = right[list(TARGET_COLUMNS)].to_numpy(dtype="<f8")
    computed_target = np.asarray(original + native_delta, dtype="<f8")
    if not np.array_equal(computed_target.view(np.uint64), target.view(np.uint64)):
        raise FixedPolicyStop("STOP_NATIVE_TARGET_NOT_EXACT_ORIGINAL_PLUS_DELTA")
    per_image = right.groupby("target_uid", sort=False).size()
    if len(per_image) != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_IMAGES_WITH_DELTA_COUNT")
    return {
        "native_schedule_rows": len(schedule),
        "nominal_augment_on_count": int(schedule["augment_on"].sum()),
        "descriptor_policy_valid_active_count": int(active.sum()),
        "identity_count": int((~active).sum()),
        "invalid_nominal_augmentation_identity_count": int(
            (schedule["augment_on"] & ~schedule["valid"]).sum()
        ),
        "candidate_row_count": len(right),
        "candidate_image_count": len(per_image),
        "candidate_delta_sha_unique_count": int(right["native_delta_sha256"].nunique()),
        "delta_float64_bits_exact": True,
        "delta_sha_recomputed_exact": True,
        "target_descriptor_addition_bits_exact": True,
    }


def build_policy_tables(
    whitelist: pd.DataFrame,
    schedule: pd.DataFrame,
    deltas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.Series]]:
    grouped = {
        str(uid): group.sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)
        for uid, group in deltas.groupby("target_uid", sort=False)
    }
    selected_rows: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for uid in sorted(whitelist["uid"].astype(str).tolist()):
        group = grouped.get(uid)
        count = 0 if group is None else len(group)
        hash_input, digest, integer = selector_digest(uid)
        indices = [] if group is None else group["formal_visit_index"].astype(int).tolist()
        delta_shas = [] if group is None else group["native_delta_sha256"].astype(str).tolist()
        list_sha = canonical_json_sha256(
            [{"formal_visit_index": index, "delta_sha256": sha}
             for index, sha in zip(indices, delta_shas)]
        )
        selected_index: int | None = None
        selected: pd.Series | None = None
        if count:
            selected_index = integer % count
            selected = group.iloc[selected_index]
            selected_rows[uid] = selected
        row: dict[str, Any] = {
            "schema": SELECTION_SCHEMA,
            "canonical_uid": uid,
            "valid_delta_count": count,
            "fixed_delta_exists": count > 0,
            "ordered_formal_visit_indices": json.dumps(indices, separators=(",", ":")),
            "ordered_delta_sha256": json.dumps(delta_shas, separators=(",", ":")),
            "valid_delta_list_sha256": list_sha,
            "hash_input_utf8": hash_input,
            "hash_input_sha256": digest,
            "hash_integer_decimal": str(integer),
            "selected_index": selected_index,
            "selected_formal_visit_index": None if selected is None else int(selected["formal_visit_index"]),
            "selected_epoch": None if selected is None else int(selected["epoch"]),
            "selected_target_visit_ordinal": None if selected is None else int(selected["target_visit_ordinal"]),
            "selected_schedule_uid": "" if selected is None else str(selected["schedule_uid"]),
            "selected_visit_key": "" if selected is None else str(selected["visit_key"]),
            "selected_delta_sha": "" if selected is None else str(selected["native_delta_sha256"]),
            "selected_intended_geometric_dose": None if selected is None else float(selected["intended_geometric_dose"]),
        }
        for index, column in enumerate(DELTA_COLUMNS):
            row[f"selected_delta_{index}"] = None if selected is None else float(selected[column])
        for index, column in enumerate(TARGET_COLUMNS):
            row[f"selected_target_std_{index}"] = None if selected is None else float(selected[column])
        rows.append(row)
    selection = pd.DataFrame(rows, columns=SELECTION_COLUMNS)
    for column in (
        "selected_index", "selected_formal_visit_index", "selected_epoch",
        "selected_target_visit_ordinal",
    ):
        selection[column] = pd.array(selection[column], dtype="Int64")
    if len(selection) != EXPECTED_IMAGES or selection["canonical_uid"].duplicated().any():
        raise FixedPolicyStop("STOP_SELECTION_CARDINALITY")
    exists = _strict_bool(selection["fixed_delta_exists"], "selection.fixed_delta_exists")
    if int(exists.sum()) != EXPECTED_WITH_DELTA or int((~exists).sum()) != EXPECTED_WITHOUT_DELTA:
        raise FixedPolicyStop("STOP_SELECTION_EXISTS_COUNTS")
    if selection.loc[exists, "selected_delta_sha"].nunique() != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_SELECTED_DELTA_NOT_UNIQUE")

    selected_sha = dict(zip(selection["canonical_uid"], selection["selected_delta_sha"]))
    exists_map = dict(zip(selection["canonical_uid"], exists))
    exposure = schedule.copy()
    exposure["resample_active"] = (
        exposure["augment_on"] & exposure["valid"]
        & exposure["policy_flag"].eq("APPLY_OFFLINE_THETA")
    )
    exposure["resample_identity"] = ~exposure["resample_active"]
    exposure["fixed_delta_exists"] = exposure["q2_uid"].map(exists_map)
    if exposure["fixed_delta_exists"].isna().any():
        raise FixedPolicyStop("STOP_EXPOSURE_SELECTION_JOIN_MISSING")
    if not exposure.loc[exposure["resample_active"], "fixed_delta_exists"].all():
        raise FixedPolicyStop("STOP_ACTIVE_VISIT_WITHOUT_FIXED_DELTA")
    exposure["fixed_active"] = exposure["resample_active"].copy()
    exposure["fixed_identity"] = ~exposure["fixed_active"]
    exposure["fixed_selected_delta_sha"] = exposure["q2_uid"].map(selected_sha).fillna("")
    exposure["fixed_delta_sha_used"] = np.where(
        exposure["fixed_active"], exposure["fixed_selected_delta_sha"], ""
    )
    exposure["active_match"] = exposure["fixed_active"].eq(exposure["resample_active"])
    exposure["identity_match"] = exposure["fixed_identity"].eq(exposure["resample_identity"])
    exposure_output = pd.DataFrame({
        "schema": EXPOSURE_SCHEMA,
        "formal_visit_index": exposure["formal_visit_index"].astype(np.int64),
        "adaptation_seed": exposure["adaptation_seed"].astype(np.int64),
        "epoch": exposure["epoch"].astype(np.int64),
        "target_visit_ordinal": exposure["target_visit_ordinal"].astype(np.int64),
        "schedule_uid": exposure["schedule_uid"].astype(str),
        "canonical_uid": exposure["q2_uid"].astype(str),
        "resample_augment_on": exposure["augment_on"].astype(bool),
        "resample_descriptor_policy_valid": exposure["valid"].astype(bool),
        "resample_policy_flag": exposure["policy_flag"].astype(str),
        "resample_failure_reason": exposure["failure_reason"].astype(str),
        "resample_active": exposure["resample_active"].astype(bool),
        "resample_identity": exposure["resample_identity"].astype(bool),
        "fixed_delta_exists": exposure["fixed_delta_exists"].astype(bool),
        "fixed_active": exposure["fixed_active"].astype(bool),
        "fixed_identity": exposure["fixed_identity"].astype(bool),
        "fixed_selected_delta_sha": exposure["fixed_selected_delta_sha"].astype(str),
        "fixed_delta_sha_used": exposure["fixed_delta_sha_used"].astype(str),
        "active_match": exposure["active_match"].astype(bool),
        "identity_match": exposure["identity_match"].astype(bool),
    }, columns=EXPOSURE_COLUMNS)
    mismatch = int((~exposure_output["active_match"] | ~exposure_output["identity_match"]).sum())
    if (
        len(exposure_output) != EXPECTED_VISITS
        or int(exposure_output["fixed_active"].sum()) != EXPECTED_ACTIVE
        or int(exposure_output["fixed_identity"].sum()) != EXPECTED_IDENTITY
        or mismatch != 0
    ):
        raise FixedPolicyStop("STOP_EXPOSURE_GLOBAL_EQUIVALENCE")
    per_image = exposure_output.groupby("canonical_uid", sort=True).agg(
        total_visits_resample=("resample_active", "size"),
        active_visits_resample=("resample_active", "sum"),
        active_visits_fixed=("fixed_active", "sum"),
        identity_visits_resample=("resample_identity", "sum"),
        identity_visits_fixed=("fixed_identity", "sum"),
    ).reset_index()
    per_epoch = exposure_output.groupby("epoch", sort=True).agg(
        total_visits_resample=("resample_active", "size"),
        active_visits_resample=("resample_active", "sum"),
        active_visits_fixed=("fixed_active", "sum"),
        identity_visits_resample=("resample_identity", "sum"),
        identity_visits_fixed=("fixed_identity", "sum"),
    ).reset_index()
    per_image["active_match"] = per_image["active_visits_resample"].eq(per_image["active_visits_fixed"])
    per_image["identity_match"] = per_image["identity_visits_resample"].eq(per_image["identity_visits_fixed"])
    per_epoch["active_match"] = per_epoch["active_visits_resample"].eq(per_epoch["active_visits_fixed"])
    per_epoch["identity_match"] = per_epoch["identity_visits_resample"].eq(per_epoch["identity_visits_fixed"])
    if not per_image[["active_match", "identity_match"]].to_numpy(dtype=bool).all():
        raise FixedPolicyStop("STOP_EXPOSURE_PER_IMAGE_EQUIVALENCE")
    if not per_epoch[["active_match", "identity_match"]].to_numpy(dtype=bool).all():
        raise FixedPolicyStop("STOP_EXPOSURE_PER_EPOCH_EQUIVALENCE")
    equivalence = {
        "schema": "m1_native_a_fixed_exposure_equivalence_v1",
        "status": "PASS",
        "total_visits_resample": EXPECTED_VISITS,
        "total_visits_fixed": len(exposure_output),
        "active_visits_resample": int(exposure_output["resample_active"].sum()),
        "active_visits_fixed": int(exposure_output["fixed_active"].sum()),
        "identity_visits_resample": int(exposure_output["resample_identity"].sum()),
        "identity_visits_fixed": int(exposure_output["fixed_identity"].sum()),
        "global_mismatch_count": mismatch,
        "per_image_mismatch_count": int((~per_image["active_match"] | ~per_image["identity_match"]).sum()),
        "per_epoch_mismatch_count": int((~per_epoch["active_match"] | ~per_epoch["identity_match"]).sum()),
        "per_image_active_counts_sha256": canonical_json_sha256(
            per_image[["canonical_uid", "active_visits_resample", "active_visits_fixed"]].to_dict("records")
        ),
        "per_epoch_counts": per_epoch.to_dict("records"),
        "descriptor_invalid_nominal_augmentation_remains_identity": True,
        "exposure_mask_exactly_matched_to_native_a_resample": True,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "TARGET_ADAPT_LABEL_READ_COUNT": 0,
        "TARGET_DEV_READ_COUNT": 0,
        "TARGET_TEST_READ_COUNT": 0,
        "DATASET_MUTATION_COUNT": 0,
    }
    return selection, exposure_output, equivalence, selected_rows


def _policy_paths(output: Path) -> dict[str, Path]:
    return {
        "selection": output / "fixed_a_image_delta_selection.csv",
        "exposure": output / "fixed_a_visit_exposure_audit.csv",
        "equivalence": output / "fixed_a_exposure_equivalence.json",
        "policy_audit": output / RAW_DIRNAME / "fixed_a_policy_build_audit.json",
    }


def write_or_validate_policy_outputs(
    output: Path,
    selection: pd.DataFrame,
    exposure: pd.DataFrame,
    equivalence: Mapping[str, Any],
    lineage_audit: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    contract_path: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> dict[str, Any]:
    paths = _policy_paths(output)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not (overwrite or resume):
        raise FixedPolicyStop(
            "STOP_POLICY_OUTPUT_EXISTS_USE_OVERWRITE_OR_RESUME:"
            + "|".join(str(path) for path in existing)
        )
    if resume and existing:
        required = set(paths.values())
        if set(existing) != required:
            raise FixedPolicyStop("STOP_RESUME_POLICY_OUTPUT_SET_INCOMPLETE")
        if (
            paths["selection"].read_bytes() != _csv_bytes(selection)
            or paths["exposure"].read_bytes() != _csv_bytes(exposure)
        ):
            raise FixedPolicyStop("STOP_RESUME_POLICY_OUTPUT_CONTENT_MISMATCH")
        if json.loads(paths["equivalence"].read_text(encoding="utf-8")) != dict(equivalence):
            raise FixedPolicyStop("STOP_RESUME_EXPOSURE_JSON_MISMATCH")
    else:
        atomic_csv(paths["selection"], selection)
        atomic_csv(paths["exposure"], exposure)
        atomic_json(paths["equivalence"], dict(equivalence))
    audit = {
        "schema": "m1_native_a_fixed_policy_build_audit_v1",
        "status": "PASS",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": CONTRACT_SHA256,
        "input_sha256": dict(input_hashes),
        "lineage": dict(lineage_audit),
        "selection_rows": len(selection),
        "images_with_fixed_delta": int(selection["fixed_delta_exists"].sum()),
        "images_without_fixed_delta": int((~selection["fixed_delta_exists"]).sum()),
        "exposure_rows": len(exposure),
        "active_visits": int(exposure["fixed_active"].sum()),
        "identity_visits": int(exposure["fixed_identity"].sum()),
        "selector_uses_realization_fields": False,
        "selector_uses_semantic_fields": False,
        "selector_uses_performance_fields": False,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "TARGET_ADAPT_LABEL_READ_COUNT": 0,
        "TARGET_DEV_READ_COUNT": 0,
        "TARGET_TEST_READ_COUNT": 0,
        "DATASET_MUTATION_COUNT": 0,
        "POWELL_CALL_COUNT": 0,
    }
    atomic_json(paths["policy_audit"], audit)
    return audit


def _normalization_and_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    schema = json.loads(DESCRIPTOR_SCHEMA.read_text(encoding="utf-8"))["standardization"]
    median = np.asarray([schema[name]["median"] for name in BASE], dtype="<f8")
    scale = np.asarray(
        [schema[name]["iqr"] + schema[name]["epsilon_scale"] for name in BASE],
        dtype="<f8",
    )
    events = pd.read_csv(EVENT_PROTOTYPES, dtype={"domain": str})
    events = events.loc[events["domain"].eq("TARGET_ADAPT")]
    event_std = events[[f"std__{name}" for name in BASE]].to_numpy(dtype="<f8")
    if event_std.shape != (2373, 7):
        raise FixedPolicyStop(f"STOP_SUPPORT_MATRIX_SHAPE:{event_std.shape}")
    return median, scale, event_std


def _internal_image_path(uid: str, whitelist_index: pd.DataFrame) -> Path:
    if uid not in whitelist_index.index:
        raise FixedPolicyStop(f"STOP_UID_NOT_IN_WHITELIST:{uid}")
    basename = Path(str(whitelist_index.at[uid, "image_path"])).name
    path = (INTERNAL_IMAGE_ROOT / basename).resolve()
    normalized = str(path).replace("\\", "/").casefold()
    if "/target/dev/" in normalized or "/target/test/" in normalized:
        raise FixedPolicyStop("STOP_FORBIDDEN_TARGET_SPLIT_PATH")
    if INTERNAL_IMAGE_ROOT.resolve() not in path.parents or not path.is_file():
        raise FixedPolicyStop(f"STOP_TARGET_ADAPT_IMAGE_MISSING:{path}")
    return path


def build_realization_plan(
    selection: pd.DataFrame,
    selected_rows: Mapping[str, pd.Series],
    whitelist: pd.DataFrame,
    descriptors: pd.DataFrame,
    config: ScientificConfigIdentity,
) -> tuple[ExecutionPlan, pd.DataFrame]:
    whitelist_index = whitelist.set_index("uid", verify_integrity=True)
    descriptors_index = descriptors.set_index("uid", verify_integrity=True)
    selection_index = selection.set_index("canonical_uid", verify_integrity=True)
    pairs = []
    for uid in sorted(selected_rows):
        selected = selected_rows[uid]
        selection_row = selection_index.loc[uid]
        original = descriptors_index.loc[uid, list(ORIGINAL_COLUMNS)].to_numpy(dtype="<f8")
        target = np.asarray(
            [selection_row[f"selected_target_std_{index}"] for index in range(7)],
            dtype="<f8",
        )
        delta_hash_material = np.asarray(
            [selection_row[f"selected_delta_{index}"] for index in range(7)],
            dtype="<f8",
        )
        path = _internal_image_path(uid, whitelist_index)
        declared_image_sha = str(whitelist_index.at[uid, "image_sha256"])
        request = make_explicit_target_solve_request(
            image_path=path,
            input_image_sha256=declared_image_sha,
            original_std=original,
            target_std=target,
            delta_hash_material_std=delta_hash_material,
            delta_sha256=str(selection_row["selected_delta_sha"]),
            intended_dose=float(selection_row["selected_intended_geometric_dose"]),
            config=config,
        )
        alias = VisitAlias(
            adaptation_seed=42,
            epoch=int(selected["epoch"]),
            target_visit_ordinal=int(selected["target_visit_ordinal"]),
            schedule_uid=str(selected["schedule_uid"]),
            q2_uid=uid,
            capture_event_id=str(selected["capture_event_id"]),
            branch=BRANCH_NATIVE,
            donor_schedule_uid=str(selected["schedule_uid"]),
            donor_q2_uid=uid,
            donor_event_id=str(selected["capture_event_id"]),
        )
        pairs.append((request, alias))
    plan = build_execution_plan(pairs)
    plan.validate()
    if (
        plan.total_solve_requests != EXPECTED_WITH_DELTA
        or plan.unique_exact_solve_count != EXPECTED_WITH_DELTA
        or plan.unique_namespaced_cache_count != EXPECTED_WITH_DELTA
    ):
        raise FixedPolicyStop("STOP_FIXED_REALIZATION_PLAN_COUNTS")
    aliases = pd.DataFrame(plan.visit_alias_rows())
    if len(aliases) != EXPECTED_WITH_DELTA or aliases["q2_uid"].duplicated().any():
        raise FixedPolicyStop("STOP_FIXED_REALIZATION_ALIAS_COUNTS")
    solve_rows = pd.DataFrame(plan.solve_key_rows())
    plan_rows = aliases.merge(
        solve_rows,
        on=["solve_key", "cache_namespace", "cache_key_sha256"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_solve"),
    )
    plan_rows = plan_rows.rename(columns={"q2_uid": "canonical_uid"})
    plan_rows["selected_delta_sha"] = plan_rows["canonical_uid"].map(
        selection_index["selected_delta_sha"]
    )
    return plan, plan_rows


def _chunk_plan(groups: Sequence[Any]) -> ExecutionPlan:
    solves = [solve for group in groups for solve in group.unique_solves]
    aliases = [alias for solve in solves for alias in solve.aliases]
    exact_keys = {solve.exact_solve_key_sha256 for solve in solves}
    cache_keys = {solve.cache_key_sha256 for solve in solves}
    plan = ExecutionPlan(
        groups=tuple(groups),
        total_solve_requests=len(aliases),
        unique_exact_solve_count=len(exact_keys),
        unique_namespaced_cache_count=len(cache_keys),
        subset_reuse_authorized=False,
        subset_evidence=None,
    )
    plan.validate()
    return plan


def _chunk_paths(checkpoint_dir: Path, index: int) -> dict[str, Path]:
    stem = f"chunk_{index:04d}"
    return {
        "science": checkpoint_dir / f"{stem}_science.parquet",
        "performance": checkpoint_dir / f"{stem}_performance.csv",
        "commit": checkpoint_dir / f"{stem}_commit.json",
    }


def _verify_committed_chunk(
    paths: Mapping[str, Path], chunk_plan: ExecutionPlan, index: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not paths["commit"].is_file():
        raise FixedPolicyStop(f"STOP_COMMITTED_CHUNK_MARKER_MISSING:{index}")
    commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    if commit.get("status") != "COMMITTED" or int(commit.get("chunk_index", -1)) != index:
        raise FixedPolicyStop(f"STOP_CHUNK_COMMIT_SCHEMA:{index}")
    expected_keys = sorted(row["cache_key_sha256"] for row in chunk_plan.solve_key_rows())
    if commit.get("contract_sha256") != CONTRACT_SHA256:
        raise FixedPolicyStop(f"STOP_CHUNK_CONTRACT_SHA:{index}")
    if commit.get("chunk_plan_sha256") != canonical_json_sha256(
        list(chunk_plan.solve_key_rows())
    ):
        raise FixedPolicyStop(f"STOP_CHUNK_PLAN_SHA:{index}")
    if commit.get("cache_key_list_sha256") != canonical_json_sha256(expected_keys):
        raise FixedPolicyStop(f"STOP_CHUNK_PLAN_KEY_SHA:{index}")
    if int(commit.get("unique_solve_count", -1)) != len(expected_keys):
        raise FixedPolicyStop(f"STOP_CHUNK_SOLVE_COUNT:{index}")
    for role in ("science", "performance"):
        path = paths[role]
        if not path.is_file() or sha256_file(path) != str(commit.get(f"{role}_sha256", "")):
            raise FixedPolicyStop(f"STOP_CHUNK_FILE_SHA:{index}:{role}")
    science = pd.read_parquet(paths["science"])
    performance = pd.read_csv(paths["performance"])
    observed_keys = sorted(science["cache_key_sha256"].astype(str).tolist())
    if observed_keys != expected_keys or len(science) != len(expected_keys):
        raise FixedPolicyStop(f"STOP_CHUNK_SCIENCE_KEY_SET:{index}")
    return science, performance, commit


def _aggregate_checkpoint_state(
    raw: Path,
    plan: ExecutionPlan,
    commits: Sequence[Mapping[str, Any]],
    science_frames: Sequence[pd.DataFrame],
    performance_frames: Sequence[pd.DataFrame],
    contract_path: Path,
    input_hashes: Mapping[str, str],
    started: float,
    total_chunks: int,
) -> None:
    science = (
        pd.concat(science_frames, ignore_index=True).sort_values("cache_key_sha256", kind="mergesort")
        if science_frames else pd.DataFrame()
    )
    performance = (
        pd.concat(performance_frames, ignore_index=True)
        if performance_frames else pd.DataFrame()
    )
    if not science.empty:
        atomic_parquet(raw / "fixed_a_science_cache.parquet", science)
    if not performance.empty:
        atomic_csv(raw / "fixed_a_realization_performance.csv", performance)
    events = [
        {
            "schema": "m1_native_a_fixed_realization_event_v1",
            "event": "chunk_committed",
            **dict(commit),
        }
        for commit in commits
    ]
    atomic_jsonl(raw / "fixed_a_realization_events.jsonl", events)
    completed = len(science)
    checkpoint = {
        "schema": "m1_native_a_fixed_realization_checkpoint_v1",
        "status": "COMPLETE" if completed == EXPECTED_WITH_DELTA else "PARTIAL_RESUMABLE",
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": CONTRACT_SHA256,
        "input_sha256": dict(input_hashes),
        "plan_sha256": canonical_json_sha256(list(plan.solve_key_rows())),
        "expected_unique_solves": EXPECTED_WITH_DELTA,
        "completed_unique_solves": completed,
        "committed_chunk_count": len(commits),
        "total_chunk_count": total_chunks,
        "runner_wall_seconds_current_process": time.perf_counter() - started,
        "workers": FROZEN_WORKERS,
        "persistent_worker_pool": True,
        "online_powell_calls": 0,
        "offline_powell_attempt_count": completed,
        "solver_non_success_reselection_count": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "TARGET_ADAPT_LABEL_READ_COUNT": 0,
        "TARGET_DEV_READ_COUNT": 0,
        "TARGET_TEST_READ_COUNT": 0,
        "DATASET_MUTATION_COUNT": 0,
    }
    atomic_json(raw / "fixed_a_realization_checkpoint.json", checkpoint)


def _materialize_theta_table(
    selection: pd.DataFrame,
    plan: ExecutionPlan,
    science: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    aliases = pd.DataFrame(plan.visit_alias_rows())
    if len(aliases) != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_THETA_ALIAS_COUNT")
    science_by_key = science.set_index("cache_key_sha256", verify_integrity=True)
    solve_performance = performance.loc[
        performance["record_type"].eq("unique_solve")
    ].copy()
    if (
        len(solve_performance) != EXPECTED_WITH_DELTA
        or solve_performance["cache_key_sha256"].duplicated().any()
    ):
        raise FixedPolicyStop("STOP_THETA_PERFORMANCE_SOLVE_COUNT")
    performance_by_key = solve_performance.set_index(
        "cache_key_sha256", verify_integrity=True
    )
    alias_by_uid = aliases.set_index("q2_uid", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for item in selection.to_dict("records"):
        uid = str(item["canonical_uid"])
        exists = bool(item["fixed_delta_exists"])
        row: dict[str, Any] = {
            "schema": THETA_SCHEMA,
            "canonical_uid": uid,
            "fixed_delta_exists": exists,
            "realization_status": (
                "SOLVED_RETAINED_NO_RESELECTION" if exists
                else "IDENTITY_NO_VALID_DELTA"
            ),
            "selected_index": item["selected_index"],
            "selected_formal_visit_index": item["selected_formal_visit_index"],
            "selected_epoch": item["selected_epoch"],
            "selected_target_visit_ordinal": item["selected_target_visit_ordinal"],
            "selected_schedule_uid": item["selected_schedule_uid"],
            "selected_delta_sha": item["selected_delta_sha"],
            "selected_intended_geometric_dose": item["selected_intended_geometric_dose"],
        }
        for column in THETA_SCIENCE_COLUMNS:
            row[column] = None if column not in {"solve_key", "cache_namespace", "cache_key_sha256"} else ""
        if exists:
            alias = alias_by_uid.loc[uid]
            cache_key = str(alias["cache_key_sha256"])
            if cache_key not in science_by_key.index:
                raise FixedPolicyStop(f"STOP_THETA_SCIENCE_ENTRY_MISSING:{uid}")
            entry = science_by_key.loc[cache_key]
            row["solve_key"] = str(alias["solve_key"])
            row["cache_namespace"] = str(alias["cache_namespace"])
            row["cache_key_sha256"] = cache_key
            for column in THETA_SCIENCE_COLUMNS:
                if column in {"solve_key", "cache_namespace", "cache_key_sha256"}:
                    continue
                if column == "elapsed_seconds":
                    if cache_key not in performance_by_key.index:
                        raise FixedPolicyStop(
                            f"STOP_THETA_PERFORMANCE_ENTRY_MISSING:{uid}"
                        )
                    row[column] = performance_by_key.at[
                        cache_key, "solver_reported_elapsed_seconds"
                    ]
                    continue
                if column not in entry.index:
                    raise FixedPolicyStop(f"STOP_THETA_SCIENCE_FIELD_MISSING:{column}")
                row[column] = entry[column]
        rows.append(row)
    result = pd.DataFrame(rows, columns=THETA_COLUMNS)
    for column in (
        "selected_index", "selected_formal_visit_index", "selected_epoch",
        "selected_target_visit_ordinal",
    ):
        result[column] = pd.array(result[column], dtype="Int64")
    if len(result) != EXPECTED_IMAGES:
        raise FixedPolicyStop("STOP_THETA_TABLE_CARDINALITY")
    solved = result["realization_status"].eq("SOLVED_RETAINED_NO_RESELECTION")
    if int(solved.sum()) != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_THETA_SOLVED_COUNT")
    if result.loc[solved, "cache_key_sha256"].nunique() != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_THETA_UNIQUE_SOLVE_COUNT")
    return result


def execute_realizations(
    output: Path,
    contract_path: Path,
    input_hashes: Mapping[str, str],
    selection: pd.DataFrame,
    selected_rows: Mapping[str, pd.Series],
    whitelist: pd.DataFrame,
    descriptors: pd.DataFrame,
    *,
    resume: bool,
    groups_per_checkpoint: int,
) -> dict[str, Any]:
    if groups_per_checkpoint <= 0:
        raise FixedPolicyStop("STOP_CHECKPOINT_GROUP_COUNT_NONPOSITIVE")
    raw = output / RAW_DIRNAME
    checkpoint_dir = raw / "realization_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    subset = json.loads(SUBSET_AUDIT.read_text(encoding="utf-8"))
    config = ScientificConfigIdentity.from_config_mapping(subset["configs"])
    if config.as_dict() != EXPECTED_CONFIG_SHAS:
        raise FixedPolicyStop("STOP_SCIENTIFIC_CONFIG_IDENTITY_DRIFT")
    plan, plan_rows = build_realization_plan(
        selection, selected_rows, whitelist, descriptors, config
    )
    atomic_parquet(raw / "fixed_a_solve_plan.parquet", plan_rows)
    median, scale, event_std = _normalization_and_support()
    runtime = build_frozen_runtime_spec(
        realization_core_path=FROZEN_CORE,
        q2_snapshot_dir=Q2_SNAPSHOT,
        allowed_image_roots=[INTERNAL_IMAGE_ROOT],
        median=median,
        scale=scale,
        event_std=event_std,
        subset_evidence=subset,
        descriptor_schema_path=DESCRIPTOR_SCHEMA,
        image_descriptors_path=IMAGE_DESCRIPTORS,
        event_prototypes_path=EVENT_PROTOTYPES,
    )
    groups = list(plan.groups)
    chunks = [groups[index:index + groups_per_checkpoint]
              for index in range(0, len(groups), groups_per_checkpoint)]
    chunk_plans = [_chunk_plan(chunk) for chunk in chunks]
    science_frames: list[pd.DataFrame] = []
    performance_frames: list[pd.DataFrame] = []
    commits: list[dict[str, Any]] = []
    first_pending = 0
    for index, chunk_plan in enumerate(chunk_plans):
        paths = _chunk_paths(checkpoint_dir, index)
        if paths["commit"].exists():
            if not resume:
                raise FixedPolicyStop(f"STOP_REALIZATION_COMMIT_EXISTS_USE_RESUME:{index}")
            science, performance, commit = _verify_committed_chunk(paths, chunk_plan, index)
            science_frames.append(science)
            performance_frames.append(performance)
            commits.append(commit)
            first_pending = index + 1
        else:
            break
    for index in range(first_pending, len(chunk_plans)):
        if _chunk_paths(checkpoint_dir, index)["commit"].exists():
            raise FixedPolicyStop("STOP_NONCONTIGUOUS_REALIZATION_COMMITS")
    started = time.perf_counter()
    pool = PersistentSpawnPool(runtime, max_workers=FROZEN_WORKERS, chunksize=1)
    try:
        for index in range(first_pending, len(chunk_plans)):
            chunk_plan = chunk_plans[index]
            paths = _chunk_paths(checkpoint_dir, index)
            outputs = pool.execute(chunk_plan)
            science = pd.DataFrame(outputs.science_cache_entries)
            performance = pd.DataFrame(outputs.performance_rows)
            performance.insert(0, "chunk_index", index)
            expected_count = chunk_plan.unique_namespaced_cache_count
            if len(science) != expected_count:
                raise FixedPolicyStop(f"STOP_CHUNK_SCIENCE_COUNT:{index}")
            atomic_parquet(paths["science"], science)
            atomic_csv(paths["performance"], performance)
            expected_keys = sorted(row["cache_key_sha256"] for row in chunk_plan.solve_key_rows())
            commit = {
                "status": "COMMITTED",
                "chunk_index": index,
                "contract_sha256": CONTRACT_SHA256,
                "chunk_plan_sha256": canonical_json_sha256(
                    list(chunk_plan.solve_key_rows())
                ),
                "image_group_count": len(chunk_plan.groups),
                "unique_solve_count": expected_count,
                "cache_key_list_sha256": canonical_json_sha256(expected_keys),
                "science_sha256": sha256_file(paths["science"]),
                "performance_sha256": sha256_file(paths["performance"]),
                "solver_non_success_count": int((~science["solver_success"].astype(bool)).sum()),
                "solver_non_success_reselection_count": 0,
                "wall_seconds": float(
                    performance.loc[
                        performance["record_type"].eq("execution_batch"),
                        "execution_batch_wall_seconds",
                    ].iloc[0]
                ),
            }
            atomic_json(paths["commit"], commit)
            science_frames.append(science)
            performance_frames.append(performance)
            commits.append(commit)
            _aggregate_checkpoint_state(
                raw, plan, commits, science_frames, performance_frames,
                contract_path, input_hashes, started, len(chunk_plans),
            )
    finally:
        pool.close(cancel_futures=False)
    science_all = pd.concat(science_frames, ignore_index=True).sort_values(
        "cache_key_sha256", kind="mergesort"
    ).reset_index(drop=True)
    if len(science_all) != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_FINAL_SCIENCE_CACHE_COUNT")
    # Exact engine validation of every resumed and newly committed cache entry.
    validator_pool = PersistentSpawnPool(runtime, max_workers=FROZEN_WORKERS, chunksize=1)
    try:
        validated = validator_pool.execute(
            plan, existing_science_cache_entries=science_all.to_dict("records")
        )
    finally:
        validator_pool.close()
    if len(validated.science_cache_entries) != EXPECTED_WITH_DELTA:
        raise FixedPolicyStop("STOP_FINAL_ENGINE_CACHE_VALIDATION")
    atomic_parquet(raw / "fixed_a_science_cache.parquet", science_all)
    performance_all = pd.concat(performance_frames, ignore_index=True)
    theta = _materialize_theta_table(
        selection, plan, science_all, performance_all
    )
    atomic_csv(output / "fixed_a_theta_realization.csv", theta)
    total_wall = time.perf_counter() - started
    aggregate_execution_wall = float(
        sum(float(commit["wall_seconds"]) for commit in commits)
    )
    solver_non_success = int((~science_all["solver_success"].astype(bool)).sum())
    audit = {
        "schema": "m1_native_a_fixed_realization_audit_v1",
        "status": "PASS",
        "contract_sha256": CONTRACT_SHA256,
        "workers": FROZEN_WORKERS,
        "persistent_worker_pool": True,
        "checkpoint_group_count": groups_per_checkpoint,
        "image_group_count": len(plan.groups),
        "selected_image_count": EXPECTED_WITH_DELTA,
        "unique_exact_solve_count": plan.unique_exact_solve_count,
        "unique_namespaced_solve_count": plan.unique_namespaced_cache_count,
        "offline_powell_attempt_count": EXPECTED_WITH_DELTA,
        "solver_calls_per_selected_image": 1,
        "solver_non_success_count": solver_non_success,
        "solver_non_success_reselection_count": 0,
        "realization_or_semantic_rejection_count": 0,
        "wall_seconds_current_process_including_resume_validation": total_wall,
        "full_measured_execution_batch_wall_seconds": aggregate_execution_wall,
        "measured_solves_per_second": (
            EXPECTED_WITH_DELTA / aggregate_execution_wall
            if aggregate_execution_wall > 0 else None
        ),
        "science_cache_path": str((raw / "fixed_a_science_cache.parquet").resolve()),
        "science_cache_sha256": sha256_file(raw / "fixed_a_science_cache.parquet"),
        "theta_table_path": str((output / "fixed_a_theta_realization.csv").resolve()),
        "theta_table_sha256": sha256_file(output / "fixed_a_theta_realization.csv"),
        "online_powell_calls": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "TARGET_ADAPT_LABEL_READ_COUNT": 0,
        "TARGET_DEV_READ_COUNT": 0,
        "TARGET_TEST_READ_COUNT": 0,
        "DATASET_MUTATION_COUNT": 0,
        "SCIENCE_CHANGE": True,
        "SCIENCE_CHANGE_SCOPE": "PER_IMAGE_FIXED_POLICY_VS_PER_VISIT_RESAMPLED_POLICY",
    }
    atomic_json(raw / "fixed_a_realization_audit.json", audit)
    return audit


def _validate_output_root(output: Path) -> Path:
    resolved = output.resolve()
    results_root = (ROOT / "project" / "results").resolve()
    if results_root not in resolved.parents:
        raise FixedPolicyStop(f"STOP_OUTPUT_OUTSIDE_PROJECT_RESULTS:{resolved}")
    return resolved


def run(
    contract_path: Path,
    output: Path,
    *,
    build_policy_only: bool,
    overwrite: bool,
    resume: bool,
    groups_per_checkpoint: int,
) -> int:
    if overwrite and resume:
        raise FixedPolicyStop("STOP_OVERWRITE_AND_RESUME_MUTUALLY_EXCLUSIVE")
    output = _validate_output_root(output)
    contract_path = contract_path.resolve()
    if contract_path.parent != output:
        raise FixedPolicyStop("STOP_CONTRACT_NOT_IN_OUTPUT_ROOT")
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = _verify_locked_inputs(contract_path)
    load_and_validate_contract(contract_path)
    whitelist, schedule, deltas, descriptors = _load_policy_inputs()
    lineage = _validate_candidate_lineage(whitelist, schedule, deltas, descriptors)
    selection, exposure, equivalence, selected_rows = build_policy_tables(
        whitelist, schedule, deltas
    )
    policy_audit = write_or_validate_policy_outputs(
        output, selection, exposure, equivalence, lineage, input_hashes,
        contract_path, overwrite=overwrite, resume=resume,
    )
    if build_policy_only:
        print(json.dumps({
            "status": "PASS_POLICY_ONLY",
            "selection_rows": len(selection),
            "exposure_rows": len(exposure),
            "powell_calls": 0,
            "policy_audit": policy_audit,
            "TRAINING_AUTHORIZED": False,
            "FORMAL_TRAINING_STARTED": False,
        }, indent=2, sort_keys=True, default=_json_default))
        return 0
    audit = execute_realizations(
        output, contract_path, input_hashes, selection, selected_rows,
        whitelist, descriptors, resume=resume,
        groups_per_checkpoint=groups_per_checkpoint,
    )
    print(json.dumps(audit, indent=2, sort_keys=True, default=_json_default))
    return 0


def self_check() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    ast.parse(source, filename=str(Path(__file__)))
    engine = implementation_unit_tests()
    if engine.get("status") != "PASS" or int(engine.get("powell_calls", -1)) != 0:
        raise FixedPolicyStop("STOP_ENGINE_SELF_CHECK_NOT_PURE")
    first = selector_digest("img_fixed_a_selfcheck")
    second = selector_digest("img_fixed_a_selfcheck")
    if first != second or len(first[1]) != 64:
        raise FixedPolicyStop("STOP_SELECTOR_SELF_CHECK")
    if formal_visit_index(72, 0) != 0 or formal_visit_index(100, 4415) != 128063:
        raise FixedPolicyStop("STOP_FORMAL_VISIT_INDEX_SELF_CHECK")
    forbidden_calls = {"fit", "train", "backward", "optimizer_step"}
    parsed = ast.parse(source)
    observed_names = {
        node.func.attr if isinstance(node.func, ast.Attribute)
        else node.func.id if isinstance(node.func, ast.Name) else ""
        for node in ast.walk(parsed) if isinstance(node, ast.Call)
    }
    if forbidden_calls & observed_names:
        raise FixedPolicyStop("STOP_FORBIDDEN_TRAINING_CALL_IN_RUNNER")
    return {
        "schema": "m1_native_a_fixed_policy_realization_self_check_v1",
        "status": "PASS",
        "ast_parse_pass": True,
        "contract_sha256_expected": CONTRACT_SHA256,
        "selector_digest_deterministic": True,
        "formal_visit_index_boundaries_exact": True,
        "expected_images": EXPECTED_IMAGES,
        "expected_selected_solves": EXPECTED_WITH_DELTA,
        "expected_visits": EXPECTED_VISITS,
        "workers_frozen": FROZEN_WORKERS,
        "engine_self_check": engine,
        "image_reads": 0,
        "powell_calls": 0,
        "training_steps": 0,
        "TRAINING_AUTHORIZED": False,
        "FORMAL_TRAINING_STARTED": False,
        "TARGET_ADAPT_LABEL_READ_COUNT": 0,
        "TARGET_DEV_READ_COUNT": 0,
        "TARGET_TEST_READ_COUNT": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--build-policy-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--groups-per-checkpoint", type=int,
        default=DEFAULT_GROUPS_PER_CHECKPOINT,
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.root.resolve() != ROOT.resolve():
        raise FixedPolicyStop("STOP_WORKING_ROOT_MISMATCH")
    if args.self_check:
        print(json.dumps(self_check(), indent=2, sort_keys=True, default=_json_default))
        return 0
    if args.contract is None or args.output_dir is None:
        parser.error("--contract and --output-dir are required unless --self-check is used")
    try:
        return run(
            args.contract, args.output_dir,
            build_policy_only=args.build_policy_only,
            overwrite=args.overwrite,
            resume=args.resume,
            groups_per_checkpoint=args.groups_per_checkpoint,
        )
    except Exception as error:
        failure = {
            "schema": "m1_native_a_fixed_policy_realization_failure_v1",
            "status": "FAILED_BEFORE_GATE_DECISION",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "TRAINING_AUTHORIZED": False,
            "FORMAL_TRAINING_STARTED": False,
            "TARGET_ADAPT_LABEL_READ_COUNT": 0,
            "TARGET_DEV_READ_COUNT": 0,
            "TARGET_TEST_READ_COUNT": 0,
            "DATASET_MUTATION_COUNT": 0,
        }
        try:
            output = _validate_output_root(args.output_dir)
            atomic_json(output / RAW_DIRNAME / "fixed_a_policy_realization_failure.json", failure)
        except Exception:
            pass
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
