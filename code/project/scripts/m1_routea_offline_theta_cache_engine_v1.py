from __future__ import annotations

"""Exact, execution-only offline theta-cache engine for frozen Route A.

This module does not define a new photometric method.  It calls the protected
``realization_case`` implementation once for every globally unique exact
solve key.  Visit metadata is emitted as a separate alias table; it is never
mixed into the scientific cache entry.

The public execution API uses a persistent ``spawn`` process pool.  All
requests for one input-image SHA are one worker task, so that the file hash,
decode, and ``PaletteImage`` invariant preparation each occur once per image.
The caller must build one complete plan before execution.  This module has no
training entry point and never performs an online solve.
"""

import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ENGINE_SCHEMA = "m1_routea_offline_theta_cache_engine_v1"
EXACT_SOLVE_KEY_SCHEMA = "m1_routea_exact_solve_key_v1"
CACHE_KEY_SCHEMA = "m1_routea_namespaced_cache_key_v1"
CACHE_ENTRY_SCHEMA = "m1_routea_offline_theta_science_entry_v1"
VISIT_ALIAS_SCHEMA = "m1_routea_offline_theta_visit_alias_v1"
SUBSET_AUDIT_SCHEMA = "m1_routea_solve_key_subset_audit_v1"

FROZEN_REALIZATION_CORE_SHA256 = (
    "45b62bf0a192e2d527f3171661d36b19d2dd7da21c63afb93140015251b449c1"
)
FROZEN_ADAPTATION_SEED = 42
FROZEN_FIRST_EPOCH = 72
FROZEN_LAST_EPOCH = 100
FROZEN_TARGET_VISITS_PER_EPOCH = 4416

BRANCH_NATIVE = "NATIVE_A"
BRANCH_MATCHED = "MATCHED_A"
BRANCH_SMG = "SMG"
ALLOWED_BRANCHES = (BRANCH_NATIVE, BRANCH_MATCHED, BRANCH_SMG)
NATIVE_CACHE_NAMESPACE = "M1_ROUTEA_NATIVE_EXACT_CACHE_V1"
MATCHED_ISOLATED_NAMESPACE = "M1_ROUTEA_MATCHED_ISOLATED_CACHE_V1"
SMG_CACHE_NAMESPACE = "M1_ROUTEA_SMG_STRICTLY_ISOLATED_CACHE_V1"

BASE = (
    "tone_ln_y_q10",
    "tone_ln_y_q50",
    "tone_ln_y_q90",
    "contrast_global_q90_minus_q10",
    "contrast_local_grid8_median_q90_minus_q10",
    "chroma_cstar_q50",
    "chroma_cstar_q90",
)
WEIGHTS = np.asarray(
    [1 / 9, 1 / 9, 1 / 9, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    dtype=np.float64,
)

DELTA_HASH_PREFIX = b"M1_ROUTEA_DELTA_FLOAT64_LE_V1\0"
TARGET_DESCRIPTOR_HASH_PREFIX = (
    b"M1_ROUTEA_SOLVE_TARGET_DESCRIPTOR_FLOAT64_LE_V1\0"
)
ORIGINAL_DESCRIPTOR_HASH_PREFIX = (
    b"M1_ROUTEA_SOLVE_ORIGINAL_DESCRIPTOR_FLOAT64_LE_V1\0"
)
CONTINUOUS_STD_HASH_PREFIX = b"M1_ROUTEA_CONTINUOUS_STD_V1\0"
DEPLOYMENT_STD_HASH_PREFIX = b"M1_ROUTEA_DEPLOYMENT_STD_V1\0"
RAW_DESCRIPTOR_HASH_PREFIX = b"M1_ROUTEA_RAW_DESCRIPTOR_FLOAT64_LE_V1\0"
REALIZED_DELTA_HASH_PREFIX = b"M1_ROUTEA_REALIZED_DELTA_FLOAT64_LE_V1\0"
THETA_HASH_PREFIX = b"M1_ROUTEA_THETA_RAW_FLOAT64_LE_V1\0"
SUPPORT_MATRIX_HASH_PREFIX = b"M1_ROUTEA_SUPPORT_STD_MATRIX_FLOAT64_LE_V1\0"
CACHE_KEY_HASH_PREFIX = b"M1_ROUTEA_NAMESPACED_CACHE_KEY_V1\0"

EXACT_SOLVE_KEY_COMPONENTS = (
    "image_sha",
    "Delta_sha",
    "target_descriptor_sha",
    "solver_config_sha",
    "operator_config_sha",
    "original_descriptor_sha",
    "descriptor_config_sha",
    "realization_core_sha",
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExactCacheStop(RuntimeError):
    """Fail-closed cache, evidence, frozen-code, or representation stop."""


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_compatible(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, role: str) -> str:
    text = str(value)
    if _HEX_SHA256.fullmatch(text) is None:
        raise ExactCacheStop(f"STOP_INVALID_SHA256:{role}:{text}")
    return text


def _f8_vector(values: Sequence[float], role: str, length: int = 7) -> np.ndarray:
    array = np.asarray(values, dtype="<f8")
    if array.shape != (length,):
        raise ExactCacheStop(f"STOP_VECTOR_SHAPE:{role}:{array.shape}")
    if not np.isfinite(array).all():
        raise ExactCacheStop(f"STOP_VECTOR_NONFINITE:{role}")
    return array


def vector_sha256(values: Sequence[float], prefix: bytes, role: str) -> str:
    array = _f8_vector(values, role)
    return hashlib.sha256(prefix + array.tobytes(order="C")).hexdigest()


def theta_raw_f8_sha256(theta: Sequence[float]) -> str:
    array = _f8_vector(theta, "theta", length=3)
    return hashlib.sha256(THETA_HASH_PREFIX + array.tobytes(order="C")).hexdigest()


def support_matrix_sha256(values: Sequence[Sequence[float]]) -> str:
    matrix = np.asarray(values, dtype="<f8")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 7:
        raise ExactCacheStop(f"STOP_SUPPORT_MATRIX_SHAPE:{matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ExactCacheStop("STOP_SUPPORT_MATRIX_NONFINITE")
    digest = hashlib.sha256()
    digest.update(SUPPORT_MATRIX_HASH_PREFIX)
    digest.update(str(tuple(int(value) for value in matrix.shape)).encode("ascii"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _float64_bytes(value: float) -> bytes:
    array = np.asarray([value], dtype="<f8")
    if not np.isfinite(array).all():
        raise ExactCacheStop("STOP_NONFINITE_FLOAT64")
    return array.tobytes(order="C")


@dataclass(frozen=True, order=True)
class ScientificConfigIdentity:
    solver_config_sha256: str
    operator_config_sha256: str
    descriptor_config_sha256: str
    realization_core_sha256: str = FROZEN_REALIZATION_CORE_SHA256

    def validate(self) -> None:
        _require_sha256(self.solver_config_sha256, "solver_config")
        _require_sha256(self.operator_config_sha256, "operator_config")
        _require_sha256(self.descriptor_config_sha256, "descriptor_config")
        _require_sha256(self.realization_core_sha256, "realization_core")
        if self.realization_core_sha256 != FROZEN_REALIZATION_CORE_SHA256:
            raise ExactCacheStop("STOP_NONFROZEN_REALIZATION_CORE_IDENTITY")

    @classmethod
    def from_config_mapping(
        cls, value: Mapping[str, Any]
    ) -> "ScientificConfigIdentity":
        for payload_name, sha_name in (
            ("solver_config", "solver_config_sha256"),
            ("operator_config", "operator_config_sha256"),
            ("descriptor_config", "descriptor_config_sha256"),
        ):
            if payload_name not in value or sha_name not in value:
                raise ExactCacheStop(f"STOP_CONFIG_EVIDENCE_FIELD_MISSING:{payload_name}")
            observed = canonical_json_sha256(value[payload_name])
            if observed != str(value[sha_name]):
                raise ExactCacheStop(f"STOP_CONFIG_PAYLOAD_SHA_MISMATCH:{payload_name}")
        core_values = {
            str(value["solver_config"].get("frozen_realization_core_sha256", "")),
            str(value["operator_config"].get("frozen_realization_core_sha256", "")),
        }
        if core_values != {FROZEN_REALIZATION_CORE_SHA256}:
            raise ExactCacheStop("STOP_CONFIG_PAYLOAD_CORE_SHA_MISMATCH")
        q2_code_sets = [
            value[name].get("q2_code_sha256")
            for name in ("solver_config", "operator_config", "descriptor_config")
            if value[name].get("q2_code_sha256") is not None
        ]
        if (
            len(q2_code_sets) < 2
            or any(not isinstance(item, Mapping) for item in q2_code_sets)
            or any(dict(item) != dict(q2_code_sets[0]) for item in q2_code_sets[1:])
        ):
            raise ExactCacheStop("STOP_CONFIG_PAYLOAD_Q2_CODE_IDENTITY_MISMATCH")
        result = cls(
            solver_config_sha256=str(value["solver_config_sha256"]),
            operator_config_sha256=str(value["operator_config_sha256"]),
            descriptor_config_sha256=str(value["descriptor_config_sha256"]),
        )
        result.validate()
        return result

    def as_dict(self) -> dict[str, str]:
        return {
            "solver_config_sha256": self.solver_config_sha256,
            "operator_config_sha256": self.operator_config_sha256,
            "descriptor_config_sha256": self.descriptor_config_sha256,
            "realization_core_sha256": self.realization_core_sha256,
        }


@dataclass(frozen=True, order=True)
class ExactSolveKey:
    image_sha: str
    Delta_sha: str
    target_descriptor_sha: str
    solver_config_sha: str
    operator_config_sha: str
    original_descriptor_sha: str
    descriptor_config_sha: str
    realization_core_sha: str
    schema: str = EXACT_SOLVE_KEY_SCHEMA

    def validate(self) -> None:
        if self.schema != EXACT_SOLVE_KEY_SCHEMA:
            raise ExactCacheStop(f"STOP_SOLVE_KEY_SCHEMA:{self.schema}")
        for field in EXACT_SOLVE_KEY_COMPONENTS:
            _require_sha256(str(getattr(self, field)), f"solve_key.{field}")
        if self.realization_core_sha != FROZEN_REALIZATION_CORE_SHA256:
            raise ExactCacheStop("STOP_SOLVE_KEY_CORE_NOT_FROZEN")

    def as_dict(self) -> dict[str, str]:
        # Field spelling and hashing deliberately match the frozen key audit.
        return {
            "schema": self.schema,
            "image_sha": self.image_sha,
            "Delta_sha": self.Delta_sha,
            "target_descriptor_sha": self.target_descriptor_sha,
            "solver_config_sha": self.solver_config_sha,
            "operator_config_sha": self.operator_config_sha,
            "original_descriptor_sha": self.original_descriptor_sha,
            "descriptor_config_sha": self.descriptor_config_sha,
            "realization_core_sha": self.realization_core_sha,
        }

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_json_sha256(self.as_dict())


@dataclass(frozen=True)
class SolveRequest:
    """Scientific solve input only; no visit or branch metadata."""

    image_path: str
    input_image_sha256: str
    original_std: tuple[float, ...]
    delta_std: tuple[float, ...]
    delta_hash_material_std: tuple[float, ...]
    target_std: tuple[float, ...]
    delta_sha256: str
    delta_sha_is_frozen_declared_lineage: bool
    original_descriptor_sha256: str
    target_descriptor_sha256: str
    intended_dose: float
    intended_dose_is_explicit_frozen_input: bool
    config: ScientificConfigIdentity

    def validate(self) -> None:
        if not str(self.image_path):
            raise ExactCacheStop("STOP_EMPTY_IMAGE_PATH")
        _require_sha256(self.input_image_sha256, "input_image")
        original = _f8_vector(self.original_std, "original_std")
        delta = _f8_vector(self.delta_std, "delta_std")
        delta_hash_material = _f8_vector(
            self.delta_hash_material_std, "delta_hash_material_std"
        )
        target = _f8_vector(self.target_std, "target_std")
        computed_target = np.asarray(original + delta, dtype="<f8")
        if computed_target.tobytes(order="C") != target.tobytes(order="C"):
            raise ExactCacheStop("STOP_TARGET_DESCRIPTOR_NOT_EXACT_ORIGINAL_PLUS_DELTA")
        expected_delta_sha = vector_sha256(
            delta_hash_material, DELTA_HASH_PREFIX, "delta_hash_material_std"
        )
        declared_lineage = _strict_bool(
            self.delta_sha_is_frozen_declared_lineage,
            "delta_sha_is_frozen_declared_lineage",
        )
        if expected_delta_sha != self.delta_sha256 and not declared_lineage:
            raise ExactCacheStop("STOP_DELTA_SHA_MISMATCH")
        expected_original_sha = vector_sha256(
            original, ORIGINAL_DESCRIPTOR_HASH_PREFIX, "original_std"
        )
        if expected_original_sha != self.original_descriptor_sha256:
            raise ExactCacheStop("STOP_ORIGINAL_DESCRIPTOR_SHA_MISMATCH")
        expected_target_sha = vector_sha256(
            target, TARGET_DESCRIPTOR_HASH_PREFIX, "target_std"
        )
        if expected_target_sha != self.target_descriptor_sha256:
            raise ExactCacheStop("STOP_TARGET_DESCRIPTOR_SHA_MISMATCH")
        if not math.isfinite(float(self.intended_dose)):
            raise ExactCacheStop("STOP_INTENDED_DOSE_NONFINITE")
        if not _strict_bool(
            self.intended_dose_is_explicit_frozen_input,
            "intended_dose_is_explicit_frozen_input",
        ):
            expected_dose = float(np.sum(np.abs(delta) * WEIGHTS, dtype=np.float64))
            if _float64_bytes(expected_dose) != _float64_bytes(
                float(self.intended_dose)
            ):
                raise ExactCacheStop("STOP_INTENDED_DOSE_MISMATCH")
        self.config.validate()

    @property
    def exact_key(self) -> ExactSolveKey:
        self.validate()
        return ExactSolveKey(
            image_sha=self.input_image_sha256,
            Delta_sha=self.delta_sha256,
            target_descriptor_sha=self.target_descriptor_sha256,
            solver_config_sha=self.config.solver_config_sha256,
            operator_config_sha=self.config.operator_config_sha256,
            original_descriptor_sha=self.original_descriptor_sha256,
            descriptor_config_sha=self.config.descriptor_config_sha256,
            realization_core_sha=self.config.realization_core_sha256,
        )

    def exact_science_payload(self) -> tuple[Any, ...]:
        """Collision-resistant equality check using raw float64 bytes."""

        self.validate()
        return (
            self.input_image_sha256,
            _f8_vector(self.original_std, "original_std").tobytes(order="C"),
            _f8_vector(self.delta_std, "delta_std").tobytes(order="C"),
            _f8_vector(
                self.delta_hash_material_std, "delta_hash_material_std"
            ).tobytes(order="C"),
            _f8_vector(self.target_std, "target_std").tobytes(order="C"),
            bool(self.delta_sha_is_frozen_declared_lineage),
            _float64_bytes(float(self.intended_dose)),
            bool(self.intended_dose_is_explicit_frozen_input),
            self.config,
        )


def make_solve_request(
    *,
    image_path: str | Path,
    input_image_sha256: str,
    original_std: Sequence[float],
    delta_std: Sequence[float],
    delta_sha256: str,
    config: ScientificConfigIdentity,
) -> SolveRequest:
    original = _f8_vector(original_std, "original_std")
    delta = _f8_vector(delta_std, "delta_std")
    target = np.asarray(original + delta, dtype="<f8")
    request = SolveRequest(
        image_path=str(image_path),
        input_image_sha256=str(input_image_sha256),
        original_std=tuple(float(value) for value in original),
        delta_std=tuple(float(value) for value in delta),
        delta_hash_material_std=tuple(float(value) for value in delta),
        target_std=tuple(float(value) for value in target),
        delta_sha256=str(delta_sha256),
        delta_sha_is_frozen_declared_lineage=False,
        original_descriptor_sha256=vector_sha256(
            original, ORIGINAL_DESCRIPTOR_HASH_PREFIX, "original_std"
        ),
        target_descriptor_sha256=vector_sha256(
            target, TARGET_DESCRIPTOR_HASH_PREFIX, "target_std"
        ),
        intended_dose=float(np.sum(np.abs(delta) * WEIGHTS, dtype=np.float64)),
        intended_dose_is_explicit_frozen_input=False,
        config=config,
    )
    request.validate()
    return request


def make_explicit_target_solve_request(
    *,
    image_path: str | Path,
    input_image_sha256: str,
    original_std: Sequence[float],
    target_std: Sequence[float],
    delta_hash_material_std: Sequence[float],
    delta_sha256: str,
    intended_dose: float,
    config: ScientificConfigIdentity,
) -> SolveRequest:
    """Build a request when a frozen target was materialized independently.

    The formal Native realization table predates this engine and freezes its
    target descriptor independently from the CSV-serialized policy delta.
    Their values differ by a few float64 ULPs.  ``target_std - original_std``
    remains the exact objective delta.  ``delta_hash_material_std`` records
    the later CSV representation, while ``delta_sha256`` retains the earlier
    declared lineage cross-locked by the caller to the SHA-locked formal
    tables.  No tolerance or numerical substitution is used.
    """

    original = _f8_vector(original_std, "original_std")
    target = _f8_vector(target_std, "target_std")
    target_delta = np.asarray(target - original, dtype="<f8")
    hash_material = _f8_vector(
        delta_hash_material_std, "delta_hash_material_std"
    )
    _require_sha256(str(delta_sha256), "explicit_target_delta_lineage")
    if not math.isfinite(float(intended_dose)):
        raise ExactCacheStop("STOP_EXPLICIT_TARGET_INTENDED_DOSE_NONFINITE")
    request = SolveRequest(
        image_path=str(image_path),
        input_image_sha256=str(input_image_sha256),
        original_std=tuple(float(value) for value in original),
        delta_std=tuple(float(value) for value in target_delta),
        delta_hash_material_std=tuple(float(value) for value in hash_material),
        target_std=tuple(float(value) for value in target),
        delta_sha256=str(delta_sha256),
        # The old formal realization froze this SHA before a CSV serialization
        # boundary.  Its exact target descriptor remains in Parquet, but the
        # pre-CSV delta bytes do not.  This flag is collision-checked as part
        # of the scientific payload and is used only by the Golden audit path.
        delta_sha_is_frozen_declared_lineage=True,
        original_descriptor_sha256=vector_sha256(
            original, ORIGINAL_DESCRIPTOR_HASH_PREFIX, "original_std"
        ),
        target_descriptor_sha256=vector_sha256(
            target, TARGET_DESCRIPTOR_HASH_PREFIX, "target_std"
        ),
        intended_dose=float(intended_dose),
        intended_dose_is_explicit_frozen_input=True,
        config=config,
    )
    request.validate()
    return request


@dataclass(frozen=True)
class VisitAlias:
    """Frozen visit/schedule identity, kept outside the scientific cache."""

    adaptation_seed: int
    epoch: int
    target_visit_ordinal: int
    schedule_uid: str
    q2_uid: str
    capture_event_id: str
    branch: str
    augment_on: bool = True
    valid: bool = True
    policy_flag: str = "APPLY_OFFLINE_THETA"
    failure_reason: str = ""
    donor_schedule_uid: str = ""
    donor_q2_uid: str = ""
    donor_event_id: str = ""

    def validate(self) -> None:
        if int(self.adaptation_seed) != FROZEN_ADAPTATION_SEED:
            raise ExactCacheStop("STOP_ALIAS_ADAPTATION_SEED_DRIFT")
        if not (FROZEN_FIRST_EPOCH <= int(self.epoch) <= FROZEN_LAST_EPOCH):
            raise ExactCacheStop("STOP_ALIAS_EPOCH_OUTSIDE_FROZEN_HORIZON")
        if not (0 <= int(self.target_visit_ordinal) < FROZEN_TARGET_VISITS_PER_EPOCH):
            raise ExactCacheStop("STOP_ALIAS_VISIT_ORDINAL_OUT_OF_RANGE")
        if self.branch not in ALLOWED_BRANCHES:
            raise ExactCacheStop(f"STOP_ALIAS_BRANCH:{self.branch}")
        if not _strict_bool(self.augment_on, "alias.augment_on") or not _strict_bool(
            self.valid, "alias.valid"
        ):
            raise ExactCacheStop("STOP_ALIAS_IS_NOT_ACTIVE_VALID_SOLVE")
        if self.policy_flag != "APPLY_OFFLINE_THETA":
            raise ExactCacheStop(f"STOP_ALIAS_POLICY_FLAG:{self.policy_flag}")
        if not self.schedule_uid or not self.q2_uid or not self.capture_event_id:
            raise ExactCacheStop("STOP_ALIAS_REQUIRED_ID_EMPTY")

    @property
    def identity(self) -> tuple[Any, ...]:
        self.validate()
        return (
            int(self.adaptation_seed),
            int(self.epoch),
            int(self.target_visit_ordinal),
            str(self.schedule_uid),
            str(self.q2_uid),
            str(self.branch),
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        rank = {BRANCH_NATIVE: 0, BRANCH_MATCHED: 1, BRANCH_SMG: 2}[self.branch]
        return (
            int(self.epoch),
            int(self.target_visit_ordinal),
            rank,
            str(self.schedule_uid),
            str(self.q2_uid),
            str(self.donor_schedule_uid),
            str(self.donor_q2_uid),
        )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": VISIT_ALIAS_SCHEMA,
            "adaptation_seed": int(self.adaptation_seed),
            "epoch": int(self.epoch),
            "target_visit_ordinal": int(self.target_visit_ordinal),
            "schedule_uid": str(self.schedule_uid),
            "q2_uid": str(self.q2_uid),
            "capture_event_id": str(self.capture_event_id),
            "branch": str(self.branch),
            "augment_on": True,
            "valid": True,
            "policy_flag": str(self.policy_flag),
            "failure_reason": str(self.failure_reason),
            "donor_schedule_uid": str(self.donor_schedule_uid),
            "donor_q2_uid": str(self.donor_q2_uid),
            "donor_event_id": str(self.donor_event_id),
        }


def _cache_namespace(branches: set[str], subset_reuse_authorized: bool) -> str:
    if BRANCH_SMG in branches:
        if branches != {BRANCH_SMG}:
            raise ExactCacheStop("STOP_SMG_NAMESPACE_CROSS_BRANCH_COLLISION")
        return SMG_CACHE_NAMESPACE
    if BRANCH_NATIVE in branches:
        if BRANCH_MATCHED in branches and not subset_reuse_authorized:
            raise ExactCacheStop("STOP_NATIVE_MATCHED_REUSE_WITHOUT_VALIDATED_EVIDENCE")
        return NATIVE_CACHE_NAMESPACE
    if branches == {BRANCH_MATCHED}:
        # A Matched-only plan cannot silently claim Native cache provenance.
        return MATCHED_ISOLATED_NAMESPACE
    raise ExactCacheStop(f"STOP_UNSUPPORTED_BRANCH_SET:{sorted(branches)}")


def namespaced_cache_key_sha256(namespace: str, exact_solve_key_sha256: str) -> str:
    _require_sha256(exact_solve_key_sha256, "exact_solve_key")
    if namespace not in {
        NATIVE_CACHE_NAMESPACE,
        MATCHED_ISOLATED_NAMESPACE,
        SMG_CACHE_NAMESPACE,
    }:
        raise ExactCacheStop(f"STOP_CACHE_NAMESPACE:{namespace}")
    payload = {
        "schema": CACHE_KEY_SCHEMA,
        "cache_namespace": namespace,
        "exact_solve_key_sha256": exact_solve_key_sha256,
    }
    return hashlib.sha256(
        CACHE_KEY_HASH_PREFIX + canonical_json_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True)
class UniqueSolve:
    request: SolveRequest
    aliases: tuple[VisitAlias, ...]
    cache_namespace: str

    @property
    def exact_solve_key(self) -> ExactSolveKey:
        return self.request.exact_key

    @property
    def exact_solve_key_sha256(self) -> str:
        return self.exact_solve_key.sha256

    @property
    def cache_key_sha256(self) -> str:
        return namespaced_cache_key_sha256(
            self.cache_namespace, self.exact_solve_key_sha256
        )

    def validate(self) -> None:
        self.request.validate()
        if not self.aliases:
            raise ExactCacheStop("STOP_UNIQUE_SOLVE_WITHOUT_ALIAS")
        aliases = tuple(sorted(self.aliases, key=lambda item: item.sort_key))
        if aliases != self.aliases:
            raise ExactCacheStop("STOP_UNSORTED_UNIQUE_SOLVE_ALIASES")
        branches = {alias.branch for alias in aliases}
        for alias in aliases:
            alias.validate()
        if BRANCH_SMG in branches and branches != {BRANCH_SMG}:
            raise ExactCacheStop("STOP_SMG_ALIAS_NAMESPACE_LEAK")
        expected_namespace = _cache_namespace(
            branches,
            subset_reuse_authorized=(
                BRANCH_NATIVE in branches and BRANCH_MATCHED in branches
            ),
        )
        if expected_namespace != self.cache_namespace:
            raise ExactCacheStop("STOP_UNIQUE_SOLVE_NAMESPACE_MISMATCH")


@dataclass(frozen=True)
class ImageSolveGroup:
    input_image_sha256: str
    image_path: str
    unique_solves: tuple[UniqueSolve, ...]

    def validate(self) -> None:
        _require_sha256(self.input_image_sha256, "image_group")
        if not self.image_path or not self.unique_solves:
            raise ExactCacheStop("STOP_EMPTY_IMAGE_SOLVE_GROUP")
        keys = []
        for solve in self.unique_solves:
            solve.validate()
            if solve.request.input_image_sha256 != self.input_image_sha256:
                raise ExactCacheStop("STOP_IMAGE_GROUP_SHA_MISMATCH")
            keys.append(solve.cache_key_sha256)
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ExactCacheStop("STOP_IMAGE_GROUP_KEY_ORDER_OR_DUPLICATE")


@dataclass(frozen=True)
class ExecutionPlan:
    groups: tuple[ImageSolveGroup, ...]
    total_solve_requests: int
    unique_exact_solve_count: int
    unique_namespaced_cache_count: int
    subset_reuse_authorized: bool
    subset_evidence: Mapping[str, Any] | None

    def validate(self) -> None:
        image_shas = []
        exact_keys: set[str] = set()
        cache_keys: set[str] = set()
        alias_count = 0
        reconstructed_pairs: list[tuple[SolveRequest, VisitAlias]] = []
        branches_present: set[str] = set()
        for group in self.groups:
            group.validate()
            image_shas.append(group.input_image_sha256)
            for solve in group.unique_solves:
                exact_keys.add(solve.exact_solve_key_sha256)
                cache_keys.add(solve.cache_key_sha256)
                alias_count += len(solve.aliases)
                for alias in solve.aliases:
                    reconstructed_pairs.append((solve.request, alias))
                    branches_present.add(alias.branch)
        if image_shas != sorted(image_shas) or len(image_shas) != len(set(image_shas)):
            raise ExactCacheStop("STOP_PLAN_IMAGE_ORDER_OR_DUPLICATE")
        if alias_count != int(self.total_solve_requests):
            raise ExactCacheStop("STOP_PLAN_REQUEST_COUNT_MISMATCH")
        if len(exact_keys) != int(self.unique_exact_solve_count):
            raise ExactCacheStop("STOP_PLAN_EXACT_KEY_COUNT_MISMATCH")
        if len(cache_keys) != int(self.unique_namespaced_cache_count):
            raise ExactCacheStop("STOP_PLAN_CACHE_KEY_COUNT_MISMATCH")
        has_native_matched = {
            BRANCH_NATIVE,
            BRANCH_MATCHED,
        }.issubset(branches_present)
        if has_native_matched:
            if not self.subset_reuse_authorized or self.subset_evidence is None:
                raise ExactCacheStop("STOP_PLAN_NATIVE_MATCHED_EVIDENCE_NOT_AUTHORIZED")
            _validate_subset_evidence(self.subset_evidence, reconstructed_pairs)
        elif self.subset_reuse_authorized or self.subset_evidence is not None:
            raise ExactCacheStop("STOP_PLAN_SPURIOUS_NATIVE_MATCHED_AUTHORIZATION")

    def solve_key_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for group in self.groups:
            for solve in group.unique_solves:
                branches = sorted({alias.branch for alias in solve.aliases})
                rows.append(
                    {
                        **solve.exact_solve_key.as_dict(),
                        "solve_key": solve.exact_solve_key_sha256,
                        "cache_namespace": solve.cache_namespace,
                        "cache_key_sha256": solve.cache_key_sha256,
                        "request_count": len(solve.aliases),
                        "branches": "|".join(branches),
                    }
                )
        return tuple(sorted(rows, key=lambda row: row["cache_key_sha256"]))

    def visit_alias_rows(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        for group in self.groups:
            for solve in group.unique_solves:
                for alias in solve.aliases:
                    rows.append(
                        {
                            **alias.as_dict(),
                            "solve_key": solve.exact_solve_key_sha256,
                            "cache_namespace": solve.cache_namespace,
                            "cache_key_sha256": solve.cache_key_sha256,
                        }
                    )
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    int(row["epoch"]),
                    int(row["target_visit_ordinal"]),
                    {BRANCH_NATIVE: 0, BRANCH_MATCHED: 1, BRANCH_SMG: 2}[
                        str(row["branch"])
                    ],
                    str(row["schedule_uid"]),
                    str(row["q2_uid"]),
                ),
            )
        )


def _branch_exact_key_sets(
    pairs: Sequence[tuple[SolveRequest, VisitAlias]],
) -> dict[str, set[str]]:
    result = {branch: set() for branch in ALLOWED_BRANCHES}
    for request, alias in pairs:
        result[alias.branch].add(request.exact_key.sha256)
    return result


def build_native_matched_subset_evidence(
    pairs: Iterable[tuple[SolveRequest, VisitAlias]],
) -> dict[str, Any]:
    """Build machine evidence; execution still requires it to be passed back."""

    materialized = tuple(pairs)
    for request, alias in materialized:
        request.validate()
        alias.validate()
    key_sets = _branch_exact_key_sets(materialized)
    native = key_sets[BRANCH_NATIVE]
    matched = key_sets[BRANCH_MATCHED]
    smg = key_sets[BRANCH_SMG]
    missing = sorted(matched - native)
    config_identities = {request.config for request, _ in materialized}
    if len(config_identities) != 1:
        raise ExactCacheStop("STOP_MULTIPLE_SCIENTIFIC_CONFIG_IDENTITIES")
    config = next(iter(config_identities))
    return {
        "schema": SUBSET_AUDIT_SCHEMA,
        "MATCHED_A_SOLVE_KEY_SUBSET_OF_NATIVE_A": len(missing) == 0,
        "subset_boolean": len(missing) == 0,
        "missing_key_count": len(missing),
        "missing_keys": missing,
        "Native_unique_keys": len(native),
        "Matched_A_unique_keys": len(matched),
        "SMG_unique_keys": len(smg),
        "reused_key_count": len(native & matched),
        "native_only_key_count": len(native - matched),
        "SMG_intersection_with_Native_count": len(smg & native),
        "SMG_reuse_attempted": False,
        "solve_key_schema": EXACT_SOLVE_KEY_SCHEMA,
        "solve_key_components": list(EXACT_SOLVE_KEY_COMPONENTS),
        "config_identity": config.as_dict(),
    }


_VALIDATED_RAW_SUBSET_EVIDENCE: dict[str, dict[str, set[str]]] = {}


def _validate_raw_subset_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, set[str]] | None:
    if "evidence_path" not in evidence and "evidence_sha256" not in evidence:
        return None
    path = Path(str(evidence.get("evidence_path", "")))
    expected_sha = _require_sha256(
        str(evidence.get("evidence_sha256", "")), "subset_evidence_file"
    )
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ExactCacheStop("STOP_SUBSET_RAW_EVIDENCE_FILE_SHA_MISMATCH")
    cached = _VALIDATED_RAW_SUBSET_EVIDENCE.get(expected_sha)
    if cached is not None:
        return cached
    try:
        import pandas as pd
    except Exception as error:
        raise ExactCacheStop("STOP_SUBSET_RAW_EVIDENCE_PANDAS_IMPORT") from error
    required = {"branch", "solve_key", *EXACT_SOLVE_KEY_COMPONENTS}
    try:
        frame = pd.read_parquet(path, columns=sorted(required))
    except Exception as error:
        raise ExactCacheStop("STOP_SUBSET_RAW_EVIDENCE_READ") from error
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ExactCacheStop(
            "STOP_SUBSET_RAW_EVIDENCE_SCHEMA:" + ",".join(missing_columns)
        )
    key_sets = {branch: set() for branch in ALLOWED_BRANCHES}
    for row in frame.to_dict("records"):
        branch = str(row["branch"])
        if branch not in key_sets:
            raise ExactCacheStop(f"STOP_SUBSET_RAW_EVIDENCE_BRANCH:{branch}")
        payload = {"schema": EXACT_SOLVE_KEY_SCHEMA}
        for field in EXACT_SOLVE_KEY_COMPONENTS:
            payload[field] = _require_sha256(
                str(row[field]), f"raw_evidence.{field}"
            )
        observed_key = _require_sha256(str(row["solve_key"]), "raw_evidence.solve_key")
        if canonical_json_sha256(payload) != observed_key:
            raise ExactCacheStop("STOP_SUBSET_RAW_EVIDENCE_KEY_RECOMPUTE_MISMATCH")
        key_sets[branch].add(observed_key)
    native = key_sets[BRANCH_NATIVE]
    matched = key_sets[BRANCH_MATCHED]
    smg = key_sets[BRANCH_SMG]
    missing = sorted(matched - native)
    expected_values = {
        "Native_unique_keys": len(native),
        "Matched_A_unique_keys": len(matched),
        "SMG_unique_keys": len(smg),
        "reused_key_count": len(native & matched),
        "native_only_key_count": len(native - matched),
        "SMG_intersection_with_Native_count": len(smg & native),
    }
    for field, expected in expected_values.items():
        if int(evidence.get(field, -1)) != expected:
            raise ExactCacheStop(f"STOP_SUBSET_RAW_EVIDENCE_COUNT:{field}")
    if missing or list(evidence.get("missing_keys", [])) != missing:
        raise ExactCacheStop("STOP_SUBSET_RAW_EVIDENCE_MISSING_KEYS")
    immutable = {branch: set(values) for branch, values in key_sets.items()}
    _VALIDATED_RAW_SUBSET_EVIDENCE[expected_sha] = immutable
    return immutable


def _validate_subset_evidence(
    evidence: Mapping[str, Any],
    pairs: Sequence[tuple[SolveRequest, VisitAlias]],
) -> None:
    if str(evidence.get("schema", "")) != SUBSET_AUDIT_SCHEMA:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_SCHEMA")
    if str(evidence.get("solve_key_schema", "")) != EXACT_SOLVE_KEY_SCHEMA:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_KEY_SCHEMA")
    if tuple(evidence.get("solve_key_components", ())) != EXACT_SOLVE_KEY_COMPONENTS:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_KEY_COMPONENTS")
    if evidence.get("MATCHED_A_SOLVE_KEY_SUBSET_OF_NATIVE_A") is not True:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_NOT_PASS")
    if evidence.get("subset_boolean") is not True:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_BOOLEAN_NOT_PASS")
    if int(evidence.get("missing_key_count", -1)) != 0:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_MISSING_NONZERO")
    if list(evidence.get("missing_keys", [])) != []:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_MISSING_LIST_NONEMPTY")
    if evidence.get("SMG_reuse_attempted") is not False:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_SMG_REUSE_ATTEMPTED")

    derived = build_native_matched_subset_evidence(pairs)
    raw_key_sets = _validate_raw_subset_evidence(evidence)
    if raw_key_sets is None:
        for field in (
            "Native_unique_keys",
            "Matched_A_unique_keys",
            "SMG_unique_keys",
            "reused_key_count",
            "native_only_key_count",
            "SMG_intersection_with_Native_count",
        ):
            if int(evidence.get(field, -1)) != int(derived[field]):
                raise ExactCacheStop(f"STOP_SUBSET_EVIDENCE_DERIVED_COUNT:{field}")
        if derived["subset_boolean"] is not True or derived["missing_key_count"] != 0:
            raise ExactCacheStop("STOP_SUBSET_RECOMPUTATION_FAILED")
    else:
        current_sets = _branch_exact_key_sets(pairs)
        for branch in ALLOWED_BRANCHES:
            unexpected = current_sets[branch] - raw_key_sets[branch]
            if unexpected:
                raise ExactCacheStop(
                    f"STOP_PLAN_KEY_ABSENT_FROM_RAW_SUBSET_EVIDENCE:{branch}"
                )
        if not current_sets[BRANCH_MATCHED].issubset(raw_key_sets[BRANCH_NATIVE]):
            raise ExactCacheStop("STOP_PLAN_MATCHED_KEY_ABSENT_FROM_EVIDENCED_NATIVE")

    config = next(iter({request.config for request, _ in pairs}))
    if "configs" in evidence:
        observed = ScientificConfigIdentity.from_config_mapping(
            evidence["configs"]
        )
        if observed != config:
            raise ExactCacheStop("STOP_SUBSET_EVIDENCE_CONFIG_MISMATCH")
    elif "config_identity" in evidence:
        if dict(evidence["config_identity"]) != config.as_dict():
            raise ExactCacheStop("STOP_SUBSET_EVIDENCE_CONFIG_IDENTITY_MISMATCH")
    else:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_CONFIG_MISSING")

def build_execution_plan(
    pairs: Iterable[tuple[SolveRequest, VisitAlias]],
    *,
    subset_evidence: Mapping[str, Any] | None = None,
) -> ExecutionPlan:
    """Validate, globally deduplicate, group by image SHA, and sort exactly."""

    materialized = tuple(pairs)
    if not materialized:
        raise ExactCacheStop("STOP_EMPTY_SOLVE_REQUEST_SET")
    alias_identities: set[tuple[Any, ...]] = set()
    configs: set[ScientificConfigIdentity] = set()
    for request, alias in materialized:
        request.validate()
        alias.validate()
        configs.add(request.config)
        if alias.identity in alias_identities:
            raise ExactCacheStop(f"STOP_DUPLICATE_VISIT_ALIAS:{alias.identity}")
        alias_identities.add(alias.identity)
    if len(configs) != 1:
        raise ExactCacheStop("STOP_MULTIPLE_SCIENTIFIC_CONFIG_IDENTITIES")

    branches_present = {alias.branch for _, alias in materialized}
    has_native_matched = {
        BRANCH_NATIVE,
        BRANCH_MATCHED,
    }.issubset(branches_present)
    if has_native_matched:
        if subset_evidence is None:
            raise ExactCacheStop("STOP_NATIVE_MATCHED_SUBSET_EVIDENCE_REQUIRED")
        _validate_subset_evidence(subset_evidence, materialized)
    elif subset_evidence is not None:
        raise ExactCacheStop("STOP_SUBSET_EVIDENCE_WITHOUT_COMPLETE_NATIVE_MATCHED_SET")

    by_scientific_key: dict[
        str, list[tuple[SolveRequest, VisitAlias]]
    ] = {}
    for request, alias in materialized:
        by_scientific_key.setdefault(request.exact_key.sha256, []).append(
            (request, alias)
        )

    unique_solves: list[UniqueSolve] = []
    for solve_key_sha in sorted(by_scientific_key):
        values = by_scientific_key[solve_key_sha]
        baseline_payload = values[0][0].exact_science_payload()
        for request, _ in values[1:]:
            if request.exact_science_payload() != baseline_payload:
                raise ExactCacheStop("STOP_EXACT_KEY_SCIENCE_PAYLOAD_COLLISION")
        branches = {alias.branch for _, alias in values}
        if BRANCH_SMG in branches and len(branches) != 1:
            # Never combine SMG even in the cryptographically unlikely event
            # that every non-namespace solve component is equal.
            non_smg = [value for value in values if value[1].branch != BRANCH_SMG]
            smg = [value for value in values if value[1].branch == BRANCH_SMG]
            partitions = [partition for partition in (non_smg, smg) if partition]
        else:
            partitions = [values]
        for partition in partitions:
            partition_branches = {alias.branch for _, alias in partition}
            namespace = _cache_namespace(
                partition_branches,
                subset_reuse_authorized=has_native_matched,
            )
            representative = min(
                (request for request, _ in partition),
                key=lambda request: str(Path(request.image_path)),
            )
            aliases = tuple(
                sorted((alias for _, alias in partition), key=lambda item: item.sort_key)
            )
            unique = UniqueSolve(
                request=representative,
                aliases=aliases,
                cache_namespace=namespace,
            )
            unique.validate()
            unique_solves.append(unique)

    # Namespaced cache keys, not unqualified scientific keys, are the global
    # execution-dedup unit.  This is the hard SMG isolation boundary.
    cache_keys = [solve.cache_key_sha256 for solve in unique_solves]
    if len(cache_keys) != len(set(cache_keys)):
        raise ExactCacheStop("STOP_NAMESPACED_CACHE_KEY_DUPLICATE_AFTER_DEDUP")

    by_image: dict[str, list[UniqueSolve]] = {}
    for solve in unique_solves:
        by_image.setdefault(solve.request.input_image_sha256, []).append(solve)
    groups: list[ImageSolveGroup] = []
    for image_sha in sorted(by_image):
        solves = tuple(sorted(by_image[image_sha], key=lambda item: item.cache_key_sha256))
        paths = sorted({str(solve.request.image_path) for solve in solves})
        group = ImageSolveGroup(
            input_image_sha256=image_sha,
            image_path=paths[0],
            unique_solves=solves,
        )
        group.validate()
        groups.append(group)

    plan = ExecutionPlan(
        groups=tuple(groups),
        total_solve_requests=len(materialized),
        unique_exact_solve_count=len(by_scientific_key),
        unique_namespaced_cache_count=len(unique_solves),
        subset_reuse_authorized=has_native_matched,
        subset_evidence=dict(subset_evidence) if subset_evidence is not None else None,
    )
    plan.validate()
    return plan


@dataclass(frozen=True)
class FrozenRuntimeSpec:
    realization_core_path: str
    q2_snapshot_dir: str
    q2_file_sha256: tuple[tuple[str, str], ...]
    allowed_image_roots: tuple[str, ...]
    median: tuple[float, ...]
    scale: tuple[float, ...]
    event_std: tuple[tuple[float, ...], ...]
    support_reference_sha256: str
    config: ScientificConfigIdentity
    solver_config_json: str
    operator_config_json: str
    descriptor_config_json: str
    descriptor_schema_path: str
    image_descriptors_path: str
    event_prototypes_path: str

    def validate_static(self) -> None:
        self.config.validate()
        if not Path(self.realization_core_path).is_absolute():
            raise ExactCacheStop("STOP_RUNTIME_CORE_PATH_NOT_ABSOLUTE")
        if not Path(self.q2_snapshot_dir).is_absolute():
            raise ExactCacheStop("STOP_RUNTIME_Q2_PATH_NOT_ABSOLUTE")
        if not self.allowed_image_roots:
            raise ExactCacheStop("STOP_RUNTIME_ALLOWED_IMAGE_ROOTS_EMPTY")
        for root in self.allowed_image_roots:
            if not Path(root).is_absolute():
                raise ExactCacheStop("STOP_RUNTIME_ALLOWED_ROOT_NOT_ABSOLUTE")
        _f8_vector(self.median, "median")
        scale = _f8_vector(self.scale, "scale")
        if np.any(scale <= 0.0):
            raise ExactCacheStop("STOP_RUNTIME_SCALE_NONPOSITIVE")
        observed_support = support_matrix_sha256(self.event_std)
        if observed_support != self.support_reference_sha256:
            raise ExactCacheStop("STOP_RUNTIME_SUPPORT_REFERENCE_SHA_MISMATCH")
        for name, expected in self.q2_file_sha256:
            if Path(name).name != name or name in {"", ".", ".."}:
                raise ExactCacheStop("STOP_RUNTIME_Q2_MANIFEST_NAME")
            _require_sha256(expected, f"q2.{name}")
        if tuple(sorted(self.q2_file_sha256)) != self.q2_file_sha256:
            raise ExactCacheStop("STOP_RUNTIME_Q2_MANIFEST_NOT_SORTED")
        for role, payload_json, expected in (
            ("solver", self.solver_config_json, self.config.solver_config_sha256),
            ("operator", self.operator_config_json, self.config.operator_config_sha256),
            (
                "descriptor",
                self.descriptor_config_json,
                self.config.descriptor_config_sha256,
            ),
        ):
            try:
                payload = json.loads(payload_json)
            except Exception as error:
                raise ExactCacheStop(f"STOP_RUNTIME_CONFIG_JSON:{role}") from error
            if canonical_json_sha256(payload) != expected:
                raise ExactCacheStop(f"STOP_RUNTIME_CONFIG_JSON_SHA:{role}")


def build_frozen_runtime_spec(
    *,
    realization_core_path: str | Path,
    q2_snapshot_dir: str | Path,
    allowed_image_roots: Sequence[str | Path],
    median: Sequence[float],
    scale: Sequence[float],
    event_std: Sequence[Sequence[float]],
    subset_evidence: Mapping[str, Any],
    descriptor_schema_path: str | Path,
    image_descriptors_path: str | Path,
    event_prototypes_path: str | Path,
) -> FrozenRuntimeSpec:
    """Create and verify a picklable runtime from the frozen audit evidence."""

    configs = subset_evidence.get("configs")
    if not isinstance(configs, Mapping):
        raise ExactCacheStop("STOP_RUNTIME_SUBSET_CONFIGS_MISSING")
    config = ScientificConfigIdentity.from_config_mapping(configs)
    core_path = Path(realization_core_path).resolve()
    snapshot = Path(q2_snapshot_dir).resolve()
    if not core_path.is_file() or sha256_file(core_path) != config.realization_core_sha256:
        raise ExactCacheStop("STOP_RUNTIME_FROZEN_CORE_SHA_MISMATCH")
    expected_q2 = configs["descriptor_config"].get("q2_code_sha256", {})
    required_q2 = {"q2_descriptors.py", "q2_frozen.py", "q2_hpl.py"}
    if set(expected_q2) != required_q2:
        raise ExactCacheStop("STOP_RUNTIME_Q2_EVIDENCE_FILE_SET")
    q2_manifest: list[tuple[str, str]] = []
    for name in sorted(required_q2 | {"q2_rng.py"}):
        path = snapshot / name
        if not path.is_file():
            raise ExactCacheStop(f"STOP_RUNTIME_Q2_FILE_MISSING:{name}")
        observed = sha256_file(path)
        if name in expected_q2 and observed != str(expected_q2[name]):
            raise ExactCacheStop(f"STOP_RUNTIME_Q2_FILE_SHA:{name}")
        q2_manifest.append((name, observed))

    for role, path_value, config_field in (
        ("descriptor_schema", descriptor_schema_path, "descriptor_schema_sha256"),
        ("image_descriptors", image_descriptors_path, "image_descriptors_sha256"),
        ("event_prototypes", event_prototypes_path, "event_prototypes_sha256"),
    ):
        path = Path(path_value).resolve()
        expected = str(configs["descriptor_config"].get(config_field, ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise ExactCacheStop(f"STOP_RUNTIME_DESCRIPTOR_INPUT_SHA:{role}")

    runtime = FrozenRuntimeSpec(
        realization_core_path=str(core_path),
        q2_snapshot_dir=str(snapshot),
        q2_file_sha256=tuple(q2_manifest),
        allowed_image_roots=tuple(
            sorted(str(Path(root).resolve()) for root in allowed_image_roots)
        ),
        median=tuple(float(value) for value in _f8_vector(median, "median")),
        scale=tuple(float(value) for value in _f8_vector(scale, "scale")),
        event_std=tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(event_std, dtype="<f8")
        ),
        support_reference_sha256=support_matrix_sha256(event_std),
        config=config,
        solver_config_json=canonical_json_bytes(configs["solver_config"]).decode("utf-8"),
        operator_config_json=canonical_json_bytes(configs["operator_config"]).decode("utf-8"),
        descriptor_config_json=canonical_json_bytes(configs["descriptor_config"]).decode("utf-8"),
        descriptor_schema_path=str(Path(descriptor_schema_path).resolve()),
        image_descriptors_path=str(Path(image_descriptors_path).resolve()),
        event_prototypes_path=str(Path(event_prototypes_path).resolve()),
    )
    runtime.validate_static()
    return runtime


def _import_frozen_core(runtime: FrozenRuntimeSpec) -> Any:
    path = Path(runtime.realization_core_path)
    if not path.is_file() or sha256_file(path) != FROZEN_REALIZATION_CORE_SHA256:
        raise ExactCacheStop("STOP_WORKER_FROZEN_CORE_SHA")
    name = f"m1_routea_offline_cache_frozen_core_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExactCacheStop("STOP_WORKER_FROZEN_CORE_IMPORT_SPEC")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_core_against_configs(core: Any, runtime: FrozenRuntimeSpec) -> None:
    solver = json.loads(runtime.solver_config_json)
    operator = json.loads(runtime.operator_config_json)
    descriptor = json.loads(runtime.descriptor_config_json)
    solver_body = solver.get("solver", {})
    if solver_body.get("method") != "Powell":
        raise ExactCacheStop("STOP_WORKER_SOLVER_METHOD_DRIFT")
    if (
        solver_body.get("single_start") is not True
        or solver_body.get("multi_start") is not False
        or solver_body.get("random_restart") is not False
        or solver_body.get("fallback_solver") is not False
    ):
        raise ExactCacheStop("STOP_WORKER_SOLVER_START_OR_FALLBACK_DRIFT")
    expected_options = {
        "maxiter": int(solver_body.get("maxiter", -1)),
        "maxfev": int(solver_body.get("maxfev", -1)),
        "xtol": float(solver_body.get("xtol", math.nan)),
        "ftol": float(solver_body.get("ftol", math.nan)),
    }
    if dict(core.SOLVER_OPTIONS) != expected_options:
        raise ExactCacheStop("STOP_WORKER_SOLVER_OPTIONS_DRIFT")
    bounds = solver.get("bounds", {})
    if _f8_vector(bounds.get("lower", ()), "solver.lower", 3).tobytes() != np.asarray(
        core.PARAMETER_LOWER, dtype="<f8"
    ).tobytes():
        raise ExactCacheStop("STOP_WORKER_SOLVER_LOWER_BOUND_DRIFT")
    if _f8_vector(bounds.get("upper", ()), "solver.upper", 3).tobytes() != np.asarray(
        core.PARAMETER_UPPER, dtype="<f8"
    ).tobytes():
        raise ExactCacheStop("STOP_WORKER_SOLVER_UPPER_BOUND_DRIFT")
    if str(solver.get("numpy_version")) != np.__version__:
        raise ExactCacheStop("STOP_WORKER_NUMPY_VERSION_DRIFT")
    try:
        import scipy
    except Exception as error:
        raise ExactCacheStop("STOP_WORKER_SCIPY_IMPORT") from error
    if str(solver.get("scipy_version")) != scipy.__version__:
        raise ExactCacheStop("STOP_WORKER_SCIPY_VERSION_DRIFT")
    operator_body = operator.get("operator", {})
    if int(operator_body.get("degrees_of_freedom", -1)) != 3:
        raise ExactCacheStop("STOP_WORKER_OPERATOR_DOF_DRIFT")
    for field, observed in (
        ("delta_t_bounds", core.PARAMETER_LOWER[[0]].tolist() + core.PARAMETER_UPPER[[0]].tolist()),
        ("c_t_bounds", core.PARAMETER_LOWER[[1]].tolist() + core.PARAMETER_UPPER[[1]].tolist()),
        ("s_c_bounds", core.PARAMETER_LOWER[[2]].tolist() + core.PARAMETER_UPPER[[2]].tolist()),
    ):
        expected = _f8_vector(operator_body.get(field, ()), f"operator.{field}", 2)
        if expected.tobytes() != np.asarray(observed, dtype="<f8").tobytes():
            raise ExactCacheStop(f"STOP_WORKER_OPERATOR_BOUND_DRIFT:{field}")
    descriptor_space = descriptor.get("descriptor_space", {})
    if tuple(descriptor_space.get("base_descriptors", ())) != BASE:
        raise ExactCacheStop("STOP_WORKER_DESCRIPTOR_BASE_DRIFT")
    weights = _f8_vector(
        descriptor_space.get("d_style_weights", ()), "descriptor.weights"
    )
    if weights.tobytes() != np.asarray(core.WEIGHTS, dtype="<f8").tobytes():
        raise ExactCacheStop("STOP_WORKER_DESCRIPTOR_WEIGHT_DRIFT")


_WORKER_STATE: tuple[FrozenRuntimeSpec, Any, np.ndarray, np.ndarray, np.ndarray] | None = None


def _worker_initializer(runtime: FrozenRuntimeSpec) -> None:
    global _WORKER_STATE
    runtime.validate_static()
    snapshot = Path(runtime.q2_snapshot_dir)
    for name, expected in runtime.q2_file_sha256:
        path = snapshot / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ExactCacheStop(f"STOP_WORKER_Q2_SHA:{name}")
    descriptor_config = json.loads(runtime.descriptor_config_json)
    for role, path_value, config_field in (
        (
            "descriptor_schema",
            runtime.descriptor_schema_path,
            "descriptor_schema_sha256",
        ),
        (
            "image_descriptors",
            runtime.image_descriptors_path,
            "image_descriptors_sha256",
        ),
        (
            "event_prototypes",
            runtime.event_prototypes_path,
            "event_prototypes_sha256",
        ),
    ):
        path = Path(path_value)
        expected = str(descriptor_config.get(config_field, ""))
        if not path.is_file() or sha256_file(path) != expected:
            raise ExactCacheStop(f"STOP_WORKER_DESCRIPTOR_INPUT_SHA:{role}")
    for module_name in ("q2_descriptors", "q2_frozen", "q2_hpl", "q2_rng"):
        existing = sys.modules.get(module_name)
        if existing is not None:
            existing_path = Path(str(getattr(existing, "__file__", ""))).resolve()
            if existing_path.parent != snapshot.resolve():
                raise ExactCacheStop(f"STOP_WORKER_PRELOADED_Q2_MODULE:{module_name}")
    core = _import_frozen_core(runtime)
    _validate_core_against_configs(core, runtime)
    core.configure_q2_snapshot(snapshot)
    median = _f8_vector(runtime.median, "median")
    scale = _f8_vector(runtime.scale, "scale")
    event_std = np.asarray(runtime.event_std, dtype=np.float64)
    _WORKER_STATE = (runtime, core, median, scale, event_std)


def _is_allowed_image(path: Path, allowed_roots: Sequence[str]) -> bool:
    resolved = path.resolve()
    normalized = str(resolved).replace("\\", "/").casefold()
    if "/target/dev/" in normalized or "/target/test/" in normalized:
        return False
    return any(
        resolved == Path(root).resolve() or Path(root).resolve() in resolved.parents
        for root in allowed_roots
    )


def _process_memory_snapshot() -> tuple[int, int]:
    try:
        import psutil
    except Exception as error:
        raise ExactCacheStop("STOP_PSUTIL_IMPORT_FOR_RESOURCE_AUDIT") from error
    information = psutil.Process().memory_info()
    return int(information.rss), int(getattr(information, "peak_wset", information.rss))


def _strict_bool(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ExactCacheStop(f"STOP_RESULT_BOOL_TYPE:{role}:{type(value).__name__}")


def _finite_float(value: Any, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ExactCacheStop(f"STOP_RESULT_NONFINITE:{role}")
    return result


def _strict_int(value: Any, role: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ExactCacheStop(f"STOP_RESULT_INT_BOOL:{role}")
    result = int(value)
    if float(result) != float(value):
        raise ExactCacheStop(f"STOP_RESULT_INT_VALUE:{role}")
    return result


_RESULT_FLOAT_FIELDS = (
    "intended_dose",
    "target_descriptor_dose",
    "actual_effective_dose",
    "actual_to_target_dose_ratio",
    "absolute_dose_error",
    "continuous_realization_residual",
    "deployment_realization_residual",
    "quantization_residual_increment",
    "realization_residual",
    "normalized_realization_residual",
    "actual_support_deviation",
    "weighted_sign_concordance",
    "initial_delta_t",
    "initial_c_t",
    "initial_s_c",
    "solver_fun",
    "tone_preclip_pixel_fraction",
    "tone_preclip_channel_fraction",
    "tone_clip_mean_excess",
    "tone_clip_max_excess",
    "lab_to_srgb_preclip_pixel_fraction",
    "lab_to_srgb_preclip_channel_fraction",
    "gamut_clip_mean_excess",
    "gamut_clip_max_excess",
    "final_channel_at_zero_fraction",
    "final_channel_at_one_fraction",
    "chroma_weighted_mean_absolute_hue_angle_shift_degrees",
    "cstar_weighted_q50_hue_shift_degrees",
    "cstar_weighted_q90_hue_shift_degrees",
)
_RESULT_BOOL_FIELDS = (
    "actual_endpoint_in_tube",
    "actual_descriptor_internal_valid",
    "solver_success",
    "delta_t_lower_bound_hit",
    "delta_t_upper_bound_hit",
    "c_t_lower_bound_hit",
    "c_t_upper_bound_hit",
    "s_c_lower_bound_hit",
    "s_c_upper_bound_hit",
    "any_parameter_bound_hit",
)
_RESULT_INT_FIELDS = (
    "image_height",
    "image_width",
    "unique_rgb_count",
    "solver_status",
    "solver_nfev",
    "solver_nit",
)


def _science_cache_entry(
    solve: UniqueSolve,
    row: Mapping[str, Any],
    runtime: FrozenRuntimeSpec,
) -> dict[str, Any]:
    request = solve.request
    descriptor_fields = tuple(
        f"{prefix}__{name}"
        for prefix in (
            "original_raw",
            "target_raw",
            "continuous_raw",
            "deployment_raw",
            "original_std",
            "target_std",
            "continuous_std",
            "deployment_std",
            "target_delta",
            "realized_delta",
            "error",
        )
        for name in BASE
    )
    required = {
        *_RESULT_FLOAT_FIELDS,
        *_RESULT_BOOL_FIELDS,
        *_RESULT_INT_FIELDS,
        *descriptor_fields,
        "delta_t",
        "c_t",
        "s_c",
        "solver_message",
        "output_array_sha256",
        "realized_delta_sha256",
        "actual_descriptor_internal_invalid_reason",
        "target_delta_sha256",
        "input_image_sha256",
        "decoder",
        "elapsed_seconds",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ExactCacheStop("STOP_RESULT_FIELDS_MISSING:" + ",".join(missing))
    if str(row["input_image_sha256"]) != request.input_image_sha256:
        raise ExactCacheStop("STOP_RESULT_IMAGE_SHA_MISMATCH")
    if str(row["target_delta_sha256"]) != request.delta_sha256:
        raise ExactCacheStop("STOP_RESULT_DELTA_SHA_MISMATCH")
    if _float64_bytes(float(row["intended_dose"])) != _float64_bytes(
        float(request.intended_dose)
    ):
        raise ExactCacheStop("STOP_RESULT_INTENDED_DOSE_MISMATCH")
    output_sha = _require_sha256(str(row["output_array_sha256"]), "output_array")
    theta = np.asarray(
        [row["delta_t"], row["c_t"], row["s_c"]], dtype="<f8"
    )
    if theta.shape != (3,) or not np.isfinite(theta).all():
        raise ExactCacheStop("STOP_RESULT_THETA_INVALID")
    original_std = np.asarray(
        [row[f"original_std__{name}"] for name in BASE], dtype="<f8"
    )
    target_std = np.asarray(
        [row[f"target_std__{name}"] for name in BASE], dtype="<f8"
    )
    target_delta = np.asarray(
        [row[f"target_delta__{name}"] for name in BASE], dtype="<f8"
    )
    if original_std.tobytes() != _f8_vector(
        request.original_std, "request.original"
    ).tobytes():
        raise ExactCacheStop("STOP_RESULT_ORIGINAL_DESCRIPTOR_MISMATCH")
    if target_std.tobytes() != _f8_vector(
        request.target_std, "request.target"
    ).tobytes():
        raise ExactCacheStop("STOP_RESULT_TARGET_DESCRIPTOR_MISMATCH")
    # realization_case records target_delta by subtracting its frozen target
    # and original descriptors.  IEEE-754 (original + Delta) - original is
    # not generally byte-identical to the independently frozen Delta input.
    # Reproduce the old implementation's exact diagnostic operation here;
    # the original Delta remains independently guarded by Delta_sha above.
    expected_result_target_delta = np.asarray(
        _f8_vector(request.target_std, "request.target")
        - _f8_vector(request.original_std, "request.original"),
        dtype="<f8",
    )
    if target_delta.tobytes() != expected_result_target_delta.tobytes():
        raise ExactCacheStop("STOP_RESULT_TARGET_DELTA_MISMATCH")
    realized_delta = np.asarray(
        [row[f"realized_delta__{name}"] for name in BASE], dtype="<f8"
    )
    expected_realized_sha = hashlib.sha256(
        REALIZED_DELTA_HASH_PREFIX + realized_delta.tobytes(order="C")
    ).hexdigest()
    if str(row["realized_delta_sha256"]) != expected_realized_sha:
        raise ExactCacheStop("STOP_RESULT_REALIZED_DELTA_SHA_MISMATCH")

    key_payload = solve.exact_solve_key.as_dict()
    key_payload.pop("schema")
    entry: dict[str, Any] = {
        "schema": CACHE_ENTRY_SCHEMA,
        "solve_key_schema": EXACT_SOLVE_KEY_SCHEMA,
        "solve_key": solve.exact_solve_key_sha256,
        **key_payload,
        "cache_namespace": solve.cache_namespace,
        "cache_key_sha256": solve.cache_key_sha256,
        "input_image_sha256": request.input_image_sha256,
        "delta_sha256": request.delta_sha256,
        "original_descriptor_sha256": request.original_descriptor_sha256,
        "target_descriptor_sha256": request.target_descriptor_sha256,
        "support_reference_sha256": runtime.support_reference_sha256,
        **request.config.as_dict(),
        "decoder": str(row["decoder"]),
        "theta_delta_t": float(theta[0]),
        "theta_c_t": float(theta[1]),
        "theta_s_c": float(theta[2]),
        "theta_raw_f8_le_hex": theta.tobytes(order="C").hex(),
        "theta_raw_f8_sha256": theta_raw_f8_sha256(theta),
        "solver_message": str(row["solver_message"]),
        "output_array_sha256": output_sha,
        "realized_delta_sha256": expected_realized_sha,
        "actual_descriptor_internal_invalid_reason": str(
            row["actual_descriptor_internal_invalid_reason"]
        ),
    }
    for field in _RESULT_FLOAT_FIELDS:
        entry[field] = _finite_float(row[field], field)
    for field in _RESULT_BOOL_FIELDS:
        entry[field] = _strict_bool(row[field], field)
    for field in _RESULT_INT_FIELDS:
        entry[field] = _strict_int(row[field], field)
    for field in descriptor_fields:
        entry[field] = _finite_float(row[field], field)

    continuous_std = [entry[f"continuous_std__{name}"] for name in BASE]
    deployment_std = [entry[f"deployment_std__{name}"] for name in BASE]
    continuous_raw = [entry[f"continuous_raw__{name}"] for name in BASE]
    deployment_raw = [entry[f"deployment_raw__{name}"] for name in BASE]
    entry["continuous_std_descriptor_sha256"] = vector_sha256(
        continuous_std, CONTINUOUS_STD_HASH_PREFIX, "continuous_std"
    )
    entry["deployment_std_descriptor_sha256"] = vector_sha256(
        deployment_std, DEPLOYMENT_STD_HASH_PREFIX, "deployment_std"
    )
    entry["continuous_raw_descriptor_sha256"] = vector_sha256(
        continuous_raw, RAW_DESCRIPTOR_HASH_PREFIX, "continuous_raw"
    )
    entry["deployment_raw_descriptor_sha256"] = vector_sha256(
        deployment_raw, RAW_DESCRIPTOR_HASH_PREFIX, "deployment_raw"
    )
    entry["science_cache_entry_sha256"] = canonical_json_sha256(entry)

    forbidden_visit_fields = {
        "adaptation_seed",
        "epoch",
        "target_visit_ordinal",
        "schedule_uid",
        "q2_uid",
        "branch",
        "policy_flag",
        "donor_schedule_uid",
        "donor_q2_uid",
    }
    leaked = sorted(forbidden_visit_fields & set(entry))
    if leaked:
        raise ExactCacheStop("STOP_VISIT_METADATA_LEAK_IN_SCIENCE_CACHE:" + ",".join(leaked))
    return entry


@dataclass(frozen=True)
class ImageGroupExecution:
    input_image_sha256: str
    science_entries: tuple[dict[str, Any], ...]
    performance_rows: tuple[dict[str, Any], ...]


def _worker_execute_image_group(group: ImageSolveGroup) -> ImageGroupExecution:
    if _WORKER_STATE is None:
        raise ExactCacheStop("STOP_WORKER_NOT_INITIALIZED")
    runtime, frozen, median, scale, event_std = _WORKER_STATE
    group.validate()
    path = Path(group.image_path)
    if not _is_allowed_image(path, runtime.allowed_image_roots):
        raise ExactCacheStop(f"STOP_IMAGE_PATH_OUTSIDE_ALLOWED_TARGET_ADAPT:{path}")
    if not path.is_file():
        raise ExactCacheStop(f"STOP_IMAGE_FILE_MISSING:{path}")

    group_started = time.perf_counter()
    cpu_started = time.process_time()
    rss_before, peak_before = _process_memory_snapshot()
    started = time.perf_counter()
    observed_sha = sha256_file(path)
    hash_seconds = time.perf_counter() - started
    if observed_sha != group.input_image_sha256:
        raise ExactCacheStop("STOP_IMAGE_FILE_SHA_MISMATCH")

    started = time.perf_counter()
    rgb, decoder = frozen.decode_q2_rgb(path)
    decode_seconds = time.perf_counter() - started
    started = time.perf_counter()
    palette = frozen.PaletteImage.from_rgb_uint8(rgb)
    palette_seconds = time.perf_counter() - started

    science_entries: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for solve in group.unique_solves:
        if solve.request.config != runtime.config:
            raise ExactCacheStop("STOP_REQUEST_RUNTIME_CONFIG_MISMATCH")
        aliases = sorted(solve.aliases, key=lambda item: item.sort_key)
        representative = aliases[0]
        started = time.perf_counter()
        cpu_before = time.process_time()
        row = frozen.realization_case(
            palette=palette,
            uid=str(representative.q2_uid),
            event_id=str(representative.capture_event_id),
            case_type=str(representative.branch),
            input_image_sha256=solve.request.input_image_sha256,
            decoder=str(decoder),
            original_std=np.asarray(solve.request.original_std, dtype=np.float64),
            target_std=np.asarray(solve.request.target_std, dtype=np.float64),
            intended_dose=float(solve.request.intended_dose),
            target_delta_sha256=solve.request.delta_sha256,
            donor_uid=str(representative.donor_q2_uid),
            donor_event_id=str(representative.donor_event_id),
            median=median,
            scale=scale,
            event_std=event_std,
        )
        wall_seconds = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_before
        entry = _science_cache_entry(solve, row, runtime)
        science_entries.append(entry)
        rss_after_solve, peak_after_solve = _process_memory_snapshot()
        performance_rows.append(
            {
                "record_type": "unique_solve",
                "input_image_sha256": group.input_image_sha256,
                "solve_key": solve.exact_solve_key_sha256,
                "cache_namespace": solve.cache_namespace,
                "cache_key_sha256": solve.cache_key_sha256,
                "request_alias_count": len(solve.aliases),
                "solve_wall_seconds": wall_seconds,
                "solve_process_cpu_seconds": cpu_seconds,
                "solver_reported_elapsed_seconds": _finite_float(
                    row["elapsed_seconds"], "elapsed_seconds"
                ),
                "process_rss_bytes_after_solve": rss_after_solve,
                "process_peak_rss_bytes_after_solve": peak_after_solve,
            }
        )

    rss_after, peak_after = _process_memory_snapshot()
    performance_rows.append(
        {
            "record_type": "image_group",
            "input_image_sha256": group.input_image_sha256,
            "solve_key": "",
            "cache_namespace": "",
            "cache_key_sha256": "",
            "request_alias_count": sum(
                len(solve.aliases) for solve in group.unique_solves
            ),
            "unique_solve_count": len(group.unique_solves),
            "image_file_hash_count": 1,
            "image_decode_count": 1,
            "palette_invariant_preparation_count": 1,
            "realization_case_call_count": len(group.unique_solves),
            "image_hash_seconds": hash_seconds,
            "decode_seconds": decode_seconds,
            "palette_invariant_preparation_seconds": palette_seconds,
            "image_group_wall_seconds": time.perf_counter() - group_started,
            "image_group_process_cpu_seconds": time.process_time() - cpu_started,
            "process_rss_bytes_before_group": rss_before,
            "process_rss_bytes_after_group": rss_after,
            "process_peak_rss_bytes_before_group": peak_before,
            "process_peak_rss_bytes_after_group": peak_after,
        }
    )
    return ImageGroupExecution(
        input_image_sha256=group.input_image_sha256,
        science_entries=tuple(
            sorted(science_entries, key=lambda row: row["cache_key_sha256"])
        ),
        performance_rows=tuple(performance_rows),
    )


@dataclass(frozen=True)
class OfflineCacheOutputs:
    science_cache_entries: tuple[dict[str, Any], ...]
    visit_alias_rows: tuple[dict[str, Any], ...]
    performance_rows: tuple[dict[str, Any], ...]


_LEGACY_POLICY_REQUIRED_COLUMNS = (
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
_LEGACY_WORKER_REQUIRED_REALIZATION_COLUMNS = (
    "theta_delta_t",
    "theta_c_t",
    "theta_s_c",
    "input_image_sha256",
    "output_array_sha256",
    "actual_effective_dose",
    "tone_preclip_pixel_fraction",
    "lab_to_srgb_preclip_pixel_fraction",
)
_LEGACY_SAFE_SCIENCE_COLUMNS = (
    "solver_success",
    "solver_status",
    "solver_message",
    "solver_nfev",
    "solver_nit",
    "solver_fun",
    "actual_support_deviation",
    "actual_endpoint_in_tube",
    "actual_descriptor_internal_valid",
    "any_parameter_bound_hit",
    "solve_key",
    "cache_namespace",
    "cache_key_sha256",
    "theta_raw_f8_sha256",
    "solver_config_sha256",
    "operator_config_sha256",
    "descriptor_config_sha256",
    "realization_core_sha256",
    "science_cache_entry_sha256",
)
_LEGACY_FORBIDDEN_COLUMN_TOKENS = (
    "label",
    "annotation",
    "ground_truth",
    "bbox",
    "boxes",
    "class_id",
    "species",
    "is_positive",
    "box_count",
    "class",
    "positive",
    "is_empty",
    "empty",
    "groundtruth",
    "gt_",
    "_gt",
    "location",
    "gmm",
    "style_mode",
    "mode_id",
    "component",
)


@dataclass(frozen=True)
class ResourceWindowWorkerCache:
    worker_cache_rows: tuple[dict[str, Any], ...]
    audit: Mapping[str, Any]


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    return str(value)


def _schedule_bool(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(int(value))
    raise ExactCacheStop(f"STOP_RESOURCE_SCHEDULE_BOOLEAN:{role}")


def _validate_resource_schedule_state(row: Mapping[str, Any]) -> bool:
    requested = _schedule_bool(row["augment_on"], "augment_on")
    valid = _schedule_bool(row["valid"], "valid")
    flag = _optional_text(row["policy_flag"])
    failure = _optional_text(row["failure_reason"])
    delta = np.asarray([row[f"delta_{index}"] for index in range(7)], dtype="<f8")
    delta_sha = _optional_text(row["delta_sha256"])
    donor_schedule = _optional_text(row["donor_schedule_uid"])
    donor_q2 = _optional_text(row["donor_q2_uid"])
    active = requested and valid
    if not requested:
        if (
            valid
            or flag != "IDENTITY_POLICY_OFF"
            or failure != "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH"
        ):
            raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_POLICY_OFF_STATE")
    elif valid:
        if flag != "APPLY_OFFLINE_THETA" or failure:
            raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_ACTIVE_STATE")
    elif flag != "IDENTITY_FALLBACK" or not failure:
        raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_FALLBACK_STATE")
    if active:
        if not np.isfinite(delta).all():
            raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_ACTIVE_DELTA")
        if vector_sha256(delta, DELTA_HASH_PREFIX, "resource.delta") != delta_sha:
            raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_DELTA_SHA")
        if not donor_schedule or not donor_q2:
            raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_ACTIVE_DONOR")
    elif (
        not np.isnan(delta).all()
        or delta_sha
        or donor_schedule
        or donor_q2
    ):
        raise ExactCacheStop("STOP_RESOURCE_SCHEDULE_IDENTITY_PAYLOAD")
    return active


def materialize_e72_resource_worker_cache(
    *,
    schedule_rows: Iterable[Mapping[str, Any]],
    science_cache_entries: Iterable[Mapping[str, Any]],
    visit_alias_rows: Iterable[Mapping[str, Any]],
) -> ResourceWindowWorkerCache:
    """Build the frozen 240-row worker cache without altering policy columns.

    Full scientific entries and aliases remain companion tables.  This view
    exposes only the legacy worker-required realization columns plus safe
    cryptographic lineage columns.  Identity/fallback rows have null
    realization values exactly as the old left-merge cache did.
    """

    schedule = [dict(row) for row in schedule_rows]
    if len(schedule) != 240:
        raise ExactCacheStop(f"STOP_RESOURCE_WORKER_CACHE_ROW_COUNT:{len(schedule)}")
    source_columns = tuple(schedule[0]) if schedule else ()
    missing_columns = sorted(set(_LEGACY_POLICY_REQUIRED_COLUMNS) - set(source_columns))
    if missing_columns:
        raise ExactCacheStop(
            "STOP_RESOURCE_WORKER_CACHE_POLICY_COLUMNS:" + ",".join(missing_columns)
        )
    for row in schedule:
        if tuple(row) != source_columns:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_POLICY_SCHEMA_DRIFT")

    aliases: dict[tuple[Any, ...], dict[str, Any]] = {}
    for value in visit_alias_rows:
        alias = {str(key): _json_compatible(item) for key, item in value.items()}
        identity = tuple(alias[column] for column in (
            "adaptation_seed",
            "epoch",
            "target_visit_ordinal",
            "schedule_uid",
            "q2_uid",
            "branch",
        ))
        if identity in aliases:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_DUPLICATE_ALIAS")
        aliases[identity] = alias
    science: dict[str, dict[str, Any]] = {}
    for value in science_cache_entries:
        entry = {str(key): _json_compatible(item) for key, item in value.items()}
        cache_key = _require_sha256(
            str(entry.get("cache_key_sha256", "")), "resource.cache_key"
        )
        if cache_key in science:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_DUPLICATE_SCIENCE_ENTRY")
        if entry.get("schema") != CACHE_ENTRY_SCHEMA:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_SCIENCE_SCHEMA")
        science[cache_key] = entry

    realization_columns = (
        *_LEGACY_WORKER_REQUIRED_REALIZATION_COLUMNS,
        *_LEGACY_SAFE_SCIENCE_COLUMNS,
    )
    output_columns = (*source_columns, *realization_columns)
    forbidden = sorted(
        column
        for column in output_columns
        if any(token in column.casefold() for token in _LEGACY_FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ExactCacheStop(
            "STOP_RESOURCE_WORKER_CACHE_FORBIDDEN_COLUMNS:" + ",".join(forbidden)
        )

    consumed_aliases: set[tuple[Any, ...]] = set()
    output: list[dict[str, Any]] = []
    branch_rows = {branch: 0 for branch in ALLOWED_BRANCHES}
    branch_active = {branch: 0 for branch in ALLOWED_BRANCHES}
    branch_cache_keys = {branch: set() for branch in ALLOWED_BRANCHES}
    numeric_null_columns = {
        "theta_delta_t",
        "theta_c_t",
        "theta_s_c",
        "actual_effective_dose",
        "tone_preclip_pixel_fraction",
        "lab_to_srgb_preclip_pixel_fraction",
        "solver_status",
        "solver_nfev",
        "solver_nit",
        "solver_fun",
        "actual_support_deviation",
    }
    for source in sorted(
        schedule,
        key=lambda row: (
            {BRANCH_NATIVE: 0, BRANCH_MATCHED: 1, BRANCH_SMG: 2}[str(row["branch"])],
            int(row["target_visit_ordinal"]),
        ),
    ):
        branch = str(source["branch"])
        if branch not in ALLOWED_BRANCHES:
            raise ExactCacheStop(f"STOP_RESOURCE_WORKER_CACHE_BRANCH:{branch}")
        if int(source["adaptation_seed"]) != 42 or int(source["epoch"]) != 72:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_SEED_OR_EPOCH")
        branch_rows[branch] += 1
        active = _validate_resource_schedule_state(source)
        row = dict(source)
        for column in realization_columns:
            row[column] = math.nan if column in numeric_null_columns else None
        identity = tuple(source[column] for column in (
            "adaptation_seed",
            "epoch",
            "target_visit_ordinal",
            "schedule_uid",
            "q2_uid",
            "branch",
        ))
        alias = aliases.get(identity)
        if active:
            if alias is None:
                raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_ACTIVE_ALIAS_MISSING")
            consumed_aliases.add(identity)
            for field in (
                "augment_on",
                "valid",
                "policy_flag",
                "failure_reason",
                "donor_schedule_uid",
                "donor_q2_uid",
            ):
                if _json_compatible(alias.get(field)) != _json_compatible(source[field]):
                    raise ExactCacheStop(
                        f"STOP_RESOURCE_WORKER_CACHE_ALIAS_POLICY_CROSSLOCK:{field}"
                    )
            cache_key = str(alias.get("cache_key_sha256", ""))
            entry = science.get(cache_key)
            if entry is None:
                raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_SCIENCE_ENTRY_MISSING")
            if (
                str(alias.get("solve_key", "")) != str(entry.get("solve_key", ""))
                or str(alias.get("cache_namespace", ""))
                != str(entry.get("cache_namespace", ""))
                or str(source["delta_sha256"]) != str(entry.get("delta_sha256", ""))
            ):
                raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_ALIAS_SCIENCE_CROSSLOCK")
            for column in realization_columns:
                row[column] = entry[column]
            branch_active[branch] += 1
            branch_cache_keys[branch].add(cache_key)
        elif alias is not None:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_IDENTITY_ALIAS_PRESENT")
        if tuple(row) != output_columns:
            raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_OUTPUT_COLUMN_ORDER")
        output.append(row)

    if consumed_aliases != set(aliases):
        raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_UNUSED_ALIAS")
    expected_ordinals = set(range(80))
    for branch in ALLOWED_BRANCHES:
        branch_ordinals = {
            int(row["target_visit_ordinal"])
            for row in output
            if str(row["branch"]) == branch
        }
        if branch_rows[branch] != 80 or branch_ordinals != expected_ordinals:
            raise ExactCacheStop(f"STOP_RESOURCE_WORKER_CACHE_ORDINALS:{branch}")
        if branch_active[branch] != 41:
            raise ExactCacheStop(f"STOP_RESOURCE_WORKER_CACHE_ACTIVE_COUNT:{branch}")
    if branch_cache_keys[BRANCH_NATIVE] != branch_cache_keys[BRANCH_MATCHED]:
        raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_NATIVE_MATCHED_REUSE_SET")
    if branch_cache_keys[BRANCH_SMG] & branch_cache_keys[BRANCH_NATIVE]:
        raise ExactCacheStop("STOP_RESOURCE_WORKER_CACHE_SMG_NAMESPACE_INTERSECTION")

    audit = {
        "schema": "m1_routea_e72_resource_worker_cache_materialization_v1",
        "status": "PASS",
        "row_count": len(output),
        "branch_row_counts": branch_rows,
        "branch_active_counts": branch_active,
        "active_alias_count": len(consumed_aliases),
        "relevant_unique_cache_entry_count": len(
            set().union(*branch_cache_keys.values())
        ),
        "native_to_matched_reused_key_count": len(branch_cache_keys[BRANCH_NATIVE]),
        "smg_namespace_intersection_count": 0,
        "original_policy_column_count": len(source_columns),
        "original_policy_columns": list(source_columns),
        "policy_field_values_preserved_exact_in_memory": True,
        "worker_required_columns_present": True,
        "identity_realization_values_null": True,
        "forbidden_worker_column_tokens_present": False,
        "science_entries_are_companion_table": True,
        "visit_aliases_are_companion_table": True,
    }
    return ResourceWindowWorkerCache(
        worker_cache_rows=tuple(output),
        audit=audit,
    )


def _append_batch_performance(
    outputs: OfflineCacheOutputs,
    *,
    plan: ExecutionPlan,
    cache_hit_count: int,
    executed_groups: int,
    started: float,
    cpu_started: float,
    rss_before: int,
    peak_before: int,
) -> OfflineCacheOutputs:
    rss_after, peak_after = _process_memory_snapshot()
    miss_count = plan.unique_namespaced_cache_count - int(cache_hit_count)
    batch_row = {
        "record_type": "execution_batch",
        "input_image_sha256": "",
        "solve_key": "",
        "cache_namespace": "",
        "cache_key_sha256": "",
        "request_alias_count": plan.total_solve_requests,
        "unique_solve_count": plan.unique_namespaced_cache_count,
        "exact_cache_hit_count": int(cache_hit_count),
        "exact_cache_miss_count": int(miss_count),
        "image_group_count_total": len(plan.groups),
        "image_group_count_executed": int(executed_groups),
        "image_file_hash_count": int(executed_groups),
        "image_decode_count": int(executed_groups),
        "palette_invariant_preparation_count": int(executed_groups),
        "realization_case_call_count": int(miss_count),
        "execution_batch_wall_seconds": time.perf_counter() - started,
        "parent_process_cpu_seconds": time.process_time() - cpu_started,
        "parent_process_rss_bytes_before": int(rss_before),
        "parent_process_rss_bytes_after": int(rss_after),
        "parent_process_peak_rss_bytes_before": int(peak_before),
        "parent_process_peak_rss_bytes_after": int(peak_after),
    }
    return OfflineCacheOutputs(
        science_cache_entries=outputs.science_cache_entries,
        visit_alias_rows=outputs.visit_alias_rows,
        performance_rows=tuple([*outputs.performance_rows, batch_row]),
    )


def validate_existing_science_cache_entry(
    entry_value: Mapping[str, Any],
    solve: UniqueSolve,
    runtime: FrozenRuntimeSpec,
) -> dict[str, Any]:
    """Validate a disk/memory memoization entry before suppressing a solve."""

    entry = {str(key): _json_compatible(value) for key, value in entry_value.items()}
    if entry.get("schema") != CACHE_ENTRY_SCHEMA:
        raise ExactCacheStop("STOP_EXISTING_CACHE_ENTRY_SCHEMA")
    descriptor_fields = {
        f"{prefix}__{name}"
        for prefix in (
            "original_raw",
            "target_raw",
            "continuous_raw",
            "deployment_raw",
            "original_std",
            "target_std",
            "continuous_std",
            "deployment_std",
            "target_delta",
            "realized_delta",
            "error",
        )
        for name in BASE
    }
    required = {
        *_RESULT_FLOAT_FIELDS,
        *_RESULT_BOOL_FIELDS,
        *_RESULT_INT_FIELDS,
        *descriptor_fields,
        "solve_key_schema",
        "solve_key",
        "cache_namespace",
        "cache_key_sha256",
        "input_image_sha256",
        "delta_sha256",
        "original_descriptor_sha256",
        "target_descriptor_sha256",
        "support_reference_sha256",
        "theta_delta_t",
        "theta_c_t",
        "theta_s_c",
        "theta_raw_f8_le_hex",
        "theta_raw_f8_sha256",
        "solver_message",
        "output_array_sha256",
        "realized_delta_sha256",
        "actual_descriptor_internal_invalid_reason",
        "continuous_std_descriptor_sha256",
        "deployment_std_descriptor_sha256",
        "continuous_raw_descriptor_sha256",
        "deployment_raw_descriptor_sha256",
        "science_cache_entry_sha256",
        *EXACT_SOLVE_KEY_COMPONENTS,
        *solve.request.config.as_dict().keys(),
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ExactCacheStop("STOP_EXISTING_CACHE_FIELDS_MISSING:" + ",".join(missing))
    expected_entry_sha = _require_sha256(
        str(entry.get("science_cache_entry_sha256", "")),
        "existing_cache_entry",
    )
    unhashed = dict(entry)
    unhashed.pop("science_cache_entry_sha256", None)
    if canonical_json_sha256(unhashed) != expected_entry_sha:
        raise ExactCacheStop("STOP_EXISTING_CACHE_ENTRY_SHA_MISMATCH")
    if str(entry.get("solve_key", "")) != solve.exact_solve_key_sha256:
        raise ExactCacheStop("STOP_EXISTING_CACHE_SOLVE_KEY_MISMATCH")
    if str(entry.get("cache_namespace", "")) != solve.cache_namespace:
        raise ExactCacheStop("STOP_EXISTING_CACHE_NAMESPACE_MISMATCH")
    if str(entry.get("cache_key_sha256", "")) != solve.cache_key_sha256:
        raise ExactCacheStop("STOP_EXISTING_CACHE_KEY_MISMATCH")
    if str(entry.get("solve_key_schema", "")) != EXACT_SOLVE_KEY_SCHEMA:
        raise ExactCacheStop("STOP_EXISTING_CACHE_SOLVE_KEY_SCHEMA")
    key_payload = solve.exact_solve_key.as_dict()
    key_payload.pop("schema")
    for field, expected in key_payload.items():
        if str(entry.get(field, "")) != expected:
            raise ExactCacheStop(f"STOP_EXISTING_CACHE_KEY_COMPONENT:{field}")
    for field, expected in solve.request.config.as_dict().items():
        if str(entry.get(field, "")) != expected:
            raise ExactCacheStop(f"STOP_EXISTING_CACHE_CONFIG:{field}")
    if str(entry.get("support_reference_sha256", "")) != runtime.support_reference_sha256:
        raise ExactCacheStop("STOP_EXISTING_CACHE_SUPPORT_REFERENCE")
    if str(entry.get("input_image_sha256", "")) != solve.request.input_image_sha256:
        raise ExactCacheStop("STOP_EXISTING_CACHE_INPUT_IMAGE_SHA")
    if str(entry.get("delta_sha256", "")) != solve.request.delta_sha256:
        raise ExactCacheStop("STOP_EXISTING_CACHE_DELTA_SHA")

    theta_hex = str(entry.get("theta_raw_f8_le_hex", ""))
    try:
        theta_bytes = bytes.fromhex(theta_hex)
    except ValueError as error:
        raise ExactCacheStop("STOP_EXISTING_CACHE_THETA_HEX") from error
    if len(theta_bytes) != 3 * np.dtype("<f8").itemsize:
        raise ExactCacheStop("STOP_EXISTING_CACHE_THETA_BYTE_COUNT")
    theta = np.frombuffer(theta_bytes, dtype="<f8")
    expected_theta = np.asarray(
        [
            entry.get("theta_delta_t"),
            entry.get("theta_c_t"),
            entry.get("theta_s_c"),
        ],
        dtype="<f8",
    )
    if not np.isfinite(theta).all() or theta.tobytes() != expected_theta.tobytes():
        raise ExactCacheStop("STOP_EXISTING_CACHE_THETA_VALUE_MISMATCH")
    if str(entry.get("theta_raw_f8_sha256", "")) != theta_raw_f8_sha256(theta):
        raise ExactCacheStop("STOP_EXISTING_CACHE_THETA_SHA_MISMATCH")
    operator = json.loads(runtime.operator_config_json)["operator"]
    lower = np.asarray(
        [
            operator["delta_t_bounds"][0],
            operator["c_t_bounds"][0],
            operator["s_c_bounds"][0],
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            operator["delta_t_bounds"][1],
            operator["c_t_bounds"][1],
            operator["s_c_bounds"][1],
        ],
        dtype=np.float64,
    )
    if np.any(theta < lower) or np.any(theta > upper):
        raise ExactCacheStop("STOP_EXISTING_CACHE_THETA_BOUNDS")
    _require_sha256(str(entry.get("output_array_sha256", "")), "existing.output")
    for field in _RESULT_FLOAT_FIELDS:
        _finite_float(entry[field], f"existing.{field}")
    for field in _RESULT_BOOL_FIELDS:
        _strict_bool(entry[field], f"existing.{field}")
    for field in _RESULT_INT_FIELDS:
        _strict_int(entry[field], f"existing.{field}")
    for field in descriptor_fields:
        _finite_float(entry[field], f"existing.{field}")
    for prefix, hash_field, hash_prefix in (
        ("continuous_std", "continuous_std_descriptor_sha256", CONTINUOUS_STD_HASH_PREFIX),
        ("deployment_std", "deployment_std_descriptor_sha256", DEPLOYMENT_STD_HASH_PREFIX),
        ("continuous_raw", "continuous_raw_descriptor_sha256", RAW_DESCRIPTOR_HASH_PREFIX),
        ("deployment_raw", "deployment_raw_descriptor_sha256", RAW_DESCRIPTOR_HASH_PREFIX),
    ):
        values = [entry[f"{prefix}__{name}"] for name in BASE]
        if str(entry[hash_field]) != vector_sha256(values, hash_prefix, f"existing.{prefix}"):
            raise ExactCacheStop(f"STOP_EXISTING_CACHE_DESCRIPTOR_SHA:{prefix}")
    realized = np.asarray(
        [entry[f"realized_delta__{name}"] for name in BASE], dtype="<f8"
    )
    expected_realized_sha = hashlib.sha256(
        REALIZED_DELTA_HASH_PREFIX + realized.tobytes(order="C")
    ).hexdigest()
    if str(entry["realized_delta_sha256"]) != expected_realized_sha:
        raise ExactCacheStop("STOP_EXISTING_CACHE_REALIZED_DELTA_SHA")
    forbidden_visit_fields = {
        "adaptation_seed",
        "epoch",
        "target_visit_ordinal",
        "schedule_uid",
        "q2_uid",
        "branch",
        "policy_flag",
        "donor_schedule_uid",
        "donor_q2_uid",
    }
    if forbidden_visit_fields & set(entry):
        raise ExactCacheStop("STOP_EXISTING_CACHE_VISIT_METADATA_LEAK")
    return entry


def _materialize_outputs(
    plan: ExecutionPlan,
    results: Sequence[ImageGroupExecution],
    *,
    executed_groups: Sequence[ImageSolveGroup] | None = None,
    existing_entries: Sequence[dict[str, Any]] = (),
) -> OfflineCacheOutputs:
    groups = tuple(plan.groups if executed_groups is None else executed_groups)
    expected_images = [group.input_image_sha256 for group in groups]
    observed_images = [result.input_image_sha256 for result in results]
    if observed_images != expected_images:
        raise ExactCacheStop("STOP_EXECUTION_IMAGE_RESULT_ORDER_OR_SET")
    science = tuple(
        sorted(
            (
                *existing_entries,
                *(entry for result in results for entry in result.science_entries),
            ),
            key=lambda row: row["cache_key_sha256"],
        )
    )
    expected_cache_keys = {
        solve.cache_key_sha256
        for group in plan.groups
        for solve in group.unique_solves
    }
    observed_cache_keys = {str(row["cache_key_sha256"]) for row in science}
    if observed_cache_keys != expected_cache_keys or len(science) != len(expected_cache_keys):
        raise ExactCacheStop("STOP_EXECUTION_CACHE_ENTRY_SET_MISMATCH")
    aliases = plan.visit_alias_rows()
    if len(aliases) != plan.total_solve_requests:
        raise ExactCacheStop("STOP_EXECUTION_ALIAS_COUNT_MISMATCH")
    alias_counts = {
        solve.cache_key_sha256: len(solve.aliases)
        for group in plan.groups
        for solve in group.unique_solves
    }
    sorted_existing = sorted(
        existing_entries, key=lambda entry: str(entry["cache_key_sha256"])
    )
    performance = tuple(
        [
            {
                "record_type": "exact_cache_hit",
                "input_image_sha256": str(entry["input_image_sha256"]),
                "solve_key": str(entry["solve_key"]),
                "cache_namespace": str(entry["cache_namespace"]),
                "cache_key_sha256": str(entry["cache_key_sha256"]),
                "request_alias_count": alias_counts[str(entry["cache_key_sha256"])],
                "solve_wall_seconds": 0.0,
                "solve_process_cpu_seconds": 0.0,
                "solver_reported_elapsed_seconds": 0.0,
            }
            for entry in sorted_existing
        ]
        + [row for result in results for row in result.performance_rows]
    )
    return OfflineCacheOutputs(
        science_cache_entries=science,
        visit_alias_rows=aliases,
        performance_rows=performance,
    )


class PersistentSpawnPool:
    """Reusable spawn-based pool; no worker is recreated between submissions."""

    def __init__(
        self,
        runtime: FrozenRuntimeSpec,
        *,
        max_workers: int,
        chunksize: int = 1,
    ) -> None:
        runtime.validate_static()
        if int(max_workers) <= 0 or int(chunksize) <= 0:
            raise ExactCacheStop("STOP_POOL_NONPOSITIVE_CONFIGURATION")
        self.runtime = runtime
        self.max_workers = int(max_workers)
        self.chunksize = int(chunksize)
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None

    def start(self) -> "PersistentSpawnPool":
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=context,
                initializer=_worker_initializer,
                initargs=(self.runtime,),
            )
        return self

    def execute(
        self,
        plan: ExecutionPlan,
        *,
        existing_science_cache_entries: Iterable[Mapping[str, Any]] = (),
    ) -> OfflineCacheOutputs:
        batch_started = time.perf_counter()
        batch_cpu_started = time.process_time()
        batch_rss_before, batch_peak_before = _process_memory_snapshot()
        plan.validate()
        expected_solves = {
            solve.cache_key_sha256: solve
            for group in plan.groups
            for solve in group.unique_solves
        }
        for group in plan.groups:
            for solve in group.unique_solves:
                if solve.request.config != self.runtime.config:
                    raise ExactCacheStop("STOP_PLAN_POOL_CONFIG_MISMATCH")
        existing_by_key: dict[str, dict[str, Any]] = {}
        for value in existing_science_cache_entries:
            cache_key = str(value.get("cache_key_sha256", ""))
            if cache_key in existing_by_key:
                raise ExactCacheStop("STOP_EXISTING_CACHE_DUPLICATE_KEY")
            solve = expected_solves.get(cache_key)
            if solve is None:
                raise ExactCacheStop("STOP_EXISTING_CACHE_KEY_NOT_IN_PLAN")
            existing_by_key[cache_key] = validate_existing_science_cache_entry(
                value, solve, self.runtime
            )

        missing_groups: list[ImageSolveGroup] = []
        for group in plan.groups:
            missing_solves = tuple(
                solve
                for solve in group.unique_solves
                if solve.cache_key_sha256 not in existing_by_key
            )
            if missing_solves:
                missing_group = ImageSolveGroup(
                    input_image_sha256=group.input_image_sha256,
                    image_path=group.image_path,
                    unique_solves=missing_solves,
                )
                missing_group.validate()
                missing_groups.append(missing_group)
        if not missing_groups:
            outputs = _materialize_outputs(
                plan,
                (),
                executed_groups=(),
                existing_entries=tuple(existing_by_key.values()),
            )
            return _append_batch_performance(
                outputs,
                plan=plan,
                cache_hit_count=len(existing_by_key),
                executed_groups=0,
                started=batch_started,
                cpu_started=batch_cpu_started,
                rss_before=batch_rss_before,
                peak_before=batch_peak_before,
            )

        self.start()
        assert self._executor is not None
        try:
            results = list(
                self._executor.map(
                    _worker_execute_image_group,
                    tuple(missing_groups),
                    chunksize=self.chunksize,
                )
            )
        except Exception as error:
            raise ExactCacheStop("STOP_PERSISTENT_POOL_EXECUTION_FAILED") from error
        outputs = _materialize_outputs(
            plan,
            results,
            executed_groups=tuple(missing_groups),
            existing_entries=tuple(existing_by_key.values()),
        )
        return _append_batch_performance(
            outputs,
            plan=plan,
            cache_hit_count=len(existing_by_key),
            executed_groups=len(missing_groups),
            started=batch_started,
            cpu_started=batch_cpu_started,
            rss_before=batch_rss_before,
            peak_before=batch_peak_before,
        )

    def close(self, *, cancel_futures: bool = False) -> None:
        if self._executor is not None:
            self._executor.shutdown(
                wait=True,
                cancel_futures=bool(cancel_futures),
            )
            self._executor = None

    def __enter__(self) -> "PersistentSpawnPool":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(cancel_futures=exc_type is not None)


def implementation_unit_tests() -> dict[str, Any]:
    """Pure planner/hash tests; intentionally performs no image read or solve."""

    digest = lambda character: character * 64
    config = ScientificConfigIdentity(
        solver_config_sha256=digest("1"),
        operator_config_sha256=digest("2"),
        descriptor_config_sha256=digest("3"),
    )
    original = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    delta = np.asarray([0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.04], dtype=np.float64)
    delta_sha = vector_sha256(delta, DELTA_HASH_PREFIX, "test.delta")
    request = make_solve_request(
        image_path=r"D:\allowed\image_a.jpg",
        input_image_sha256=digest("a"),
        original_std=original,
        delta_std=delta,
        delta_sha256=delta_sha,
        config=config,
    )

    def alias(branch: str, ordinal: int) -> VisitAlias:
        return VisitAlias(
            adaptation_seed=42,
            epoch=72,
            target_visit_ordinal=ordinal,
            schedule_uid=f"schedule_{ordinal}.jpg",
            q2_uid="q2_a",
            capture_event_id="event_a",
            branch=branch,
        )

    native_matched = (
        (request, alias(BRANCH_NATIVE, 0)),
        (request, alias(BRANCH_MATCHED, 1)),
    )
    evidence = build_native_matched_subset_evidence(native_matched)
    plan = build_execution_plan(native_matched, subset_evidence=evidence)
    if plan.unique_namespaced_cache_count != 1 or plan.total_solve_requests != 2:
        raise AssertionError("Native/Matched exact dedup failed")
    unique = plan.groups[0].unique_solves[0]
    if unique.cache_namespace != NATIVE_CACHE_NAMESPACE:
        raise AssertionError("Native/Matched namespace failed")

    evidence_required = False
    try:
        build_execution_plan(native_matched)
    except ExactCacheStop as error:
        evidence_required = "SUBSET_EVIDENCE_REQUIRED" in str(error)
    if not evidence_required:
        raise AssertionError("Native/Matched reuse did not fail closed")

    smg_pair = (request, alias(BRANCH_SMG, 2))
    combined_pairs = native_matched + (smg_pair,)
    combined_evidence = build_native_matched_subset_evidence(combined_pairs)
    combined = build_execution_plan(
        combined_pairs, subset_evidence=combined_evidence
    )
    namespaces = {
        solve.cache_namespace
        for group in combined.groups
        for solve in group.unique_solves
    }
    if namespaces != {NATIVE_CACHE_NAMESPACE, SMG_CACHE_NAMESPACE}:
        raise AssertionError("SMG namespace isolation failed")
    if combined.unique_exact_solve_count != 1:
        raise AssertionError("Scientific solve-key equality test failed")
    if combined.unique_namespaced_cache_count != 2:
        raise AssertionError("Namespaced execution-key isolation failed")

    corrupt = dict(combined_evidence)
    corrupt["missing_key_count"] = 1
    corrupt_rejected = False
    try:
        build_execution_plan(combined_pairs, subset_evidence=corrupt)
    except ExactCacheStop as error:
        corrupt_rejected = "MISSING_NONZERO" in str(error)
    if not corrupt_rejected:
        raise AssertionError("Corrupt subset evidence was accepted")

    theta = np.asarray([0.125, 1.0, 1.5], dtype=np.float64)
    theta_sha = theta_raw_f8_sha256(theta)
    if theta_sha != theta_raw_f8_sha256(theta.copy()):
        raise AssertionError("theta raw-f8 hash is not deterministic")

    fake_row: dict[str, Any] = {field: 0.0 for field in _RESULT_FLOAT_FIELDS}
    fake_row.update({field: False for field in _RESULT_BOOL_FIELDS})
    fake_row.update({field: 1 for field in _RESULT_INT_FIELDS})
    fake_row["solver_nit"] = 0
    fake_row["intended_dose"] = request.intended_dose
    fake_row["target_descriptor_dose"] = request.intended_dose
    fake_row["solver_success"] = True
    fake_row["actual_descriptor_internal_valid"] = True
    fake_result_target_delta = np.asarray(
        np.asarray(request.target_std, dtype="<f8")
        - np.asarray(request.original_std, dtype="<f8"),
        dtype="<f8",
    )
    fake_row.update(
        {
            "delta_t": float(theta[0]),
            "c_t": float(theta[1]),
            "s_c": float(theta[2]),
            "solver_message": "synthetic-no-solve",
            "output_array_sha256": digest("b"),
            "actual_descriptor_internal_invalid_reason": "",
            "target_delta_sha256": request.delta_sha256,
            "input_image_sha256": request.input_image_sha256,
            "decoder": "synthetic-no-image-read",
            "elapsed_seconds": 0.0,
        }
    )
    for index, name in enumerate(BASE):
        fake_row[f"original_raw__{name}"] = float(original[index])
        fake_row[f"target_raw__{name}"] = float(request.target_std[index])
        fake_row[f"continuous_raw__{name}"] = float(request.target_std[index])
        fake_row[f"deployment_raw__{name}"] = float(request.target_std[index])
        fake_row[f"original_std__{name}"] = float(request.original_std[index])
        fake_row[f"target_std__{name}"] = float(request.target_std[index])
        fake_row[f"continuous_std__{name}"] = float(request.target_std[index])
        fake_row[f"deployment_std__{name}"] = float(request.target_std[index])
        fake_row[f"target_delta__{name}"] = float(fake_result_target_delta[index])
        fake_row[f"realized_delta__{name}"] = float(request.delta_std[index])
        fake_row[f"error__{name}"] = 0.0
    realized = np.asarray(request.delta_std, dtype="<f8")
    fake_row["realized_delta_sha256"] = hashlib.sha256(
        REALIZED_DELTA_HASH_PREFIX + realized.tobytes(order="C")
    ).hexdigest()

    class _RuntimeStub:
        support_reference_sha256 = digest("4")
        operator_config_json = json.dumps(
            {
                "operator": {
                    "delta_t_bounds": [-math.log(2.0), math.log(2.0)],
                    "c_t_bounds": [0.5, 2.0],
                    "s_c_bounds": [0.5, 2.0],
                }
            }
        )

    fake_entry = _science_cache_entry(unique, fake_row, _RuntimeStub())  # type: ignore[arg-type]
    if fake_entry["schema"] != CACHE_ENTRY_SCHEMA:
        raise AssertionError("science cache schema was overwritten")
    if fake_entry["theta_raw_f8_sha256"] != theta_sha:
        raise AssertionError("science cache theta hash mismatch")
    validate_existing_science_cache_entry(
        fake_entry, unique, _RuntimeStub()  # type: ignore[arg-type]
    )
    if any(
        field in fake_entry
        for field in ("epoch", "target_visit_ordinal", "schedule_uid", "branch")
    ):
        raise AssertionError("visit metadata leaked into science cache")

    smg_unique = next(
        solve
        for group in combined.groups
        for solve in group.unique_solves
        if solve.cache_namespace == SMG_CACHE_NAMESPACE
    )
    fake_smg_entry = _science_cache_entry(
        smg_unique, fake_row, _RuntimeStub()  # type: ignore[arg-type]
    )
    resource_schedule: list[dict[str, Any]] = []
    resource_aliases: list[dict[str, Any]] = []
    entry_by_branch = {
        BRANCH_NATIVE: fake_entry,
        BRANCH_MATCHED: fake_entry,
        BRANCH_SMG: fake_smg_entry,
    }
    for branch in ALLOWED_BRANCHES:
        for ordinal in range(80):
            active = ordinal < 41
            schedule_row = {
                "adaptation_seed": 42,
                "epoch": 72,
                "target_visit_ordinal": ordinal,
                "schedule_uid": f"resource_{ordinal}.jpg",
                "q2_uid": "q2_a",
                "branch": branch,
                "augment_on": active,
                "valid": active,
                "policy_flag": "APPLY_OFFLINE_THETA" if active else "IDENTITY_POLICY_OFF",
                "failure_reason": "" if active else "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH",
                "delta_sha256": request.delta_sha256 if active else "",
                **{
                    f"delta_{index}": float(request.delta_std[index]) if active else math.nan
                    for index in range(7)
                },
                "donor_schedule_uid": "donor_schedule.jpg" if active else "",
                "donor_q2_uid": "donor_q2" if active else "",
            }
            resource_schedule.append(schedule_row)
            if active:
                entry = entry_by_branch[branch]
                resource_aliases.append(
                    {
                        "adaptation_seed": 42,
                        "epoch": 72,
                        "target_visit_ordinal": ordinal,
                        "schedule_uid": schedule_row["schedule_uid"],
                        "q2_uid": "q2_a",
                        "branch": branch,
                        "augment_on": True,
                        "valid": True,
                        "policy_flag": "APPLY_OFFLINE_THETA",
                        "failure_reason": "",
                        "donor_schedule_uid": "donor_schedule.jpg",
                        "donor_q2_uid": "donor_q2",
                        "solve_key": entry["solve_key"],
                        "cache_namespace": entry["cache_namespace"],
                        "cache_key_sha256": entry["cache_key_sha256"],
                    }
                )
    worker_cache = materialize_e72_resource_worker_cache(
        schedule_rows=resource_schedule,
        science_cache_entries=(fake_entry, fake_smg_entry),
        visit_alias_rows=resource_aliases,
    )
    if len(worker_cache.worker_cache_rows) != 240:
        raise AssertionError("resource worker cache row count failed")
    if worker_cache.audit["active_alias_count"] != 123:
        raise AssertionError("resource worker cache active count failed")
    inactive = next(
        row
        for row in worker_cache.worker_cache_rows
        if not bool(row["augment_on"])
    )
    if not all(
        inactive[column] is None
        or (isinstance(inactive[column], float) and math.isnan(inactive[column]))
        for column in _LEGACY_WORKER_REQUIRED_REALIZATION_COLUMNS
    ):
        raise AssertionError("resource identity realization values are not null")
    return {
        "schema": "m1_routea_offline_theta_cache_engine_unit_tests_v1",
        "status": "PASS",
        "powell_calls": 0,
        "image_reads": 0,
        "native_matched_evidence_required": True,
        "native_matched_unique_cache_entries": 1,
        "smg_namespace_strongly_isolated": True,
        "combined_scientific_key_count": 1,
        "combined_namespaced_cache_count": 2,
        "theta_raw_f8_sha256": theta_sha,
        "science_visit_metadata_separation": True,
        "science_cache_entry_schema": fake_entry["schema"],
        "resource_worker_cache_rows": len(worker_cache.worker_cache_rows),
        "resource_worker_cache_active_aliases": worker_cache.audit["active_alias_count"],
        "resource_worker_cache_forbidden_columns": False,
    }


__all__ = [
    "ENGINE_SCHEMA",
    "EXACT_SOLVE_KEY_SCHEMA",
    "ExactCacheStop",
    "ExecutionPlan",
    "FrozenRuntimeSpec",
    "ImageSolveGroup",
    "OfflineCacheOutputs",
    "PersistentSpawnPool",
    "ResourceWindowWorkerCache",
    "ScientificConfigIdentity",
    "SolveRequest",
    "UniqueSolve",
    "VisitAlias",
    "build_execution_plan",
    "build_frozen_runtime_spec",
    "build_native_matched_subset_evidence",
    "canonical_json_sha256",
    "implementation_unit_tests",
    "materialize_e72_resource_worker_cache",
    "make_explicit_target_solve_request",
    "make_solve_request",
    "namespaced_cache_key_sha256",
    "sha256_file",
    "support_matrix_sha256",
    "theta_raw_f8_sha256",
    "validate_existing_science_cache_entry",
    "vector_sha256",
]


if __name__ == "__main__":
    if sys.argv[1:] != ["--self-test"]:
        raise SystemExit("This module only supports --self-test; it has no solve CLI.")
    print(json.dumps(implementation_unit_tests(), indent=2, sort_keys=True))
