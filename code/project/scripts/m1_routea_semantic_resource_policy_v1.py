from __future__ import annotations

"""Frozen visit-policy construction for the Route-A semantic/resource gate.

This module is deliberately limited to descriptor-space policy instantiation and
read-only aggregation of the completed photometric-realization evidence.  It
does not decode images, run a detector, solve photometric parameters, or train.

The CLI is epoch-resumable.  Each completed epoch is an immutable partition
whose audit is written last.  A rerun skips a partition only after validating
its lineage key and every recorded output SHA256.
"""

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import struct
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


EXPERIMENT = "M1_ROUTE_A_SEMANTIC_RESOURCE_GATE_MEASUREMENT_V1"
PHASE = "VISIT_POLICY_AND_EXISTING_REALIZATION_AGGREGATION_V1"

BASE = [
    "tone_ln_y_q10",
    "tone_ln_y_q50",
    "tone_ln_y_q90",
    "contrast_global_q90_minus_q10",
    "contrast_local_grid8_median_q90_minus_q10",
    "chroma_cstar_q50",
    "chroma_cstar_q90",
]
WEIGHTS = np.asarray([1 / 9, 1 / 9, 1 / 9, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
DELTA_HASH_PREFIX = b"M1_ROUTEA_DELTA_FLOAT64_LE_V1\0"

P_AUG = 0.5
FORMAL_EPOCH_START = 72
FORMAL_EPOCH_END = 100
EXPECTED_TARGET_IMAGES = 3272
EXPECTED_TARGET_EVENTS = 2373
EXPECTED_STEPS_PER_EPOCH = 552
EXPECTED_TARGET_BATCH = 8
EXPECTED_VISITS_PER_EPOCH = EXPECTED_STEPS_PER_EPOCH * EXPECTED_TARGET_BATCH
REQUIRED_NETWORKX_VERSION = "3.4.2"

# The visit key is the sole policy-randomness key.  Namespaces are independent.
NS_AUGMENT = "M1_ROUTEA_VISIT_AUGMENT_ON_V1"
NS_DOSE = "M1_ROUTEA_VISIT_TUBE_DOSE_V1"
NS_DESTINATION = "M1_ROUTEA_VISIT_TSVA_DESTINATION_V1"
# Matching retains the exact Geometry-V1 objective/integerization formula, with
# the separately frozen visit-level namespace and full formal visit identity.
NS_MATCHING_TIEBREAK = "M1_ROUTEA_VISIT_MATCHING_TIEBREAK_V1"

PRESOLVE_ADDENDUM_REL = Path(
    "project/contracts/M1_ROUTEA_EXACT_MATCHING_PRESOLVE_ENGINEERING_ADDENDUM_V1.json"
)
PRESOLVE_ADDENDUM_SHA256 = "5a6be115d65b60d21d69f3376dc533422416c220b64ed2fc2143d26848e69382"
PRESOLVE_D = 1_000_000
PRESOLVE_MARGIN_NUMERATORS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
PRESOLVE_MARGIN_DENOMINATOR = 1000
PRESOLVE_LINPROG_METHOD = "highs-ipm"
PRESOLVE_REQUIRED_SCIPY_VERSION = "1.15.3"
PRESOLVE_REQUIRED_HIGHS_VERSION = "1.8.0"
PRESOLVE_LINPROG_OPTIONS = {
    "threads": 1,
    "parallel": False,
    "random_seed": 0,
    "presolve": True,
}

WORKER_SCHEDULE_COLUMNS = [
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
] + [f"delta_{coordinate}" for coordinate in range(7)] + [
    "donor_schedule_uid",
    "donor_q2_uid",
]

DEFAULT_GEOMETRY_REL = Path(
    "project/audits/M1_ROUTE_A_DESCRIPTOR_GEOMETRY_FEASIBILITY_MEASUREMENT_V1_20260808_225541"
)
DEFAULT_REALIZATION_REL = Path(
    "project/audits/M1_ROUTE_A_PHOTOMETRIC_REALIZATION_FEASIBILITY_MEASUREMENT_V1_20260809_131730"
)
DEFAULT_Q2_REL = Path(
    "project/audits/M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
)
DEFAULT_SCHEDULE_REL = Path("project/contracts/protocol_b_target_schedule_e71_e100.jsonl")

# These hashes close the exact already-completed scientific lineage.  CLI path
# overrides are allowed only for byte-identical mirrors.
EXPECTED_HASHES = {
    "geometry_contract": "5e4bb9ab6560196e0fc37eed96d23403090901700d2d27a01217a4f526dcafb4",
    "geometry_matching_contract": "dcc9b18e139ac1f32473b4307d009fbaa32c2e0cc5a30e2ed33b412bbd103549",
    "geometry_final_audit": "a838f6350cfa8dd809191f91d328b5d74c6c4c121ea79ceb43f26fa7ce946f27",
    "geometry_support_summary": "ab9e27ecace7b060c84d35908b2de0a6ffcac7b851b36d2c99df720217c754a8",
    "geometry_image_tube": "98c5ce48b0ad5d85f725f0d090cceed43911769db8667d6faec66e4cb2be0e80",
    "geometry_equal_anchor_pairs": "8299ec41b790ee0a14280af0e82b3fc4abd4645300d3ac1b2dd8be23285458ed",
    "geometry_directed_pairs": "e2382786057232828d6a1cc81e767660e8c78392bec092d4fbbe8608422751ad",
    "q2_whitelist": "24426506f2fba321ca1c98f593d459dca3e9e76d030c0303299eacd1ad76b29f",
    "q2_image_descriptors": "2281400f9601203437df6a8fefd06e8516ff760189faba54d7126abca62068cc",
    "q2_event_prototypes": "0519d5d08b97647f9a84a95e1d1913e88d2c695d7b7876c19a8ff699f0479d02",
    "q2_descriptor_schema": "628a8b2da5927e816c4291d6622824076214d62486092ddfcabfc435c5f7f8f9",
    "target_schedule": "d4dc5afea3adacec2f5562a10cf4f10226d751ada3b52f0c5a69aa1949d5b923",
    "realization_native": "7e2e49724c7055e43969bc014a3746169653e551e3f4653f6e0e7761ea4b53f5",
    "realization_matched_native": "f4410daf5d6da4b71df71e372aaf29b08be1579d3702b05adf488f1bbf55edba",
    "realization_smg": "905e42a1c480b0543ef0206e55c5056ee9b1ac041d29e0b3383c26ff2e515e3f",
}


class PolicyStop(RuntimeError):
    """Fail-closed contract or lineage error."""


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stateless_digest(namespace: str, *parts: Any) -> bytes:
    key = namespace + "|" + "|".join(str(part) for part in parts)
    return hashlib.sha256(key.encode("utf-8")).digest()


def stateless_u01(namespace: str, *parts: Any) -> float:
    integer = int.from_bytes(stateless_digest(namespace, *parts)[:8], "big", signed=False)
    return integer / float(1 << 64)


def stateless_index(n: int, namespace: str, *parts: Any) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    modulus = 1 << 64
    limit = modulus - modulus % n
    counter = 0
    while True:
        integer = int.from_bytes(stateless_digest(namespace, *parts, counter)[:8], "big", signed=False)
        if integer < limit:
            return int(integer % n)
        counter += 1


def delta_hash(delta: Sequence[float]) -> str:
    values = np.asarray(delta, dtype="<f8")
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("delta must be one finite seven-dimensional vector")
    return hashlib.sha256(DELTA_HASH_PREFIX + values.tobytes(order="C")).hexdigest()


def dstyle_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(
        np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)) * WEIGHTS,
        axis=-1,
        dtype=np.float64,
    )


def inverse_standardize(z_std: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(z_std, dtype=np.float64) * scale + median


def validity_arrays(
    z_std: np.ndarray, median: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    standardized = np.atleast_2d(np.asarray(z_std, dtype=np.float64))
    raw = inverse_standardize(standardized, median, scale)
    finite = np.isfinite(standardized).all(axis=1) & np.isfinite(raw).all(axis=1)
    tone = (raw[:, 0] <= raw[:, 1]) & (raw[:, 1] <= raw[:, 2])
    contrast = (raw[:, 3] >= 0.0) & (raw[:, 4] >= 0.0)
    chroma = raw[:, 5] <= raw[:, 6]
    valid = finite & tone & contrast & chroma
    reasons: list[str] = []
    for row in range(len(raw)):
        parts: list[str] = []
        if not finite[row]:
            parts.append("NONFINITE")
        if not tone[row]:
            parts.append("TONE_QUANTILE_ORDER")
        if not contrast[row]:
            parts.append("NEGATIVE_CONTRAST")
        if not chroma[row]:
            parts.append("CHROMA_QUANTILE_ORDER")
        reasons.append("|".join(parts))
    return valid, reasons


def inverse_ecdf_weighted(values: Sequence[float], weights: Sequence[float], p: float) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(value) == 0 or len(value) != len(weight):
        raise ValueError("inverse ECDF requires equal, nonempty vectors")
    if not np.isfinite(value).all() or not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("inverse ECDF requires finite values and finite nonnegative weights")
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    weight = weight[order]
    unique, first = np.unique(value, return_index=True)
    merged = np.add.reduceat(weight, first)
    total = float(merged.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("inverse ECDF requires positive total weight")
    if p <= 0.0:
        return float(unique[0])
    if p >= 1.0:
        return float(unique[-1])
    cdf = np.cumsum(merged / total, dtype=np.float64)
    index = min(int(np.searchsorted(cdf, p, side="left")), len(unique) - 1)
    return float(unique[index])


def empirical_summary(values: Sequence[float]) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(value) == 0:
        return {"count": 0, "Q10": math.nan, "Q50": math.nan, "Q90": math.nan, "mean": math.nan}
    weights = np.ones(len(value), dtype=np.float64)
    return {
        "count": int(len(value)),
        "Q10": inverse_ecdf_weighted(value, weights, 0.10),
        "Q50": inverse_ecdf_weighted(value, weights, 0.50),
        "Q90": inverse_ecdf_weighted(value, weights, 0.90),
        "mean": float(value.mean(dtype=np.float64)),
    }


def wasserstein_1d(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    left = np.asarray(values_a, dtype=np.float64).reshape(-1)
    right = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if len(left) == 0 or len(right) == 0:
        return math.nan
    support = np.unique(np.concatenate([left, right]))
    if len(support) <= 1:
        return 0.0
    mass_left = np.zeros(len(support), dtype=np.float64)
    mass_right = np.zeros(len(support), dtype=np.float64)
    np.add.at(mass_left, np.searchsorted(support, left), 1.0 / len(left))
    np.add.at(mass_right, np.searchsorted(support, right), 1.0 / len(right))
    cdf_left = np.cumsum(mass_left, dtype=np.float64)
    cdf_right = np.cumsum(mass_right, dtype=np.float64)
    return float(np.sum(np.abs(cdf_left[:-1] - cdf_right[:-1]) * np.diff(support), dtype=np.float64))


def strict_bool_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    allowed = {"true", "false"}
    observed = set(normalized.unique().tolist())
    if not observed.issubset(allowed):
        raise PolicyStop(f"STOP_BOOLEAN_SCHEMA_INVALID_{label}: {sorted(observed)}")
    return normalized.map({"true": True, "false": False}).astype(bool)


def visit_key(adaptation_seed: int, epoch: int, ordinal: int, target_uid: str) -> str:
    # Length-prefixing keeps the node identity injective even if UID punctuation
    # changes in a future byte-identical mirror.
    uid = str(target_uid)
    return f"seed={adaptation_seed}|epoch={epoch:03d}|ordinal={ordinal:04d}|uidlen={len(uid)}|uid={uid}"


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_replaceable(path: Path, value: Any) -> None:
    _atomic_replace_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def write_csv_replaceable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def write_parquet_replaceable(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def _commit_immutable(path: Path, writer: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer(temporary)
    candidate_hash = sha256_file(temporary)
    if path.exists():
        existing_hash = sha256_file(path)
        temporary.unlink()
        if existing_hash != candidate_hash:
            raise PolicyStop(f"STOP_RESUME_IMMUTABLE_OUTPUT_CONFLICT: {path}")
        return existing_hash
    os.replace(temporary, path)
    return candidate_hash


def write_parquet_immutable(path: Path, frame: pd.DataFrame) -> str:
    return _commit_immutable(
        path,
        lambda temporary: frame.to_parquet(
            temporary, index=False, engine="pyarrow", compression="zstd"
        ),
    )


def write_json_immutable(path: Path, value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    return _commit_immutable(path, lambda temporary: temporary.write_bytes(payload))


def write_npz_immutable(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)

    return _commit_immutable(path, writer)


def verify_hash(role: str, path: Path) -> str:
    if role not in EXPECTED_HASHES:
        raise KeyError(role)
    if not path.is_file():
        raise PolicyStop(f"STOP_REQUIRED_INPUT_MISSING_{role}: {path}")
    actual = sha256_file(path)
    if actual != EXPECTED_HASHES[role]:
        raise PolicyStop(
            f"STOP_INPUT_SHA_MISMATCH_{role}: actual={actual} expected={EXPECTED_HASHES[role]} path={path}"
        )
    return actual


def schedule_uid_basename(value: str) -> str:
    normalized = str(value).replace("\\", "/")
    result = PurePosixPath(normalized).name
    if not result or result != normalized:
        # The formal schedule is filename-based.  Refuse path-tail inference.
        raise PolicyStop(f"STOP_SCHEDULE_UID_NOT_BASENAME_ONLY: {value}")
    return result


def build_schedule_q2_crosswalk(whitelist: pd.DataFrame, schedule_uids: Iterable[str]) -> dict[str, str]:
    required = {"uid", "image_path"}
    if not required.issubset(whitelist.columns):
        raise PolicyStop("STOP_Q2_WHITELIST_SCHEMA_MISMATCH")
    basenames = whitelist["image_path"].astype(str).map(lambda value: Path(value).name)
    if (basenames == "").any() or basenames.duplicated().any():
        duplicates = sorted(basenames[basenames.duplicated(keep=False)].unique().tolist())[:10]
        raise PolicyStop(f"STOP_Q2_SCHEDULE_BASENAME_CROSSWALK_AMBIGUOUS: {duplicates}")
    mapping = dict(zip(basenames.tolist(), whitelist["uid"].astype(str).tolist()))
    schedule_set = {schedule_uid_basename(value) for value in schedule_uids}
    if schedule_set != set(mapping):
        missing = sorted(schedule_set - set(mapping))[:10]
        extra = sorted(set(mapping) - schedule_set)[:10]
        raise PolicyStop(f"STOP_Q2_SCHEDULE_UID_SET_MISMATCH: missing={missing} extra={extra}")
    return {name: mapping[name] for name in sorted(mapping)}


def load_schedule(path: Path, epoch_start: int, epoch_end: int) -> dict[int, pd.DataFrame]:
    rows: dict[int, list[dict[str, Any]]] = {epoch: [] for epoch in range(epoch_start, epoch_end + 1)}
    ordinal_by_epoch = Counter()
    batch_seen: dict[int, list[int]] = {epoch: [] for epoch in rows}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            if epoch < epoch_start or epoch > epoch_end:
                continue
            batch = int(record["batch"])
            uids = list(record["uids"])
            if len(uids) != EXPECTED_TARGET_BATCH:
                raise PolicyStop(
                    f"STOP_TARGET_SCHEDULE_BATCH_SIZE_MISMATCH: line={line_number} count={len(uids)}"
                )
            batch_seen[epoch].append(batch)
            for position, uid in enumerate(uids):
                ordinal = int(ordinal_by_epoch[epoch])
                ordinal_by_epoch[epoch] += 1
                rows[epoch].append(
                    {
                        "epoch": epoch,
                        "batch": batch,
                        "target_position_in_batch": position,
                        "target_visit_ordinal": ordinal,
                        "schedule_uid": schedule_uid_basename(str(uid)),
                    }
                )
    result: dict[int, pd.DataFrame] = {}
    for epoch in range(epoch_start, epoch_end + 1):
        expected_batches = list(range(EXPECTED_STEPS_PER_EPOCH))
        if batch_seen[epoch] != expected_batches:
            raise PolicyStop(f"STOP_TARGET_SCHEDULE_BATCH_SEQUENCE_MISMATCH_E{epoch:03d}")
        frame = pd.DataFrame(rows[epoch])
        if len(frame) != EXPECTED_VISITS_PER_EPOCH:
            raise PolicyStop(f"STOP_TARGET_SCHEDULE_VISIT_COUNT_MISMATCH_E{epoch:03d}: {len(frame)}")
        result[epoch] = frame
    return result


@dataclass(frozen=True)
class MatchingSolution:
    pairs: list[tuple[int, int]]
    cost_int: np.ndarray
    tie_digest: np.ndarray
    tie_rank: np.ndarray
    solver_weights: list[int]
    cost_priority: int
    canonical_input_sha256: str
    presolve_certificate: dict[str, Any]
    presolve_arrays: dict[str, np.ndarray]


def _highs_version() -> str:
    try:
        from scipy.optimize._highspy import _core as highs_core

        return (
            f"{int(highs_core.HIGHS_VERSION_MAJOR)}."
            f"{int(highs_core.HIGHS_VERSION_MINOR)}."
            f"{int(highs_core.HIGHS_VERSION_PATCH)}"
        )
    except Exception:
        return "UNAVAILABLE"


def _integer_storage(values: Sequence[int] | np.ndarray) -> tuple[np.ndarray, str]:
    if isinstance(values, np.ndarray) and values.dtype == np.int64:
        return values.astype("<i8", copy=False), "INT64_LE"
    materialized = [int(value) for value in values]
    limit = np.iinfo(np.int64)
    if not materialized or (min(materialized) >= limit.min and max(materialized) <= limit.max):
        return np.asarray(materialized, dtype="<i8"), "INT64_LE"
    encoded = [str(value).encode("ascii") for value in materialized]
    width = max((len(value) for value in encoded), default=1)
    return np.asarray(encoded, dtype=f"S{width}"), "ASCII_DECIMAL_FIXED_WIDTH"


def _integer_sequence_sha256(values: Sequence[int] | np.ndarray) -> tuple[str, np.ndarray, str]:
    stored, encoding = _integer_storage(values)
    digest = hashlib.sha256()
    digest.update(b"M1_ROUTEA_EXACT_INTEGER_SEQUENCE_V1\0")
    digest.update(encoding.encode("ascii") + b"\0")
    digest.update(struct.pack("<Q", len(stored)))
    digest.update(stored.tobytes(order="C"))
    return digest.hexdigest(), stored, encoding


def _edge_index_sha256(indices: np.ndarray) -> str:
    normalized = np.asarray(indices, dtype="<i8").reshape(-1)
    digest = hashlib.sha256()
    digest.update(b"M1_ROUTEA_FULL_EDGE_INDICES_V1\0")
    digest.update(struct.pack("<Q", len(normalized)))
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _matching_edge_indices(
    pairs: Sequence[tuple[int, int]], left: np.ndarray, right: np.ndarray, node_count: int
) -> np.ndarray:
    edge_code = left * np.int64(node_count) + right
    if len(edge_code) and np.any(edge_code[1:] <= edge_code[:-1]):
        order = np.argsort(edge_code, kind="mergesort")
        sorted_code = edge_code[order]
    else:
        order = None
        sorted_code = edge_code
    if len(sorted_code) and np.any(sorted_code[1:] == sorted_code[:-1]):
        raise PolicyStop("STOP_MATCHING_FULL_EDGE_DUPLICATE")
    pair_code = np.asarray(
        [int(a) * node_count + int(b) for a, b in pairs], dtype=np.int64
    )
    positions = np.searchsorted(sorted_code, pair_code)
    if np.any(positions >= len(sorted_code)) or np.any(sorted_code[positions] != pair_code):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_OUTPUT_NOT_FULL_EDGE")
    return positions if order is None else order[positions]


def _networkx_matching_on_edge_indices(
    node_count: int,
    left: np.ndarray,
    right: np.ndarray,
    solver_weights: Sequence[int],
    edge_indices: np.ndarray,
) -> list[tuple[int, int]]:
    selected = np.asarray(edge_indices, dtype=np.int64).reshape(-1)
    if len(selected) and (np.any(selected < 0) or np.any(selected >= len(left))):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_EDGE_INDEX_OUT_OF_RANGE")
    if len(selected) > 1 and np.any(selected[1:] <= selected[:-1]):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_EDGE_ORDER_CHANGED")
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    for raw_index in selected:
        index = int(raw_index)
        graph.add_edge(
            int(left[index]),
            int(right[index]),
            solver_weight=solver_weights[index],
            edge_index=index,
        )
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=True, weight="solver_weight"
    )
    return sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in matching)


def _exact_optimality_safe_presolve(
    node_keys: Sequence[str],
    left: np.ndarray,
    right: np.ndarray,
    cost_int: np.ndarray,
    tie_rank: np.ndarray,
    solver_weights: list[int],
    cost_priority: int,
    canonical_input_sha256: str,
    allow_full_graph_fallback: bool,
) -> tuple[list[tuple[int, int]], dict[str, Any], dict[str, np.ndarray]]:
    node_count = len(node_keys)
    edge_count = len(left)
    cardinality_target = node_count // 2
    full_indices = np.arange(edge_count, dtype=np.int64)
    cost_sha, cost_stored, cost_encoding = _integer_sequence_sha256(cost_int)
    tie_sha, tie_stored, tie_encoding = _integer_sequence_sha256(tie_rank)
    weight_sha, weight_stored, weight_encoding = _integer_sequence_sha256(solver_weights)
    certificate: dict[str, Any] = {
        "schema": "m1_routea_exact_matching_presolve_certificate_v1",
        "engineering_addendum_path": str(PRESOLVE_ADDENDUM_REL).replace("\\", "/"),
        "engineering_addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
        "scientific_contract_modified": False,
        "matching_mathematical_objective_modified": False,
        "full_node_count": node_count,
        "full_edge_count": edge_count,
        "full_node_keys_unique_verified": True,
        "full_simple_undirected_edges_unique_verified": True,
        "fixed_cardinality_k": cardinality_target,
        "full_matching_input_sha256": canonical_input_sha256,
        "full_primary_cost_int_sha256": cost_sha,
        "full_primary_cost_int_encoding": cost_encoding,
        "full_global_tie_rank_sha256": tie_sha,
        "full_global_tie_rank_encoding": tie_encoding,
        "full_combined_solver_weight_sha256": weight_sha,
        "full_combined_solver_weight_encoding": weight_encoding,
        "global_cost_priority": int(cost_priority),
        "full_fields_frozen_before_presolve": True,
        "tie_rank_recomputed_after_pruning": False,
        "cost_priority_recomputed_after_pruning": False,
        "edge_order_recomputed_after_pruning": False,
        "D": PRESOLVE_D,
        "margin_numerators": list(PRESOLVE_MARGIN_NUMERATORS),
        "margin_denominator": PRESOLVE_MARGIN_DENOMINATOR,
        "linprog_api": "scipy.optimize.linprog",
        "linprog_method": PRESOLVE_LINPROG_METHOD,
        "linprog_options": dict(PRESOLVE_LINPROG_OPTIONS),
        "scipy_version": scipy.__version__,
        "required_scipy_version": PRESOLVE_REQUIRED_SCIPY_VERSION,
        "highs_version": _highs_version(),
        "required_highs_version": PRESOLVE_REQUIRED_HIGHS_VERSION,
        "networkx_solver": "networkx.algorithms.matching.max_weight_matching",
        "networkx_version": nx.__version__,
        "candidate_attempts": [],
        "certified_candidate_accepted": False,
        "fallback_full_graph": True,
        "fallback_reason": "UNSET",
    }
    arrays: dict[str, np.ndarray] = {
        "full_primary_cost_int": cost_stored,
        "full_global_tie_rank": tie_stored,
        "full_combined_solver_weight": weight_stored,
    }
    selected_pairs: list[tuple[int, int]] | None = None
    retained_indices = full_indices
    pruned_indices = np.asarray([], dtype=np.int64)
    fallback_reason = "PRESOLVE_NOT_CERTIFIED"

    try:
        if node_count < 2 or edge_count == 0 or cardinality_target == 0:
            raise RuntimeError("EMPTY_OR_ZERO_CARDINALITY_GRAPH")
        if scipy.__version__ != PRESOLVE_REQUIRED_SCIPY_VERSION:
            raise RuntimeError("SCIPY_VERSION_MISMATCH")
        if _highs_version() != PRESOLVE_REQUIRED_HIGHS_VERSION:
            raise RuntimeError("HIGHS_VERSION_MISMATCH")
        if int(cost_int.max(initial=0)) > 2**53:
            raise RuntimeError("PRIMARY_COST_NOT_EXACTLY_REPRESENTABLE_IN_FLOAT64")

        nonzeros = edge_count * 3
        column_dtype = np.int32 if node_count + 1 <= np.iinfo(np.int32).max else np.int64
        columns = np.empty(nonzeros, dtype=column_dtype)
        columns[0::3] = 0
        columns[1::3] = left + 1
        columns[2::3] = right + 1
        coefficients = np.empty(nonzeros, dtype=np.float64)
        coefficients[0::3] = 1.0
        coefficients[1::3] = -1.0
        coefficients[2::3] = -1.0
        indptr = np.arange(0, nonzeros + 1, 3, dtype=np.int64)
        constraints = csr_matrix(
            (coefficients, columns, indptr), shape=(edge_count, node_count + 1)
        )
        objective = np.ones(node_count + 1, dtype=np.float64)
        objective[0] = -float(cardinality_target)
        bounds = [(None, None)] + [(0.0, None)] * node_count
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lp = linprog(
                objective,
                A_ub=constraints,
                b_ub=cost_int.astype(np.float64, copy=False),
                bounds=bounds,
                method=PRESOLVE_LINPROG_METHOD,
                options=dict(PRESOLVE_LINPROG_OPTIONS),
            )
        certificate["linprog_warnings"] = [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ]
        certificate["linprog_success"] = bool(lp.success)
        certificate["linprog_status"] = int(lp.status)
        certificate["linprog_message"] = str(lp.message)
        lp_nit = getattr(lp, "nit", None)
        lp_crossover_nit = getattr(lp, "crossover_nit", None)
        certificate["linprog_nit"] = int(lp_nit) if lp_nit is not None else -1
        certificate["linprog_crossover_nit"] = (
            int(lp_crossover_nit) if lp_crossover_nit is not None else -1
        )
        lp_fun = getattr(lp, "fun", None)
        certificate["linprog_fun"] = (
            float(lp_fun)
            if lp_fun is not None and math.isfinite(float(lp_fun))
            else None
        )
        if not lp.success or lp.x is None or len(lp.x) != node_count + 1:
            raise RuntimeError(f"LINPROG_FAILURE_STATUS_{int(lp.status)}")
        if not np.isfinite(lp.x).all():
            raise RuntimeError("LINPROG_NONFINITE_DUAL")

        alpha = float(lp.x[0])
        lambdas = np.asarray(lp.x[1:], dtype=np.float64)
        scaled_alpha = int(math.floor(alpha * PRESOLVE_D))
        scaled_lambda_values = [
            max(0, int(math.ceil(float(value) * PRESOLVE_D))) for value in lambdas
        ]
        lambda_sha, lambda_stored, lambda_encoding = _integer_sequence_sha256(
            scaled_lambda_values
        )
        arrays["dual_lambda_scaled"] = lambda_stored
        certificate["dual_alpha_float"] = alpha
        certificate["dual_alpha_scaled_A"] = scaled_alpha
        certificate["dual_lambda_scaled_sha256"] = lambda_sha
        certificate["dual_lambda_scaled_encoding"] = lambda_encoding
        certificate["dual_lambda_scaled_count"] = len(scaled_lambda_values)

        sum_lambda = sum(scaled_lambda_values)
        global_lb = cardinality_target * scaled_alpha - sum_lambda
        base = (cardinality_target - 1) * scaled_alpha - sum_lambda
        limit = np.iinfo(np.int64)
        max_lambda = max(scaled_lambda_values, default=0)
        max_cost = int(cost_int.max(initial=0))
        vector_bound = max(
            abs(scaled_alpha) + 2 * max_lambda + PRESOLVE_D * max_cost,
            abs(base) + 2 * max_lambda + PRESOLVE_D * max_cost,
        )
        if (
            lambda_encoding == "INT64_LE"
            and limit.min <= scaled_alpha <= limit.max
            and vector_bound <= limit.max
        ):
            lambda_i64 = lambda_stored.astype(np.int64, copy=False)
            rhs = cost_int * np.int64(PRESOLVE_D)
            constraint_delta = (
                np.int64(scaled_alpha) - lambda_i64[left] - lambda_i64[right] - rhs
            )
            constraint_max = int(constraint_delta.max(initial=limit.min))
            constraint_violation_count = int(np.count_nonzero(constraint_delta > 0))
            forced_lb_values: Sequence[int] | np.ndarray = (
                rhs + np.int64(base) + lambda_i64[left] + lambda_i64[right]
            )
            exact_arithmetic_mode = "VECTORIZED_INT64_PREBOUNDED"
        else:
            constraint_max_python: int | None = None
            constraint_violation_count = 0
            forced_python: list[int] = []
            for u, v, primary in zip(left, right, cost_int):
                lambda_u = scaled_lambda_values[int(u)]
                lambda_v = scaled_lambda_values[int(v)]
                delta_exact = (
                    scaled_alpha - lambda_u - lambda_v - PRESOLVE_D * int(primary)
                )
                constraint_max_python = (
                    delta_exact
                    if constraint_max_python is None
                    else max(constraint_max_python, delta_exact)
                )
                constraint_violation_count += int(delta_exact > 0)
                forced_python.append(
                    PRESOLVE_D * int(primary) + base + lambda_u + lambda_v
                )
            forced_lb_values = forced_python
            constraint_max = 0 if constraint_max_python is None else constraint_max_python
            exact_arithmetic_mode = "PYTHON_INT"
        certificate["exact_integer_arithmetic_mode"] = exact_arithmetic_mode
        certificate["dual_constraint_max_lhs_minus_rhs"] = constraint_max
        certificate["dual_constraint_violation_count"] = constraint_violation_count
        certificate["dual_full_edge_constraint_count"] = edge_count
        certificate["dual_full_edge_constraints_valid"] = constraint_max <= 0
        certificate["global_LB_scaled"] = int(global_lb)
        certificate["sum_dual_lambda_scaled"] = int(sum_lambda)
        if constraint_violation_count > 0:
            raise RuntimeError("EXACT_DUAL_CONSTRAINT_VIOLATION")

        forced_sha, forced_stored, forced_encoding = _integer_sequence_sha256(
            forced_lb_values
        )
        arrays["forced_edge_LB_scaled"] = forced_stored
        certificate["forced_edge_LB_scaled_sha256"] = forced_sha
        certificate["forced_edge_LB_scaled_encoding"] = forced_encoding
        threshold_scale = max(abs(global_lb), PRESOLVE_D)

        for numerator in PRESOLVE_MARGIN_NUMERATORS:
            margin_scaled = (
                numerator * threshold_scale + PRESOLVE_MARGIN_DENOMINATOR - 1
            ) // PRESOLVE_MARGIN_DENOMINATOR
            threshold = global_lb + margin_scaled
            if isinstance(forced_lb_values, np.ndarray):
                if threshold < limit.min:
                    retained_mask = np.zeros(edge_count, dtype=bool)
                elif threshold >= limit.max:
                    retained_mask = np.ones(edge_count, dtype=bool)
                else:
                    retained_mask = forced_lb_values <= np.int64(threshold)
            else:
                retained_mask = np.fromiter(
                    (int(value) <= threshold for value in forced_lb_values),
                    dtype=bool,
                    count=edge_count,
                )
            candidate_indices = np.flatnonzero(retained_mask).astype(np.int64, copy=False)
            excluded_indices = np.flatnonzero(~retained_mask).astype(np.int64, copy=False)
            attempt: dict[str, Any] = {
                "margin_numerator": int(numerator),
                "margin_denominator": PRESOLVE_MARGIN_DENOMINATOR,
                "threshold_margin_scaled": int(margin_scaled),
                "threshold_T_scaled": int(threshold),
                "retained_edge_count": int(len(candidate_indices)),
                "pruned_edge_count": int(len(excluded_indices)),
                "retained_edge_indices_sha256": _edge_index_sha256(candidate_indices),
                "pruned_edge_indices_sha256": _edge_index_sha256(excluded_indices),
                "all_full_nodes_inserted": True,
                "edge_insertion_order_preserved": True,
                "global_weights_reused": True,
            }
            try:
                candidate_pairs = _networkx_matching_on_edge_indices(
                    node_count, left, right, solver_weights, candidate_indices
                )
                candidate_pair_indices = _matching_edge_indices(
                    candidate_pairs, left, right, node_count
                )
                candidate_cardinality = len(candidate_pairs)
                candidate_primary = sum(int(cost_int[index]) for index in candidate_pair_indices)
                candidate_tie_rank_sum = sum(
                    int(tie_rank[index]) for index in candidate_pair_indices
                )
                candidate_solver_weight_sum = sum(
                    int(solver_weights[index]) for index in candidate_pair_indices
                )
                candidate_combined_penalty = -candidate_solver_weight_sum
                candidate_combined_expected = (
                    candidate_primary * cost_priority + candidate_tie_rank_sum
                )
                if candidate_combined_penalty != candidate_combined_expected:
                    raise PolicyStop("STOP_MATCHING_PRESOLVE_COMBINED_SUM_INVALID")
                excluded_min = (
                    min(int(forced_lb_values[index]) for index in excluded_indices)
                    if len(excluded_indices)
                    else None
                )
                excluded_strict = excluded_min is None or excluded_min > threshold
                primary_bound = PRESOLVE_D * candidate_primary
                accepted = (
                    candidate_cardinality == cardinality_target
                    and primary_bound <= threshold
                    and excluded_strict
                )
                attempt.update(
                    {
                        "solver_success": True,
                        "candidate_cardinality": candidate_cardinality,
                        "candidate_primary_cost_int_sum": int(candidate_primary),
                        "candidate_primary_cost_scaled_D": int(primary_bound),
                        "candidate_tie_rank_sum": int(candidate_tie_rank_sum),
                        "candidate_solver_weight_sum_decimal": str(candidate_solver_weight_sum),
                        "candidate_combined_penalty_sum_decimal": str(candidate_combined_penalty),
                        "minimum_excluded_forced_edge_LB_scaled": excluded_min,
                        "excluded_LB_strictly_above_T": bool(excluded_strict),
                        "D_times_Cprimary_le_T": bool(primary_bound <= threshold),
                        "certificate_accepted": bool(accepted),
                    }
                )
                certificate["candidate_attempts"].append(attempt)
                if accepted:
                    selected_pairs = candidate_pairs
                    retained_indices = candidate_indices
                    pruned_indices = excluded_indices
                    certificate["certified_candidate_accepted"] = True
                    certificate["fallback_full_graph"] = False
                    certificate["fallback_reason"] = ""
                    certificate["accepted_margin_numerator"] = int(numerator)
                    certificate["accepted_threshold_T_scaled"] = int(threshold)
                    certificate["certificate_proof"] = (
                        "candidate_cardinality==k AND D*Cprimary<=T AND every excluded forced-edge LB>T"
                    )
                    break
            except PolicyStop:
                raise
            except Exception as error:
                attempt.update(
                    {
                        "solver_success": False,
                        "solver_error": f"{type(error).__name__}: {error}",
                        "certificate_accepted": False,
                    }
                )
                certificate["candidate_attempts"].append(attempt)
                fallback_reason = f"CANDIDATE_SOLVER_FAILURE_{type(error).__name__}"
                break
        if selected_pairs is None and fallback_reason == "PRESOLVE_NOT_CERTIFIED":
            fallback_reason = "ALL_MARGIN_CERTIFICATES_REJECTED"
    except PolicyStop:
        raise
    except Exception as error:
        fallback_reason = f"{type(error).__name__}: {error}"
        certificate.setdefault("linprog_success", False)
        certificate.setdefault("linprog_warnings", [])

    if selected_pairs is None:
        if not allow_full_graph_fallback:
            raise PolicyStop(
                "STOP_MATCHING_PRESOLVE_CERTIFICATE_NOT_ACCEPTED_FULL_FALLBACK_DISABLED"
            )
        selected_pairs = _networkx_matching_on_edge_indices(
            node_count, left, right, solver_weights, full_indices
        )
        retained_indices = full_indices
        pruned_indices = np.asarray([], dtype=np.int64)
        certificate["certified_candidate_accepted"] = False
        certificate["fallback_full_graph"] = True
        certificate["fallback_reason"] = fallback_reason

    selected_edge_indices = _matching_edge_indices(selected_pairs, left, right, node_count)
    selected_primary = sum(int(cost_int[index]) for index in selected_edge_indices)
    selected_tie_rank_sum = sum(int(tie_rank[index]) for index in selected_edge_indices)
    selected_solver_weight_sum = sum(int(solver_weights[index]) for index in selected_edge_indices)
    selected_combined_penalty = -selected_solver_weight_sum
    selected_combined_expected = selected_primary * cost_priority + selected_tie_rank_sum
    if selected_combined_penalty != selected_combined_expected:
        raise PolicyStop("STOP_MATCHING_PRESOLVE_COMBINED_SUM_INVALID")
    certificate.update(
        {
            "final_retained_edge_count": int(len(retained_indices)),
            "final_pruned_edge_count": int(len(pruned_indices)),
            "final_retained_edge_indices_sha256": _edge_index_sha256(retained_indices),
            "final_pruned_edge_indices_sha256": _edge_index_sha256(pruned_indices),
            "selected_matching_cardinality": int(len(selected_pairs)),
            "selected_primary_cost_int_sum": int(selected_primary),
            "selected_tie_rank_sum": int(selected_tie_rank_sum),
            "selected_solver_weight_sum_decimal": str(selected_solver_weight_sum),
            "selected_combined_penalty_sum_decimal": str(selected_combined_penalty),
            "selected_pair_full_edge_membership": True,
            "selected_pair_edge_indices_sha256": _edge_index_sha256(selected_edge_indices),
            "exact_combined_sum_from_full_frozen_weights": True,
            "networkx_maxcardinality": True,
            "greedy_fallback": False,
        }
    )
    arrays["final_retained_edge_indices"] = retained_indices.astype("<i8", copy=False)
    arrays["final_pruned_edge_indices"] = pruned_indices.astype("<i8", copy=False)
    arrays["selected_pair_full_edge_indices"] = selected_edge_indices.astype("<i8", copy=False)
    return selected_pairs, certificate, arrays


def solve_general_graph_matching(
    node_keys: Sequence[str],
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    fairness_cost: np.ndarray,
    *,
    allow_full_graph_fallback: bool = True,
) -> MatchingSolution:
    """Exact Geometry-V1 objective with certified execution-only safe presolve."""
    if nx.__version__ != REQUIRED_NETWORKX_VERSION:
        raise PolicyStop(
            f"STOP_MATCHING_IMPLEMENTATION_VERSION_MISMATCH: {nx.__version__} != {REQUIRED_NETWORKX_VERSION}"
        )
    left = np.asarray(edge_i, dtype=np.int64).reshape(-1)
    right = np.asarray(edge_j, dtype=np.int64).reshape(-1)
    cost = np.asarray(fairness_cost, dtype=np.float64).reshape(-1)
    if not (len(left) == len(right) == len(cost)):
        raise ValueError("matching edge arrays must have equal length")
    if np.any(left >= right) or np.any(left < 0) or np.any(right >= len(node_keys)):
        raise ValueError("matching edges must be unique upper-triangle node pairs")
    if len(set(map(str, node_keys))) != len(node_keys):
        raise PolicyStop("STOP_MATCHING_NODE_KEY_DUPLICATE")
    if len(left) > 1:
        strictly_lexicographic = (left[1:] > left[:-1]) | (
            (left[1:] == left[:-1]) & (right[1:] > right[:-1])
        )
        if not strictly_lexicographic.all():
            pair_order = np.lexsort((right, left))
            ordered_left = left[pair_order]
            ordered_right = right[pair_order]
            duplicate = (ordered_left[1:] == ordered_left[:-1]) & (
                ordered_right[1:] == ordered_right[:-1]
            )
            if duplicate.any():
                raise PolicyStop("STOP_MATCHING_FULL_EDGE_DUPLICATE")
    if not np.isfinite(cost).all() or np.any(cost < 0.0):
        raise ValueError("matching costs must be finite and nonnegative")

    rounded = np.rint(cost * 1e9)
    limit = np.iinfo(np.int64)
    if np.any(rounded < limit.min) or np.any(rounded > limit.max):
        raise PolicyStop("STOP_MATCHING_COST_INTEGER_OVERFLOW")
    cost_int = rounded.astype(np.int64)

    digests: list[bytes] = []
    key_lo: list[str] = []
    key_hi: list[str] = []
    for a, b in zip(left.tolist(), right.tolist()):
        lo, hi = sorted((str(node_keys[a]), str(node_keys[b])))
        key_lo.append(lo)
        key_hi.append(hi)
        digests.append(stateless_digest(NS_MATCHING_TIEBREAK, lo, hi))
    digest_array = np.asarray(digests, dtype="S32") if digests else np.asarray([], dtype="S32")
    tie_order = np.argsort(digest_array, kind="mergesort") if len(digests) else np.asarray([], dtype=np.int64)
    if len(digests):
        sorted_digest = digest_array[tie_order]
        start = 0
        while start < len(tie_order):
            end = start + 1
            while end < len(tie_order) and sorted_digest[end] == sorted_digest[start]:
                end += 1
            if end - start > 1:
                members = tie_order[start:end].tolist()
                members.sort(key=lambda index: (key_lo[index], key_hi[index]))
                tie_order[start:end] = members
            start = end
    tie_rank = np.empty(len(digests), dtype=np.int64)
    if len(digests):
        tie_rank[tie_order] = np.arange(len(digests), dtype=np.int64)

    max_matching_edges = len(node_keys) // 2
    tie_rank_max = max(len(digests) - 1, 1)
    cost_priority = max_matching_edges * tie_rank_max + 1
    solver_weights = [
        -(int(cost_int[index]) * cost_priority + int(tie_rank[index]))
        for index in range(len(digests))
    ]

    canonical = hashlib.sha256()
    canonical.update(b"M1_ROUTEA_VISIT_MATCHING_INPUT_V1\0")
    for key in node_keys:
        encoded = str(key).encode("utf-8")
        canonical.update(struct.pack("<Q", len(encoded)))
        canonical.update(encoded)
    canonical.update(np.asarray(left, dtype="<i8").tobytes(order="C"))
    canonical.update(np.asarray(right, dtype="<i8").tobytes(order="C"))
    canonical.update(np.asarray(cost, dtype="<f8").tobytes(order="C"))
    canonical.update(np.asarray(cost_int, dtype="<i8").tobytes(order="C"))
    canonical.update(digest_array.tobytes(order="C"))
    canonical.update(np.asarray(tie_rank, dtype="<i8").tobytes(order="C"))
    canonical_sha256 = canonical.hexdigest()

    pairs, presolve_certificate, presolve_arrays = _exact_optimality_safe_presolve(
        node_keys,
        left,
        right,
        cost_int,
        tie_rank,
        solver_weights,
        cost_priority,
        canonical_sha256,
        allow_full_graph_fallback,
    )
    return MatchingSolution(
        pairs=pairs,
        cost_int=cost_int,
        tie_digest=digest_array,
        tie_rank=tie_rank,
        solver_weights=solver_weights,
        cost_priority=cost_priority,
        canonical_input_sha256=canonical_sha256,
        presolve_certificate=presolve_certificate,
        presolve_arrays=presolve_arrays,
    )


@dataclass
class PolicyContext:
    median: np.ndarray
    scale: np.ndarray
    tau: float
    q10_tube: float
    q90_tube: float
    pmf_values: np.ndarray
    pmf_weights: np.ndarray
    pmf_cdf: np.ndarray
    event_ids: np.ndarray
    event_index: dict[str, int]
    z_event: np.ndarray
    event_tree: cKDTree
    neighbors: dict[str, tuple[np.ndarray, np.ndarray]]
    image_uids: np.ndarray
    image_index: dict[str, int]
    image_event_ids: np.ndarray
    z_image: np.ndarray
    original_in_tube: np.ndarray
    original_internal_valid: np.ndarray
    original_invalid_reason: np.ndarray
    schedule_to_q2_uid: dict[str, str]
    input_hashes: dict[str, str]
    lineage_sha256: str


def _input_paths(
    geometry_dir: Path, realization_dir: Path, q2_dir: Path, schedule_path: Path
) -> dict[str, Path]:
    return {
        "geometry_contract": geometry_dir / "ROUTEA_DESCRIPTOR_GEOMETRY_CONTRACT.json",
        "geometry_matching_contract": geometry_dir / "ROUTEA_MATCHING_CONTRACT.json",
        "geometry_final_audit": geometry_dir / "FINAL_AUDIT.json",
        "geometry_support_summary": geometry_dir / "ROUTEA_SUPPORT_TUBE_SUMMARY.csv",
        "geometry_image_tube": geometry_dir / "raw/image_to_event_tube_distances.csv",
        "geometry_equal_anchor_pairs": geometry_dir / "raw/equal_anchor_pair_weights.parquet",
        "geometry_directed_pairs": geometry_dir / "raw/directed_tube_pairs.parquet",
        "q2_whitelist": q2_dir / "TARGET_ADAPT_STYLE_AUDIT_WHITELIST.csv",
        "q2_image_descriptors": q2_dir / "IMAGE_STYLE_DESCRIPTORS.parquet",
        "q2_event_prototypes": q2_dir / "EVENT_STYLE_PROTOTYPES.csv",
        "q2_descriptor_schema": q2_dir / "STYLE_DESCRIPTOR_SCHEMA.json",
        "target_schedule": schedule_path,
        "realization_native": realization_dir / "raw/realization_native_cases.parquet",
        "realization_matched_native": realization_dir / "raw/realization_matched_native_cases.parquet",
        "realization_smg": realization_dir / "raw/realization_smg_cases.parquet",
    }


def load_policy_context(
    geometry_dir: Path,
    q2_dir: Path,
    schedule_path: Path,
    schedule_uids: Iterable[str],
) -> PolicyContext:
    paths = _input_paths(geometry_dir, Path("."), q2_dir, schedule_path)
    policy_roles = [
        "geometry_contract",
        "geometry_matching_contract",
        "geometry_final_audit",
        "geometry_support_summary",
        "geometry_image_tube",
        "geometry_equal_anchor_pairs",
        "geometry_directed_pairs",
        "q2_whitelist",
        "q2_image_descriptors",
        "q2_event_prototypes",
        "q2_descriptor_schema",
        "target_schedule",
    ]
    input_hashes = {role: verify_hash(role, paths[role]) for role in policy_roles}

    geometry_final = json.loads(paths["geometry_final_audit"].read_text(encoding="utf-8"))
    if geometry_final.get("status") != "MEASUREMENT_COMPLETE":
        raise PolicyStop("STOP_GEOMETRY_FORMAL_STATUS_INVALID")
    if geometry_final.get("matching_solver_version") != REQUIRED_NETWORKX_VERSION:
        raise PolicyStop("STOP_GEOMETRY_MATCHING_VERSION_LINEAGE_INVALID")
    metrics = geometry_final["metrics"]
    tau = float(metrics["tau"])
    q10_tube = float(metrics["Q10_tube"])
    q90_tube = float(metrics["Q90_tube"])
    if not (0.0 < q10_tube <= q90_tube <= 2.0 * tau):
        raise PolicyStop("STOP_FROZEN_TUBE_DOSE_BOUNDS_INVALID")

    schema = json.loads(paths["q2_descriptor_schema"].read_text(encoding="utf-8"))
    if schema.get("base_descriptors") != BASE or schema.get("double_normalization") is not False:
        raise PolicyStop("STOP_Q2_DESCRIPTOR_SCHEMA_MISMATCH")
    standardization = schema["standardization"]
    median = np.asarray([standardization[name]["median"] for name in BASE], dtype=np.float64)
    scale = np.asarray(
        [standardization[name]["iqr"] + standardization[name]["epsilon_scale"] for name in BASE],
        dtype=np.float64,
    )
    if not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise PolicyStop("STOP_Q2_NORMALIZATION_INVALID")

    whitelist = pd.read_csv(
        paths["q2_whitelist"],
        usecols=["uid", "image_path", "image_sha256", "capture_event_id", "location_id"],
        dtype=str,
        keep_default_na=False,
    )
    if len(whitelist) != EXPECTED_TARGET_IMAGES or whitelist["uid"].nunique() != len(whitelist):
        raise PolicyStop("STOP_TARGET_WHITELIST_COUNT_OR_UID_MISMATCH")
    crosswalk = build_schedule_q2_crosswalk(whitelist, schedule_uids)

    image_columns = ["domain", "uid", "capture_event_id"] + [f"std__whole__{name}" for name in BASE]
    images = pd.read_parquet(paths["q2_image_descriptors"], columns=image_columns, engine="pyarrow")
    images = images.loc[images["domain"] == "TARGET_ADAPT"].copy()
    images = images.sort_values("uid", kind="mergesort").reset_index(drop=True)
    if len(images) != EXPECTED_TARGET_IMAGES or images["uid"].nunique() != len(images):
        raise PolicyStop("STOP_TARGET_DESCRIPTOR_COUNT_OR_UID_MISMATCH")
    if set(images["uid"].astype(str)) != set(whitelist["uid"].astype(str)):
        raise PolicyStop("STOP_TARGET_DESCRIPTOR_WHITELIST_UID_MISMATCH")
    whitelist_events = whitelist.set_index("uid")["capture_event_id"].astype(str).sort_index()
    descriptor_events = images.set_index("uid")["capture_event_id"].astype(str).sort_index()
    if not whitelist_events.equals(descriptor_events):
        raise PolicyStop("STOP_TARGET_DESCRIPTOR_EVENT_MAPPING_MISMATCH")

    event_columns = ["domain", "capture_event_id"] + [f"std__{name}" for name in BASE]
    events = pd.read_csv(paths["q2_event_prototypes"], usecols=event_columns, dtype={"domain": str, "capture_event_id": str})
    events = events.loc[events["domain"] == "TARGET_ADAPT"].copy()
    events = events.sort_values("capture_event_id", kind="mergesort").reset_index(drop=True)
    if len(events) != EXPECTED_TARGET_EVENTS or events["capture_event_id"].nunique() != len(events):
        raise PolicyStop("STOP_TARGET_EVENT_PROTOTYPE_COUNT_MISMATCH")

    event_ids = events["capture_event_id"].astype(str).to_numpy()
    z_event = events[[f"std__{name}" for name in BASE]].to_numpy(dtype=np.float64)
    event_valid, event_reason = validity_arrays(z_event, median, scale)
    if not event_valid.all():
        bad = [event_reason[index] for index in np.flatnonzero(~event_valid)[:10]]
        raise PolicyStop(f"STOP_TARGET_EVENT_PROTOTYPE_INTERNAL_INVALID: {bad}")
    event_index = {event_id: index for index, event_id in enumerate(event_ids.tolist())}
    z_image = images[[f"std__whole__{name}" for name in BASE]].to_numpy(dtype=np.float64)
    image_uids = images["uid"].astype(str).to_numpy()
    image_event_ids = images["capture_event_id"].astype(str).to_numpy()
    if not set(image_event_ids.tolist()).issubset(event_index):
        raise PolicyStop("STOP_TARGET_IMAGE_EVENT_NOT_IN_PROTOTYPES")

    frozen_tube = pd.read_csv(
        paths["geometry_image_tube"],
        usecols=[
            "uid",
            "capture_event_id",
            "image_in_tube",
            "original_descriptor_internal_valid",
            "original_descriptor_internal_invalid_reasons",
        ],
        dtype={"uid": str, "capture_event_id": str, "original_descriptor_internal_invalid_reasons": str},
        keep_default_na=False,
    ).sort_values("uid", kind="mergesort").reset_index(drop=True)
    if len(frozen_tube) != len(images) or frozen_tube["uid"].tolist() != image_uids.tolist():
        raise PolicyStop("STOP_GEOMETRY_IMAGE_TUBE_UID_ORDER_MISMATCH")
    if frozen_tube["capture_event_id"].astype(str).tolist() != image_event_ids.tolist():
        raise PolicyStop("STOP_GEOMETRY_IMAGE_TUBE_EVENT_MAPPING_MISMATCH")
    original_in_tube = strict_bool_series(frozen_tube["image_in_tube"], "image_in_tube").to_numpy()
    original_internal = strict_bool_series(
        frozen_tube["original_descriptor_internal_valid"], "original_descriptor_internal_valid"
    ).to_numpy()

    pair_columns = ["anchor_event_id", "reference_event_id", "distance", "anchor_degree", "equal_anchor_weight"]
    equal_pairs = pd.read_parquet(paths["geometry_equal_anchor_pairs"], columns=pair_columns, engine="pyarrow")
    directed_pairs = pd.read_parquet(paths["geometry_directed_pairs"], columns=pair_columns, engine="pyarrow")
    if len(equal_pairs) == 0 or len(equal_pairs) != len(directed_pairs):
        raise PolicyStop("STOP_FROZEN_DIRECTED_PAIR_COUNT_MISMATCH")
    for name in ["anchor_event_id", "reference_event_id", "anchor_degree"]:
        if not np.array_equal(equal_pairs[name].to_numpy(), directed_pairs[name].to_numpy()):
            raise PolicyStop(f"STOP_FROZEN_DIRECTED_PAIR_IDENTITY_MISMATCH_{name}")
    for name in ["distance", "equal_anchor_weight"]:
        if not np.array_equal(
            equal_pairs[name].to_numpy(dtype=np.float64), directed_pairs[name].to_numpy(dtype=np.float64)
        ):
            raise PolicyStop(f"STOP_FROZEN_DIRECTED_PAIR_VALUE_MISMATCH_{name}")
    distance = equal_pairs["distance"].to_numpy(dtype=np.float64)
    pair_weight = equal_pairs["equal_anchor_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(distance).all() or not np.isfinite(pair_weight).all() or np.any(pair_weight <= 0.0):
        raise PolicyStop("STOP_FROZEN_EQUAL_ANCHOR_DISTRIBUTION_INVALID")
    if abs(float(pair_weight.sum(dtype=np.float64)) - 1.0) > 1e-10:
        raise PolicyStop("STOP_FROZEN_EQUAL_ANCHOR_WEIGHTS_NOT_NORMALIZED")
    if abs(inverse_ecdf_weighted(distance, pair_weight, 0.10) - q10_tube) > 1e-15:
        raise PolicyStop("STOP_FROZEN_Q10_TUBE_REPLAY_MISMATCH")
    if abs(inverse_ecdf_weighted(distance, pair_weight, 0.90) - q90_tube) > 1e-15:
        raise PolicyStop("STOP_FROZEN_Q90_TUBE_REPLAY_MISMATCH")
    truncated = (distance >= q10_tube) & (distance <= q90_tube)
    pmf_values = distance[truncated]
    pmf_weights = pair_weight[truncated]
    pmf_weights = pmf_weights / pmf_weights.sum(dtype=np.float64)
    order = np.argsort(pmf_values, kind="mergesort")
    pmf_values = pmf_values[order]
    pmf_weights = pmf_weights[order]
    pmf_cdf = np.cumsum(pmf_weights, dtype=np.float64)
    pmf_cdf[-1] = 1.0

    neighbors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for anchor, group in directed_pairs.groupby("anchor_event_id", sort=False):
        ordered = group.sort_values("reference_event_id", kind="mergesort")
        refs = ordered["reference_event_id"].astype(str).to_numpy()
        distances = ordered["distance"].to_numpy(dtype=np.float64)
        if len(refs) != len(set(refs.tolist())) or np.any(distances > 2.0 * tau + 1e-14):
            raise PolicyStop(f"STOP_FROZEN_NEIGHBOR_SET_INVALID: {anchor}")
        neighbors[str(anchor)] = (refs, distances)

    lineage_payload = {
        "phase": PHASE,
        "input_hashes": input_hashes,
        "p_aug": P_AUG,
        "formal_epochs": [FORMAL_EPOCH_START, FORMAL_EPOCH_END],
        "namespaces": [NS_AUGMENT, NS_DOSE, NS_DESTINATION, NS_MATCHING_TIEBREAK],
        "networkx": REQUIRED_NETWORKX_VERSION,
        "matching_execution_version": "EXACT_OPTIMALITY_SAFE_PRESOLVE_V1",
        "matching_presolve_addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
        "matching_presolve_D": PRESOLVE_D,
        "matching_presolve_margin_numerators": list(PRESOLVE_MARGIN_NUMERATORS),
        "matching_presolve_margin_denominator": PRESOLVE_MARGIN_DENOMINATOR,
        "matching_presolve_linprog_method": PRESOLVE_LINPROG_METHOD,
        "matching_presolve_linprog_options": dict(PRESOLVE_LINPROG_OPTIONS),
        "tau": tau,
        "q10_tube": q10_tube,
        "q90_tube": q90_tube,
    }
    lineage_sha = hashlib.sha256(canonical_json_bytes(lineage_payload)).hexdigest()
    return PolicyContext(
        median=median,
        scale=scale,
        tau=tau,
        q10_tube=q10_tube,
        q90_tube=q90_tube,
        pmf_values=pmf_values,
        pmf_weights=pmf_weights,
        pmf_cdf=pmf_cdf,
        event_ids=event_ids,
        event_index=event_index,
        z_event=z_event,
        event_tree=cKDTree(z_event * WEIGHTS),
        neighbors=neighbors,
        image_uids=image_uids,
        image_index={uid: index for index, uid in enumerate(image_uids.tolist())},
        image_event_ids=image_event_ids,
        z_image=z_image,
        original_in_tube=original_in_tube,
        original_internal_valid=original_internal,
        original_invalid_reason=frozen_tube["original_descriptor_internal_invalid_reasons"].astype(str).to_numpy(),
        schedule_to_q2_uid=crosswalk,
        input_hashes=input_hashes,
        lineage_sha256=lineage_sha,
    )


def _sample_dose(context: PolicyContext, u01: float) -> float:
    index = min(int(np.searchsorted(context.pmf_cdf, u01, side="right")), len(context.pmf_values) - 1)
    return float(context.pmf_values[index])


def _resume_epoch_audit(audit_path: Path, context: PolicyContext, adaptation_seed: int) -> dict[str, Any] | None:
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE":
        raise PolicyStop(f"STOP_RESUME_EPOCH_AUDIT_NOT_COMPLETE: {audit_path}")
    if audit.get("policy_lineage_sha256") != context.lineage_sha256:
        raise PolicyStop(f"STOP_RESUME_EPOCH_LINEAGE_MISMATCH: {audit_path}")
    if int(audit.get("adaptation_seed", -1)) != adaptation_seed:
        raise PolicyStop(f"STOP_RESUME_EPOCH_SEED_MISMATCH: {audit_path}")
    for record in audit.get("output_files", []):
        path = Path(record["absolute_path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise PolicyStop(f"STOP_RESUME_EPOCH_OUTPUT_HASH_MISMATCH: {path}")
    return audit


def instantiate_epoch_policy(
    epoch: int,
    schedule: pd.DataFrame,
    context: PolicyContext,
    adaptation_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    partition = f"epoch_{epoch:03d}"
    audit_path = output_dir / "raw/visit_delta_multiset_audit" / f"{partition}.json"
    resumed = _resume_epoch_audit(audit_path, context, adaptation_seed)
    if resumed is not None:
        return resumed

    n_visits = len(schedule)
    if n_visits != EXPECTED_VISITS_PER_EPOCH:
        raise PolicyStop(f"STOP_EPOCH_VISIT_COUNT_MISMATCH_E{epoch:03d}")
    frame = schedule.copy().reset_index(drop=True)
    frame["target_uid"] = frame["schedule_uid"].map(context.schedule_to_q2_uid)
    if frame["target_uid"].isna().any():
        raise PolicyStop(f"STOP_SCHEDULE_CROSSWALK_MISSING_E{epoch:03d}")
    frame["adaptation_seed"] = adaptation_seed
    frame["visit_key"] = [
        visit_key(adaptation_seed, epoch, int(ordinal), uid)
        for ordinal, uid in zip(frame["target_visit_ordinal"], frame["target_uid"])
    ]
    if frame["visit_key"].duplicated().any():
        raise PolicyStop(f"STOP_VISIT_KEY_DUPLICATE_E{epoch:03d}")

    image_indices = np.asarray([context.image_index[str(uid)] for uid in frame["target_uid"]], dtype=np.int64)
    event_ids = context.image_event_ids[image_indices]
    z_image = context.z_image[image_indices]
    original_in_tube = context.original_in_tube[image_indices]
    original_internal = context.original_internal_valid[image_indices]
    original_reason = context.original_invalid_reason[image_indices]

    augment_u = np.asarray(
        [
            stateless_u01(
                NS_AUGMENT,
                adaptation_seed,
                epoch,
                int(ordinal),
                str(uid),
                "AUGMENT_ON",
            )
            for ordinal, uid in zip(frame["target_visit_ordinal"], frame["target_uid"])
        ],
        dtype=np.float64,
    )
    augment_on = augment_u < P_AUG
    dose_u = np.full(n_visits, np.nan, dtype=np.float64)
    intended_dose = np.full(n_visits, np.nan, dtype=np.float64)
    destination_count = np.zeros(n_visits, dtype=np.int64)
    destination_pick = np.full(n_visits, -1, dtype=np.int64)
    destination_event = np.full(n_visits, "", dtype=object)
    destination_distance = np.full(n_visits, np.nan, dtype=np.float64)
    interpolation_lambda = np.full(n_visits, np.nan, dtype=np.float64)
    delta = np.full((n_visits, 7), np.nan, dtype=np.float64)
    native_target = np.full((n_visits, 7), np.nan, dtype=np.float64)
    event_target = np.full((n_visits, 7), np.nan, dtype=np.float64)
    native_dev = np.full(n_visits, np.nan, dtype=np.float64)
    event_target_dev = np.full(n_visits, np.nan, dtype=np.float64)
    native_valid = np.zeros(n_visits, dtype=bool)
    failure = np.full(n_visits, "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH", dtype=object)

    for row in np.flatnonzero(augment_on):
        uid = str(frame.at[row, "target_uid"])
        ordinal = int(frame.at[row, "target_visit_ordinal"])
        failure[row] = ""
        if not original_in_tube[row]:
            failure[row] = "UNAVAILABLE_ORIGINAL_IMAGE_OFF_TUBE"
            continue
        if not original_internal[row]:
            suffix = str(original_reason[row])
            failure[row] = "ORIGINAL_DESCRIPTOR_INTERNAL_INVALID" + (":" + suffix if suffix else "")
            continue
        u = stateless_u01(
            NS_DOSE,
            adaptation_seed,
            epoch,
            ordinal,
            uid,
            "INTENDED_GEOMETRIC_DOSE",
        )
        dose_u[row] = u
        dstar = _sample_dose(context, u)
        intended_dose[row] = dstar
        anchor_event = str(event_ids[row])
        refs, distances = context.neighbors.get(
            anchor_event, (np.asarray([], dtype=object), np.asarray([], dtype=np.float64))
        )
        legal = np.flatnonzero(distances >= dstar)
        destination_count[row] = len(legal)
        if len(legal) == 0:
            failure[row] = "NATIVE_TSVA_UNAVAILABLE_NO_DESTINATION_AT_DOSE"
            continue
        pick = stateless_index(
            len(legal),
            NS_DESTINATION,
            adaptation_seed,
            epoch,
            ordinal,
            uid,
            "DESTINATION",
        )
        selected = int(legal[pick])
        destination_pick[row] = pick
        destination_event[row] = str(refs[selected])
        distance = float(distances[selected])
        destination_distance[row] = distance
        if not (dstar > 0.0 and dstar <= distance <= 2.0 * context.tau + 1e-14):
            failure[row] = "NATIVE_TSVA_DESTINATION_DISTANCE_INVALID"
            continue
        lam = dstar / distance
        interpolation_lambda[row] = lam
        if not (0.0 < lam <= 1.0):
            failure[row] = "NATIVE_TSVA_UNAVAILABLE_LAMBDA_OUT_OF_RANGE"
            continue
        anchor_index = context.event_index[anchor_event]
        destination_index = context.event_index[str(refs[selected])]
        event_target[row] = context.z_event[anchor_index] + lam * (
            context.z_event[destination_index] - context.z_event[anchor_index]
        )
        delta[row] = event_target[row] - context.z_event[anchor_index]
        measured_dose = float(np.sum(np.abs(delta[row]) * WEIGHTS, dtype=np.float64))
        tolerance = max(1e-12, 1e-10 * max(1.0, abs(dstar)))
        if abs(measured_dose - dstar) > tolerance:
            failure[row] = "NATIVE_DOSE_NUMERIC_MISMATCH"
            continue
        native_target[row] = z_image[row] + delta[row]

    generated = np.flatnonzero(np.isfinite(delta).all(axis=1))
    if len(generated):
        event_valid, event_reasons = validity_arrays(event_target[generated], context.median, context.scale)
        target_valid, target_reasons = validity_arrays(native_target[generated], context.median, context.scale)
        queried_event, _ = context.event_tree.query(
            event_target[generated] * WEIGHTS, k=1, p=1, eps=0, workers=1
        )
        queried_native, _ = context.event_tree.query(
            native_target[generated] * WEIGHTS, k=1, p=1, eps=0, workers=1
        )
        event_target_dev[generated] = queried_event
        native_dev[generated] = queried_native
        for local, row in enumerate(generated.tolist()):
            if failure[row]:
                continue
            if not event_valid[local]:
                failure[row] = "DESCRIPTOR_INTERNAL_INVALID_EVENT_TARGET:" + event_reasons[local]
            elif event_target_dev[row] > context.tau:
                failure[row] = "EVENT_TARGET_OFF_EMPIRICAL_TUBE"
            elif not target_valid[local]:
                failure[row] = "DESCRIPTOR_INTERNAL_INVALID_NATIVE:" + target_reasons[local]
            elif native_dev[row] > context.tau:
                failure[row] = "NATIVE_ENDPOINT_OFF_EMPIRICAL_TUBE"
            else:
                native_valid[row] = True

    delta_sha = np.full(n_visits, "", dtype=object)
    for row in np.flatnonzero(np.isfinite(delta).all(axis=1)):
        delta_sha[row] = delta_hash(delta[row])

    frame["capture_event_id"] = event_ids
    frame["augment_hash_u01"] = augment_u
    frame["augment_on_requested"] = augment_on
    frame["original_image_in_tube"] = original_in_tube
    frame["original_descriptor_internal_valid"] = original_internal
    frame["dose_hash_u01"] = dose_u
    frame["intended_geometric_dose"] = intended_dose
    frame["destination_candidate_count"] = destination_count
    frame["destination_hash_index"] = destination_pick
    frame["destination_event_id"] = destination_event
    frame["anchor_destination_distance"] = destination_distance
    frame["lambda"] = interpolation_lambda
    frame["event_target_tube_deviation"] = event_target_dev
    frame["native_tube_deviation"] = native_dev
    frame["native_augmented_valid"] = native_valid
    frame["native_identity"] = ~native_valid
    frame["native_failure_reason"] = failure
    frame["native_delta_sha256"] = delta_sha
    for coordinate, name in enumerate(BASE):
        frame[f"original_std__{name}"] = z_image[:, coordinate]
        frame[f"native_delta__{name}"] = delta[:, coordinate]
        frame[f"native_target_std__{name}"] = native_target[:, coordinate]

    node_rows = np.flatnonzero(native_valid)
    node_keys = frame.loc[node_rows, "visit_key"].astype(str).tolist()
    node_events = event_ids[node_rows]
    node_x = z_image[node_rows]
    node_delta = delta[node_rows]
    node_dev_native = native_dev[node_rows]
    n_nodes = len(node_rows)
    cross_dev = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    cross_ok = np.zeros((n_nodes, n_nodes), dtype=bool)
    donor_block = 32
    for start in range(0, n_nodes, donor_block):
        end = min(start + donor_block, n_nodes)
        targets = node_x[:, None, :] + node_delta[None, start:end, :]
        flat = targets.reshape(-1, 7)
        valid_flat, _ = validity_arrays(flat, context.median, context.scale)
        deviation_flat = np.full(len(flat), np.inf, dtype=np.float64)
        if valid_flat.any():
            queried, _ = context.event_tree.query(
                flat[valid_flat] * WEIGHTS, k=1, p=1, eps=0, workers=1
            )
            deviation_flat[valid_flat] = queried
        deviation = deviation_flat.reshape(n_nodes, end - start)
        compatible = valid_flat.reshape(n_nodes, end - start) & (deviation <= context.tau)
        cross_dev[:, start:end] = deviation
        cross_ok[:, start:end] = compatible
    event_code, _ = pd.factorize(node_events, sort=True)
    mutual = cross_ok & cross_ok.T & (event_code[:, None] != event_code[None, :])
    np.fill_diagonal(mutual, False)
    edge_i, edge_j = np.nonzero(np.triu(mutual, k=1))
    dev_x_y = cross_dev[edge_i, edge_j]
    dev_y_x = cross_dev[edge_j, edge_i]
    fairness_cost = np.abs(dev_x_y - node_dev_native[edge_i]) + np.abs(
        dev_y_x - node_dev_native[edge_j]
    )
    del mutual, cross_ok
    solution = solve_general_graph_matching(node_keys, edge_i, edge_j, fairness_cost)

    partner = np.full(n_nodes, -1, dtype=np.int64)
    for left, right in solution.pairs:
        if left == right or partner[left] >= 0 or partner[right] >= 0:
            raise PolicyStop(f"STOP_VISIT_MATCHING_NOT_DERANGEMENT_E{epoch:03d}")
        partner[left] = right
        partner[right] = left
    matched_nodes = np.flatnonzero(partner >= 0)
    matched_rows = node_rows[matched_nodes]
    matched_mask = np.zeros(n_visits, dtype=bool)
    matched_mask[matched_rows] = True
    frame["matched_pair_id"] = -1
    frame["matched_partner_visit_key"] = ""
    frame["smg_donor_uid"] = ""
    frame["smg_donor_event_id"] = ""
    frame["smg_delta_sha256"] = ""
    frame["smg_target_tube_deviation"] = np.nan
    frame["matched_a_augmented_valid"] = matched_mask
    frame["smg_augmented_valid"] = matched_mask
    frame["matched_a_identity"] = ~matched_mask
    frame["smg_identity"] = ~matched_mask
    for name in BASE:
        frame[f"smg_delta__{name}"] = np.nan
        frame[f"smg_target_std__{name}"] = np.nan

    selected_pair_edge_indices = np.asarray(
        solution.presolve_arrays["selected_pair_full_edge_indices"], dtype=np.int64
    )
    if len(selected_pair_edge_indices) != len(solution.pairs):
        raise PolicyStop(f"STOP_MATCHING_SELECTED_EDGE_INDEX_COUNT_MISMATCH_E{epoch:03d}")
    pair_rows: list[dict[str, Any]] = []
    for pair_id, ((left, right), raw_edge_index) in enumerate(
        zip(solution.pairs, selected_pair_edge_indices)
    ):
        edge_index = int(raw_edge_index)
        if int(edge_i[edge_index]) != left or int(edge_j[edge_index]) != right:
            raise PolicyStop(f"STOP_MATCHING_SELECTED_EDGE_INDEX_IDENTITY_MISMATCH_E{epoch:03d}")
        row_left = int(node_rows[left])
        row_right = int(node_rows[right])
        for receiver, donor, receiver_row in [(left, right, row_left), (right, left, row_right)]:
            donor_row = int(node_rows[donor])
            frame.at[receiver_row, "matched_pair_id"] = pair_id
            frame.at[receiver_row, "matched_partner_visit_key"] = node_keys[donor]
            frame.at[receiver_row, "smg_donor_uid"] = str(frame.at[donor_row, "target_uid"])
            frame.at[receiver_row, "smg_donor_event_id"] = str(node_events[donor])
            frame.at[receiver_row, "smg_delta_sha256"] = str(delta_sha[donor_row])
            frame.at[receiver_row, "smg_target_tube_deviation"] = float(cross_dev[receiver, donor])
            for coordinate, name in enumerate(BASE):
                frame.at[receiver_row, f"smg_delta__{name}"] = float(node_delta[donor, coordinate])
                frame.at[receiver_row, f"smg_target_std__{name}"] = float(
                    node_x[receiver, coordinate] + node_delta[donor, coordinate]
                )
        pair_rows.append(
            {
                "epoch": epoch,
                "pair_id": pair_id,
                "visit_key_x": node_keys[left],
                "visit_key_y": node_keys[right],
                "target_visit_ordinal_x": int(frame.at[row_left, "target_visit_ordinal"]),
                "target_visit_ordinal_y": int(frame.at[row_right, "target_visit_ordinal"]),
                "uid_x": str(frame.at[row_left, "target_uid"]),
                "uid_y": str(frame.at[row_right, "target_uid"]),
                "event_x": str(node_events[left]),
                "event_y": str(node_events[right]),
                "delta_x_sha256": str(delta_sha[row_left]),
                "delta_y_sha256": str(delta_sha[row_right]),
                "matched_a_x_uses_delta_sha256": str(delta_sha[row_left]),
                "matched_a_y_uses_delta_sha256": str(delta_sha[row_right]),
                "smg_x_uses_delta_sha256": str(delta_sha[row_right]),
                "smg_y_uses_delta_sha256": str(delta_sha[row_left]),
                "dose_x": float(intended_dose[row_left]),
                "dose_y": float(intended_dose[row_right]),
                "dev_native_x": float(node_dev_native[left]),
                "dev_native_y": float(node_dev_native[right]),
                "dev_smg_x": float(cross_dev[left, right]),
                "dev_smg_y": float(cross_dev[right, left]),
                "fairness_cost": float(fairness_cost[edge_index]),
                "cost_int_1e9": int(solution.cost_int[edge_index]),
                "tiebreak_sha256": solution.tie_digest.view(np.uint8).reshape(-1, 32)[edge_index].tobytes().hex(),
                "tiebreak_rank": int(solution.tie_rank[edge_index]),
                "cost_priority": int(solution.cost_priority),
                "solver_weight_decimal": str(solution.solver_weights[edge_index]),
            }
        )

    matched_a_multiset = sorted(frame.loc[matched_mask, "native_delta_sha256"].astype(str).tolist())
    smg_multiset = sorted(frame.loc[matched_mask, "smg_delta_sha256"].astype(str).tolist())
    exact_multiset = matched_a_multiset == smg_multiset
    if not exact_multiset:
        raise PolicyStop(f"STOP_VISIT_LEVEL_CAUSAL_MATCH_INVALID_E{epoch:03d}")
    if not np.array_equal(
        frame["matched_a_augmented_valid"].to_numpy(dtype=bool),
        frame["smg_augmented_valid"].to_numpy(dtype=bool),
    ):
        raise PolicyStop(f"STOP_VISIT_LEVEL_BRANCH_COVERAGE_MISMATCH_E{epoch:03d}")

    native_delta_columns = [
        "epoch",
        "target_visit_ordinal",
        "visit_key",
        "schedule_uid",
        "target_uid",
        "capture_event_id",
        "intended_geometric_dose",
        "destination_event_id",
        "anchor_destination_distance",
        "lambda",
        "native_tube_deviation",
        "native_delta_sha256",
    ] + [f"native_delta__{name}" for name in BASE] + [f"native_target_std__{name}" for name in BASE]
    native_delta_frame = frame.loc[native_valid, native_delta_columns].copy()
    pairs_frame = pd.DataFrame(pair_rows)

    visit_key_to_row = {
        str(key): int(index) for index, key in enumerate(frame["visit_key"].astype(str).tolist())
    }
    worker_rows: list[dict[str, Any]] = []
    for row in range(n_visits):
        base_record = {
            "adaptation_seed": adaptation_seed,
            "epoch": epoch,
            "target_visit_ordinal": int(frame.at[row, "target_visit_ordinal"]),
            "schedule_uid": str(frame.at[row, "schedule_uid"]),
            "q2_uid": str(frame.at[row, "target_uid"]),
        }
        policy_requested = bool(augment_on[row])
        native_ok = bool(native_valid[row])
        matched_ok = bool(matched_mask[row])
        native_reason = str(failure[row])
        for branch in ["NATIVE_A", "MATCHED_A", "SMG"]:
            if branch == "NATIVE_A":
                valid = native_ok
                branch_delta = delta[row] if valid else np.full(7, np.nan, dtype=np.float64)
                branch_sha = str(delta_sha[row]) if valid else ""
                donor_schedule_uid = str(frame.at[row, "schedule_uid"]) if valid else ""
                donor_q2_uid = str(frame.at[row, "target_uid"]) if valid else ""
                reason = "" if valid else native_reason
            elif branch == "MATCHED_A":
                valid = matched_ok
                branch_delta = delta[row] if valid else np.full(7, np.nan, dtype=np.float64)
                branch_sha = str(delta_sha[row]) if valid else ""
                donor_schedule_uid = str(frame.at[row, "schedule_uid"]) if valid else ""
                donor_q2_uid = str(frame.at[row, "target_uid"]) if valid else ""
                if valid:
                    reason = ""
                elif not policy_requested:
                    reason = "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH"
                elif not native_ok:
                    reason = native_reason
                else:
                    reason = "MATCHED_CONTROL_UNAVAILABLE_NO_2CYCLE"
            else:
                valid = matched_ok
                branch_delta = (
                    frame.loc[row, [f"smg_delta__{name}" for name in BASE]].to_numpy(dtype=np.float64)
                    if valid
                    else np.full(7, np.nan, dtype=np.float64)
                )
                branch_sha = str(frame.at[row, "smg_delta_sha256"]) if valid else ""
                if valid:
                    partner_key = str(frame.at[row, "matched_partner_visit_key"])
                    partner_row = visit_key_to_row[partner_key]
                    donor_schedule_uid = str(frame.at[partner_row, "schedule_uid"])
                    donor_q2_uid = str(frame.at[partner_row, "target_uid"])
                    reason = ""
                elif not policy_requested:
                    donor_schedule_uid = ""
                    donor_q2_uid = ""
                    reason = "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH"
                elif not native_ok:
                    donor_schedule_uid = ""
                    donor_q2_uid = ""
                    reason = native_reason
                else:
                    donor_schedule_uid = ""
                    donor_q2_uid = ""
                    reason = "MATCHED_CONTROL_UNAVAILABLE_NO_2CYCLE"
            if valid:
                policy_flag = "APPLY_OFFLINE_THETA"
            elif not policy_requested:
                policy_flag = "IDENTITY_POLICY_OFF"
            else:
                policy_flag = "IDENTITY_FALLBACK"
            record = {
                **base_record,
                "branch": branch,
                # This is the common Bernoulli request.  The worker applies a
                # view only when augment_on AND valid AND policy_flag=APPLY.
                "augment_on": policy_requested,
                "valid": valid,
                "policy_flag": policy_flag,
                "failure_reason": reason,
                "delta_sha256": branch_sha,
                "donor_schedule_uid": donor_schedule_uid,
                "donor_q2_uid": donor_q2_uid,
            }
            for coordinate in range(7):
                record[f"delta_{coordinate}"] = float(branch_delta[coordinate])
            worker_rows.append(record)
    worker_schedule = pd.DataFrame(worker_rows, columns=WORKER_SCHEDULE_COLUMNS)
    expected_worker_rows = n_visits * 3
    if len(worker_schedule) != expected_worker_rows:
        raise PolicyStop(f"STOP_WORKER_SCHEDULE_ROW_COUNT_MISMATCH_E{epoch:03d}")
    matched_worker = worker_schedule.loc[worker_schedule["branch"] == "MATCHED_A"]
    smg_worker = worker_schedule.loc[worker_schedule["branch"] == "SMG"]
    if not np.array_equal(
        matched_worker[["augment_on", "valid"]].to_numpy(dtype=bool),
        smg_worker[["augment_on", "valid"]].to_numpy(dtype=bool),
    ):
        raise PolicyStop(f"STOP_WORKER_MATCHED_BRANCH_FLAG_MISMATCH_E{epoch:03d}")

    schedule_path = output_dir / "raw/visit_policy_schedule" / f"{partition}.parquet"
    internal_path = output_dir / "raw/visit_policy_internal" / f"{partition}.parquet"
    delta_path = output_dir / "raw/visit_native_deltas" / f"{partition}.parquet"
    pairs_path = output_dir / "raw/visit_matched_pairs" / f"{partition}.parquet"
    presolve_arrays_path = (
        output_dir / "raw/visit_matching_presolve" / f"{partition}_arrays.npz"
    )
    presolve_certificate_path = (
        output_dir / "raw/visit_matching_presolve" / f"{partition}.json"
    )
    presolve_arrays_sha256 = write_npz_immutable(
        presolve_arrays_path, solution.presolve_arrays
    )
    presolve_certificate = dict(solution.presolve_certificate)
    presolve_certificate.update(
        {
            "experiment": EXPERIMENT,
            "phase": PHASE,
            "epoch": epoch,
            "adaptation_seed": adaptation_seed,
            "policy_lineage_sha256": context.lineage_sha256,
            "arrays_artifact": {
                "absolute_path": str(presolve_arrays_path.resolve()),
                "sha256": presolve_arrays_sha256,
                "members": {
                    name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                    for name, value in sorted(solution.presolve_arrays.items())
                },
            },
        }
    )
    presolve_certificate_sha256 = write_json_immutable(
        presolve_certificate_path, presolve_certificate
    )
    output_records = [
        {
            "role": "visit_matching_presolve_certificate",
            "absolute_path": str(presolve_certificate_path.resolve()),
            "sha256": presolve_certificate_sha256,
            "rows": 1,
        },
        {
            "role": "visit_matching_presolve_arrays",
            "absolute_path": str(presolve_arrays_path.resolve()),
            "sha256": presolve_arrays_sha256,
            "rows": int(solution.presolve_certificate["full_edge_count"]),
        },
        {
            "role": "visit_policy_schedule",
            "absolute_path": str(schedule_path.resolve()),
            "sha256": write_parquet_immutable(schedule_path, worker_schedule),
            "rows": len(worker_schedule),
        },
        {
            "role": "visit_policy_internal",
            "absolute_path": str(internal_path.resolve()),
            "sha256": write_parquet_immutable(internal_path, frame),
            "rows": len(frame),
        },
        {
            "role": "visit_native_deltas",
            "absolute_path": str(delta_path.resolve()),
            "sha256": write_parquet_immutable(delta_path, native_delta_frame),
            "rows": len(native_delta_frame),
        },
        {
            "role": "visit_matched_pairs",
            "absolute_path": str(pairs_path.resolve()),
            "sha256": write_parquet_immutable(pairs_path, pairs_frame),
            "rows": len(pairs_frame),
        },
    ]
    audit = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "status": "COMPLETE",
        "epoch": epoch,
        "adaptation_seed": adaptation_seed,
        "policy_lineage_sha256": context.lineage_sha256,
        "formal_visit_key": ["adaptation_seed", "epoch", "target_visit_ordinal", "target_uid"],
        "scheduled_visit_count": n_visits,
        "augment_on_visit_count": int(augment_on.sum()),
        "native_augmented_valid_count": int(native_valid.sum()),
        "native_identity_count": int((~native_valid).sum()),
        "native_invalid_identity_fallback_count": int((augment_on & ~native_valid).sum()),
        "matching_node_count": n_nodes,
        "mutual_compatible_edge_count": int(len(edge_i)),
        "matched_pair_count": int(len(solution.pairs)),
        "matched_visit_count": int(matched_mask.sum()),
        "matched_a_augmentation_on_visit_count": int(augment_on.sum()),
        "smg_augmentation_on_visit_count": int(augment_on.sum()),
        "matched_a_valid_visit_count": int(matched_mask.sum()),
        "smg_valid_visit_count": int(matched_mask.sum()),
        "matched_a_identity_count": int((~matched_mask).sum()),
        "smg_identity_count": int((~matched_mask).sum()),
        "matched_a_sorted_delta_multiset_sha256": hashlib.sha256(
            "\n".join(matched_a_multiset).encode("ascii")
        ).hexdigest(),
        "smg_sorted_delta_multiset_sha256": hashlib.sha256(
            "\n".join(smg_multiset).encode("ascii")
        ).hexdigest(),
        "VISIT_LEVEL_EXACT_DELTA_MATCH": exact_multiset,
        "matching_solver": "networkx.algorithms.matching.max_weight_matching",
        "matching_solver_version": nx.__version__,
        "matching_greedy_fallback": False,
        "matching_cost": "abs(dev_smg_x-dev_native_x)+abs(dev_smg_y-dev_native_y)",
        "matching_cost_integerization": "int(numpy.rint(cost*1e9)); IEEE-754 ties-to-even",
        "matching_tiebreak_namespace": NS_MATCHING_TIEBREAK,
        "matching_cost_priority": int(solution.cost_priority),
        "matching_canonical_input_sha256": solution.canonical_input_sha256,
        "matching_execution_presolve": {
            "name": "EXACT_OPTIMALITY_SAFE_PRESOLVE_V1",
            "engineering_addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
            "certificate_path": str(presolve_certificate_path.resolve()),
            "certificate_sha256": presolve_certificate_sha256,
            "arrays_path": str(presolve_arrays_path.resolve()),
            "arrays_sha256": presolve_arrays_sha256,
            "certified_candidate_accepted": bool(
                presolve_certificate["certified_candidate_accepted"]
            ),
            "fallback_full_graph": bool(presolve_certificate["fallback_full_graph"]),
            "fallback_reason": str(presolve_certificate["fallback_reason"]),
            "full_edge_count": int(presolve_certificate["full_edge_count"]),
            "final_retained_edge_count": int(
                presolve_certificate["final_retained_edge_count"]
            ),
            "final_pruned_edge_count": int(
                presolve_certificate["final_pruned_edge_count"]
            ),
            "selected_matching_cardinality": int(
                presolve_certificate["selected_matching_cardinality"]
            ),
            "selected_primary_cost_int_sum": int(
                presolve_certificate["selected_primary_cost_int_sum"]
            ),
            "selected_combined_penalty_sum_decimal": str(
                presolve_certificate["selected_combined_penalty_sum_decimal"]
            ),
            "exact_combined_sum_from_full_frozen_weights": bool(
                presolve_certificate["exact_combined_sum_from_full_frozen_weights"]
            ),
        },
        "failure_reason_counts": dict(sorted(Counter(failure.tolist()).items())),
        "output_files": output_records,
        "hard_zeros": {
            "target_adapt_label_reads": 0,
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "image_decode_calls": 0,
            "detector_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "ema_updates": 0,
            "photometric_solver_calls": 0,
        },
    }
    write_json_immutable(audit_path, audit)
    return audit


def _metric_row(
    scope: str, epoch: int | str, metric: str, value: Any, unit: str, notes: str
) -> dict[str, Any]:
    return {"scope": scope, "epoch": epoch, "metric": metric, "value": value, "unit": unit, "notes": notes}


def write_policy_aggregate(output_dir: Path, adaptation_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    delta_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    for epoch in range(FORMAL_EPOCH_START, FORMAL_EPOCH_END + 1):
        audit_path = output_dir / "raw/visit_delta_multiset_audit" / f"epoch_{epoch:03d}.json"
        schedule_path = output_dir / "raw/visit_policy_schedule" / f"epoch_{epoch:03d}.parquet"
        if not audit_path.exists():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if int(audit.get("adaptation_seed", -1)) != adaptation_seed:
            raise PolicyStop(f"STOP_AGGREGATE_EPOCH_SEED_MISMATCH_E{epoch:03d}")
        if audit.get("VISIT_LEVEL_EXACT_DELTA_MATCH") is not True:
            raise PolicyStop(f"STOP_VISIT_LEVEL_CAUSAL_MATCH_INVALID_E{epoch:03d}")
        audits.append(audit)
        frames.append(pd.read_parquet(schedule_path, engine="pyarrow"))
        delta_frames.append(
            pd.read_parquet(
                output_dir / "raw/visit_native_deltas" / f"epoch_{epoch:03d}.parquet",
                engine="pyarrow",
            )
        )
        pair_frames.append(
            pd.read_parquet(
                output_dir / "raw/visit_matched_pairs" / f"epoch_{epoch:03d}.parquet",
                engine="pyarrow",
            )
        )
    if not audits:
        raise PolicyStop("STOP_NO_COMPLETED_VISIT_POLICY_EPOCHS")

    summary_rows: list[dict[str, Any]] = []
    for audit in audits:
        epoch = int(audit["epoch"])
        total = int(audit["scheduled_visit_count"])
        native = int(audit["native_augmented_valid_count"])
        matched = int(audit["matched_visit_count"])
        values = [
            ("scheduled_visit_count", total, "visits", "frozen Target schedule"),
            ("nominal_augment_on_count", int(audit["augment_on_visit_count"]), "visits", "stateless Bernoulli(0.5)"),
            ("nominal_augment_on_rate", int(audit["augment_on_visit_count"]) / total, "fraction", "all scheduled visits"),
            ("native_augmented_valid_count", native, "visits", "Native-A apply count"),
            ("native_augmented_valid_rate", native / total, "fraction", "all scheduled visits"),
            ("matched_augmented_valid_count", matched, "visits", "common Matched-A/SMG apply count"),
            ("matched_augmented_valid_rate", matched / total, "fraction", "all scheduled visits"),
            ("native_minus_matched_coverage_gap", (native - matched) / total, "fraction", "all scheduled visits"),
            ("native_invalid_identity_fallback_count", int(audit["native_invalid_identity_fallback_count"]), "visits", "augment requested but descriptor-invalid"),
            ("matched_a_identity_count", int(audit["matched_a_identity_count"]), "visits", "policy-off, invalid, or unmatched"),
            ("smg_identity_count", int(audit["smg_identity_count"]), "visits", "exactly matched to Matched-A"),
            ("mutual_compatible_edge_count", int(audit["mutual_compatible_edge_count"]), "edges", "epoch-specific mutual graph"),
            ("matched_pair_count", int(audit["matched_pair_count"]), "pairs", "maximum-cardinality 2-cycles"),
        ]
        summary_rows.extend(_metric_row("EPOCH", epoch, *value) for value in values)

    combined = pd.concat(frames, ignore_index=True)
    native_rows = combined.loc[combined["branch"] == "NATIVE_A"].copy()
    matched_rows = combined.loc[combined["branch"] == "MATCHED_A"].copy()
    smg_rows = combined.loc[combined["branch"] == "SMG"].copy()
    if not (len(native_rows) == len(matched_rows) == len(smg_rows)):
        raise PolicyStop("STOP_WORKER_BRANCH_SCHEDULE_COUNT_MISMATCH")
    total = len(native_rows)
    native_count = int(strict_bool_series(native_rows["valid"], "NATIVE_A.valid").sum())
    matched_count = int(strict_bool_series(matched_rows["valid"], "MATCHED_A.valid").sum())
    smg_count = int(strict_bool_series(smg_rows["valid"], "SMG.valid").sum())
    if matched_count != smg_count:
        raise PolicyStop("STOP_WORKER_MATCHED_BRANCH_COVERAGE_MISMATCH")
    augment_count = int(strict_bool_series(native_rows["augment_on"], "NATIVE_A.augment_on").sum())
    full_values = [
        ("completed_epoch_count", len(audits), "epochs", "available immutable partitions"),
        ("scheduled_visit_count", total, "visits", "completed epochs"),
        ("nominal_augment_on_count", augment_count, "visits", "stateless Bernoulli(0.5)"),
        ("nominal_augment_on_rate", augment_count / total, "fraction", "completed scheduled visits"),
        ("native_augmented_valid_count", native_count, "visits", "Native-A apply count"),
        ("native_augmented_valid_rate", native_count / total, "fraction", "completed scheduled visits"),
        ("matched_augmented_valid_count", matched_count, "visits", "common Matched-A/SMG apply count"),
        ("matched_augmented_valid_rate", matched_count / total, "fraction", "completed scheduled visits"),
        ("native_minus_matched_coverage_gap", (native_count - matched_count) / total, "fraction", "completed scheduled visits"),
    ]
    summary_rows.extend(_metric_row("FULL_HORIZON", "ALL", *value) for value in full_values)
    summary = pd.DataFrame(summary_rows)
    write_csv_replaceable(output_dir / "VISIT_LEVEL_POLICY_SUMMARY.csv", summary)

    matched_multiset: list[str] = []
    smg_multiset: list[str] = []
    for frame in frames:
        matched_branch = frame.loc[frame["branch"] == "MATCHED_A"].copy()
        smg_branch = frame.loc[frame["branch"] == "SMG"].copy()
        selected_matched = strict_bool_series(matched_branch["valid"], "MATCHED_A.valid")
        selected_smg = strict_bool_series(smg_branch["valid"], "SMG.valid")
        matched_multiset.extend(matched_branch.loc[selected_matched, "delta_sha256"].astype(str).tolist())
        smg_multiset.extend(smg_branch.loc[selected_smg, "delta_sha256"].astype(str).tolist())
    matched_multiset.sort()
    smg_multiset.sort()
    exact = matched_multiset == smg_multiset and all(
        audit["VISIT_LEVEL_EXACT_DELTA_MATCH"] is True for audit in audits
    )
    if not exact:
        raise PolicyStop("STOP_VISIT_LEVEL_CAUSAL_MATCH_INVALID")
    presolve_epochs = [audit.get("matching_execution_presolve") for audit in audits]
    if any(not isinstance(item, dict) for item in presolve_epochs):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_EPOCH_AUDIT_MISSING")
    if any(
        item.get("engineering_addendum_sha256") != PRESOLVE_ADDENDUM_SHA256
        for item in presolve_epochs
    ):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_ADDENDUM_LINEAGE_MISMATCH")
    if any(
        item.get("exact_combined_sum_from_full_frozen_weights") is not True
        for item in presolve_epochs
    ):
        raise PolicyStop("STOP_MATCHING_PRESOLVE_COMBINED_SUM_INVALID")
    retained_counts = [int(item["final_retained_edge_count"]) for item in presolve_epochs]
    full_counts = [int(item["full_edge_count"]) for item in presolve_epochs]
    presolve_summary = {
        "name": "EXACT_OPTIMALITY_SAFE_PRESOLVE_V1",
        "engineering_addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
        "scientific_contract_modified": False,
        "matching_mathematical_objective_modified": False,
        "epoch_count": len(presolve_epochs),
        "certified_candidate_epoch_count": sum(
            item["certified_candidate_accepted"] is True for item in presolve_epochs
        ),
        "full_graph_fallback_epoch_count": sum(
            item["fallback_full_graph"] is True for item in presolve_epochs
        ),
        "full_edge_count_total": sum(full_counts),
        "retained_edge_count_total": sum(retained_counts),
        "retained_edge_count_min": min(retained_counts),
        "retained_edge_count_max": max(retained_counts),
        "retained_edge_count_median_inverse_ecdf": int(
            np.sort(np.asarray(retained_counts, dtype=np.int64), kind="mergesort")[
                max(0, math.ceil(0.50 * len(retained_counts)) - 1)
            ]
        ),
        "exact_combined_sum_all_epochs": True,
        "greedy_fallback": False,
        "per_epoch_certificate_sha256": {
            f"epoch_{int(audit['epoch']):03d}": str(
                audit["matching_execution_presolve"]["certificate_sha256"]
            )
            for audit in audits
        },
    }
    matching_audit = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "status": "COMPLETE" if len(audits) == FORMAL_EPOCH_END - FORMAL_EPOCH_START + 1 else "PARTIAL_RESUMABLE",
        "adaptation_seed": adaptation_seed,
        "formal_epoch_range": [FORMAL_EPOCH_START, FORMAL_EPOCH_END],
        "completed_epochs": [int(audit["epoch"]) for audit in audits],
        "formal_horizon_complete": len(audits) == FORMAL_EPOCH_END - FORMAL_EPOCH_START + 1,
        "matched_a_delta_count": len(matched_multiset),
        "smg_delta_count": len(smg_multiset),
        "matched_a_sorted_delta_multiset_sha256": hashlib.sha256(
            "\n".join(matched_multiset).encode("ascii")
        ).hexdigest(),
        "smg_sorted_delta_multiset_sha256": hashlib.sha256(
            "\n".join(smg_multiset).encode("ascii")
        ).hexdigest(),
        "VISIT_LEVEL_EXACT_DELTA_MATCH": exact,
        "augmentation_on_visit_count_equal": all(
            int(audit["matched_a_augmentation_on_visit_count"])
            == int(audit["smg_augmentation_on_visit_count"])
            for audit in audits
        ),
        "matching_solver": "networkx.algorithms.matching.max_weight_matching",
        "matching_solver_version": nx.__version__,
        "matching_greedy_fallback": False,
        "matching_execution_presolve_summary": presolve_summary,
        "per_epoch_audit_sha256": {
            f"epoch_{int(audit['epoch']):03d}": sha256_file(
                output_dir / "raw/visit_delta_multiset_audit" / f"epoch_{int(audit['epoch']):03d}.json"
            )
            for audit in audits
        },
    }
    write_json_replaceable(output_dir / "VISIT_LEVEL_MATCHING_AUDIT.json", matching_audit)
    if matching_audit["formal_horizon_complete"]:
        write_parquet_replaceable(
            output_dir / "raw/visit_policy_schedule.parquet",
            pd.concat(frames, ignore_index=True),
        )
        write_parquet_replaceable(
            output_dir / "raw/visit_native_deltas.parquet",
            pd.concat(delta_frames, ignore_index=True),
        )
        write_parquet_replaceable(
            output_dir / "raw/visit_matched_pairs.parquet",
            pd.concat(pair_frames, ignore_index=True),
        )
        write_json_replaceable(
            output_dir / "raw/visit_delta_multiset_audit.json",
            matching_audit,
        )
    return summary, matching_audit


CLIPPING_METRICS = [
    "tone_preclip_pixel_fraction",
    "tone_preclip_channel_fraction",
    "lab_to_srgb_preclip_pixel_fraction",
    "lab_to_srgb_preclip_channel_fraction",
    "tone_clip_mean_excess",
    "tone_clip_max_excess",
    "gamut_clip_mean_excess",
    "gamut_clip_max_excess",
]

RISK_METRICS = [
    "realization_residual",
    "normalized_realization_residual",
    "actual_effective_dose",
    "actual_support_deviation",
    "actual_endpoint_in_tube",
    "tone_preclip_pixel_fraction",
    "tone_preclip_channel_fraction",
    "tone_clip_mean_excess",
    "tone_clip_max_excess",
    "lab_to_srgb_preclip_pixel_fraction",
    "lab_to_srgb_preclip_channel_fraction",
    "gamut_clip_mean_excess",
    "gamut_clip_max_excess",
]


def clipping_fairness_closure(matched_native: pd.DataFrame, smg: pd.DataFrame) -> pd.DataFrame:
    if len(matched_native) != len(smg) or set(matched_native["uid"].astype(str)) != set(smg["uid"].astype(str)):
        raise PolicyStop("STOP_REALIZATION_MATCHED_UID_SET_MISMATCH")
    rows: list[dict[str, Any]] = []
    for metric in CLIPPING_METRICS:
        native_values = matched_native[metric].to_numpy(dtype=np.float64)
        smg_values = smg[metric].to_numpy(dtype=np.float64)
        if not np.isfinite(native_values).all() or not np.isfinite(smg_values).all():
            raise PolicyStop(f"STOP_REALIZATION_CLIPPING_NONFINITE_{metric}")
        native_summary = empirical_summary(native_values)
        smg_summary = empirical_summary(smg_values)
        rows.append(
            {
                "metric": metric,
                "matched_native_count": len(native_values),
                "matched_native_Q10": native_summary["Q10"],
                "matched_native_Q50": native_summary["Q50"],
                "matched_native_Q90": native_summary["Q90"],
                "matched_native_mean": native_summary["mean"],
                "matched_smg_count": len(smg_values),
                "matched_smg_Q10": smg_summary["Q10"],
                "matched_smg_Q50": smg_summary["Q50"],
                "matched_smg_Q90": smg_summary["Q90"],
                "matched_smg_mean": smg_summary["mean"],
                "W1_matched_native_vs_smg": wasserstein_1d(native_values, smg_values),
                "scientific_recomputation": False,
                "realization_rerun": False,
            }
        )
    return pd.DataFrame(rows)


def solver_status_risk_stratification(branches: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for branch, frame in branches.items():
        success = strict_bool_series(frame["solver_success"], f"{branch}.solver_success")
        for status_value, status_name in [(True, "SOLVER_SUCCESS"), (False, "SOLVER_NON_SUCCESS")]:
            subset = frame.loc[success == status_value]
            for metric in RISK_METRICS:
                if metric == "actual_endpoint_in_tube":
                    values = strict_bool_series(subset[metric], f"{branch}.{metric}").astype(float).to_numpy()
                    unit = "fraction"
                else:
                    values = subset[metric].to_numpy(dtype=np.float64)
                    unit = "D_style" if metric in {
                        "realization_residual",
                        "actual_effective_dose",
                        "actual_support_deviation",
                    } else "scalar"
                if len(values) and not np.isfinite(values).all():
                    raise PolicyStop(f"STOP_REALIZATION_RISK_NONFINITE_{branch}_{status_name}_{metric}")
                summary = empirical_summary(values)
                rows.append(
                    {
                        "branch": branch,
                        "solver_stratum": status_name,
                        "metric": metric,
                        "case_count": len(subset),
                        "Q10": summary["Q10"],
                        "Q50": summary["Q50"],
                        "Q90": summary["Q90"],
                        "mean": summary["mean"],
                        "unit": unit,
                        "read_only_aggregation": True,
                    }
                )
    return pd.DataFrame(rows)


def write_realization_aggregates(realization_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _input_paths(Path("."), realization_dir, Path("."), Path("."))
    for role in ["realization_native", "realization_matched_native", "realization_smg"]:
        verify_hash(role, paths[role])
    columns = sorted(set(["uid", "solver_success"] + CLIPPING_METRICS + RISK_METRICS))
    native = pd.read_parquet(paths["realization_native"], columns=columns, engine="pyarrow")
    matched_native = pd.read_parquet(paths["realization_matched_native"], columns=columns, engine="pyarrow")
    smg = pd.read_parquet(paths["realization_smg"], columns=columns, engine="pyarrow")
    if len(native) != 3213 or len(matched_native) != 3212 or len(smg) != 3212:
        raise PolicyStop(
            f"STOP_REALIZATION_CASE_COUNT_MISMATCH: native={len(native)} matched_native={len(matched_native)} smg={len(smg)}"
        )
    clipping = clipping_fairness_closure(matched_native, smg)
    risk = solver_status_risk_stratification(
        {"NATIVE_ALL": native, "MATCHED_NATIVE": matched_native, "MATCHED_SMG": smg}
    )
    write_csv_replaceable(output_dir / "CLIPPING_FAIRNESS_CLOSURE.csv", clipping)
    write_csv_replaceable(output_dir / "SOLVER_STATUS_RISK_STRATIFICATION.csv", risk)
    return clipping, risk


def run_unit_tests() -> dict[str, Any]:
    tests: list[dict[str, Any]] = []

    def record(name: str, passed: bool, actual: Any = None, expected: Any = None) -> None:
        tests.append({"name": name, "pass": bool(passed), "actual": actual, "expected": expected})

    digest = stateless_digest(NS_AUGMENT, 42, 72, 0, "img_test", "AUGMENT_ON").hex()
    record(
        "stateless_digest_known_value",
        digest == "eccd49acf8c97b2f0101b5efad7e867e4273dc1fd1984d1c8212e78bb6be2c90",
        digest,
        "eccd49acf8c97b2f0101b5efad7e867e4273dc1fd1984d1c8212e78bb6be2c90",
    )
    index = stateless_index(17, NS_DESTINATION, 42, 72, 0, "img_test", "DESTINATION")
    record("stateless_index_known_value", index == 13, index, 13)
    record(
        "visit_matching_namespace_frozen",
        NS_MATCHING_TIEBREAK == "M1_ROUTEA_VISIT_MATCHING_TIEBREAK_V1",
        NS_MATCHING_TIEBREAK,
        "M1_ROUTEA_VISIT_MATCHING_TIEBREAK_V1",
    )
    quantile = inverse_ecdf_weighted([0.0, 10.0], [0.5, 0.5], 0.5)
    record("inverse_ecdf_no_interpolation", quantile == 0.0, quantile, 0.0)
    distance = float(dstyle_rows(np.zeros((1, 7)), np.ones((1, 7)))[0])
    record("axis_balanced_dstyle", abs(distance - 1.0) <= 1e-15, distance, 1.0)
    valid, reason = validity_arrays(
        np.asarray([[0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 2.0]]),
        np.zeros(7),
        np.ones(7),
    )
    record("descriptor_internal_valid", bool(valid[0]) and reason[0] == "", [bool(valid[0]), reason[0]], [True, ""])
    invalid, reason = validity_arrays(
        np.asarray([[1.0, 0.0, 2.0, -1.0, 0.0, 3.0, 2.0]]),
        np.zeros(7),
        np.ones(7),
    )
    expected_reasons = "TONE_QUANTILE_ORDER|NEGATIVE_CONTRAST|CHROMA_QUANTILE_ORDER"
    record("descriptor_internal_invalid_reasons", not bool(invalid[0]) and reason[0] == expected_reasons, reason[0], expected_reasons)
    whitelist = pd.DataFrame(
        {"uid": ["q2_a", "q2_b"], "image_path": [r"D:\target\a.jpg", r"D:\target\b.jpg"]}
    )
    crosswalk = build_schedule_q2_crosswalk(whitelist, ["b.jpg", "a.jpg"])
    record("schedule_q2_crosswalk_bijection", crosswalk == {"a.jpg": "q2_a", "b.jpg": "q2_b"}, crosswalk)
    record(
        "worker_schedule_schema_frozen",
        WORKER_SCHEDULE_COLUMNS
        == [
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
            "delta_0",
            "delta_1",
            "delta_2",
            "delta_3",
            "delta_4",
            "delta_5",
            "delta_6",
            "donor_schedule_uid",
            "donor_q2_uid",
        ],
        WORKER_SCHEDULE_COLUMNS,
    )
    solution = solve_general_graph_matching(
        ["a", "b", "c", "d"],
        np.asarray([0, 2, 0, 1], dtype=np.int64),
        np.asarray([1, 3, 2, 3], dtype=np.int64),
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    )
    record("matching_maxcardinality_min_cost", solution.pairs == [(0, 1), (2, 3)], solution.pairs, [(0, 1), (2, 3)])
    repeated = solve_general_graph_matching(
        ["a", "b", "c", "d"],
        np.asarray([0, 2, 0, 1], dtype=np.int64),
        np.asarray([1, 3, 2, 3], dtype=np.int64),
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    )
    record(
        "matching_deterministic_replay",
        solution.pairs == repeated.pairs
        and solution.canonical_input_sha256 == repeated.canonical_input_sha256,
    )
    certificate_hash_fields = [
        "dual_lambda_scaled_sha256",
        "forced_edge_LB_scaled_sha256",
        "final_retained_edge_indices_sha256",
        "final_pruned_edge_indices_sha256",
        "selected_pair_edge_indices_sha256",
    ]
    record(
        "presolve_dual_candidate_deterministic_replay",
        all(
            solution.presolve_certificate.get(field)
            == repeated.presolve_certificate.get(field)
            for field in certificate_hash_fields
        )
        and solution.presolve_certificate["candidate_attempts"]
        == repeated.presolve_certificate["candidate_attempts"],
    )
    full_pairs = _networkx_matching_on_edge_indices(
        4,
        np.asarray([0, 2, 0, 1], dtype=np.int64),
        np.asarray([1, 3, 2, 3], dtype=np.int64),
        solution.solver_weights,
        np.arange(4, dtype=np.int64),
    )
    record(
        "presolve_small_graph_full_pair_regression",
        solution.pairs == full_pairs,
        solution.pairs,
        full_pairs,
    )
    record(
        "presolve_exact_threshold_equality_retained",
        bool(np.asarray([10, 11], dtype=np.int64)[0] <= 10)
        and not bool(np.asarray([10, 11], dtype=np.int64)[1] <= 10),
    )
    odd = solve_general_graph_matching(
        ["a", "b", "c"],
        np.asarray([0, 0, 1], dtype=np.int64),
        np.asarray([1, 2, 2], dtype=np.int64),
        np.asarray([0.0, 2.0, 3.0], dtype=np.float64),
    )
    record(
        "presolve_odd_node_fixed_k",
        len(odd.pairs) == 1 and odd.presolve_certificate["fixed_cardinality_k"] == 1,
        [len(odd.pairs), odd.presolve_certificate["fixed_cardinality_k"]],
        [1, 1],
    )
    deficient = solve_general_graph_matching(
        ["a", "b", "c", "d"],
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([0.0], dtype=np.float64),
    )
    record(
        "presolve_cardinality_deficient_full_fallback",
        deficient.pairs == [(0, 1)]
        and deficient.presolve_certificate["fallback_full_graph"] is True
        and deficient.presolve_certificate["selected_matching_cardinality"] == 1,
        deficient.presolve_certificate["fallback_reason"],
        "full graph fallback with one selected edge",
    )
    fallback_disabled_rejected = False
    try:
        solve_general_graph_matching(
            ["a", "b", "c", "d"],
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0.0], dtype=np.float64),
            allow_full_graph_fallback=False,
        )
    except PolicyStop as error:
        fallback_disabled_rejected = str(error) == (
            "STOP_MATCHING_PRESOLVE_CERTIFICATE_NOT_ACCEPTED_FULL_FALLBACK_DISABLED"
        )
    record(
        "presolve_geometry_helper_full_fallback_disabled",
        fallback_disabled_rejected,
    )
    duplicate_rejected = False
    try:
        solve_general_graph_matching(
            ["a", "b"],
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        )
    except PolicyStop as error:
        duplicate_rejected = str(error) == "STOP_MATCHING_FULL_EDGE_DUPLICATE"
    record("presolve_duplicate_full_edge_rejected", duplicate_rejected)
    _, big_storage, big_encoding = _integer_sequence_sha256([10**20, -(10**20)])
    record(
        "presolve_python_bigint_certificate_encoding",
        big_encoding == "ASCII_DECIMAL_FIXED_WIDTH" and big_storage.dtype.kind == "S",
        [big_encoding, str(big_storage.dtype)],
        ["ASCII_DECIMAL_FIXED_WIDTH", "bytes"],
    )
    all_pass = all(test["pass"] for test in tests)
    return {
        "phase": PHASE,
        "all_pass": all_pass,
        "networkx_version": nx.__version__,
        "required_networkx_version": REQUIRED_NETWORKX_VERSION,
        "scipy_version": scipy.__version__,
        "highs_version": _highs_version(),
        "matching_presolve_addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
        "tests": tests,
    }


def write_phase_input_hashes(output_dir: Path, paths: Mapping[str, Path], roles: Sequence[str]) -> None:
    rows = []
    for role in roles:
        path = paths[role]
        rows.append(
            {
                "role": role,
                "absolute_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv_replaceable(output_dir / "VISIT_POLICY_PHASE_INPUT_HASHES.csv", pd.DataFrame(rows))


def _resolve_contract_input(root: Path, value: Any, default: Path) -> Path:
    if value in (None, ""):
        return (root / default).resolve()
    candidate = Path(str(value))
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _derived_policy_contract(
    contract_path: Path,
    provided_contract: Mapping[str, Any],
    paths: Mapping[str, Path],
    context: PolicyContext,
    adaptation_seed: int,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "source_contract_path": str(contract_path.resolve()),
        "source_contract_sha256": sha256_file(contract_path),
        "source_contract_experiment": provided_contract.get("experiment", ""),
        "formal_policy": "PER_VISIT_STATELESS_RESAMPLING_PLUS_OFFLINE_POLICY_INSTANTIATION",
        "formal_visit_key": ["adaptation_seed", "epoch", "target_visit_ordinal", "target_uid"],
        "adaptation_seed": adaptation_seed,
        "epoch_range": [FORMAL_EPOCH_START, FORMAL_EPOCH_END],
        "policy_application_probability": P_AUG,
        "policy_application_probability_not_searched": True,
        "rng": {
            "algorithm": "SHA-256",
            "stateless": True,
            "mutable_global_rng": False,
            "namespaces": [NS_AUGMENT, NS_DOSE, NS_DESTINATION, NS_MATCHING_TIEBREAK],
        },
        "dose": {
            "source": "frozen equal-anchor Q_T_tube Q10-Q90 inclusive truncated weighted empirical PMF",
            "q10_tube": context.q10_tube,
            "q90_tube": context.q90_tube,
            "sampling": "weighted inverse empirical CDF; not Uniform(bounds)",
        },
        "destination": {
            "rule": "d_star <= D_style(A,B) <= 2*tau",
            "tau": context.tau,
            "sampling": "uniform over lexicographically ordered legal destination event IDs",
        },
        "matching": {
            "per_epoch": True,
            "solver": "networkx.algorithms.matching.max_weight_matching(G,maxcardinality=True,weight='solver_weight')",
            "networkx_version": REQUIRED_NETWORKX_VERSION,
            "greedy_fallback": False,
            "derangement": "maximum-cardinality 2-cycle matching",
            "cost": "abs(dev_smg_x-dev_native_x)+abs(dev_smg_y-dev_native_y)",
            "cost_integerization": "int(numpy.rint(cost*1e9)); IEEE-754 ties-to-even",
            "tiebreak_namespace": NS_MATCHING_TIEBREAK,
            "tie_rank": "rank by (SHA256(namespace|visit_key_lo|visit_key_hi),visit_key_lo,visit_key_hi)",
            "cost_priority": "floor(native_valid_node_count/2)*max(mutual_edge_count-1,1)+1",
            "solver_weight": "-(cost_int*cost_priority+tie_rank)",
            "execution_presolve": {
                "name": "EXACT_OPTIMALITY_SAFE_PRESOLVE_V1",
                "engineering_only": True,
                "scientific_contract_modified": False,
                "addendum_path": str(paths["matching_presolve_addendum"].resolve()),
                "addendum_sha256": PRESOLVE_ADDENDUM_SHA256,
                "D": PRESOLVE_D,
                "linprog_method": PRESOLVE_LINPROG_METHOD,
                "linprog_options": dict(PRESOLVE_LINPROG_OPTIONS),
                "margin_numerators": list(PRESOLVE_MARGIN_NUMERATORS),
                "margin_denominator": PRESOLVE_MARGIN_DENOMINATOR,
                "failure_action": "FULL_GRAPH_NETWORKX_FALLBACK",
            },
        },
        "worker_schedule_schema": WORKER_SCHEDULE_COLUMNS,
        "worker_apply_rule": "augment_on AND valid AND policy_flag == APPLY_OFFLINE_THETA",
        "branches": ["NATIVE_A", "MATCHED_A", "SMG"],
        "q2_uid_crosswalk": {
            "rule": "unique basename of approved Q2 whitelist image_path maps schedule_uid to Q2 uid",
            "filesystem_search": False,
            "fallback": False,
        },
        "input_hashes": {
            role: {"absolute_path": str(paths[role].resolve()), "sha256": sha}
            for role, sha in context.input_hashes.items()
        },
        "hard_zeros": {
            "target_adapt_label_reads": 0,
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "image_decode_calls": 0,
            "detector_forward_calls": 0,
            "training_steps": 0,
            "photometric_solver_calls": 0,
        },
    }


def _run_epoch_process(
    root: str,
    output_dir: str,
    contract_path: str,
    epoch: int,
) -> tuple[int, dict[str, Any]]:
    """Spawn-safe entry point; a child owns exactly one immutable epoch partition."""
    audit = run_policy(
        Path(root),
        Path(output_dir),
        Path(contract_path),
        epoch_only=epoch,
        aggregate_only=False,
        parallel_workers=1,
    )
    return epoch, audit


def run_policy(
    root: Path,
    output_dir: Path,
    contract_path: Path,
    *,
    epoch_only: int | None = None,
    aggregate_only: bool = False,
    parallel_workers: int = 2,
) -> dict[str, Any]:
    """Instantiate the full frozen e72-e100 policy and aggregate prior evidence.

    The function is safe to call again after interruption: completed epoch
    partitions are SHA-validated and skipped; no greedy matching fallback exists.
    Full runs use at most two independent Windows-spawn workers.  Each child
    executes the unchanged per-epoch construction and owns disjoint immutable
    partition paths.  Aggregation remains single-process and epoch ordered.
    """
    if epoch_only is not None and aggregate_only:
        raise PolicyStop("STOP_POLICY_EXECUTION_MODE_CONFLICT")
    if epoch_only is not None and not (FORMAL_EPOCH_START <= int(epoch_only) <= FORMAL_EPOCH_END):
        raise PolicyStop(f"STOP_EPOCH_ONLY_OUT_OF_RANGE: {epoch_only}")
    if parallel_workers not in (1, 2):
        raise PolicyStop(f"STOP_POLICY_PARALLEL_WORKER_COUNT_INVALID: {parallel_workers}")
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    contract_path = Path(contract_path).resolve()
    if str(root).lower() != r"d:\cv_yolo11_protocolb_formal":
        raise PolicyStop(f"STOP_WORKING_ROOT_MISMATCH: {root}")
    presolve_addendum_path = (root / PRESOLVE_ADDENDUM_REL).resolve()
    if not presolve_addendum_path.is_file():
        raise PolicyStop(f"STOP_MATCHING_PRESOLVE_ADDENDUM_MISSING: {presolve_addendum_path}")
    presolve_addendum_sha256 = sha256_file(presolve_addendum_path)
    if presolve_addendum_sha256 != PRESOLVE_ADDENDUM_SHA256:
        raise PolicyStop(
            "STOP_MATCHING_PRESOLVE_ADDENDUM_SHA_MISMATCH: "
            f"actual={presolve_addendum_sha256} expected={PRESOLVE_ADDENDUM_SHA256}"
        )
    if not contract_path.is_file():
        raise PolicyStop(f"STOP_SEMANTIC_RESOURCE_CONTRACT_MISSING: {contract_path}")
    provided_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(provided_contract, dict):
        raise PolicyStop("STOP_SEMANTIC_RESOURCE_CONTRACT_SCHEMA_INVALID")
    if provided_contract.get("schema") != "m1_routea_semantic_resource_gate_measurement_contract_v1":
        raise PolicyStop(
            f"STOP_SEMANTIC_RESOURCE_CONTRACT_SCHEMA_MISMATCH: {provided_contract.get('schema')}"
        )
    declared_experiment = provided_contract.get("experiment", EXPERIMENT)
    if declared_experiment != EXPERIMENT:
        raise PolicyStop(
            f"STOP_SEMANTIC_RESOURCE_CONTRACT_EXPERIMENT_MISMATCH: {declared_experiment}"
        )
    policy_section = provided_contract.get("visit_policy", {})
    if not isinstance(policy_section, dict):
        raise PolicyStop("STOP_VISIT_POLICY_CONTRACT_SECTION_INVALID")
    required_policy_fields = {
        "timeline_epochs",
        "adaptation_seed",
        "application_probability",
        "application_probability_searched",
        "matched_control",
    }
    missing_policy_fields = sorted(required_policy_fields - set(policy_section))
    if missing_policy_fields:
        raise PolicyStop(f"STOP_VISIT_POLICY_CONTRACT_FIELDS_MISSING: {missing_policy_fields}")
    adaptation_seed = int(policy_section["adaptation_seed"])
    declared_p_aug = float(policy_section["application_probability"])
    if declared_p_aug != P_AUG:
        raise PolicyStop(f"STOP_POLICY_APPLICATION_PROBABILITY_CHANGED: {declared_p_aug}")
    if policy_section["application_probability_searched"] is not False:
        raise PolicyStop("STOP_POLICY_APPLICATION_PROBABILITY_SEARCH_FLAG_INVALID")
    timeline = policy_section["timeline_epochs"]
    if not isinstance(timeline, list) or len(timeline) != 2:
        raise PolicyStop(f"STOP_FORMAL_VISIT_TIMELINE_SCHEMA_INVALID: {timeline}")
    declared_start = int(timeline[0])
    declared_end = int(timeline[1])
    if (declared_start, declared_end) != (FORMAL_EPOCH_START, FORMAL_EPOCH_END):
        raise PolicyStop(
            f"STOP_FORMAL_VISIT_EPOCH_RANGE_CHANGED: {(declared_start, declared_end)}"
        )
    matched_control = policy_section["matched_control"]
    if not isinstance(matched_control, dict) or "tie_break_namespace" not in matched_control:
        raise PolicyStop("STOP_VISIT_MATCHED_CONTROL_CONTRACT_SCHEMA_INVALID")
    declared_namespace = matched_control["tie_break_namespace"]
    if declared_namespace != NS_MATCHING_TIEBREAK:
        raise PolicyStop(
            f"STOP_VISIT_MATCHING_NAMESPACE_CHANGED: {declared_namespace}"
        )

    inputs = provided_contract.get("inputs", {})
    if not isinstance(inputs, dict):
        raise PolicyStop("STOP_SEMANTIC_RESOURCE_INPUT_CONTRACT_INVALID")
    geometry_dir = _resolve_contract_input(
        root,
        inputs.get("geometry_formal_directory", inputs.get("geometry_dir")),
        DEFAULT_GEOMETRY_REL,
    )
    realization_dir = _resolve_contract_input(
        root,
        inputs.get("realization_formal_directory", inputs.get("realization_dir")),
        DEFAULT_REALIZATION_REL,
    )
    q2_dir = _resolve_contract_input(
        root,
        inputs.get("q2_formal_directory", inputs.get("q2_dir")),
        DEFAULT_Q2_REL,
    )
    schedule_path = _resolve_contract_input(
        root,
        inputs.get("target_schedule_path", inputs.get("schedule_path")),
        DEFAULT_SCHEDULE_REL,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    unit = run_unit_tests()
    if epoch_only is None:
        write_json_replaceable(output_dir / "VISIT_POLICY_IMPLEMENTATION_UNIT_TESTS.json", unit)
    if not unit["all_pass"]:
        raise PolicyStop("STOP_VISIT_POLICY_IMPLEMENTATION_UNIT_TEST_FAIL")

    verify_hash("target_schedule", schedule_path)
    schedules = load_schedule(schedule_path, FORMAL_EPOCH_START, FORMAL_EPOCH_END)
    all_schedule_uids = [
        uid for frame in schedules.values() for uid in frame["schedule_uid"].astype(str).tolist()
    ]
    context = load_policy_context(geometry_dir, q2_dir, schedule_path, all_schedule_uids)
    paths = _input_paths(geometry_dir, realization_dir, q2_dir, schedule_path)
    paths["matching_presolve_addendum"] = presolve_addendum_path
    policy_contract = _derived_policy_contract(
        contract_path, provided_contract, paths, context, adaptation_seed
    )
    if epoch_only is None:
        write_json_replaceable(output_dir / "VISIT_LEVEL_POLICY_CONTRACT.json", policy_contract)

    if epoch_only is not None:
        epoch = int(epoch_only)
        audit = instantiate_epoch_policy(
            epoch, schedules[epoch], context, adaptation_seed, output_dir
        )
        print(
            f"epoch {epoch}: native={audit['native_augmented_valid_count']} "
            f"matched={audit['matched_visit_count']} pairs={audit['matched_pair_count']}",
            flush=True,
        )
        return audit

    epoch_audits: list[dict[str, Any]] = []
    if aggregate_only:
        for epoch in range(FORMAL_EPOCH_START, FORMAL_EPOCH_END + 1):
            audit_path = output_dir / "raw/visit_delta_multiset_audit" / f"epoch_{epoch:03d}.json"
            audit = _resume_epoch_audit(audit_path, context, adaptation_seed)
            if audit is None:
                raise PolicyStop(f"STOP_AGGREGATE_EPOCH_PARTITION_MISSING_E{epoch:03d}")
            epoch_audits.append(audit)
    elif parallel_workers == 1:
        for epoch in range(FORMAL_EPOCH_START, FORMAL_EPOCH_END + 1):
            audit = instantiate_epoch_policy(
                epoch, schedules[epoch], context, adaptation_seed, output_dir
            )
            epoch_audits.append(audit)
            print(
                f"epoch {epoch}: native={audit['native_augmented_valid_count']} "
                f"matched={audit['matched_visit_count']} pairs={audit['matched_pair_count']}",
                flush=True,
            )
    else:
        completed: dict[int, dict[str, Any]] = {}
        pending: list[int] = []
        for epoch in range(FORMAL_EPOCH_START, FORMAL_EPOCH_END + 1):
            audit_path = output_dir / "raw/visit_delta_multiset_audit" / f"epoch_{epoch:03d}.json"
            audit = _resume_epoch_audit(audit_path, context, adaptation_seed)
            if audit is None:
                pending.append(epoch)
            else:
                completed[epoch] = audit
                print(
                    f"epoch {epoch}: resumed native={audit['native_augmented_valid_count']} "
                    f"matched={audit['matched_visit_count']} pairs={audit['matched_pair_count']}",
                    flush=True,
                )
        if pending:
            spawn_context = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=parallel_workers,
                mp_context=spawn_context,
            )
            futures: dict[Any, int] = {}
            try:
                futures = {
                    executor.submit(
                        _run_epoch_process,
                        str(root),
                        str(output_dir),
                        str(contract_path),
                        epoch,
                    ): epoch
                    for epoch in pending
                }
                for future in as_completed(futures):
                    epoch, audit = future.result()
                    completed[epoch] = audit
                    print(
                        f"epoch {epoch}: completed native={audit['native_augmented_valid_count']} "
                        f"matched={audit['matched_visit_count']} pairs={audit['matched_pair_count']}",
                        flush=True,
                    )
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
        epoch_audits = [completed[epoch] for epoch in range(FORMAL_EPOCH_START, FORMAL_EPOCH_END + 1)]
    _, matching_audit = write_policy_aggregate(output_dir, adaptation_seed)
    if matching_audit.get("formal_horizon_complete") is not True:
        raise PolicyStop("STOP_FORMAL_VISIT_POLICY_HORIZON_INCOMPLETE")
    write_realization_aggregates(realization_dir, output_dir)

    used_roles = list(context.input_hashes) + [
        "realization_native",
        "realization_matched_native",
        "realization_smg",
        "matching_presolve_addendum",
    ]
    paths_with_contract = dict(paths)
    paths_with_contract["semantic_resource_contract"] = contract_path
    write_phase_input_hashes(
        output_dir,
        paths_with_contract,
        sorted(set(used_roles)) + ["semantic_resource_contract"],
    )
    phase_audit = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "status": "COMPLETE",
        "unit_tests_pass": True,
        "adaptation_seed": adaptation_seed,
        "epoch_start": FORMAL_EPOCH_START,
        "epoch_end": FORMAL_EPOCH_END,
        "policy_application_probability": P_AUG,
        "policy_application_probability_not_searched": True,
        "formal_policy": "PER_VISIT_STATELESS_RESAMPLING_PLUS_OFFLINE_POLICY_INSTANTIATION",
        "training_step_powell_calls": 0,
        "matching_solver": "networkx.algorithms.matching.max_weight_matching",
        "matching_solver_version": nx.__version__,
        "matching_greedy_fallback": False,
        "matching_execution_presolve_summary": matching_audit[
            "matching_execution_presolve_summary"
        ],
        "q2_schedule_crosswalk": "unique approved-whitelist image_path basename to Q2 uid",
        "filesystem_uid_search": False,
        "hard_zeros": {
            "target_adapt_label_reads": 0,
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "image_decode_calls": 0,
            "detector_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "ema_updates": 0,
            "photometric_realization_calls": 0,
            "geometry_recomputation": False,
            "tau_modification": False,
            "matching_rule_modification": False,
        },
    }
    write_json_replaceable(output_dir / "VISIT_POLICY_PHASE_AUDIT.json", phase_audit)
    return phase_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--epoch-only",
        type=int,
        help="Generate exactly one immutable epoch partition; do not aggregate shared outputs.",
    )
    execution.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Validate all immutable epoch partitions and generate only shared aggregate outputs.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        choices=(1, 2),
        default=2,
        help="Full-run epoch worker count; fixed upper bound 2. Ignored by --epoch-only/--aggregate-only.",
    )
    args = parser.parse_args(argv)
    result = run_policy(
        Path(args.root),
        Path(args.output_dir),
        Path(args.contract),
        epoch_only=args.epoch_only,
        aggregate_only=args.aggregate_only,
        parallel_workers=args.parallel_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PolicyStop as error:
        print(str(error), flush=True)
        raise SystemExit(2)
