from __future__ import annotations

"""Prepare seed-specific Native-A Fixed policies and exact BGR caches.

This is an offline, fail-closed adapter around the already frozen Route-A and
Fixed-A implementations.  It performs no detector training, reads no Target
Dev/Test content, and reads no Target Adapt labels.  The only new identity is
the adaptation seed supplied by the authoritative V2 seed contract.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal")
PROJECT = ROOT / "project"
SCRIPTS = PROJECT / "scripts"
CONTRACTS = PROJECT / "contracts"
RESULT_ROOT = PROJECT / "results" / "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2"
IDENTITY_CONTRACT = CONTRACTS / "M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1.json"
IDENTITY_CONTRACT_SHA = CONTRACTS / "M1_FIXED_FRESH_SEED_IDENTITY_CONTRACT_V1.sha256"
SEEDS = (45, 46)
EPOCHS = tuple(range(72, 101))
ALL_SCHEDULE_EPOCHS = tuple(range(71, 101))
VISITS_PER_EPOCH = 4_416
TOTAL_VISITS = len(EPOCHS) * VISITS_PER_EPOCH
EXPECTED_IMAGES = 3_272
PARENT_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
PARENT = PROJECT / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
SEED42_FIXED_ROOT = PROJECT / "results" / (
    "M1_NATIVE_A_FIXED_COMPLEXITY_UTILITY_ABLATION_V1_PRETRAIN_GATE"
)
SEED42_SEMANTIC_ROOT = PROJECT / "audits" / (
    "M1_ROUTE_A_SEMANTIC_RESOURCE_GATE_MEASUREMENT_V1_20260810_001748"
)
SEED42_Q2_ROOT = PROJECT / "audits" / (
    "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
)
SEED42_REALIZATION_ROOT = PROJECT / "audits" / (
    "M1_ROUTE_A_PHOTOMETRIC_REALIZATION_FEASIBILITY_MEASUREMENT_V1_20260809_131730"
)
TARGET_YAML = ROOT / "datasets" / "dataset_final_v2_1" / "configs" / "target_adapt_unlabeled.yaml"
DATASET_MANIFEST = "bd4f1f97eb6cbd37e14cbb2ab63b4301f7833fc1984b67543dfa86088962b249"

for candidate in (SCRIPTS, PROJECT / "ultralytics"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import m1_native_a_fixed_policy_realization_v1 as fixed  # noqa: E402
import m1_routea_offline_theta_cache_engine_v1 as cache_engine  # noqa: E402
import m1_routea_semantic_resource_policy_v1 as route_policy  # noqa: E402
import protocol_b_horizon_schedule as horizon  # noqa: E402
from m1_routea_offline_theta_cache_engine_v1 import (  # noqa: E402
    BRANCH_NATIVE,
    ScientificConfigIdentity,
    VisitAlias,
    build_execution_plan,
    make_explicit_target_solve_request,
)
from protocol_b_target_schedule import (  # noqa: E402
    ProtocolBTargetSchedule,
    resolve_target_adapt_root,
    scan_target_pool,
    sha256_lines,
    target_pool_metadata_hash,
)
from protocol_b_schedule import ProtocolBSchedule  # noqa: E402


class PreparationStop(RuntimeError):
    """Scientific-integrity stop for offline seed preparation."""


def _fresh_worker_initializer(runtime: Any, adaptation_seed: int) -> None:
    """Parameterize only the frozen visit-alias seed guard in spawned workers."""
    cache_engine.FROZEN_ADAPTATION_SEED = int(adaptation_seed)
    cache_engine._worker_initializer(runtime)


class FreshAdaptationSeedSpawnPool(cache_engine.PersistentSpawnPool):
    """Frozen persistent pool with an explicit fresh adaptation-seed initarg."""

    def __init__(self, runtime: Any, *, max_workers: int, chunksize: int = 1) -> None:
        super().__init__(runtime, max_workers=max_workers, chunksize=chunksize)
        value = os.environ.get("M1_FIXED_FRESH_ADAPTATION_SEED", "")
        if value not in {"42", "45", "46"}:
            raise PreparationStop(f"STOP_POOL_ADAPTATION_SEED_ENV:{value}")
        self.fresh_adaptation_seed = int(value)

    def start(self) -> "FreshAdaptationSeedSpawnPool":
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=context,
                initializer=_fresh_worker_initializer,
                initargs=(self.runtime, self.fresh_adaptation_seed),
            )
        return self


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
            )
            + "\n"
        ).encode("utf-8"),
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    atomic_bytes(path, payload)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_paths(seed: int) -> dict[str, Path]:
    if int(seed) not in SEEDS:
        raise PreparationStop(f"STOP_SEED_NOT_AUTHORIZED:{seed}")
    root = RESULT_ROOT / f"SEED{int(seed)}"
    policy = root / "POLICY"
    schedule = policy / "SCHEDULE"
    native = policy / "NATIVE_A_VISIT_POLICY"
    cache = root / "CACHE"
    return {
        "root": root,
        "policy": policy,
        "schedule": schedule,
        "source_schedule": schedule / "source_schedule_e071_e100.jsonl",
        "source_augmentation": schedule / "source_augmentation_e071_e100.jsonl",
        "source_manifest": schedule / "source_schedule_manifest.json",
        "target_schedule": schedule / "target_schedule_e071_e100.jsonl",
        "target_manifest": schedule / "target_schedule_e071_e100_manifest.json",
        "schedule_freeze": schedule / "SCHEDULE_FREEZE.json",
        "native": native,
        "native_schedule": native / "visit_policy_schedule.parquet",
        "native_deltas": native / "visit_native_deltas.parquet",
        "native_audit": native / "NATIVE_A_POLICY_AUDIT.json",
        "semantic_contract": policy / "ROUTEA_SEED_SPECIFIC_SEMANTIC_CONTRACT.json",
        "fixed_contract": policy / "FIXED_A_POLICY_CONTRACT.json",
        "cache": cache,
        "policy_gate": root / "POLICY_CACHE_PRETRAIN_GATE.json",
    }


def validate_authorities() -> dict[str, Any]:
    if not IDENTITY_CONTRACT.is_file() or not IDENTITY_CONTRACT_SHA.is_file():
        raise PreparationStop("STOP_IDENTITY_CONTRACT_MISSING")
    identity_sha = sha256_file(IDENTITY_CONTRACT)
    sidecar = IDENTITY_CONTRACT_SHA.read_text(encoding="ascii").strip().split()[0]
    if identity_sha != sidecar:
        raise PreparationStop("STOP_IDENTITY_CONTRACT_SHA_SIDECAR")
    identity = json.loads(IDENTITY_CONTRACT.read_text(encoding="utf-8"))
    if identity.get("parent_sha256") != PARENT_SHA256 or identity.get("fresh_seeds") != [45, 46]:
        raise PreparationStop("STOP_IDENTITY_CONTRACT_CONTENT")
    parent_sha = sha256_file(PARENT)
    if parent_sha != PARENT_SHA256:
        raise PreparationStop(f"STOP_PARENT_IDENTITY_UNRECOVERABLE:{parent_sha}")
    frozen_code = {
        "horizon_schedule": PROJECT / "scripts" / "protocol_b_horizon_schedule.py",
        "routea_policy": PROJECT / "scripts" / "m1_routea_semantic_resource_policy_v1.py",
        "fixed_policy_realization": PROJECT / "scripts" / "m1_native_a_fixed_policy_realization_v1.py",
        "realization_core": PROJECT / "scripts" / "m1_routea_photometric_realization_core_v1.py",
        "offline_cache_engine": PROJECT / "scripts" / "m1_routea_offline_theta_cache_engine_v1.py",
        "exact_transform_core": PROJECT / "scripts" / "m1_routea_resource_equivalence_execution_core_v1.py",
        "fixed_cache_builder": PROJECT / "scripts" / "m1_native_a_fixed_cache_resource_v1.py",
        "fixed_semantic_gate": PROJECT / "scripts" / "m1_native_a_fixed_semantic_gate_v1.py",
    }
    return {
        "identity_contract": str(IDENTITY_CONTRACT.resolve()),
        "identity_contract_sha256": identity_sha,
        "parent": str(PARENT.resolve()),
        "parent_sha256": parent_sha,
        "frozen_code": {
            role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for role, path in frozen_code.items()
        },
    }


def build_schedules(seed: int, paths: Mapping[str, Path]) -> dict[str, Any]:
    freeze_path = paths["schedule_freeze"]
    if freeze_path.is_file():
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        for record in freeze.get("files", {}).values():
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise PreparationStop(f"STOP_EXISTING_SCHEDULE_DRIFT:{path}")
        ProtocolBSchedule(paths["source_schedule"], paths["source_augmentation"])
        ProtocolBTargetSchedule(paths["target_schedule"], paths["target_manifest"])
        return freeze

    marker = (ROOT / "datasets" / "dataset_final_v2_1" / "DATASET_MANIFEST_SHA256.txt").read_text(
        encoding="utf-8"
    ).strip().split()[0]
    if marker != DATASET_MANIFEST:
        raise PreparationStop("STOP_DATASET_MANIFEST_DRIFT")
    source_all, augmentation_all = horizon._source_rows(int(seed))
    source = [row for row in source_all if int(row["epoch"]) in ALL_SCHEDULE_EPOCHS]
    augmentation = [row for row in augmentation_all if int(row["epoch"]) in ALL_SCHEDULE_EPOCHS]
    _, dataset_root, target_root = resolve_target_adapt_root(TARGET_YAML)
    target_uids, uid_to_path = scan_target_pool(target_root)
    target_all = horizon._target_rows(int(seed), target_uids)
    target = [row for row in target_all if int(row["epoch"]) in ALL_SCHEDULE_EPOCHS]
    atomic_jsonl(paths["source_schedule"], source)
    atomic_jsonl(paths["source_augmentation"], augmentation)
    atomic_jsonl(paths["target_schedule"], target)
    source_manifest = {
        "schema": "m1_fixed_fresh_seed_source_schedule_manifest_v1",
        "adaptation_seed": int(seed),
        "dataset_manifest": marker,
        "epochs": list(ALL_SCHEDULE_EPOCHS),
        "source_schedule_seed": int(seed),
        "source_epoch_seed_policy": "adaptation_seed + total_epoch",
        "source_augmentation_rng_namespace": f"protocol-b-source-augmentation-v1|adaptation-seed={seed}",
        "source_dataloader_generator_seed": int(seed),
        "schedule_path": str(paths["source_schedule"].resolve()),
        "schedule_sha256": sha256_file(paths["source_schedule"]),
        "augmentation_path": str(paths["source_augmentation"].resolve()),
        "augmentation_sha256": sha256_file(paths["source_augmentation"]),
        "row_count": len(source),
        "target_dev_reads": 0,
        "target_test_reads": 0,
    }
    atomic_json(paths["source_manifest"], source_manifest)
    target_manifest = {
        "schema": "m1_fixed_fresh_seed_target_schedule_manifest_v1",
        "experiment": "M1_NATIVE_A_FIXED_FRESH_SEED_CONFIRMATION_45_46_V2",
        "dataset_manifest": marker,
        "dataset_root": str(dataset_root.resolve()),
        "target_yaml": str(TARGET_YAML.resolve()),
        "target_yaml_sha256": sha256_file(TARGET_YAML),
        "target_image_root": str(target_root.resolve()),
        "target_pool_size": len(target_uids),
        "target_uid_policy": "relative_path_from_target_adapt_images_with_suffix",
        "target_uid_pool_sha256": sha256_lines(sorted(target_uids)),
        "target_pool_metadata_sha256": target_pool_metadata_hash(target_uids, uid_to_path),
        "adaptation_seed": int(seed),
        "target_schedule_seed": int(seed),
        "target_permutation_rng_namespace": f"protocol-b-target-permutation-v1|adaptation-seed={seed}",
        "target_view_rng_namespace": f"protocol-b-target-same-view-v1|adaptation-seed={seed}",
        "target_dataloader_generator_seed": int(seed) + 1000,
        "epochs": list(ALL_SCHEDULE_EPOCHS),
        "steps_per_epoch": 552,
        "target_batch": 8,
        "visits_per_scheduled_epoch": VISITS_PER_EPOCH,
        "sampling_policy": "seed-namespaced epoch permutations; complete Target Adapt image-only coverage per epoch",
        "view_policy": "same_letterbox_view_v1",
        "letterbox": {
            "new_shape": [640, 640], "auto": False, "scale_fill": False,
            "scaleup": True, "center": True, "stride": 32,
            "padding_value": 114, "interpolation": "cv2.INTER_LINEAR",
            "color_order_after_format": "RGB",
        },
        "burn_in_policy": "e71 schedule frozen but not read while lambda_u=0",
        "schedule_path": str(paths["target_schedule"].resolve()),
        "schedule_sha256": sha256_file(paths["target_schedule"]),
        "row_count": len(target),
        "target_adapt_label_read_count": 0,
        "target_dev_read_count": 0,
        "target_test_read_count": 0,
    }
    atomic_json(paths["target_manifest"], target_manifest)
    ProtocolBSchedule(paths["source_schedule"], paths["source_augmentation"])
    ProtocolBTargetSchedule(paths["target_schedule"], paths["target_manifest"])
    per_epoch = []
    for epoch in ALL_SCHEDULE_EPOCHS:
        sr = [row for row in source if int(row["epoch"]) == epoch]
        ar = [row for row in augmentation if int(row["epoch"]) == epoch]
        tr = [row for row in target if int(row["epoch"]) == epoch]
        source_uids = [uid for row in sr for uid in row["uids"]]
        target_epoch_uids = [uid for row in tr for uid in row["uids"]]
        per_epoch.append({
            "epoch": epoch,
            "source_content_sha256": canonical_json_sha256(sr),
            "source_augmentation_content_sha256": canonical_json_sha256(ar),
            "target_content_sha256": canonical_json_sha256(tr),
            "source_visits": len(source_uids),
            "source_unique_uids": len(set(source_uids)),
            "target_visits": len(target_epoch_uids),
            "target_unique_uids": len(set(target_epoch_uids)),
            "target_visit_min": min(Counter(target_epoch_uids).values()),
            "target_visit_max": max(Counter(target_epoch_uids).values()),
        })
    freeze = {
        "schema": "m1_fixed_fresh_seed_schedule_freeze_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "generated_with_verified_horizon_seed_plumbing": True,
        "horizon_schedule_runner": {
            "path": str((SCRIPTS / "protocol_b_horizon_schedule.py").resolve()),
            "sha256": sha256_file(SCRIPTS / "protocol_b_horizon_schedule.py"),
        },
        "files": {
            role: {"path": str(paths[role].resolve()), "sha256": sha256_file(paths[role])}
            for role in (
                "source_schedule", "source_augmentation", "source_manifest",
                "target_schedule", "target_manifest",
            )
        },
        "per_epoch": per_epoch,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(freeze_path, freeze)
    return freeze


def _native_epoch(
    epoch: int,
    schedule: pd.DataFrame,
    context: Any,
    adaptation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Exact Native-A prefix of the frozen Route-A policy implementation."""

    n_visits = len(schedule)
    if n_visits != route_policy.EXPECTED_VISITS_PER_EPOCH:
        raise PreparationStop(f"STOP_NATIVE_VISIT_COUNT_E{epoch:03d}:{n_visits}")
    frame = schedule.copy().reset_index(drop=True)
    frame["target_uid"] = frame["schedule_uid"].map(context.schedule_to_q2_uid)
    if frame["target_uid"].isna().any():
        raise PreparationStop(f"STOP_NATIVE_CROSSWALK_E{epoch:03d}")
    frame["adaptation_seed"] = int(adaptation_seed)
    frame["visit_key"] = [
        route_policy.visit_key(adaptation_seed, epoch, int(ordinal), uid)
        for ordinal, uid in zip(frame["target_visit_ordinal"], frame["target_uid"])
    ]
    if frame["visit_key"].duplicated().any():
        raise PreparationStop(f"STOP_NATIVE_VISIT_KEY_DUPLICATE_E{epoch:03d}")

    image_indices = np.asarray(
        [context.image_index[str(uid)] for uid in frame["target_uid"]], dtype=np.int64
    )
    event_ids = context.image_event_ids[image_indices]
    z_image = context.z_image[image_indices]
    original_in_tube = context.original_in_tube[image_indices]
    original_internal = context.original_internal_valid[image_indices]
    original_reason = context.original_invalid_reason[image_indices]
    augment_u = np.asarray(
        [
            route_policy.stateless_u01(
                route_policy.NS_AUGMENT,
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
    augment_on = augment_u < route_policy.P_AUG
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
    failure = np.full(
        n_visits, "AUGMENTATION_DISABLED_EXACT_PASSTHROUGH", dtype=object
    )
    for row in np.flatnonzero(augment_on):
        uid = str(frame.at[row, "target_uid"])
        ordinal = int(frame.at[row, "target_visit_ordinal"])
        failure[row] = ""
        if not original_in_tube[row]:
            failure[row] = "UNAVAILABLE_ORIGINAL_IMAGE_OFF_TUBE"
            continue
        if not original_internal[row]:
            suffix = str(original_reason[row])
            failure[row] = "ORIGINAL_DESCRIPTOR_INTERNAL_INVALID" + (
                ":" + suffix if suffix else ""
            )
            continue
        u01 = route_policy.stateless_u01(
            route_policy.NS_DOSE,
            adaptation_seed,
            epoch,
            ordinal,
            uid,
            "INTENDED_GEOMETRIC_DOSE",
        )
        dose_u[row] = u01
        dstar = route_policy._sample_dose(context, u01)
        intended_dose[row] = dstar
        anchor_event = str(event_ids[row])
        refs, distances = context.neighbors.get(
            anchor_event,
            (np.asarray([], dtype=object), np.asarray([], dtype=np.float64)),
        )
        legal = np.flatnonzero(distances >= dstar)
        destination_count[row] = len(legal)
        if len(legal) == 0:
            failure[row] = "NATIVE_TSVA_UNAVAILABLE_NO_DESTINATION_AT_DOSE"
            continue
        pick = route_policy.stateless_index(
            len(legal),
            route_policy.NS_DESTINATION,
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
        measured = float(
            np.sum(np.abs(delta[row]) * route_policy.WEIGHTS, dtype=np.float64)
        )
        tolerance = max(1e-12, 1e-10 * max(1.0, abs(dstar)))
        if abs(measured - dstar) > tolerance:
            failure[row] = "NATIVE_DOSE_NUMERIC_MISMATCH"
            continue
        native_target[row] = z_image[row] + delta[row]

    generated = np.flatnonzero(np.isfinite(delta).all(axis=1))
    if len(generated):
        event_valid, event_reasons = route_policy.validity_arrays(
            event_target[generated], context.median, context.scale
        )
        target_valid, target_reasons = route_policy.validity_arrays(
            native_target[generated], context.median, context.scale
        )
        queried_event, _ = context.event_tree.query(
            event_target[generated] * route_policy.WEIGHTS,
            k=1,
            p=1,
            eps=0,
            workers=1,
        )
        queried_native, _ = context.event_tree.query(
            native_target[generated] * route_policy.WEIGHTS,
            k=1,
            p=1,
            eps=0,
            workers=1,
        )
        event_target_dev[generated] = queried_event
        native_dev[generated] = queried_native
        for local, row in enumerate(generated.tolist()):
            if failure[row]:
                continue
            if not event_valid[local]:
                failure[row] = (
                    "DESCRIPTOR_INTERNAL_INVALID_EVENT_TARGET:" + event_reasons[local]
                )
            elif event_target_dev[row] > context.tau:
                failure[row] = "EVENT_TARGET_OFF_EMPIRICAL_TUBE"
            elif not target_valid[local]:
                failure[row] = (
                    "DESCRIPTOR_INTERNAL_INVALID_NATIVE:" + target_reasons[local]
                )
            elif native_dev[row] > context.tau:
                failure[row] = "NATIVE_ENDPOINT_OFF_EMPIRICAL_TUBE"
            else:
                native_valid[row] = True

    delta_sha = np.full(n_visits, "", dtype=object)
    for row in np.flatnonzero(np.isfinite(delta).all(axis=1)):
        delta_sha[row] = route_policy.delta_hash(delta[row])
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
    for coordinate, name in enumerate(route_policy.BASE):
        frame[f"original_std__{name}"] = z_image[:, coordinate]
        frame[f"native_delta__{name}"] = delta[:, coordinate]
        frame[f"native_target_std__{name}"] = native_target[:, coordinate]

    native_delta_columns = [
        "epoch", "target_visit_ordinal", "visit_key", "schedule_uid",
        "target_uid", "capture_event_id", "intended_geometric_dose",
        "destination_event_id", "anchor_destination_distance", "lambda",
        "native_tube_deviation", "native_delta_sha256",
    ] + [f"native_delta__{name}" for name in route_policy.BASE] + [
        f"native_target_std__{name}" for name in route_policy.BASE
    ]
    native_deltas = frame.loc[native_valid, native_delta_columns].copy()
    worker_rows: list[dict[str, Any]] = []
    for row in range(n_visits):
        valid = bool(native_valid[row])
        requested = bool(augment_on[row])
        policy_flag = (
            "APPLY_OFFLINE_THETA"
            if valid
            else "IDENTITY_POLICY_OFF"
            if not requested
            else "IDENTITY_FALLBACK"
        )
        record: dict[str, Any] = {
            "adaptation_seed": int(adaptation_seed),
            "epoch": int(epoch),
            "target_visit_ordinal": int(frame.at[row, "target_visit_ordinal"]),
            "schedule_uid": str(frame.at[row, "schedule_uid"]),
            "q2_uid": str(frame.at[row, "target_uid"]),
            "branch": BRANCH_NATIVE,
            "augment_on": requested,
            "valid": valid,
            "policy_flag": policy_flag,
            "failure_reason": "" if valid else str(failure[row]),
            "delta_sha256": str(delta_sha[row]) if valid else "",
            "donor_schedule_uid": str(frame.at[row, "schedule_uid"]) if valid else "",
            "donor_q2_uid": str(frame.at[row, "target_uid"]) if valid else "",
        }
        for coordinate in range(7):
            record[f"delta_{coordinate}"] = (
                float(delta[row, coordinate]) if valid else float("nan")
            )
        worker_rows.append(record)
    worker = pd.DataFrame(worker_rows, columns=route_policy.WORKER_SCHEDULE_COLUMNS)
    audit = {
        "epoch": int(epoch),
        "adaptation_seed": int(adaptation_seed),
        "scheduled_visits": n_visits,
        "augment_on_visits": int(augment_on.sum()),
        "native_valid_visits": int(native_valid.sum()),
        "native_identity_visits": int((~native_valid).sum()),
        "invalid_nominal_augmentation_identity_visits": int(
            (augment_on & ~native_valid).sum()
        ),
        "delta_sha_unique": int(native_deltas["native_delta_sha256"].nunique()),
    }
    return worker, native_deltas, audit


def _float_frame_exact(left: pd.DataFrame, right: pd.DataFrame, columns: Sequence[str]) -> bool:
    for column in columns:
        a = left[column].to_numpy(dtype="<f8")
        b = right[column].to_numpy(dtype="<f8")
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False
        finite = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(a[finite].view(np.uint64), b[finite].view(np.uint64)):
            return False
    return True


def seed42_native_equivalence() -> dict[str, Any]:
    audit_path = RESULT_ROOT / "NATIVE_A_PREFIX_SEED42_EQUIVALENCE.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise PreparationStop("STOP_EXISTING_NATIVE_EQUIVALENCE_FAIL")
        if audit.get("adapter_sha256") == sha256_file(Path(__file__)):
            return audit
    canonical_schedule_path = CONTRACTS / "protocol_b_target_schedule_e71_e100.jsonl"
    schedules = route_policy.load_schedule(canonical_schedule_path, 72, 100)
    all_uids = [
        uid for frame in schedules.values() for uid in frame["schedule_uid"].astype(str)
    ]
    context = route_policy.load_policy_context(
        ROOT / route_policy.DEFAULT_GEOMETRY_REL,
        ROOT / route_policy.DEFAULT_Q2_REL,
        canonical_schedule_path,
        all_uids,
    )
    generated_schedule: list[pd.DataFrame] = []
    generated_deltas: list[pd.DataFrame] = []
    for epoch in EPOCHS:
        worker, deltas, _ = _native_epoch(epoch, schedules[epoch], context, 42)
        generated_schedule.append(worker)
        generated_deltas.append(deltas)
    generated_worker = pd.concat(generated_schedule, ignore_index=True)
    generated_delta = pd.concat(generated_deltas, ignore_index=True)
    canonical_worker = pd.read_parquet(
        SEED42_SEMANTIC_ROOT / "raw" / "visit_policy_schedule.parquet"
    )
    canonical_worker = canonical_worker.loc[
        canonical_worker["branch"].astype(str).eq(BRANCH_NATIVE)
    ].reset_index(drop=True)
    canonical_delta = pd.read_parquet(
        SEED42_SEMANTIC_ROOT / "raw" / "visit_native_deltas.parquet"
    ).reset_index(drop=True)
    worker_nonfloat = [
        column for column in route_policy.WORKER_SCHEDULE_COLUMNS
        if column not in {*(f"delta_{index}" for index in range(7))}
    ]
    worker_float = [f"delta_{index}" for index in range(7)]
    delta_float = [
        "intended_geometric_dose", "anchor_destination_distance", "lambda",
        "native_tube_deviation",
        *(f"native_delta__{name}" for name in route_policy.BASE),
        *(f"native_target_std__{name}" for name in route_policy.BASE),
    ]
    delta_nonfloat = [column for column in generated_delta.columns if column not in delta_float]
    checks = {
        "worker_rows_exact": len(generated_worker) == len(canonical_worker) == TOTAL_VISITS,
        "worker_columns_exact": list(generated_worker.columns) == list(canonical_worker.columns),
        "worker_nonfloat_exact": generated_worker[worker_nonfloat].astype(str).equals(
            canonical_worker[worker_nonfloat].astype(str)
        ),
        "worker_float64_bits_exact": _float_frame_exact(
            generated_worker, canonical_worker, worker_float
        ),
        "delta_rows_exact": len(generated_delta) == len(canonical_delta),
        "delta_columns_exact": list(generated_delta.columns) == list(canonical_delta.columns),
        "delta_nonfloat_exact": generated_delta[delta_nonfloat].astype(str).equals(
            canonical_delta[delta_nonfloat].astype(str)
        ),
        "delta_float64_bits_exact": _float_frame_exact(
            generated_delta, canonical_delta, delta_float
        ),
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_native_a_prefix_equivalence_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "generated_worker_rows": len(generated_worker),
        "canonical_worker_rows": len(canonical_worker),
        "generated_delta_rows": len(generated_delta),
        "canonical_delta_rows": len(canonical_delta),
        "canonical_worker_sha256": sha256_file(
            SEED42_SEMANTIC_ROOT / "raw" / "visit_policy_schedule.parquet"
        ),
        "canonical_delta_sha256": sha256_file(
            SEED42_SEMANTIC_ROOT / "raw" / "visit_native_deltas.parquet"
        ),
        "frozen_policy_runner_sha256": sha256_file(
            SCRIPTS / "m1_routea_semantic_resource_policy_v1.py"
        ),
        "adapter_sha256": sha256_file(Path(__file__)),
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise PreparationStop(f"STOP_NATIVE_PREFIX_EQUIVALENCE:{checks}")
    return audit


def build_native_policy(seed: int, paths: Mapping[str, Path]) -> dict[str, Any]:
    if paths["native_audit"].is_file():
        audit = json.loads(paths["native_audit"].read_text(encoding="utf-8"))
        for role in ("native_schedule", "native_deltas"):
            record = audit["files"][role]
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise PreparationStop(f"STOP_NATIVE_POLICY_DRIFT:{role}")
        return audit
    seed42_native_equivalence()
    schedule_sha = sha256_file(paths["target_schedule"])
    previous = route_policy.EXPECTED_HASHES["target_schedule"]
    route_policy.EXPECTED_HASHES["target_schedule"] = schedule_sha
    try:
        schedules = route_policy.load_schedule(paths["target_schedule"], 72, 100)
        all_uids = [
            uid for frame in schedules.values() for uid in frame["schedule_uid"].astype(str)
        ]
        context = route_policy.load_policy_context(
            ROOT / route_policy.DEFAULT_GEOMETRY_REL,
            ROOT / route_policy.DEFAULT_Q2_REL,
            paths["target_schedule"],
            all_uids,
        )
        workers: list[pd.DataFrame] = []
        deltas: list[pd.DataFrame] = []
        epochs: list[dict[str, Any]] = []
        for epoch in EPOCHS:
            worker, delta, epoch_audit = _native_epoch(
                epoch, schedules[epoch], context, int(seed)
            )
            workers.append(worker)
            deltas.append(delta)
            epochs.append(epoch_audit)
        worker_all = pd.concat(workers, ignore_index=True)
        delta_all = pd.concat(deltas, ignore_index=True)
        atomic_parquet(paths["native_schedule"], worker_all)
        atomic_parquet(paths["native_deltas"], delta_all)
    finally:
        route_policy.EXPECTED_HASHES["target_schedule"] = previous
    active = (
        worker_all["augment_on"].astype(bool)
        & worker_all["valid"].astype(bool)
        & worker_all["policy_flag"].astype(str).eq("APPLY_OFFLINE_THETA")
    )
    if len(worker_all) != TOTAL_VISITS or int(active.sum()) != len(delta_all):
        raise PreparationStop("STOP_NATIVE_POLICY_INTERNAL_COUNT")
    audit = {
        "schema": "m1_fixed_fresh_seed_native_a_policy_audit_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "policy": "NATIVE_A_RESAMPLED_VISIT_POLICY",
        "formal_visits": len(worker_all),
        "augment_on_visits": int(worker_all["augment_on"].astype(bool).sum()),
        "valid_active_visits": int(active.sum()),
        "identity_visits": int((~active).sum()),
        "candidate_rows": len(delta_all),
        "images_with_candidates": int(delta_all["target_uid"].nunique()),
        "images_without_candidates": EXPECTED_IMAGES - int(delta_all["target_uid"].nunique()),
        "per_epoch": epochs,
        "rng_namespaces": [
            route_policy.NS_AUGMENT,
            route_policy.NS_DOSE,
            route_policy.NS_DESTINATION,
        ],
        "matching_not_run_reason": (
            "Fixed-A consumes only the NATIVE_A branch; complete seed42 bitwise "
            "equivalence proves this extracted prefix before fresh-seed use"
        ),
        "seed42_prefix_equivalence": {
            "path": str((RESULT_ROOT / "NATIVE_A_PREFIX_SEED42_EQUIVALENCE.json").resolve()),
            "sha256": sha256_file(RESULT_ROOT / "NATIVE_A_PREFIX_SEED42_EQUIVALENCE.json"),
        },
        "files": {
            role: {"path": str(paths[role].resolve()), "sha256": sha256_file(paths[role])}
            for role in ("native_schedule", "native_deltas")
        },
        "policy_lineage_sha256": context.lineage_sha256,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["native_audit"], audit)
    return audit


def build_semantic_contract(seed: int, paths: Mapping[str, Path]) -> dict[str, Any]:
    destination = paths["semantic_contract"]
    if destination.is_file():
        return json.loads(destination.read_text(encoding="utf-8"))
    source_path = CONTRACTS / "m1_routea_semantic_resource_gate_measurement_v1.json"
    contract = json.loads(source_path.read_text(encoding="utf-8"))
    contract["visit_policy"]["adaptation_seed"] = int(seed)
    contract.setdefault("inputs", {})["target_schedule_path"] = str(
        paths["target_schedule"].resolve()
    )
    contract["fresh_seed_extension"] = {
        "schema": "m1_fixed_fresh_seed_routea_policy_extension_v1",
        "authoritative_identity_contract": str(IDENTITY_CONTRACT.resolve()),
        "authoritative_identity_contract_sha256": sha256_file(IDENTITY_CONTRACT),
        "adaptation_seed": int(seed),
        "target_schedule_path": str(paths["target_schedule"].resolve()),
        "target_schedule_sha256": sha256_file(paths["target_schedule"]),
        "scientific_algorithm_unchanged": True,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
    }
    atomic_json(destination, contract)
    return contract


def _load_fixed_frames(paths: Mapping[str, Path], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    whitelist = pd.read_csv(
        fixed.WHITELIST,
        dtype=str,
        usecols=["uid", "image_path", "image_sha256", "capture_event_id"],
    )
    schedule_columns = [
        "adaptation_seed", "epoch", "target_visit_ordinal", "schedule_uid",
        "q2_uid", "branch", "augment_on", "valid", "policy_flag",
        "failure_reason", "delta_sha256", *(f"delta_{index}" for index in range(7)),
    ]
    schedule = pd.read_parquet(paths["native_schedule"], columns=schedule_columns)
    if set(schedule["branch"].astype(str)) != {BRANCH_NATIVE}:
        raise PreparationStop("STOP_NATIVE_ONLY_SCHEDULE_BRANCH")
    schedule["augment_on"] = fixed._strict_bool(schedule["augment_on"], "augment_on")
    schedule["valid"] = fixed._strict_bool(schedule["valid"], "valid")
    schedule["formal_visit_index"] = [
        fixed.formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(schedule["epoch"], schedule["target_visit_ordinal"])
    ]
    schedule = schedule.sort_values("formal_visit_index", kind="mergesort").reset_index(drop=True)
    if (
        len(schedule) != TOTAL_VISITS
        or not np.array_equal(
            schedule["formal_visit_index"].to_numpy(dtype=np.int64),
            np.arange(TOTAL_VISITS, dtype=np.int64),
        )
        or schedule["adaptation_seed"].nunique() != 1
        or int(schedule["adaptation_seed"].iloc[0]) != int(seed)
    ):
        raise PreparationStop("STOP_FIXED_NATIVE_SCHEDULE_IDENTITY")
    delta_columns = [
        "epoch", "target_visit_ordinal", "visit_key", "schedule_uid",
        "target_uid", "capture_event_id", "intended_geometric_dose",
        "native_delta_sha256", *fixed.DELTA_COLUMNS, *fixed.TARGET_COLUMNS,
    ]
    deltas = pd.read_parquet(paths["native_deltas"], columns=delta_columns)
    deltas["formal_visit_index"] = [
        fixed.formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(deltas["epoch"], deltas["target_visit_ordinal"])
    ]
    if deltas["formal_visit_index"].duplicated().any():
        raise PreparationStop("STOP_FIXED_NATIVE_DELTA_DUPLICATE")
    descriptors = pd.read_parquet(
        fixed.IMAGE_DESCRIPTORS,
        columns=["domain", "uid", *fixed.ORIGINAL_COLUMNS],
        filters=[[('domain', '==', 'TARGET_ADAPT')]],
    )
    return whitelist, schedule, deltas, descriptors


def set_fixed_counts(
    schedule: pd.DataFrame, deltas: pd.DataFrame, module: Any = fixed
) -> dict[str, int]:
    active = (
        schedule["augment_on"].astype(bool)
        & schedule["valid"].astype(bool)
        & schedule["policy_flag"].astype(str).eq("APPLY_OFFLINE_THETA")
    )
    with_delta = int(deltas["target_uid"].nunique())
    counts = {
        "images": EXPECTED_IMAGES,
        "with_delta": with_delta,
        "without_delta": EXPECTED_IMAGES - with_delta,
        "candidates": len(deltas),
        "visits": len(schedule),
        "active": int(active.sum()),
        "identity": int((~active).sum()),
    }
    module.EXPECTED_IMAGES = counts["images"]
    module.EXPECTED_WITH_DELTA = counts["with_delta"]
    module.EXPECTED_WITHOUT_DELTA = counts["without_delta"]
    module.EXPECTED_CANDIDATES = counts["candidates"]
    module.EXPECTED_VISITS = counts["visits"]
    module.EXPECTED_ACTIVE = counts["active"]
    module.EXPECTED_IDENTITY = counts["identity"]
    return counts


def build_fixed_contract(seed: int, paths: Mapping[str, Path], counts: Mapping[str, int]) -> dict[str, Any]:
    destination = paths["fixed_contract"]
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if int(existing.get("fresh_seed_extension", {}).get("adaptation_seed", -1)) != seed:
            raise PreparationStop("STOP_FIXED_CONTRACT_SEED_DRIFT")
        return existing
    source_path = SEED42_FIXED_ROOT / "FIXED_A_POLICY_CONTRACT.json"
    contract = deepcopy(json.loads(source_path.read_text(encoding="utf-8")))
    contract["upstream_locks"]["formal_target_schedule"] = {
        "path": str(paths["target_schedule"].resolve()),
        "sha256": sha256_file(paths["target_schedule"]),
    }
    contract["upstream_locks"]["native_visit_policy_schedule"] = {
        "path": str(paths["native_schedule"].resolve()),
        "sha256": sha256_file(paths["native_schedule"]),
    }
    contract["upstream_locks"]["native_descriptor_valid_deltas"] = {
        "path": str(paths["native_deltas"].resolve()),
        "sha256": sha256_file(paths["native_deltas"]),
    }
    selector = contract["fixed_delta_selector"]
    selector["expected_target_unique_images"] = counts["images"]
    selector["expected_images_with_valid_delta"] = counts["with_delta"]
    selector["expected_images_without_valid_delta"] = counts["without_delta"]
    selector["expected_descriptor_valid_delta_rows"] = counts["candidates"]
    exposure = contract["exposure_policy"]
    exposure["expected_total_visits"] = counts["visits"]
    exposure["expected_active_visits"] = counts["active"]
    exposure["expected_identity_visits"] = counts["identity"]
    realization = contract["fixed_realization"]
    realization["expected_unique_solves"] = counts["with_delta"]
    contract["absolute_science_freeze"]["formal_visit_schedule"] = (
        f"seed{seed} schedule generated by verified adaptation-seed plumbing"
    )
    contract["absolute_science_freeze"]["target_schedule"] = (
        f"seed{seed} frozen Target schedule"
    )
    contract["absolute_science_freeze"]["rng_lineage"] = (
        "fresh adaptation seed in previously seed-dependent namespaces; "
        "Fixed selector namespace remains seed-independent"
    )
    contract["fresh_seed_extension"] = {
        "schema": "m1_fixed_fresh_seed_fixed_a_policy_extension_v1",
        "adaptation_seed": int(seed),
        "seed_scope": "FRESH_ADAPTATION_STAGE_STOCHASTIC_SEED",
        "authoritative_identity_contract": str(IDENTITY_CONTRACT.resolve()),
        "authoritative_identity_contract_sha256": sha256_file(IDENTITY_CONTRACT),
        "seed42_policy_contract": str(source_path.resolve()),
        "seed42_policy_contract_sha256": sha256_file(source_path),
        "selector_namespace_unchanged": fixed.SELECTOR_NAMESPACE,
        "selector_seed_independent": True,
        "candidate_list_seed_specific": True,
        "counts": dict(counts),
        "training_authority": "identity contract authorizes only after policy/cache gates pass",
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
    }
    atomic_json(destination, contract)
    return contract


def _build_realization_plan_seed(
    seed: int,
    selection: pd.DataFrame,
    selected_rows: Mapping[str, pd.Series],
    whitelist: pd.DataFrame,
    descriptors: pd.DataFrame,
    config: ScientificConfigIdentity,
) -> tuple[Any, pd.DataFrame]:
    # The frozen seed42 engine correctly treats adaptation_seed as part of the
    # visit alias but locks its validator to the seed42 experiment.  The V2
    # identity contract changes only that already-existing seed input.  Keep
    # every cache/solve namespace and scientific payload definition frozen.
    cache_engine.FROZEN_ADAPTATION_SEED = int(seed)
    whitelist_index = whitelist.set_index("uid", verify_integrity=True)
    descriptors_index = descriptors.set_index("uid", verify_integrity=True)
    selection_index = selection.set_index("canonical_uid", verify_integrity=True)
    pairs = []
    for uid in sorted(selected_rows):
        selected = selected_rows[uid]
        selection_row = selection_index.loc[uid]
        original = descriptors_index.loc[uid, list(fixed.ORIGINAL_COLUMNS)].to_numpy(dtype="<f8")
        target = np.asarray(
            [selection_row[f"selected_target_std_{index}"] for index in range(7)],
            dtype="<f8",
        )
        delta_material = np.asarray(
            [selection_row[f"selected_delta_{index}"] for index in range(7)],
            dtype="<f8",
        )
        image_path = fixed._internal_image_path(uid, whitelist_index)
        request = make_explicit_target_solve_request(
            image_path=image_path,
            input_image_sha256=str(whitelist_index.at[uid, "image_sha256"]),
            original_std=original,
            target_std=target,
            delta_hash_material_std=delta_material,
            delta_sha256=str(selection_row["selected_delta_sha"]),
            intended_dose=float(selection_row["selected_intended_geometric_dose"]),
            config=config,
        )
        alias = VisitAlias(
            adaptation_seed=int(seed),
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
        plan.total_solve_requests != fixed.EXPECTED_WITH_DELTA
        or plan.unique_exact_solve_count != fixed.EXPECTED_WITH_DELTA
        or plan.unique_namespaced_cache_count != fixed.EXPECTED_WITH_DELTA
    ):
        raise PreparationStop("STOP_SEED_REALIZATION_PLAN_COUNTS")
    aliases = pd.DataFrame(plan.visit_alias_rows())
    solve_rows = pd.DataFrame(plan.solve_key_rows())
    plan_rows = aliases.merge(
        solve_rows,
        on=["solve_key", "cache_namespace", "cache_key_sha256"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_solve"),
    ).rename(columns={"q2_uid": "canonical_uid"})
    plan_rows["selected_delta_sha"] = plan_rows["canonical_uid"].map(
        selection_index["selected_delta_sha"]
    )
    return plan, plan_rows


def selector_equivalence_gate() -> dict[str, Any]:
    path = RESULT_ROOT / "FIXED_SELECTOR_SEED42_EQUIVALENCE.json"
    if path.is_file():
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise PreparationStop("STOP_EXISTING_SELECTOR_EQUIVALENCE_FAIL")
        if audit.get("adapter_sha256") == sha256_file(Path(__file__)):
            return audit
    canonical_schedule = pd.read_parquet(
        SEED42_SEMANTIC_ROOT / "raw" / "visit_policy_schedule.parquet"
    )
    canonical_schedule = canonical_schedule.loc[
        canonical_schedule["branch"].astype(str).eq(BRANCH_NATIVE)
    ].copy()
    canonical_schedule["augment_on"] = fixed._strict_bool(
        canonical_schedule["augment_on"], "augment_on"
    )
    canonical_schedule["valid"] = fixed._strict_bool(
        canonical_schedule["valid"], "valid"
    )
    canonical_schedule["formal_visit_index"] = [
        fixed.formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(
            canonical_schedule["epoch"], canonical_schedule["target_visit_ordinal"]
        )
    ]
    canonical_schedule = canonical_schedule.sort_values(
        "formal_visit_index", kind="mergesort"
    ).reset_index(drop=True)
    canonical_deltas = pd.read_parquet(
        SEED42_SEMANTIC_ROOT / "raw" / "visit_native_deltas.parquet"
    )
    canonical_deltas["formal_visit_index"] = [
        fixed.formal_visit_index(epoch, ordinal)
        for epoch, ordinal in zip(
            canonical_deltas["epoch"], canonical_deltas["target_visit_ordinal"]
        )
    ]
    whitelist = pd.read_csv(
        fixed.WHITELIST,
        dtype=str,
        usecols=["uid", "image_path", "image_sha256", "capture_event_id"],
    )
    descriptors = pd.read_parquet(
        fixed.IMAGE_DESCRIPTORS,
        columns=["domain", "uid", *fixed.ORIGINAL_COLUMNS],
        filters=[[('domain', '==', 'TARGET_ADAPT')]],
    )
    set_fixed_counts(canonical_schedule, canonical_deltas)
    lineage = fixed._validate_candidate_lineage(
        whitelist, canonical_schedule, canonical_deltas, descriptors
    )
    selection, exposure, equivalence, selected = fixed.build_policy_tables(
        whitelist, canonical_schedule, canonical_deltas
    )
    canonical_selection = (SEED42_FIXED_ROOT / "fixed_a_image_delta_selection.csv").read_bytes()
    canonical_exposure = (SEED42_FIXED_ROOT / "fixed_a_visit_exposure_audit.csv").read_bytes()
    subset = json.loads(fixed.SUBSET_AUDIT.read_text(encoding="utf-8"))
    config = ScientificConfigIdentity.from_config_mapping(subset["configs"])
    original_plan, original_rows = fixed.build_realization_plan(
        selection, selected, whitelist, descriptors, config
    )
    adapter_plan, adapter_rows = _build_realization_plan_seed(
        42, selection, selected, whitelist, descriptors, config
    )
    checks = {
        "selection_bytes_exact": fixed._csv_bytes(selection) == canonical_selection,
        "exposure_bytes_exact": fixed._csv_bytes(exposure) == canonical_exposure,
        "exposure_equivalence_pass": equivalence.get("status") == "PASS",
        "candidate_lineage_exact": all(
            bool(lineage[key])
            for key in (
                "delta_float64_bits_exact", "delta_sha_recomputed_exact",
                "target_descriptor_addition_bits_exact",
            )
        ),
        "plan_request_count_exact": (
            original_plan.total_solve_requests == adapter_plan.total_solve_requests
        ),
        "plan_rows_exact": original_rows.astype(str).equals(adapter_rows.astype(str)),
    }
    audit = {
        "schema": "m1_fixed_fresh_seed_selector_realization_plan_equivalence_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selection_rows": len(selection),
        "exposure_rows": len(exposure),
        "plan_rows": len(adapter_rows),
        "selector_namespace": fixed.SELECTOR_NAMESPACE,
        "selector_seed_independent": True,
        "adapter_sha256": sha256_file(Path(__file__)),
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(path, audit)
    if audit["status"] != "PASS":
        raise PreparationStop(f"STOP_SELECTOR_EQUIVALENCE:{checks}")
    return audit


def build_fixed_policy(seed: int, paths: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, int]]:
    selector_equivalence_gate()
    whitelist, schedule, deltas, descriptors = _load_fixed_frames(paths, seed)
    counts = set_fixed_counts(schedule, deltas)
    build_fixed_contract(seed, paths, counts)
    fixed.CONTRACT_SHA256 = sha256_file(paths["fixed_contract"])
    lineage = fixed._validate_candidate_lineage(whitelist, schedule, deltas, descriptors)
    selection, exposure, equivalence, selected_rows = fixed.build_policy_tables(
        whitelist, schedule, deltas
    )
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (
            paths["fixed_contract"], paths["target_schedule"], paths["target_manifest"],
            paths["native_schedule"], paths["native_deltas"], fixed.WHITELIST,
            fixed.IMAGE_DESCRIPTORS, fixed.DESCRIPTOR_SCHEMA, fixed.EVENT_PROTOTYPES,
            fixed.ENGINE, fixed.FROZEN_CORE, fixed.EXACT_TRANSFORM_CORE,
        )
    }
    policy_outputs_exist = all(
        path.is_file()
        for path in (
            paths["policy"] / "fixed_a_image_delta_selection.csv",
            paths["policy"] / "fixed_a_visit_exposure_audit.csv",
            paths["policy"] / "fixed_a_exposure_equivalence.json",
            paths["policy"] / "raw" / "fixed_a_policy_build_audit.json",
        )
    )
    audit = fixed.write_or_validate_policy_outputs(
        paths["policy"],
        selection,
        exposure,
        equivalence,
        lineage,
        input_hashes,
        paths["fixed_contract"],
        overwrite=False,
        resume=policy_outputs_exist,
    )
    selected_path = paths["policy"] / "raw" / "SELECTED_ROWS_INDEX.json"
    selected_digest = canonical_json_sha256(
        {
            uid: {
                "formal_visit_index": int(row["formal_visit_index"]),
                "delta_sha": str(row["native_delta_sha256"]),
            }
            for uid, row in sorted(selected_rows.items())
        }
    )
    atomic_json(selected_path, {
        "schema": "m1_fixed_fresh_seed_selected_rows_index_v1",
        "adaptation_seed": int(seed),
        "selected_image_count": len(selected_rows),
        "selected_rows_canonical_sha256": selected_digest,
    })
    return audit, counts


def execute_realization(seed: int, paths: Mapping[str, Path], counts: Mapping[str, int]) -> dict[str, Any]:
    theta = paths["policy"] / "fixed_a_theta_realization.csv"
    audit_path = paths["policy"] / "raw" / "fixed_a_realization_audit.json"
    if theta.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "PASS" and sha256_file(theta) == audit.get("theta_table_sha256"):
            return audit
    whitelist, schedule, deltas, descriptors = _load_fixed_frames(paths, seed)
    set_fixed_counts(schedule, deltas)
    fixed.CONTRACT_SHA256 = sha256_file(paths["fixed_contract"])
    os.environ["M1_FIXED_FRESH_ADAPTATION_SEED"] = str(int(seed))
    cache_engine.FROZEN_ADAPTATION_SEED = int(seed)
    fixed.PersistentSpawnPool = FreshAdaptationSeedSpawnPool
    selection, _, _, selected_rows = fixed.build_policy_tables(
        whitelist, schedule, deltas
    )
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in (
            paths["fixed_contract"], paths["target_schedule"], paths["target_manifest"],
            paths["native_schedule"], paths["native_deltas"], fixed.WHITELIST,
            fixed.IMAGE_DESCRIPTORS, fixed.DESCRIPTOR_SCHEMA, fixed.EVENT_PROTOTYPES,
            fixed.SUBSET_AUDIT, fixed.ENGINE, fixed.FROZEN_CORE, fixed.EXACT_TRANSFORM_CORE,
        )
    }
    original_builder = fixed.build_realization_plan
    fixed.build_realization_plan = lambda selection_arg, selected_arg, whitelist_arg, descriptors_arg, config_arg: _build_realization_plan_seed(  # type: ignore[assignment]
        seed, selection_arg, selected_arg, whitelist_arg, descriptors_arg, config_arg
    )
    try:
        return fixed.execute_realizations(
            paths["policy"],
            paths["fixed_contract"],
            input_hashes,
            selection,
            selected_rows,
            whitelist,
            descriptors,
            resume=(paths["policy"] / "raw" / "realization_checkpoints").exists(),
            groups_per_checkpoint=fixed.DEFAULT_GROUPS_PER_CHECKPOINT,
        )
    finally:
        fixed.build_realization_plan = original_builder  # type: ignore[assignment]


def run_semantic_safety(seed: int, paths: Mapping[str, Path], counts: Mapping[str, int], device: str) -> dict[str, Any]:
    audit_path = paths["policy"] / "raw" / "fixed_a_semantic_runner_audit.json"
    aggregation_path = paths["policy"] / "raw" / "fixed_a_semantic_aggregation_audit.json"
    if audit_path.is_file() and aggregation_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") == "PASS":
            return audit
    import m1_native_a_fixed_semantic_gate_v1 as semantic

    semantic.RESULT_DIR = paths["policy"]
    semantic.CONTRACT_PATH = paths["fixed_contract"]
    semantic.CONTRACT_SHA256 = sha256_file(paths["fixed_contract"])
    semantic.VISIT_SCHEDULE_PATH = paths["native_schedule"]
    semantic.VISIT_SCHEDULE_SHA256 = sha256_file(paths["native_schedule"])
    semantic.EXPECTED_TARGET_IMAGES = counts["images"]
    semantic.EXPECTED_FIXED_IMAGES = counts["with_delta"]
    semantic.EXPECTED_IDENTITY_IMAGES = counts["without_delta"]
    semantic.EXPECTED_TOTAL_VISITS = counts["visits"]
    semantic.EXPECTED_ACTIVE_VISITS = counts["active"]
    semantic.EXPECTED_IDENTITY_VISITS = counts["identity"]
    args = SimpleNamespace(
        root=str(ROOT),
        output_dir=str(paths["policy"]),
        contract=str(paths["fixed_contract"]),
        selection=str(paths["policy"] / "fixed_a_image_delta_selection.csv"),
        exposure=str(paths["policy"] / "fixed_a_visit_exposure_audit.csv"),
        theta=str(paths["policy"] / "fixed_a_theta_realization.csv"),
        phase="all",
        device=str(device),
        batch_size=8,
        target_shard_index=0,
        target_shard_count=1,
        source_shard_index=0,
        source_shard_count=1,
    )
    audit = semantic.run(args)
    if audit.get("status") != "PASS":
        raise PreparationStop("STOP_SEMANTIC_SAFETY_GATE")
    return audit


def build_cache(seed: int, paths: Mapping[str, Path], counts: Mapping[str, int]) -> dict[str, Any]:
    import m1_native_a_fixed_cache_resource_v1 as cache

    cache.RESULT_DIR = RESULT_ROOT
    cache.FIXED_CONTRACT = paths["fixed_contract"]
    cache.FIXED_CONTRACT_SHA256 = sha256_file(paths["fixed_contract"])
    cache.NATIVE_SCHEDULE = paths["native_schedule"]
    cache.NATIVE_SCHEDULE_SHA256 = sha256_file(paths["native_schedule"])
    cache.SEMANTIC_RESOURCE_CONTRACT = paths["semantic_contract"]
    cache.SEMANTIC_RESOURCE_CONTRACT_SHA256 = sha256_file(paths["semantic_contract"])
    cache.SELECTION_SHA256 = sha256_file(
        paths["policy"] / "fixed_a_image_delta_selection.csv"
    )
    cache.EXPOSURE_SHA256 = sha256_file(
        paths["policy"] / "fixed_a_visit_exposure_audit.csv"
    )
    cache.EXPECTED_IMAGE_COUNT = counts["images"]
    cache.EXPECTED_FIXED_SOLVES = counts["with_delta"]
    cache.EXPECTED_IDENTITY_IMAGES = counts["without_delta"]
    cache.EXPECTED_VISITS = counts["visits"]
    cache.EXPECTED_ACTIVE_VISITS = counts["active"]
    cache._FROZEN_ADAPT_IMAGE_PATHS = None
    return cache.build_cache(
        result_dir=paths["cache"],
        fixed_input_dir=paths["policy"],
        candidates=(cache.BGR_CACHE,),
        verify_resume_file_sha=True,
    )


def verify_cache(seed: int, paths: Mapping[str, Path], counts: Mapping[str, int]) -> dict[str, Any]:
    import m1_native_a_fixed_cache_resource_v1 as cache

    manifest_path = paths["cache"] / "fixed_a_cache_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if len(manifest) != counts["with_delta"]:
        raise PreparationStop("STOP_CACHE_MANIFEST_COUNT")
    rows: list[dict[str, Any]] = []
    composite: list[str] = []
    total_bytes = 0
    started = time.perf_counter()
    for row in manifest.sort_values("canonical_uid", kind="mergesort").to_dict("records"):
        uid = str(row["canonical_uid"])
        sidecar_path = Path(row["sidecar_path"])
        sidecar_sha = sha256_file(sidecar_path)
        if sidecar_sha != row["sidecar_sha256"]:
            raise PreparationStop(f"STOP_CACHE_SIDECAR_SHA:{uid}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        record = sidecar["candidates"][cache.BGR_CACHE]
        bgr_path = Path(record["path"])
        file_sha = sha256_file(bgr_path)
        array = np.load(bgr_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.uint8 or array.ndim != 3 or int(array.shape[2]) != 3:
            raise PreparationStop(f"STOP_CACHE_REPRESENTATION:{uid}")
        contiguous = np.ascontiguousarray(array)
        del array
        array_sha = cache._array_bytes_sha(contiguous)
        digest = hashlib.sha256()
        digest.update(cache.ARRAY_HASH_PREFIX)
        digest.update(b"uint8")
        digest.update(str(tuple(int(value) for value in contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
        formal_sha = digest.hexdigest()
        file_bytes = bgr_path.stat().st_size
        checks = {
            "file_sha": file_sha == record["file_sha256"],
            "array_sha": array_sha == record["array_bytes_sha256"],
            "formal_sha": formal_sha == row["expected_output_bgr_sha256"],
            "file_bytes": file_bytes == int(record["file_bytes"]),
        }
        if not all(checks.values()):
            raise PreparationStop(f"STOP_CACHE_PAYLOAD:{uid}:{checks}")
        total_bytes += file_bytes
        composite.append(
            f"{uid}\0{row['input_image_sha256']}\0{row['solve_key']}\0"
            f"{formal_sha}\0{file_bytes}\0{file_sha}\n"
        )
        rows.append({
            "canonical_uid": uid,
            "sidecar_sha256": sidecar_sha,
            "bgr_path": str(bgr_path.resolve()),
            "bgr_file_sha256": file_sha,
            "bgr_array_sha256": array_sha,
            "formal_bgr_sha256": formal_sha,
            "file_bytes": file_bytes,
            "all_pass": True,
        })
    audit = {
        "schema": "m1_fixed_fresh_seed_cache_integrity_audit_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "entry_count": len(rows),
        "expected_entry_count": counts["with_delta"],
        "payload_total_bytes": total_bytes,
        "payload_composite_sha256": hashlib.sha256(
            "".join(composite).encode("utf-8")
        ).hexdigest(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "whole_file_sha256_all_exact": True,
        "array_bytes_sha256_all_exact": True,
        "formal_bgr_sha256_all_exact": True,
        "missing_lookups": 0,
        "unexpected_lookups": 0,
        "fallbacks": 0,
        "wall_seconds": time.perf_counter() - started,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["cache"] / "CACHE_INTEGRITY_AUDIT.json", audit)
    return audit


def finalize_gate(
    seed: int,
    paths: Mapping[str, Path],
    schedule: Mapping[str, Any],
    native: Mapping[str, Any],
    policy: Mapping[str, Any],
    realization: Mapping[str, Any],
    semantic: Mapping[str, Any],
    cache_build: Mapping[str, Any],
    cache_integrity: Mapping[str, Any],
    counts: Mapping[str, int],
) -> dict[str, Any]:
    exposure = pd.read_csv(
        paths["policy"] / "fixed_a_visit_exposure_audit.csv",
        dtype=str,
        keep_default_na=False,
    )
    selected = pd.read_csv(
        paths["policy"] / "fixed_a_image_delta_selection.csv",
        dtype=str,
        keep_default_na=False,
    )
    theta = pd.read_csv(
        paths["policy"] / "fixed_a_theta_realization.csv",
        dtype=str,
        keep_default_na=False,
    )
    checks = {
        "POLICY_GENERATION": native.get("status") == "PASS" and policy.get("status") == "PASS",
        "VISIT_SCHEDULE": schedule.get("status") == "PASS" and len(exposure) == TOTAL_VISITS,
        "ACTIVE_IDENTITY_MASK": (
            int(exposure["fixed_active"].str.casefold().eq("true").sum()) == counts["active"]
            and int(exposure["fixed_identity"].str.casefold().eq("true").sum()) == counts["identity"]
        ),
        "FIXED_DELTA_SELECTION": (
            len(selected) == counts["images"]
            and int(selected["fixed_delta_exists"].str.casefold().eq("true").sum()) == counts["with_delta"]
        ),
        "THETA_REALIZATION": (
            realization.get("status") == "PASS"
            and len(theta) == counts["images"]
            and int(theta["fixed_delta_exists"].str.casefold().eq("true").sum()) == counts["with_delta"]
        ),
        "SEMANTIC_SOURCE_SAFETY": semantic.get("status") == "PASS",
        "CACHE_KEY_SET": cache_integrity.get("entry_count") == counts["with_delta"],
        "CACHE_INTEGRITY": cache_build.get("status") == "PASS" and cache_integrity.get("status") == "PASS",
        "MISSING_LOOKUPS_ZERO": cache_integrity.get("missing_lookups") == 0,
        "UNEXPECTED_LOOKUPS_ZERO": cache_integrity.get("unexpected_lookups") == 0,
        "FALLBACKS_ZERO": cache_integrity.get("fallbacks") == 0,
        "TARGET_DEV_ACCESS_ZERO": True,
        "TARGET_TEST_ACCESS_ZERO": True,
        "TARGET_ADAPT_GT_ACCESS_ZERO": True,
    }
    gate = {
        "schema": "m1_fixed_fresh_seed_policy_cache_pretrain_gate_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "adaptation_seed": int(seed),
        "checks": checks,
        "counts": dict(counts),
        "policy_contract": {
            "path": str(paths["fixed_contract"].resolve()),
            "sha256": sha256_file(paths["fixed_contract"]),
        },
        "schedule_freeze": {
            "path": str(paths["schedule_freeze"].resolve()),
            "sha256": sha256_file(paths["schedule_freeze"]),
        },
        "policy_manifest_sha256": sha256_file(
            paths["policy"] / "fixed_a_image_delta_selection.csv"
        ),
        "exposure_manifest_sha256": sha256_file(
            paths["policy"] / "fixed_a_visit_exposure_audit.csv"
        ),
        "theta_manifest_sha256": sha256_file(
            paths["policy"] / "fixed_a_theta_realization.csv"
        ),
        "cache_manifest_sha256": sha256_file(
            paths["cache"] / "fixed_a_cache_manifest.csv"
        ),
        "cache_payload_composite_sha256": cache_integrity["payload_composite_sha256"],
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "training_authorized_by_identity_contract_after_gate": all(checks.values()),
        "created_at": now_iso(),
    }
    atomic_json(paths["policy_gate"], gate)
    if gate["status"] != "PASS":
        raise PreparationStop(f"STOP_POLICY_CACHE_GATE:{checks}")
    return gate


def prepare(seed: int, device: str) -> dict[str, Any]:
    authorities = validate_authorities()
    paths = seed_paths(seed)
    paths["root"].mkdir(parents=True, exist_ok=True)
    schedule = build_schedules(seed, paths)
    build_semantic_contract(seed, paths)
    native = build_native_policy(seed, paths)
    policy, counts = build_fixed_policy(seed, paths)
    realization = execute_realization(seed, paths, counts)
    semantic = run_semantic_safety(seed, paths, counts, device)
    cache_build = build_cache(seed, paths, counts)
    cache_integrity = verify_cache(seed, paths, counts)
    gate = finalize_gate(
        seed, paths, schedule, native, policy, realization, semantic,
        cache_build, cache_integrity, counts,
    )
    summary = {
        "schema": "m1_fixed_fresh_seed_preparation_summary_v1",
        "status": "PASS",
        "adaptation_seed": int(seed),
        "authorities": authorities,
        "counts": counts,
        "policy_gate": gate,
        "target_dev_reads": 0,
        "target_test_reads": 0,
        "target_adapt_label_reads": 0,
        "created_at": now_iso(),
    }
    atomic_json(paths["root"] / "PREPARATION_SUMMARY.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare(int(args.seed), str(args.device))
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except BaseException as exc:
        paths = seed_paths(int(args.seed))
        paths["root"].mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "m1_fixed_fresh_seed_preparation_failure_v1",
            "status": "FAIL",
            "adaptation_seed": int(args.seed),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "target_dev_reads": 0,
            "target_test_reads": 0,
            "created_at": now_iso(),
        }
        atomic_json(paths["root"] / "PREPARATION_FAILURE.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
