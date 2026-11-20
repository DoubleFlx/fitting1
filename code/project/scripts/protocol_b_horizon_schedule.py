from __future__ import annotations

"""One-shot, seed-namespaced image-only schedules for e71-e105."""

import json
import random
from collections import Counter, namedtuple
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from protocol_b_horizon_common import (
    ALL_TRAIN_EPOCHS,
    CORE_EPOCHS,
    DATASET,
    EXPECTED_DATASET_MANIFEST,
    EXPECTED_SOURCE_BATCH,
    EXPECTED_SOURCE_VISITS,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_TARGET_BATCH,
    EXPECTED_TARGET_POOL,
    EXPECTED_TARGET_VISITS,
    TAIL_EPOCHS,
    atomic_json,
    atomic_jsonl,
    canonical_json_sha256,
    require_seed,
    schedule_paths,
    sha256_file,
)
from protocol_b_schedule import ResettableProtocolBDataLoader, ScheduledSourceDataset
from protocol_b_target_schedule import (
    ProtocolBTargetDataset,
    ResettableProtocolBTargetDataLoader,
    collate_target_batch,
    derive_seed,
    resolve_target_adapt_root,
    scan_target_pool,
    sha256_lines,
    target_pool_metadata_hash,
)


SourceToken = namedtuple(
    "SourceToken",
    ["dataset_index", "aug_seed", "uid", "total_epoch", "batch_index", "position"],
)
TargetToken = namedtuple(
    "TargetToken", ["uid", "view_seed", "total_epoch", "batch_index", "position"]
)


class HorizonSourceSchedule:
    def __init__(
        self,
        schedule_path: str | Path,
        augmentation_path: str | Path,
        *,
        expected_epochs: Sequence[int],
        adaptation_seed: int,
    ) -> None:
        self.schedule_path = Path(schedule_path).resolve()
        self.augmentation_path = Path(augmentation_path).resolve()
        self.expected_epochs = tuple(int(value) for value in expected_epochs)
        self.adaptation_seed = require_seed(adaptation_seed)
        schedule_rows = [
            json.loads(line)
            for line in self.schedule_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        augmentation_rows = [
            json.loads(line)
            for line in self.augmentation_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        augmentation_by_key = {row["key"]: row for row in augmentation_rows}
        if len(augmentation_by_key) != len(augmentation_rows):
            raise RuntimeError("Duplicate Source augmentation schedule keys")
        self.epochs: dict[int, list[dict[str, Any]]] = {}
        for row in schedule_rows:
            augmentation = augmentation_by_key.get(row["key"])
            if augmentation is None:
                raise KeyError(f"Missing Source augmentation row: {row['key']}")
            if int(row.get("adaptation_seed", -1)) != self.adaptation_seed:
                raise RuntimeError(f"Source schedule seed drift: {row['key']}")
            if int(augmentation.get("adaptation_seed", -1)) != self.adaptation_seed:
                raise RuntimeError(f"Source augmentation seed drift: {row['key']}")
            if len(row["uids"]) != len(augmentation["sample_aug_seeds"]):
                raise RuntimeError(f"Source UID/augmentation count drift: {row['key']}")
            merged = dict(row)
            merged["sample_aug_seeds"] = [
                int(value) for value in augmentation["sample_aug_seeds"]
            ]
            self.epochs.setdefault(int(row["epoch"]), []).append(merged)
        if tuple(sorted(self.epochs)) != self.expected_epochs:
            raise RuntimeError(f"Source epochs drift: {sorted(self.epochs)}")
        for epoch, rows in self.epochs.items():
            rows.sort(key=lambda row: int(row["batch"]))
            if len(rows) != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"Source e{epoch} row count={len(rows)}")
            if [int(row["batch"]) for row in rows] != list(range(EXPECTED_STEPS_PER_EPOCH)):
                raise RuntimeError(f"Source e{epoch} batch indices incomplete")
            uids = [uid for row in rows for uid in row["uids"]]
            if len(uids) != EXPECTED_SOURCE_VISITS or len(set(uids)) != EXPECTED_SOURCE_VISITS:
                raise RuntimeError(f"Source e{epoch} coverage drift")

    def rows_for_epoch(self, total_epoch: int) -> list[dict[str, Any]]:
        try:
            return self.epochs[int(total_epoch)]
        except KeyError as exc:
            raise KeyError(f"No Source schedule for e{total_epoch}") from exc


class HorizonSourceBatchSampler(Sampler[list[SourceToken]]):
    def __init__(
        self,
        schedule: HorizonSourceSchedule,
        uid_to_index: dict[str, int],
        total_epoch: int,
    ) -> None:
        self.schedule = schedule
        self.uid_to_index = uid_to_index
        self.total_epoch = int(total_epoch)
        scheduled = {
            uid for rows in schedule.epochs.values() for row in rows for uid in row["uids"]
        }
        missing = sorted(scheduled - set(uid_to_index))
        if missing:
            raise KeyError(f"Scheduled Source UID missing from dataset: {missing[:20]}")

    def set_total_epoch(self, total_epoch: int) -> None:
        self.schedule.rows_for_epoch(total_epoch)
        self.total_epoch = int(total_epoch)

    def __iter__(self) -> Iterator[list[SourceToken]]:
        for row in self.schedule.rows_for_epoch(self.total_epoch):
            yield [
                SourceToken(
                    self.uid_to_index[uid],
                    int(seed),
                    uid,
                    self.total_epoch,
                    int(row["batch"]),
                    position,
                )
                for position, (uid, seed) in enumerate(
                    zip(row["uids"], row["sample_aug_seeds"], strict=True)
                )
            ]

    def __len__(self) -> int:
        return len(self.schedule.rows_for_epoch(self.total_epoch))


def build_horizon_source_loader(
    base_dataset: Dataset,
    schedule: HorizonSourceSchedule,
    *,
    workers: int,
    total_epoch: int,
    adaptation_seed: int,
) -> ResettableProtocolBDataLoader:
    adaptation_seed = require_seed(adaptation_seed)
    uid_to_index: dict[str, int] = {}
    for index, image_path in enumerate(base_dataset.im_files):
        uid = Path(image_path).stem
        if uid in uid_to_index:
            raise RuntimeError(f"Duplicate Source UID: {uid}")
        uid_to_index[uid] = index
    wrapped = ScheduledSourceDataset(base_dataset)
    sampler = HorizonSourceBatchSampler(schedule, uid_to_index, total_epoch)
    generator = torch.Generator()
    generator.manual_seed(adaptation_seed)
    return ResettableProtocolBDataLoader(
        wrapped,
        batch_sampler=sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=base_dataset.collate_fn,
        generator=generator,
    )


class HorizonTargetSchedule:
    def __init__(
        self,
        schedule_path: str | Path,
        manifest_path: str | Path,
        *,
        expected_epochs: Sequence[int],
        adaptation_seed: int,
    ) -> None:
        self.schedule_path = Path(schedule_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.expected_epochs = tuple(int(value) for value in expected_epochs)
        self.adaptation_seed = require_seed(adaptation_seed)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        if self.manifest.get("dataset_manifest") != EXPECTED_DATASET_MANIFEST:
            raise RuntimeError("Target schedule dataset manifest drift")
        if int(self.manifest.get("adaptation_seed", -1)) != self.adaptation_seed:
            raise RuntimeError("Target schedule seed drift")
        if int(self.manifest.get("target_pool_size", -1)) != EXPECTED_TARGET_POOL:
            raise RuntimeError("Target schedule pool-size drift")
        if self.manifest.get("view_policy") != "same_letterbox_view_v1":
            raise RuntimeError("Target view-policy drift")
        if sha256_file(self.schedule_path) != self.manifest.get("schedule_sha256"):
            raise RuntimeError("Target schedule SHA drift")
        self.epochs: dict[int, list[dict[str, Any]]] = {}
        for line in self.schedule_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("adaptation_seed", -1)) != self.adaptation_seed:
                raise RuntimeError(f"Target row seed drift: {row.get('key')}")
            if len(row["uids"]) != EXPECTED_TARGET_BATCH:
                raise RuntimeError(f"Target batch-size drift: {row.get('key')}")
            if len(row["sample_view_seeds"]) != EXPECTED_TARGET_BATCH:
                raise RuntimeError(f"Target view-seed count drift: {row.get('key')}")
            self.epochs.setdefault(int(row["epoch"]), []).append(row)
        if tuple(sorted(self.epochs)) != self.expected_epochs:
            raise RuntimeError(f"Target epochs drift: {sorted(self.epochs)}")
        for epoch, rows in self.epochs.items():
            rows.sort(key=lambda row: int(row["batch"]))
            if len(rows) != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"Target e{epoch} row count={len(rows)}")
            if [int(row["batch"]) for row in rows] != list(range(EXPECTED_STEPS_PER_EPOCH)):
                raise RuntimeError(f"Target e{epoch} batch indices incomplete")
            uids = [uid for row in rows for uid in row["uids"]]
            if len(uids) != EXPECTED_TARGET_VISITS or len(set(uids)) != EXPECTED_TARGET_POOL:
                raise RuntimeError(f"Target e{epoch} coverage drift")

    def rows_for_epoch(self, total_epoch: int) -> list[dict[str, Any]]:
        try:
            return self.epochs[int(total_epoch)]
        except KeyError as exc:
            raise KeyError(f"No Target schedule for e{total_epoch}") from exc


class HorizonTargetBatchSampler(Sampler[list[TargetToken]]):
    def __init__(self, schedule: HorizonTargetSchedule, total_epoch: int) -> None:
        self.schedule = schedule
        self.total_epoch = int(total_epoch)

    def set_total_epoch(self, total_epoch: int) -> None:
        self.schedule.rows_for_epoch(total_epoch)
        self.total_epoch = int(total_epoch)

    def __iter__(self) -> Iterator[list[TargetToken]]:
        for row in self.schedule.rows_for_epoch(self.total_epoch):
            yield [
                TargetToken(uid, int(seed), self.total_epoch, int(row["batch"]), position)
                for position, (uid, seed) in enumerate(
                    zip(row["uids"], row["sample_view_seeds"], strict=True)
                )
            ]

    def __len__(self) -> int:
        return len(self.schedule.rows_for_epoch(self.total_epoch))


def build_horizon_target_loader(
    *,
    target_yaml: str | Path,
    schedule: HorizonTargetSchedule,
    workers: int,
    imgsz: int,
    total_epoch: int,
    adaptation_seed: int,
) -> ResettableProtocolBTargetDataLoader:
    adaptation_seed = require_seed(adaptation_seed)
    _, _, image_root = resolve_target_adapt_root(target_yaml)
    uids, uid_to_path = scan_target_pool(image_root)
    scheduled = {
        uid for rows in schedule.epochs.values() for row in rows for uid in row["uids"]
    }
    if set(uids) != scheduled:
        raise RuntimeError("Target image-only pool/schedule membership drift")
    dataset = ProtocolBTargetDataset(image_root, uid_to_path, imgsz=int(imgsz))
    sampler = HorizonTargetBatchSampler(schedule, total_epoch)
    generator = torch.Generator()
    generator.manual_seed(adaptation_seed + 1000)
    return ResettableProtocolBTargetDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_target_batch,
        generator=generator,
    )


def _source_rows(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = DATASET / "source" / "train" / "images"
    paths = sorted(path for path in source_root.iterdir() if path.is_file())
    uids = [path.stem for path in paths]
    if len(uids) != EXPECTED_SOURCE_VISITS or len(set(uids)) != EXPECTED_SOURCE_VISITS:
        raise RuntimeError(f"Source Train pool drift: {len(uids)}")
    schedule_rows: list[dict[str, Any]] = []
    augmentation_rows: list[dict[str, Any]] = []
    namespace = f"protocol-b-source-augmentation-v1|adaptation-seed={seed}"
    for epoch in ALL_TRAIN_EPOCHS:
        epoch_seed = seed + epoch
        rng = random.Random(epoch_seed)
        epoch_uids = list(uids)
        rng.shuffle(epoch_uids)
        for batch in range(EXPECTED_STEPS_PER_EPOCH):
            start = batch * EXPECTED_SOURCE_BATCH
            batch_uids = epoch_uids[start : start + EXPECTED_SOURCE_BATCH]
            key = f"e{epoch:03d}_b{batch:04d}"
            schedule_rows.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "key": key,
                    "n_samples": len(batch_uids),
                    "is_last_batch": batch == EXPECTED_STEPS_PER_EPOCH - 1,
                    "uids": batch_uids,
                    "adaptation_seed": seed,
                    "source_schedule_seed": seed,
                    "epoch_seed": epoch_seed,
                }
            )
            augmentation_rows.append(
                {
                    "epoch": epoch,
                    "batch": batch,
                    "key": key,
                    "sample_aug_seeds": [
                        rng.randint(0, 2**31 - 1) for _ in range(len(batch_uids))
                    ],
                    "adaptation_seed": seed,
                    "epoch_rng_seed": epoch_seed,
                    "augmentation_rng_namespace": namespace,
                }
            )
    return schedule_rows, augmentation_rows


def _target_rows(seed: int, uids: list[str]) -> list[dict[str, Any]]:
    permutation_namespace = f"protocol-b-target-permutation-v1|adaptation-seed={seed}"
    view_namespace = f"protocol-b-target-same-view-v1|adaptation-seed={seed}"
    rows: list[dict[str, Any]] = []
    for epoch in ALL_TRAIN_EPOCHS:
        stream: list[str] = []
        cycle = 0
        while len(stream) < EXPECTED_TARGET_VISITS:
            rng = np.random.default_rng(derive_seed(permutation_namespace, epoch, cycle))
            stream.extend(uids[int(index)] for index in rng.permutation(len(uids)))
            cycle += 1
        stream = stream[:EXPECTED_TARGET_VISITS]
        for batch in range(EXPECTED_STEPS_PER_EPOCH):
            start = batch * EXPECTED_TARGET_BATCH
            batch_uids = stream[start : start + EXPECTED_TARGET_BATCH]
            rows.append(
                {
                    "key": f"e{epoch:03d}_b{batch:03d}",
                    "epoch": epoch,
                    "batch": batch,
                    "uids": batch_uids,
                    "sample_view_seeds": [
                        derive_seed(view_namespace, epoch, batch, position, uid)
                        for position, uid in enumerate(batch_uids)
                    ],
                    "adaptation_seed": seed,
                    "target_schedule_seed": seed,
                }
            )
    return rows


def _segment(rows: list[dict[str, Any]], epochs: Sequence[int]) -> list[dict[str, Any]]:
    wanted = set(int(value) for value in epochs)
    return [row for row in rows if int(row["epoch"]) in wanted]


def generate_and_freeze_schedules(seed: int) -> dict[str, Any]:
    """Generate all e71-e105 schedules in one call without labels or Target Dev/Test."""

    seed = require_seed(seed)
    paths = schedule_paths(seed)
    for key, path in paths.items():
        if key == "directory":
            continue
        if path.exists():
            raise FileExistsError(f"Frozen schedule target already exists: {path}")
    marker = (DATASET / "DATASET_MANIFEST_SHA256.txt").read_text(
        encoding="utf-8"
    ).strip().split()[0]
    if marker != EXPECTED_DATASET_MANIFEST:
        raise RuntimeError(f"Dataset manifest drift: {marker}")

    source_rows, augmentation_rows = _source_rows(seed)
    target_yaml = DATASET / "configs" / "target_adapt_unlabeled.yaml"
    _, dataset_root, target_image_root = resolve_target_adapt_root(target_yaml)
    if dataset_root.resolve() != DATASET.resolve():
        raise RuntimeError("Target image-only YAML root drift")
    target_uids, uid_to_path = scan_target_pool(target_image_root)
    target_rows = _target_rows(seed, target_uids)

    source_core = _segment(source_rows, CORE_EPOCHS)
    augmentation_core = _segment(augmentation_rows, CORE_EPOCHS)
    source_tail = _segment(source_rows, TAIL_EPOCHS)
    augmentation_tail = _segment(augmentation_rows, TAIL_EPOCHS)
    target_core = _segment(target_rows, CORE_EPOCHS)
    target_tail = _segment(target_rows, TAIL_EPOCHS)
    atomic_jsonl(paths["source_core"], source_core)
    atomic_jsonl(paths["source_core_augmentation"], augmentation_core)
    atomic_jsonl(paths["source_tail"], source_tail)
    atomic_jsonl(paths["source_tail_augmentation"], augmentation_tail)
    atomic_jsonl(paths["target_core"], target_core)
    atomic_jsonl(paths["target_tail"], target_tail)

    common_target = {
        "schema": "b2_generic_horizon_target_schedule_segment_v1",
        "experiment": "B2_GENERIC_HORIZON_CONFIRMATION_E100_VS_E105_V1",
        "dataset_manifest": marker,
        "dataset_root": str(DATASET.resolve()),
        "target_yaml": str(target_yaml.resolve()),
        "target_yaml_sha256": sha256_file(target_yaml),
        "target_image_root": str(target_image_root.resolve()),
        "target_pool_size": len(target_uids),
        "target_uid_policy": "relative_path_from_target_adapt_images_with_suffix",
        "target_uid_pool_sha256": sha256_lines(sorted(target_uids)),
        "target_pool_metadata_sha256": target_pool_metadata_hash(target_uids, uid_to_path),
        "adaptation_seed": seed,
        "target_schedule_seed": seed,
        "target_permutation_rng_namespace": f"protocol-b-target-permutation-v1|adaptation-seed={seed}",
        "target_view_rng_namespace": f"protocol-b-target-same-view-v1|adaptation-seed={seed}",
        "target_dataloader_generator_seed": seed + 1000,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "target_batch": EXPECTED_TARGET_BATCH,
        "visits_per_scheduled_epoch": EXPECTED_TARGET_VISITS,
        "sampling_policy": "seed-namespaced epoch permutations; full Target Adapt image-only coverage per epoch",
        "view_policy": "same_letterbox_view_v1",
        "letterbox": {
            "new_shape": [640, 640],
            "auto": False,
            "scale_fill": False,
            "scaleup": True,
            "center": True,
            "stride": 32,
            "padding_value": 114,
            "interpolation": "cv2.INTER_LINEAR",
            "color_order_after_format": "RGB",
        },
        "target_adapt_manifest_read_count": 0,
        "target_adapt_label_read_count": 0,
        "target_dev_read_count": 0,
        "target_test_read_count": 0,
    }
    core_target_manifest = {
        **common_target,
        "epochs": list(CORE_EPOCHS),
        "burn_in_policy": "e71 schedule frozen but not read while lambda_u=0",
        "schedule_path": str(paths["target_core"].resolve()),
        "schedule_sha256": sha256_file(paths["target_core"]),
        "row_count": len(target_core),
    }
    tail_target_manifest = {
        **common_target,
        "epochs": list(TAIL_EPOCHS),
        "burn_in_policy": None,
        "schedule_path": str(paths["target_tail"].resolve()),
        "schedule_sha256": sha256_file(paths["target_tail"]),
        "row_count": len(target_tail),
    }
    atomic_json(paths["target_core_manifest"], core_target_manifest)
    atomic_json(paths["target_tail_manifest"], tail_target_manifest)

    per_epoch: list[dict[str, Any]] = []
    for epoch in ALL_TRAIN_EPOCHS:
        source_epoch = [row for row in source_rows if int(row["epoch"]) == epoch]
        augmentation_epoch = [
            row for row in augmentation_rows if int(row["epoch"]) == epoch
        ]
        target_epoch = [row for row in target_rows if int(row["epoch"]) == epoch]
        per_epoch.append(
            {
                "seed": seed,
                "epoch": epoch,
                "source_content_sha256": canonical_json_sha256(source_epoch),
                "source_augmentation_content_sha256": canonical_json_sha256(
                    augmentation_epoch
                ),
                "target_content_sha256": canonical_json_sha256(target_epoch),
                "source_visits": sum(len(row["uids"]) for row in source_epoch),
                "source_unique_uid_count": len(
                    {uid for row in source_epoch for uid in row["uids"]}
                ),
                "target_visits": sum(len(row["uids"]) for row in target_epoch),
                "target_unique_uid_count": len(
                    {uid for row in target_epoch for uid in row["uids"]}
                ),
                "target_visit_count_min": min(
                    Counter(uid for row in target_epoch for uid in row["uids"]).values()
                ),
                "target_visit_count_max": max(
                    Counter(uid for row in target_epoch for uid in row["uids"]).values()
                ),
            }
        )
    source_manifest = {
        "schema": "b2_generic_horizon_source_schedule_e071_e105_manifest_v1",
        "adaptation_seed": seed,
        "source_schedule_seed": seed,
        "source_epoch_seed_policy": "source_schedule_seed + total_epoch",
        "source_dataloader_generator_seed": seed,
        "augmentation_rng_namespace": f"protocol-b-source-augmentation-v1|adaptation-seed={seed}",
        "epochs": list(ALL_TRAIN_EPOCHS),
        "generated_in_one_operation": True,
        "dataset_manifest": marker,
        "segments": {
            "core_schedule": {
                "path": str(paths["source_core"].resolve()),
                "sha256": sha256_file(paths["source_core"]),
            },
            "core_augmentation": {
                "path": str(paths["source_core_augmentation"].resolve()),
                "sha256": sha256_file(paths["source_core_augmentation"]),
            },
            "tail_schedule": {
                "path": str(paths["source_tail"].resolve()),
                "sha256": sha256_file(paths["source_tail"]),
            },
            "tail_augmentation": {
                "path": str(paths["source_tail_augmentation"].resolve()),
                "sha256": sha256_file(paths["source_tail_augmentation"]),
            },
        },
        "per_epoch": per_epoch,
        "target_dev_read_count": 0,
        "target_test_read_count": 0,
    }
    target_manifest = {
        "schema": "b2_generic_horizon_target_schedule_e071_e105_manifest_v1",
        "adaptation_seed": seed,
        "target_schedule_seed": seed,
        "target_permutation_rng_namespace": common_target[
            "target_permutation_rng_namespace"
        ],
        "target_view_rng_namespace": common_target["target_view_rng_namespace"],
        "target_dataloader_generator_seed": seed + 1000,
        "epochs": list(ALL_TRAIN_EPOCHS),
        "generated_in_one_operation": True,
        "dataset_manifest": marker,
        "target_uid_pool_sha256": common_target["target_uid_pool_sha256"],
        "target_pool_metadata_sha256": common_target[
            "target_pool_metadata_sha256"
        ],
        "segments": {
            "core": {
                "path": str(paths["target_core"].resolve()),
                "sha256": sha256_file(paths["target_core"]),
                "manifest_path": str(paths["target_core_manifest"].resolve()),
                "manifest_sha256": sha256_file(paths["target_core_manifest"]),
            },
            "tail": {
                "path": str(paths["target_tail"].resolve()),
                "sha256": sha256_file(paths["target_tail"]),
                "manifest_path": str(paths["target_tail_manifest"].resolve()),
                "manifest_sha256": sha256_file(paths["target_tail_manifest"]),
            },
        },
        "per_epoch": per_epoch,
        "target_adapt_manifest_read_count": 0,
        "target_adapt_label_read_count": 0,
        "target_dev_read_count": 0,
        "target_test_read_count": 0,
    }
    atomic_json(paths["source_manifest"], source_manifest)
    atomic_json(paths["target_manifest"], target_manifest)
    freeze = {
        "schema": "b2_generic_horizon_schedule_freeze_v1",
        "experiment": "B2_GENERIC_HORIZON_CONFIRMATION_E100_VS_E105_V1",
        "adaptation_seed": seed,
        "epochs": list(ALL_TRAIN_EPOCHS),
        "generated_before_resource_gate_or_training": True,
        "generated_in_one_operation": True,
        "files": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in paths.items()
            if key not in {"directory", "freeze"}
        },
        "source_schedule_seed": seed,
        "target_schedule_seed": seed,
        "augmentation_rng_namespace": f"protocol-b-source-augmentation-v1|adaptation-seed={seed}",
        "source_dataloader_generator_seed": seed,
        "target_dataloader_generator_seed": seed + 1000,
        "target_loader_image_only": True,
        "target_adapt_label_reads": 0,
        "target_dev_reads": 0,
        "target_test_reads": 0,
    }
    atomic_json(paths["freeze"], freeze)

    HorizonSourceSchedule(
        paths["source_core"],
        paths["source_core_augmentation"],
        expected_epochs=CORE_EPOCHS,
        adaptation_seed=seed,
    )
    HorizonSourceSchedule(
        paths["source_tail"],
        paths["source_tail_augmentation"],
        expected_epochs=TAIL_EPOCHS,
        adaptation_seed=seed,
    )
    HorizonTargetSchedule(
        paths["target_core"],
        paths["target_core_manifest"],
        expected_epochs=CORE_EPOCHS,
        adaptation_seed=seed,
    )
    HorizonTargetSchedule(
        paths["target_tail"],
        paths["target_tail_manifest"],
        expected_epochs=TAIL_EPOCHS,
        adaptation_seed=seed,
    )
    return {**freeze, "freeze_sha256": sha256_file(paths["freeze"]), "per_epoch": per_epoch}


__all__ = [
    "HorizonSourceSchedule",
    "HorizonTargetSchedule",
    "build_horizon_source_loader",
    "build_horizon_target_loader",
    "generate_and_freeze_schedules",
]
