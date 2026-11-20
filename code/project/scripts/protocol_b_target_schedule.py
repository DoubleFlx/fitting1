from __future__ import annotations

import hashlib
import io
import json
import os
import random
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Sampler
from ultralytics.data.augment import LetterBox

EXPECTED_DATASET_MANIFEST = "bd4f1f97eb6cbd37e14cbb2ab63b4301f7833fc1984b67543dfa86088962b249"
EXPECTED_TARGET_POOL_SIZE = 3272
EXPECTED_TOTAL_EPOCHS = tuple(range(71, 101))
EXPECTED_STEPS_PER_EPOCH = 552
EXPECTED_TARGET_BATCH = 8
EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH = EXPECTED_STEPS_PER_EPOCH * EXPECTED_TARGET_BATCH
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

TargetScheduleToken = namedtuple(
    "TargetScheduleToken",
    ["uid", "view_seed", "total_epoch", "batch_index", "position"],
)


def read_target_image_readonly(image_path: str | Path) -> tuple[np.ndarray, str]:
    """Decode one Target Adapt image without modifying the dataset.

    Policy:
    1. Use OpenCV's normal decoder.
    2. If OpenCV returns None, use a read-only Pillow fallback with
       LOAD_TRUNCATED_IMAGES enabled for this call only.
    3. Return a BGR uint8 array so downstream preprocessing remains unchanged.

    The fallback is deterministic for fixed file bytes and is shared by B2/B3.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Target Adapt image is missing: {image_path}")
    if image_path.stat().st_size <= 0:
        raise RuntimeError(f"Target Adapt image is empty: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"OpenCV returned an invalid Target Adapt image shape "
                f"{tuple(image.shape)} for {image_path}"
            )
        return np.ascontiguousarray(image), "cv2"

    raw = image_path.read_bytes()
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(io.BytesIO(raw)) as pil_image:
            pil_image.load()
            rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(
            f"Both OpenCV and Pillow failed to decode Target Adapt image: "
            f"{image_path}; size={len(raw)} bytes; pillow={type(exc).__name__}: {exc}"
        ) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(
            f"Pillow fallback returned an invalid Target Adapt image shape "
            f"{tuple(rgb.shape)} for {image_path}"
        )
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return bgr, "pillow_truncated_readonly"


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_seed(namespace: str, *parts: Any) -> int:
    text = "|".join([namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little") & 0x7FFFFFFF


def resolve_target_adapt_root(target_yaml: str | Path) -> tuple[dict[str, Any], Path, Path]:
    target_yaml = Path(target_yaml)
    if not target_yaml.is_file():
        raise FileNotFoundError(target_yaml)
    data = yaml.safe_load(target_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid Target Adapt YAML: {target_yaml}")
    if int(data.get("nc", -1)) != 1:
        raise RuntimeError(f"Target Adapt must use nc=1, got {data.get('nc')}")
    dataset_root = Path(str(data["path"]))
    image_root = dataset_root / str(data["train"])
    normalized = str(image_root).replace("\\", "/").lower()
    if "/target/adapt/images" not in normalized:
        raise RuntimeError(f"Target YAML must resolve only target/adapt/images, got {image_root}")
    if "/target/dev/" in normalized or "/target/test/" in normalized:
        raise RuntimeError(f"Target Dev/Test is forbidden in training: {image_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    return data, dataset_root, image_root


def scan_target_pool(image_root: Path) -> tuple[list[str], dict[str, Path]]:
    paths = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    uid_to_path: dict[str, Path] = {}
    for path in paths:
        uid = path.relative_to(image_root).as_posix()
        if uid in uid_to_path:
            raise RuntimeError(f"Duplicate Target Adapt UID: {uid}")
        uid_to_path[uid] = path
    if len(uid_to_path) != EXPECTED_TARGET_POOL_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGET_POOL_SIZE} Target Adapt images, found {len(uid_to_path)} in {image_root}"
        )
    return list(uid_to_path), uid_to_path


def target_pool_metadata_hash(uids: list[str], uid_to_path: dict[str, Path]) -> str:
    lines = []
    for uid in sorted(uids):
        stat = uid_to_path[uid].stat()
        lines.append(f"{uid}\t{stat.st_size}")
    return sha256_lines(lines)


def generate_target_schedule(
    *,
    target_yaml: str | Path,
    dataset_manifest_path: str | Path,
    output_schedule: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    target_yaml = Path(target_yaml)
    dataset_manifest_path = Path(dataset_manifest_path)
    output_schedule = Path(output_schedule)
    output_manifest = Path(output_manifest)

    dataset_manifest = dataset_manifest_path.read_text(encoding="utf-8").strip().split()[0]
    if dataset_manifest != EXPECTED_DATASET_MANIFEST:
        raise RuntimeError(f"Dataset manifest mismatch: {dataset_manifest}")

    _, dataset_root, image_root = resolve_target_adapt_root(target_yaml)
    uids, uid_to_path = scan_target_pool(image_root)
    uid_pool_sha256 = sha256_lines(sorted(uids))
    metadata_sha256 = target_pool_metadata_hash(uids, uid_to_path)

    rows: list[dict[str, Any]] = []
    visits = EXPECTED_TARGET_VISITS_PER_ACTIVE_EPOCH
    for total_epoch in EXPECTED_TOTAL_EPOCHS:
        stream: list[str] = []
        cycle = 0
        while len(stream) < visits:
            rng = np.random.default_rng(derive_seed("protocol-b-target-permutation-v1", total_epoch, cycle))
            permutation = [uids[int(index)] for index in rng.permutation(len(uids))]
            stream.extend(permutation)
            cycle += 1
        stream = stream[:visits]

        for batch_index in range(EXPECTED_STEPS_PER_EPOCH):
            start = batch_index * EXPECTED_TARGET_BATCH
            batch_uids = stream[start : start + EXPECTED_TARGET_BATCH]
            view_seeds = [
                derive_seed("protocol-b-target-same-view-v1", total_epoch, batch_index, position, uid)
                for position, uid in enumerate(batch_uids)
            ]
            rows.append(
                {
                    "key": f"e{total_epoch:03d}_b{batch_index:03d}",
                    "epoch": total_epoch,
                    "batch": batch_index,
                    "uids": batch_uids,
                    "sample_view_seeds": view_seeds,
                }
            )

    output_schedule.parent.mkdir(parents=True, exist_ok=True)
    temp_schedule = output_schedule.with_name(output_schedule.name + ".tmp")
    with temp_schedule.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_schedule, output_schedule)

    schedule_sha256 = sha256_file(output_schedule)
    manifest = {
        "schema": "protocol_b_target_schedule_manifest_v1",
        "dataset_manifest": dataset_manifest,
        "dataset_root": str(dataset_root),
        "target_yaml": str(target_yaml.resolve()),
        "target_yaml_sha256": sha256_file(target_yaml),
        "target_image_root": str(image_root),
        "target_pool_size": len(uids),
        "target_uid_policy": "relative_path_from_target_adapt_images_with_suffix",
        "target_uid_pool_sha256": uid_pool_sha256,
        "target_pool_metadata_sha256": metadata_sha256,
        "epochs": list(EXPECTED_TOTAL_EPOCHS),
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "target_batch": EXPECTED_TARGET_BATCH,
        "visits_per_scheduled_epoch": visits,
        "sampling_policy": (
            "epoch-specific deterministic permutations; every image appears at least once per epoch; "
            "a second independent permutation supplies the remaining visits"
        ),
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
        "burn_in_policy": "e71 schedule exists but B2/B3 must not read Target Adapt while lambda_u=0",
        "schedule_path": str(output_schedule.resolve()),
        "schedule_sha256": schedule_sha256,
        "row_count": len(rows),
    }
    temp_manifest = output_manifest.with_name(output_manifest.name + ".tmp")
    temp_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_manifest, output_manifest)
    return manifest


class ProtocolBTargetSchedule:
    def __init__(self, schedule_path: str | Path, manifest_path: str | Path) -> None:
        self.schedule_path = Path(schedule_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("dataset_manifest") != EXPECTED_DATASET_MANIFEST:
            raise RuntimeError("Target schedule dataset manifest mismatch")
        if int(self.manifest.get("target_pool_size", -1)) != EXPECTED_TARGET_POOL_SIZE:
            raise RuntimeError("Target schedule pool-size mismatch")
        if self.manifest.get("view_policy") != "same_letterbox_view_v1":
            raise RuntimeError(f"Unexpected Target view policy: {self.manifest.get('view_policy')}")
        actual_sha = sha256_file(self.schedule_path)
        if actual_sha != self.manifest.get("schedule_sha256"):
            raise RuntimeError(
                f"Target schedule SHA256 mismatch: manifest={self.manifest.get('schedule_sha256')}, actual={actual_sha}"
            )

        self.epochs: dict[int, list[dict[str, Any]]] = {}
        for line in self.schedule_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if len(row["uids"]) != EXPECTED_TARGET_BATCH:
                raise RuntimeError(f"Target schedule batch size mismatch for {row.get('key')}")
            if len(row["sample_view_seeds"]) != EXPECTED_TARGET_BATCH:
                raise RuntimeError(f"Target view-seed count mismatch for {row.get('key')}")
            self.epochs.setdefault(int(row["epoch"]), []).append(row)

        if tuple(sorted(self.epochs)) != EXPECTED_TOTAL_EPOCHS:
            raise RuntimeError(f"Target schedule epochs mismatch: {sorted(self.epochs)}")
        for epoch, rows in self.epochs.items():
            rows.sort(key=lambda row: int(row["batch"]))
            if len(rows) != EXPECTED_STEPS_PER_EPOCH:
                raise RuntimeError(f"Target epoch {epoch} has {len(rows)} rows")
            if [int(row["batch"]) for row in rows] != list(range(EXPECTED_STEPS_PER_EPOCH)):
                raise RuntimeError(f"Target epoch {epoch} has incomplete batch indices")

    def rows_for_epoch(self, total_epoch: int) -> list[dict[str, Any]]:
        try:
            return self.epochs[int(total_epoch)]
        except KeyError as exc:
            raise KeyError(f"No Target schedule for e{total_epoch}") from exc


class ProtocolBTargetBatchSampler(Sampler[list[TargetScheduleToken]]):
    def __init__(self, schedule: ProtocolBTargetSchedule, total_epoch: int = 71) -> None:
        self.schedule = schedule
        self.total_epoch = int(total_epoch)

    def set_total_epoch(self, total_epoch: int) -> None:
        self.schedule.rows_for_epoch(total_epoch)
        self.total_epoch = int(total_epoch)

    def __iter__(self) -> Iterator[list[TargetScheduleToken]]:
        for row in self.schedule.rows_for_epoch(self.total_epoch):
            yield [
                TargetScheduleToken(
                    uid,
                    int(seed),
                    self.total_epoch,
                    int(row["batch"]),
                    position,
                )
                for position, (uid, seed) in enumerate(zip(row["uids"], row["sample_view_seeds"]))
            ]

    def __len__(self) -> int:
        return len(self.schedule.rows_for_epoch(self.total_epoch))


class ProtocolBTargetDataset(Dataset):
    def __init__(self, image_root: str | Path, uid_to_path: dict[str, Path], imgsz: int = 640) -> None:
        self.image_root = Path(image_root)
        self.uid_to_path = {str(uid): Path(path) for uid, path in uid_to_path.items()}
        self.imgsz = int(imgsz)
        self.letterbox = LetterBox(
            new_shape=(self.imgsz, self.imgsz),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            stride=32,
            padding_value=114,
            interpolation=cv2.INTER_LINEAR,
        )

    def __len__(self) -> int:
        return len(self.uid_to_path)

    def __getitem__(self, token: TargetScheduleToken | tuple) -> dict[str, Any]:
        token = TargetScheduleToken(*token)
        if token.uid not in self.uid_to_path:
            raise KeyError(f"Target UID not present in pool: {token.uid}")
        image_path = self.uid_to_path[token.uid]
        normalized = str(image_path).replace("\\", "/").lower()
        if "/target/dev/" in normalized or "/target/test/" in normalized:
            raise RuntimeError(f"Forbidden target split read: {image_path}")

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(token.view_seed)
            np.random.seed(token.view_seed & 0xFFFFFFFF)
            torch.manual_seed(token.view_seed)

            image, decoder = read_target_image_readonly(image_path)
            labels = {"img": image}
            params = self.letterbox.get_params(labels)
            image = self.letterbox.apply_image(labels, params)["img"]
            image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
            image_tensor = torch.from_numpy(image)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

        return {
            "img": image_tensor,
            "protocol_b_target_uid": token.uid,
            "protocol_b_target_view_seed": int(token.view_seed),
            "protocol_b_target_total_epoch": int(token.total_epoch),
            "protocol_b_target_batch_index": int(token.batch_index),
            "protocol_b_target_position": int(token.position),
            "im_file": str(image_path),
            "protocol_b_target_decoder": decoder,
            "orig_shape": torch.tensor(params["orig_shape"], dtype=torch.int32),
            "ratio": torch.tensor(params["ratio"], dtype=torch.float32),
            "pad": torch.tensor([params["left"], params["top"]], dtype=torch.float32),
        }


def collate_target_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "img": torch.stack([sample["img"] for sample in samples], dim=0),
        "protocol_b_target_uid": [sample["protocol_b_target_uid"] for sample in samples],
        "protocol_b_target_view_seed": [sample["protocol_b_target_view_seed"] for sample in samples],
        "protocol_b_target_total_epoch": [sample["protocol_b_target_total_epoch"] for sample in samples],
        "protocol_b_target_batch_index": [sample["protocol_b_target_batch_index"] for sample in samples],
        "protocol_b_target_position": [sample["protocol_b_target_position"] for sample in samples],
        "im_file": [sample["im_file"] for sample in samples],
        "protocol_b_target_decoder": [
            sample["protocol_b_target_decoder"] for sample in samples
        ],
        "orig_shape": torch.stack([sample["orig_shape"] for sample in samples], dim=0),
        "ratio": torch.stack([sample["ratio"] for sample in samples], dim=0),
        "pad": torch.stack([sample["pad"] for sample in samples], dim=0),
    }


class ResettableProtocolBTargetDataLoader(DataLoader):
    def reset(self) -> None:
        return None


def build_protocol_b_target_loader(
    *,
    target_yaml: str | Path,
    schedule: ProtocolBTargetSchedule,
    workers: int,
    imgsz: int = 640,
    total_epoch: int = 71,
) -> ResettableProtocolBTargetDataLoader:
    _, _, image_root = resolve_target_adapt_root(target_yaml)
    uids, uid_to_path = scan_target_pool(image_root)
    scheduled_uids = {
        uid
        for rows in schedule.epochs.values()
        for row in rows
        for uid in row["uids"]
    }
    missing = sorted(scheduled_uids - set(uid_to_path))
    if missing:
        raise KeyError(f"Scheduled Target UIDs missing from pool: {missing[:20]}")
    if set(uids) != set(scheduled_uids):
        unscheduled = sorted(set(uids) - scheduled_uids)
        raise RuntimeError(f"Target pool contains unscheduled UIDs: {unscheduled[:20]}")

    dataset = ProtocolBTargetDataset(image_root, uid_to_path, imgsz=imgsz)
    batch_sampler = ProtocolBTargetBatchSampler(schedule, total_epoch=total_epoch)
    generator = torch.Generator()
    generator.manual_seed(1042)
    return ResettableProtocolBTargetDataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_target_batch,
        generator=generator,
    )
