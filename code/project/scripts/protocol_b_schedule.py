from __future__ import annotations

import json
import random
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

ScheduleToken = namedtuple(
    "ScheduleToken",
    ["dataset_index", "aug_seed", "uid", "total_epoch", "batch_index", "position"],
)


class ProtocolBSchedule:
    def __init__(self, schedule_path: str | Path, augmentation_path: str | Path) -> None:
        self.schedule_path = Path(schedule_path)
        self.augmentation_path = Path(augmentation_path)
        self.epochs: dict[int, list[dict[str, Any]]] = {}
        schedule_rows = [json.loads(line) for line in self.schedule_path.read_text(encoding="utf-8").splitlines() if line]
        aug_rows = [json.loads(line) for line in self.augmentation_path.read_text(encoding="utf-8").splitlines() if line]
        aug_by_key = {row["key"]: row for row in aug_rows}
        if len(aug_by_key) != len(aug_rows):
            raise RuntimeError("Duplicate augmentation schedule keys")
        for row in schedule_rows:
            key = row["key"]
            if key not in aug_by_key:
                raise KeyError(f"Missing augmentation row for {key}")
            aug = aug_by_key[key]
            if len(row["uids"]) != len(aug["sample_aug_seeds"]):
                raise RuntimeError(f"UID/seed count mismatch for {key}")
            merged = dict(row)
            merged["sample_aug_seeds"] = list(map(int, aug["sample_aug_seeds"]))
            self.epochs.setdefault(int(row["epoch"]), []).append(merged)
        if sorted(self.epochs) != list(range(71, 101)):
            raise RuntimeError(f"Schedule epochs are not exactly 71..100: {sorted(self.epochs)}")
        for epoch, rows in self.epochs.items():
            rows.sort(key=lambda item: int(item["batch"]))
            if len(rows) != 552:
                raise RuntimeError(f"Epoch {epoch} contains {len(rows)} batches, expected 552")
            if [int(row["batch"]) for row in rows] != list(range(552)):
                raise RuntimeError(f"Epoch {epoch} batch indices are incomplete")

    def rows_for_epoch(self, total_epoch: int) -> list[dict[str, Any]]:
        try:
            return self.epochs[int(total_epoch)]
        except KeyError as exc:
            raise KeyError(f"No Protocol B source schedule for total epoch {total_epoch}") from exc


class ProtocolBBatchSampler(Sampler[list[ScheduleToken]]):
    def __init__(self, schedule: ProtocolBSchedule, uid_to_index: dict[str, int], total_epoch: int = 71) -> None:
        self.schedule = schedule
        self.uid_to_index = uid_to_index
        self.total_epoch = int(total_epoch)
        all_scheduled = {uid for rows in schedule.epochs.values() for row in rows for uid in row["uids"]}
        missing = sorted(all_scheduled - set(uid_to_index))
        if missing:
            raise KeyError(f"Scheduled UIDs missing from dataset: {missing[:20]}")

    def set_total_epoch(self, total_epoch: int) -> None:
        self.schedule.rows_for_epoch(total_epoch)
        self.total_epoch = int(total_epoch)

    def __iter__(self) -> Iterator[list[ScheduleToken]]:
        for row in self.schedule.rows_for_epoch(self.total_epoch):
            tokens = [
                ScheduleToken(
                    self.uid_to_index[uid],
                    int(seed),
                    uid,
                    self.total_epoch,
                    int(row["batch"]),
                    position,
                )
                for position, (uid, seed) in enumerate(zip(row["uids"], row["sample_aug_seeds"]))
            ]
            yield tokens

    def __len__(self) -> int:
        return len(self.schedule.rows_for_epoch(self.total_epoch))


class ScheduledSourceDataset(Dataset):
    def __init__(self, base_dataset: Dataset) -> None:
        object.__setattr__(self, "base_dataset", base_dataset)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        # Windows DataLoader uses spawn and unpickles the wrapper before
        # ``base_dataset`` has necessarily been restored. Delegating through
        # ``self.base_dataset`` at that point recursively calls __getattr__.
        if name == "base_dataset":
            raise AttributeError(name)
        state = object.__getattribute__(self, "__dict__")
        base_dataset = state.get("base_dataset")
        if base_dataset is None:
            raise AttributeError(name)
        return getattr(base_dataset, name)

    def __getstate__(self) -> dict[str, Any]:
        return {"base_dataset": object.__getattribute__(self, "base_dataset")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        if "base_dataset" not in state:
            raise RuntimeError("ScheduledSourceDataset pickle state is missing base_dataset")
        object.__setattr__(self, "base_dataset", state["base_dataset"])

    @property
    def mosaic(self) -> Any:
        return getattr(self.base_dataset, "mosaic", False)

    @mosaic.setter
    def mosaic(self, value: Any) -> None:
        setattr(self.base_dataset, "mosaic", value)

    def close_mosaic(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_dataset.close_mosaic(*args, **kwargs)

    def __getitem__(self, token: ScheduleToken | tuple | int) -> dict[str, Any]:
        if isinstance(token, int):
            return self.base_dataset[token]
        token = ScheduleToken(*token)
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(token.aug_seed)
            np.random.seed(token.aug_seed & 0xFFFFFFFF)
            torch.manual_seed(token.aug_seed)
            sample = dict(self.base_dataset[token.dataset_index])
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
        sample["protocol_b_uid"] = token.uid
        sample["protocol_b_aug_seed"] = int(token.aug_seed)
        sample["protocol_b_total_epoch"] = int(token.total_epoch)
        sample["protocol_b_batch_index"] = int(token.batch_index)
        sample["protocol_b_position"] = int(token.position)
        return sample


class ResettableProtocolBDataLoader(DataLoader):
    """Standard finite DataLoader with the reset method expected by Ultralytics."""

    def reset(self) -> None:
        # persistent_workers=False, so the next epoch creates workers from the current dataset state.
        return None


def build_protocol_b_source_loader(
    base_dataset: Dataset,
    schedule: ProtocolBSchedule,
    *,
    workers: int,
    total_epoch: int = 71,
) -> ResettableProtocolBDataLoader:
    uid_to_index: dict[str, int] = {}
    for index, image_path in enumerate(base_dataset.im_files):
        uid = Path(image_path).stem
        if uid in uid_to_index:
            raise RuntimeError(f"Duplicate source UID in dataset: {uid}")
        uid_to_index[uid] = index
    wrapped = ScheduledSourceDataset(base_dataset)
    batch_sampler = ProtocolBBatchSampler(schedule, uid_to_index, total_epoch=total_epoch)
    generator = torch.Generator()
    generator.manual_seed(42)
    return ResettableProtocolBDataLoader(
        wrapped,
        batch_sampler=batch_sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=base_dataset.collate_fn,
        generator=generator,
    )
