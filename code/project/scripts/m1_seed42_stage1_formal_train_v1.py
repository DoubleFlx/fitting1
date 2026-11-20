from __future__ import annotations

"""Formal seed-42 Stage-1 runner for Control versus Native-A-Fixed.

This file is deliberately self contained at the engineering boundary.  It
reuses the frozen Protocol-B source/target schedules, B2 Teacher/Student loss,
HPL generator, optimizer, EMA and AMP implementation.  The only branch
difference is the frozen Student Target pre-letterbox input: identity for
CONTROL, and the already-audited exact Fixed-A BGR cache on exposure-active
visits for NATIVE_A_FIXED.  There is no online photometric realization and no
solver import in this module.

The public execution boundary is intentionally split into:

* ``self-check``: CPU-only structural tests; no checkpoint or image read.
* ``logging-preflight``: four fresh-e71, ten-microbatch isolated workers.
* ``formal-branch``: one singleton-protected e72-e100 branch.
* ``convert-parquet``: post-training child process; the live training process
  never imports pandas or pyarrow.  ``torchvision`` remains the frozen
  transitive Ultralytics dependency outside each HPL call; the frozen resource
  protocol temporarily removes only its root registry key during that call so
  local ``TorchNMS.nms`` is selected, then restores the identical object.

The RUN_CONTRACT is external and immutable.  It is frozen after this file's
SHA256 is known, and every runtime verifies ``code.formal_train_runner_sha256``
against ``sha256(__file__)`` before touching CUDA or Target Adapt images.
"""

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import time
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch


ROOT = Path(r"D:\CV_YOLO11_ProtocolB_Formal")
SCRIPT_DIR = ROOT / "project" / "scripts"
LOCAL_ULTRALYTICS = ROOT / "project" / "ultralytics"
for _path in (SCRIPT_DIR, LOCAL_ULTRALYTICS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ultralytics.utils import RANK  # noqa: E402
from ultralytics.utils.torch_utils import autocast, unwrap_model  # noqa: E402

import protocol_b_b2_trainer as _b2_module  # noqa: E402
from protocol_b_b2_trainer import ProtocolBB2Trainer  # noqa: E402
from protocol_b_full_state import (  # noqa: E402
    EXPECTED_DATASET_MANIFEST,
    atomic_torch_save,
    restore_rng_state,
    rotate_last_checkpoint,
    state_dict_max_abs_diff,
    tensor_state_hash,
)
from protocol_b_m1_common import (  # noqa: E402
    bn_running_state_clone,
    recursive_state_sha256,
    rng_state_hashes,
    rng_state_payload,
    tensor_hash,
)
from protocol_b_m1_data_isolation import (  # noqa: E402
    ensure_m1_data_isolation_guard,
    install_m1_data_isolation_guard,
)
from protocol_b_m1_model import ProtocolBM1DetectionModel  # noqa: E402
from protocol_b_pseudo import generate_high_confidence_pseudo  # noqa: E402
from protocol_b_target_schedule import (  # noqa: E402
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_TARGET_BATCH,
    EXPECTED_TARGET_POOL_SIZE,
    ProtocolBTargetBatchSampler,
    ProtocolBTargetDataset,
    ProtocolBTargetSchedule,
    ResettableProtocolBTargetDataLoader,
    TargetScheduleToken,
    read_target_image_readonly,
    resolve_target_adapt_root,
    scan_target_pool,
)
from protocol_b_training_singleton import acquire_training_singleton  # noqa: E402


SCHEMA = "m1_seed42_formal_stage1_control_vs_native_a_fixed_run_contract_v1"
EXPERIMENT = "M1_SEED42_FORMAL_STAGE1_CONTROL_VS_NATIVE_A_FIXED_V1"
BRANCHES = ("CONTROL", "NATIVE_A_FIXED")
BRANCH_RUN_NAMES = {"CONTROL": "control", "NATIVE_A_FIXED": "native_a_fixed"}
BRANCH_CONTRACT_METADATA = {
    "execution_order": ["CONTROL", "NATIVE_A_FIXED"],
    "fresh_exact_e71_restore_per_branch": True,
    "fresh_process_per_branch": True,
    "order_is_execution_only_not_scientific": True,
}
BRANCH_CONTRACT_KEYS = frozenset({*BRANCHES, *BRANCH_CONTRACT_METADATA})
RESULT_ROOT = ROOT / "project" / "results" / EXPERIMENT
DEFAULT_CONTRACT = RESULT_ROOT / "RUN_CONTRACT.json"
FIXED_CACHE_PRELAUNCH = RESULT_ROOT / "fixed_cache_prelaunch.json"

EPOCHS = tuple(range(72, 101))
SOURCE_BATCH = 32
TARGET_BATCH = 8
TARGET_VISITS_PER_EPOCH = EXPECTED_STEPS_PER_EPOCH * TARGET_BATCH
TOTAL_TARGET_VISITS = len(EPOCHS) * TARGET_VISITS_PER_EPOCH
ACCUMULATION = 2
WINDOWS_PER_EPOCH = EXPECTED_STEPS_PER_EPOCH // ACCUMULATION
TOTAL_WINDOWS = len(EPOCHS) * WINDOWS_PER_EPOCH
PREFLIGHT_MICROBATCHES = 10
PREFLIGHT_WINDOWS = 5

E71_FULL_STATE = ROOT / "project" / "runs" / "final_v2_1" / (
    "protocol_b_b2_tsda_s410b_core_e71_e100_20260724_225511"
) / "full_state" / "last_full_state.pt"
E71_FILE_SHA256 = "01ee8b07e10d80b54ffa922d7dce05f046e72422e81d383f331e7e35f237a324"
E71_RAW_TENSOR_SHA256 = "50da5d059f8910f78edcd62757bdf09359c69e74d77059b5f51ff0854adeb6e3"
E71_EMA_TENSOR_SHA256 = "740217d1efa41368f2eef9c59ed663b7cd5c05edb6e2ebae226f2fa82097b621"
E71_GLOBAL_STEP = 39_192
E71_EMA_UPDATES = 20_010
E70_BOOTSTRAP = ROOT / "project" / "runs" / "final_v2_1" / "protocol_b_common_fork_e70_frozen.pt"
E70_BOOTSTRAP_SHA256 = "2366ec68c56032fdab78cd6ab80cd2c485f06c04a92f358304caf177411cd90f"

CONTRACTS = ROOT / "project" / "contracts"
DATASET = ROOT / "datasets" / "dataset_final_v2_1"
SOURCE_YAML = DATASET / "configs" / "source_supervised.yaml"
TARGET_YAML = DATASET / "configs" / "target_adapt_unlabeled.yaml"
SOURCE_SCHEDULE = CONTRACTS / "protocol_b_source_schedule_e71_e100.jsonl"
SOURCE_AUGMENTATION = CONTRACTS / "protocol_b_source_augmentation_e71_e100.jsonl"
TARGET_SCHEDULE = CONTRACTS / "protocol_b_target_schedule_e71_e100.jsonl"
TARGET_SCHEDULE_MANIFEST = CONTRACTS / "protocol_b_target_schedule_manifest.json"
B2_TEMPLATE = CONTRACTS / "protocol_b_b2_contract_template_v2.json"
COMPARISON_RULE = CONTRACTS / "Protocol_B_comparison_rule_v1.json"
B1_METRICS = ROOT / "project" / "runs" / "final_v2_1" / (
    "protocol_b_b1_source_only_e71_e100_20260724_181826"
) / "train_epoch_metrics.csv"

FIXED_ROOT = ROOT / "project" / "results" / (
    "M1_NATIVE_A_FIXED_COMPLEXITY_UTILITY_ABLATION_V1_PRETRAIN_GATE"
)
FIXED_POLICY_CONTRACT = FIXED_ROOT / "FIXED_A_POLICY_CONTRACT.json"
FIXED_EXPOSURE = FIXED_ROOT / "fixed_a_visit_exposure_audit.csv"
FIXED_CACHE_MANIFEST = FIXED_ROOT / "fixed_a_cache_manifest.csv"
FIXED_CACHE_BUILD = FIXED_ROOT / "fixed_a_cache_build.json"
FIXED_PRETRAIN_FINAL_STATUS = FIXED_ROOT / "FINAL_STATUS.json"
FIXED_CACHE_ROOT = FIXED_ROOT / "working_cache"
PREFLIGHT_CONTROL_REFERENCE = FIXED_ROOT / (
    "resource_windows_attempt4/EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE/"
    "control/raw/resource_worker_CONTROL.json"
)
PREFLIGHT_FIXED_REFERENCE = FIXED_ROOT / (
    "resource_windows_attempt4/EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE/"
    "fixed_a/raw/resource_worker_NATIVE_A.json"
)

EXPECTED_INPUT_SHA256 = {
    "source_schedule": "d525929849744fe18b64c967226a774a983302832e750e9a36b8c67864715e86",
    "source_augmentation": "6d3bbee615dbc5a5e149e138d660e239b4e60f22004b83b7bbfdbd45d7c04216",
    "target_schedule": "d4dc5afea3adacec2f5562a10cf4f10226d751ada3b52f0c5a69aa1949d5b923",
    "fixed_policy_contract": "f26b856ab54cd3a55ff720a8215a3300fee16f82388fa045a1d05e840bb27d0c",
    "fixed_exposure": "1be0e1b15051d9d68a115c7a6f223ded55bec9c6b32edfea253d29edde51b231",
    "fixed_cache_manifest": "e07af146a73941adae88460f6dd0f023133ec5f51bb1c61ed2a032cce91667a7",
    "fixed_cache_build": "0a345389a51907dd7b953df300bebf7fb55e8788461f223bfc44600279e15e70",
    "fixed_pretrain_final_status": "de448700c426bbfd936d190c1afd7c84fb1fcf9781b3e9abc43406548c2554bd",
    "preflight_control_reference": "973ae9030b89d462c8270da2100aeec97b8466013b53eca7b320f12242864b32",
    "preflight_fixed_reference": "c18f608efaaf02551c5903e4d30001b2de884681e8b5f8952ad02b8495925fd8",
    "b1_train_metrics": "d7a6d5b17faa83b8948ee156387281f2d78ebe79a438beb3218e5af78ef2ea55",
}

PREFLIGHT_ENDPOINTS = {
    "CONTROL": {
        "raw": "9066f4e46461b9f32d2ab664cbeab1e46a5d85c1185f8935afb77b1883a14798",
        "ema": "ba709ef195993b5ebed14ea08fab299f39824b232324c3b04af93ed8a3820dec",
        "optimizer": "8d6408e8a3f8ee82487762a444c2456ccffca6f25e980aae4b44579f49905214",
        "scheduler": "e565f62bc2b71e76e7fa0feb142c510a149610b49f8f26eaba06230eef77e6fa",
        "scaler": "49c9aac11460be13b4f5bf967b9b8784741273dce93c92b43abe226e2c3191d2",
        "bn": "a61fe9c31ce49aed82ab194f2296dae3b15a7a64e6fb88468533468111cb48d8",
    },
    "NATIVE_A_FIXED": {
        "raw": "793b0daf5aed87f319919547757977704bb39c75219ae52d4ee4001af70e2270",
        "ema": "c22e401aa6eb11996056c67a9a499fe990f07e55383597d13f6f0edce0d35705",
        "optimizer": "3da64e21897492107e7ebdeb72ceb441f032993d7223a45ba0acb21f5ea05f7a",
        "scheduler": "e565f62bc2b71e76e7fa0feb142c510a149610b49f8f26eaba06230eef77e6fa",
        "scaler": "49c9aac11460be13b4f5bf967b9b8784741273dce93c92b43abe226e2c3191d2",
        "bn": "9578dc29c936fea15d73c3046e8483a6145020e72bc4450881dff054afa8d40b",
    },
}
PREFLIGHT_ENDPOINT_RNG_HASHES = {
    "rng_python": "dd70f5b371e14766767bec2931338361204e5f92c1612e79aa248c908d09cbc0",
    "rng_numpy": "a63c7ac6d1a2a8e5dc3df2127c03090c9d64658e70b5f44164bc408ccce095a8",
    "rng_torch_cpu": "066a6e59eb94c4c6c6e6696438b6627b3975da1547ea4fe1266bff8bc0d51c45",
    "rng_torch_cuda": "de5f5f7464b78e9778aa3bed7c57f0fb40bd64f671455e8a564bdfae3476a430",
}

EXPECTED_CODE_SHA256 = {
    "project/scripts/protocol_b_b1_trainer.py": "48cf2c8b205e67daacf1ab3b776a8100cf0252c2f2b059b6429dd8b28eac429f",
    "project/scripts/protocol_b_b2_trainer.py": "1f0ea0ca0ed07cc0480af16034d1311b0d72fd76837561f082601a554cf47ac5",
    "project/scripts/protocol_b_b2_model.py": "42d2e9b18d49a93796e56e1d238e3651eb708829748873235446f8a33669e4b7",
    "project/scripts/protocol_b_b2_full_state.py": "3cdd56c962489619e6a4ac38add57e8b212304cb76e75c6028d59a708b5aed1f",
    "project/scripts/protocol_b_full_state.py": "1c24c2bc4dad0bebf3a3c362af5cb3ddce6707100bd491c2f49538ae9658546f",
    "project/scripts/protocol_b_schedule.py": "cd7c69641690bcc450de6d8956334352a351c96338c999f3917996c6988fb12a",
    "project/scripts/protocol_b_target_schedule.py": "a2612a8c6c68e319a066c94cb9a34dbd201d4772a18440dc6970cccca5655101",
    "project/scripts/protocol_b_pseudo.py": "be97be4428e8df0e6412dbe3f1ba990f7214814059fcee06a80cc1caeebc44dd",
    "project/scripts/protocol_b_m1_model.py": "3ab4bd494a0ebf38685891dac20383465d2bae7445e9900422bcfc0d3a01ac13",
    "project/scripts/protocol_b_m1_common.py": "28aac71b72f7e7572ef312a35197841285b893082fe19aa066027ce814fde66c",
    "project/scripts/protocol_b_m1_data_isolation.py": "19c8da47872f9776b38ad15f9878be2902105c7033fd13c7c32414589d7df6ba",
    "project/scripts/protocol_b_training_singleton.py": "967333f12302a6fa702901782c267160558226ec789575678335b63b6addf396",
    "project/ultralytics/engine/trainer.py": "5232397fd8893c513dff824b52965de3394d75abea588a3b63a4b523de8f8111",
    "project/ultralytics/utils/nms.py": "f17f8bfba204ff96e95be41747504d23949f32a68fd29fe4a17b151cdf08b212",
}

ARRAY_HASH_PREFIX = b"M1_ROUTEA_REALIZED_BGR_UINT8_HWC_V1\0"
T_EVENTS = {10: "T10", 50: "T50", 100: "T100"}
E_EVENTS = {72: "E72_END", 75: "E75", 80: "E80", 90: "E90", 100: "E100"}
FORBIDDEN_RUNTIME_IMPORTS = ("pandas", "pyarrow", "torchvision")


class FormalStop(RuntimeError):
    """Fail-closed training/preflight stop."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    fieldnames = list(rows[0]) if rows else []
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_exclusive(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def parse_bool(value: Any, field: str) -> bool:
    text = str(value).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise FormalStop(f"STOP_BOOLEAN_SCHEMA:{field}:{value!r}")


def require_sha(value: Any, field: str) -> str:
    text = str(value).strip().casefold()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise FormalStop(f"STOP_SHA256_SCHEMA:{field}")
    return text


def inside_root(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(ROOT):
        raise FormalStop(f"STOP_PATH_OUTSIDE_ROOT:{value}")
    return value


def tensor_bytes_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(int(v) for v in value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def gradient_state_sha(model: torch.nn.Module) -> str:
    state = {
        name: parameter.grad
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    return tensor_state_hash(state)


def float32_hex(value: float) -> str:
    return struct.pack("<f", float(value)).hex()


def array_bytes_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def formal_bgr_sha(bgr: np.ndarray) -> str:
    value = np.ascontiguousarray(bgr)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise FormalStop("STOP_FIXED_BGR_REPRESENTATION")
    digest = hashlib.sha256()
    digest.update(ARRAY_HASH_PREFIX)
    digest.update(b"uint8")
    digest.update(str(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def exact_bgr_payload_hashes(bgr: np.ndarray) -> tuple[str, str]:
    """Return cache-array and formal-BGR SHA from one immutable byte snapshot."""
    value = np.ascontiguousarray(bgr)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise FormalStop("STOP_FIXED_BGR_REPRESENTATION")
    raw = value.tobytes(order="C")
    shape = str(tuple(int(item) for item in value.shape)).encode("ascii")
    array_digest = hashlib.sha256()
    array_digest.update(b"uint8")
    array_digest.update(shape)
    array_digest.update(raw)
    formal_digest = hashlib.sha256()
    formal_digest.update(ARRAY_HASH_PREFIX)
    formal_digest.update(b"uint8")
    formal_digest.update(shape)
    formal_digest.update(raw)
    return array_digest.hexdigest(), formal_digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FormalStop(f"STOP_JSON_NOT_OBJECT:{path}")
    return value


def _contract_path_value(value: Any, field: str, *, must_exist: bool = True) -> Path:
    path = inside_root(Path(str(value)))
    if must_exist and not path.exists():
        raise FormalStop(f"STOP_CONTRACT_PATH_MISSING:{field}:{path}")
    return path


def load_run_contract(path: Path, *, authority: str) -> tuple[dict[str, Any], str]:
    path = inside_root(path)
    if path != DEFAULT_CONTRACT.resolve():
        raise FormalStop(f"STOP_RUN_CONTRACT_NONCANONICAL_PATH:{path}")
    if not path.is_file():
        raise FormalStop(f"STOP_RUN_CONTRACT_MISSING:{path}")
    contract = _load_json(path)
    if contract.get("schema") != SCHEMA or contract.get("experiment") != EXPERIMENT:
        raise FormalStop("STOP_RUN_CONTRACT_IDENTITY")
    runner_sha = sha256_file(Path(__file__))
    if contract.get("code", {}).get("formal_train_runner_sha256") != runner_sha:
        raise FormalStop("STOP_RUNNER_SHA_NOT_FROZEN_IN_CONTRACT")
    authorization = contract.get("authorization", {})
    required_authority = (
        "logging_preflight_authorized" if authority == "logging-preflight" else "formal_training_authorized"
    )
    if authorization.get(required_authority) is not True:
        raise FormalStop(f"STOP_NOT_AUTHORIZED:{required_authority}")
    if any(
        authorization.get(key) is not False
        for key in ("target_dev_read_authorized", "target_test_read_authorized")
    ):
        raise FormalStop("STOP_FORBIDDEN_DATA_AUTHORIZATION")

    lock_rows = contract.get("input_locks")
    if not isinstance(lock_rows, list):
        raise FormalStop("STOP_CONTRACT_INPUT_LOCKS_SCHEMA")
    lock_by_role: dict[str, dict[str, Any]] = {}
    lock_keys = {
        "schema", "role", "path", "expected_bytes", "actual_bytes",
        "expected_sha256", "actual_sha256", "exact",
    }
    for row in lock_rows:
        if not isinstance(row, dict) or set(row) != lock_keys:
            raise FormalStop("STOP_CONTRACT_INPUT_LOCK_ROW_SCHEMA")
        role = str(row["role"])
        if role in lock_by_role:
            raise FormalStop(f"STOP_CONTRACT_INPUT_LOCK_DUPLICATE:{role}")
        lock_by_role[role] = row
    resource_locks = {
        "fixed_control_bgr_resource_worker_endpoint": (
            PREFLIGHT_CONTROL_REFERENCE,
            EXPECTED_INPUT_SHA256["preflight_control_reference"],
        ),
        "fixed_native_a_fixed_bgr_resource_worker_endpoint": (
            PREFLIGHT_FIXED_REFERENCE,
            EXPECTED_INPUT_SHA256["preflight_fixed_reference"],
        ),
        "b1_train_metrics": (
            B1_METRICS,
            EXPECTED_INPUT_SHA256["b1_train_metrics"],
        ),
    }
    for role, (locked_path, locked_sha) in resource_locks.items():
        row = lock_by_role.get(role)
        actual_bytes = locked_path.stat().st_size
        if not (
            row
            and row.get("schema") == "m1_seed42_stage1_locked_file_v1"
            and _contract_path_value(row.get("path"), f"input_locks.{role}.path")
            == locked_path.resolve()
            and int(row.get("expected_bytes", -1)) == actual_bytes
            and int(row.get("actual_bytes", -1)) == actual_bytes
            and row.get("expected_sha256") == locked_sha
            and row.get("actual_sha256") == locked_sha
            and row.get("exact") is True
        ):
            raise FormalStop(f"STOP_CONTRACT_RESOURCE_LOCK:{role}")

    start = contract.get("start_state", {})
    start_expected = {
        "file_sha256": E71_FILE_SHA256,
        "epoch": 71,
        "global_step": E71_GLOBAL_STEP,
        "optimizer_updates_absolute": E71_EMA_UPDATES,
        "ema_updates": E71_EMA_UPDATES,
        "raw_tensor_sha256": E71_RAW_TENSOR_SHA256,
        "ema_tensor_sha256": E71_EMA_TENSOR_SHA256,
    }
    if any(start.get(key) != value for key, value in start_expected.items()):
        raise FormalStop("STOP_CONTRACT_E71_METADATA")
    if _contract_path_value(start.get("path"), "start_state.path") != E71_FULL_STATE.resolve():
        raise FormalStop("STOP_CONTRACT_E71_PATH")

    schedules = contract.get("schedules", {})
    schedule_expectations = {
        "source_path": SOURCE_SCHEDULE,
        "source_augmentation_path": SOURCE_AUGMENTATION,
        "target_path": TARGET_SCHEDULE,
        "target_manifest_path": TARGET_SCHEDULE_MANIFEST,
    }
    for field, expected in schedule_expectations.items():
        if _contract_path_value(schedules.get(field), f"schedules.{field}") != expected.resolve():
            raise FormalStop(f"STOP_CONTRACT_SCHEDULE_PATH:{field}")

    fixed = contract.get("fixed_a", {})
    fixed_expectations = {
        "policy_contract_path": FIXED_POLICY_CONTRACT,
        "exposure_path": FIXED_EXPOSURE,
        "cache_manifest_path": FIXED_CACHE_MANIFEST,
        "cache_build_path": FIXED_CACHE_BUILD,
        "cache_root": FIXED_CACHE_ROOT,
    }
    for field, expected in fixed_expectations.items():
        if _contract_path_value(fixed.get(field), f"fixed_a.{field}") != expected.resolve():
            raise FormalStop(f"STOP_CONTRACT_FIXED_PATH:{field}")
    if fixed.get("cache_candidate") != "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE":
        raise FormalStop("STOP_CONTRACT_FIXED_CACHE_CANDIDATE")
    fixed_scalar_expectations = {
        "cache_manifest_sha256": EXPECTED_INPUT_SHA256["fixed_cache_manifest"],
        "cache_build_sha256": EXPECTED_INPUT_SHA256["fixed_cache_build"],
        "cache_miss_policy": "STOP_NO_FALLBACK",
        "online_powell_policy": "FORBIDDEN_ZERO",
    }
    if any(
        fixed.get(key) != value
        for key, value in fixed_scalar_expectations.items()
    ):
        raise FormalStop("STOP_CONTRACT_FIXED_CACHE_POLICY")

    cache_lock = contract.get("prelaunch_evidence_locks", {}).get("fixed_cache")
    if not isinstance(cache_lock, dict) or set(cache_lock) != {
        "path", "sha256", "bytes"
    }:
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_LOCK_SCHEMA")
    cache_evidence_path = _contract_path_value(
        cache_lock["path"], "prelaunch_evidence_locks.fixed_cache.path"
    )
    if cache_evidence_path != FIXED_CACHE_PRELAUNCH.resolve():
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_PATH")
    cache_evidence_bytes = cache_evidence_path.stat().st_size
    cache_evidence_sha = sha256_file(cache_evidence_path)
    if not (
        int(cache_lock["bytes"]) == cache_evidence_bytes
        and require_sha(cache_lock["sha256"], "fixed cache prelaunch lock SHA")
        == cache_evidence_sha
    ):
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_FILE_LOCK")
    cache_evidence = _load_json(cache_evidence_path)
    if contract.get("fixed_cache_evidence") != cache_evidence:
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_EMBEDDED_MISMATCH")
    cache_evidence_keys = {
        "schema", "status", "fixed_final_status_checks",
        "exposure_equivalence_checks", "cache_build_checks",
        "cache_equivalence_checks", "cache_resource_checks",
        "selection_images", "fixed_delta_images",
        "identity_no_valid_delta_images", "cache_entries", "cache_root",
        "candidate", "payload_hash_mode",
        "full_payload_hash_required_for_freeze", "payload_file_count",
        "payload_total_bytes", "payload_composite_sha256",
        "payload_hash_wall_seconds", "cache_miss_policy",
        "online_powell_policy", "identity_images_use_original_preletterbox_BGR",
        "working_cache_excluded_from_prior_core_zip",
    }
    if set(cache_evidence) != cache_evidence_keys:
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_EVIDENCE_SCHEMA")
    nested_key_sets = {
        "fixed_final_status_checks": {
            "status", "old_phase_training_authorized_false",
            "formal_training_not_started", "target_adapt_label_reads_zero",
            "target_dev_reads_zero", "target_test_reads_zero",
            "dataset_mutation_zero",
        },
        "exposure_equivalence_checks": {
            "status", "mask", "total", "active", "identity",
            "global_mismatch_zero", "per_image_mismatch_zero",
            "per_epoch_mismatch_zero",
        },
        "cache_build_checks": {
            "status", "entries", "online_powell_zero", "bgr_complete",
            "bgr_payload_bytes",
        },
        "cache_equivalence_checks": {
            "status", "strict_rows", "online_powell_zero",
            "cache_fallback_zero",
        },
        "cache_resource_checks": {
            "status", "best_candidate", "online_powell_zero",
            "cache_fallback_zero", "target_dev_zero", "target_test_zero",
        },
    }
    for field, expected_keys in nested_key_sets.items():
        values = cache_evidence.get(field)
        if (
            not isinstance(values, dict)
            or set(values) != expected_keys
            or not all(value is True for value in values.values())
        ):
            raise FormalStop(f"STOP_FIXED_CACHE_PRELAUNCH_CHECKS:{field}")
    cache_evidence_expected = {
        "schema": "m1_seed42_stage1_fixed_cache_prelaunch_v1",
        "status": "PASS",
        "selection_images": 3_272,
        "fixed_delta_images": 3_226,
        "identity_no_valid_delta_images": 46,
        "cache_entries": 3_226,
        "candidate": "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE",
        "payload_hash_mode": "FULL_FILE_SHA256",
        "full_payload_hash_required_for_freeze": True,
        "payload_file_count": 3_226,
        "payload_total_bytes": 29_612_305_664,
        "cache_miss_policy": "STOP_NO_FALLBACK",
        "online_powell_policy": "FORBIDDEN_ZERO",
        "identity_images_use_original_preletterbox_BGR": True,
        "working_cache_excluded_from_prior_core_zip": True,
    }
    if any(
        cache_evidence.get(key) != value
        for key, value in cache_evidence_expected.items()
    ):
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_CONSTANTS")
    if (
        _contract_path_value(cache_evidence.get("cache_root"), "fixed_cache_evidence.cache_root")
        != FIXED_CACHE_ROOT.resolve()
        or float(cache_evidence.get("payload_hash_wall_seconds", 0.0)) <= 0.0
    ):
        raise FormalStop("STOP_FIXED_CACHE_PRELAUNCH_RESOURCE")
    require_sha(
        cache_evidence.get("payload_composite_sha256"),
        "fixed cache payload composite SHA",
    )

    protocol = contract.get("protocol", {})
    protocol_expected = {
        "seed": 42,
        "formal_horizon": 100,
        "start_completed_epoch": 71,
        "source_batch": SOURCE_BATCH,
        "target_batch": TARGET_BATCH,
        "workers": 2,
        "accumulation": ACCUMULATION,
        "preflight_microbatches": PREFLIGHT_MICROBATCHES,
        "preflight_optimizer_windows": PREFLIGHT_WINDOWS,
        "preflight_fixed_active_visits": 41,
        "preflight_fixed_identity_visits": 39,
    }
    if any(protocol.get(key) != value for key, value in protocol_expected.items()):
        raise FormalStop("STOP_CONTRACT_PROTOCOL")
    if protocol.get("epochs") != list(EPOCHS):
        raise FormalStop("STOP_CONTRACT_EPOCHS")
    if protocol.get("t_events") != {"T0": 0, "T10": 10, "T50": 50, "T100": 100}:
        raise FormalStop("STOP_CONTRACT_T_EVENTS")
    if protocol.get("epoch_events") != list(E_EVENTS.values()):
        raise FormalStop("STOP_CONTRACT_E_EVENTS")
    if protocol.get("registered_checkpoint_epochs") != [94, 98, 100]:
        raise FormalStop("STOP_CONTRACT_REGISTERED_CHECKPOINT_EPOCHS")

    paths = contract.get("paths")
    branches = contract.get("branches")
    if not isinstance(paths, dict) or not isinstance(branches, dict):
        raise FormalStop("STOP_CONTRACT_RESULT_PATHS_SCHEMA")
    if (
        _contract_path_value(
            paths.get("result_root"), "paths.result_root", must_exist=False
        )
        != RESULT_ROOT.resolve()
    ):
        raise FormalStop("STOP_CONTRACT_RESULT_ROOT")
    branch_runs = paths.get("branch_runs")
    if (
        not isinstance(branch_runs, dict)
        or set(branch_runs) != set(BRANCHES)
        or set(branches) != BRANCH_CONTRACT_KEYS
        or any(not isinstance(branches[branch], dict) for branch in BRANCHES)
        or any(
            branches.get(key) != expected
            for key, expected in BRANCH_CONTRACT_METADATA.items()
        )
    ):
        raise FormalStop("STOP_CONTRACT_BRANCH_RUN_KEYS")
    for branch in BRANCHES:
        branch_run_dir = _contract_path_value(
            branches[branch].get("run_dir"),
            f"branches.{branch}.run_dir",
            must_exist=False,
        )
        paths_run_dir = _contract_path_value(
            branch_runs[branch],
            f"paths.branch_runs.{branch}",
            must_exist=False,
        )
        expected_run_dir = (
            RESULT_ROOT / "runs" / BRANCH_RUN_NAMES[branch]
        ).resolve()
        if branch_run_dir != paths_run_dir or branch_run_dir != expected_run_dir:
            raise FormalStop(f"STOP_BRANCH_RUN_PATH:{branch}")

    return contract, sha256_file(path)


def verify_static_locks() -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in EXPECTED_CODE_SHA256.items():
        path = ROOT / relative
        digest = sha256_file(path)
        actual[relative] = digest
        if digest != expected:
            raise FormalStop(f"STOP_FROZEN_CODE_SHA:{relative}:{digest}")
    inputs = {
        "source_schedule": SOURCE_SCHEDULE,
        "source_augmentation": SOURCE_AUGMENTATION,
        "target_schedule": TARGET_SCHEDULE,
        "fixed_policy_contract": FIXED_POLICY_CONTRACT,
        "fixed_exposure": FIXED_EXPOSURE,
        "fixed_cache_manifest": FIXED_CACHE_MANIFEST,
        "fixed_cache_build": FIXED_CACHE_BUILD,
        "fixed_pretrain_final_status": FIXED_PRETRAIN_FINAL_STATUS,
        "preflight_control_reference": PREFLIGHT_CONTROL_REFERENCE,
        "preflight_fixed_reference": PREFLIGHT_FIXED_REFERENCE,
        "b1_train_metrics": B1_METRICS,
    }
    for name, path in inputs.items():
        digest = sha256_file(path)
        actual[str(path.relative_to(ROOT))] = digest
        if digest != EXPECTED_INPUT_SHA256[name]:
            raise FormalStop(f"STOP_FROZEN_INPUT_SHA:{name}:{digest}")
    build = _load_json(FIXED_CACHE_BUILD)
    bgr_counts = build.get("candidate_counts", {}).get(
        "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE", {}
    )
    if not (
        build.get("schema") == "m1_native_a_fixed_cache_build_v1"
        and build.get("status") == "PASS"
        and int(build.get("expected_active_entries", -1)) == 3_226
        and int(bgr_counts.get("complete", -1)) == 3_226
        and build.get("manifest_csv_sha256")
        == EXPECTED_INPUT_SHA256["fixed_cache_manifest"]
    ):
        raise FormalStop("STOP_FIXED_CACHE_BUILD_AUTHORITY")
    pretrain = _load_json(FIXED_PRETRAIN_FINAL_STATUS)
    if not (
        pretrain.get("schema") == "m1_native_a_fixed_pretrain_final_status_v1"
        and pretrain.get("FINAL_STATUS") == "FIXED_A_PRETRAIN_GATE_PASS"
        and pretrain.get("FIXED_A_PRETRAIN_GATE_PASS") is True
        and int(pretrain.get("hard_check_fail_count", -1)) == 0
    ):
        raise FormalStop("STOP_FIXED_PRETRAIN_AUTHORITY")
    for branch, reference_path in (
        ("CONTROL", PREFLIGHT_CONTROL_REFERENCE),
        ("NATIVE_A_FIXED", PREFLIGHT_FIXED_REFERENCE),
    ):
        reference = _load_json(reference_path)
        final = reference.get("final", {})
        expected = PREFLIGHT_ENDPOINTS[branch]
        if not (
            reference.get("backward_calls") == 10
            and int(reference.get("ema_updates_during_worker", -1)) == 5
            and all(reference.get("checks", {}).values())
            and final.get("raw_state_sha256") == expected["raw"]
            and final.get("ema_state_sha256") == expected["ema"]
            and final.get("optimizer_state_sha256") == expected["optimizer"]
            and final.get("scheduler_state_sha256") == expected["scheduler"]
            and final.get("scaler_state_sha256") == expected["scaler"]
            and final.get("bn_running_state_sha256") == expected["bn"]
            and final.get("rng_hashes") == PREFLIGHT_ENDPOINT_RNG_HASHES
            and int(final.get("scheduler_last_epoch", -1)) == 71
            and int(final.get("ema_updates", -1)) == 20_015
            and int(final.get("optimizer_steps", -1)) == 5
        ):
            raise FormalStop(f"STOP_PREFLIGHT_RESOURCE_REFERENCE:{branch}")
    if sha256_file(E70_BOOTSTRAP) != E70_BOOTSTRAP_SHA256:
        raise FormalStop("STOP_E70_BOOTSTRAP_SHA")
    if sha256_file(E71_FULL_STATE) != E71_FILE_SHA256:
        raise FormalStop("STOP_E71_FILE_SHA")
    return actual


@dataclass(frozen=True)
class ExposureEntry:
    formal_visit_index: int
    epoch: int
    target_visit_ordinal: int
    schedule_uid: str
    canonical_uid: str
    fixed_active: bool
    fixed_identity: bool
    fixed_delta_exists: bool
    selected_delta_sha: str


@dataclass(frozen=True)
class CacheEntry:
    canonical_uid: str
    input_image_sha256: str
    solve_key: str
    expected_output_bgr_sha256: str
    sidecar_path: str
    sidecar_sha256: str
    bgr_path: str
    bgr_file_sha256: str
    bgr_array_bytes_sha256: str
    bgr_shape: tuple[int, int, int]
    bgr_file_bytes: int


def load_fixed_inputs(
    target_schedule: ProtocolBTargetSchedule,
) -> tuple[list[ExposureEntry], dict[str, CacheEntry], dict[str, Any]]:
    entries: list[ExposureEntry] = []
    active = 0
    with FIXED_EXPOSURE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schema", "formal_visit_index", "adaptation_seed", "epoch",
            "target_visit_ordinal", "schedule_uid", "canonical_uid",
            "fixed_delta_exists", "fixed_active", "fixed_identity",
            "fixed_selected_delta_sha", "active_match", "identity_match",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise FormalStop("STOP_EXPOSURE_SCHEMA")
        for expected_index, row in enumerate(reader):
            index = int(row["formal_visit_index"])
            epoch = int(row["epoch"])
            ordinal = int(row["target_visit_ordinal"])
            if index != expected_index:
                raise FormalStop("STOP_EXPOSURE_ORDER")
            if int(row["adaptation_seed"]) != 42:
                raise FormalStop("STOP_EXPOSURE_ADAPTATION_SEED")
            if index != (epoch - 72) * TARGET_VISITS_PER_EPOCH + ordinal:
                raise FormalStop("STOP_EXPOSURE_INDEX_FORMULA")
            if epoch not in EPOCHS or not 0 <= ordinal < TARGET_VISITS_PER_EPOCH:
                raise FormalStop("STOP_EXPOSURE_RANGE")
            fixed_active = parse_bool(row["fixed_active"], "fixed_active")
            fixed_identity = parse_bool(row["fixed_identity"], "fixed_identity")
            if fixed_active == fixed_identity:
                raise FormalStop("STOP_EXPOSURE_COMPLEMENT")
            if not parse_bool(row["active_match"], "active_match") or not parse_bool(row["identity_match"], "identity_match"):
                raise FormalStop("STOP_EXPOSURE_UPSTREAM_MISMATCH")
            selected = str(row["fixed_selected_delta_sha"]).strip().casefold()
            if selected:
                require_sha(selected, "fixed_selected_delta_sha")
            fixed_delta_exists = parse_bool(
                row["fixed_delta_exists"], "fixed_delta_exists"
            )
            if fixed_delta_exists != bool(selected) or (
                fixed_active and not fixed_delta_exists
            ):
                raise FormalStop("STOP_EXPOSURE_FIXED_DELTA_CONSISTENCY")
            entries.append(
                ExposureEntry(
                    index, epoch, ordinal, str(row["schedule_uid"]),
                    str(row["canonical_uid"]), fixed_active, fixed_identity,
                    fixed_delta_exists, selected,
                )
            )
            active += int(fixed_active)
    if len(entries) != TOTAL_TARGET_VISITS or active != 62_679:
        raise FormalStop(f"STOP_EXPOSURE_COUNTS:{len(entries)}:{active}")

    # Cross-lock every schedule UID.  adaptation_seed is the global fixed 42;
    # it is intentionally never compared with the per-visit view_seed.
    for epoch in EPOCHS:
        rows = target_schedule.rows_for_epoch(epoch)
        for batch_index, schedule_row in enumerate(rows):
            for position, (uid, _view_seed) in enumerate(
                zip(schedule_row["uids"], schedule_row["sample_view_seeds"], strict=True)
            ):
                ordinal = batch_index * TARGET_BATCH + position
                index = (epoch - 72) * TARGET_VISITS_PER_EPOCH + ordinal
                entry = entries[index]
                if entry.epoch != epoch or entry.target_visit_ordinal != ordinal or entry.schedule_uid != str(uid):
                    raise FormalStop(f"STOP_EXPOSURE_TARGET_SCHEDULE_CROSSLOCK:{index}")

    cache: dict[str, CacheEntry] = {}
    with FIXED_CACHE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "schema", "canonical_uid", "input_image_sha256",
            "solve_key", "expected_output_bgr_sha256", "sidecar_path", "sidecar_sha256",
            "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise FormalStop("STOP_CACHE_MANIFEST_SCHEMA")
        for row in reader:
            uid = str(row["canonical_uid"])
            if (
                row["schema"] != "m1_native_a_fixed_cache_manifest_entry_v1"
                or uid in cache
                or not parse_bool(
                    row["EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE_complete"],
                    "bgr_complete",
                )
            ):
                raise FormalStop("STOP_CACHE_MANIFEST_DUPLICATE_OR_INCOMPLETE")
            input_sha = require_sha(row["input_image_sha256"], "input_image_sha256")
            solve_key = require_sha(row["solve_key"], "solve_key")
            output_sha = require_sha(
                row["expected_output_bgr_sha256"], "expected_output_bgr_sha256"
            )
            sidecar_path = inside_root(Path(row["sidecar_path"]))
            sidecar_sha = require_sha(row["sidecar_sha256"], "sidecar_sha256")
            if sha256_file(sidecar_path) != sidecar_sha:
                raise FormalStop(f"STOP_FIXED_SIDECAR_SHA:{uid}")
            sidecar = _load_json(sidecar_path)
            candidate = sidecar.get("candidates", {}).get(
                "EXACT_FIXED_OUTPUT_PRELETTERBOX_BGR_CACHE", {}
            )
            shape = tuple(int(value) for value in candidate.get("shape", []))
            bgr_path = inside_root(Path(str(candidate.get("path", ""))))
            sidecar_checks = {
                "schema": sidecar.get("schema") == "m1_native_a_fixed_cache_entry_v1",
                "uid": sidecar.get("canonical_uid") == uid,
                "input": sidecar.get("input_image_sha256") == input_sha,
                "solve_key": sidecar.get("solve_key") == solve_key,
                "output": sidecar.get("expected_output_bgr_sha256") == output_sha,
                "complete": candidate.get("complete") is True,
                "dtype": candidate.get("dtype") == "uint8",
                "shape": len(shape) == 3 and shape[2] == 3,
                "file_exists": bgr_path.is_file(),
                "file_size": bgr_path.is_file()
                and bgr_path.stat().st_size == int(candidate.get("file_bytes", -1)),
            }
            if not all(sidecar_checks.values()):
                raise FormalStop(f"STOP_FIXED_SIDECAR_CROSSLOCK:{uid}:{sidecar_checks}")
            cache[uid] = CacheEntry(
                uid,
                input_sha,
                solve_key,
                output_sha,
                str(sidecar_path),
                sidecar_sha,
                str(bgr_path),
                require_sha(candidate.get("file_sha256"), "bgr_file_sha256"),
                require_sha(
                    candidate.get("array_bytes_sha256"), "bgr_array_bytes_sha256"
                ),
                shape,
                int(candidate["file_bytes"]),
            )
    active_uids = {entry.canonical_uid for entry in entries if entry.fixed_delta_exists}
    if len(cache) != 3_226 or set(cache) != active_uids:
        raise FormalStop(f"STOP_CACHE_UID_SET:{len(cache)}:{len(active_uids)}")
    summary = {
        "exposure_rows": len(entries),
        "active_visits": active,
        "identity_visits": len(entries) - active,
        "cache_entries": len(cache),
        "preflight_active_visits": sum(int(row.fixed_active) for row in entries[:80]),
        "preflight_identity_visits": sum(int(row.fixed_identity) for row in entries[:80]),
        "adaptation_seed_compared_only_to_global_42": True,
        "view_seed_compared_only_by_frozen_target_schedule": True,
    }
    if summary["preflight_active_visits"] != 41 or summary["preflight_identity_visits"] != 39:
        raise FormalStop("STOP_PREFLIGHT_EXPOSURE_COUNTS")
    return entries, cache, summary


class FixedStage1TargetDataset(ProtocolBTargetDataset):
    """Original Teacher tensor plus exact cached Fixed-A Student tensor."""

    def __init__(
        self,
        image_root: str | Path,
        uid_to_path: dict[str, Path],
        *,
        exposure_entries: list[ExposureEntry],
        cache_entries: dict[str, CacheEntry],
        branch: str,
        imgsz: int,
        data_audit_dir: str | Path,
        verify_payload_on_read: bool,
    ) -> None:
        super().__init__(image_root, uid_to_path, imgsz=imgsz)
        if branch not in BRANCHES:
            raise FormalStop("STOP_TARGET_DATASET_BRANCH")
        self.exposure_entries = exposure_entries
        self.cache_entries = cache_entries
        self.branch = branch
        self.data_audit_dir = str(inside_root(Path(data_audit_dir)))
        self.verify_payload_on_read = bool(verify_payload_on_read)

    def _load_exact_bgr(
        self, entry: ExposureEntry, original_shape: tuple[int, ...]
    ) -> tuple[np.ndarray, CacheEntry, float, float, bool]:
        try:
            cache_entry = self.cache_entries[entry.canonical_uid]
        except KeyError as exc:
            raise FormalStop(f"STOP_FIXED_CACHE_MISS:{entry.canonical_uid}") from exc
        bgr_path = inside_root(Path(cache_entry.bgr_path))
        read_started = time.perf_counter()
        mmap = np.load(bgr_path, mmap_mode="r", allow_pickle=False)
        if (
            mmap.dtype != np.uint8
            or tuple(int(value) for value in mmap.shape) != cache_entry.bgr_shape
        ):
            raise FormalStop(f"STOP_FIXED_BGR_NPY_REPRESENTATION:{entry.canonical_uid}")
        bgr = np.ascontiguousarray(mmap)
        del mmap
        read_seconds = time.perf_counter() - read_started
        if tuple(bgr.shape) != tuple(original_shape):
            raise FormalStop(f"STOP_FIXED_BGR_GEOMETRY:{entry.canonical_uid}")
        payload_validation = self.verify_payload_on_read
        validation_started = time.perf_counter()
        if payload_validation:
            if sha256_file(bgr_path) != cache_entry.bgr_file_sha256:
                raise FormalStop(f"STOP_FIXED_BGR_FILE_SHA:{entry.canonical_uid}")
            if array_bytes_sha(bgr) != cache_entry.bgr_array_bytes_sha256:
                raise FormalStop(f"STOP_FIXED_BGR_ARRAY_SHA:{entry.canonical_uid}")
            if formal_bgr_sha(bgr) != cache_entry.expected_output_bgr_sha256:
                raise FormalStop(f"STOP_FIXED_BGR_FORMAL_SHA:{entry.canonical_uid}")
        validation_seconds = (
            time.perf_counter() - validation_started if payload_validation else 0.0
        )
        return bgr, cache_entry, read_seconds, validation_seconds, payload_validation

    def __getitem__(self, token: TargetScheduleToken | tuple) -> dict[str, Any]:
        isolation_guard = ensure_m1_data_isolation_guard(
            phase=f"{EXPERIMENT}_{self.branch}",
            audit_dir=Path(self.data_audit_dir) / f"e{int(token.total_epoch):03d}",
            allow_target_dev_images=False,
        )
        token = TargetScheduleToken(*token)
        if token.uid not in self.uid_to_path:
            raise FormalStop(f"STOP_TARGET_UID_NOT_IN_POOL:{token.uid}")
        if not 72 <= int(token.total_epoch) <= 100:
            raise FormalStop(f"STOP_TARGET_TOKEN_EPOCH:{token.total_epoch}")
        ordinal = int(token.batch_index) * TARGET_BATCH + int(token.position)
        index = (int(token.total_epoch) - 72) * TARGET_VISITS_PER_EPOCH + ordinal
        entry = self.exposure_entries[index]
        if (
            entry.formal_visit_index != index
            or entry.epoch != int(token.total_epoch)
            or entry.target_visit_ordinal != ordinal
            or entry.schedule_uid != str(token.uid)
        ):
            raise FormalStop(f"STOP_FIXED_VISIT_LOOKUP:{index}")
        image_path = self.uid_to_path[token.uid]
        normalized = str(image_path).replace("\\", "/").casefold()
        if "/target/dev/" in normalized or "/target/test/" in normalized:
            raise FormalStop("STOP_FORBIDDEN_TARGET_SPLIT")

        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(int(token.view_seed))
            np.random.seed(int(token.view_seed) & 0xFFFFFFFF)
            torch.manual_seed(int(token.view_seed))
            original_bgr, decoder = read_target_image_readonly(image_path)
            weak_labels = {"img": original_bgr}
            params = self.letterbox.get_params(weak_labels)
            weak_hwc = self.letterbox.apply_image(weak_labels, params)["img"]
            weak_chw = np.ascontiguousarray(weak_hwc[:, :, ::-1].transpose(2, 0, 1))
            weak_tensor = torch.from_numpy(weak_chw)

            apply_fixed = self.branch == "NATIVE_A_FIXED" and entry.fixed_active
            output_sha = ""
            cache_array_sha = ""
            cache_read_seconds = 0.0
            cache_validation_seconds = 0.0
            cache_first_validation = False
            student_letterbox_seconds = 0.0
            if apply_fixed:
                (
                    transformed_bgr,
                    cache_entry,
                    cache_read_seconds,
                    cache_validation_seconds,
                    cache_first_validation,
                ) = self._load_exact_bgr(entry, tuple(original_bgr.shape))
                letterbox_started = time.perf_counter()
                strong_hwc = self.letterbox.apply_image({"img": transformed_bgr}, params)["img"]
                strong_chw = np.ascontiguousarray(strong_hwc[:, :, ::-1].transpose(2, 0, 1))
                student_tensor = torch.from_numpy(strong_chw)
                student_letterbox_seconds = time.perf_counter() - letterbox_started
                output_sha = cache_entry.expected_output_bgr_sha256
                cache_array_sha = cache_entry.bgr_array_bytes_sha256
            else:
                student_tensor = weak_tensor
            if weak_tensor.dtype != torch.uint8 or student_tensor.dtype != torch.uint8:
                raise FormalStop("STOP_TARGET_DTYPE")
            if weak_tensor.shape != student_tensor.shape:
                raise FormalStop("STOP_TARGET_VIEW_GEOMETRY")
            top, bottom = int(params["top"]), int(params["bottom"])
            left, right = int(params["left"]), int(params["right"])
            mask = torch.zeros((weak_tensor.shape[1], weak_tensor.shape[2]), dtype=torch.bool)
            mask[top : weak_tensor.shape[1] - bottom, left : weak_tensor.shape[2] - right] = True
            delta = (student_tensor.to(torch.int16) - weak_tensor.to(torch.int16)).abs()
            padding = delta[:, ~mask]
            padding_max = int(padding.max()) if padding.numel() else 0
            if padding_max != 0:
                raise FormalStop("STOP_FIXED_LETTERBOX_PADDING_CHANGED")
            if not apply_fixed and not torch.equal(student_tensor, weak_tensor):
                raise FormalStop("STOP_IDENTITY_NOT_EXACT")
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

        sample = {
            "img": weak_tensor,
            "fixed_student_img": student_tensor,
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
            "fixed_formal_visit_index": index,
            "fixed_target_visit_ordinal": ordinal,
            "fixed_canonical_uid": entry.canonical_uid,
            "fixed_exposure_active": int(entry.fixed_active),
            "fixed_exposure_identity": int(entry.fixed_identity),
            "fixed_transform_applied": int(apply_fixed),
            "fixed_output_bgr_sha256": output_sha,
            "fixed_cache_array_bytes_sha256": cache_array_sha,
            "fixed_cache_read_seconds": cache_read_seconds,
            "fixed_cache_payload_validation_seconds": cache_validation_seconds,
            "fixed_cache_payload_validation": int(cache_first_validation),
            "fixed_student_letterbox_seconds": student_letterbox_seconds,
            "fixed_padding_max_abs_byte_delta": padding_max,
        }
        # With persistent_workers=False, the final two scheduled batches are
        # dispatched round-robin to the two workers.  Explicit export avoids
        # relying on interpreter atexit behavior in multiprocessing workers.
        if (
            int(token.batch_index) in {8, 9, 550, 551}
            and int(token.position) == TARGET_BATCH - 1
        ):
            event_path = isolation_guard.export_process_audit()
            counters = {
                key: int(value or 0)
                for key, value in isolation_guard.counters().items()
            }
            summary_path = Path(self.data_audit_dir) / f"e{int(token.total_epoch):03d}" / (
                f"process_{os.getpid()}_DATA_ACCESS_SUMMARY.json"
            )
            atomic_json(
                summary_path,
                {
                    "schema": "m1_seed42_stage1_worker_data_access_summary_v1",
                    "branch": self.branch,
                    "epoch": int(token.total_epoch),
                    "process_id": os.getpid(),
                    "last_export_trigger_batch_index": int(token.batch_index),
                    "counters": counters,
                    "event_path": str(Path(event_path).resolve()),
                    "event_file_sha256": sha256_file(event_path),
                    "timestamp": now_iso(),
                },
            )
        return sample


_FIXED_SCALARS = (
    "fixed_formal_visit_index", "fixed_target_visit_ordinal", "fixed_canonical_uid",
    "fixed_exposure_active", "fixed_exposure_identity", "fixed_transform_applied",
    "fixed_output_bgr_sha256", "fixed_cache_array_bytes_sha256",
    "fixed_cache_read_seconds", "fixed_cache_payload_validation_seconds",
    "fixed_cache_payload_validation", "fixed_student_letterbox_seconds",
    "fixed_padding_max_abs_byte_delta",
)


class PrefixTargetBatchSampler(ProtocolBTargetBatchSampler):
    """Deterministic schedule prefix used only by the bounded logging gate.

    Giving DataLoader the exact ten-batch cardinality prevents background
    worker prefetch from reading unscheduled-for-this-gate Target images after
    the tenth consumed batch.  The yielded tokens are byte-for-byte the first
    ten tokens of the frozen canonical sampler.
    """

    def __init__(
        self,
        schedule: ProtocolBTargetSchedule,
        *,
        total_epoch: int,
        max_batches: int,
    ) -> None:
        super().__init__(schedule, total_epoch=total_epoch)
        self.max_batches = int(max_batches)
        if not 1 <= self.max_batches <= EXPECTED_STEPS_PER_EPOCH:
            raise FormalStop(f"STOP_TARGET_PREFIX_LENGTH:{self.max_batches}")

    def __iter__(self) -> Iterator[list[TargetScheduleToken]]:
        for batch_index, tokens in enumerate(super().__iter__()):
            if batch_index >= self.max_batches:
                break
            yield tokens

    def __len__(self) -> int:
        return self.max_batches


def collate_fixed_target(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "img": torch.stack([sample["img"] for sample in samples]),
        "fixed_student_img": torch.stack([sample["fixed_student_img"] for sample in samples]),
        "protocol_b_target_uid": [sample["protocol_b_target_uid"] for sample in samples],
        "protocol_b_target_view_seed": [sample["protocol_b_target_view_seed"] for sample in samples],
        "protocol_b_target_total_epoch": [sample["protocol_b_target_total_epoch"] for sample in samples],
        "protocol_b_target_batch_index": [sample["protocol_b_target_batch_index"] for sample in samples],
        "protocol_b_target_position": [sample["protocol_b_target_position"] for sample in samples],
        "im_file": [sample["im_file"] for sample in samples],
        "protocol_b_target_decoder": [sample["protocol_b_target_decoder"] for sample in samples],
        "orig_shape": torch.stack([sample["orig_shape"] for sample in samples]),
        "ratio": torch.stack([sample["ratio"] for sample in samples]),
        "pad": torch.stack([sample["pad"] for sample in samples]),
    }
    for key in _FIXED_SCALARS:
        batch[key] = [sample[key] for sample in samples]
    return batch


def build_fixed_target_loader(
    *,
    schedule: ProtocolBTargetSchedule,
    exposure_entries: list[ExposureEntry],
    cache_entries: dict[str, CacheEntry],
    branch: str,
    workers: int,
    imgsz: int,
    data_audit_dir: Path,
    verify_payload_on_read: bool,
    max_batches: int | None,
) -> ResettableProtocolBTargetDataLoader:
    _, _, image_root = resolve_target_adapt_root(TARGET_YAML)
    uids, uid_to_path = scan_target_pool(image_root)
    scheduled = {uid for rows in schedule.epochs.values() for row in rows for uid in row["uids"]}
    if set(uids) != scheduled:
        raise FormalStop("STOP_TARGET_POOL_SCHEDULE_MEMBERSHIP")
    dataset = FixedStage1TargetDataset(
        image_root,
        uid_to_path,
        exposure_entries=exposure_entries,
        cache_entries=cache_entries,
        branch=branch,
        imgsz=imgsz,
        data_audit_dir=data_audit_dir,
        verify_payload_on_read=verify_payload_on_read,
    )
    sampler: ProtocolBTargetBatchSampler
    if max_batches is None:
        sampler = ProtocolBTargetBatchSampler(schedule, total_epoch=72)
    else:
        sampler = PrefixTargetBatchSampler(
            schedule, total_epoch=72, max_batches=max_batches
        )
    generator = torch.Generator()
    generator.manual_seed(1042)
    return ResettableProtocolBTargetDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=collate_fixed_target,
        generator=generator,
    )


class CapturingIterator(Iterator[dict[str, Any]]):
    def __init__(self, inner: Iterator[dict[str, Any]], owner: "Stage1Trainer") -> None:
        self.inner = inner
        self.owner = owner

    def __iter__(self) -> "CapturingIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        value = next(self.inner)
        self.owner.last_target_raw = value
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class Stage1Trainer(ProtocolBB2Trainer):
    """Frozen B2 trainer with e71 restore, exact Fixed cache and read-only logging."""

    def __init__(
        self,
        *,
        branch: str,
        run_contract_path: Path,
        run_contract_sha256: str,
        start_state: dict[str, Any],
        start_state_path: Path,
        completed_start_epoch: int,
        exposure_entries: list[ExposureEntry],
        cache_entries: dict[str, CacheEntry],
        data_guard: Any,
        data_access_attempt_dir: Path,
        run_attempt_id: str,
        logging_enabled: bool,
        preflight_mode: bool,
        **kwargs: Any,
    ) -> None:
        if branch not in BRANCHES:
            raise FormalStop("STOP_TRAINER_BRANCH")
        self.stage1_branch = branch
        self.run_contract_path = run_contract_path.resolve()
        self.run_contract_sha256 = run_contract_sha256
        self.start_state = start_state
        self.start_state_path = start_state_path.resolve()
        self.completed_start_epoch = int(completed_start_epoch)
        self.stage1_start_global_step = int(start_state["global_step"])
        self.stage1_start_ema_updates = int(start_state["ema_updates"])
        self.exposure_entries = exposure_entries
        self.cache_entries = cache_entries
        self.data_guard = data_guard
        self.data_access_attempt_dir = inside_root(data_access_attempt_dir)
        self.run_attempt_id = str(run_attempt_id)
        if not self.run_attempt_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in self.run_attempt_id
        ):
            raise FormalStop("STOP_RUN_ATTEMPT_ID")
        self.logging_enabled = bool(logging_enabled)
        self.preflight_mode = bool(preflight_mode)
        self.last_target_raw: dict[str, Any] | None = None
        self._observed_pseudo: Any | None = None
        self._observer_calls_batch = 0
        self.observer_original_call_count = int(start_state.get("teacher_target_forward_count", 0))
        self.observer_return_identity_count = int(start_state.get("observer_return_identity_count", 0))
        self.teacher_target_forward_count = int(start_state.get("teacher_target_forward_count", 0))
        self.target_unsup_microbatch_count = int(start_state.get("target_unsup_microbatch_count", 0))
        self.hpl_torchvision_scope_call_count = int(
            start_state.get("hpl_torchvision_scope_call_count", 0)
        )
        self.hpl_torchvision_identity_restore_count = int(
            start_state.get("hpl_torchvision_identity_restore_count", 0)
        )
        self._epoch_trace_box_rows: list[dict[str, Any]] = []
        self._epoch_trace_batch_rows: list[dict[str, Any]] = []
        self._preflight_common_rows: list[dict[str, Any]] = []
        self._window_target_term_included = False
        self._window_target_microbatch_count = 0
        self._last_window_target_included = False
        self.target_contributing_optimizer_updates = int(
            start_state.get("target_contributing_optimizer_updates", 0)
        )
        self.actual_unskipped_optimizer_updates = int(
            start_state.get("actual_unskipped_optimizer_updates", 0)
        )
        self.amp_found_inf_windows = int(start_state.get("amp_found_inf_windows", 0))
        self.protocol_last_optimizer_window_found_inf = False
        self.protocol_last_optimizer_parameter_update_skipped = False
        self.protocol_last_found_inf_json = "{}"
        self._snapshot_rows: list[dict[str, Any]] = list(start_state.get("snapshot_rows", []))
        self._snapshot_names = {str(row["event_name"]) for row in self._snapshot_rows}
        self._t0_hook_handle: Any | None = None
        self._t0_hook_fire_count = int(start_state.get("t0_hook_fire_count", 0))
        self._first_window = True
        self._last_step_underlying_called = False
        self._last_step_target_contributing = False
        self._last_step_grad_sha = ""
        self._last_step_scale_before = float("nan")
        self._last_step_scale_after = float("nan")
        self.optimizer_update_rows: list[dict[str, Any]] = list(
            start_state.get("optimizer_update_rows", [])
        )
        self.training_epoch_metric_rows: list[dict[str, Any]] = list(
            start_state.get("training_epoch_metric_rows", [])
        )
        self.fixed_cache_runtime_rows: list[dict[str, Any]] = []
        self.fixed_exposure_runtime_rows: list[dict[str, Any]] = []
        self.checkpoint_rows: list[dict[str, Any]] = list(start_state.get("checkpoint_rows", []))
        self.recovery_rows: list[dict[str, Any]] = list(
            start_state.get("recovery_rows", [])
        )
        self.completed_data_access_epoch_rows: list[dict[str, Any]] = list(
            start_state.get("completed_data_access_epoch_rows", [])
        )
        super().__init__(**kwargs)
        # Recovery must preserve cumulative evidence, not merely mutable model
        # state.  These are primitive-only frozen B1/B2 audit rows and scalar
        # counters; restoring them cannot alter the next schedule/RNG node.
        self.protocol_epoch_rows = list(start_state.get("protocol_epoch_rows", []))
        self.protocol_b2_epoch_rows = list(
            start_state.get("protocol_b2_epoch_rows", [])
        )
        self.protocol_b2_batch_rows = list(
            start_state.get("protocol_b2_batch_rows", [])
        )
        for name in (
            "protocol_target_read_count_total",
            "protocol_target_schedule_mismatches_total",
            "protocol_source_hash_mismatches_total",
            "protocol_b2_pseudo_boxes_total",
            "protocol_target_cv2_decode_total",
            "protocol_target_pillow_fallback_total",
        ):
            setattr(self, name, int(start_state.get(name, 0)))
        self.method_config = dict(self.method_config)
        self.method_config.update(
            {
                "experiment": EXPERIMENT,
                "branch": branch,
                "primary_comparison": "MATCHED_CONTROL_VS_PER_IMAGE_FIXED_TARGET_CONDITIONED_POLICY",
                "control_student_target": "original exact identity",
                "fixed_student_target": "frozen exact pre-letterbox BGR cache on frozen active mask",
                "teacher_target_view": "original Target",
                "online_powell_calls": 0,
                "hpl_observer_extra_forward_count": 0,
            }
        )

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = ProtocolBM1DetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)
        return model

    def _restore_start_state(self) -> dict[str, Any]:
        model = unwrap_model(self.model)
        raw_result = model.load_state_dict(self.start_state["model_state_dict"], strict=True)
        ema_result = self.ema.ema.load_state_dict(self.start_state["ema_state_dict"], strict=True)
        if raw_result.missing_keys or raw_result.unexpected_keys or ema_result.missing_keys or ema_result.unexpected_keys:
            raise FormalStop("STOP_STRICT_START_TENSOR_RESTORE")
        self.optimizer.load_state_dict(self.start_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(self.start_state["scheduler_state_dict"])
        self.scaler.load_state_dict(self.start_state["scaler_state_dict"])
        restore_rng_state(self.start_state)
        self.ema.updates = int(self.start_state["ema_updates"])
        self.base_ema_updates = int(self.start_state["ema_updates"])
        self.protocol_optimizer_steps = int(self.start_state.get("optimizer_steps_this_branch", 0))
        self.start_epoch = self.completed_start_epoch
        expected_scheduler = self.completed_start_epoch - 1
        if int(self.scheduler.last_epoch) != expected_scheduler:
            raise FormalStop(
                f"STOP_START_SCHEDULER_NODE:{self.scheduler.last_epoch}:{expected_scheduler}"
            )
        raw_hash = tensor_state_hash(model.state_dict())
        ema_hash = tensor_state_hash(self.ema.ema.state_dict())
        if raw_hash != str(self.start_state.get("model_hash", raw_hash)) or ema_hash != str(self.start_state.get("ema_hash", ema_hash)):
            raise FormalStop("STOP_START_STORED_TENSOR_HASH")
        if self.completed_start_epoch == 71 and (
            raw_hash != E71_RAW_TENSOR_SHA256 or ema_hash != E71_EMA_TENSOR_SHA256
        ):
            raise FormalStop("STOP_E71_RUNTIME_TENSOR_HASH")
        return {
            "raw_tensor_sha256": raw_hash,
            "ema_tensor_sha256": ema_hash,
            "optimizer_state_sha256": recursive_state_sha256(self.optimizer.state_dict()),
            "scheduler_state_sha256": recursive_state_sha256(self.scheduler.state_dict()),
            "scaler_state_sha256": recursive_state_sha256(self.scaler.state_dict()),
            "rng_hashes": rng_state_hashes(),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "ema_updates": int(self.ema.updates),
        }

    def _setup_train(self):
        # Frozen B2 construction uses e70 only as an architecture/optimizer
        # bootstrap.  All mutable state is then strictly overwritten by the
        # canonical e71 (or a contract-validated complete-epoch recovery).
        super()._setup_train()
        for misleading in (
            Path(self.save_dir) / "audits" / "b1_restore_audit.json",
            Path(self.save_dir) / "audits" / "b2_restore_audit.json",
        ):
            misleading.unlink(missing_ok=True)
        self.target_loader = build_fixed_target_loader(
            schedule=self.target_schedule,
            exposure_entries=self.exposure_entries,
            cache_entries=self.cache_entries,
            branch=self.stage1_branch,
            workers=int(self.args.workers),
            imgsz=int(self.args.imgsz),
            data_audit_dir=self.data_access_attempt_dir,
            verify_payload_on_read=self.preflight_mode,
            max_batches=PREFLIGHT_MICROBATCHES if self.preflight_mode else None,
        )
        expected_loader_batches = (
            PREFLIGHT_MICROBATCHES
            if self.preflight_mode
            else EXPECTED_STEPS_PER_EPOCH
        )
        if len(self.target_loader) != expected_loader_batches or len(self.target_loader.dataset) != EXPECTED_TARGET_POOL_SIZE:
            raise FormalStop("STOP_FIXED_TARGET_LOADER_CARDINALITY")
        restore = self._restore_start_state()
        self.ema.ema.eval()
        if any(parameter.requires_grad for parameter in self.ema.ema.parameters()):
            raise FormalStop("STOP_TEACHER_REQUIRES_GRAD")
        if self.logging_enabled and not self.preflight_mode and "T0" not in self._snapshot_names:
            self._install_t0_hook()
        audit = {
            "schema": "m1_seed42_stage1_start_restore_audit_v1",
            "status": "PASS",
            "branch": self.stage1_branch,
            "start_state_path": str(self.start_state_path),
            "start_state_file_sha256": sha256_file(self.start_state_path),
            "completed_start_epoch": self.completed_start_epoch,
            "restore": restore,
            "run_contract_sha256": self.run_contract_sha256,
            "e70_bootstrap_not_lineage_parent": True,
            "teacher_original_view": True,
            "online_powell_calls": 0,
            "target_dev_read_count": 0,
            "target_test_read_count": 0,
            "timestamp": now_iso(),
        }
        atomic_json(Path(self.save_dir) / "START_RESTORE_AUDIT.json", audit)

    def _install_t0_hook(self) -> None:
        if self._t0_hook_handle is not None:
            raise FormalStop("STOP_T0_HOOK_ALREADY_INSTALLED")

        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if self._t0_hook_fire_count != 0 or "T0" in self._snapshot_names:
                raise FormalStop("STOP_T0_HOOK_MULTIPLE_FIRE")
            if self.last_target_raw is None or not inputs or not torch.is_tensor(inputs[0]):
                raise FormalStop("STOP_T0_NOT_AT_TEACHER_TARGET_FORWARD")
            target_tensor = inputs[0]
            if target_tensor.ndim != 4 or int(target_tensor.shape[0]) != TARGET_BATCH:
                raise FormalStop("STOP_T0_TEACHER_FORWARD_SHAPE")
            self._t0_hook_fire_count += 1
            self._save_snapshot("T0", event_kind="TEACHER_FORWARD_ENTRY")
            handle = self._t0_hook_handle
            if handle is not None:
                handle.remove()
                self._t0_hook_handle = None

        self._t0_hook_handle = self.ema.ema.register_forward_pre_hook(hook)

    def _observing_generate(self, output: Any, **kwargs: Any):
        self._observer_calls_batch += 1
        self.observer_original_call_count += 1
        pseudo = generate_high_confidence_pseudo(output, **kwargs)
        returned = pseudo
        if returned is not pseudo:
            raise FormalStop("STOP_OBSERVER_RETURN_IDENTITY")
        self.observer_return_identity_count += 1
        self._observed_pseudo = pseudo
        return returned

    def _protocol_on_epoch_start(self, trainer: "Stage1Trainer") -> None:
        super()._protocol_on_epoch_start(trainer)
        if trainer.protocol_current_total_epoch not in EPOCHS:
            raise FormalStop("STOP_FORMAL_EPOCH_RANGE")
        trainer.last_target_raw = None
        if trainer.protocol_target_iterator is None:
            raise FormalStop("STOP_ACTIVE_TARGET_ITERATOR_MISSING")
        trainer.protocol_target_iterator = CapturingIterator(trainer.protocol_target_iterator, self)
        self._epoch_trace_box_rows = []
        self._epoch_trace_batch_rows = []
        self._preflight_common_rows = []
        self.fixed_cache_runtime_rows = []
        self.fixed_exposure_runtime_rows = []

    def _protocol_on_batch_start(self, trainer: "Stage1Trainer") -> None:
        super()._protocol_on_batch_start(trainer)
        if self._first_window and trainer.protocol_batch_index == 0:
            trainer.accumulate = self.stage1_start_global_step + 2
        else:
            trainer.accumulate = ACCUMULATION
        if int(trainer.batch_size) != SOURCE_BATCH or int(trainer.accumulate) not in {ACCUMULATION, self.stage1_start_global_step + 2}:
            raise FormalStop("STOP_ACCUMULATION_PROTOCOL")

    def _capture_pseudo_trace(self, target_raw: dict[str, Any], pseudo: Any, source_batch_index: int) -> None:
        uids = list(target_raw["protocol_b_target_uid"])
        canonical = list(target_raw["fixed_canonical_uid"])
        ordinals = [int(value) for value in target_raw["fixed_formal_visit_index"]]
        boxes = pseudo.bboxes.detach().cpu().contiguous()
        classes = pseudo.cls.detach().cpu().contiguous()
        batch_idx = pseudo.batch_idx.detach().cpu().contiguous()
        scores = [float(value) for value in pseudo.stats["confidences"]]
        if len(scores) != len(boxes) or len(classes) != len(boxes) or len(batch_idx) != len(boxes):
            raise FormalStop("STOP_HPL_TRACE_CARDINALITY")
        batch_row = {
            "schema": "m1_seed42_stage1_pseudo_batch_trace_v1",
            "branch": self.stage1_branch,
            "epoch": int(self.protocol_current_total_epoch),
            "source_batch_index": int(source_batch_index),
            "target_uid_json": json.dumps(uids, separators=(",", ":")),
            "canonical_uid_json": json.dumps(canonical, separators=(",", ":")),
            "formal_visit_index_json": json.dumps(ordinals, separators=(",", ":")),
            "view_seed_json": json.dumps([int(v) for v in target_raw["protocol_b_target_view_seed"]], separators=(",", ":")),
            "fixed_exposure_active_json": json.dumps([int(v) for v in target_raw["fixed_exposure_active"]], separators=(",", ":")),
            "fixed_transform_applied_json": json.dumps([int(v) for v in target_raw["fixed_transform_applied"]], separators=(",", ":")),
            "teacher_uint8_sha256": tensor_bytes_sha(target_raw["img"]),
            "student_uint8_sha256": tensor_bytes_sha(target_raw["fixed_student_img"]),
            "pseudo_cls_sha256": tensor_bytes_sha(classes),
            "pseudo_bboxes_sha256": tensor_bytes_sha(boxes),
            "pseudo_batch_idx_sha256": tensor_bytes_sha(batch_idx),
            "pseudo_scores_f32le_sha256": hashlib.sha256(b"".join(struct.pack("<f", value) for value in scores)).hexdigest(),
            "pseudo_final_boxes": int(len(boxes)),
            "pseudo_pre_nms_per_image_json": json.dumps(
                [int(v) for v in pseudo.stats["pre_nms_per_image"]], separators=(",", ":")
            ),
            "pseudo_post_nms_per_image_json": json.dumps(
                [int(v) for v in pseudo.stats["post_nms_per_image"]], separators=(",", ":")
            ),
            "pseudo_final_per_image_json": json.dumps(
                [int(v) for v in pseudo.stats["final_per_image"]], separators=(",", ":")
            ),
            "pre_nms_candidates": int(pseudo.stats["pre_nms_candidates"]),
            "post_nms_boxes": int(pseudo.stats["post_nms_boxes"]),
            "area_filter_removed": int(pseudo.stats["area_filter_removed"]),
            "aspect_filter_removed": int(pseudo.stats["aspect_filter_removed"]),
            "observer_original_call_count_this_batch": int(self._observer_calls_batch),
            "observer_returned_same_object": True,
            "extra_teacher_forward_count": 0,
            "extra_nms_call_count": 0,
        }
        self._epoch_trace_batch_rows.append(batch_row)
        for box_index in range(len(boxes)):
            image_index = int(batch_idx[box_index])
            if not 0 <= image_index < TARGET_BATCH:
                raise FormalStop("STOP_HPL_BATCH_INDEX")
            values = [float(v) for v in boxes[box_index].tolist()]
            self._epoch_trace_box_rows.append(
                {
                    "schema": "m1_seed42_stage1_hpl_box_trace_v1",
                    "branch": self.stage1_branch,
                    "epoch": int(self.protocol_current_total_epoch),
                    "source_batch_index": int(source_batch_index),
                    "formal_visit_index": ordinals[image_index],
                    "target_position": image_index,
                    "schedule_uid": uids[image_index],
                    "canonical_uid": canonical[image_index],
                    "box_index_in_batch": box_index,
                    "class_id": float(classes[box_index, 0]),
                    "score": scores[box_index],
                    "score_f32_le_hex": float32_hex(scores[box_index]),
                    "x_center": values[0], "y_center": values[1],
                    "width": values[2], "height": values[3],
                    "area": values[2] * values[3],
                    "box_f32_le_base64": base64.b64encode(boxes[box_index].numpy().astype("<f4", copy=False).tobytes()).decode("ascii"),
                }
            )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.last_target_raw = None
        self._observed_pseudo = None
        self._observer_calls_batch = 0
        previous = _b2_module.generate_high_confidence_pseudo
        if previous is not generate_high_confidence_pseudo:
            raise FormalStop("STOP_HPL_GLOBAL_ALREADY_PATCHED")
        confidence_start = len(self.protocol_epoch_pseudo_confidences)
        area_start = len(self.protocol_epoch_pseudo_areas)
        # Frozen Route-A resource protocol (authoritative worker lines
        # 1307-1318): Ultralytics selects its NMS backend from the root
        # torchvision registry key.  Hide only that key during the exact B2
        # HPL call, then restore the identical module object in finally.
        torchvision_module = sys.modules.pop("torchvision", None)
        if torchvision_module is None:
            raise FormalStop("STOP_TORCHVISION_TRANSITIVE_MODULE_NOT_LOADED")
        self.hpl_torchvision_scope_call_count += 1
        try:
            if "torchvision" in sys.modules:
                raise FormalStop("STOP_HPL_NMS_BACKEND_IMPORT_ORDER_ASSERTION")
            if self.logging_enabled:
                _b2_module.generate_high_confidence_pseudo = self._observing_generate
                try:
                    source_batch = super().preprocess_batch(batch)
                finally:
                    _b2_module.generate_high_confidence_pseudo = previous
                if self._observer_calls_batch != 1 or self._observed_pseudo is None:
                    raise FormalStop("STOP_OBSERVER_NOT_EXACTLY_ONCE")
            else:
                source_batch = super().preprocess_batch(batch)
                if self._observer_calls_batch != 0 or self._observed_pseudo is not None:
                    raise FormalStop("STOP_LOGGING_OFF_OBSERVER_INTERVENTION")
        finally:
            scope_imported_torchvision = "torchvision" in sys.modules
            sys.modules["torchvision"] = torchvision_module
            if sys.modules.get("torchvision") is not torchvision_module:
                raise FormalStop("STOP_TORCHVISION_ROOT_IDENTITY_RESTORE")
            self.hpl_torchvision_identity_restore_count += 1
            if scope_imported_torchvision:
                raise FormalStop("STOP_HPL_SCOPE_IMPORTED_TORCHVISION")
        target_raw = self.last_target_raw
        target_batch = source_batch.get("_protocol_b_target_batch")
        if target_raw is None or target_batch is None:
            raise FormalStop("STOP_TARGET_RAW_OR_BATCH_MISSING")
        self.teacher_target_forward_count += 1
        batch_confidences = self.protocol_epoch_pseudo_confidences[
            confidence_start:
        ]
        batch_areas = self.protocol_epoch_pseudo_areas[area_start:]
        confidence_f32_sha = hashlib.sha256(
            b"".join(struct.pack("<f", float(value)) for value in batch_confidences)
        ).hexdigest()
        area_f32_sha = hashlib.sha256(
            b"".join(struct.pack("<f", float(value)) for value in batch_areas)
        ).hexdigest()
        if len(batch_confidences) != int(target_batch["_protocol_b_pseudo_count"]):
            raise FormalStop("STOP_HPL_SCORE_SLICE_CARDINALITY")
        if self.logging_enabled and (
            target_batch["cls"] is not self._observed_pseudo.cls
            or target_batch["bboxes"] is not self._observed_pseudo.bboxes
            or target_batch["batch_idx"] is not self._observed_pseudo.batch_idx
        ):
            raise FormalStop("STOP_OBSERVER_OBJECT_NOT_USED_BY_LOSS")
        teacher_uint8 = target_raw["img"]
        student_uint8 = target_raw["fixed_student_img"]
        if tensor_bytes_sha(teacher_uint8) != self._b2_pending["target_image_hash"]:
            raise FormalStop("STOP_TEACHER_ORIGINAL_VIEW_HASH")
        if self.stage1_branch == "CONTROL" and not torch.equal(teacher_uint8, student_uint8):
            raise FormalStop("STOP_CONTROL_NOT_IDENTITY")
        if any(int(v) != 0 for v in target_raw["fixed_padding_max_abs_byte_delta"]):
            raise FormalStop("STOP_FIXED_PADDING_NONZERO")
        target_batch["img"] = student_uint8.to(
            self.device, non_blocking=self.device.type == "cuda"
        ).float() / 255.0
        source_batch_index = int(self._b2_pending["source_batch"])
        pseudo_count = int(target_batch["_protocol_b_pseudo_count"])
        target_term_included = bool(self.protocol_lambda_u > 0.0 and pseudo_count > 0)
        self.target_unsup_microbatch_count += int(target_term_included)
        self._window_target_term_included |= target_term_included
        self._window_target_microbatch_count += int(target_term_included)
        if self.logging_enabled:
            self._capture_pseudo_trace(target_raw, self._observed_pseudo, source_batch_index)
        self._preflight_common_rows.append(
            {
                "epoch": int(self.protocol_current_total_epoch),
                "source_batch_index": source_batch_index,
                "target_uid_json": json.dumps(
                    list(target_raw["protocol_b_target_uid"]), separators=(",", ":")
                ),
                "formal_visit_index_json": json.dumps(
                    [int(v) for v in target_raw["fixed_formal_visit_index"]],
                    separators=(",", ":"),
                ),
                "fixed_exposure_active_json": json.dumps(
                    [int(v) for v in target_raw["fixed_exposure_active"]],
                    separators=(",", ":"),
                ),
                "teacher_uint8_sha256": tensor_bytes_sha(teacher_uint8),
                "student_uint8_sha256": tensor_bytes_sha(student_uint8),
                "hpl_cls_sha256": tensor_bytes_sha(target_batch["cls"]),
                "hpl_bboxes_sha256": tensor_bytes_sha(target_batch["bboxes"]),
                "hpl_batch_idx_sha256": tensor_bytes_sha(target_batch["batch_idx"]),
                "hpl_score_count": len(batch_confidences),
                "hpl_scores_f32le_sha256": confidence_f32_sha,
                "hpl_area_count": len(batch_areas),
                "hpl_areas_f32le_sha256": area_f32_sha,
                "pseudo_count": pseudo_count,
                "torchvision_transitive_loaded_outside_hpl_scope": True,
                "torchvision_root_present_inside_hpl_scope": False,
                "hpl_nms_backend": "ultralytics.utils.nms.TorchNMS.nms",
                "torchvision_root_identity_restored": True,
                "cache_payload_validation_count": sum(
                    int(v) for v in target_raw["fixed_cache_payload_validation"]
                ),
            }
        )
        runtime_common = {
            "schema": "m1_seed42_stage1_fixed_exposure_runtime_v1",
            "branch": self.stage1_branch,
            "epoch": int(self.protocol_current_total_epoch),
            "source_batch_index": source_batch_index,
            "formal_visit_indices_json": json.dumps(
                [int(v) for v in target_raw["fixed_formal_visit_index"]], separators=(",", ":")
            ),
            "schedule_uids_json": json.dumps(
                list(target_raw["protocol_b_target_uid"]), separators=(",", ":")
            ),
            "canonical_uids_json": json.dumps(
                list(target_raw["fixed_canonical_uid"]), separators=(",", ":")
            ),
            "fixed_active_count": sum(int(v) for v in target_raw["fixed_exposure_active"]),
            "fixed_identity_count": sum(int(v) for v in target_raw["fixed_exposure_identity"]),
            "fixed_transform_applied_count": sum(int(v) for v in target_raw["fixed_transform_applied"]),
            "exposure_lookup_mismatch_count": 0,
        }
        self.fixed_exposure_runtime_rows.append(runtime_common)
        if self.stage1_branch == "NATIVE_A_FIXED":
            self.fixed_cache_runtime_rows.append(
                {
                    "schema": "m1_seed42_stage1_fixed_cache_runtime_v1",
                    "branch": self.stage1_branch,
                    "epoch": int(self.protocol_current_total_epoch),
                    "source_batch_index": source_batch_index,
                    "cache_hit_count": runtime_common["fixed_transform_applied_count"],
                    "cache_miss_count": 0,
                    "cache_fallback_count": 0,
                    "online_theta_solve_count": 0,
                    "cache_build_authority_sha256": EXPECTED_INPUT_SHA256[
                        "fixed_cache_build"
                    ],
                    "fixed_pretrain_authority_sha256": EXPECTED_INPUT_SHA256[
                        "fixed_pretrain_final_status"
                    ],
                    "cache_read_seconds": sum(
                        float(v) for v in target_raw["fixed_cache_read_seconds"]
                    ),
                    "cache_payload_validation_seconds": sum(
                        float(v)
                        for v in target_raw["fixed_cache_payload_validation_seconds"]
                    ),
                    "cache_payload_validation_count": sum(
                        int(v) for v in target_raw["fixed_cache_payload_validation"]
                    ),
                    "student_letterbox_seconds": sum(
                        float(v) for v in target_raw["fixed_student_letterbox_seconds"]
                    ),
                    "identity_cache_access_count": sum(
                        int(not bool(active) and (float(read_seconds) != 0.0))
                        for active, read_seconds in zip(
                            target_raw["fixed_transform_applied"],
                            target_raw["fixed_cache_read_seconds"],
                            strict=True,
                        )
                    ),
                    "output_bgr_sha256_json": json.dumps(
                        list(target_raw["fixed_output_bgr_sha256"]), separators=(",", ":")
                    ),
                    "cache_array_bytes_sha256_json": json.dumps(
                        list(target_raw["fixed_cache_array_bytes_sha256"]), separators=(",", ":")
                    ),
                    "padding_max_abs_byte_delta": max(
                        int(v) for v in target_raw["fixed_padding_max_abs_byte_delta"]
                    ),
                }
            )
        self._b2_pending.update(
            {
                "stage1_branch": self.stage1_branch,
                "target_term_included": int(target_term_included),
                "fixed_exposure_active_count": sum(int(v) for v in target_raw["fixed_exposure_active"]),
                "fixed_exposure_identity_count": sum(int(v) for v in target_raw["fixed_exposure_identity"]),
                "fixed_transform_applied_count": sum(int(v) for v in target_raw["fixed_transform_applied"]),
                "teacher_original_uint8_sha256": tensor_bytes_sha(teacher_uint8),
                "student_target_uint8_sha256": tensor_bytes_sha(student_uint8),
                "online_powell_call_count": 0,
                "observer_original_call_count": int(self.logging_enabled),
                "observer_returned_same_object": int(self.logging_enabled),
            }
        )
        return source_batch

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        scaler_state = self.scaler._per_optimizer_states.get(id(self.optimizer), {})
        found_inf_map = scaler_state.get("found_inf_per_device", {})
        found_inf_values = {
            str(device): float(value.detach().cpu()) for device, value in found_inf_map.items()
        }
        found_inf = any(value != 0.0 for value in found_inf_values.values())
        grad_sha = gradient_state_sha(self.model) if self.preflight_mode else ""
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm) and not found_inf:
            raise FloatingPointError("STOP_NONFINITE_GRAD_WITHOUT_FOUND_INF")
        self.protocol_last_grad_norm = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        underlying_called = {"value": False}
        original_step = self.optimizer.step

        def observed_step(*args: Any, **kwargs: Any):
            underlying_called["value"] = True
            return original_step(*args, **kwargs)

        scale_before = float(self.scaler.get_scale())
        self.optimizer.step = observed_step  # type: ignore[method-assign]
        try:
            self.scaler.step(self.optimizer)
        finally:
            self.optimizer.step = original_step  # type: ignore[method-assign]
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self.protocol_optimizer_steps += 1
        self._first_window = False
        self.protocol_last_optimizer_window_found_inf = found_inf
        self.protocol_last_optimizer_parameter_update_skipped = not underlying_called["value"]
        self.protocol_last_found_inf_json = json.dumps(found_inf_values, sort_keys=True, separators=(",", ":"))
        self.amp_found_inf_windows += int(found_inf)
        self.actual_unskipped_optimizer_updates += int(underlying_called["value"])
        target_contributing = bool(underlying_called["value"] and self._window_target_term_included)
        self.target_contributing_optimizer_updates += int(target_contributing)
        self._last_window_target_included = bool(self._window_target_term_included)
        self._last_step_underlying_called = bool(underlying_called["value"])
        self._last_step_target_contributing = target_contributing
        self._last_step_grad_sha = grad_sha
        self._last_step_scale_before = scale_before
        self._last_step_scale_after = scale_after
        self.optimizer_update_rows.append(
            {
                "schema": "m1_seed42_stage1_optimizer_update_trace_v1",
                "branch": self.stage1_branch,
                "epoch": int(self.protocol_current_total_epoch),
                "optimizer_window_this_branch": int(self.protocol_optimizer_steps),
                "optimizer_window_in_epoch": (int(self.protocol_optimizer_steps) - 1)
                % WINDOWS_PER_EPOCH,
                "underlying_optimizer_step_called": int(underlying_called["value"]),
                "found_inf": int(found_inf),
                "found_inf_per_device": self.protocol_last_found_inf_json,
                "window_target_term_included": int(self._window_target_term_included),
                "window_target_unsup_microbatch_count": int(
                    self._window_target_microbatch_count
                ),
                "target_contributing_optimizer_update": int(target_contributing),
                "target_contributing_optimizer_updates_cumulative": int(
                    self.target_contributing_optimizer_updates
                ),
                "actual_unskipped_optimizer_updates_cumulative": int(
                    self.actual_unskipped_optimizer_updates
                ),
                "ema_updates_after": int(self.ema.updates),
                "grad_state_sha256_before_step": grad_sha,
                "grad_norm_before_clip": self.protocol_last_grad_norm,
                "grad_scaler_scale_before": scale_before,
                "grad_scaler_scale_after": scale_after,
            }
        )
        self._window_target_term_included = False
        self._window_target_microbatch_count = 0
        event = T_EVENTS.get(self.target_contributing_optimizer_updates)
        if event is not None and event not in self._snapshot_names:
            self._save_snapshot(event, event_kind="TARGET_CONTRIBUTING_OPTIMIZER_UPDATE")

    def _protocol_b2_on_batch_end(self, trainer: "Stage1Trainer") -> None:
        super()._protocol_b2_on_batch_end(trainer)
        row = trainer.protocol_b2_batch_rows[-1]
        diagnostics = dict(unwrap_model(trainer.model).protocol_b_last_diagnostics)
        row.update(
            {
                "target_bn_buffer_max_abs_delta": float(diagnostics["target_forward_bn_buffer_max_abs_delta"]),
                "target_bn_nonfloat_mismatch_count": int(diagnostics["target_forward_bn_nonfloat_mismatch_count"]),
                "optimizer_underlying_step_called": int(self._last_step_underlying_called) if row["optimizer_step_occurred"] else 0,
                "optimizer_window_found_inf": int(self.protocol_last_optimizer_window_found_inf) if row["optimizer_step_occurred"] else 0,
                "found_inf_per_device": self.protocol_last_found_inf_json if row["optimizer_step_occurred"] else "{}",
                "target_contributing_optimizer_updates": self.target_contributing_optimizer_updates,
                "actual_unskipped_optimizer_updates": self.actual_unskipped_optimizer_updates,
                "gradient_state_sha256_before_step": self._last_step_grad_sha if row["optimizer_step_occurred"] else "",
                "grad_scaler_scale_before": self._last_step_scale_before if row["optimizer_step_occurred"] else float(self.scaler.get_scale()),
                "grad_scaler_scale_after": self._last_step_scale_after if row["optimizer_step_occurred"] else float(self.scaler.get_scale()),
            }
        )
        if row["target_bn_buffer_max_abs_delta"] != 0.0 or row["target_bn_nonfloat_mismatch_count"] != 0:
            raise FormalStop("STOP_TARGET_BN_BEHAVIOR")
        common = self._preflight_common_rows[-1]
        common.update(
            {
                "global_microbatch_index": E71_GLOBAL_STEP
                + (int(self.protocol_current_total_epoch) - 72)
                * EXPECTED_STEPS_PER_EPOCH
                + int(common["source_batch_index"]),
                "global_step_after_microbatch": E71_GLOBAL_STEP
                + (int(self.protocol_current_total_epoch) - 72)
                * EXPECTED_STEPS_PER_EPOCH
                + int(common["source_batch_index"])
                + 1,
                "source_loss_total": float(diagnostics["source_loss_total"]),
                "target_loss_total": float(diagnostics["target_loss_total"]),
                "weighted_target_loss": float(diagnostics["weighted_target_loss"]),
                "combined_loss_total": float(diagnostics["combined_loss_total"]),
                "target_term_included": int(row["target_term_included"]),
                "optimizer_step_occurred": int(row["optimizer_step_occurred"]),
                "optimizer_underlying_step_called": int(row["optimizer_underlying_step_called"]),
                "ema_update_occurred": int(row["ema_update_occurred"]),
                "target_contributing_optimizer_updates": self.target_contributing_optimizer_updates,
                "target_unsupervised_optimizer_update_count": self.target_contributing_optimizer_updates,
            }
        )
        if self.logging_enabled:
            trace = self._epoch_trace_batch_rows[-1]
            trace.update(common)

    def _snapshot_payload(self, event_name: str, role: str, state: dict[str, Any]) -> dict[str, Any]:
        global_step = self._snapshot_global_step(event_name)
        return {
            "format": "m1_seed42_stage1_inference_snapshot_v1",
            "experiment": EXPERIMENT,
            "branch": self.stage1_branch,
            "event_name": event_name,
            "role": role,
            "epoch": int(self.protocol_current_total_epoch),
            "global_step": global_step,
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
            "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
            "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
            "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
            "target_unsup_optimizer_update_count": int(self.target_contributing_optimizer_updates),
            "teacher_target_forward_count": int(self.teacher_target_forward_count),
            "observer_return_identity_count": int(self.observer_return_identity_count),
            "hpl_torchvision_scope_call_count": int(
                self.hpl_torchvision_scope_call_count
            ),
            "hpl_torchvision_identity_restore_count": int(
                self.hpl_torchvision_identity_restore_count
            ),
            "ema_updates": int(self.ema.updates),
            "inference_only": True,
            "resume_forbidden": True,
            "model_yaml": "yolo11n.yaml",
            "nc": 1,
            "model_state_dict": deepcopy(state),
            "tensor_canonical_sha256": tensor_state_hash(state),
            "run_contract_sha256": self.run_contract_sha256,
        }

    def _snapshot_global_step(self, event_name: str) -> int:
        epoch = int(self.protocol_current_total_epoch)
        epoch_start = E71_GLOBAL_STEP + (epoch - 72) * EXPECTED_STEPS_PER_EPOCH
        if event_name == "T0":
            return epoch_start
        if event_name.startswith("T"):
            # Frozen B1 preprocess increments protocol_batch_index before the
            # training loop reaches optimizer_step.  At this callback it is
            # already the exact number of completed microbatches.
            return epoch_start + int(self.protocol_batch_index)
        return E71_GLOBAL_STEP + (epoch - 71) * EXPECTED_STEPS_PER_EPOCH

    def _save_snapshot(self, event_name: str, *, event_kind: str) -> None:
        if event_name in self._snapshot_names:
            raise FormalStop(f"STOP_DUPLICATE_SNAPSHOT:{event_name}")
        raw_state = unwrap_model(self.model).state_dict()
        ema_state = self.ema.ema.state_dict()
        before = {
            "raw_tensor_sha256": tensor_state_hash(raw_state),
            "ema_tensor_sha256": tensor_state_hash(ema_state),
            "optimizer_state_sha256": recursive_state_sha256(
                self.optimizer.state_dict()
            ),
            "scheduler_state_sha256": recursive_state_sha256(
                self.scheduler.state_dict()
            ),
            "scaler_state_sha256": recursive_state_sha256(self.scaler.state_dict()),
            "rng_state_sha256": recursive_state_sha256(rng_state_payload()),
            "rng_hashes": rng_state_hashes(),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "ema_updates": int(self.ema.updates),
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
        }
        directory = (
            Path(self.save_dir)
            / "inference_state"
            / "attempts"
            / self.run_attempt_id
        )
        raw_path = directory / f"{event_name}_raw.pt"
        ema_path = directory / f"{event_name}_ema.pt"
        if raw_path.exists() or ema_path.exists():
            raise FormalStop(f"STOP_SNAPSHOT_PATH_EXISTS:{event_name}")
        atomic_torch_save(self._snapshot_payload(event_name, "raw", raw_state), raw_path)
        atomic_torch_save(self._snapshot_payload(event_name, "ema", ema_state), ema_path)
        after = {
            "raw_tensor_sha256": tensor_state_hash(
                unwrap_model(self.model).state_dict()
            ),
            "ema_tensor_sha256": tensor_state_hash(self.ema.ema.state_dict()),
            "optimizer_state_sha256": recursive_state_sha256(
                self.optimizer.state_dict()
            ),
            "scheduler_state_sha256": recursive_state_sha256(
                self.scheduler.state_dict()
            ),
            "scaler_state_sha256": recursive_state_sha256(self.scaler.state_dict()),
            "rng_state_sha256": recursive_state_sha256(rng_state_payload()),
            "rng_hashes": rng_state_hashes(),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "ema_updates": int(self.ema.updates),
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
        }
        if before != after:
            raise FormalStop(f"STOP_SNAPSHOT_STATE_INTERVENTION:{event_name}")
        nonintervention_path = (
            Path(self.save_dir)
            / "raw_trace"
            / "attempts"
            / self.run_attempt_id
            / f"snapshot_nonintervention_{event_name}.json"
        )
        atomic_json(
            nonintervention_path,
            {
                "schema": "m1_seed42_stage1_snapshot_nonintervention_v1",
                "status": "PASS",
                "all_pass": True,
                "branch": self.stage1_branch,
                "event_name": event_name,
                "event_kind": event_kind,
                "before": before,
                "after": after,
                "all_state_rng_scheduler_exact": True,
                "extra_forward_count": 0,
                "extra_nms_count": 0,
                "target_dev_read_count": 0,
                "target_test_read_count": 0,
                "timestamp": now_iso(),
            },
        )
        row = {
            "schema": "m1_seed42_stage1_checkpoint_manifest_v1",
            "branch": self.stage1_branch,
            "event_kind": event_kind,
            "event_name": event_name,
            "epoch": int(self.protocol_current_total_epoch),
            "global_step": self._snapshot_global_step(event_name),
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
            "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
            "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
            "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
            "target_unsup_optimizer_update_count": int(
                self.target_contributing_optimizer_updates
            ),
            "teacher_target_forward_count": int(self.teacher_target_forward_count),
            "observer_return_identity_count": int(self.observer_return_identity_count),
            "hpl_torchvision_scope_call_count": int(
                self.hpl_torchvision_scope_call_count
            ),
            "hpl_torchvision_identity_restore_count": int(
                self.hpl_torchvision_identity_restore_count
            ),
            "ema_updates": int(self.ema.updates),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "full_state_path": "",
            "full_state_file_sha256": "",
            "raw_inference_path": str(raw_path.resolve()),
            "raw_inference_file_sha256": sha256_file(raw_path),
            "raw_tensor_sha256": before["raw_tensor_sha256"],
            "ema_inference_path": str(ema_path.resolve()),
            "ema_inference_file_sha256": sha256_file(ema_path),
            "ema_tensor_sha256": before["ema_tensor_sha256"],
            "optimizer_state_sha256": before["optimizer_state_sha256"],
            "scheduler_state_sha256": before["scheduler_state_sha256"],
            "scaler_state_sha256": before["scaler_state_sha256"],
            "rng_state_sha256": before["rng_state_sha256"],
            "run_contract_sha256": self.run_contract_sha256,
            "resume_allowed": False,
            "snapshot_save_nonintervention_exact": True,
            "snapshot_before_state_sha256": canonical_json_sha256(before),
            "snapshot_after_state_sha256": canonical_json_sha256(after),
            "snapshot_rng_hashes_before_json": json.dumps(
                before["rng_hashes"], sort_keys=True, separators=(",", ":")
            ),
            "snapshot_rng_hashes_after_json": json.dumps(
                after["rng_hashes"], sort_keys=True, separators=(",", ":")
            ),
            "snapshot_nonintervention_audit_path": str(
                nonintervention_path.resolve()
            ),
            "snapshot_nonintervention_audit_sha256": sha256_file(
                nonintervention_path
            ),
        }
        self._snapshot_rows.append(row)
        self._snapshot_names.add(event_name)
        atomic_csv(
            Path(self.save_dir) / "CHECKPOINT_MANIFEST.csv",
            [*self._snapshot_rows, *self.checkpoint_rows, *self.recovery_rows],
        )
        atomic_csv(
            Path(self.save_dir) / "COLD_START_SNAPSHOT_MANIFEST.csv",
            self._snapshot_rows,
        )

    def _capture_completed_epoch_data_access(self, epoch: int) -> None:
        epoch_dir = self.data_access_attempt_dir / f"e{epoch:03d}"
        summaries = sorted(
            epoch_dir.glob("process_*_DATA_ACCESS_SUMMARY.json")
        )
        if len(summaries) != 2:
            raise FormalStop(
                f"STOP_EPOCH_WORKER_DATA_ACCESS_SUMMARY_COUNT:{epoch}:"
                f"{len(summaries)}"
            )
        summary_rows: list[dict[str, Any]] = []
        forbidden_fields = (
            "formal_forbidden_event_count",
            "target_adapt_label_file_open_count",
            "target_adapt_gt_cache_open_count",
            "target_dev_gt_reads",
            "target_test_reads",
            "dataset_mutation_attempts",
        )
        for path in summaries:
            summary = _load_json(path)
            event_path = inside_root(Path(str(summary.get("event_path", ""))))
            counters = summary.get("counters")
            if not isinstance(counters, dict):
                raise FormalStop(f"STOP_WORKER_DATA_ACCESS_COUNTERS:{path}")
            checks = {
                "schema": summary.get("schema")
                == "m1_seed42_stage1_worker_data_access_summary_v1",
                "branch": summary.get("branch") == self.stage1_branch,
                "epoch": int(summary.get("epoch", -1)) == epoch,
                "trigger": int(summary.get("last_export_trigger_batch_index", -1))
                in {550, 551},
                "event_inside_epoch": event_path.is_relative_to(epoch_dir.resolve()),
                "event_exists": event_path.is_file(),
                "event_sha": event_path.is_file()
                and sha256_file(event_path)
                == summary.get("event_file_sha256"),
                "reads": int(
                    counters.get("target_adapt_image_read_completed_count", -1)
                )
                == TARGET_VISITS_PER_EPOCH // 2,
                "forbidden_zero": all(
                    int(counters.get(field, -1)) == 0
                    for field in forbidden_fields
                ),
            }
            if not all(checks.values()):
                raise FormalStop(
                    f"STOP_EPOCH_WORKER_DATA_ACCESS:{epoch}:{path}:{checks}"
                )
            summary_rows.append(
                {
                    "summary_path": str(path.resolve()),
                    "summary_file_sha256": sha256_file(path),
                    "event_path": str(event_path),
                    "event_file_sha256": str(summary["event_file_sha256"]),
                    "process_id": int(summary["process_id"]),
                    "image_reads": int(
                        counters["target_adapt_image_read_completed_count"]
                    ),
                }
            )
        if any(
            int(row["epoch"]) == epoch
            for row in self.completed_data_access_epoch_rows
        ):
            raise FormalStop(f"STOP_DUPLICATE_COMPLETED_DATA_ACCESS_EPOCH:{epoch}")
        self.completed_data_access_epoch_rows.append(
            {
                "schema": "m1_seed42_stage1_completed_epoch_data_access_v1",
                "branch": self.stage1_branch,
                "epoch": epoch,
                "attempt_dir": str(self.data_access_attempt_dir.resolve()),
                "worker_count": len(summary_rows),
                "target_adapt_image_read_count": sum(
                    int(row["image_reads"]) for row in summary_rows
                ),
                "worker_summaries_json": json.dumps(
                    summary_rows, sort_keys=True, separators=(",", ":")
                ),
                "all_pass": True,
            }
        )
        atomic_csv(
            Path(self.save_dir) / "DATA_ACCESS_COMPLETED_EPOCHS.csv",
            self.completed_data_access_epoch_rows,
        )

    def _protocol_on_epoch_end(self, trainer: "Stage1Trainer") -> None:
        super()._protocol_on_epoch_end(trainer)
        epoch = int(self.protocol_current_total_epoch)
        self._capture_completed_epoch_data_access(epoch)
        if len(self._epoch_trace_batch_rows) != EXPECTED_STEPS_PER_EPOCH:
            raise FormalStop(f"STOP_TRACE_BATCH_COUNT_EPOCH:{epoch}:{len(self._epoch_trace_batch_rows)}")
        raw = (
            Path(self.save_dir)
            / "raw_trace"
            / "attempts"
            / self.run_attempt_id
        )
        box_path = raw / f"hpl_boxes_e{epoch:03d}.jsonl"
        batch_path = raw / f"pseudo_batches_e{epoch:03d}.jsonl"
        exposure_runtime_path = raw / f"fixed_exposure_runtime_e{epoch:03d}.jsonl"
        cache_runtime_path = raw / f"fixed_cache_runtime_e{epoch:03d}.jsonl"
        write_jsonl_exclusive(box_path, self._epoch_trace_box_rows)
        write_jsonl_exclusive(batch_path, self._epoch_trace_batch_rows)
        write_jsonl_exclusive(
            exposure_runtime_path,
            self.fixed_exposure_runtime_rows,
        )
        if self.stage1_branch == "NATIVE_A_FIXED":
            write_jsonl_exclusive(
                cache_runtime_path,
                self.fixed_cache_runtime_rows,
            )
        # Canonical state identities are taken once, at the complete epoch
        # boundary after all optimizer/EMA updates and before checkpoint I/O.
        # This is the same callback node in both branches and performs no
        # forward, NMS, RNG draw, optimizer step, or mutable-state write.
        raw_state_sha = tensor_state_hash(unwrap_model(self.model).state_dict())
        ema_state_sha = tensor_state_hash(self.ema.ema.state_dict())
        optimizer_state_sha = recursive_state_sha256(self.optimizer.state_dict())
        scheduler_state_sha = recursive_state_sha256(self.scheduler.state_dict())
        scaler_state_sha = recursive_state_sha256(self.scaler.state_dict())
        rng_state_sha = recursive_state_sha256(rng_state_payload())
        per_image_counts: list[int] = []
        for batch_row in self._epoch_trace_batch_rows:
            per_image_counts.extend(
                int(value)
                for value in json.loads(batch_row["pseudo_final_per_image_json"])
            )
        image_histogram: dict[str, int] = {}
        for count in per_image_counts:
            image_histogram[str(count)] = image_histogram.get(str(count), 0) + 1
        confidence_edges = np.asarray(
            [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0000001],
            dtype=np.float64,
        )
        confidence_histogram = np.histogram(
            np.asarray(self.protocol_epoch_pseudo_confidences, dtype=np.float64),
            bins=confidence_edges,
        )[0]
        epoch_metric = {
            "schema": "m1_seed42_stage1_training_epoch_metrics_v1",
            "branch": self.stage1_branch,
            **self.protocol_b2_epoch_rows[-1],
            "raw_state_sha256": raw_state_sha,
            "ema_state_sha256": ema_state_sha,
            "raw_tensor_sha256": raw_state_sha,
            "ema_tensor_sha256": ema_state_sha,
            "optimizer_state_sha256": optimizer_state_sha,
            "scheduler_state_sha256": scheduler_state_sha,
            "scaler_state_sha256": scaler_state_sha,
            "rng_state_sha256": rng_state_sha,
            "fixed_exposure_active_visits": sum(
                int(row["fixed_active_count"])
                for row in self.fixed_exposure_runtime_rows
            ),
            "fixed_exposure_identity_visits": sum(
                int(row["fixed_identity_count"])
                for row in self.fixed_exposure_runtime_rows
            ),
            "fixed_transform_applied_visits": sum(
                int(row["fixed_transform_applied_count"])
                for row in self.fixed_exposure_runtime_rows
            ),
            "cache_hit_count": sum(
                int(row["cache_hit_count"]) for row in self.fixed_cache_runtime_rows
            ),
            "cache_miss_count": sum(
                int(row["cache_miss_count"]) for row in self.fixed_cache_runtime_rows
            ),
            "cache_fallback_count": sum(
                int(row["cache_fallback_count"])
                for row in self.fixed_cache_runtime_rows
            ),
            "hpl_image_count": len(per_image_counts),
            "hpl_per_image_count_histogram_json": json.dumps(
                image_histogram, sort_keys=True, separators=(",", ":")
            ),
            "hpl_confidence_count": len(self.protocol_epoch_pseudo_confidences),
            "hpl_confidence_f32le_sha256": hashlib.sha256(
                b"".join(
                    struct.pack("<f", float(value))
                    for value in self.protocol_epoch_pseudo_confidences
                )
            ).hexdigest(),
            "hpl_confidence_histogram_edges_json": json.dumps(
                confidence_edges.tolist(), separators=(",", ":")
            ),
            "hpl_confidence_histogram_counts_json": json.dumps(
                [int(value) for value in confidence_histogram], separators=(",", ":")
            ),
            "state_identity_node": "COMPLETE_EPOCH_END_BEFORE_CHECKPOINT_IO",
            "run_attempt_id": self.run_attempt_id,
            "pseudo_batch_trace_path": str(batch_path.resolve()),
            "pseudo_batch_trace_sha256": sha256_file(batch_path),
            "pseudo_batch_trace_rows": len(self._epoch_trace_batch_rows),
            "hpl_box_trace_path": str(box_path.resolve()),
            "hpl_box_trace_sha256": sha256_file(box_path),
            "hpl_box_trace_rows": len(self._epoch_trace_box_rows),
            "fixed_exposure_runtime_path": str(exposure_runtime_path.resolve()),
            "fixed_exposure_runtime_sha256": sha256_file(
                exposure_runtime_path
            ),
            "fixed_exposure_runtime_rows": len(
                self.fixed_exposure_runtime_rows
            ),
            "fixed_cache_runtime_path": (
                str(cache_runtime_path.resolve())
                if self.stage1_branch == "NATIVE_A_FIXED"
                else ""
            ),
            "fixed_cache_runtime_sha256": (
                sha256_file(cache_runtime_path)
                if self.stage1_branch == "NATIVE_A_FIXED"
                else ""
            ),
            "fixed_cache_runtime_rows": len(self.fixed_cache_runtime_rows),
        }
        if any(int(row["epoch"]) == epoch for row in self.training_epoch_metric_rows):
            raise FormalStop(f"STOP_DUPLICATE_TRAINING_EPOCH_METRIC:{epoch}")
        self.training_epoch_metric_rows.append(epoch_metric)
        atomic_csv(
            Path(self.save_dir) / "TRAINING_EPOCH_METRICS.csv",
            self.training_epoch_metric_rows,
        )
        atomic_csv(
            Path(self.save_dir) / "OPTIMIZER_UPDATE_TRACE.csv",
            self.optimizer_update_rows,
        )
        event = E_EVENTS.get(epoch)
        if self.logging_enabled and event is not None and event not in self._snapshot_names:
            self._save_snapshot(event, event_kind="COMPLETE_EPOCH_END")

    def _full_state_payload(self, epoch: int) -> dict[str, Any]:
        model_state = deepcopy(unwrap_model(self.model).state_dict())
        ema_state = deepcopy(self.ema.ema.state_dict())
        global_step = E71_GLOBAL_STEP + (epoch - 71) * EXPECTED_STEPS_PER_EPOCH
        return {
            "format": "m1_seed42_stage1_full_state_v1",
            "experiment": EXPERIMENT,
            "branch": self.stage1_branch,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step_count": E71_EMA_UPDATES + int(self.protocol_optimizer_steps),
            "optimizer_updates_absolute": E71_EMA_UPDATES + int(self.protocol_optimizer_steps),
            "optimizer_steps_this_branch": int(self.protocol_optimizer_steps),
            "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
            "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
            "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
            "teacher_target_forward_count": int(self.teacher_target_forward_count),
            "observer_return_identity_count": int(self.observer_return_identity_count),
            "hpl_torchvision_scope_call_count": int(
                self.hpl_torchvision_scope_call_count
            ),
            "hpl_torchvision_identity_restore_count": int(
                self.hpl_torchvision_identity_restore_count
            ),
            "amp_found_inf_windows": int(self.amp_found_inf_windows),
            "ema_updates": int(self.ema.updates),
            "model_state_dict": model_state,
            "model_hash": tensor_state_hash(model_state),
            "ema_state_dict": ema_state,
            "ema_hash": tensor_state_hash(ema_state),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            **rng_state_payload(),
            "source_schedule_cursor": {"completed_total_epoch": epoch, "next_total_epoch": epoch + 1, "batches_per_epoch": EXPECTED_STEPS_PER_EPOCH},
            "target_schedule_cursor": {"completed_total_epoch": epoch, "next_total_epoch": epoch + 1, "batches_per_active_epoch": EXPECTED_STEPS_PER_EPOCH, "target_batch": TARGET_BATCH},
            "base_checkpoint_sha256": E71_FILE_SHA256,
            "base_checkpoint_path": str(E71_FULL_STATE.resolve()),
            "dataset_manifest": EXPECTED_DATASET_MANIFEST,
            "source_schedule_sha256": EXPECTED_INPUT_SHA256["source_schedule"],
            "source_augmentation_sha256": EXPECTED_INPUT_SHA256["source_augmentation"],
            "target_schedule_sha256": EXPECTED_INPUT_SHA256["target_schedule"],
            "fixed_exposure_sha256": EXPECTED_INPUT_SHA256["fixed_exposure"],
            "fixed_cache_manifest_sha256": EXPECTED_INPUT_SHA256["fixed_cache_manifest"],
            "run_contract_path": str(self.run_contract_path),
            "run_contract_sha256": self.run_contract_sha256,
            "method_config": self.method_config,
            "train_args": vars(self.args),
            "epoch_audit": self.protocol_b2_epoch_rows[-1] if self.protocol_b2_epoch_rows else {},
            "snapshot_rows": self._snapshot_rows,
            "checkpoint_rows": self.checkpoint_rows,
            "recovery_rows": self.recovery_rows,
            "optimizer_update_rows": self.optimizer_update_rows,
            "training_epoch_metric_rows": self.training_epoch_metric_rows,
            "completed_data_access_epoch_rows": self.completed_data_access_epoch_rows,
            "protocol_epoch_rows": self.protocol_epoch_rows,
            "protocol_b2_epoch_rows": self.protocol_b2_epoch_rows,
            "protocol_b2_batch_rows": self.protocol_b2_batch_rows,
            "protocol_target_read_count_total": int(
                self.protocol_target_read_count_total
            ),
            "protocol_target_schedule_mismatches_total": int(
                self.protocol_target_schedule_mismatches_total
            ),
            "protocol_source_hash_mismatches_total": int(
                self.protocol_source_hash_mismatches_total
            ),
            "protocol_b2_pseudo_boxes_total": int(
                self.protocol_b2_pseudo_boxes_total
            ),
            "protocol_target_cv2_decode_total": int(
                self.protocol_target_cv2_decode_total
            ),
            "protocol_target_pillow_fallback_total": int(
                self.protocol_target_pillow_fallback_total
            ),
            "t0_hook_fire_count": self._t0_hook_fire_count,
            "complete_epoch_boundary": True,
        }

    def save_model(self):
        epoch = int(self.protocol_current_total_epoch)
        payload = self._full_state_payload(epoch)
        directory = Path(self.save_dir) / "full_state"
        previous_last = next(
            (
                dict(row)
                for row in self.recovery_rows
                if row.get("event_kind") == "RECOVERY_ONLY_EPOCH"
                and row.get("event_name") == "RECOVERY_LAST"
            ),
            None,
        )
        last_path = rotate_last_checkpoint(payload, directory)
        recovery_rows: list[dict[str, Any]] = []
        prev_path = directory / "prev_full_state.pt"
        if prev_path.exists():
            if previous_last is None or int(previous_last["epoch"]) != epoch - 1:
                raise FormalStop(f"STOP_RECOVERY_PREV_LINEAGE:{epoch}")
            previous_last.update(
                {
                    "event_name": "RECOVERY_PREV",
                    "full_state_path": str(prev_path.resolve()),
                    "full_state_file_sha256": sha256_file(prev_path),
                    "resume_allowed": int(previous_last["epoch"]) < 100,
                }
            )
            recovery_rows.append(previous_last)
        recovery_rows.append(
            {
                "schema": "m1_seed42_stage1_checkpoint_manifest_v1",
                "branch": self.stage1_branch,
                "event_kind": "RECOVERY_ONLY_EPOCH",
                "event_name": "RECOVERY_LAST",
                "epoch": epoch,
                "global_step": payload["global_step"],
                "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
                "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
                "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
                "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
                "target_unsup_optimizer_update_count": int(self.target_contributing_optimizer_updates),
                "teacher_target_forward_count": int(self.teacher_target_forward_count),
                "observer_return_identity_count": int(self.observer_return_identity_count),
                "ema_updates": int(self.ema.updates),
                "scheduler_last_epoch": int(self.scheduler.last_epoch),
                "full_state_path": str(last_path.resolve()),
                "full_state_file_sha256": sha256_file(last_path),
                "raw_inference_path": "",
                "raw_inference_file_sha256": "",
                "raw_tensor_sha256": payload["model_hash"],
                "ema_inference_path": "",
                "ema_inference_file_sha256": "",
                "ema_tensor_sha256": payload["ema_hash"],
                "optimizer_state_sha256": recursive_state_sha256(payload["optimizer_state_dict"]),
                "scheduler_state_sha256": recursive_state_sha256(payload["scheduler_state_dict"]),
                "scaler_state_sha256": recursive_state_sha256(payload["scaler_state_dict"]),
                "rng_state_sha256": recursive_state_sha256(
                    {
                        key: payload[key]
                        for key in (
                            "rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda"
                        )
                    }
                ),
                "run_contract_sha256": self.run_contract_sha256,
                "resume_allowed": epoch < 100,
            }
        )
        self.recovery_rows = recovery_rows
        if epoch not in {94, 98, 100}:
            atomic_csv(
                Path(self.save_dir) / "CHECKPOINT_MANIFEST.csv",
                [*self._snapshot_rows, *self.checkpoint_rows, *self.recovery_rows],
            )
            return False
        registered_path = (
            directory
            / "registered"
            / self.run_attempt_id
            / f"epoch_{epoch:03d}_full_state.pt"
        )
        if registered_path.exists():
            raise FormalStop(f"STOP_REGISTERED_FULL_STATE_EXISTS:{registered_path}")
        registered_path.parent.mkdir(parents=True, exist_ok=True)
        # Contract-frozen formal registration cadence.  Rolling last/prev is
        # recovery-only engineering state: it is SHA/manifest registered and
        # resume eligible, but never an evaluation-eligible model checkpoint.
        temporary = registered_path.with_name(registered_path.name + ".tmp")
        try:
            with last_path.open("rb") as source, temporary.open("wb") as target:
                while block := source.read(16 * 1024 * 1024):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, registered_path)
        finally:
            temporary.unlink(missing_ok=True)
        inference_dir = (
            Path(self.save_dir)
            / "inference_state"
            / "attempts"
            / self.run_attempt_id
        )
        event_name = f"E{epoch}_REGISTERED"
        raw_path = inference_dir / f"{event_name}_raw.pt"
        ema_path = inference_dir / f"{event_name}_ema.pt"
        if raw_path.exists() or ema_path.exists():
            raise FormalStop(f"STOP_REGISTERED_INFERENCE_EXISTS:E{epoch}")
        atomic_torch_save(
            self._snapshot_payload(event_name, "raw", payload["model_state_dict"]),
            raw_path,
        )
        atomic_torch_save(
            self._snapshot_payload(event_name, "ema", payload["ema_state_dict"]),
            ema_path,
        )
        row = {
            "schema": "m1_seed42_stage1_checkpoint_manifest_v1",
            "branch": self.stage1_branch,
            "event_kind": "FULL_STATE_EPOCH",
            "event_name": f"E{epoch}_FULL_STATE",
            "epoch": epoch,
            "global_step": payload["global_step"],
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
            "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
            "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
            "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
            "target_unsup_optimizer_update_count": int(self.target_contributing_optimizer_updates),
            "teacher_target_forward_count": int(self.teacher_target_forward_count),
            "observer_return_identity_count": int(self.observer_return_identity_count),
            "ema_updates": int(self.ema.updates),
            "scheduler_last_epoch": int(self.scheduler.last_epoch),
            "full_state_path": str(registered_path.resolve()),
            "full_state_file_sha256": sha256_file(registered_path),
            "raw_inference_path": str(raw_path.resolve()),
            "raw_inference_file_sha256": sha256_file(raw_path),
            "raw_tensor_sha256": payload["model_hash"],
            "ema_inference_path": str(ema_path.resolve()),
            "ema_inference_file_sha256": sha256_file(ema_path),
            "ema_tensor_sha256": payload["ema_hash"],
            "optimizer_state_sha256": recursive_state_sha256(payload["optimizer_state_dict"]),
            "scheduler_state_sha256": recursive_state_sha256(payload["scheduler_state_dict"]),
            "scaler_state_sha256": recursive_state_sha256(payload["scaler_state_dict"]),
            "rng_state_sha256": recursive_state_sha256({key: payload[key] for key in ("rng_python", "rng_numpy", "rng_torch_cpu", "rng_torch_cuda")}),
            "run_contract_sha256": self.run_contract_sha256,
            "resume_allowed": epoch < 100,
        }
        self.checkpoint_rows.append(row)
        atomic_csv(
            Path(self.save_dir) / "CHECKPOINT_MANIFEST.csv",
            [*self._snapshot_rows, *self.checkpoint_rows, *self.recovery_rows],
        )
        return False

    def _protocol_on_train_end(self, trainer: "Stage1Trainer") -> None:
        super()._protocol_on_train_end(trainer)
        checks = {
            "completed_e100": int(self.protocol_current_total_epoch) == 100,
            "optimizer_windows_exact": int(self.protocol_optimizer_steps) == TOTAL_WINDOWS,
            "ema_updates_exact": int(self.ema.updates) == E71_EMA_UPDATES + TOTAL_WINDOWS,
            "source_epoch_rows_exact": len(self.protocol_epoch_rows) == len(EPOCHS)
            and [int(row["epoch"]) for row in self.protocol_epoch_rows]
            == list(EPOCHS),
            "b2_epoch_rows_exact": len(self.protocol_b2_epoch_rows) == len(EPOCHS)
            and [int(row["epoch"]) for row in self.protocol_b2_epoch_rows]
            == list(EPOCHS),
            "training_epoch_metric_rows_exact": len(
                self.training_epoch_metric_rows
            )
            == len(EPOCHS)
            and [int(row["epoch"]) for row in self.training_epoch_metric_rows]
            == list(EPOCHS)
            and all(
                require_sha(row["raw_state_sha256"], "raw_state_sha256")
                and require_sha(row["ema_state_sha256"], "ema_state_sha256")
                for row in self.training_epoch_metric_rows
            ),
            "b2_batch_rows_exact": len(self.protocol_b2_batch_rows)
            == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "optimizer_update_rows_exact": len(self.optimizer_update_rows)
            == TOTAL_WINDOWS,
            "completed_epoch_worker_data_access_exact": len(
                self.completed_data_access_epoch_rows
            )
            == len(EPOCHS)
            and [
                int(row["epoch"])
                for row in self.completed_data_access_epoch_rows
            ]
            == list(EPOCHS)
            and sum(
                int(row["target_adapt_image_read_count"])
                for row in self.completed_data_access_epoch_rows
            )
            == TOTAL_TARGET_VISITS
            and all(
                int(row["worker_count"]) == 2 and bool(row["all_pass"])
                for row in self.completed_data_access_epoch_rows
            ),
            "source_schedule_mismatch_zero": all(int(row["source_schedule_mismatch_count"]) == 0 for row in self.protocol_epoch_rows),
            "target_schedule_mismatch_zero": int(self.protocol_target_schedule_mismatches_total) == 0,
            "target_reads_exact": int(self.protocol_target_read_count_total) == TOTAL_TARGET_VISITS,
            "observer_call_count_exact": int(self.observer_original_call_count) == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "observer_return_identity_exact": int(self.observer_return_identity_count) == int(self.observer_original_call_count),
            "teacher_target_forward_count_exact": int(
                self.teacher_target_forward_count
            ) == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "hpl_torchvision_scoped_backend_calls_exact": int(
                self.hpl_torchvision_scope_call_count
            )
            == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "hpl_torchvision_root_identity_restores_exact": int(
                self.hpl_torchvision_identity_restore_count
            )
            == int(self.hpl_torchvision_scope_call_count),
            "target_contributing_events_reached_T100": int(
                self.target_contributing_optimizer_updates
            ) >= 100,
            "t0_forward_pre_hook_exact_once": int(self._t0_hook_fire_count) == 1,
            "required_snapshots_complete": set(["T0", *T_EVENTS.values(), *E_EVENTS.values()]).issubset(self._snapshot_names),
            "target_bn_frozen": all(float(row.get("target_bn_buffer_max_abs_delta", -1)) == 0.0 and int(row.get("target_bn_nonfloat_mismatch_count", -1)) == 0 for row in self.protocol_b2_batch_rows),
            "online_powell_calls_zero": all(int(row.get("online_powell_call_count", -1)) == 0 for row in self.protocol_b2_batch_rows),
            "target_adapt_label_read_count_zero": self.data_guard.counters()["target_adapt_label_file_open_count"] == 0,
            "target_dev_read_count_zero": self.data_guard.counters()["target_dev_gt_reads"] == 0,
            "target_test_read_count_zero": self.data_guard.counters()["target_test_reads"] == 0,
            "dataset_mutation_zero": self.data_guard.counters()["dataset_mutation_attempts"] == 0,
        }
        core = {
            "schema": "m1_seed42_stage1_training_core_audit_v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "completed": all(checks.values()),
            "all_pass": all(checks.values()),
            "branch": self.stage1_branch,
            "final_epoch": int(self.protocol_current_total_epoch),
            "final_global_step": E71_GLOBAL_STEP + len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "optimizer_windows_this_branch": int(self.protocol_optimizer_steps),
            "actual_unskipped_optimizer_updates": int(self.actual_unskipped_optimizer_updates),
            "target_contributing_optimizer_updates": int(self.target_contributing_optimizer_updates),
            "target_unsup_microbatch_count": int(self.target_unsup_microbatch_count),
            "teacher_target_forward_count": int(self.teacher_target_forward_count),
            "hpl_torchvision_scope_call_count": int(
                self.hpl_torchvision_scope_call_count
            ),
            "hpl_torchvision_identity_restore_count": int(
                self.hpl_torchvision_identity_restore_count
            ),
            "amp_found_inf_windows": int(self.amp_found_inf_windows),
            "ema_updates_final": int(self.ema.updates),
            "target_adapt_image_reads": int(self.protocol_target_read_count_total),
            "target_adapt_label_read_count": 0,
            "target_dev_read_count": 0,
            "target_test_read_count": 0,
            "dataset_mutation_count": 0,
            "checks": checks,
            "run_contract_sha256": self.run_contract_sha256,
            "timestamp": now_iso(),
        }
        atomic_json(Path(self.save_dir) / "TRAINING_CORE_AUDIT.json", core)
        if not core["all_pass"]:
            raise FormalStop(f"STOP_TRAINING_CORE_AUDIT:{checks}")


def validate_start_state(path: Path, expected_sha: str, *, branch: str | None = None) -> dict[str, Any]:
    if sha256_file(path) != expected_sha:
        raise FormalStop("STOP_START_STATE_FILE_SHA")
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format", "epoch", "global_step", "ema_updates", "model_state_dict",
        "ema_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "scaler_state_dict", "rng_python", "rng_numpy", "rng_torch_cpu",
        "rng_torch_cuda", "dataset_manifest",
    }
    if required - set(state):
        raise FormalStop(f"STOP_START_STATE_FIELDS:{sorted(required-set(state))}")
    epoch = int(state["epoch"])
    if path.resolve() == E71_FULL_STATE.resolve():
        checks = {
            "format": state["format"] == "protocol_b_full_state_v2",
            "epoch": epoch == 71,
            "global": int(state["global_step"]) == E71_GLOBAL_STEP,
            "ema": int(state["ema_updates"]) == E71_EMA_UPDATES,
            "optimizer": int(state.get("optimizer_step_count", -1)) == E71_EMA_UPDATES,
            "dataset": state["dataset_manifest"] == EXPECTED_DATASET_MANIFEST,
            "source": state.get("source_schedule_sha256") == EXPECTED_INPUT_SHA256["source_schedule"],
            "target": state.get("target_schedule_sha256") == EXPECTED_INPUT_SHA256["target_schedule"],
        }
        if not all(checks.values()):
            raise FormalStop(f"STOP_E71_IDENTITY:{checks}")
        raw_hash = tensor_state_hash(state["model_state_dict"])
        ema_hash = tensor_state_hash(state["ema_state_dict"])
        if raw_hash != E71_RAW_TENSOR_SHA256 or ema_hash != E71_EMA_TENSOR_SHA256:
            raise FormalStop("STOP_E71_TENSOR_HASH")
    else:
        if state["format"] != "m1_seed42_stage1_full_state_v1" or state.get("experiment") != EXPERIMENT or state.get("branch") != branch:
            raise FormalStop("STOP_RECOVERY_STATE_IDENTITY")
        if not 72 <= epoch < 100 or state.get("complete_epoch_boundary") is not True:
            raise FormalStop("STOP_RECOVERY_NOT_COMPLETE_EPOCH")
        if state.get("run_contract_sha256") != sha256_file(DEFAULT_CONTRACT):
            raise FormalStop("STOP_RECOVERY_CONTRACT_SHA")
        if int(state["global_step"]) != E71_GLOBAL_STEP + (epoch - 71) * EXPECTED_STEPS_PER_EPOCH:
            raise FormalStop("STOP_RECOVERY_GLOBAL_STEP")
        if int(state["scheduler_state_dict"]["last_epoch"]) != epoch - 1:
            raise FormalStop("STOP_RECOVERY_SCHEDULER_NODE")
        if state["dataset_manifest"] != EXPECTED_DATASET_MANIFEST:
            raise FormalStop("STOP_RECOVERY_DATASET")
        if tensor_state_hash(state["model_state_dict"]) != state["model_hash"] or tensor_state_hash(state["ema_state_dict"]) != state["ema_hash"]:
            raise FormalStop("STOP_RECOVERY_TENSOR_HASH")
    return state


def trainer_overrides(run_dir: Path, device: str, *, preflight: bool, resume: bool) -> dict[str, Any]:
    return {
        "model": "yolo11n.yaml", "data": str(SOURCE_YAML), "epochs": 100,
        "imgsz": 640, "batch": SOURCE_BATCH, "workers": 2, "device": device,
        "seed": 42, "deterministic": True, "single_cls": True,
        "pretrained": False, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
        "momentum": 0.937, "weight_decay": 0.0005, "warmup_epochs": 3.0,
        "warmup_momentum": 0.8, "warmup_bias_lr": 0.1, "nbs": 64,
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "close_mosaic": 10,
        "amp": True, "cache": False, "rect": False, "multi_scale": 0.0,
        "val": False, "plots": False, "save": not preflight, "save_period": -1,
        "patience": 1000, "project": str(run_dir.parent), "name": run_dir.name,
        "exist_ok": bool(resume or preflight), "resume": False,
        "verbose": not preflight, "cos_lr": False, "compile": False,
    }


def build_trainer(
    *,
    branch: str,
    contract_path: Path,
    contract_sha: str,
    run_dir: Path,
    device: str,
    start_state: dict[str, Any],
    start_state_path: Path,
    exposure_entries: list[ExposureEntry],
    cache_entries: dict[str, CacheEntry],
    data_guard: Any,
    data_access_attempt_dir: Path,
    run_attempt_id: str,
    logging_enabled: bool,
    preflight: bool,
    resume: bool,
) -> Stage1Trainer:
    return Stage1Trainer(
        branch=branch,
        run_contract_path=contract_path,
        run_contract_sha256=contract_sha,
        start_state=start_state,
        start_state_path=start_state_path,
        completed_start_epoch=int(start_state["epoch"]),
        exposure_entries=exposure_entries,
        cache_entries=cache_entries,
        data_guard=data_guard,
        data_access_attempt_dir=data_access_attempt_dir,
        run_attempt_id=run_attempt_id,
        logging_enabled=logging_enabled,
        preflight_mode=preflight,
        full_state_path=E70_BOOTSTRAP,
        source_schedule_path=SOURCE_SCHEDULE,
        source_augmentation_path=SOURCE_AUGMENTATION,
        contract_path=B2_TEMPLATE,
        target_yaml_path=TARGET_YAML,
        target_schedule_path=TARGET_SCHEDULE,
        target_schedule_manifest_path=TARGET_SCHEDULE_MANIFEST,
        b1_train_metrics_path=B1_METRICS,
        comparison_rule_path=COMPARISON_RULE,
        code_snapshot_sha256=sha256_file(Path(__file__)),
        environment_sha256=canonical_json_sha256({
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "device": device,
        }),
        overrides=trainer_overrides(run_dir, device, preflight=preflight, resume=resume),
    )


def _preflight_projection(trainer: Stage1Trainer, source_rows: list[dict[str, Any]], window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "branch": trainer.stage1_branch,
        "source_rows": source_rows,
        "target_hpl_common_rows": trainer._preflight_common_rows,
        "window_rows": window_rows,
        "student_tensor_sha256": tensor_state_hash(unwrap_model(trainer.model).state_dict()),
        "teacher_tensor_sha256": tensor_state_hash(trainer.ema.ema.state_dict()),
        "optimizer_state_sha256": recursive_state_sha256(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": recursive_state_sha256(trainer.scheduler.state_dict()),
        "scaler_state_sha256": recursive_state_sha256(trainer.scaler.state_dict()),
        "rng_state_sha256": recursive_state_sha256(rng_state_payload()),
        "rng_hashes": rng_state_hashes(),
        "bn_running_state_sha256": tensor_state_hash(
            bn_running_state_clone(unwrap_model(trainer.model))
        ),
        "scheduler_last_epoch": int(trainer.scheduler.last_epoch),
        "optimizer_windows": int(trainer.protocol_optimizer_steps),
        "actual_unskipped_optimizer_updates": int(trainer.actual_unskipped_optimizer_updates),
        "target_contributing_optimizer_updates": int(trainer.target_contributing_optimizer_updates),
        "target_unsup_microbatch_count": int(trainer.target_unsup_microbatch_count),
        "teacher_target_forward_count": int(trainer.teacher_target_forward_count),
        "hpl_torchvision_scope_call_count": int(
            trainer.hpl_torchvision_scope_call_count
        ),
        "hpl_torchvision_identity_restore_count": int(
            trainer.hpl_torchvision_identity_restore_count
        ),
        "ema_updates": int(trainer.ema.updates),
        "target_read_count": int(trainer.protocol_target_read_count_total),
    }


def run_preflight_worker(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve()
    contract, contract_sha = load_run_contract(contract_path, authority="logging-preflight")
    verify_static_locks()
    branch = str(args.branch)
    logging_enabled = str(args.logging).upper() == "ON"
    output = inside_root(Path(args.output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data_guard = install_m1_data_isolation_guard(
        phase=f"{EXPERIMENT}_LOGGING_PREFLIGHT_{branch}_{args.logging}",
        audit_dir=output / "data_access",
        allow_target_dev_images=False,
    )
    schedule = ProtocolBTargetSchedule(TARGET_SCHEDULE, TARGET_SCHEDULE_MANIFEST)
    exposure, cache, fixed_summary = load_fixed_inputs(schedule)
    state = validate_start_state(E71_FULL_STATE, E71_FILE_SHA256)
    trainer = build_trainer(
        branch=branch, contract_path=contract_path, contract_sha=contract_sha,
        run_dir=output / "trainer_sandbox", device=args.device,
        start_state=state, start_state_path=E71_FULL_STATE,
        exposure_entries=exposure, cache_entries=cache, data_guard=data_guard,
        data_access_attempt_dir=output / "trainer_sandbox" / "data_access",
        run_attempt_id=f"PREFLIGHT_{branch}_{str(args.logging).upper()}",
        logging_enabled=logging_enabled, preflight=True, resume=False,
    )
    trainer._setup_train()
    trainer.epoch = 71
    trainer._protocol_on_epoch_start(trainer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.scheduler.step()
    if int(trainer.scheduler.last_epoch) != 71:
        raise FormalStop("STOP_PREFLIGHT_E72_SCHEDULER_NODE")
    trainer._model_train()
    trainer.optimizer.zero_grad()
    source_iterator = iter(trainer.train_loader)
    source_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for micro_index in range(PREFLIGHT_MICROBATCHES):
        raw = next(source_iterator)
        source_uid = list(raw["protocol_b_uid"])
        source_seed = [int(value) for value in raw["protocol_b_aug_seed"]]
        trainer._protocol_on_batch_start(trainer)
        with autocast(bool(trainer.amp)):
            processed = trainer.preprocess_batch(raw)
            loss, loss_items = trainer.model(processed)
            total = loss.sum()
        trainer.scaler.scale(total).backward()
        source_rows.append(
            {
                "microbatch": micro_index,
                "source_uids": source_uid,
                "source_aug_seeds": source_seed,
                "source_tensor_sha256": tensor_bytes_sha(processed["img"]),
                "loss_sha256": tensor_bytes_sha(loss),
                "loss_items_sha256": tensor_bytes_sha(loss_items),
                "gradient_sha256_after_backward": gradient_state_sha(trainer.model),
            }
        )
        if micro_index % ACCUMULATION == ACCUMULATION - 1:
            trainer.optimizer_step()
            window_rows.append(
                {
                    "window": len(window_rows) + 1,
                    "underlying_optimizer_step_called": trainer._last_step_underlying_called,
                    "found_inf": trainer.protocol_last_optimizer_window_found_inf,
                    "target_term_included": trainer._last_window_target_included,
                    "student_tensor_sha256": tensor_state_hash(unwrap_model(trainer.model).state_dict()),
                    "teacher_tensor_sha256": tensor_state_hash(trainer.ema.ema.state_dict()),
                    "optimizer_state_sha256": recursive_state_sha256(trainer.optimizer.state_dict()),
                    "ema_updates": int(trainer.ema.updates),
                    "rng_state_sha256": recursive_state_sha256(rng_state_payload()),
                }
            )
        trainer._protocol_b2_on_batch_end(trainer)
    target_iterator = trainer.protocol_target_iterator
    if not isinstance(target_iterator, CapturingIterator):
        raise FormalStop("STOP_PREFLIGHT_TARGET_ITERATOR_TYPE")
    shutdown_workers = getattr(target_iterator.inner, "_shutdown_workers", None)
    if not callable(shutdown_workers):
        raise FormalStop("STOP_PREFLIGHT_WORKER_SHUTDOWN_UNAVAILABLE")
    shutdown_workers()
    trainer.protocol_target_iterator = None
    projection = _preflight_projection(trainer, source_rows, window_rows)
    fixed_active = sum(
        sum(json.loads(row["fixed_exposure_active_json"]))
        for row in projection["target_hpl_common_rows"]
    )
    cache_payload_validations = sum(
        int(row["cache_payload_validation_count"])
        for row in projection["target_hpl_common_rows"]
    )
    parent_events_path = data_guard.export_process_audit(
        output / "DATA_ACCESS_EVENTS.json"
    )
    child_summary_paths = sorted(
        (output / "trainer_sandbox" / "data_access").rglob(
            "process_*_DATA_ACCESS_SUMMARY.json"
        )
    )
    process_counters = [data_guard.counters()]
    child_event_paths: list[Path] = []
    child_summaries: list[dict[str, Any]] = []
    for summary_path in child_summary_paths:
        summary = _load_json(summary_path)
        child = summary.get("counters")
        event_path = inside_root(Path(str(summary.get("event_path", ""))))
        if not (
            summary.get("schema")
            == "m1_seed42_stage1_worker_data_access_summary_v1"
            and isinstance(child, dict)
            and event_path.is_file()
            and sha256_file(event_path) == summary.get("event_file_sha256")
        ):
            raise FormalStop(f"STOP_PREFLIGHT_WORKER_DATA_ACCESS:{summary_path}")
        process_counters.append(child)
        child_event_paths.append(event_path)
        child_summaries.append(summary)
    isolation_fields = (
        "formal_forbidden_event_count",
        "target_adapt_label_file_open_count",
        "target_dev_gt_reads",
        "target_test_reads",
        "dataset_mutation_attempts",
        "target_adapt_image_read_completed_count",
    )
    isolation = {
        field: sum(int(row.get(field, 0) or 0) for row in process_counters)
        for field in isolation_fields
    }
    endpoint_reference = PREFLIGHT_ENDPOINTS[branch]
    checks = {
        "microbatches_exact": len(source_rows) == 10,
        "optimizer_windows_exact": len(window_rows) == 5,
        "target_visits_exact": projection["target_read_count"] == 80,
        "fixed_active_exact": fixed_active == 41,
        "fixed_identity_exact": 80 - fixed_active == 39,
        "fixed_payload_hash_cases_exact": cache_payload_validations
        == (41 if branch == "NATIVE_A_FIXED" else 0),
        "observer_mode_exact": (
            trainer.observer_original_call_count == (10 if logging_enabled else 0)
            and trainer.observer_return_identity_count == (10 if logging_enabled else 0)
        ),
        "hpl_local_nms_scope_exact": projection[
            "hpl_torchvision_scope_call_count"
        ]
        == PREFLIGHT_MICROBATCHES
        and projection["hpl_torchvision_identity_restore_count"]
        == PREFLIGHT_MICROBATCHES,
        "logging_off_no_trace_capture": logging_enabled
        or (not trainer._epoch_trace_batch_rows and not trainer._epoch_trace_box_rows),
        "logging_on_full_trace_capture": (not logging_enabled)
        or len(trainer._epoch_trace_batch_rows) == 10,
        "optimizer_step_accounting_exact": sum(
            int(bool(row["underlying_optimizer_step_called"]))
            + int(bool(row["found_inf"]))
            for row in window_rows
        ) == PREFLIGHT_WINDOWS,
        "endpoint_raw_reference_exact": projection["student_tensor_sha256"]
        == endpoint_reference["raw"],
        "endpoint_ema_reference_exact": projection["teacher_tensor_sha256"]
        == endpoint_reference["ema"],
        "endpoint_optimizer_reference_exact": projection[
            "optimizer_state_sha256"
        ]
        == endpoint_reference["optimizer"],
        "endpoint_scheduler_reference_exact": projection[
            "scheduler_state_sha256"
        ]
        == endpoint_reference["scheduler"]
        and projection["scheduler_last_epoch"] == 71,
        "endpoint_scaler_reference_exact": projection["scaler_state_sha256"]
        == endpoint_reference["scaler"],
        "endpoint_rng_reference_exact": projection["rng_hashes"]
        == PREFLIGHT_ENDPOINT_RNG_HASHES,
        "endpoint_bn_reference_exact": projection["bn_running_state_sha256"]
        == endpoint_reference["bn"],
        "endpoint_update_counts_reference_exact": projection["optimizer_windows"]
        == 5
        and projection["ema_updates"] == 20_015,
        "target_worker_process_evidence_exact": len(child_summary_paths) == 2
        and len(child_event_paths) == 2
        and {
            int(row["last_export_trigger_batch_index"])
            for row in child_summaries
        }
        == {8, 9}
        and all(
            int(row["counters"]["target_adapt_image_read_completed_count"])
            == 40
            for row in child_summaries
        ),
        "target_workers_shutdown_before_audit": trainer.protocol_target_iterator
        is None,
        "target_image_reads_exact": isolation[
            "target_adapt_image_read_completed_count"
        ]
        == 80,
        "checkpoint_and_snapshot_writes_zero": not any(output.rglob("*.pt")),
        "forbidden_events_zero": isolation["formal_forbidden_event_count"] == 0,
        "target_dev_zero": isolation["target_dev_gt_reads"] == 0,
        "target_test_zero": isolation["target_test_reads"] == 0,
        "target_labels_zero": isolation["target_adapt_label_file_open_count"] == 0,
        "dataset_mutation_zero": isolation["dataset_mutation_attempts"] == 0,
    }
    result = {
        "schema": "m1_seed42_stage1_logging_preflight_worker_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()),
        "branch": branch,
        "logging": str(args.logging).upper(),
        "run_contract_sha256": contract_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "fixed_input_summary": fixed_summary,
        "endpoint_reference": endpoint_reference,
        "endpoint_reference_file": str(
            (
                PREFLIGHT_CONTROL_REFERENCE
                if branch == "CONTROL"
                else PREFLIGHT_FIXED_REFERENCE
            ).resolve()
        ),
        "endpoint_reference_file_sha256": EXPECTED_INPUT_SHA256[
            "preflight_control_reference"
            if branch == "CONTROL"
            else "preflight_fixed_reference"
        ],
        "checks": checks,
        "projection": projection,
        "projection_sha256": canonical_json_sha256(projection),
        "data_access_counters": isolation,
        "checkpoint_write_count": 0,
        "snapshot_write_count": 0,
        "data_access_evidence": [
            {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
            }
            for path in [
                parent_events_path,
                *child_summary_paths,
                *child_event_paths,
            ]
        ],
        "timestamp": now_iso(),
    }
    atomic_json(output / "result.json", result)
    if not result["all_pass"]:
        raise FormalStop(f"STOP_PREFLIGHT_WORKER:{checks}")
    return 0


def run_logging_preflight(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve()
    contract, contract_sha = load_run_contract(contract_path, authority="logging-preflight")
    verify_static_locks()
    output_root = RESULT_ROOT / "pseudo_trace_logging_preflight_raw"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite logging preflight: {output_root}")
    output_root.mkdir(parents=True)
    lease = acquire_training_singleton(
        project_root=ROOT,
        experiment_id=f"{EXPERIMENT}_LOGGING_PREFLIGHT",
        run_name="logging_preflight",
        run_path=output_root,
        checkpoint_path=E71_FULL_STATE,
        checkpoint_sha256=E71_FILE_SHA256,
        device_id=args.device,
        entrypoint=__file__,
    )
    try:
        worker_results: list[dict[str, Any]] = []
        for branch in BRANCHES:
            for logging_mode in ("OFF", "ON"):
                destination = output_root / f"{branch}_{logging_mode}"
                command = [
                    sys.executable, "-B", str(Path(__file__).resolve()), "logging-preflight",
                    "--contract", str(contract_path), "--device", str(args.device),
                    "--worker", "--branch", branch, "--logging", logging_mode,
                    "--output", str(destination),
                ]
                env = dict(os.environ)
                env.update({"PYTHONHASHSEED": "42", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
                completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
                if completed.returncode != 0:
                    raise FormalStop(f"STOP_PREFLIGHT_CHILD_EXIT:{branch}:{logging_mode}:{completed.returncode}")
                worker_results.append(_load_json(destination / "result.json"))
        comparisons: list[dict[str, Any]] = []
        for branch in BRANCHES:
            off = next(row for row in worker_results if row["branch"] == branch and row["logging"] == "OFF")
            on = next(row for row in worker_results if row["branch"] == branch and row["logging"] == "ON")
            exact = off["projection"] == on["projection"]
            comparisons.append(
                {
                    "branch": branch,
                    "logging_off_projection_sha256": off["projection_sha256"],
                    "logging_on_projection_sha256": on["projection_sha256"],
                    "uid_source_target_mask_hpl_loss_grad_step_ema_rng_student_teacher_exact": exact,
                }
            )
        checks = {
            "four_workers_pass": len(worker_results) == 4 and all(row["all_pass"] for row in worker_results),
            "two_off_on_comparisons_exact": len(comparisons) == 2 and all(row["uid_source_target_mask_hpl_loss_grad_step_ema_rng_student_teacher_exact"] for row in comparisons),
            "fresh_e71_each_worker": all(row["projection"]["optimizer_windows"] == 5 for row in worker_results),
            "checkpoint_writes_zero": all(
                int(row["checkpoint_write_count"]) == 0
                and int(row["snapshot_write_count"]) == 0
                for row in worker_results
            ),
        }
        audit = {
            "schema": "m1_seed42_stage1_logging_preflight_audit_v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "all_pass": all(checks.values()),
            "run_contract_sha256": contract_sha,
            "runner_sha256": sha256_file(Path(__file__)),
            "workers": [{key: row[key] for key in ("branch", "logging", "status", "projection_sha256")} for row in worker_results],
            "comparisons": comparisons,
            "checks": checks,
            "formal_training_started": False,
            "checkpoint_writes": 0,
            "snapshot_writes": 0,
            "timestamp": now_iso(),
        }
        atomic_json(RESULT_ROOT / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json", audit)
        if not audit["all_pass"]:
            raise FormalStop(f"STOP_LOGGING_PREFLIGHT:{checks}")
        lease.mark_completed()
    except BaseException as exc:
        lease.mark_failed(exc)
        atomic_json(output_root / "FAILURE.json", {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "timestamp": now_iso()})
        raise
    return 0


def _validate_recovery(path: Path, branch: str, run_dir: Path, contract_sha: str) -> tuple[dict[str, Any], str]:
    path = inside_root(path)
    if not path.is_relative_to((run_dir / "full_state").resolve()):
        raise FormalStop("STOP_RECOVERY_PATH_OUTSIDE_BRANCH")
    completed_audit = run_dir / "TRAINING_FINAL_AUDIT.json"
    if completed_audit.is_file() and _load_json(completed_audit).get("completed") is True:
        raise FormalStop("STOP_RECOVERY_AFTER_COMPLETED_FORMAL_RUN")
    manifest_path = run_dir / "CHECKPOINT_MANIFEST.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    recovery_kinds = {"FULL_STATE_EPOCH", "RECOVERY_ONLY_EPOCH"}
    matches = [
        row
        for row in rows
        if row["event_kind"] in recovery_kinds
        and str(row.get("resume_allowed", "")).casefold() == "true"
        and Path(row["full_state_path"]).resolve() == path
    ]
    if len(matches) != 1:
        raise FormalStop("STOP_RECOVERY_MANIFEST_IDENTITY")
    expected_sha = require_sha(matches[0]["full_state_file_sha256"], "recovery manifest SHA")
    state = validate_start_state(path, expected_sha, branch=branch)
    if state["run_contract_sha256"] != contract_sha:
        raise FormalStop("STOP_RECOVERY_CONTRACT")
    complete_epochs = sorted(
        int(row["epoch"])
        for row in rows
        if row["event_kind"] in recovery_kinds
        and str(row.get("resume_allowed", "")).casefold() == "true"
    )
    if int(state["epoch"]) != max(complete_epochs):
        raise FormalStop("STOP_RECOVERY_NOT_LATEST_COMPLETE_EPOCH")
    # The external manifest can contain a snapshot written during a later,
    # incomplete attempt (snapshot I/O happens before the complete-epoch full
    # state).  Such a row is evidence from the aborted attempt, not accepted
    # recovery lineage.  Inherit snapshots only from the selected complete
    # full-state payload, then cross-lock every inherited artifact against the
    # manifest.  Registered complete-epoch checkpoints at or before the
    # selected epoch are independently safe to reconstruct from the manifest;
    # the selected checkpoint payload is necessarily created just before its
    # own registered manifest row is appended.
    payload_snapshots = state.get("snapshot_rows")
    if not isinstance(payload_snapshots, list):
        raise FormalStop("STOP_RECOVERY_SNAPSHOT_PAYLOAD_SCHEMA")
    manifest_snapshots = [
        row
        for row in rows
        if row["event_kind"] not in {"FULL_STATE_EPOCH", "RECOVERY_ONLY_EPOCH"}
    ]
    manifest_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_snapshots:
        manifest_by_event.setdefault(str(row["event_name"]), []).append(row)
    accepted_snapshots: list[dict[str, Any]] = []
    accepted_names: set[str] = set()
    for payload_row in payload_snapshots:
        if not isinstance(payload_row, dict):
            raise FormalStop("STOP_RECOVERY_SNAPSHOT_PAYLOAD_ROW")
        event_name = str(payload_row.get("event_name", ""))
        candidates = manifest_by_event.get(event_name, [])
        if not event_name or event_name in accepted_names or len(candidates) != 1:
            raise FormalStop(f"STOP_RECOVERY_SNAPSHOT_MANIFEST_IDENTITY:{event_name}")
        manifest_row = candidates[0]
        exact_fields = (
            "schema", "branch", "event_kind", "event_name",
            "full_state_path", "full_state_file_sha256",
            "raw_inference_path", "raw_inference_file_sha256",
            "raw_tensor_sha256", "ema_inference_path",
            "ema_inference_file_sha256", "ema_tensor_sha256",
            "run_contract_sha256", "snapshot_nonintervention_audit_path",
            "snapshot_nonintervention_audit_sha256",
        )
        if any(str(manifest_row.get(key, "")) != str(payload_row.get(key, "")) for key in exact_fields):
            raise FormalStop(f"STOP_RECOVERY_SNAPSHOT_MANIFEST_FIELDS:{event_name}")
        if any(
            int(manifest_row.get(key, -1)) != int(payload_row.get(key, -1))
            for key in ("epoch", "global_step")
        ):
            raise FormalStop(f"STOP_RECOVERY_SNAPSHOT_MANIFEST_COUNTERS:{event_name}")
        for path_field, sha_field in (
            ("raw_inference_path", "raw_inference_file_sha256"),
            ("ema_inference_path", "ema_inference_file_sha256"),
            (
                "snapshot_nonintervention_audit_path",
                "snapshot_nonintervention_audit_sha256",
            ),
        ):
            artifact = inside_root(Path(str(payload_row[path_field])))
            expected_artifact_sha = require_sha(
                payload_row[sha_field], f"recovery snapshot {sha_field}"
            )
            if not artifact.is_file() or sha256_file(artifact) != expected_artifact_sha:
                raise FormalStop(
                    f"STOP_RECOVERY_SNAPSHOT_ARTIFACT:{event_name}:{path_field}"
                )
        accepted_names.add(event_name)
        accepted_snapshots.append(dict(payload_row))
    state["checkpoint_rows"] = [
        row
        for row in rows
        if row["event_kind"] == "FULL_STATE_EPOCH"
        and int(row["epoch"]) <= int(state["epoch"])
    ]
    state["snapshot_rows"] = accepted_snapshots
    state["recovery_rows"] = [
        row for row in rows if row["event_kind"] == "RECOVERY_ONLY_EPOCH"
    ]
    return state, expected_sha


def _write_data_access_audit(
    run_dir: Path,
    branch: str,
    guard: Any,
    completed_epoch_rows: Sequence[dict[str, Any]],
    current_attempt_root: Path,
) -> dict[str, Any]:
    """Aggregate the registered training lineage and retain aborted evidence.

    Completed-epoch rows are part of each rolling full-state.  Therefore a
    resumed run consumes exactly the two worker summaries that produced every
    accepted epoch, while partial/aborted-attempt files stay present and are
    separately scanned for any forbidden access.
    """

    current_attempt_root = inside_root(current_attempt_root)
    parent_unique = current_attempt_root / "parent" / (
        f"process_{os.getpid()}_DATA_ACCESS_EVENTS.json"
    )
    parent_events_path = guard.export_process_audit(parent_unique)
    # Compatibility alias; the attempt-scoped original is immutable evidence.
    atomic_json(run_dir / "DATA_ACCESS_EVENTS.json", _load_json(parent_events_path))

    summary_paths_all = sorted(
        (run_dir / "data_access" / "attempts").rglob(
            "process_*_DATA_ACCESS_SUMMARY.json"
        )
    )
    summary_by_path: dict[Path, dict[str, Any]] = {}
    event_paths_referenced: set[Path] = set()
    all_attempt_counter_rows: list[dict[str, Any]] = [
        {key: int(value or 0) for key, value in guard.counters().items()}
    ]
    raw_evidence_exact = True
    for summary_path in summary_paths_all:
        summary = _load_json(summary_path)
        counters = summary.get("counters")
        event_path = inside_root(Path(str(summary.get("event_path", ""))))
        summary_exact = (
            summary.get("schema")
            == "m1_seed42_stage1_worker_data_access_summary_v1"
            and isinstance(counters, dict)
            and event_path.is_file()
            and require_sha(
                summary.get("event_file_sha256"),
                "worker data access event SHA",
            )
            == sha256_file(event_path)
        )
        raw_evidence_exact &= bool(summary_exact)
        if not isinstance(counters, dict):
            counters = {}
        normalized_counters = {
            key: int(value or 0) for key, value in counters.items()
        }
        all_attempt_counter_rows.append(normalized_counters)
        resolved_summary = summary_path.resolve()
        summary_by_path[resolved_summary] = summary
        event_paths_referenced.add(event_path.resolve())

    all_event_paths = sorted(
        (run_dir / "data_access" / "attempts").rglob(
            "process_*_DATA_ACCESS_EVENTS.json"
        )
    )
    orphan_event_paths = [
        path
        for path in all_event_paths
        if path.resolve() not in event_paths_referenced
        and path.resolve() != Path(parent_events_path).resolve()
    ]
    for orphan_path in orphan_event_paths:
        orphan_counters = _load_json(orphan_path).get("counters")
        if not isinstance(orphan_counters, dict):
            raw_evidence_exact = False
            orphan_counters = {}
        all_attempt_counter_rows.append(
            {key: int(value or 0) for key, value in orphan_counters.items()}
        )

    lineage_summary_paths: list[Path] = []
    lineage_rows_valid = (
        len(completed_epoch_rows) == len(EPOCHS)
        and [int(row.get("epoch", -1)) for row in completed_epoch_rows]
        == list(EPOCHS)
    )
    for epoch_row in completed_epoch_rows:
        if epoch_row.get("schema") != "m1_seed42_stage1_completed_epoch_data_access_v1":
            lineage_rows_valid = False
            continue
        try:
            workers = json.loads(str(epoch_row["worker_summaries_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            lineage_rows_valid = False
            continue
        if not isinstance(workers, list) or len(workers) != 2:
            lineage_rows_valid = False
            continue
        for worker in workers:
            path = inside_root(Path(str(worker.get("summary_path", ""))))
            summary = summary_by_path.get(path.resolve())
            if not (
                summary is not None
                and sha256_file(path)
                == worker.get("summary_file_sha256")
                and summary.get("event_file_sha256")
                == worker.get("event_file_sha256")
                and int(worker.get("image_reads", -1))
                == TARGET_VISITS_PER_EPOCH // 2
            ):
                lineage_rows_valid = False
            lineage_summary_paths.append(path.resolve())

    lineage_counter_rows = [
        {
            key: int(value or 0)
            for key, value in summary_by_path[path]["counters"].items()
        }
        for path in lineage_summary_paths
        if path in summary_by_path
    ]
    summed_fields = (
        "formal_forbidden_event_count", "target_adapt_manifest_read_count",
        "target_adapt_label_file_open_count", "target_adapt_gt_cache_open_count",
        "target_adapt_image_read_completed_count", "target_dev_gt_reads",
        "target_test_reads", "dataset_mutation_attempts",
    )
    all_attempt_counters = {
        field: sum(int(row.get(field, 0) or 0) for row in all_attempt_counter_rows)
        for field in summed_fields
    }
    lineage_counters = {
        field: sum(int(row.get(field, 0) or 0) for row in lineage_counter_rows)
        for field in summed_fields
    }
    counters = dict(lineage_counters)
    counters.update(
        {
            "process_audit_count": 1 + len(summary_paths_all)
            + len(orphan_event_paths),
            "target_worker_process_audit_count": len(lineage_summary_paths),
            "all_attempt_worker_summary_count": len(summary_paths_all),
            "aborted_or_orphan_worker_event_count": len(orphan_event_paths),
            "all_attempt_target_adapt_image_read_completed_count": all_attempt_counters[
                "target_adapt_image_read_completed_count"
            ],
        }
    )
    forbidden_fields = (
        "formal_forbidden_event_count",
        "target_adapt_label_file_open_count",
        "target_adapt_gt_cache_open_count",
        "target_dev_gt_reads",
        "target_test_reads",
        "dataset_mutation_attempts",
    )
    checks = {
        "registered_completed_epochs_exact": lineage_rows_valid,
        "registered_worker_summaries_exact": len(lineage_summary_paths)
        == len(EPOCHS) * 2
        and len(lineage_counter_rows) == len(lineage_summary_paths),
        "raw_worker_event_sha_evidence_exact": raw_evidence_exact,
        "all_attempt_forbidden_events_zero": all(
            all_attempt_counters[field] == 0 for field in forbidden_fields
        ),
        "target_adapt_image_reads_exact": lineage_counters[
            "target_adapt_image_read_completed_count"
        ]
        == TOTAL_TARGET_VISITS,
        "target_worker_process_evidence_complete": len(lineage_summary_paths)
        == len(EPOCHS) * 2,
    }
    audit = {
        "schema": "m1_seed42_stage1_data_access_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()),
        "branch": branch,
        "target_adapt_label_read_count": all_attempt_counters[
            "target_adapt_label_file_open_count"
        ],
        "target_dev_read_count": all_attempt_counters["target_dev_gt_reads"],
        "target_test_read_count": all_attempt_counters["target_test_reads"],
        "dataset_mutation_count": all_attempt_counters[
            "dataset_mutation_attempts"
        ],
        "target_adapt_image_read_count": lineage_counters[
            "target_adapt_image_read_completed_count"
        ],
        "all_attempt_target_adapt_image_read_count": all_attempt_counters[
            "target_adapt_image_read_completed_count"
        ],
        "completed_epoch_evidence_count": len(completed_epoch_rows),
        "lineage_worker_evidence_count": len(lineage_summary_paths),
        "all_attempt_worker_summary_count": len(summary_paths_all),
        "orphan_worker_event_count": len(orphan_event_paths),
        "counters": counters,
        "all_attempt_counters": all_attempt_counters,
        "checks": checks,
        "event_log_path": str(Path(parent_events_path).resolve()),
        "event_log_sha256": sha256_file(parent_events_path),
        "completed_epoch_evidence_path": str(
            (run_dir / "DATA_ACCESS_COMPLETED_EPOCHS.csv").resolve()
        ),
        "completed_epoch_evidence_sha256": sha256_file(
            run_dir / "DATA_ACCESS_COMPLETED_EPOCHS.csv"
        )
        if (run_dir / "DATA_ACCESS_COMPLETED_EPOCHS.csv").is_file()
        else "",
        "timestamp": now_iso(),
    }
    atomic_json(run_dir / "DATA_ACCESS_AUDIT.json", audit)
    atomic_json(current_attempt_root / "ATTEMPT_DATA_ACCESS_AUDIT.json", audit)
    return audit


def run_formal_branch(args: argparse.Namespace) -> int:
    contract_path = Path(args.contract).resolve()
    contract, contract_sha = load_run_contract(contract_path, authority="formal-branch")
    verify_static_locks()
    branch = str(args.branch)
    if branch not in BRANCHES:
        raise FormalStop("STOP_FORMAL_BRANCH")
    preflight_path = RESULT_ROOT / "PSEUDO_TRACE_LOGGING_PREFLIGHT.json"
    if not preflight_path.is_file():
        raise FormalStop("STOP_LOGGING_PREFLIGHT_NOT_RUN")
    preflight = _load_json(preflight_path)
    if not (
        preflight.get("schema") == "m1_seed42_stage1_logging_preflight_audit_v1"
        and preflight.get("all_pass") is True
        and preflight.get("run_contract_sha256") == contract_sha
        and preflight.get("runner_sha256") == sha256_file(Path(__file__))
    ):
        raise FormalStop("STOP_LOGGING_PREFLIGHT_NOT_FROZEN_PASS")
    run_dir = inside_root(Path(contract["branches"][branch]["run_dir"]))
    resume = args.resume_from is not None
    if resume:
        if not run_dir.is_dir():
            raise FormalStop("STOP_RECOVERY_RUN_DIR_MISSING")
        state, checkpoint_sha = _validate_recovery(Path(args.resume_from), branch, run_dir, contract_sha)
        start_path = Path(args.resume_from).resolve()
    else:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite formal run: {run_dir}")
        state = validate_start_state(E71_FULL_STATE, E71_FILE_SHA256)
        checkpoint_sha = E71_FILE_SHA256
        start_path = E71_FULL_STATE
    attempt_id = (
        f"start_e{int(state['epoch']) + 1:03d}_pid{os.getpid()}_"
        f"{time.time_ns()}"
    )
    data_access_attempt_root = (
        run_dir / "data_access" / "attempts" / attempt_id
    )
    worker_data_access_dir = data_access_attempt_root / "workers"
    lease = acquire_training_singleton(
        project_root=ROOT,
        experiment_id=f"{EXPERIMENT}_{branch}",
        run_name=run_dir.name,
        run_path=run_dir,
        checkpoint_path=start_path,
        checkpoint_sha256=checkpoint_sha,
        device_id=args.device,
        entrypoint=__file__,
    )
    trainer: Stage1Trainer | None = None
    data_guard: Any | None = None
    try:
        data_guard = install_m1_data_isolation_guard(
            phase=f"{EXPERIMENT}_FORMAL_{branch}_{attempt_id}",
            audit_dir=data_access_attempt_root / "parent",
            allow_target_dev_images=False,
        )
        schedule = ProtocolBTargetSchedule(
            TARGET_SCHEDULE, TARGET_SCHEDULE_MANIFEST
        )
        exposure, cache, fixed_summary = load_fixed_inputs(schedule)
        import_state_before = {
            name: name in sys.modules
            for name in ("pandas", "pyarrow", "torchvision")
        }
        torchvision_root_object = sys.modules.get("torchvision")
        if import_state_before != {
            "pandas": False,
            "pyarrow": False,
            "torchvision": True,
        } or torchvision_root_object is None:
            raise FormalStop(
                f"STOP_TRAINING_IMPORT_STATE_BEFORE:{import_state_before}"
            )
        trainer = build_trainer(
            branch=branch, contract_path=contract_path, contract_sha=contract_sha,
            run_dir=run_dir, device=args.device, start_state=state,
            start_state_path=start_path, exposure_entries=exposure,
            cache_entries=cache, data_guard=data_guard,
            data_access_attempt_dir=worker_data_access_dir,
            run_attempt_id=attempt_id,
            logging_enabled=True,
            preflight=False, resume=resume,
        )
        registration = {
                "schema": "m1_seed42_stage1_run_registration_v1",
                "status": "REGISTERED_BEFORE_TRAINING",
                "branch": branch,
                "attempt_id": attempt_id,
                "data_access_attempt_root": str(
                    data_access_attempt_root.resolve()
                ),
                "run_contract_path": str(contract_path),
                "run_contract_sha256": contract_sha,
                "runner_sha256": sha256_file(Path(__file__)),
                "start_state_path": str(start_path),
                "start_state_file_sha256": checkpoint_sha,
                "resume": resume,
                "fixed_input_summary": fixed_summary,
                "target_dev_read_count": 0,
                "target_test_read_count": 0,
                "timestamp": now_iso(),
            }
        atomic_json(run_dir / "RUN_REGISTRATION.json", registration)
        atomic_json(
            run_dir / "raw_attempts" / f"RUN_REGISTRATION_{attempt_id}.json",
            registration,
        )
        trainer.train()
        import_state_after = {
            name: name in sys.modules
            for name in ("pandas", "pyarrow", "torchvision")
        }
        import_checks = {
            "pandas_absent_before_after": not import_state_before["pandas"]
            and not import_state_after["pandas"],
            "pyarrow_absent_before_after": not import_state_before["pyarrow"]
            and not import_state_after["pyarrow"],
            "torchvision_transitive_loaded_outside_scope": import_state_before[
                "torchvision"
            ]
            and import_state_after["torchvision"],
            "torchvision_root_object_identity_restored": sys.modules.get(
                "torchvision"
            )
            is torchvision_root_object,
            "hpl_scope_calls_exact": trainer.hpl_torchvision_scope_call_count
            == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "hpl_scope_restores_exact": trainer.hpl_torchvision_identity_restore_count
            == trainer.hpl_torchvision_scope_call_count,
            "pseudo_code_sha_exact": sha256_file(
                ROOT / "project/scripts/protocol_b_pseudo.py"
            )
            == EXPECTED_CODE_SHA256["project/scripts/protocol_b_pseudo.py"],
            "nms_code_sha_exact": sha256_file(
                ROOT / "project/ultralytics/utils/nms.py"
            )
            == EXPECTED_CODE_SHA256["project/ultralytics/utils/nms.py"],
        }
        import_audit = {
            "schema": "m1_seed42_stage1_training_process_import_audit_v1",
            "status": "PASS" if all(import_checks.values()) else "FAIL",
            "all_pass": all(import_checks.values()),
            "branch": branch,
            "before": import_state_before,
            "after": import_state_after,
            "torchvision_provenance": "TRANSITIVE_FROZEN_ULTRALYTICS_DEPENDENCY",
            "hpl_nms_backend": "ultralytics.utils.nms.TorchNMS.nms",
            "hpl_backend_protocol": "FROZEN_SCOPED_ROOT_KEY_POP_RESTORE_RESOURCE_WORKER_LINES_1307_1318",
            "sys_modules_root_identity_restored_each_call": True,
            "hpl_scope_call_count": trainer.hpl_torchvision_scope_call_count,
            "hpl_scope_restore_count": trainer.hpl_torchvision_identity_restore_count,
            "pseudo_code_sha256": EXPECTED_CODE_SHA256[
                "project/scripts/protocol_b_pseudo.py"
            ],
            "nms_code_sha256": EXPECTED_CODE_SHA256[
                "project/ultralytics/utils/nms.py"
            ],
            "checks": import_checks,
            "timestamp": now_iso(),
        }
        atomic_json(run_dir / "TRAINING_PROCESS_IMPORT_AUDIT.json", import_audit)
        atomic_json(run_dir / "HPL_RUNTIME_AUDIT.json", import_audit)
        if not import_audit["all_pass"]:
            raise FormalStop(f"STOP_TRAINING_IMPORT_AUDIT:{import_checks}")
        data_audit = _write_data_access_audit(
            run_dir,
            branch,
            data_guard,
            trainer.completed_data_access_epoch_rows,
            data_access_attempt_root,
        )
        converter = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "convert-parquet", "--run-dir", str(run_dir)],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "42"},
            check=False,
        )
        if converter.returncode != 0:
            raise FormalStop(f"STOP_OFFLINE_PARQUET_CONVERSION:{converter.returncode}")
        parquet_audit = _load_json(run_dir / "PARQUET_CONVERSION_AUDIT.json")
        core = _load_json(run_dir / "TRAINING_CORE_AUDIT.json")
        with (run_dir / "CHECKPOINT_MANIFEST.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        e100_full = [row for row in manifest if row["event_kind"] == "FULL_STATE_EPOCH" and int(row["epoch"]) == 100]
        e100_event = [row for row in manifest if row["event_name"] == "E100"]
        registered_rows = [
            row for row in manifest if row["event_kind"] == "FULL_STATE_EPOCH"
        ]
        cold_rows = [
            row
            for row in manifest
            if row["event_kind"]
            in {"TEACHER_FORWARD_ENTRY", "TARGET_CONTRIBUTING_OPTIMIZER_UPDATE", "COMPLETE_EPOCH_END"}
        ]
        cold_event_names = [str(row["event_name"]) for row in cold_rows]
        required_cold_events = {
            "T0", *T_EVENTS.values(), *E_EVENTS.values()
        }
        cold_by_name = {str(row["event_name"]): row for row in cold_rows}
        t_event_counters = {
            "T0": 0,
            **{event_name: count for count, event_name in T_EVENTS.items()},
        }
        e_event_epochs = {
            event_name: epoch for epoch, event_name in E_EVENTS.items()
        }
        recovery_rows = [
            row for row in manifest if row["event_kind"] == "RECOVERY_ONLY_EPOCH"
        ]
        checks = {
            "training_core_pass": core.get("all_pass") is True,
            "data_access_pass": data_audit.get("all_pass") is True,
            "parquet_conversion_pass": parquet_audit.get("all_pass") is True,
            "training_process_import_audit_pass": import_audit.get("all_pass")
            is True,
            "e100_full_state_exactly_one": len(e100_full) == 1,
            "e100_inference_event_exactly_one": len(e100_event) == 1,
            "registered_checkpoint_cadence_exact": [
                int(row["epoch"]) for row in registered_rows
            ]
            == [94, 98, 100]
            and all(
                row["full_state_path"]
                and row["raw_inference_path"]
                and row["ema_inference_path"]
                for row in registered_rows
            ),
            # Preserve the real append chronology.  Target-contributing events
            # can cross an epoch boundary when HPL is empty or an AMP window is
            # skipped, so no cross-kind chronology is scientifically fixed.
            "required_snapshot_names_exact": len(cold_event_names) == 9
            and len(set(cold_event_names)) == 9
            and set(cold_event_names) == required_cold_events,
            "t_snapshot_counters_exact": all(
                event_name in cold_by_name
                and int(
                    cold_by_name[event_name][
                        "target_contributing_optimizer_updates"
                    ]
                )
                == expected_count
                and cold_by_name[event_name]["event_kind"]
                == (
                    "TEACHER_FORWARD_ENTRY"
                    if event_name == "T0"
                    else "TARGET_CONTRIBUTING_OPTIMIZER_UPDATE"
                )
                for event_name, expected_count in t_event_counters.items()
            ),
            "epoch_snapshot_epochs_exact": all(
                event_name in cold_by_name
                and int(cold_by_name[event_name]["epoch"]) == expected_epoch
                and cold_by_name[event_name]["event_kind"]
                == "COMPLETE_EPOCH_END"
                for event_name, expected_epoch in e_event_epochs.items()
            ),
            "epoch_snapshot_internal_order_exact": [
                name for name in cold_event_names if name in e_event_epochs
            ]
            == list(E_EVENTS.values()),
            "snapshot_nonintervention_exact": all(
                str(row.get("snapshot_save_nonintervention_exact", "")).casefold()
                == "true"
                for row in cold_rows
            ),
            "recovery_only_not_evaluation_rows_exact": len(recovery_rows) == 2
            and {row["event_name"] for row in recovery_rows}
            == {"RECOVERY_PREV", "RECOVERY_LAST"}
            and all(not row["raw_inference_path"] and not row["ema_inference_path"] for row in recovery_rows),
            "target_dev_read_count_zero": data_audit["target_dev_read_count"] == 0,
            "target_test_read_count_zero": data_audit["target_test_read_count"] == 0,
        }
        final = {
            "schema": "m1_seed42_stage1_training_final_audit_v1",
            "status": "COMPLETED_PASS" if all(checks.values()) else "FAILED",
            "completed": all(checks.values()),
            "all_pass": all(checks.values()),
            "experiment": EXPERIMENT,
            "branch": branch,
            "final_epoch": 100,
            "end_epoch": 100,
            "final_global_step": E71_GLOBAL_STEP + len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
            "run_dir": str(run_dir.resolve()),
            "run_contract_path": str(contract_path),
            "run_contract_sha256": contract_sha,
            "runner_sha256": sha256_file(Path(__file__)),
            "optimizer_windows_attempted": int(
                core["optimizer_windows_this_branch"]
            ),
            "optimizer_updates_unskipped": int(
                core["actual_unskipped_optimizer_updates"]
            ),
            "optimizer_updates_skipped": int(
                core["optimizer_windows_this_branch"]
            )
            - int(core["actual_unskipped_optimizer_updates"]),
            "amp_found_inf_windows": int(core["amp_found_inf_windows"]),
            "ema_updates_final": int(core["ema_updates_final"]),
            "target_unsup_microbatch_count": int(
                core["target_unsup_microbatch_count"]
            ),
            "target_unsup_optimizer_update_count": int(
                core["target_contributing_optimizer_updates"]
            ),
            "teacher_target_forward_count": int(
                core["teacher_target_forward_count"]
            ),
            "fixed_cache_prelaunch_evidence_path": str(
                FIXED_CACHE_PRELAUNCH.resolve()
            ),
            "fixed_cache_prelaunch_evidence_sha256": sha256_file(
                FIXED_CACHE_PRELAUNCH
            ),
            "checkpoint_manifest_path": str((run_dir / "CHECKPOINT_MANIFEST.csv").resolve()),
            "checkpoint_manifest_sha256": sha256_file(run_dir / "CHECKPOINT_MANIFEST.csv"),
            "e100": {
                "full_state_path": e100_full[0]["full_state_path"] if e100_full else "",
                "full_state_file_sha256": e100_full[0]["full_state_file_sha256"] if e100_full else "",
                "raw_inference_path": e100_event[0]["raw_inference_path"] if e100_event else "",
                "raw_inference_file_sha256": e100_event[0]["raw_inference_file_sha256"] if e100_event else "",
                "raw_tensor_sha256": e100_event[0]["raw_tensor_sha256"] if e100_event else "",
                "ema_inference_path": e100_event[0]["ema_inference_path"] if e100_event else "",
                "ema_inference_file_sha256": e100_event[0]["ema_inference_file_sha256"] if e100_event else "",
                "ema_tensor_sha256": e100_event[0]["ema_tensor_sha256"] if e100_event else "",
            },
            "target_adapt_label_read_count": 0,
            "target_dev_read_count": 0,
            "target_test_read_count": 0,
            "dataset_mutation_count": 0,
            "checks": checks,
            "timestamp": now_iso(),
        }
        atomic_json(run_dir / "TRAINING_FINAL_AUDIT.json", final)
        if not final["all_pass"]:
            raise FormalStop(f"STOP_FORMAL_FINAL_AUDIT:{checks}")
        lease.mark_completed()
    except BaseException as exc:
        lease.mark_failed(exc)
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            run_dir / "FAILURE.json",
            {
                "schema": "m1_seed42_stage1_failure_v1", "status": "FAILED",
                "branch": branch, "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "target_dev_read_count": 0,
                "target_test_read_count": 0, "timestamp": now_iso(),
            },
        )
        try:
            if data_guard is not None:
                _write_data_access_audit(
                    run_dir,
                    branch,
                    data_guard,
                    (
                        trainer.completed_data_access_epoch_rows
                        if trainer is not None
                        else list(
                            state.get("completed_data_access_epoch_rows", [])
                        )
                    ),
                    data_access_attempt_root,
                )
        except Exception:
            pass
        raise
    return 0


def _postrun_verify_fixed_cache(run_dir: Path) -> dict[str, Any]:
    """One post-training, full-cache payload verification outside the train loop."""
    contract, _ = load_run_contract(
        DEFAULT_CONTRACT, authority="formal-branch"
    )
    cache_evidence = contract["fixed_cache_evidence"]
    if sha256_file(FIXED_CACHE_BUILD) != EXPECTED_INPUT_SHA256["fixed_cache_build"]:
        raise FormalStop("STOP_POSTRUN_CACHE_BUILD_AUTHORITY_SHA")
    if sha256_file(FIXED_CACHE_MANIFEST) != EXPECTED_INPUT_SHA256["fixed_cache_manifest"]:
        raise FormalStop("STOP_POSTRUN_CACHE_MANIFEST_SHA")
    schedule = ProtocolBTargetSchedule(TARGET_SCHEDULE, TARGET_SCHEDULE_MANIFEST)
    _, cache, _ = load_fixed_inputs(schedule)
    rows: list[dict[str, Any]] = []
    composite_entries: list[str] = []
    started = time.perf_counter()
    failure = ""
    for uid in sorted(cache):
        entry = cache[uid]
        item_started = time.perf_counter()
        try:
            path = inside_root(Path(entry.bgr_path))
            file_bytes = path.stat().st_size
            mmap = np.load(path, mmap_mode="r", allow_pickle=False)
            shape = tuple(int(value) for value in mmap.shape)
            dtype = str(mmap.dtype)
            representation_exact = (
                dtype == "uint8"
                and shape == entry.bgr_shape
                and file_bytes == entry.bgr_file_bytes
            )
            if not representation_exact:
                raise FormalStop(f"STOP_POSTRUN_CACHE_REPRESENTATION:{uid}")
            file_sha = sha256_file(path)
            file_sha_exact = file_sha == entry.bgr_file_sha256
            array_sha, bgr_sha = exact_bgr_payload_hashes(mmap)
            del mmap
            array_exact = array_sha == entry.bgr_array_bytes_sha256
            bgr_exact = bgr_sha == entry.expected_output_bgr_sha256
            passed = (
                representation_exact and file_sha_exact and array_exact and bgr_exact
            )
            composite_entries.append(
                f"{uid}\0{entry.input_image_sha256}\0{entry.solve_key}\0"
                f"{entry.expected_output_bgr_sha256}\0{file_bytes}\0{file_sha}\n"
            )
            row = {
                "schema": "m1_seed42_stage1_fixed_cache_postrun_reverification_entry_v1",
                "canonical_uid": uid,
                "bgr_path": str(path.resolve()),
                "file_bytes": file_bytes,
                "expected_file_bytes": entry.bgr_file_bytes,
                "input_image_sha256": entry.input_image_sha256,
                "solve_key": entry.solve_key,
                "dtype": dtype,
                "shape_json": json.dumps(shape, separators=(",", ":")),
                "expected_shape_json": json.dumps(
                    entry.bgr_shape, separators=(",", ":")
                ),
                "expected_array_bytes_sha256": entry.bgr_array_bytes_sha256,
                "actual_array_bytes_sha256": array_sha,
                "expected_file_sha256": entry.bgr_file_sha256,
                "actual_file_sha256": file_sha,
                "expected_formal_bgr_sha256": entry.expected_output_bgr_sha256,
                "actual_formal_bgr_sha256": bgr_sha,
                "representation_exact": representation_exact,
                "whole_file_sha256_exact": file_sha_exact,
                "array_bytes_sha256_exact": array_exact,
                "formal_bgr_sha256_exact": bgr_exact,
                "all_pass": passed,
                "wall_seconds": time.perf_counter() - item_started,
                "error": "",
            }
        except BaseException as exc:
            row = {
                "schema": "m1_seed42_stage1_fixed_cache_postrun_reverification_entry_v1",
                "canonical_uid": uid,
                "bgr_path": entry.bgr_path,
                "file_bytes": "",
                "expected_file_bytes": entry.bgr_file_bytes,
                "input_image_sha256": entry.input_image_sha256,
                "solve_key": entry.solve_key,
                "dtype": "",
                "shape_json": "",
                "expected_shape_json": json.dumps(
                    entry.bgr_shape, separators=(",", ":")
                ),
                "expected_array_bytes_sha256": entry.bgr_array_bytes_sha256,
                "actual_array_bytes_sha256": "",
                "expected_file_sha256": entry.bgr_file_sha256,
                "actual_file_sha256": "",
                "expected_formal_bgr_sha256": entry.expected_output_bgr_sha256,
                "actual_formal_bgr_sha256": "",
                "representation_exact": False,
                "whole_file_sha256_exact": False,
                "array_bytes_sha256_exact": False,
                "formal_bgr_sha256_exact": False,
                "all_pass": False,
                "wall_seconds": time.perf_counter() - item_started,
                "error": f"{type(exc).__name__}:{exc}",
            }
            failure = row["error"]
        rows.append(row)
        if not row["all_pass"]:
            break
    csv_path = run_dir / "FIXED_CACHE_POSTRUN_REVERIFICATION.csv"
    atomic_csv(csv_path, rows)
    actual_composite_sha = hashlib.sha256(
        "".join(sorted(composite_entries)).encode("utf-8")
    ).hexdigest()
    expected_composite_sha = require_sha(
        cache_evidence["payload_composite_sha256"],
        "fixed cache prelaunch payload composite SHA",
    )
    checks = {
        "entries_exact": len(rows) == 3_226,
        "all_payloads_exact": len(rows) == 3_226
        and all(bool(row["all_pass"]) for row in rows),
        "cache_build_authority_sha_exact": sha256_file(FIXED_CACHE_BUILD)
        == EXPECTED_INPUT_SHA256["fixed_cache_build"],
        "cache_manifest_sha_exact": sha256_file(FIXED_CACHE_MANIFEST)
        == EXPECTED_INPUT_SHA256["fixed_cache_manifest"],
        "whole_file_sha256_all_exact": len(rows) == 3_226
        and all(bool(row["whole_file_sha256_exact"]) for row in rows),
        "payload_total_bytes_exact": len(rows) == 3_226
        and all(row["file_bytes"] != "" for row in rows)
        and sum(int(row["file_bytes"]) for row in rows)
        == int(cache_evidence["payload_total_bytes"])
        == 29_612_305_664,
        "payload_composite_sha256_exact": len(composite_entries) == 3_226
        and actual_composite_sha == expected_composite_sha,
    }
    audit = {
        "schema": "m1_seed42_stage1_fixed_cache_postrun_reverification_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()),
        "entry_count": len(rows),
        "mismatch_count": sum(not bool(row["all_pass"]) for row in rows),
        "cache_build_authority_sha256": EXPECTED_INPUT_SHA256["fixed_cache_build"],
        "fixed_pretrain_authority_sha256": EXPECTED_INPUT_SHA256[
            "fixed_pretrain_final_status"
        ],
        "cache_manifest_sha256": EXPECTED_INPUT_SHA256["fixed_cache_manifest"],
        "payload_hash_mode": "FULL_FILE_SHA256",
        "expected_payload_composite_sha256": expected_composite_sha,
        "actual_payload_composite_sha256": actual_composite_sha,
        "payload_total_bytes": sum(
            int(row["file_bytes"]) for row in rows if row["file_bytes"] != ""
        ),
        "evidence_csv": str(csv_path.resolve()),
        "evidence_csv_sha256": sha256_file(csv_path),
        "wall_seconds": time.perf_counter() - started,
        "failure": failure,
        "checks": checks,
        "timestamp": now_iso(),
    }
    atomic_json(run_dir / "FIXED_CACHE_POSTRUN_REVERIFICATION.json", audit)
    if not audit["all_pass"]:
        raise FormalStop(f"STOP_POSTRUN_CACHE_REVERIFICATION:{failure}")
    return audit


def run_convert_parquet(args: argparse.Namespace) -> int:
    # This subcommand is intentionally a fresh process.  The training process
    # invokes it only after trainer.train() has returned.
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_dir = inside_root(Path(args.run_dir))
    raw_dir = run_dir / "raw_trace"

    def read_jsonl(paths: list[Path], label: str) -> list[dict[str, Any]]:
        if len(paths) != len(EPOCHS):
            raise FormalStop(f"STOP_OFFLINE_RAW_EPOCH_COUNT:{label}:{len(paths)}")
        rows: list[dict[str, Any]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        return rows

    with (run_dir / "TRAINING_EPOCH_METRICS.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        epoch_metrics = list(csv.DictReader(handle))
    if (
        len(epoch_metrics) != len(EPOCHS)
        or [int(row["epoch"]) for row in epoch_metrics] != list(EPOCHS)
    ):
        raise FormalStop("STOP_OFFLINE_EPOCH_METRIC_LINEAGE")
    branch = str(epoch_metrics[0]["branch"])
    if branch not in BRANCHES or any(
        row["branch"] != branch for row in epoch_metrics
    ):
        raise FormalStop("STOP_OFFLINE_BRANCH_LINEAGE")

    def registered_paths(
        path_field: str,
        sha_field: str,
        rows_field: str,
        *,
        expected_rows: int | None,
        applicable: bool = True,
    ) -> list[Path]:
        if not applicable:
            if any(
                row.get(path_field) or row.get(sha_field)
                or int(row.get(rows_field, -1)) != 0
                for row in epoch_metrics
            ):
                raise FormalStop(f"STOP_OFFLINE_NONAPPLICABLE_SHARD:{path_field}")
            return []
        paths: list[Path] = []
        for row in epoch_metrics:
            path = inside_root(Path(str(row[path_field])))
            registered_rows = int(row[rows_field])
            if not (
                path.is_relative_to((raw_dir / "attempts").resolve())
                and path.is_file()
                and sha256_file(path) == require_sha(row[sha_field], sha_field)
                and registered_rows >= 0
                and (expected_rows is None or registered_rows == expected_rows)
            ):
                raise FormalStop(f"STOP_OFFLINE_RAW_SHARD:{path_field}:{path}")
            paths.append(path)
        return paths

    box_paths = registered_paths(
        "hpl_box_trace_path", "hpl_box_trace_sha256", "hpl_box_trace_rows",
        expected_rows=None,
    )
    batch_paths = registered_paths(
        "pseudo_batch_trace_path", "pseudo_batch_trace_sha256",
        "pseudo_batch_trace_rows", expected_rows=EXPECTED_STEPS_PER_EPOCH,
    )
    exposure_paths = registered_paths(
        "fixed_exposure_runtime_path", "fixed_exposure_runtime_sha256",
        "fixed_exposure_runtime_rows", expected_rows=EXPECTED_STEPS_PER_EPOCH,
    )
    cache_paths = registered_paths(
        "fixed_cache_runtime_path", "fixed_cache_runtime_sha256",
        "fixed_cache_runtime_rows", expected_rows=EXPECTED_STEPS_PER_EPOCH,
        applicable=branch == "NATIVE_A_FIXED",
    )
    box_rows = read_jsonl(box_paths, "hpl_boxes")
    batch_rows = read_jsonl(batch_paths, "pseudo_batches")
    exposure_rows = read_jsonl(exposure_paths, "fixed_exposure")
    cache_rows = (
        read_jsonl(cache_paths, "fixed_cache")
        if branch == "NATIVE_A_FIXED"
        else []
    )
    training_import_audit = _load_json(
        run_dir / "TRAINING_PROCESS_IMPORT_AUDIT.json"
    )
    postrun_cache_audit = (
        _postrun_verify_fixed_cache(run_dir)
        if branch == "NATIVE_A_FIXED"
        else {"status": "NOT_APPLICABLE_CONTROL", "all_pass": True}
    )
    if len(batch_rows) != len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH:
        raise FormalStop("STOP_PSEUDO_BATCH_ROW_COUNT")
    atomic_csv(run_dir / "PSEUDO_LABEL_BATCH_TRACE.csv", batch_rows)
    tensor_manifest = [
        {
            "schema": "m1_seed42_stage1_hpl_tensor_manifest_v1",
            "branch": row["branch"], "epoch": row["epoch"],
            "source_batch_index": row["source_batch_index"],
            "formal_visit_index_json": row["formal_visit_index_json"],
            "teacher_uint8_sha256": row["teacher_uint8_sha256"],
            "student_uint8_sha256": row["student_uint8_sha256"],
            "pseudo_cls_sha256": row["pseudo_cls_sha256"],
            "pseudo_bboxes_sha256": row["pseudo_bboxes_sha256"],
            "pseudo_batch_idx_sha256": row["pseudo_batch_idx_sha256"],
            "pseudo_scores_f32le_sha256": row["pseudo_scores_f32le_sha256"],
            "pseudo_final_boxes": row["pseudo_final_boxes"],
        }
        for row in batch_rows
    ]
    atomic_csv(run_dir / "HPL_TENSOR_MANIFEST.csv", tensor_manifest)
    boxes_by_visit: dict[int, list[dict[str, Any]]] = {}
    for box in box_rows:
        boxes_by_visit.setdefault(int(box["formal_visit_index"]), []).append(box)
    image_rows: list[dict[str, Any]] = []
    for row in batch_rows:
        uids = json.loads(row["target_uid_json"])
        canonical = json.loads(row["canonical_uid_json"])
        visits = json.loads(row["formal_visit_index_json"])
        active = json.loads(row["fixed_exposure_active_json"])
        applied = json.loads(row["fixed_transform_applied_json"])
        pre = json.loads(row["pseudo_pre_nms_per_image_json"])
        post = json.loads(row["pseudo_post_nms_per_image_json"])
        final = json.loads(row["pseudo_final_per_image_json"])
        if not all(
            len(values) == TARGET_BATCH
            for values in (uids, canonical, visits, active, applied, pre, post, final)
        ):
            raise FormalStop("STOP_PSEUDO_IMAGE_BATCH_CARDINALITY")
        for position in range(TARGET_BATCH):
            visit_boxes = boxes_by_visit.get(int(visits[position]), [])
            if len(visit_boxes) != int(final[position]):
                raise FormalStop(
                    f"STOP_HPL_BOX_LINK_CARDINALITY:{visits[position]}:"
                    f"{len(visit_boxes)}:{final[position]}"
                )
            scores = [float(item["score"]) for item in visit_boxes]
            boxes_xywh = [
                [
                    float(item["x_center"]), float(item["y_center"]),
                    float(item["width"]), float(item["height"]),
                ]
                for item in visit_boxes
            ]
            areas = [float(item["area"]) for item in visit_boxes]
            image_rows.append(
                {
                    "schema": "m1_seed42_stage1_pseudo_label_image_trace_v1",
                    "branch": row["branch"], "epoch": row["epoch"],
                    "source_batch_index": row["source_batch_index"],
                    "global_microbatch_index": row["global_microbatch_index"],
                    "global_step_after_microbatch": row["global_step_after_microbatch"],
                    "optimizer_step_occurred": row["optimizer_step_occurred"],
                    "optimizer_underlying_step_called": row["optimizer_underlying_step_called"],
                    "target_unsupervised_optimizer_update_count": row[
                        "target_unsupervised_optimizer_update_count"
                    ],
                    "target_position": position,
                    "formal_visit_index": visits[position],
                    "schedule_uid": uids[position],
                    "canonical_uid": canonical[position],
                    "fixed_exposure_active": active[position],
                    "fixed_transform_applied": applied[position],
                    "pre_nms_candidates": pre[position],
                    "post_nms_boxes": post[position],
                    "final_hpl_boxes": final[position],
                    "hpl_positive": int(final[position] > 0),
                    "hpl_empty": int(final[position] == 0),
                    "hpl_scores_json": json.dumps(scores, separators=(",", ":")),
                    "hpl_boxes_xywh_json": json.dumps(boxes_xywh, separators=(",", ":")),
                    "hpl_areas_json": json.dumps(areas, separators=(",", ":")),
                    "hpl_box_trace_link_sha256": canonical_json_sha256(
                        {
                            "formal_visit_index": visits[position],
                            "scores": scores,
                            "boxes_xywh": boxes_xywh,
                            "areas": areas,
                        }
                    ),
                }
            )

    parquet_payloads = {
        "HPL_BOX_TRACE.parquet": box_rows,
        "PSEUDO_LABEL_IMAGE_TRACE.parquet": image_rows,
    }
    outputs: list[dict[str, Any]] = []
    for name, rows in parquet_payloads.items():
        destination = run_dir / name
        temporary = destination.with_name(destination.name + ".tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, destination)
        outputs.append({
            "path": str(destination.resolve()),
            "file_sha256": sha256_file(destination), "row_count": len(rows),
        })
    if branch == "NATIVE_A_FIXED":
        atomic_csv(run_dir / "FIXED_CACHE_RUNTIME_AUDIT.csv", cache_rows)
        atomic_csv(run_dir / "FIXED_EXPOSURE_RUNTIME_AUDIT.csv", exposure_rows)
    checks = {
        "outputs_two": len(outputs) == 2,
        "pseudo_batch_rows_exact": len(batch_rows) == len(EPOCHS) * EXPECTED_STEPS_PER_EPOCH,
        "pseudo_image_rows_exact": len(image_rows) == TOTAL_TARGET_VISITS,
        "pseudo_image_visit_indices_exact": sorted(
            int(row["formal_visit_index"]) for row in image_rows
        )
        == list(range(TOTAL_TARGET_VISITS)),
        "hpl_box_rows_exactly_linked": sum(
            int(row["final_hpl_boxes"]) for row in image_rows
        )
        == len(box_rows),
        "hpl_tensor_manifest_rows_exact": len(tensor_manifest) == len(batch_rows),
        "fixed_exposure_rows_exact": len(exposure_rows) == len(batch_rows),
        "fixed_active_identity_counts_exact": (
            sum(int(row["fixed_active_count"]) for row in exposure_rows) == 62_679
            and sum(int(row["fixed_identity_count"]) for row in exposure_rows) == 65_385
        ),
        "fixed_cache_rows_exact_if_applicable": branch != "NATIVE_A_FIXED"
        or len(cache_rows) == len(batch_rows),
        "fixed_cache_no_miss_or_identity_access": branch != "NATIVE_A_FIXED"
        or all(
            int(row["cache_miss_count"]) == 0
            and int(row["cache_fallback_count"]) == 0
            and int(row["identity_cache_access_count"]) == 0
            and int(row["online_theta_solve_count"]) == 0
            for row in cache_rows
        ),
        "fixed_runtime_payload_hashing_disabled": branch != "NATIVE_A_FIXED"
        or all(int(row["cache_payload_validation_count"]) == 0 for row in cache_rows),
        "fixed_cache_hits_exact": branch != "NATIVE_A_FIXED"
        or sum(int(row["cache_hit_count"]) for row in cache_rows) == 62_679,
        "fixed_cache_authority_exact": branch != "NATIVE_A_FIXED"
        or all(
            row["cache_build_authority_sha256"]
            == EXPECTED_INPUT_SHA256["fixed_cache_build"]
            and row["fixed_pretrain_authority_sha256"]
            == EXPECTED_INPUT_SHA256["fixed_pretrain_final_status"]
            for row in cache_rows
        ),
        "fixed_postrun_full_payload_reverification_pass": postrun_cache_audit.get(
            "all_pass"
        )
        is True,
        "training_process_offline_only_imports_absent": training_import_audit.get(
            "all_pass"
        )
        is True
        and training_import_audit.get("before", {}).get("pandas") is False
        and training_import_audit.get("after", {}).get("pandas") is False
        and training_import_audit.get("before", {}).get("pyarrow") is False
        and training_import_audit.get("after", {}).get("pyarrow") is False,
        "torchvision_transitive_dependency_scoped_local_nms_exact": training_import_audit.get(
            "before", {}
        ).get("torchvision")
        is True
        and training_import_audit.get("after", {}).get("torchvision") is True
        and training_import_audit.get("hpl_nms_backend")
        == "ultralytics.utils.nms.TorchNMS.nms",
    }
    audit = {
        "schema": "m1_seed42_stage1_parquet_conversion_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()), "outputs": outputs,
        "fixed_cache_postrun_reverification": postrun_cache_audit,
        "checks": checks,
        "process_id": os.getpid(), "timestamp": now_iso(),
    }
    atomic_json(run_dir / "PARQUET_CONVERSION_AUDIT.json", audit)
    if not audit["all_pass"]:
        raise FormalStop("STOP_PARQUET_AUDIT")
    return 0


def run_self_check() -> int:
    source = Path(__file__).read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    # pyarrow occurs only inside the isolated converter; pandas and torchvision
    # must never be imported anywhere by this file.
    top_level_forbidden = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_forbidden.extend(
                alias.name for alias in node.names if alias.name.split(".")[0] in FORBIDDEN_RUNTIME_IMPORTS
            )
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in FORBIDDEN_RUNTIME_IMPORTS:
            top_level_forbidden.append(node.module)
    converter_node = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_convert_parquet"
    )
    pyarrow_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "pyarrow" for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "pyarrow"
        )
    ]
    dummy_view_seed = 987654321
    dummy_global_seed = 42
    synthetic_branch_section: dict[str, Any] = {
        branch: {"run_dir": BRANCH_RUN_NAMES[branch]} for branch in BRANCHES
    }
    synthetic_branch_section.update(deepcopy(BRANCH_CONTRACT_METADATA))
    synthetic_missing_branch_metadata = dict(synthetic_branch_section)
    synthetic_missing_branch_metadata.pop("fresh_process_per_branch")
    synthetic_extra_branch_metadata = dict(synthetic_branch_section)
    synthetic_extra_branch_metadata["unfrozen_extra"] = True

    def source_block(text: str, start: str, end: str) -> str:
        start_at = text.find(start)
        end_at = text.find(end, start_at + len(start)) if start_at >= 0 else -1
        return text[start_at:end_at] if start_at >= 0 and end_at > start_at else ""

    b1_path = SCRIPT_DIR / "protocol_b_b1_trainer.py"
    b1_source = b1_path.read_text(encoding="utf-8")
    b1_preprocess = source_block(
        b1_source, "    def preprocess_batch(", "    def optimizer_step("
    )
    b1_optimizer = source_block(
        b1_source, "    def optimizer_step(", "    def _protocol_on_epoch_end("
    )
    stage1_init = source_block(source, "    def __init__(", "    def get_model(")
    stage1_load_exact_bgr = source_block(
        source, "    def _load_exact_bgr(", "    def __getitem__("
    )
    stage1_preprocess = source_block(
        source, "    def preprocess_batch(", "    def optimizer_step("
    )
    stage1_optimizer = source_block(
        source, "    def optimizer_step(", "    def _protocol_b2_on_batch_end("
    )
    stage1_snapshot_global = source_block(
        source, "    def _snapshot_global_step(", "    def _save_snapshot("
    )
    stage1_snapshot_save = source_block(
        source, "    def _save_snapshot(", "    def _protocol_on_epoch_end("
    )
    stage1_epoch_end = source_block(
        source, "    def _protocol_on_epoch_end(", "    def _full_state_payload("
    )
    stage1_full_state = source_block(
        source, "    def _full_state_payload(", "    def save_model("
    )
    stage1_core_end = source_block(
        source, "    def _protocol_on_train_end(", "def validate_start_state("
    )
    converter_source = source_block(
        source, "def run_convert_parquet(", "def run_self_check("
    )
    nms_path = LOCAL_ULTRALYTICS / "utils" / "nms.py"
    nms_source = nms_path.read_text(encoding="utf-8")

    b1_counter_at = b1_preprocess.find("self.protocol_batch_index += 1")
    b1_return_at = b1_preprocess.find("return super().preprocess_batch(batch)")
    b1_optimizer_order = [
        b1_optimizer.find(value)
        for value in (
            "self.scaler.step(self.optimizer)",
            "self.scaler.update()",
            "self.optimizer.zero_grad()",
            "self.ema.update(self.model)",
            "self.protocol_optimizer_steps += 1",
        )
    ]
    e100_dict_key_sets = [
        [
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
        and key.value == "e100"
        and isinstance(value, ast.Dict)
    ]
    checks = {
        "ast_parse": True,
        "single_new_runner_identity": Path(__file__).name == "m1_seed42_stage1_formal_train_v1.py",
        "no_top_level_forbidden_imports": not top_level_forbidden,
        "pandas_not_imported_by_source": "pandas" not in imports,
        "torchvision_not_imported_by_source": "torchvision" not in imports,
        "pyarrow_only_in_converter": len(pyarrow_import_lines) == 2
        and all(
            converter_node.lineno <= line <= int(converter_node.end_lineno or line)
            for line in pyarrow_import_lines
        ),
        "epochs_exact": EPOCHS == tuple(range(72, 101)),
        "counts_exact": TOTAL_TARGET_VISITS == 128_064 and TOTAL_WINDOWS == 8_004,
        "events_exact": list(T_EVENTS.values()) == ["T10", "T50", "T100"] and list(E_EVENTS.values()) == ["E72_END", "E75", "E80", "E90", "E100"],
        "snapshot_gate_allows_cross_kind_interleaving": all(
            value in source
            for value in (
                "set(cold_event_names) == required_cold_events",
                '"t_snapshot_counters_exact"',
                '"epoch_snapshot_epochs_exact"',
                '"epoch_snapshot_internal_order_exact"',
            )
        ),
        "adaptation_seed_not_view_seed": dummy_global_seed == 42 and dummy_view_seed != dummy_global_seed,
        "fixed_index_formula": (100 - 72) * 4_416 + 4_415 == 128_063,
        "contract_cycle_broken": "formal_train_runner_sha256" in source
        and DEFAULT_CONTRACT.name == "RUN_CONTRACT.json"
        and "STOP_RUN_CONTRACT_NONCANONICAL_PATH" in source,
        "contract_result_root_and_branch_runs_exact": all(
            value in source
            for value in (
                'paths.get("result_root")',
                'set(branch_runs) != set(BRANCHES)',
                'set(branches) != BRANCH_CONTRACT_KEYS',
                'BRANCH_CONTRACT_METADATA.items()',
                'branches[branch].get("run_dir")',
                'branch_runs[branch]',
                'STOP_CONTRACT_RESULT_ROOT',
                'STOP_CONTRACT_BRANCH_RUN_KEYS',
            )
        ),
        "contract_branch_metadata_keyset_synthetic": (
            set(synthetic_branch_section) == BRANCH_CONTRACT_KEYS
            and all(
                synthetic_branch_section[key] == expected
                for key, expected in BRANCH_CONTRACT_METADATA.items()
            )
            and set(synthetic_missing_branch_metadata) != BRANCH_CONTRACT_KEYS
            and set(synthetic_extra_branch_metadata) != BRANCH_CONTRACT_KEYS
        ),
        "b1_train_metrics_authority_exact": (
            B1_METRICS
            == ROOT
            / "project"
            / "runs"
            / "final_v2_1"
            / "protocol_b_b1_source_only_e71_e100_20260724_181826"
            / "train_epoch_metrics.csv"
            and B1_METRICS.is_file()
            and sha256_file(B1_METRICS)
            == EXPECTED_INPUT_SHA256["b1_train_metrics"]
            and '"b1_train_metrics": (' in source
            and '"b1_train_metrics": B1_METRICS' in source
        ),
        "observer_calls_original_once_in_function": source[source.index("def _observing_generate"):source.index("def _protocol_on_epoch_start")].count("generate_high_confidence_pseudo(") == 1,
        "no_online_solver_import": "scipy" not in imports,
        "registered_checkpoint_cadence_exact": "if epoch not in {94, 98, 100}" in source,
        "preflight_snapshot_writes_disabled": "not self.preflight_mode" in source,
        "preflight_sampler_is_exact_ten_batch_prefix_and_workers_shutdown": (
            "class PrefixTargetBatchSampler" in source
            and "max_batches=PREFLIGHT_MICROBATCHES if self.preflight_mode else None"
            in source
            and 'getattr(target_iterator.inner, "_shutdown_workers", None)'
            in source
            and '"target_image_reads_exact"' in source
        ),
        "prelaunch_full_cache_payload_evidence_contract_locked": all(
            value in source
            for value in (
                '"prelaunch_evidence_locks"',
                '"fixed_cache_evidence"',
                '"payload_hash_mode": "FULL_FILE_SHA256"',
                '"payload_file_count": 3_226',
                '"payload_total_bytes": 29_612_305_664',
                '"cache_miss_policy": "STOP_NO_FALLBACK"',
                '"online_powell_policy": "FORBIDDEN_ZERO"',
            )
        ),
        "frozen_b1_counter_source_sha_exact": sha256_file(b1_path)
        == EXPECTED_CODE_SHA256["project/scripts/protocol_b_b1_trainer.py"],
        "b1_counter_increment_precedes_optimizer_callback": b1_counter_at >= 0
        and b1_return_at > b1_counter_at,
        "tn_global_step_uses_completed_protocol_counter_without_plus_one": (
            "return epoch_start + int(self.protocol_batch_index)"
            in stage1_snapshot_global
            and "return epoch_start + int(self.protocol_batch_index) + 1"
            not in stage1_snapshot_global
        ),
        "scoped_torchvision_root_key_pop_restore_exact": all(
            value in stage1_preprocess
            for value in (
                'torchvision_module = sys.modules.pop("torchvision", None)',
                'if "torchvision" in sys.modules:',
                'sys.modules["torchvision"] = torchvision_module',
                'sys.modules.get("torchvision") is not torchvision_module',
                'if scope_imported_torchvision:',
            )
        ),
        "frozen_local_torchnms_branch_exact": (
            sha256_file(nms_path)
            == EXPECTED_CODE_SHA256["project/ultralytics/utils/nms.py"]
            and 'if "torchvision" in sys.modules:' in nms_source
            and "i = TorchNMS.nms(boxes, scores, iou_thres)" in nms_source
            and '"hpl_nms_backend": "ultralytics.utils.nms.TorchNMS.nms"'
            in stage1_preprocess
        ),
        "long_run_has_no_per_visit_bgr_payload_rehash": (
            "verify_payload_on_read=self.preflight_mode" in source
            and "if payload_validation:" in stage1_load_exact_bgr
            and stage1_load_exact_bgr.count("array_bytes_sha(bgr)") == 1
            and stage1_load_exact_bgr.count("formal_bgr_sha(bgr)") == 1
        ),
        "snapshot_state_and_rng_nonintervention_exact": (
            "if before != after:" in stage1_snapshot_save
            and all(
                stage1_snapshot_save.count(key) >= 2
                for key in (
                    '"raw_tensor_sha256"', '"ema_tensor_sha256"',
                    '"optimizer_state_sha256"', '"scheduler_state_sha256"',
                    '"scaler_state_sha256"', '"rng_state_sha256"',
                    '"rng_hashes"',
                )
            )
        ),
        "resume_cumulative_state_and_shards_preserved": (
            all(
                key in stage1_full_state and key in stage1_init
                for key in (
                    "optimizer_update_rows", "training_epoch_metric_rows",
                    "completed_data_access_epoch_rows",
                    "protocol_epoch_rows", "protocol_b2_epoch_rows",
                    "protocol_b2_batch_rows",
                )
            )
            and "write_jsonl_exclusive(" in stage1_epoch_end
            and 'raw / f"fixed_exposure_runtime_e{epoch:03d}.jsonl"'
            in stage1_epoch_end
            and 'raw / f"fixed_cache_runtime_e{epoch:03d}.jsonl"'
            in stage1_epoch_end
            and 'len(paths) != len(EPOCHS)' in converter_source
            and '== TOTAL_TARGET_VISITS' in converter_source
            and '== 62_679' in converter_source
            and '== 65_385' in converter_source
            and 'raw_trace"\n            / "attempts"' in source
            and 'inference_state"\n            / "attempts"' in source
        ),
        "recovery_inherits_only_complete_state_snapshots": all(
            value in source
            for value in (
                'payload_snapshots = state.get("snapshot_rows")',
                'manifest_by_event',
                'state["snapshot_rows"] = accepted_snapshots',
                'STOP_RECOVERY_SNAPSHOT_MANIFEST_FIELDS',
                'STOP_RECOVERY_SNAPSHOT_ARTIFACT',
            )
        ),
        "final_e100_hybrid_endpoint_has_exact_eight_fields": e100_dict_key_sets
        == [[
            "full_state_path", "full_state_file_sha256",
            "raw_inference_path", "raw_inference_file_sha256",
            "raw_tensor_sha256", "ema_inference_path",
            "ema_inference_file_sha256", "ema_tensor_sha256",
        ]]
        and '"raw_inference_path": e100_event[0]' in source
        and '"full_state_path": e100_full[0]' in source,
        "postrun_full_file_and_composite_cache_reverification_exact": all(
            value in source
            for value in (
                'file_sha_exact = file_sha == entry.bgr_file_sha256',
                '"whole_file_sha256_all_exact"',
                '"payload_composite_sha256_exact"',
                '"".join(sorted(composite_entries)).encode("utf-8")',
            )
        ),
        "found_inf_uses_frozen_gradscaler_ema_semantics_without_actual_step_hard_gate": (
            all(position >= 0 for position in b1_optimizer_order)
            and b1_optimizer_order == sorted(b1_optimizer_order)
            and all(
                value in stage1_optimizer
                for value in (
                    "self.scaler.step(self.optimizer)",
                    "self.scaler.update()",
                    "self.optimizer.zero_grad()",
                    "self.ema.update(self.model)",
                    "self.protocol_optimizer_steps += 1",
                    "self.actual_unskipped_optimizer_updates +=",
                    "self.amp_found_inf_windows +=",
                )
            )
            and '"optimizer_windows_exact": int(self.protocol_optimizer_steps) == TOTAL_WINDOWS'
            in stage1_core_end
            and '"ema_updates_exact": int(self.ema.updates) == E71_EMA_UPDATES + TOTAL_WINDOWS'
            in stage1_core_end
            and "actual_unskipped_optimizer_updates) == TOTAL_WINDOWS"
            not in stage1_core_end
            and '"amp_found_inf_windows_zero"' not in stage1_core_end
        ),
        "required_output_names_present": all(
            name in source
            for name in (
                "TRAINING_FINAL_AUDIT.json", "CHECKPOINT_MANIFEST.csv",
                "TRAINING_EPOCH_METRICS.csv", "COLD_START_SNAPSHOT_MANIFEST.csv",
                "DATA_ACCESS_AUDIT.json", "PSEUDO_LABEL_BATCH_TRACE.csv",
                "PSEUDO_LABEL_IMAGE_TRACE.parquet", "HPL_TENSOR_MANIFEST.csv",
                "HPL_BOX_TRACE.parquet", "OPTIMIZER_UPDATE_TRACE.csv",
                "FIXED_CACHE_RUNTIME_AUDIT.csv",
                "FIXED_EXPOSURE_RUNTIME_AUDIT.csv",
            )
        ),
    }
    result = {
        "schema": "m1_seed42_stage1_formal_train_self_check_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "runner_sha256": sha256_file(Path(__file__)),
        "contract_expected_path": str(DEFAULT_CONTRACT.resolve()),
        "contract_expected_schema": SCHEMA,
        "gpu_used": False, "checkpoint_read": False, "target_image_read": False,
        "timestamp": now_iso(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check", help="CPU-only static/self checks; no contract required")

    preflight = sub.add_parser("logging-preflight", help="four fresh-e71 OFF/ON equivalence workers")
    preflight.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    preflight.add_argument("--device", default="0")
    preflight.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    preflight.add_argument("--branch", choices=BRANCHES, help=argparse.SUPPRESS)
    preflight.add_argument("--logging", choices=("OFF", "ON"), help=argparse.SUPPRESS)
    preflight.add_argument("--output", help=argparse.SUPPRESS)

    formal = sub.add_parser("formal-branch", help="run one authorized formal branch e72-e100")
    formal.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    formal.add_argument("--branch", choices=BRANCHES, required=True)
    formal.add_argument("--device", default="0")
    formal.add_argument("--resume-from", default=None)

    convert = sub.add_parser("convert-parquet", help="offline post-training JSONL to Parquet child")
    convert.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-check":
        return run_self_check()
    if args.command == "logging-preflight":
        if args.worker:
            if not (args.branch and args.logging and args.output):
                raise FormalStop("STOP_PREFLIGHT_WORKER_ARGUMENTS")
            return run_preflight_worker(args)
        return run_logging_preflight(args)
    if args.command == "formal-branch":
        return run_formal_branch(args)
    if args.command == "convert-parquet":
        return run_convert_parquet(args)
    raise FormalStop(f"STOP_UNKNOWN_COMMAND:{args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if isinstance(exc, SystemExit):
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
