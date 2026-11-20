from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import scale_boxes, xywh2xyxy

from q2_frozen import (
    B2_CHECKPOINT,
    B2_CHECKPOINT_SHA256,
    B2_EMA_CANONICAL_SHA256,
    B2_RAW_CANONICAL_SHA256,
    B2_RUN_CONTRACT,
    B2_RUN_CONTRACT_SHA256,
    FROZEN_B2_CODE,
    HPL,
    PROJECT_SCRIPTS,
    ROOT,
)
from q2_rng import matched_random_translation


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_readonly(image_path: str | Path) -> tuple[np.ndarray, str]:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image is missing: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Image is empty: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"OpenCV returned invalid image shape {image.shape}: {path}")
        return np.ascontiguousarray(image), "cv2"

    raw = path.read_bytes()
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(io.BytesIO(raw)) as pil_image:
            pil_image.load()
            rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RuntimeError(
            f"Both OpenCV and Pillow failed to decode image {path}; "
            f"size={len(raw)}; pillow={type(exc).__name__}: {exc}"
        ) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"Pillow returned invalid image shape {rgb.shape}: {path}")
    return np.ascontiguousarray(rgb[:, :, ::-1]), "pillow_truncated_readonly"


def rectangle_mask(height: int, width: int, boxes_xyxy: np.ndarray) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    for x1, y1, x2, y2 in boxes:
        ix1 = max(0, min(width, int(np.floor(x1))))
        iy1 = max(0, min(height, int(np.floor(y1))))
        ix2 = max(0, min(width, int(np.ceil(x2))))
        iy2 = max(0, min(height, int(np.ceil(y2))))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return mask


def matched_random_mask(uid: str, hpl_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = hpl_mask.shape
    dx, dy, redraw_attempt = matched_random_translation(uid, width, height)
    random_mask = np.roll(hpl_mask, shift=(dy, dx), axis=(0, 1))
    source_area = int(hpl_mask.sum())
    random_area = int(random_mask.sum())
    if source_area != random_area:
        raise RuntimeError("Toroidal mask translation did not preserve binary area")
    if dx == 0 and dy == 0:
        raise RuntimeError("Forbidden zero random-mask translation")
    return random_mask, {
        "dx": dx,
        "dy": dy,
        "redraw_attempt": redraw_attempt,
        "hpl_binary_area": source_area,
        "random_binary_area": random_area,
        "area_preserved": True,
        "zero_translation_excluded": True,
    }


class FrozenB2HPLDetector:
    def __init__(self) -> None:
        if sha256_file(B2_CHECKPOINT) != B2_CHECKPOINT_SHA256:
            raise RuntimeError("B2 e100 checkpoint file SHA mismatch")
        if sha256_file(B2_RUN_CONTRACT) != B2_RUN_CONTRACT_SHA256:
            raise RuntimeError("B2 run contract SHA mismatch")
        for name, (path, expected) in FROZEN_B2_CODE.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"Frozen B2 code SHA mismatch: {name}")
        if not torch.cuda.is_available():
            raise RuntimeError("STOP_HPL_DEVICE_UNAVAILABLE")

        if str(PROJECT_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(PROJECT_SCRIPTS))
        from protocol_b_b2_model import ProtocolBTSDetectionModel
        from protocol_b_full_state import tensor_state_hash
        from protocol_b_pseudo import generate_high_confidence_pseudo

        state = torch.load(B2_CHECKPOINT, map_location="cpu", weights_only=False)
        raw_hash = tensor_state_hash(state["model_state_dict"])
        ema_hash = tensor_state_hash(state["ema_state_dict"])
        if raw_hash != B2_RAW_CANONICAL_SHA256 or ema_hash != B2_EMA_CANONICAL_SHA256:
            raise RuntimeError("B2 canonical tensor SHA mismatch")
        model = ProtocolBTSDetectionModel(
            "yolo11n.yaml",
            nc=HPL["single_class_nc"],
            ch=3,
            verbose=False,
        )
        incompatible = model.load_state_dict(state["ema_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict EMA load mismatch: {incompatible}")
        self.device = torch.device("cuda:0")
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("HPL model parameters unexpectedly require gradients")
        self.tensor_state_hash = tensor_state_hash
        self.generate_high_confidence_pseudo = generate_high_confidence_pseudo
        self.initial_state_sha256 = tensor_state_hash(self.model.state_dict())
        if self.initial_state_sha256 != B2_EMA_CANONICAL_SHA256:
            raise RuntimeError("Device-loaded EMA state SHA mismatch")
        self.letterbox = LetterBox(
            new_shape=(HPL["imgsz"], HPL["imgsz"]),
            auto=False,
            scale_fill=False,
            scaleup=True,
            center=True,
            stride=32,
            padding_value=114,
            interpolation=cv2.INTER_LINEAR,
        )
        self.inference_batches = 0
        self.inference_images = 0

    def _letterbox_tensor(self, bgr_images: list[np.ndarray]) -> torch.Tensor:
        formatted = []
        for image in bgr_images:
            labels = {"img": image}
            params = self.letterbox.get_params(labels)
            letterboxed = self.letterbox.apply_image(labels, params)["img"]
            chw_rgb = np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1))
            formatted.append(chw_rgb)
        uint8 = torch.from_numpy(np.stack(formatted, axis=0))
        return uint8.to(self.device, non_blocking=False).float() / 255.0

    def predict_batch(self, bgr_images: list[np.ndarray]) -> tuple[list[np.ndarray], dict[str, Any]]:
        if not bgr_images:
            return [], {"final_per_image": []}
        target_images = self._letterbox_tensor(bgr_images)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
            output = self.model(target_images)
        pseudo = self.generate_high_confidence_pseudo(
            output,
            conf=HPL["conf"],
            iou=HPL["iou"],
            max_det=HPL["max_det"],
            imgsz=HPL["imgsz"],
            area_min=HPL["area_min_strict"],
            aspect_min=HPL["aspect_min_inclusive"],
            aspect_max=HPL["aspect_max_inclusive"],
        )
        boxes_per_image: list[np.ndarray] = []
        for image_index, image in enumerate(bgr_images):
            select = pseudo.batch_idx == image_index
            normalized_xywh = pseudo.bboxes[select]
            if len(normalized_xywh) == 0:
                boxes_per_image.append(np.zeros((0, 4), dtype=np.float64))
                continue
            boxes_letterbox = xywh2xyxy(normalized_xywh * float(HPL["imgsz"]))
            height, width = image.shape[:2]
            boxes_original = scale_boxes(
                (HPL["imgsz"], HPL["imgsz"]),
                boxes_letterbox.clone(),
                (height, width),
                padding=True,
            )
            boxes_per_image.append(boxes_original.detach().cpu().numpy().astype(np.float64))
        self.inference_batches += 1
        self.inference_images += len(bgr_images)
        del target_images, output
        return boxes_per_image, pseudo.stats

    def finalize_audit(self) -> dict[str, Any]:
        final_hash = self.tensor_state_hash(self.model.state_dict())
        if final_hash != self.initial_state_sha256:
            raise RuntimeError("HPL model state changed during read-only inference")
        ultralytics_root = ROOT / "project" / "ultralytics"
        package_files = {
            "nms": ultralytics_root / "utils" / "nms.py",
            "letterbox": ultralytics_root / "data" / "augment.py",
            "geometry": ultralytics_root / "utils" / "ops.py",
        }
        return {
            "status": "PASS",
            "checkpoint_path": str(B2_CHECKPOINT),
            "checkpoint_file_sha256": B2_CHECKPOINT_SHA256,
            "ema_canonical_tensor_sha256": B2_EMA_CANONICAL_SHA256,
            "file_sha_and_ema_canonical_sha_are_distinct": True,
            "model_state_sha256_before": self.initial_state_sha256,
            "model_state_sha256_after": final_hash,
            "model_parameter_updates": 0,
            "parameter_requires_grad_count": sum(int(p.requires_grad) for p in self.model.parameters()),
            "inference_batches": self.inference_batches,
            "inference_images": self.inference_images,
            "ultralytics_version": __import__("ultralytics").__version__,
            "hpl_parameters": HPL,
            "frozen_project_code": {
                name: {"path": str(path), "sha256": expected}
                for name, (path, expected) in FROZEN_B2_CODE.items()
            },
            "package_implementation_files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in package_files.items()
            },
            "source_and_target_identical_rule": True,
            "source_gt_used": False,
            "masked_pixels_excluded_not_filled": True,
        }
