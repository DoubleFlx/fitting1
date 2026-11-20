from __future__ import annotations

"""Strict execution-only Route-A photometric transform.

This module deliberately contains no policy, descriptor, solver, or training
logic.  It loads the protected realization snapshot by SHA256 and reuses only
its exact palette preparation, three-degree-of-freedom continuous operator,
and formal BGR-array hash helper.
"""

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FROZEN_REALIZATION_AUDIT = ROOT / "project" / "audits" / (
    "M1_ROUTE_A_PHOTOMETRIC_REALIZATION_FEASIBILITY_MEASUREMENT_V1_20260809_131730"
)
FROZEN_REALIZATION_CORE = (
    FROZEN_REALIZATION_AUDIT
    / "CODE_SNAPSHOT"
    / "m1_routea_photometric_realization_core_v1.py"
)
FROZEN_Q2_SNAPSHOT = (
    ROOT
    / "project"
    / "audits"
    / "M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520"
    / "CODE_SNAPSHOT"
)
EXPECTED_FROZEN_REALIZATION_CORE_SHA256 = (
    "45b62bf0a192e2d527f3171661d36b19d2dd7da21c63afb93140015251b449c1"
)
EXPECTED_ARRAY_HASH_PREFIX = b"M1_ROUTEA_REALIZED_BGR_UINT8_HWC_V1\0"
_FROZEN_MODULE_NAME = "m1_routea_resource_equivalence_frozen_realization_core_v1"


class RouteAExecutionCoreStop(RuntimeError):
    """Fail-closed execution-core integrity or representation error."""


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


_FROZEN_MODULE: Any | None = None
_FROZEN_MODULE_LOCK = threading.Lock()


def load_frozen_realization_core() -> Any:
    """Load and configure the protected realization snapshot exactly once."""

    global _FROZEN_MODULE
    if _FROZEN_MODULE is not None:
        return _FROZEN_MODULE
    with _FROZEN_MODULE_LOCK:
        if _FROZEN_MODULE is not None:
            return _FROZEN_MODULE
        if not FROZEN_REALIZATION_CORE.is_file():
            raise RouteAExecutionCoreStop("STOP_FROZEN_REALIZATION_CORE_MISSING")
        actual_sha256 = sha256_file(FROZEN_REALIZATION_CORE)
        if actual_sha256 != EXPECTED_FROZEN_REALIZATION_CORE_SHA256:
            raise RouteAExecutionCoreStop(
                "STOP_FROZEN_REALIZATION_CORE_SHA256_MISMATCH:"
                f"{actual_sha256}"
            )
        if not FROZEN_Q2_SNAPSHOT.is_dir():
            raise RouteAExecutionCoreStop("STOP_FROZEN_Q2_SNAPSHOT_MISSING")

        spec = importlib.util.spec_from_file_location(
            _FROZEN_MODULE_NAME, FROZEN_REALIZATION_CORE
        )
        if spec is None or spec.loader is None:
            raise RouteAExecutionCoreStop("STOP_FROZEN_REALIZATION_IMPORT_UNAVAILABLE")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_FROZEN_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
            module.configure_q2_snapshot(FROZEN_Q2_SNAPSHOT)
        except Exception:
            sys.modules.pop(_FROZEN_MODULE_NAME, None)
            raise

        if not callable(getattr(module, "_array_hash_from_palette", None)):
            raise RouteAExecutionCoreStop("STOP_FROZEN_ARRAY_HASH_HELPER_MISSING")
        if getattr(module, "ARRAY_HASH_PREFIX", None) != EXPECTED_ARRAY_HASH_PREFIX:
            raise RouteAExecutionCoreStop("STOP_FROZEN_ARRAY_HASH_PREFIX_MISMATCH")
        if not callable(getattr(module.PaletteImage, "continuous_output", None)):
            raise RouteAExecutionCoreStop("STOP_FROZEN_CONTINUOUS_OPERATOR_MISSING")
        _FROZEN_MODULE = module
        return module


def apply_theta_training_rgb_uint8(
    rgb_uint8: np.ndarray,
    theta: Sequence[float],
) -> tuple[np.ndarray, str]:
    """Apply exact frozen theta and return contiguous RGB uint8 plus BGR SHA.

    The returned SHA is the formal deployment hash over the reconstructed BGR
    uint8 HWC array.  This hot path intentionally performs no descriptor,
    diagnostic, policy, or solver work.
    """

    image = np.asarray(rgb_uint8)
    if image.dtype != np.uint8:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_INPUT_DTYPE_MISMATCH")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_INPUT_SHAPE_MISMATCH")
    if not image.flags.c_contiguous:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_INPUT_NOT_C_CONTIGUOUS")

    frozen = load_frozen_realization_core()
    theta_array = np.asarray(theta, dtype=np.float64)
    if theta_array.shape != (3,) or not np.isfinite(theta_array).all():
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_THETA_INVALID")
    lower = np.asarray(frozen.PARAMETER_LOWER, dtype=np.float64)
    upper = np.asarray(frozen.PARAMETER_UPPER, dtype=np.float64)
    if np.any(theta_array < lower) or np.any(theta_array > upper):
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_THETA_BOUND_VIOLATION")

    palette = frozen.PaletteImage.from_rgb_uint8(image)
    continuous_rgb, _, _, _ = palette.continuous_output(theta_array)
    if (
        continuous_rgb.dtype != np.float64
        or continuous_rgb.shape != (len(palette.palette_uint8), 3)
        or not continuous_rgb.flags.c_contiguous
        or not np.isfinite(continuous_rgb).all()
    ):
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_CONTINUOUS_OUTPUT_INVALID")

    deployment_palette = np.rint(continuous_rgb * 255.0).astype(np.uint8)
    if (
        deployment_palette.dtype != np.uint8
        or deployment_palette.shape != continuous_rgb.shape
        or not deployment_palette.flags.c_contiguous
    ):
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_QUANTIZED_PALETTE_INVALID")

    output_bgr_array_sha256 = frozen._array_hash_from_palette(
        deployment_palette,
        palette.inverse_flat,
        palette.height,
        palette.width,
    )
    if (
        not isinstance(output_bgr_array_sha256, str)
        or len(output_bgr_array_sha256) != 64
        or any(character not in "0123456789abcdef" for character in output_bgr_array_sha256)
    ):
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_OUTPUT_HASH_INVALID")

    full_rgb = deployment_palette[palette.inverse_flat].reshape(
        palette.height, palette.width, 3
    )
    full_rgb = np.ascontiguousarray(full_rgb)
    if full_rgb.dtype != np.uint8:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_OUTPUT_DTYPE_MISMATCH")
    if full_rgb.shape != image.shape:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_OUTPUT_SHAPE_MISMATCH")
    if not full_rgb.flags.c_contiguous:
        raise RouteAExecutionCoreStop("STOP_ROUTEA_FAST_OUTPUT_NOT_C_CONTIGUOUS")
    return full_rgb, output_bgr_array_sha256


__all__ = [
    "EXPECTED_FROZEN_REALIZATION_CORE_SHA256",
    "FROZEN_REALIZATION_CORE",
    "RouteAExecutionCoreStop",
    "apply_theta_training_rgb_uint8",
    "load_frozen_realization_core",
    "sha256_file",
]
