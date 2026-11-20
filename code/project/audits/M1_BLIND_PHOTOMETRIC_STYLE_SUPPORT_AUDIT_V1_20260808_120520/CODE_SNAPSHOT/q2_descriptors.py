from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from q2_frozen import (
    AXES,
    BASE_DESCRIPTORS,
    D65_REFERENCE_WHITE,
    LAB_REFERENCE_TOLERANCE_LAB,
    LAB_REFERENCE_TOLERANCE_XYZ,
    LAB_REFERENCE_VECTORS,
    LUMA_COEFFICIENTS,
    LUMA_LOG_EPSILON,
    SRGB_TO_XYZ_D65,
)


def inverse_ecdf_quantiles(values: np.ndarray, probabilities: Iterable[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Quantile input must be nonempty and finite")
    probabilities = list(probabilities)
    indices = []
    for probability in probabilities:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid probability: {probability}")
        index = max(0, int(math.ceil(probability * array.size)) - 1)
        indices.append(index)
    partitioned = np.partition(array, np.asarray(sorted(set(indices)), dtype=np.int64))
    return [float(partitioned[index]) for index in indices]


def srgb_inverse_companding(srgb: np.ndarray) -> np.ndarray:
    array = np.asarray(srgb, dtype=np.float64)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("sRGB values must be in [0,1]")
    return np.where(
        array <= 0.04045,
        array / 12.92,
        ((array + 0.055) / 1.055) ** 2.4,
    )


def srgb_to_xyz_lab(srgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(srgb, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB last dimension of 3, got {array.shape}")
    linear = srgb_inverse_companding(array)
    matrix = np.asarray(SRGB_TO_XYZ_D65, dtype=np.float64)
    xyz = linear @ matrix.T
    white = np.asarray(D65_REFERENCE_WHITE, dtype=np.float64)
    ratio = xyz / white
    delta = 6.0 / 29.0
    threshold = delta**3
    f = np.where(
        ratio > threshold,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(xyz, dtype=np.float64)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return xyz, lab


def pixel_planes(rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 HxWx3 RGB image, got {rgb.dtype} {rgb.shape}")
    srgb = rgb.astype(np.float64) / 255.0
    coefficients = np.asarray(LUMA_COEFFICIENTS, dtype=np.float64)
    y_prime = srgb @ coefficients
    _, lab = srgb_to_xyz_lab(srgb)
    cstar = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return y_prime, cstar


def _local_contrast(
    y_prime: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float | None, int]:
    height, width = y_prime.shape
    contrasts: list[float] = []
    valid_cells = 0
    for row in range(8):
        y0 = (row * height) // 8
        y1 = ((row + 1) * height) // 8
        for column in range(8):
            x0 = (column * width) // 8
            x1 = ((column + 1) * width) // 8
            total = (y1 - y0) * (x1 - x0)
            if total <= 0:
                continue
            cell_valid = valid_mask[y0:y1, x0:x1]
            remaining = int(cell_valid.sum())
            if remaining <= 0 or remaining * 2 < total:
                continue
            q10, q90 = inverse_ecdf_quantiles(
                y_prime[y0:y1, x0:x1][cell_valid],
                [0.10, 0.90],
            )
            valid_cells += 1
            contrasts.append(q90 - q10)
    if valid_cells < 32:
        return None, valid_cells
    return inverse_ecdf_quantiles(np.asarray(contrasts), [0.50])[0], valid_cells


def compute_descriptor_state(
    y_prime: np.ndarray,
    cstar: np.ndarray,
    excluded_mask: np.ndarray | None,
) -> dict:
    if y_prime.shape != cstar.shape:
        raise ValueError("Y and C* planes must share shape")
    if excluded_mask is None:
        valid = np.ones(y_prime.shape, dtype=bool)
    else:
        excluded = np.asarray(excluded_mask, dtype=bool)
        if excluded.shape != y_prime.shape:
            raise ValueError("Excluded mask shape mismatch")
        valid = ~excluded
    total_pixels = int(valid.size)
    remaining_pixels = int(valid.sum())
    global_available = remaining_pixels > 0 and remaining_pixels * 2 >= total_pixels

    values = {name: math.nan for name in BASE_DESCRIPTORS}
    local_value: float | None = None
    valid_cells = 0
    if global_available:
        y_valid = y_prime[valid]
        log_y = np.log(y_valid + LUMA_LOG_EPSILON)
        tone_q10, tone_q50, tone_q90 = inverse_ecdf_quantiles(log_y, [0.10, 0.50, 0.90])
        y_q10, y_q90 = inverse_ecdf_quantiles(y_valid, [0.10, 0.90])
        chroma_q50, chroma_q90 = inverse_ecdf_quantiles(cstar[valid], [0.50, 0.90])
        local_value, valid_cells = _local_contrast(y_prime, valid)
        values.update(
            {
                "tone_ln_y_q10": tone_q10,
                "tone_ln_y_q50": tone_q50,
                "tone_ln_y_q90": tone_q90,
                "contrast_global_q90_minus_q10": y_q90 - y_q10,
                "chroma_cstar_q50": chroma_q50,
                "chroma_cstar_q90": chroma_q90,
            }
        )
        if local_value is not None:
            values["contrast_local_grid8_median_q90_minus_q10"] = local_value

    descriptor_available = {
        name: bool(np.isfinite(value)) for name, value in values.items()
    }
    axis_available = {
        axis: all(descriptor_available[name] for name in descriptors)
        for axis, descriptors in AXES.items()
    }
    return {
        "values": values,
        "descriptor_available": descriptor_available,
        "axis_available": axis_available,
        "total_pixels": total_pixels,
        "remaining_pixels": remaining_pixels,
        "remaining_fraction": remaining_pixels / total_pixels if total_pixels else 0.0,
        "local_valid_cells": valid_cells,
        "global_available": global_available,
        "local_available": local_value is not None,
    }


def descriptor_unit_tests() -> dict:
    luma_cases = [
        ("black", [0.0, 0.0, 0.0], 0.0),
        ("white", [1.0, 1.0, 1.0], 1.0),
        ("red", [1.0, 0.0, 0.0], 0.2126),
        ("green", [0.0, 1.0, 0.0], 0.7152),
        ("blue", [0.0, 0.0, 1.0], 0.0722),
    ]
    luma_results = []
    luma_pass = True
    coefficients = np.asarray(LUMA_COEFFICIENTS, dtype=np.float64)
    for name, rgb, expected in luma_cases:
        actual = float(np.asarray(rgb, dtype=np.float64) @ coefficients)
        passed = abs(actual - expected) <= 1e-12
        luma_pass &= passed
        luma_results.append(
            {"name": name, "rgb": rgb, "expected_y_prime": expected, "actual_y_prime": actual, "pass": passed}
        )

    lab_results = []
    lab_pass = True
    for vector in LAB_REFERENCE_VECTORS:
        rgb = np.asarray(vector["rgb"], dtype=np.float64)
        xyz, lab = srgb_to_xyz_lab(rgb)
        expected_xyz = np.asarray(vector["xyz"], dtype=np.float64)
        expected_lab = np.asarray(vector["lab"], dtype=np.float64)
        xyz_error = float(np.max(np.abs(xyz - expected_xyz)))
        lab_error = float(np.max(np.abs(lab - expected_lab)))
        # The frozen tolerance remains unchanged.  Add one float64 machine
        # epsilon only to the comparison so a subtraction such as
        # 1.0000001 - 1.0 is not rejected solely because its representable
        # result lies infinitesimally above the decimal tolerance.
        comparison_guard = float(np.finfo(np.float64).eps)
        passed = (
            xyz_error <= LAB_REFERENCE_TOLERANCE_XYZ + comparison_guard
            and lab_error <= LAB_REFERENCE_TOLERANCE_LAB + comparison_guard
        )
        lab_pass &= passed
        lab_results.append(
            {
                "name": vector["name"],
                "rgb": vector["rgb"],
                "expected_xyz": vector["xyz"],
                "actual_xyz": xyz.tolist(),
                "expected_lab": vector["lab"],
                "actual_lab": lab.tolist(),
                "max_abs_xyz_error": xyz_error,
                "max_abs_lab_error": lab_error,
                "float64_comparison_guard": comparison_guard,
                "pass": passed,
            }
        )

    synthetic = np.zeros((8, 8, 3), dtype=np.uint8)
    synthetic[:, 4:, :] = 255
    y_prime, cstar = pixel_planes(synthetic)
    whole = compute_descriptor_state(y_prime, cstar, None)
    half_mask = np.zeros((8, 8), dtype=bool)
    half_mask[:, :4] = True
    masked = compute_descriptor_state(y_prime, cstar, half_mask)
    availability_pass = (
        whole["axis_available"]["TONE"]
        and whole["axis_available"]["CONTRAST"]
        and whole["axis_available"]["CHROMA"]
        and masked["remaining_fraction"] == 0.5
    )
    all_pass = bool(luma_pass and lab_pass and availability_pass)
    return {
        "all_pass": all_pass,
        "luma_known_values": luma_results,
        "cielab_d65_reference_vectors": lab_results,
        "mask_exclusion_availability_test": {
            "remaining_fraction": masked["remaining_fraction"],
            "pass": availability_pass,
        },
        "pixel_quantile_algorithm": "inverse empirical CDF; no interpolation",
    }
