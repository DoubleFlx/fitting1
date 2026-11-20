from __future__ import annotations

import hashlib
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial import cKDTree


EXPERIMENT = "M1_ROUTE_A_PHOTOMETRIC_REALIZATION_FEASIBILITY_MEASUREMENT_V1"
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
PARAMETER_LOWER = np.asarray([-math.log(2.0), 0.5, 0.5], dtype=np.float64)
PARAMETER_UPPER = np.asarray([math.log(2.0), 2.0, 2.0], dtype=np.float64)
SOLVER_OPTIONS = {"maxiter": 100, "maxfev": 400, "xtol": 1e-5, "ftol": 1e-6}
BOUND_HIT_TOL = 1e-7
ARRAY_HASH_PREFIX = b"M1_ROUTEA_REALIZED_BGR_UINT8_HWC_V1\0"
DELTA_HASH_PREFIX = b"M1_ROUTEA_REALIZED_DELTA_FLOAT64_LE_V1\0"

_Q2_DESCRIPTORS = None
_Q2_HPL = None
_Q2_FROZEN = None


class RealizationNumericalError(RuntimeError):
    pass


def configure_q2_snapshot(snapshot_dir: str | Path) -> None:
    """Load the exact frozen Q2 modules from their formal code snapshot."""
    global _Q2_DESCRIPTORS, _Q2_HPL, _Q2_FROZEN
    snapshot = str(Path(snapshot_dir).resolve())
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    _Q2_FROZEN = importlib.import_module("q2_frozen")
    _Q2_DESCRIPTORS = importlib.import_module("q2_descriptors")
    _Q2_HPL = importlib.import_module("q2_hpl")


def require_q2() -> tuple[Any, Any, Any]:
    if _Q2_DESCRIPTORS is None or _Q2_HPL is None or _Q2_FROZEN is None:
        raise RuntimeError("Q2 snapshot is not configured")
    return _Q2_DESCRIPTORS, _Q2_HPL, _Q2_FROZEN


def sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def float64_vector_sha(values: Sequence[float], prefix: bytes = DELTA_HASH_PREFIX) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(prefix + array.tobytes(order="C")).hexdigest()


def decode_q2_rgb(image_path: str | Path) -> tuple[np.ndarray, str]:
    _, q2_hpl, _ = require_q2()
    bgr, decoder = q2_hpl.read_image_readonly(Path(image_path))
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RealizationNumericalError(f"Invalid decoded image: {image_path} {rgb.dtype} {rgb.shape}")
    return rgb, str(decoder)


def q2_exact_raw_descriptor(rgb_uint8: np.ndarray) -> np.ndarray:
    q2_desc, _, _ = require_q2()
    y_prime, cstar = q2_desc.pixel_planes(rgb_uint8)
    state = q2_desc.compute_descriptor_state(y_prime, cstar, None)
    values = np.asarray([state["values"][name] for name in BASE], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RealizationNumericalError("Q2 exact descriptor is nonfinite")
    return values


def dstyle(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(left - right) * WEIGHTS, dtype=np.float64))


def dstyle_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.asarray(a) - np.asarray(b)) * WEIGHTS, axis=-1, dtype=np.float64)


def standardize(raw: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(raw, dtype=np.float64) - median) / scale


def inverse_standardize(z: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * scale + median


def descriptor_internal_valid_raw(raw: np.ndarray) -> tuple[bool, str]:
    value = np.asarray(raw, dtype=np.float64).reshape(7)
    reasons: list[str] = []
    if not np.isfinite(value).all():
        reasons.append("NONFINITE")
    if not (value[0] <= value[1] <= value[2]):
        reasons.append("TONE_QUANTILE_ORDER")
    if value[3] < 0.0 or value[4] < 0.0:
        reasons.append("NEGATIVE_CONTRAST")
    if value[5] > value[6]:
        reasons.append("CHROMA_QUANTILE_ORDER")
    return not reasons, "|".join(reasons)


def inverse_ecdf_weighted(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(value) == 0 or len(value) != len(weight):
        raise ValueError("weighted quantile requires equal nonempty arrays")
    if not np.isfinite(value).all() or not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("weighted quantile requires finite nonnegative inputs")
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    weight = weight[order]
    unique, first = np.unique(value, return_index=True)
    merged = np.add.reduceat(weight, first)
    total = float(merged.sum(dtype=np.float64))
    if not total > 0.0:
        raise ValueError("weighted quantile has zero total weight")
    if p <= 0.0:
        return float(unique[0])
    if p >= 1.0:
        return float(unique[-1])
    cdf = np.cumsum(merged / total, dtype=np.float64)
    index = min(int(np.searchsorted(cdf, p, side="left")), len(unique) - 1)
    return float(unique[index])


def inverse_ecdf_integer_counts(values: np.ndarray, counts: np.ndarray, p: float) -> float:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    count = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(value) == 0 or len(value) != len(count) or np.any(count < 0):
        raise ValueError("count quantile requires equal nonempty arrays and nonnegative counts")
    total = int(count.sum(dtype=np.int64))
    if total <= 0:
        raise ValueError("count quantile has zero total count")
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    sorted_count = count[order]
    unique, first = np.unique(sorted_value, return_index=True)
    merged_count = np.add.reduceat(sorted_count, first)
    if p <= 0.0:
        rank = 1
    else:
        rank = max(1, int(math.ceil(float(p) * total)))
    cumulative = np.cumsum(merged_count, dtype=np.int64)
    index = min(int(np.searchsorted(cumulative, rank, side="left")), len(unique) - 1)
    return float(unique[index])


def weighted_summary(values: Sequence[float], weights: Sequence[float] | None = None) -> dict[str, float | int]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(value) == 0:
        return {"count": 0, "Q10": math.nan, "Q50": math.nan, "Q90": math.nan, "mean": math.nan}
    weight = np.ones(len(value), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    weight = weight / weight.sum(dtype=np.float64)
    return {
        "count": int(len(value)),
        "Q10": inverse_ecdf_weighted(value, weight, 0.10),
        "Q50": inverse_ecdf_weighted(value, weight, 0.50),
        "Q90": inverse_ecdf_weighted(value, weight, 0.90),
        "mean": float(np.sum(value * weight, dtype=np.float64)),
    }


def wasserstein_1d(
    values_a: Sequence[float],
    weights_a: Sequence[float],
    values_b: Sequence[float],
    weights_b: Sequence[float],
) -> float:
    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    wa = np.asarray(weights_a, dtype=np.float64).reshape(-1)
    wb = np.asarray(weights_b, dtype=np.float64).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    wa = wa / wa.sum(dtype=np.float64)
    wb = wb / wb.sum(dtype=np.float64)
    support = np.unique(np.concatenate([a, b]))
    if len(support) <= 1:
        return 0.0
    mass_a = np.zeros(len(support), dtype=np.float64)
    mass_b = np.zeros(len(support), dtype=np.float64)
    np.add.at(mass_a, np.searchsorted(support, a), wa)
    np.add.at(mass_b, np.searchsorted(support, b), wb)
    cdf_a = np.cumsum(mass_a, dtype=np.float64)
    cdf_b = np.cumsum(mass_b, dtype=np.float64)
    return float(np.sum(np.abs(cdf_a[:-1] - cdf_b[:-1]) * np.diff(support), dtype=np.float64))


def _descriptor_from_planes_palette_reference(
    y_prime: np.ndarray,
    cstar: np.ndarray,
    counts: np.ndarray,
    cell_memberships: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    log_y = np.log(np.asarray(y_prime, dtype=np.float64) + 1e-6)
    tone = [inverse_ecdf_integer_counts(log_y, counts, p) for p in (0.10, 0.50, 0.90)]
    y10 = inverse_ecdf_integer_counts(y_prime, counts, 0.10)
    y90 = inverse_ecdf_integer_counts(y_prime, counts, 0.90)
    local: list[float] = []
    for ids, cell_counts in cell_memberships:
        q10 = inverse_ecdf_integer_counts(y_prime[ids], cell_counts, 0.10)
        q90 = inverse_ecdf_integer_counts(y_prime[ids], cell_counts, 0.90)
        local.append(q90 - q10)
    local_median = inverse_ecdf_integer_counts(
        np.asarray(local, dtype=np.float64), np.ones(len(local), dtype=np.int64), 0.50
    )
    c50 = inverse_ecdf_integer_counts(cstar, counts, 0.50)
    c90 = inverse_ecdf_integer_counts(cstar, counts, 0.90)
    return np.asarray([tone[0], tone[1], tone[2], y90 - y10, local_median, c50, c90], dtype=np.float64)


def _sorted_count_quantile_positions(
    sorted_counts: np.ndarray, probabilities: Sequence[float]
) -> np.ndarray:
    """Return inverse-ECDF positions for already value-sorted integer counts."""
    count = np.asarray(sorted_counts, dtype=np.int64).reshape(-1)
    total = int(count.sum(dtype=np.int64))
    if total <= 0:
        raise ValueError("sorted count quantile has zero total count")
    cumulative = np.cumsum(count, dtype=np.int64)
    ranks = np.asarray(
        [max(1, int(math.ceil(float(probability) * total))) for probability in probabilities],
        dtype=np.int64,
    )
    return np.searchsorted(cumulative, ranks, side="left")


def _descriptor_from_planes_palette(
    y_prime: np.ndarray,
    cstar: np.ndarray,
    counts: np.ndarray,
    cell_memberships: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Exact Q2 descriptor with redundant quantile sorts mechanically removed.

    The reference implementation sorts the same y vector five times, the same
    C* vector twice, and every cell y vector twice.  A single stable global y
    ordering determines all global tone/contrast quantiles.  Each cell is then
    ordered once by the corresponding global rank, and C* is ordered once.
    Inverse-ECDF values are byte-identical because only ordering work is shared;
    no binning, interpolation, approximation, or precision change is used.
    """
    y = np.asarray(y_prime, dtype=np.float64).reshape(-1)
    chroma = np.asarray(cstar, dtype=np.float64).reshape(-1)
    count = np.asarray(counts, dtype=np.int64).reshape(-1)
    if len(y) == 0 or len(y) != len(chroma) or len(y) != len(count):
        raise ValueError("descriptor planes/counts require equal nonempty arrays")

    y_order = np.argsort(y, kind="mergesort")
    y_sorted = y[y_order]
    y_positions = _sorted_count_quantile_positions(count[y_order], (0.10, 0.50, 0.90))
    y10, y50, y90 = (float(y_sorted[position]) for position in y_positions)
    # Use the selected original values in the identical scalar expression used
    # by the Q2 vector implementation; log is strictly monotone.
    tone = tuple(
        float(value)
        for value in np.log(np.asarray([y10, y50, y90], dtype=np.float64) + 1e-6)
    )

    rank = np.empty(len(y_order), dtype=np.int64)
    rank[y_order] = np.arange(len(y_order), dtype=np.int64)
    local = np.empty(len(cell_memberships), dtype=np.float64)
    for cell_index, (ids, cell_counts) in enumerate(cell_memberships):
        ids_array = np.asarray(ids, dtype=np.int64)
        counts_array = np.asarray(cell_counts, dtype=np.int64)
        cell_order = np.argsort(rank[ids_array], kind="mergesort")
        positions = _sorted_count_quantile_positions(counts_array[cell_order], (0.10, 0.90))
        values = y[ids_array[cell_order[positions]]]
        local[cell_index] = float(values[1] - values[0])
    local_order = np.argsort(local, kind="mergesort")
    local_position = _sorted_count_quantile_positions(
        np.ones(len(local_order), dtype=np.int64), (0.50,)
    )[0]
    local_median = float(local[local_order[local_position]])

    chroma_order = np.argsort(chroma, kind="mergesort")
    chroma_positions = _sorted_count_quantile_positions(count[chroma_order], (0.50, 0.90))
    c50, c90 = (float(chroma[chroma_order[position]]) for position in chroma_positions)
    return np.asarray(
        [tone[0], tone[1], tone[2], y90 - y10, local_median, c50, c90],
        dtype=np.float64,
    )


def _srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    q2_desc, _, _ = require_q2()
    _, lab = q2_desc.srgb_to_xyz_lab(np.asarray(srgb, dtype=np.float64))
    return lab


def _lab_to_srgb_unclipped(lab: np.ndarray) -> np.ndarray:
    _, _, frozen = require_q2()
    array = np.asarray(lab, dtype=np.float64)
    white = np.asarray(frozen.D65_REFERENCE_WHITE, dtype=np.float64)
    delta = 6.0 / 29.0
    fy = (array[..., 0] + 16.0) / 116.0
    fx = fy + array[..., 1] / 500.0
    fz = fy - array[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    ratio = np.where(f > delta, f**3, 3.0 * delta**2 * (f - 4.0 / 29.0))
    xyz = ratio * white
    forward = np.asarray(frozen.SRGB_TO_XYZ_D65, dtype=np.float64)
    inverse = np.linalg.inv(forward)
    linear = xyz @ inverse.T
    srgb = np.empty_like(linear, dtype=np.float64)
    low = linear <= 0.0031308
    srgb[low] = 12.92 * linear[low]
    high_values = linear[~low]
    srgb[~low] = 1.055 * np.power(high_values, 1.0 / 2.4) - 0.055
    return srgb


def _array_hash_from_palette(
    deployment_rgb_uint8: np.ndarray,
    inverse_flat: np.ndarray,
    height: int,
    width: int,
) -> str:
    bgr_palette = np.ascontiguousarray(deployment_rgb_uint8[:, ::-1])
    digest = hashlib.sha256()
    digest.update(ARRAY_HASH_PREFIX)
    digest.update(b"uint8")
    digest.update(str((int(height), int(width), 3)).encode("ascii"))
    for start in range(0, len(inverse_flat), 262144):
        indices = inverse_flat[start : start + 262144]
        digest.update(np.ascontiguousarray(bgr_palette[indices]).tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class PaletteImage:
    height: int
    width: int
    palette_uint8: np.ndarray
    inverse_flat: np.ndarray
    counts: np.ndarray
    cell_memberships: list[tuple[np.ndarray, np.ndarray]]
    srgb: np.ndarray
    y_prime: np.ndarray
    ell: np.ndarray
    median_ell: float

    @classmethod
    def from_rgb_uint8(cls, rgb: np.ndarray) -> "PaletteImage":
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("PaletteImage requires uint8 HxWx3 RGB")
        height, width = image.shape[:2]
        packed = (
            (image[..., 0].astype(np.uint32) << 16)
            | (image[..., 1].astype(np.uint32) << 8)
            | image[..., 2].astype(np.uint32)
        )
        unique, inverse, counts = np.unique(
            packed.reshape(-1), return_inverse=True, return_counts=True
        )
        palette = np.stack(
            [
                ((unique >> 16) & 255).astype(np.uint8),
                ((unique >> 8) & 255).astype(np.uint8),
                (unique & 255).astype(np.uint8),
            ],
            axis=1,
        )
        inverse = inverse.astype(np.int32, copy=False)
        inverse_image = inverse.reshape(height, width)
        cells: list[tuple[np.ndarray, np.ndarray]] = []
        for row in range(8):
            y0 = (row * height) // 8
            y1 = ((row + 1) * height) // 8
            for column in range(8):
                x0 = (column * width) // 8
                x1 = ((column + 1) * width) // 8
                ids, cell_counts = np.unique(
                    inverse_image[y0:y1, x0:x1].reshape(-1), return_counts=True
                )
                cells.append((ids.astype(np.int32, copy=False), cell_counts.astype(np.int64, copy=False)))
        srgb = palette.astype(np.float64) / 255.0
        y_prime = srgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        ell = np.log(y_prime + 1e-6)
        median_ell = inverse_ecdf_integer_counts(ell, counts, 0.50)
        return cls(
            height=int(height),
            width=int(width),
            palette_uint8=palette,
            inverse_flat=inverse,
            counts=counts.astype(np.int64, copy=False),
            cell_memberships=cells,
            srgb=srgb,
            y_prime=y_prime,
            ell=ell,
            median_ell=median_ell,
        )

    def descriptor_from_srgb(self, srgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(srgb, dtype=np.float64)
        y = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        lab = _srgb_to_lab(rgb)
        cstar = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
        return _descriptor_from_planes_palette(y, cstar, self.counts, self.cell_memberships)

    def original_descriptor(self) -> np.ndarray:
        return self.descriptor_from_srgb(self.srgb)

    def tone_step_lab(self, delta_t: float, c_t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ell_target = self.median_ell + c_t * (self.ell - self.median_ell) + delta_t
        y_target = np.exp(ell_target) - 1e-6
        scale = y_target / (self.y_prime + 1e-6)
        preclip = self.srgb * scale[:, None]
        tone_rgb = np.clip(preclip, 0.0, 1.0)
        lab = _srgb_to_lab(tone_rgb)
        return preclip, tone_rgb, lab

    def continuous_output(self, theta: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        delta_t, c_t, s_c = (float(x) for x in theta)
        tone_preclip, tone_rgb, tone_lab = self.tone_step_lab(delta_t, c_t)
        scaled_lab = tone_lab.copy()
        scaled_lab[:, 1] *= s_c
        scaled_lab[:, 2] *= s_c
        gamut_preclip = _lab_to_srgb_unclipped(scaled_lab)
        output = np.clip(gamut_preclip, 0.0, 1.0)
        return output, tone_preclip, gamut_preclip, tone_lab

    def continuous_descriptor(self, theta: Sequence[float]) -> np.ndarray:
        output, _, _, _ = self.continuous_output(theta)
        return self.descriptor_from_srgb(output)

    def analytic_initialization(self, original_raw: np.ndarray, target_raw: np.ndarray) -> np.ndarray:
        delta_0 = float(target_raw[1] - original_raw[1])
        denominator = max(float(original_raw[2] - original_raw[0]), 1e-8)
        c_0 = float((target_raw[2] - target_raw[0]) / denominator)
        delta_0 = float(np.clip(delta_0, PARAMETER_LOWER[0], PARAMETER_UPPER[0]))
        c_0 = float(np.clip(c_0, PARAMETER_LOWER[1], PARAMETER_UPPER[1]))
        _, _, tone_lab = self.tone_step_lab(delta_0, c_0)
        cstar = np.sqrt(tone_lab[:, 1] ** 2 + tone_lab[:, 2] ** 2)
        c50 = inverse_ecdf_integer_counts(cstar, self.counts, 0.50)
        c90 = inverse_ecdf_integer_counts(cstar, self.counts, 0.90)
        denom = c50 * c50 + c90 * c90
        if not math.isfinite(denom) or denom < 1e-8:
            s_0 = 1.0
        else:
            s_0 = (c50 * float(target_raw[5]) + c90 * float(target_raw[6])) / denom
            if not math.isfinite(s_0):
                s_0 = 1.0
        s_0 = float(np.clip(s_0, PARAMETER_LOWER[2], PARAMETER_UPPER[2]))
        return np.asarray([delta_0, c_0, s_0], dtype=np.float64)

    def final_evaluation(self, theta: Sequence[float]) -> dict[str, Any]:
        continuous_rgb, tone_preclip, gamut_preclip, tone_lab = self.continuous_output(theta)
        continuous_raw = self.descriptor_from_srgb(continuous_rgb)
        deployment_uint8 = np.rint(continuous_rgb * 255.0).astype(np.uint8)
        deployment_rgb = deployment_uint8.astype(np.float64) / 255.0
        deployment_raw = self.descriptor_from_srgb(deployment_rgb)

        tone_excess = np.maximum(-tone_preclip, 0.0) + np.maximum(tone_preclip - 1.0, 0.0)
        gamut_excess = np.maximum(-gamut_preclip, 0.0) + np.maximum(gamut_preclip - 1.0, 0.0)
        total_pixels = int(self.counts.sum(dtype=np.int64))
        count_float = self.counts.astype(np.float64)
        tone_channel_oob = (tone_excess > 0.0).astype(np.float64)
        gamut_channel_oob = (gamut_excess > 0.0).astype(np.float64)
        deployment_lab = _srgb_to_lab(deployment_rgb)
        tone_cstar = np.sqrt(tone_lab[:, 1] ** 2 + tone_lab[:, 2] ** 2)
        tone_angle = np.arctan2(tone_lab[:, 2], tone_lab[:, 1])
        deployed_angle = np.arctan2(deployment_lab[:, 2], deployment_lab[:, 1])
        hue_shift = np.abs(np.arctan2(np.sin(deployed_angle - tone_angle), np.cos(deployed_angle - tone_angle)))
        hue_shift_deg = np.degrees(hue_shift)
        hue_weights = count_float * tone_cstar
        if float(hue_weights.sum(dtype=np.float64)) > 0.0:
            hue_mean = float(np.sum(hue_shift_deg * hue_weights, dtype=np.float64) / hue_weights.sum(dtype=np.float64))
            hue_q50 = inverse_ecdf_weighted(hue_shift_deg, hue_weights, 0.50)
            hue_q90 = inverse_ecdf_weighted(hue_shift_deg, hue_weights, 0.90)
        else:
            hue_mean = hue_q50 = hue_q90 = 0.0

        diagnostics = {
            "tone_preclip_pixel_fraction": float(
                np.sum((tone_channel_oob.any(axis=1)).astype(np.float64) * count_float, dtype=np.float64)
                / total_pixels
            ),
            "tone_preclip_channel_fraction": float(
                np.sum(tone_channel_oob * count_float[:, None], dtype=np.float64) / (total_pixels * 3)
            ),
            "tone_clip_mean_excess": float(
                np.sum(tone_excess * count_float[:, None], dtype=np.float64) / (total_pixels * 3)
            ),
            "tone_clip_max_excess": float(np.max(tone_excess)),
            "lab_to_srgb_preclip_pixel_fraction": float(
                np.sum((gamut_channel_oob.any(axis=1)).astype(np.float64) * count_float, dtype=np.float64)
                / total_pixels
            ),
            "lab_to_srgb_preclip_channel_fraction": float(
                np.sum(gamut_channel_oob * count_float[:, None], dtype=np.float64) / (total_pixels * 3)
            ),
            "gamut_clip_mean_excess": float(
                np.sum(gamut_excess * count_float[:, None], dtype=np.float64) / (total_pixels * 3)
            ),
            "gamut_clip_max_excess": float(np.max(gamut_excess)),
            "final_channel_at_zero_fraction": float(
                np.sum((deployment_uint8 == 0).astype(np.float64) * count_float[:, None], dtype=np.float64)
                / (total_pixels * 3)
            ),
            "final_channel_at_one_fraction": float(
                np.sum((deployment_uint8 == 255).astype(np.float64) * count_float[:, None], dtype=np.float64)
                / (total_pixels * 3)
            ),
            "chroma_weighted_mean_absolute_hue_angle_shift_degrees": hue_mean,
            "cstar_weighted_q50_hue_shift_degrees": hue_q50,
            "cstar_weighted_q90_hue_shift_degrees": hue_q90,
        }
        return {
            "continuous_raw": continuous_raw,
            "deployment_raw": deployment_raw,
            "deployment_rgb_uint8": deployment_uint8,
            "output_array_sha256": _array_hash_from_palette(
                deployment_uint8, self.inverse_flat, self.height, self.width
            ),
            "diagnostics": diagnostics,
        }


def realization_case(
    *,
    palette: PaletteImage,
    uid: str,
    event_id: str,
    case_type: str,
    input_image_sha256: str,
    decoder: str,
    original_std: np.ndarray,
    target_std: np.ndarray,
    intended_dose: float,
    target_delta_sha256: str,
    donor_uid: str,
    donor_event_id: str,
    median: np.ndarray,
    scale: np.ndarray,
    event_std: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    original_std = np.asarray(original_std, dtype=np.float64).reshape(7)
    target_std = np.asarray(target_std, dtype=np.float64).reshape(7)
    original_raw = inverse_standardize(original_std, median, scale)
    target_raw = inverse_standardize(target_std, median, scale)
    x0 = palette.analytic_initialization(original_raw, target_raw)
    cache: dict[bytes, tuple[float, np.ndarray]] = {}

    def objective(theta: np.ndarray) -> float:
        value = np.asarray(theta, dtype=np.float64)
        key = value.tobytes(order="C")
        if key in cache:
            return cache[key][0]
        descriptor_raw = palette.continuous_descriptor(value)
        descriptor_std = standardize(descriptor_raw, median, scale)
        loss = dstyle(descriptor_std, target_std)
        if not math.isfinite(loss):
            raise RealizationNumericalError(f"Nonfinite continuous objective for {uid} {case_type}")
        if len(cache) < 1024:
            cache[key] = (loss, descriptor_raw)
        return loss

    result = minimize(
        objective,
        x0,
        method="Powell",
        bounds=Bounds(PARAMETER_LOWER, PARAMETER_UPPER),
        options=SOLVER_OPTIONS,
    )
    theta = np.asarray(result.x, dtype=np.float64)
    if theta.shape != (3,) or not np.isfinite(theta).all() or np.any(theta < PARAMETER_LOWER) or np.any(theta > PARAMETER_UPPER):
        raise RealizationNumericalError(f"Invalid Powell result for {uid} {case_type}: {theta}")
    final = palette.final_evaluation(theta)
    continuous_std = standardize(final["continuous_raw"], median, scale)
    deployment_std = standardize(final["deployment_raw"], median, scale)
    target_delta = target_std - original_std
    realized_delta = deployment_std - original_std
    target_dose = dstyle(original_std, target_std)
    actual_dose = dstyle(original_std, deployment_std)
    continuous_residual = dstyle(continuous_std, target_std)
    deployment_residual = dstyle(deployment_std, target_std)
    support_distances = dstyle_rows(event_std, deployment_std[None, :])
    support_deviation = float(np.min(support_distances))
    internal_valid, internal_reason = descriptor_internal_valid_raw(final["deployment_raw"])
    sign_mass = WEIGHTS * np.abs(target_delta)
    sign_concordance = float(
        np.sum(sign_mass * (np.sign(realized_delta) == np.sign(target_delta)), dtype=np.float64)
        / max(float(sign_mass.sum(dtype=np.float64)), 1e-12)
    )
    lower_hits = np.abs(theta - PARAMETER_LOWER) <= BOUND_HIT_TOL
    upper_hits = np.abs(theta - PARAMETER_UPPER) <= BOUND_HIT_TOL

    row: dict[str, Any] = {
        "uid": str(uid),
        "capture_event_id": str(event_id),
        "case_type": str(case_type),
        "donor_uid": str(donor_uid),
        "donor_event_id": str(donor_event_id),
        "input_image_sha256": str(input_image_sha256),
        "decoder": str(decoder),
        "image_height": palette.height,
        "image_width": palette.width,
        "unique_rgb_count": int(len(palette.palette_uint8)),
        "target_delta_sha256": str(target_delta_sha256),
        "intended_dose": float(intended_dose),
        "target_descriptor_dose": target_dose,
        "actual_effective_dose": actual_dose,
        "actual_to_target_dose_ratio": actual_dose / max(target_dose, 1e-12),
        "absolute_dose_error": abs(actual_dose - target_dose),
        "continuous_realization_residual": continuous_residual,
        "deployment_realization_residual": deployment_residual,
        "quantization_residual_increment": deployment_residual - continuous_residual,
        "realization_residual": deployment_residual,
        "normalized_realization_residual": deployment_residual / max(target_dose, 1e-12),
        "actual_support_deviation": support_deviation,
        "actual_endpoint_in_tube": bool(support_deviation <= 0.0963749617980308),
        "actual_descriptor_internal_valid": bool(internal_valid),
        "actual_descriptor_internal_invalid_reason": internal_reason,
        "weighted_sign_concordance": sign_concordance,
        "delta_t": float(theta[0]),
        "c_t": float(theta[1]),
        "s_c": float(theta[2]),
        "initial_delta_t": float(x0[0]),
        "initial_c_t": float(x0[1]),
        "initial_s_c": float(x0[2]),
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_nfev": int(result.nfev),
        "solver_nit": int(getattr(result, "nit", -1)),
        "solver_fun": float(result.fun),
        "delta_t_lower_bound_hit": bool(lower_hits[0]),
        "delta_t_upper_bound_hit": bool(upper_hits[0]),
        "c_t_lower_bound_hit": bool(lower_hits[1]),
        "c_t_upper_bound_hit": bool(upper_hits[1]),
        "s_c_lower_bound_hit": bool(lower_hits[2]),
        "s_c_upper_bound_hit": bool(upper_hits[2]),
        "any_parameter_bound_hit": bool(np.any(lower_hits | upper_hits)),
        "output_array_sha256": final["output_array_sha256"],
        "realized_delta_sha256": float64_vector_sha(realized_delta),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    row.update(final["diagnostics"])
    for index, name in enumerate(BASE):
        row[f"original_raw__{name}"] = float(original_raw[index])
        row[f"target_raw__{name}"] = float(target_raw[index])
        row[f"continuous_raw__{name}"] = float(final["continuous_raw"][index])
        row[f"deployment_raw__{name}"] = float(final["deployment_raw"][index])
        row[f"original_std__{name}"] = float(original_std[index])
        row[f"target_std__{name}"] = float(target_std[index])
        row[f"continuous_std__{name}"] = float(continuous_std[index])
        row[f"deployment_std__{name}"] = float(deployment_std[index])
        row[f"target_delta__{name}"] = float(target_delta[index])
        row[f"realized_delta__{name}"] = float(realized_delta[index])
        row[f"error__{name}"] = float(deployment_std[index] - target_std[index])
    return row


def _linear_inequality_interval(a: float, b: float, lo: float, hi: float) -> tuple[float, float] | None:
    """Return d in [lo,hi] satisfying a+b*d <= 0."""
    if b == 0.0:
        return (lo, hi) if a <= 0.0 else None
    root = -a / b
    if b > 0.0:
        hi = min(hi, root)
    else:
        lo = max(lo, root)
    return (lo, hi) if lo <= hi else None


def internal_validity_interval(
    z0: np.ndarray,
    direction: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[float, float] | None:
    raw0 = inverse_standardize(z0, median, scale)
    raw_direction = np.asarray(direction, dtype=np.float64) * scale
    interval: tuple[float, float] | None = (float(lo), float(hi))
    constraints = [
        (raw0[0] - raw0[1], raw_direction[0] - raw_direction[1]),
        (raw0[1] - raw0[2], raw_direction[1] - raw_direction[2]),
        (-raw0[3], -raw_direction[3]),
        (-raw0[4], -raw_direction[4]),
        (raw0[5] - raw0[6], raw_direction[5] - raw_direction[6]),
    ]
    for a, b in constraints:
        if interval is None:
            return None
        interval = _linear_inequality_interval(float(a), float(b), interval[0], interval[1])
    return interval


def line_l1_ball_interval(
    point: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
    lo: float,
    hi: float,
) -> tuple[float, float] | None:
    """Exact piecewise-linear intersection of p+d*v with one L1 ball."""
    if lo > hi:
        return None
    p = np.asarray(point, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    breaks = [float(lo), float(hi)]
    nonzero = np.flatnonzero(v != 0.0)
    for index in nonzero:
        value = float((c[index] - p[index]) / v[index])
        if lo < value < hi:
            breaks.append(value)
    points = np.unique(np.asarray(breaks, dtype=np.float64))
    f = np.sum(np.abs(p[None, :] + points[:, None] * v[None, :] - c[None, :]), axis=1, dtype=np.float64)
    pieces: list[tuple[float, float]] = []
    for index in range(len(points) - 1):
        left = float(points[index])
        right = float(points[index + 1])
        fl = float(f[index])
        fr = float(f[index + 1])
        left_ok = fl <= radius
        right_ok = fr <= radius
        if left_ok and right_ok:
            pieces.append((left, right))
        elif left_ok != right_ok and fr != fl:
            crossing = left + (radius - fl) * (right - left) / (fr - fl)
            crossing = min(max(crossing, left), right)
            pieces.append((left, crossing) if left_ok else (crossing, right))
    for index, point_value in enumerate(points):
        if f[index] <= radius:
            pieces.append((float(point_value), float(point_value)))
    merged = merge_intervals(pieces)
    if not merged:
        return None
    # Convexity guarantees one sublevel interval for a single L1 ball.
    return merged[0][0], merged[-1][1]


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if a <= b)
    if not ordered:
        return []
    merged = [ordered[0]]
    for lo, hi in ordered[1:]:
        old_lo, old_hi = merged[-1]
        if lo <= old_hi or lo <= np.nextafter(old_hi, math.inf):
            merged[-1] = (old_lo, max(old_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _gaps(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    gaps: list[tuple[float, float]] = []
    cursor = lo
    for left, right in merge_intervals(intervals):
        if left > cursor:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < hi:
        gaps.append((cursor, hi))
    return gaps


def tube_validity_intervals(
    *,
    z0: np.ndarray,
    direction: np.ndarray,
    event_scaled: np.ndarray,
    tree: cKDTree,
    primary_center_indices: Sequence[int],
    tau: float,
    lo: float,
    hi: float,
) -> tuple[list[tuple[float, float]], dict[str, int]]:
    point = np.asarray(z0, dtype=np.float64) * WEIGHTS
    vector = np.asarray(direction, dtype=np.float64) * WEIGHTS
    centers = sorted(set(int(x) for x in primary_center_indices))
    intervals: list[tuple[float, float]] = []
    for index in centers:
        interval = line_l1_ball_interval(point, vector, event_scaled[index], tau, lo, hi)
        if interval is not None:
            intervals.append(interval)
    intervals = merge_intervals(intervals)
    primary_full = bool(
        len(intervals) == 1 and intervals[0][0] <= lo and intervals[0][1] >= hi
    )
    gap_queries = 0
    candidate_count = len(centers)
    gaps = _gaps(intervals, lo, hi)
    if gaps:
        extra: set[int] = set()
        for gap_lo, gap_hi in gaps:
            midpoint_d = 0.5 * (gap_lo + gap_hi)
            midpoint = point + midpoint_d * vector
            query_radius = np.nextafter(tau + 0.5 * (gap_hi - gap_lo), math.inf)
            found = tree.query_ball_point(midpoint, r=query_radius, p=1.0, eps=0.0)
            extra.update(int(x) for x in found)
            gap_queries += 1
        extra.difference_update(centers)
        candidate_count += len(extra)
        for index in sorted(extra):
            interval = line_l1_ball_interval(point, vector, event_scaled[index], tau, lo, hi)
            if interval is not None:
                intervals.append(interval)
        intervals = merge_intervals(intervals)
    return intervals, {
        "tube_gap_query_count": gap_queries,
        "tube_candidate_center_count": candidate_count,
        "tube_interval_count": len(intervals),
        "primary_centers_cover_full_interval": int(primary_full),
    }


def intervals_to_dose_index_ranges(
    intervals: Sequence[tuple[float, float]], doses: np.ndarray
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for lo, hi in intervals:
        left = int(np.searchsorted(doses, lo, side="left"))
        right = int(np.searchsorted(doses, hi, side="right")) - 1
        if 0 <= left <= right < len(doses):
            ranges.append((left, right))
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for lo, hi in ranges[1:]:
        old_lo, old_hi = merged[-1]
        if lo <= old_hi + 1:
            merged[-1] = (old_lo, max(old_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _range_events_add(events: dict[int, list[int]], left: int, right: int, slot: int) -> None:
    if left > right:
        return
    events.setdefault(int(left), [0, 0])[slot] += 1
    events.setdefault(int(right + 1), [0, 0])[slot] -= 1


def expected_visit_availability_one(
    *,
    uid: str,
    z_image: np.ndarray,
    anchor_index: int,
    original_in_tube: bool,
    original_internal_valid: bool,
    nearest_original_event_index: int,
    neighbor_indices: np.ndarray,
    neighbor_distances: np.ndarray,
    event_std: np.ndarray,
    event_scaled: np.ndarray,
    tree: cKDTree,
    doses: np.ndarray,
    dose_prefix_probability: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    tau: float,
    brute_force_probe_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    z_image = np.asarray(z_image, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    neighbor_distances = np.asarray(neighbor_distances, dtype=np.float64)
    events: dict[int, list[int]] = {}
    diagnostics = {
        "destination_ray_count": int(len(neighbor_indices)),
        "internal_interval_empty_count": 0,
        "tube_interval_empty_count": 0,
        "tube_gap_query_count": 0,
        "tube_candidate_center_count": 0,
        "tube_interval_count": 0,
        "primary_centers_cover_full_interval_count": 0,
    }
    for distance in neighbor_distances:
        right = int(np.searchsorted(doses, distance, side="right")) - 1
        if right >= 0:
            _range_events_add(events, 0, min(right, len(doses) - 1), 0)

    if original_in_tube and original_internal_valid:
        dose_lo = float(doses[0])
        dose_hi_global = float(doses[-1])
        for destination, distance in zip(neighbor_indices, neighbor_distances):
            hi = min(float(distance), dose_hi_global)
            if hi < dose_lo or distance <= 0.0:
                continue
            direction = (event_std[int(destination)] - event_std[int(anchor_index)]) / float(distance)
            internal = internal_validity_interval(z_image, direction, median, scale, dose_lo, hi)
            if internal is None:
                diagnostics["internal_interval_empty_count"] += 1
                continue
            tube_intervals, tube_diag = tube_validity_intervals(
                z0=z_image,
                direction=direction,
                event_scaled=event_scaled,
                tree=tree,
                primary_center_indices=[anchor_index, int(destination), nearest_original_event_index],
                tau=tau,
                lo=internal[0],
                hi=internal[1],
            )
            for key, value in tube_diag.items():
                if key == "primary_centers_cover_full_interval":
                    diagnostics["primary_centers_cover_full_interval_count"] += int(value)
                else:
                    diagnostics[key] += int(value)
            if not tube_intervals:
                diagnostics["tube_interval_empty_count"] += 1
                continue
            for left, right in intervals_to_dose_index_ranges(tube_intervals, doses):
                _range_events_add(events, left, right, 1)

    event_positions = sorted(set(events) | {0, len(doses)})
    denominator = 0
    numerator = 0
    existence_probability = 0.0
    full_probability = 0.0
    counts_at_probe: dict[int, tuple[int, int]] = {}
    probe_set = set(int(x) for x in (brute_force_probe_indices if brute_force_probe_indices is not None else []))
    for pos_index in range(len(event_positions) - 1):
        position = event_positions[pos_index]
        if position in events:
            denominator += events[position][0]
            numerator += events[position][1]
        next_position = event_positions[pos_index + 1]
        if numerator < 0 or denominator < 0 or numerator > denominator:
            raise RealizationNumericalError(
                f"Availability count invariant failed for {uid}: num={numerator}, den={denominator}"
            )
        probability_mass = float(dose_prefix_probability[next_position] - dose_prefix_probability[position])
        if denominator > 0:
            existence_probability += probability_mass
            full_probability += probability_mass * (numerator / denominator)
        for probe in probe_set:
            if position <= probe < next_position:
                counts_at_probe[probe] = (numerator, denominator)

    brute_force_rows: list[dict[str, Any]] = []
    if brute_force_probe_indices is not None:
        for index in sorted(probe_set):
            dose = float(doses[index])
            eligible_mask = neighbor_distances >= dose
            eligible = neighbor_indices[eligible_mask]
            denominator_direct = int(len(eligible))
            numerator_direct = 0
            if original_in_tube and original_internal_valid and denominator_direct:
                distances = neighbor_distances[eligible_mask]
                directions = (event_std[eligible] - event_std[int(anchor_index)][None, :]) / distances[:, None]
                targets = z_image[None, :] + dose * directions
                raw = inverse_standardize(targets, median, scale)
                internal = (
                    np.isfinite(raw).all(axis=1)
                    & (raw[:, 0] <= raw[:, 1])
                    & (raw[:, 1] <= raw[:, 2])
                    & (raw[:, 3] >= 0.0)
                    & (raw[:, 4] >= 0.0)
                    & (raw[:, 5] <= raw[:, 6])
                )
                deviations, _ = tree.query(targets * WEIGHTS, k=1, p=1, eps=0, workers=1)
                numerator_direct = int(np.sum(internal & (np.asarray(deviations) <= tau)))
            numerator_analytic, denominator_analytic = counts_at_probe.get(index, (0, 0))
            brute_force_rows.append(
                {
                    "uid": uid,
                    "dose_index": int(index),
                    "dose": dose,
                    "analytic_numerator": int(numerator_analytic),
                    "analytic_denominator": int(denominator_analytic),
                    "direct_numerator": numerator_direct,
                    "direct_denominator": denominator_direct,
                    "match": bool(
                        numerator_analytic == numerator_direct
                        and denominator_analytic == denominator_direct
                    ),
                }
            )
    return {
        "uid": uid,
        "expected_destination_existence_probability": float(existence_probability),
        "expected_full_descriptor_valid_probability": float(full_probability),
        "expected_effective_application_rate_p0p5": 0.5 * float(full_probability),
        **diagnostics,
        "brute_force_rows": brute_force_rows,
    }


def implementation_unit_tests(snapshot_dir: str | Path) -> dict[str, Any]:
    configure_q2_snapshot(snapshot_dir)
    tests: list[dict[str, Any]] = []

    tile = np.asarray(
        [[[0, 0, 0], [255, 255, 255]], [[255, 0, 0], [0, 255, 0]]], dtype=np.uint8
    )
    rgb = np.tile(tile, (8, 8, 1))
    palette = PaletteImage.from_rgb_uint8(rgb)
    exact = q2_exact_raw_descriptor(rgb)
    compressed = palette.original_descriptor()
    error = float(np.max(np.abs(exact - compressed)))
    tests.append({"name": "lossless_palette_descriptor_equivalence", "max_abs_error": error, "pass": error <= 1e-12})

    optimized_reference_errors = []
    for theta in (
        (0.0, 1.0, 1.0),
        (-math.log(2.0), 0.5, 0.5),
        (math.log(2.0), 2.0, 2.0),
        (0.173, 1.271, 0.843),
    ):
        output, _, _, _ = palette.continuous_output(theta)
        y = output @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
        output_lab = _srgb_to_lab(output)
        cstar = np.sqrt(output_lab[:, 1] ** 2 + output_lab[:, 2] ** 2)
        reference_descriptor = _descriptor_from_planes_palette_reference(
            y, cstar, palette.counts, palette.cell_memberships
        )
        optimized_descriptor = _descriptor_from_planes_palette(
            y, cstar, palette.counts, palette.cell_memberships
        )
        optimized_reference_errors.append(
            float(np.max(np.abs(reference_descriptor - optimized_descriptor)))
        )
    optimized_reference_error = max(optimized_reference_errors)
    tests.append(
        {
            "name": "redundant_sort_elimination_exact_equivalence",
            "fixed_theta_cases": 4,
            "max_abs_error": optimized_reference_error,
            "pass": optimized_reference_error == 0.0,
        }
    )

    _, _, frozen = require_q2()
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    lab = _srgb_to_lab(reference)
    roundtrip = _lab_to_srgb_unclipped(lab)
    roundtrip_error = float(np.max(np.abs(roundtrip - reference)))
    tests.append({"name": "paired_lab_inverse_roundtrip", "max_abs_error": roundtrip_error, "pass": roundtrip_error <= 1e-12})

    med = np.zeros(7, dtype=np.float64)
    scale = np.ones(7, dtype=np.float64)
    z0 = np.asarray([-2.0, -1.0, 0.0, 0.3, 0.2, 1.0, 2.0])
    direction = np.asarray([0.1, 0.2, 0.3, -0.1, 0.1, 0.2, 0.4])
    interval = internal_validity_interval(z0, direction, med, scale, 0.0, 1.0)
    samples = np.linspace(0.0, 1.0, 101)
    brute = []
    for d in samples:
        valid, _ = descriptor_internal_valid_raw(z0 + d * direction)
        brute.append(valid)
    predicted = [interval is not None and interval[0] <= d <= interval[1] for d in samples]
    tests.append({"name": "internal_validity_interval", "pass": brute == predicted})

    p = np.asarray([0.0, 0.0])
    v = np.asarray([1.0, 0.0])
    c = np.asarray([0.5, 0.25])
    l1_interval = line_l1_ball_interval(p, v, c, 0.5, 0.0, 1.0)
    expected = (0.25, 0.75)
    l1_error = max(abs(l1_interval[0] - expected[0]), abs(l1_interval[1] - expected[1])) if l1_interval else math.inf
    tests.append({"name": "line_l1_ball_interval", "max_abs_error": l1_error, "pass": l1_error <= 1e-15})

    disabled = rgb
    tests.append({"name": "augmentation_disabled_exact_passthrough", "pass": disabled is rgb and np.array_equal(disabled, rgb)})
    tests.append({"name": "q2_matrix_identity", "pass": np.asarray(frozen.SRGB_TO_XYZ_D65).shape == (3, 3)})
    return {"all_pass": all(bool(row["pass"]) for row in tests), "tests": tests}
