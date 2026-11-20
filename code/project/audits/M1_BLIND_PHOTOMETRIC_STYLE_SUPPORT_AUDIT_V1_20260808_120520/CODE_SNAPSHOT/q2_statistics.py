from __future__ import annotations

import math
from typing import Sequence

import dcor
import numpy as np
from scipy.stats import wasserstein_distance

from q2_descriptors import inverse_ecdf_quantiles
from q2_frozen import (
    AXIS_ORDER,
    B_DCOR_PERM,
    B_GATE1_BOOT,
    B_LOCATION_PERM,
    B_NULL,
    RNG_SUBKEYS,
)
from q2_rng import (
    hash_bit,
    sha256_u64,
    stable_keys_u64,
    stateless_draw_indices,
    stateless_scores,
)


def weighted_inverse_ecdf_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    probability: float,
) -> float:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value_array.size == 0 or value_array.size != weight_array.size:
        raise ValueError("Values and weights must be same nonzero length")
    if not np.all(np.isfinite(value_array)) or not np.all(np.isfinite(weight_array)):
        raise ValueError("Values and weights must be finite")
    if np.any(weight_array < 0.0) or not float(weight_array.sum()) > 0.0:
        raise ValueError("Weights must be nonnegative with positive total")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Invalid probability: {probability}")
    order = np.argsort(value_array, kind="mergesort")
    sorted_values = value_array[order]
    sorted_weights = weight_array[order]
    unique_values, first = np.unique(sorted_values, return_index=True)
    merged_weights = np.add.reduceat(sorted_weights, first)
    normalized = merged_weights / merged_weights.sum(dtype=np.float64)
    cdf = np.cumsum(normalized, dtype=np.float64)
    index = int(np.searchsorted(cdf, probability, side="left"))
    index = min(index, len(unique_values) - 1)
    return float(unique_values[index])


def weighted_quantile_unit_tests() -> dict:
    cases = [
        {
            "name": "two_point_median_inverse_ecdf",
            "values": [0.0, 10.0],
            "weights": [0.5, 0.5],
            "p": 0.5,
            "expected": 0.0,
        },
        {
            "name": "two_point_just_above_median",
            "values": [0.0, 10.0],
            "weights": [0.5, 0.5],
            "p": 0.5000000001,
            "expected": 10.0,
        },
        {
            "name": "duplicate_weights_merged",
            "values": [1.0, 1.0, 2.0],
            "weights": [0.2, 0.3, 0.5],
            "p": 0.5,
            "expected": 1.0,
        },
        {
            "name": "unnormalized_weights_normalized",
            "values": [3.0, 1.0, 2.0],
            "weights": [5.0, 2.0, 3.0],
            "p": 0.25,
            "expected": 2.0,
        },
    ]
    results = []
    all_pass = True
    for case in cases:
        actual = weighted_inverse_ecdf_quantile(case["values"], case["weights"], case["p"])
        passed = actual == case["expected"]
        all_pass &= passed
        results.append({**case, "actual": actual, "pass": passed})
    return {
        "all_pass": bool(all_pass),
        "algorithm": "sort; merge equal values; normalize weights; inverse weighted ECDF; no interpolation",
        "cases": results,
    }


def empirical_quantile(values: Sequence[float] | np.ndarray, probability: float) -> float:
    return inverse_ecdf_quantiles(np.asarray(values, dtype=np.float64), [probability])[0]


def empirical_iqr(values: Sequence[float] | np.ndarray) -> float:
    q25, q75 = inverse_ecdf_quantiles(np.asarray(values, dtype=np.float64), [0.25, 0.75])
    return q75 - q25


def w1(
    source: Sequence[float] | np.ndarray,
    target: Sequence[float] | np.ndarray,
    source_weights: Sequence[float] | np.ndarray | None = None,
    target_weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if source_array.size == 0 or target_array.size == 0:
        raise ValueError("W1 requires nonempty distributions")
    return float(
        wasserstein_distance(
            source_array,
            target_array,
            u_weights=None if source_weights is None else np.asarray(source_weights, dtype=np.float64),
            v_weights=None if target_weights is None else np.asarray(target_weights, dtype=np.float64),
        )
    )


def location_event_weights(location_ids: Sequence[str]) -> np.ndarray:
    locations = np.asarray([str(value) for value in location_ids], dtype=object)
    unique, counts = np.unique(locations, return_counts=True)
    count_map = {location: int(count) for location, count in zip(unique, counts)}
    total_locations = len(unique)
    return np.asarray(
        [1.0 / (total_locations * count_map[location]) for location in locations],
        dtype=np.float64,
    )


def build_half_split_assignments(
    event_ids: Sequence[str],
    location_ids: Sequence[str],
    domain: str,
    replicates: int = B_NULL,
) -> np.ndarray:
    namespace = "M1_Q2_PSEUDODOMAIN_NULL_V1"
    events = np.asarray([str(value) for value in event_ids], dtype=object)
    locations = np.asarray([str(value) for value in location_ids], dtype=object)
    if len(events) != len(locations) or len(events) < 2:
        raise ValueError("Pseudo-domain split requires aligned events and locations")
    assignments = np.zeros((replicates, len(events)), dtype=bool)
    for location in sorted(set(locations.tolist())):
        indices = np.flatnonzero(locations == location)
        location_events = events[indices].tolist()
        stable = stable_keys_u64(namespace, location_events)
        tie = np.asarray(location_events, dtype=object)
        n = len(indices)
        for replicate in range(replicates):
            scores = stateless_scores(
                namespace,
                stable,
                "pseudo-split",
                replicate,
                domain,
                location,
            )
            order = np.lexsort((tie, scores))
            extra_to_b = bool(hash_bit(namespace, "odd-extra", replicate, domain, location))
            n_a = n // 2
            if n % 2 == 1 and not extra_to_b:
                n_a += 1
            assignments[replicate, indices[order[n_a:]]] = True
    if np.any(assignments.sum(axis=1) == 0) or np.any(assignments.sum(axis=1) == len(events)):
        raise RuntimeError(f"Pseudo-domain split produced empty side for {domain}")
    return assignments


def half_split_axis_statistics(
    axis_values: dict[str, np.ndarray],
    assignments_b: np.ndarray,
    keep_mask: np.ndarray | None = None,
    block_size: int = 64,
) -> dict[str, dict[str, np.ndarray]]:
    assignments = np.asarray(assignments_b, dtype=bool)
    if keep_mask is None:
        keep = np.ones(assignments.shape[1], dtype=bool)
    else:
        keep = np.asarray(keep_mask, dtype=bool)
    if keep.shape != (assignments.shape[1],) or int(keep.sum()) < 2:
        raise ValueError("Invalid pseudo-domain keep mask")
    result: dict[str, dict[str, np.ndarray]] = {}
    for axis, full_values in axis_values.items():
        values = np.asarray(full_values, dtype=np.float64)[keep]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_assignments = assignments[:, keep][:, order]
        differences = np.diff(sorted_values)
        w1_values = np.empty(assignments.shape[0], dtype=np.float64)
        signed_median = np.empty(assignments.shape[0], dtype=np.float64)
        positions = np.arange(1, len(values) + 1, dtype=np.int64)[None, :]
        for start in range(0, assignments.shape[0], block_size):
            end = min(assignments.shape[0], start + block_size)
            b = sorted_assignments[start:end]
            n_b = b.sum(axis=1).astype(np.int64)
            n_a = len(values) - n_b
            if np.any(n_a <= 0) or np.any(n_b <= 0):
                raise RuntimeError("Pseudo-domain split produced empty side after LOO")
            cum_b = np.cumsum(b, axis=1, dtype=np.int64)
            cum_a = positions - cum_b
            cdf_difference = cum_a / n_a[:, None] - cum_b / n_b[:, None]
            w1_values[start:end] = np.sum(
                np.abs(cdf_difference[:, :-1]) * differences[None, :],
                axis=1,
            )
            target_a = ((n_a + 1) // 2)[:, None]
            target_b = ((n_b + 1) // 2)[:, None]
            median_a_index = np.argmax(cum_a >= target_a, axis=1)
            median_b_index = np.argmax(cum_b >= target_b, axis=1)
            signed_median[start:end] = (
                sorted_values[median_b_index] - sorted_values[median_a_index]
            )
        result[axis] = {
            "w1": w1_values,
            "signed_median": signed_median,
            "absolute_signed_median": np.abs(signed_median),
        }
    return result


def max_null(
    source_stats: dict[str, dict[str, np.ndarray]],
    target_stats: dict[str, dict[str, np.ndarray]],
    field: str,
) -> np.ndarray:
    arrays = []
    for axis in AXIS_ORDER:
        arrays.append(np.asarray(source_stats[axis][field], dtype=np.float64))
        arrays.append(np.asarray(target_stats[axis][field], dtype=np.float64))
    return np.max(np.stack(arrays, axis=1), axis=1)


def gate1_bootstrap_iqr(
    axis_values: dict[str, np.ndarray],
    replicates: int = B_GATE1_BOOT,
) -> dict[str, np.ndarray]:
    subkey = RNG_SUBKEYS["gate1_event_bootstrap"]
    n = len(next(iter(axis_values.values())))
    output = {axis: np.empty(replicates, dtype=np.float64) for axis in axis_values}
    for replicate in range(replicates):
        indices = stateless_draw_indices(subkey, n, n, "target-event", replicate)
        for axis, values in axis_values.items():
            output[axis][replicate] = empirical_iqr(np.asarray(values)[indices])
    return output


def _bootstrap_count_matrix(
    namespace_or_subkey: str,
    n: int,
    replicate_indices: Sequence[int],
    *parts: object,
) -> np.ndarray:
    counts = np.empty((len(replicate_indices), n), dtype=np.int32)
    for row, replicate in enumerate(replicate_indices):
        draws = stateless_draw_indices(namespace_or_subkey, n, n, *parts, int(replicate))
        counts[row] = np.bincount(draws, minlength=n).astype(np.int32)
    return counts


def _w1_count_bootstrap_block(
    source_values: np.ndarray,
    target_values: np.ndarray,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    combined = np.concatenate([source, target])
    origin = np.concatenate(
        [np.arange(len(source), dtype=np.int64), np.arange(len(target), dtype=np.int64)]
    )
    is_source = np.concatenate(
        [np.ones(len(source), dtype=bool), np.zeros(len(target), dtype=bool)]
    )
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    sorted_origin = origin[order]
    sorted_source = is_source[order]
    contribution = np.empty((source_counts.shape[0], len(combined)), dtype=np.float64)
    contribution[:, sorted_source] = (
        source_counts[:, sorted_origin[sorted_source]] / float(len(source))
    )
    contribution[:, ~sorted_source] = -(
        target_counts[:, sorted_origin[~sorted_source]] / float(len(target))
    )
    cdf = np.cumsum(contribution, axis=1)
    return np.sum(np.abs(cdf[:, :-1]) * np.diff(sorted_values)[None, :], axis=1)


def foreground_paired_bootstrap(
    axis: str,
    source_states: dict[str, np.ndarray],
    target_states: dict[str, np.ndarray],
    replicates: int,
    block_size: int = 16,
) -> dict[str, np.ndarray]:
    subkey = RNG_SUBKEYS["foreground_event_bootstrap"]
    n_source = len(source_states["WHOLE"])
    n_target = len(target_states["WHOLE"])
    for state in ("HPL_MASK", "RANDOM_MASK"):
        if len(source_states[state]) != n_source or len(target_states[state]) != n_target:
            raise ValueError("Foreground paired states are not event-aligned")
    result = {
        "S_WHOLE": np.empty(replicates, dtype=np.float64),
        "S_HPL_MASK": np.empty(replicates, dtype=np.float64),
        "S_RANDOM_MASK": np.empty(replicates, dtype=np.float64),
        "E": np.empty(replicates, dtype=np.float64),
    }
    for start in range(0, replicates, block_size):
        end = min(replicates, start + block_size)
        reps = list(range(start, end))
        source_counts = _bootstrap_count_matrix(
            subkey, n_source, reps, "foreground", axis, "source"
        )
        target_counts = _bootstrap_count_matrix(
            subkey, n_target, reps, "foreground", axis, "target"
        )
        state_w1 = {}
        for state in ("WHOLE", "HPL_MASK", "RANDOM_MASK"):
            state_w1[state] = _w1_count_bootstrap_block(
                source_states[state],
                target_states[state],
                source_counts,
                target_counts,
            )
            result[f"S_{state}"][start:end] = state_w1[state]
        delta_fg = np.abs(state_w1["WHOLE"] - state_w1["HPL_MASK"])
        delta_random = np.abs(state_w1["WHOLE"] - state_w1["RANDOM_MASK"])
        result["E"][start:end] = delta_fg - delta_random
    return result


def dcor_unit_tests() -> dict:
    tolerance = 1e-12
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    same = float(dcor.distance_correlation(x, x))
    constant = float(dcor.distance_correlation(x, np.ones_like(x)))
    y = np.asarray([0.0, 1.0, 4.0, 9.0])
    nontrivial = float(dcor.distance_correlation(x, y))
    reverse = float(dcor.distance_correlation(y, x))
    expected_nontrivial = 0.9684641640120555
    checks = {
        "identical_equals_one": abs(same - 1.0) <= tolerance,
        "constant_equals_zero": abs(constant) <= tolerance,
        "nontrivial_known_value": abs(nontrivial - expected_nontrivial) <= tolerance,
        "symmetric": abs(nontrivial - reverse) <= tolerance,
        "bounded": 0.0 <= nontrivial <= 1.0,
    }
    return {
        "all_pass": all(checks.values()),
        "implementation": "dcor.distance_correlation",
        "values": {
            "identical": same,
            "constant": constant,
            "nontrivial": nontrivial,
            "reverse": reverse,
            "expected_nontrivial": expected_nontrivial,
            "tolerance": tolerance,
        },
        "checks": checks,
    }


def dcor_permutation_max_null(
    axis_values: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    replicates: int = B_DCOR_PERM,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], np.ndarray], np.ndarray]:
    namespace = "M1_Q2_DCOR_PERM_V1"
    n = len(next(iter(axis_values.values())))
    stable_indices = stable_keys_u64(namespace, [str(index) for index in range(n)])
    actual = {
        pair: float(dcor.distance_correlation(axis_values[pair[0]], axis_values[pair[1]]))
        for pair in pairs
    }
    null = {pair: np.empty(replicates, dtype=np.float64) for pair in pairs}
    max_values = np.empty(replicates, dtype=np.float64)
    tie = np.arange(n, dtype=np.int64)
    for replicate in range(replicates):
        shared_root = sha256_u64(namespace, "replicate-root", replicate)
        values = []
        for pair in pairs:
            pair_root = sha256_u64(namespace, "pair-subkey", pair[0], pair[1], shared_root)
            scores = stateless_scores(namespace, stable_indices, "permutation", shared_root, pair_root)
            permutation = np.lexsort((tie, scores))
            value = float(
                dcor.distance_correlation(
                    axis_values[pair[0]],
                    np.asarray(axis_values[pair[1]])[permutation],
                )
            )
            null[pair][replicate] = value
            values.append(value)
        max_values[replicate] = max(values)
    return actual, null, max_values


def mode_location_permutation(
    mode_labels: np.ndarray,
    location_ids: Sequence[str],
    replicates: int = B_LOCATION_PERM,
) -> tuple[dict[int, dict], float, np.ndarray]:
    namespace = "M1_Q2_LOCATION_PERM_V1"
    modes = np.asarray(mode_labels, dtype=np.int64)
    locations = np.asarray([str(value) for value in location_ids], dtype=object)
    unique_modes = sorted(int(value) for value in np.unique(modes))

    def concentration(permuted_locations: np.ndarray) -> tuple[dict[int, dict], float]:
        details = {}
        shares = []
        for mode in unique_modes:
            values, counts = np.unique(permuted_locations[modes == mode], return_counts=True)
            top = int(np.argmax(counts))
            share = float(counts[top] / counts.sum())
            details[mode] = {
                "top_location": str(values[top]),
                "top_location_count": int(counts[top]),
                "event_count": int(counts.sum()),
                "top_location_share": share,
            }
            shares.append(share)
        return details, max(shares)

    actual_details, actual_max = concentration(locations)
    stable = stable_keys_u64(namespace, [str(index) for index in range(len(locations))])
    tie = np.arange(len(locations), dtype=np.int64)
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        scores = stateless_scores(namespace, stable, "location-label-permutation", replicate)
        permutation = np.lexsort((tie, scores))
        _, null[replicate] = concentration(locations[permutation])
    return actual_details, actual_max, null


def pair_anchor_bootstrap(
    anchor_distances: np.ndarray,
    mode_labels: np.ndarray | None,
    replicates: int,
) -> np.ndarray:
    distances = np.asarray(anchor_distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 8:
        raise ValueError("Anchor distances must be N x 8")
    output = np.empty((replicates, 3), dtype=np.float64)
    if mode_labels is None:
        strata = [("all", np.arange(len(distances), dtype=np.int64))]
    else:
        labels = np.asarray(mode_labels, dtype=np.int64)
        strata = [
            (str(mode), np.flatnonzero(labels == mode))
            for mode in sorted(np.unique(labels).tolist())
        ]
    namespace = "M1_Q2_PAIR_BOOTSTRAP_V1"
    for replicate in range(replicates):
        sampled_rows = []
        for stratum, indices in strata:
            draws = stateless_draw_indices(
                namespace,
                len(indices),
                len(indices),
                "anchor-bootstrap",
                replicate,
                stratum,
            )
            sampled_rows.append(indices[draws])
        selected = np.concatenate(sampled_rows)
        output[replicate] = inverse_ecdf_quantiles(
            distances[selected].reshape(-1),
            [0.10, 0.50, 0.90],
        )
    return output
