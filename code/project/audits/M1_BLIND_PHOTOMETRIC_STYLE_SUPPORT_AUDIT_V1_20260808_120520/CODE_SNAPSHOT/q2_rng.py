from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

from q2_frozen import RNG_NAMESPACES


MASK64 = (1 << 64) - 1
SM64_GAMMA = np.uint64(0x9E3779B97F4A7C15)
SM64_M1 = np.uint64(0xBF58476D1CE4E5B9)
SM64_M2 = np.uint64(0x94D049BB133111EB)
ALLOWED_ROOTS = frozenset(RNG_NAMESPACES)


def _root_namespace(namespace_or_subkey: str) -> str:
    return namespace_or_subkey.split("|", 1)[0]


def _validate_namespace(namespace_or_subkey: str) -> None:
    if _root_namespace(namespace_or_subkey) not in ALLOWED_ROOTS:
        raise ValueError(f"Unregistered RNG namespace: {namespace_or_subkey}")


def _canonical_bytes(parts: Iterable[object]) -> bytes:
    payload = bytearray()
    for part in parts:
        encoded = str(part).encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big", signed=False))
        payload.extend(encoded)
    return bytes(payload)


def sha256_u64(namespace_or_subkey: str, *parts: object) -> int:
    _validate_namespace(namespace_or_subkey)
    digest = hashlib.sha256(
        _canonical_bytes((namespace_or_subkey, *parts))
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sha256_hex(namespace_or_subkey: str, *parts: object) -> str:
    _validate_namespace(namespace_or_subkey)
    return hashlib.sha256(_canonical_bytes((namespace_or_subkey, *parts))).hexdigest()


def splitmix64(values: np.ndarray | Sequence[int] | int) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + SM64_GAMMA
        z = (z ^ (z >> np.uint64(30))) * SM64_M1
        z = (z ^ (z >> np.uint64(27))) * SM64_M2
        z = z ^ (z >> np.uint64(31))
    return z.astype(np.uint64, copy=False)


def stable_key_u64(namespace_or_subkey: str, key: object) -> np.uint64:
    return np.uint64(sha256_u64(namespace_or_subkey, "stable-key", key))


def stable_keys_u64(namespace_or_subkey: str, keys: Sequence[object]) -> np.ndarray:
    return np.asarray(
        [stable_key_u64(namespace_or_subkey, key) for key in keys],
        dtype=np.uint64,
    )


def stateless_scores(
    namespace_or_subkey: str,
    stable_keys: np.ndarray,
    *parts: object,
) -> np.ndarray:
    root = np.uint64(sha256_u64(namespace_or_subkey, "root", *parts))
    return splitmix64(np.asarray(stable_keys, dtype=np.uint64) ^ root)


def stateless_order(
    namespace_or_subkey: str,
    keys: Sequence[str],
    *parts: object,
) -> np.ndarray:
    stable = stable_keys_u64(namespace_or_subkey, keys)
    scores = stateless_scores(namespace_or_subkey, stable, *parts)
    key_array = np.asarray([str(key) for key in keys], dtype=object)
    return np.lexsort((key_array, scores)).astype(np.int64)


def stateless_draw_indices(
    namespace_or_subkey: str,
    n_pool: int,
    n_draw: int,
    *parts: object,
) -> np.ndarray:
    if n_pool <= 0 or n_draw < 0:
        raise ValueError(f"Invalid draw sizes: pool={n_pool}, draw={n_draw}")
    root = np.uint64(sha256_u64(namespace_or_subkey, "draw-root", *parts))
    counters = np.arange(n_draw, dtype=np.uint64)
    mixed = splitmix64(counters ^ root)
    return (mixed % np.uint64(n_pool)).astype(np.int64)


def hash_bit(namespace_or_subkey: str, *parts: object) -> int:
    return sha256_u64(namespace_or_subkey, "bit", *parts) & 1


def matched_random_translation(uid: str, width: int, height: int) -> tuple[int, int, int]:
    namespace = "M1_Q2_RANDOM_MASK_V1"
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image extent: {width}x{height}")
    attempt = 0
    while True:
        dx = sha256_u64(namespace, "dx", uid, width, height, attempt) % width
        dy = sha256_u64(namespace, "dy", uid, width, height, attempt) % height
        if dx != 0 or dy != 0:
            return int(dx), int(dy), attempt
        attempt += 1
        if attempt > 1024:
            raise RuntimeError("Unable to derive nonzero toroidal translation")


def select_distinct_hash_ranked(
    namespace_or_subkey: str,
    anchor: str,
    candidates: Sequence[str],
    count: int,
    *parts: object,
) -> list[int]:
    if count < 0 or len(candidates) < count:
        raise ValueError(f"Reference pool too small: {len(candidates)} < {count}")
    stable = stable_keys_u64(namespace_or_subkey, candidates)
    scores = stateless_scores(namespace_or_subkey, stable, "anchor", anchor, *parts)
    candidate_array = np.asarray([str(value) for value in candidates], dtype=object)
    order = np.lexsort((candidate_array, scores))
    return [int(index) for index in order[:count]]


def sample_distinct_references(
    namespace_or_subkey: str,
    anchor: str,
    candidate_indices: Sequence[int],
    candidate_uids: Sequence[str],
    count: int,
) -> list[int]:
    """Select a fixed small number without enumerating anchor-reference pairs."""
    if len(candidate_indices) != len(candidate_uids):
        raise ValueError("Candidate indices and UIDs must align")
    if count < 0:
        raise ValueError("Negative sample count")
    eligible_count = sum(str(uid) != str(anchor) for uid in candidate_uids)
    if eligible_count < count:
        raise ValueError(f"Reference pool too small: {eligible_count} < {count}")
    selected: list[int] = []
    selected_set: set[int] = set()
    attempt = 0
    max_attempts = max(4096, 128 * count)
    while len(selected) < count:
        position = sha256_u64(
            namespace_or_subkey,
            "compatible-reference-draw",
            anchor,
            attempt,
        ) % len(candidate_indices)
        candidate_index = int(candidate_indices[position])
        candidate_uid = str(candidate_uids[position])
        attempt += 1
        if candidate_uid == str(anchor) or candidate_index in selected_set:
            if attempt > max_attempts:
                raise RuntimeError("Deterministic reference rejection sampler exhausted")
            continue
        selected.append(candidate_index)
        selected_set.add(candidate_index)
    return selected


def rng_algorithm_audit() -> dict:
    return {
        "cryptographic_root_hash": "SHA-256",
        "root_hash_encoding": "length-prefixed UTF-8 fields; first 64 bits big-endian",
        "stateless_mixer": "SplitMix64 finalizer",
        "ordering_tie_break": "lexicographic stable key",
        "mutable_global_rng_used": False,
        "call_order_dependent": False,
        "registered_root_namespaces": list(RNG_NAMESPACES),
        "random_translation_redraw": "increment integer attempt in same registered namespace until (dx,dy)!=(0,0)",
    }
