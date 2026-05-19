from __future__ import annotations

import numpy as np

from src.utils_hash import derive_u64s, sha512_bytes


def _sanitize_seq(seq: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(seq, dtype=np.float64)


def permutation_from_chaos(seq: np.ndarray) -> np.ndarray:
    seq = _sanitize_seq(seq)
    return np.argsort(seq, kind='mergesort').astype(np.int64)


def block_permutation_from_state(state_bytes: bytes, n_blocks: int) -> np.ndarray:
    seeds = derive_u64s(sha512_bytes(state_bytes), n_blocks)
    return np.argsort(seeds, kind='mergesort').astype(np.int64)


def keystream_from_state(state_bytes: bytes, n: int, rounds: int = 2) -> np.ndarray:
    material = state_bytes
    out = bytearray()
    counter = 0
    while len(out) < n:
        payload = material + counter.to_bytes(8, 'big')
        digest = sha512_bytes(payload)
        for _ in range(rounds - 1):
            digest = sha512_bytes(digest)
        out.extend(digest)
        counter += 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8).copy()
