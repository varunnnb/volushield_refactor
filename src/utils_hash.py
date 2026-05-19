from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def sha512_bytes(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def sha512_hex(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def hash_json(obj: Any) -> bytes:
    if is_dataclass(obj):
        obj = asdict(obj)
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    return sha512_bytes(payload)


def hash_array(arr: np.ndarray) -> bytes:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha512()
    h.update(str(a.shape).encode('utf-8'))
    h.update(str(a.dtype).encode('utf-8'))
    h.update(a.tobytes(order='C'))
    return h.digest()


def combine_hashes(*parts: bytes) -> bytes:
    h = hashlib.sha512()
    for p in parts:
        h.update(p)
    return h.digest()


def derive_u64s(seed: bytes, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.uint64)
    counter = 0
    filled = 0
    while filled < n:
        h = hashlib.sha512(seed + counter.to_bytes(8, 'big')).digest()
        block = np.frombuffer(h, dtype=np.uint64)
        take = min(len(block), n - filled)
        out[filled:filled + take] = block[:take]
        filled += take
        counter += 1
    return out
