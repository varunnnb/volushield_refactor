from __future__ import annotations

import os
import time
from typing import Callable, Any

import numpy as np

try:
    import psutil
except Exception:
    psutil = None


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 3,
    **kwargs: Any
) -> dict:

    times = []
    result = None

    for _ in range(repeats):
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        dt = time.perf_counter() - t0
        times.append(dt)

    out = {
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "result": result,
    }

    if psutil is not None:
        rss = psutil.Process(
            os.getpid()
        ).memory_info().rss

        out["rss_mb"] = rss / (1024 ** 2)

    return out