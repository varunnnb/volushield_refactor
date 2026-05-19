from __future__ import annotations

import numpy as np


def entropy_uint8(volume: np.ndarray) -> float:
    """
    Shannon entropy for uint8 data
    """
    flat = np.asarray(
        volume,
        dtype=np.uint8
    ).reshape(-1)

    hist = np.bincount(
        flat,
        minlength=256
    ).astype(np.float64)

    probs = hist / hist.sum()
    probs = probs[probs > 0]

    return float(
        -(probs * np.log2(probs)).sum()
    )


def correlation(
    volume: np.ndarray,
    axis: int = 0
) -> float:
    """
    Adjacent voxel correlation
    """
    arr = np.asarray(
        volume,
        dtype=np.float64
    )

    if arr.ndim != 3:
        raise ValueError("Expected 3D")

    if axis == 0:
        x = arr[:-1, :, :].ravel()
        y = arr[1:, :, :].ravel()

    elif axis == 1:
        x = arr[:, :-1, :].ravel()
        y = arr[:, 1:, :].ravel()

    elif axis == 2:
        x = arr[:, :, :-1].ravel()
        y = arr[:, :, 1:].ravel()

    else:
        raise ValueError("axis must be 0/1/2")

    if x.size == 0:
        return 0.0

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def npcr(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Number of Pixel Change Rate
    """
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    return float(
        (a != b).mean() * 100.0
    )


def uaci(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Unified Average Changing Intensity
    """
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    a = a.astype(np.float64)
    b = b.astype(np.float64)

    return float(
        np.mean(np.abs(a - b)) / 255.0 * 100.0
    )


def chi_square_uniformity(
    volume: np.ndarray
) -> float:
    """
    Chi-square score against uniform histogram.
    Lower is better.
    """
    flat = np.asarray(
        volume,
        dtype=np.uint8
    ).ravel()

    hist = np.bincount(
        flat,
        minlength=256
    ).astype(np.float64)

    expected = flat.size / 256.0

    score = np.sum(
        (hist - expected) ** 2 / expected
    )

    return float(score)