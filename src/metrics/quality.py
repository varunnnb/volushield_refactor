from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None


def psnr(
    a: np.ndarray,
    b: np.ndarray,
    data_range: float = 255.0
) -> float:
    """
    Peak Signal-to-Noise Ratio
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    mse = np.mean((a - b) ** 2)

    if mse == 0:
        return float("inf")

    return float(
        10.0 * np.log10((data_range ** 2) / mse)
    )


def ssim_3d(
    a: np.ndarray,
    b: np.ndarray,
    data_range: float = 255.0
) -> float:
    """
    Slice-wise SSIM averaged across axis-0
    """
    if _ssim is None:
        raise ImportError(
            "scikit-image required for SSIM"
        )

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError("Shape mismatch")

    vals = []

    for i in range(a.shape[0]):
        vals.append(
            _ssim(
                a[i],
                b[i],
                data_range=data_range
            )
        )

    return float(np.mean(vals))