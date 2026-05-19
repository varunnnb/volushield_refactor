from __future__ import annotations

import numpy as np
import nibabel as nib

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pqc.signatures import DilithiumSigner
from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package

from src.metrics.security import (
    entropy_uint8,
    correlation,
    npcr,
    uaci,
    chi_square_uniformity
)

from src.metrics.quality import (
    psnr,
    ssim_3d
)

from src.metrics.performance import benchmark


def normalize_uint8(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)

    mn = vol.min()
    mx = vol.max()

    if mx == mn:
        return np.zeros_like(
            vol,
            dtype=np.uint8
        )

    out = (vol - mn) / (mx - mn)
    out *= 255.0

    return out.astype(np.uint8)


if __name__ == "__main__":

    path = r"data/raw/BraTS-GLI-00467-000/BraTS-GLI-00467-000-t1c.nii.gz"

    img = nib.load(path)
    vol = img.get_fdata()

    vol = normalize_uint8(vol)

    signer = DilithiumSigner()
    pk, sk = signer.keygen()

    perf = benchmark(
        encrypt_volume,
        vol,
        pk,
        sk,
        repeats=3
    )

    pkg = perf["result"]

    rec = decrypt_package(pkg, sk)

    vol2 = vol.copy()
    vol2.flat[0] ^= 1

    pkg2 = encrypt_volume(vol2, pk, sk)

    enc1 = pkg.payload
    enc2 = pkg2.payload

    print("\n===== SECURITY =====")
    print("Entropy:", entropy_uint8(enc1))
    print("Corr X:", correlation(enc1, 0))
    print("Corr Y:", correlation(enc1, 1))
    print("Corr Z:", correlation(enc1, 2))
    print("NPCR:", npcr(enc1, enc2))
    print("UACI:", uaci(enc1, enc2))
    print("ChiSquare:", chi_square_uniformity(enc1))

    print("\n===== QUALITY =====")
    print("PSNR:", psnr(vol, rec))
    print("SSIM:", ssim_3d(vol, rec))

    print("\n===== PERFORMANCE =====")
    print("Mean Time:", perf["mean_seconds"])
    print("Std Time:", perf["std_seconds"])
    print("Memory MB:", perf.get("rss_mb", "N/A"))
    print("Throughput Vox/sec:",
          vol.size / perf["mean_seconds"])