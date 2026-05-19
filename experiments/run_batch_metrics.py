from __future__ import annotations

import time
from pathlib import Path
import sys
import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.pqc.signatures import DilithiumSigner
from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package

from src.metrics.security import (
    entropy_uint8,
    correlation,
    npcr,
    uaci,
)

from src.metrics.quality import (
    psnr,
    ssim_3d,
)

DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "batch_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CASES = 2

def normalize_uint8(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)

    mn = vol.min()
    mx = vol.max()

    if mx == mn:
        return np.zeros_like(vol, dtype=np.uint8)

    vol = (vol - mn) / (mx - mn)
    vol = vol * 255.0

    return vol.astype(np.uint8)


def get_middle_slice(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    idx = vol.shape[axis] // 2

    if axis == 0:
        return vol[idx, :, :]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def find_first_nii(case_dir: Path) -> Path | None:
    files = list(case_dir.glob("*.nii")) + list(case_dir.glob("*.nii.gz"))
    if not files:
        return None
    return files[0]


def save_visual(
    case_name: str,
    original: np.ndarray,
    encrypted: np.ndarray,
    decrypted: np.ndarray,
):
    orig = get_middle_slice(original)
    enc = get_middle_slice(encrypted)
    dec = get_middle_slice(decrypted)

    diff = np.abs(
        orig.astype(np.int16) -
        dec.astype(np.int16)
    )

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))

    ax[0].imshow(orig, cmap="gray")
    ax[0].set_title("Original")

    ax[1].imshow(enc, cmap="gray")
    ax[1].set_title("Encrypted")

    ax[2].imshow(dec, cmap="gray")
    ax[2].set_title("Decrypted")

    ax[3].imshow(diff, cmap="hot")
    ax[3].set_title("Difference")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    out_path = OUTPUT_DIR / f"{case_name}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():

    case_dirs = sorted(
        [p for p in DATA_DIR.iterdir() if p.is_dir()]
    )[:MAX_CASES]

    if not case_dirs:
        print("No cases found in:", DATA_DIR)
        return

    signer = DilithiumSigner()
    pk, sk = signer.keygen()

    print("=" * 90)
    print("VOLUSHIELD BATCH TEST")
    print("=" * 90)

    all_times = []

    for idx, case_dir in enumerate(case_dirs, start=1):

        try:
            nii_path = find_first_nii(case_dir)

            if nii_path is None:
                print(f"[{idx}] {case_dir.name} -> No NIfTI found")
                continue

            print("\n" + "=" * 90)
            print(f"[{idx}] CASE: {case_dir.name}")
            print("=" * 90)

            img = nib.load(str(nii_path))
            vol = img.get_fdata()

            if vol.ndim == 4:
                print("Detected 4D volume → selecting first time frame")
                vol = vol[:, :, :, 0]

            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

            vol = normalize_uint8(vol)

            print("Shape:", vol.shape)
            print("Dtype:", vol.dtype)

            t0 = time.perf_counter()

            pkg = encrypt_volume(
                vol,
                pk,
                sk,
                source_path=str(nii_path),
                source_format="nii.gz"
            )

            enc_time = time.perf_counter() - t0

            t1 = time.perf_counter()

            rec = decrypt_package(pkg, sk)

            dec_time = time.perf_counter() - t1

            all_times.append(enc_time)

            save_visual(
                case_dir.name,
                vol,
                pkg.payload,
                rec
            )

            vol2 = vol.copy()
            vol2.flat[0] ^= 1

            pkg2 = encrypt_volume(
                vol2,
                pk,
                sk,
                source_path=str(nii_path),
                source_format="nii.gz"
            )

            exact = np.array_equal(vol, rec)

            print("\n--- QUALITY ---")
            print("Exact Roundtrip :", exact)
            print("PSNR            :", psnr(vol, rec))
            print("SSIM            :", ssim_3d(vol, rec))

            print("\n--- SECURITY ---")
            print("Entropy         :", entropy_uint8(pkg.payload))
            print("Corr X          :", correlation(pkg.payload, 0))
            print("Corr Y          :", correlation(pkg.payload, 1))
            print("Corr Z          :", correlation(pkg.payload, 2))
            print("NPCR            :", npcr(pkg.payload, pkg2.payload))
            print("UACI            :", uaci(pkg.payload, pkg2.payload))

            print("\n--- PERFORMANCE ---")
            print("Encrypt Time(s) :", round(enc_time, 4))
            print("Decrypt Time(s) :", round(dec_time, 4))
            print("Voxels/sec      :", round(vol.size / enc_time, 2))

            print("\nSaved Visual ->", OUTPUT_DIR / f"{case_dir.name}.png")

        except Exception as e:
            print(f"\n[{idx}] ERROR in {case_dir.name}")
            print(str(e))

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if all_times:
        print("Cases Processed :", len(all_times))
        print("Avg Encrypt(s)  :", round(float(np.mean(all_times)), 4))
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()