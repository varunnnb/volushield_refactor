from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package
from src.pqc.signatures import DilithiumSigner


def get_slice(volume: np.ndarray, axis: int = 2, index: int | None = None) -> np.ndarray:
    """
    Extract slice from 3D volume

    axis=0 sagittal
    axis=1 coronal
    axis=2 axial
    """
    if index is None:
        index = volume.shape[axis] // 2

    if axis == 0:
        return volume[index, :, :]
    elif axis == 1:
        return volume[:, index, :]
    else:
        return volume[:, :, index]


def normalize_display(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0,1] for plotting
    """
    img = img.astype(np.float32)
    img -= img.min()

    if img.max() > 0:
        img /= img.max()

    return img


def main():

    path = r"data/raw/BraTS-GLI-00671-000/BraTS-GLI-00671-000-t1c.nii.gz"

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")

    img = nib.load(path)
    volume = img.get_fdata()

    volume = (
        (volume - volume.min()) /
        (volume.max() - volume.min()) * 255
    ).astype(np.uint8)

    print("Loaded shape:", volume.shape)

    signer = DilithiumSigner()
    pk, sk = signer.keygen()

    package = encrypt_volume(
        volume,
        pk,
        sk,
        source_path=path,
        source_format="nii.gz"
    )

    encrypted = package.payload

    decrypted = decrypt_package(package, sk).astype(np.uint8)

    slice_index = volume.shape[2] // 2

    orig_slice = get_slice(volume, axis=2, index=slice_index)
    enc_slice = get_slice(encrypted, axis=2, index=slice_index)
    dec_slice = get_slice(decrypted, axis=2, index=slice_index)

    diff = np.abs(
        orig_slice.astype(np.int16) -
        dec_slice.astype(np.int16)
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(normalize_display(orig_slice), cmap="gray")
    axes[0].set_title("Original MRI")

    axes[1].imshow(normalize_display(enc_slice), cmap="gray")
    axes[1].set_title("Encrypted")

    axes[2].imshow(normalize_display(dec_slice), cmap="gray")
    axes[2].set_title("Decrypted")

    axes[3].imshow(diff, cmap="hot")
    axes[3].set_title("Difference")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("Exact roundtrip:", np.array_equal(volume, decrypted))


if __name__ == "__main__":
    main()