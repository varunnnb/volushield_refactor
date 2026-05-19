from __future__ import annotations

import os
import numpy as np
import nibabel as nib

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package
from src.pqc.signatures import DilithiumSigner


DATASET_DIR = r"data/raw"
MAX_CASES = 10


if __name__ == '__main__':

    signer = DilithiumSigner()
    pk, sk = signer.keygen()

    count = 0

    for folder in os.listdir(DATASET_DIR):

        case_path = os.path.join(DATASET_DIR, folder)

        if not os.path.isdir(case_path):
            continue

        nii_file = os.path.join(case_path, f"{folder}-t1c.nii.gz")

        if not os.path.exists(nii_file):
            continue

        print(f"\nProcessing: {folder}")

        img = nib.load(nii_file)
        vol = img.get_fdata()

        vol = ((vol - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)

        print("Shape:", vol.shape)

        pkg = encrypt_volume(
            vol,
            pk,
            sk,
            source_path=nii_file,
            source_format='nii.gz'
        )

        rec = decrypt_package(pkg, sk)

        ok = np.array_equal(rec.astype(np.uint8), vol)

        print("Roundtrip:", ok)

        count += 1

        if count >= MAX_CASES:
            break

    print(f"\nProcessed {count} cases.")