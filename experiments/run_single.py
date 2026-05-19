from __future__ import annotations

import numpy as np
import nibabel as nib

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package
from src.pqc.signatures import DilithiumSigner


if __name__ == '__main__':

    path = r"data/raw/BraTS-GLI-00467-000/BraTS-GLI-00467-000-t1c.nii.gz"

    img = nib.load(path)
    vol = img.get_fdata()

    vol = ((vol - vol.min()) / (vol.max() - vol.min()) * 255).astype(np.uint8)

    print("Volume shape:", vol.shape)

    signer = DilithiumSigner()
    pk, sk = signer.keygen()

    pkg = encrypt_volume(vol, pk, sk, source_path=path, source_format='nii.gz')

    rec = decrypt_package(pkg, sk)

    print("Roundtrip exact:", np.array_equal(rec.astype(np.uint8), vol))