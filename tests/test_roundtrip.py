import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package
from src.pqc.signatures import DilithiumSigner


def test_roundtrip_uint8():
    rng = np.random.default_rng(123)
    vol = rng.integers(0, 256, size=(8, 8, 8), dtype=np.uint8)
    signer = DilithiumSigner()
    pk, sk = signer.keygen()
    pkg = encrypt_volume(vol, pk, sk, source_path='synthetic.npy', source_format='npy')
    rec = decrypt_package(pkg, sk)
    assert rec.shape == vol.shape
    assert rec.dtype == vol.dtype
    assert np.array_equal(rec, vol)
