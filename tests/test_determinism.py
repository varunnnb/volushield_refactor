import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.pqc.signatures import DilithiumSigner


def test_deterministic_same_input():
    vol = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    signer = DilithiumSigner()
    pk, sk = signer.keygen()
    p1 = encrypt_volume(vol, pk, sk)
    p2 = encrypt_volume(vol, pk, sk)
    assert np.array_equal(p1.payload, p2.payload)
    assert p1.header.package_hash == p2.header.package_hash
