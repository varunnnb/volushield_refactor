import base64
import numpy as np
import pytest

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_package
from src.pqc.signatures import DilithiumSigner


def test_tamper_rejected():
    vol = np.zeros((4, 4, 4), dtype=np.uint8)
    signer = DilithiumSigner()
    pk, sk = signer.keygen()
    pkg = encrypt_volume(vol, pk, sk)
    pkg.header.signature_b64 = base64.b64encode(b'bad').decode('ascii')
    with pytest.raises(ValueError):
        decrypt_package(pkg, sk)
