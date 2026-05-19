import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.normalizer import normalize_to_uint8, denormalize_from_uint8


def test_normalize_roundtrip_metadata():
    vol = np.array([[[0.0, 2.0], [4.0, 6.0]]], dtype=np.float32)
    norm, meta = normalize_to_uint8(vol)
    rec = denormalize_from_uint8(norm, meta)
    assert rec.shape == vol.shape
