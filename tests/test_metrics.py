import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics.security import entropy_uint8, npcr, uaci
from src.metrics.quality import psnr


def test_metrics_sanity():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    b = np.ones((4, 4, 4), dtype=np.uint8)
    assert entropy_uint8(a) == 0.0
    assert npcr(a, b) == 100.0
    assert uaci(a, b) > 0.0
    assert psnr(a, a) == float('inf')
