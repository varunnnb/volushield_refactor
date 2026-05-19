from __future__ import annotations

import numpy as np

from src.encryption.pipeline import encrypt_volume
from src.metrics.performance import benchmark
from src.pqc.signatures import DilithiumSigner

if __name__ == '__main__':
    signer = DilithiumSigner()
    pk, sk = signer.keygen()
    for size in [(8, 8, 8), (16, 16, 16), (24, 24, 24)]:
        vol = np.random.default_rng(0).integers(0, 256, size=size, dtype=np.uint8)
        res = benchmark(encrypt_volume, vol, pk, sk, repeats=1)
        print(size, res['mean_seconds'])
