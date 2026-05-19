from __future__ import annotations

import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chaos.hyperchaos import HyperChaoticSystem


def largest_lyapunov(
    steps=20000,
    dt=1e-4,
    eps=1e-8,
    renorm_every=10
):
    sys = HyperChaoticSystem()

    s1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    s2 = s1 + eps

    log_sum = 0.0
    count = 0

    for i in range(steps):

        s1 = sys.rk4_step(s1, dt)
        s2 = sys.rk4_step(s2, dt)

        d = s2 - s1
        dist = np.linalg.norm(d)

        if dist == 0:
            continue

        if (i + 1) % renorm_every == 0:
            log_sum += np.log(dist / eps)
            count += 1

            d = (eps / dist) * d
            s2 = s1 + d

    lle = log_sum / (count * renorm_every * dt)
    return lle


if __name__ == "__main__":
    val = largest_lyapunov()

    print("Largest Lyapunov Exponent =", val)

    if val > 0:
        print("Chaotic behavior detected")
    else:
        print("Not chaotic / stable / periodic")