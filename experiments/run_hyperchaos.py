from __future__ import annotations

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chaos.hyperchaos import HyperChaoticSystem


if __name__ == '__main__':

    chaos = HyperChaoticSystem()

    init = chaos.seed_from_bytes(b'volushield')

    seq = chaos.generate(init, 5000)

    x = seq[:, 0]
    y = seq[:, 1]
    z = seq[:, 2]
    w = seq[:, 3]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, linewidth=0.4)
    ax.set_title("Hyperchaotic Attractor")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(x[:1000], label='x')
    plt.plot(y[:1000], label='y')
    plt.plot(z[:1000], label='z')
    plt.plot(w[:1000], label='w')
    plt.title("Chaotic State Trajectories")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.hist(x, bins=50)
    plt.title("Distribution of x sequence")
    plt.show()