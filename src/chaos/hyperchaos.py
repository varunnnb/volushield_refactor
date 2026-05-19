from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.utils_hash import sha512_bytes

@dataclass
class HyperChaosState:
    """
    Container for 4D hyperchaotic initial state.
    """
    x: float
    y: float
    z: float
    w: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z, self.w],
            dtype=np.float64
        )


class HyperChaoticSystem:
    """
    True 4D Hyperchaotic Lorenz Generator for VoluShield
    """

    def __init__(
        self,
        a: float = 10.0,
        b: float = 8.0 / 3.0,
        c: float = 28.0,
        d: float = -1.0,
        r: float = -1.0,
        divergence_limit: float = 10000.0, 
    ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.r = r
        self.divergence_limit = divergence_limit

    def f(self, s: np.ndarray) -> np.ndarray:
        """
        Compute the derivatives for the Hyperchaotic Lorenz System.
        """
        x, y, z, w = s

        dx = self.a * (y - x) + w
        dy = self.c * x - x * z + self.d * y
        dz = x * y - self.b * z
        dw = -y * z + self.r * w

        return np.array(
            [dx, dy, dz, dw],
            dtype=np.float64
        )

    def rk4_step(
        self,
        state: np.ndarray,
        dt: float
    ) -> np.ndarray:

        k1 = self.f(state)
        k2 = self.f(state + 0.5 * dt * k1)
        k3 = self.f(state + 0.5 * dt * k2)
        k4 = self.f(state + dt * k3)

        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        nxt = np.clip(nxt, -self.divergence_limit, self.divergence_limit)

        if np.any(np.isnan(nxt)):
            raise RuntimeError("Hyperchaotic system diverged to NaN.")

        return nxt

    def seed_from_bytes(
        self,
        seed: bytes
    ) -> HyperChaosState:
        
        digest = sha512_bytes(seed)
        vals = np.frombuffer(digest, dtype=np.uint16).astype(np.float64)
        vals = vals[:4] / 65535.0

        vals = 2.0 * vals - 1.0

        vals += np.array([0.11, 0.23, 0.37, 0.41])

        return HyperChaosState(
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
        )

    def generate(
        self,
        initial: HyperChaosState,
        n: int,
        dt: float = 0.001,
        transient: int = 5000
    ) -> np.ndarray:

        state = initial.as_array()

        for _ in range(transient):
            state = self.rk4_step(state, dt)

        out = np.empty((n, 4), dtype=np.float64)

        for i in range(n):
            state = self.rk4_step(state, dt)
            out[i] = state

        return out

    def generate_uint8_sequence(
        self,
        initial: HyperChaosState,
        n: int,
        dt: float = 0.001,
        transient: int = 5000
    ) -> np.ndarray:

        states = self.generate(
            initial=initial,
            n=n,
            dt=dt,
            transient=transient
        )

        x, y, z, w = states[:, 0], states[:, 1], states[:, 2], states[:, 3]

        mix = (
            np.sin(x * 7.31) +
            np.cos(y * 5.17) +
            np.sin(z * 3.91) +
            np.cos(w * 9.13)
        )

        seq = np.mod(np.abs(mix) * 1e8, 256).astype(np.uint8)
        return seq

    def sensitivity_test(
        self,
        seed_a: bytes,
        seed_b: bytes,
        n: int = 1000
    ) -> float:

        s1 = self.seed_from_bytes(seed_a)
        s2 = self.seed_from_bytes(seed_b)

        q1 = self.generate_uint8_sequence(s1, n)
        q2 = self.generate_uint8_sequence(s2, n)

        return float(np.mean(np.abs(q1.astype(np.int16) - q2.astype(np.int16))))