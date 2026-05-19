from __future__ import annotations

import base64
from pathlib import Path
import numpy as np

from config import CONFIG
from src.chaos.hyperchaos import HyperChaoticSystem
from src.chaos.sequence import permutation_from_chaos, keystream_from_state
from src.encryption.package import EncryptedPackage
from src.pqc.signatures import DilithiumSigner
from src.utils_hash import combine_hashes, sha512_bytes


def _invert_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return inv


def _perm_from_seed(seed: bytes, n: int) -> np.ndarray:
    chaos_material = sha512_bytes(seed)

    rand = np.frombuffer(
        chaos_material,
        dtype=np.uint8
    )

    extra = keystream_from_state(
        chaos_material,
        n,
        rounds=1
    ).astype(np.uint8)

    floats = np.concatenate(
        [
            rand.astype(np.float64),
            extra.astype(np.float64)
        ]
    )

    if floats.size < n:
        repeats = int(np.ceil(n / floats.size))
        floats = np.tile(floats, repeats)

    return permutation_from_chaos(floats[:n])


def decrypt_package(
    package: EncryptedPackage,
    owner_secret_key: bytes
) -> np.ndarray:

    cipher = np.ascontiguousarray(
        package.payload.astype(np.uint8)
    )

    header = package.header
    meta = package.metadata

    volume_hash = bytes.fromhex(header.volume_hash)
    metadata_hash = bytes.fromhex(header.metadata_hash)
    package_hash = bytes.fromhex(header.package_hash)

    signature = base64.b64decode(
        header.signature_b64
    )

    signer_public_key = base64.b64decode(
        header.signer_public_key_b64
    )

    signer = DilithiumSigner()

    if not signer.verify(
        package_hash,
        signature,
        signer_public_key
    ):
        raise ValueError(
            "Signature verification failed"
        )

    owner_public_key = base64.b64decode(
        header.owner_public_key_b64
    )

    session_secret = sha512_bytes(
        owner_secret_key +
        owner_public_key +
        volume_hash +
        metadata_hash
    )

    chaos = HyperChaoticSystem()

    initial = chaos.seed_from_bytes(
        session_secret +
        volume_hash +
        metadata_hash
    )

    chaos_state = chaos.generate(
        initial,
        n=CONFIG.crypto.chaos_seed_rounds + 1,
        dt=CONFIG.crypto.chaos_dt,
        transient=CONFIG.crypto.chaos_transient
    )

    chaos_seed = sha512_bytes(
        chaos_state.tobytes()
    )

    flat = cipher.reshape(-1)

    keystream = keystream_from_state(
        combine_hashes(
            session_secret,
            chaos_seed,
            volume_hash
        ),
        flat.size,
        rounds=CONFIG.crypto.keystream_hash_rounds
    )

    permuted_flat = np.bitwise_xor(
        flat,
        keystream
    )

    perm = _perm_from_seed(
        chaos_seed,
        flat.size
    )

    inv = _invert_perm(perm)

    restored_flat = permuted_flat[inv]

    restored = restored_flat.reshape(
        cipher.shape
    )

    return restored.astype(
        np.uint8,
        copy=False
    )


def decrypt_package_from_path(
    path: str | Path,
    owner_secret_key: bytes
) -> np.ndarray:
    return decrypt_package(
        EncryptedPackage.load(path),
        owner_secret_key
    )