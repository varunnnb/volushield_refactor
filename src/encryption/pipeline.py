from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from config import CONFIG
from src.chaos.hyperchaos import HyperChaoticSystem
from src.chaos.sequence import permutation_from_chaos, keystream_from_state
from src.encryption.package import EncryptedPackage, PackageHeader
from src.pqc.lwe import LWEKEM, derive_session_secret
from src.pqc.signatures import DilithiumSigner
from src.preprocessing.normalizer import load_volume, normalize_to_uint8
from src.utils_hash import combine_hashes, hash_array, hash_json, sha512_bytes


def _permute_flat(flat: np.ndarray, seed: bytes) -> tuple[np.ndarray, np.ndarray]:
    chaos_material = sha512_bytes(seed)
    rand = np.frombuffer(chaos_material, dtype=np.uint8)
    extra = keystream_from_state(chaos_material, flat.size, rounds=1).astype(np.uint8)
    floats = np.concatenate([rand.astype(np.float64), extra.astype(np.float64)])
    if floats.size < flat.size:
        repeats = int(np.ceil(flat.size / floats.size))
        floats = np.tile(floats, repeats)
    perm = permutation_from_chaos(floats[: flat.size])
    return flat[perm], perm


def encrypt_volume(
    volume: np.ndarray,
    owner_public_key: bytes,
    owner_secret_key: bytes,
    source_path: str = '',
    source_format: str = ''
) -> EncryptedPackage:

    norm, meta = normalize_to_uint8(
        volume,
        source_path=source_path,
        source_format=source_format
    )

    volume_hash = hash_array(norm)
    metadata_hash = hash_json(meta)

    session_secret = sha512_bytes(
        owner_secret_key +
        owner_public_key +
        volume_hash +
        metadata_hash
    )

    kem = LWEKEM()
    kem_ct = b''
    try:
        kem_pk, _ = kem.keygen()
        kem_ct, _ = kem.encapsulate(kem_pk)
    except Exception:
        kem_ct = b''

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

    chaos_seed = sha512_bytes(chaos_state.tobytes())

    flat = norm.reshape(-1)

    permuted_flat, _ = _permute_flat(
        flat,
        chaos_seed
    )

    keystream = keystream_from_state(
        combine_hashes(
            session_secret,
            chaos_seed,
            volume_hash
        ),
        flat.size,
        rounds=CONFIG.crypto.keystream_hash_rounds
    )

    cipher_flat = np.bitwise_xor(
        permuted_flat,
        keystream
    )

    cipher = cipher_flat.reshape(norm.shape)

    signer = DilithiumSigner()

    signer._pk, signer._sk = signer.keygen()
    
    signer.keygen = lambda: (owner_public_key, owner_secret_key)

    package_hash = sha512_bytes(
        cipher.tobytes() +
        volume_hash +
        metadata_hash 
    )

    signature = signer.sign(package_hash)

    header = PackageHeader(
        version='1.0',
        algorithm='VoluShield-Hybrid-3D',
        kem_algorithm=kem.algorithm,
        signature_algorithm=signer.algorithm,
        backend=f'{kem.backend}+{signer.backend}',
        shape=list(norm.shape),
        dtype='uint8',
        block_size=CONFIG.crypto.block_size,
        volume_hash=volume_hash.hex(),
        metadata_hash=metadata_hash.hex(),
        package_hash=package_hash.hex(),
        kem_ciphertext_b64=base64.b64encode(kem_ct).decode('ascii'),
        signature_b64=base64.b64encode(signature).decode('ascii'),
        signer_public_key_b64=base64.b64encode(signer._pk).decode('ascii'),
        owner_public_key_b64=base64.b64encode(owner_public_key).decode('ascii'),
        owner_secret_hint=base64.b64encode(
            sha512_bytes(owner_secret_key)[:8]
        ).decode('ascii'),
    )

    return EncryptedPackage(
        header=header,
        payload=cipher.astype(np.uint8),
        metadata=meta
    )

def encrypt_volume_from_path(path: str | Path, owner_public_key: bytes, owner_secret_key: bytes) -> EncryptedPackage:
    volume = load_volume(path)
    suffixes = ''.join(Path(path).suffixes).lower()
    source_format = 'npy' if suffixes.endswith('.npy') else ('nii.gz' if suffixes.endswith('.nii.gz') else ('nii' if suffixes.endswith('.nii') else 'unknown'))
    return encrypt_volume(volume, owner_public_key, owner_secret_key, source_path=str(path), source_format=source_format)
