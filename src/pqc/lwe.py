from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from config import CONFIG

try:
    import oqs
except Exception:
    oqs = None


@dataclass
class KEMCiphertext:
    ciphertext: bytes
    algorithm: str
    backend: str


class LWEKEM:
    """KEM adapter.

    If oqs is installed, this can use a real post-quantum KEM backend.
    Otherwise it uses a deterministic fallback KEM-style interface so the
    research scaffold remains runnable.
    """

    def __init__(self, algorithm: str | None = None):
        self.algorithm = algorithm or CONFIG.crypto.kem_algorithm
        self.backend = 'oqs' if oqs is not None and CONFIG.crypto.use_oqs_if_available else 'fallback_hash_kem'
        self._pk: bytes | None = None
        self._sk: bytes | None = None

    def keygen(self) -> tuple[bytes, bytes]:
        if self.backend == 'oqs':
            kem = oqs.KeyEncapsulation(self.algorithm)
            pk = kem.generate_keypair()
            sk = kem.export_secret_key()
            self._pk, self._sk = pk, sk
            return pk, sk
        sk = os.urandom(32)
        pk = hashlib.sha512(sk).digest()
        self._pk, self._sk = pk, sk
        return pk, sk

    def encapsulate(self, public_key: bytes):
        if self.backend == 'oqs':
            kem = oqs.KeyEncapsulation(self.algorithm)

            ciphertext, secret = kem.encap_secret(public_key)

            return ciphertext, secret

    def decapsulate(self, ciphertext: bytes, secret_key: bytes):
        if self.backend == 'oqs':
            kem = oqs.KeyEncapsulation(self.algorithm, secret_key)
            return kem.decap_secret(ciphertext)
        pk = hashlib.sha512(secret_key).digest()
        return hashlib.sha512(pk + ciphertext).digest()


def derive_session_secret(kem_shared_secret: bytes, volume_hash: bytes, metadata_hash: bytes) -> bytes:
    return hashlib.sha512(kem_shared_secret + volume_hash + metadata_hash).digest()
