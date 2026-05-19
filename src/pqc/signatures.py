from __future__ import annotations

import hashlib
from dataclasses import dataclass

from config import CONFIG

try:
    import oqs
except Exception:
    oqs = None


@dataclass
class SignatureBundle:
    signature: bytes
    public_key: bytes
    algorithm: str
    backend: str


class DilithiumSigner:


    def __init__(self, algorithm: str | None = None):
        self.algorithm = algorithm or CONFIG.crypto.signature_algorithm

        self.backend = (
            "oqs"
            if oqs is not None and CONFIG.crypto.use_oqs_if_available
            else "fallback_hash_sig"
        )

        self._sk: bytes | None = None
        self._pk: bytes | None = None
        self._signer = None

    def keygen(self) -> tuple[bytes, bytes]:

        if self.backend == "oqs":

            self._signer = oqs.Signature(self.algorithm)

            pk = self._signer.generate_keypair()
            sk = self._signer.export_secret_key()

            self._pk = pk
            self._sk = sk

            return pk, sk

        sk = hashlib.sha512(
            b"volushield-dilithium-fallback-secret"
        ).digest() * 2

        pk = hashlib.sha512(sk).digest()

        self._pk = pk
        self._sk = sk

        return pk, sk

    def sign(
        self,
        message: bytes,
        secret_key: bytes | None = None,
    ) -> bytes:

        if self.backend == "oqs":

            if self._signer is None:
                self.keygen()

            if self._signer is None:
                raise ValueError("OQS signer initialization failed")

            return self._signer.sign(message)

        if secret_key is None:
            secret_key = self._sk

        if secret_key is None:
            raise ValueError("Secret key required for signing")

        pk = hashlib.sha512(secret_key).digest()

        return hashlib.sha512(pk + message).digest()

    def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:

        if self.backend == "oqs":
            verifier = oqs.Signature(self.algorithm)

            try:
                return bool(
                    verifier.verify(
                        message,
                        signature,
                        public_key,
                    )
                )
            except Exception:
                return False

        expected = hashlib.sha512(public_key + message).digest()

        return expected == signature

    def sign_bundle(self, message: bytes) -> SignatureBundle:

        if self._pk is None:
            self.keygen()

        signature = self.sign(message)

        return SignatureBundle(
            signature=signature,
            public_key=self._pk,
            algorithm=self.algorithm,
            backend=self.backend,
        )