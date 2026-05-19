from __future__ import annotations

import base64
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from src.preprocessing.normalizer import VolumeMetadata


@dataclass
class PackageHeader:
    version: str
    algorithm: str
    kem_algorithm: str
    signature_algorithm: str
    backend: str
    shape: list[int]
    dtype: str
    block_size: int
    volume_hash: str
    metadata_hash: str
    package_hash: str
    kem_ciphertext_b64: str
    signature_b64: str
    signer_public_key_b64: str
    owner_public_key_b64: str
    owner_secret_hint: str = ''


@dataclass
class EncryptedPackage:
    header: PackageHeader
    payload: np.ndarray
    metadata: VolumeMetadata

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            payload=self.payload,
            header=np.array([json.dumps(asdict(self.header), sort_keys=True)], dtype=object),
            metadata=np.array([json.dumps(self.metadata.to_dict(), sort_keys=True)], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> 'EncryptedPackage':
        data = np.load(Path(path), allow_pickle=True)
        header = PackageHeader(**json.loads(str(data['header'][0])))
        metadata = VolumeMetadata.from_dict(json.loads(str(data['metadata'][0])))
        payload = np.asarray(data['payload'])
        return cls(header=header, payload=payload, metadata=metadata)
