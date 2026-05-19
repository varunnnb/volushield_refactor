from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None


@dataclass
class VolumeMetadata:
    source_path: str
    source_format: str
    original_dtype: str
    original_shape: tuple[int, int, int]
    original_min: float
    original_max: float
    normalization_mode: str = 'minmax_uint8'
    scale: float = 1.0
    offset: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['original_shape'] = list(self.original_shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'VolumeMetadata':
        d = dict(d)
        d['original_shape'] = tuple(d['original_shape'])
        return cls(**d)


def _ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f'Expected a 3D volume, got shape {arr.shape}')
    return np.ascontiguousarray(arr)


def load_npy(path: str | Path) -> np.ndarray:
    arr = np.load(Path(path), allow_pickle=False)
    return _ensure_3d(np.asarray(arr))


def load_nifti(path: str | Path) -> np.ndarray:
    if nib is None:
        raise ImportError('nibabel is required to load NIfTI files')
    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32))
    return _ensure_3d(arr)


def load_volume(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.npy'):
        return load_npy(path)
    if suffixes.endswith('.nii') or suffixes.endswith('.nii.gz'):
        return load_nifti(path)
    raise ValueError(f'Unsupported input format: {path}')


def normalize_to_uint8(volume: np.ndarray, source_path: str = '', source_format: str = '') -> tuple[np.ndarray, VolumeMetadata]:
    arr = _ensure_3d(np.asarray(volume))
    original_dtype = str(arr.dtype)
    arr = arr.astype(np.float64, copy=False)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax == vmin:
        norm = np.zeros_like(arr, dtype=np.uint8)
        meta = VolumeMetadata(source_path, source_format, original_dtype, tuple(arr.shape), vmin, vmax, scale=1.0, offset=vmin)
        return norm, meta
    scale = 255.0 / (vmax - vmin)
    norm = np.rint((arr - vmin) * scale).clip(0, 255).astype(np.uint8)
    meta = VolumeMetadata(source_path, source_format, original_dtype, tuple(arr.shape), vmin, vmax, scale=scale, offset=vmin)
    return norm, meta


def denormalize_from_uint8(volume: np.ndarray, meta: VolumeMetadata) -> np.ndarray:
    arr = _ensure_3d(np.asarray(volume))
    if meta.original_max == meta.original_min:
        out = np.full(meta.original_shape, meta.original_min, dtype=np.float64)
    else:
        out = arr.astype(np.float64) / meta.scale + meta.offset
    return out.astype(np.dtype(meta.original_dtype), copy=False)
