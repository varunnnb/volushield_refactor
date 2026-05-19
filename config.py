from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
ENCRYPTED_DIR = DATA_DIR / 'encrypted'
RESULTS_DIR = ROOT / 'results'
PLOTS_DIR = ROOT / 'plots'
LEGACY_DIR = ROOT / 'legacy'


def ensure_dirs() -> None:
    for _p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, ENCRYPTED_DIR, RESULTS_DIR, PLOTS_DIR, LEGACY_DIR]:
        try:
            _p.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            # Keep import side-effect free in restricted environments.
            pass


@dataclass(frozen=True)
class CryptoConfig:
    # Practical defaults. Set to oqs-backed Dilithium/Kyber when available.
    lwe_dimension: int = 64
    lwe_modulus: int = 3329
    block_size: int = 4096
    chaos_dt: float = 1e-3
    chaos_transient: int = 1000
    chaos_seed_rounds: int = 8
    keystream_hash_rounds: int = 2
    signature_algorithm: str = 'Dilithium3'
    kem_algorithm: str = 'Kyber768'
    use_oqs_if_available: bool = True
    normalization_clip_percentile: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    root: Path = ROOT
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    encrypted_dir: Path = ENCRYPTED_DIR
    results_dir: Path = RESULTS_DIR
    plots_dir: Path = PLOTS_DIR
    legacy_dir: Path = LEGACY_DIR
    crypto: CryptoConfig = CryptoConfig()


CONFIG = AppConfig()


def config_dict() -> dict:
    d = asdict(CONFIG)
    d['root'] = str(d['root'])
    d['raw_dir'] = str(d['raw_dir'])
    d['processed_dir'] = str(d['processed_dir'])
    d['encrypted_dir'] = str(d['encrypted_dir'])
    d['results_dir'] = str(d['results_dir'])
    d['plots_dir'] = str(d['plots_dir'])
    d['legacy_dir'] = str(d['legacy_dir'])
    return d
