# VoluShield refactor

This repository is a research-grade scaffold for 3D MRI volume encryption with:

- 4D hyperchaos as seed/control material
- LWE-style or oqs-backed KEM for session establishment
- Dilithium-backed signatures when `oqs` is installed
- SHA-512 binding for volume/metadata/package hashes
- exact round-trip reconstruction of the normalized `uint8` volume

## Main entry points

- `src/encryption/pipeline.py`
- `src/decryption/pipeline.py`
- `tests/`

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib nibabel scikit-image pytest psutil
# optional for real PQC backends:
pip install git+https://github.com/open-quantum-safe/liboqs-python.git
```

## Run

```bash
pytest -q
```

## Important note

If `oqs` is not installed, the code falls back to deterministic toy implementations so the project remains runnable. For publication or deployment, switch to the `oqs` backend.
