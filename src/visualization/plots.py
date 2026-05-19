from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(original: np.ndarray, encrypted: np.ndarray, save_path: str | Path | None = None) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(original.reshape(-1), bins=256)
    ax[0].set_title('Original')
    ax[1].hist(encrypted.reshape(-1), bins=256)
    ax[1].set_title('Encrypted')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_slice_triplet(original: np.ndarray, encrypted: np.ndarray, decrypted: np.ndarray, slice_idx: int | None = None, save_path: str | Path | None = None) -> None:
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, title in zip(ax, [original, encrypted, decrypted], ['Original', 'Encrypted', 'Decrypted']):
        a.imshow(img[slice_idx], cmap='gray')
        a.set_title(title)
        a.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
