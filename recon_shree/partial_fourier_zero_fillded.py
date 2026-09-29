import os

import h5py
from matplotlib import pyplot as plt
import numpy as np


if __name__ == "__main__":

    h5_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../data/grappa_reconstructed_kspace.h5"
        )
    )

    with h5py.File(h5_file, "r") as hf:
        k_full = hf["kspace"][:]

    print("k-space shape:", k_full.shape)

    n_ro, n_pe, n_ch = k_full.shape

    pf_fraction = 0.625  
    n_acquired = int(round(pf_fraction * n_pe))
    start = n_pe - n_acquired  

    mask_pf = np.zeros_like(k_full).real
    mask_pf[:, start:, :] = 1

    k_partial = k_full * mask_pf

    im_coils = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_partial, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )

    im_sos = np.sqrt(np.sum(np.abs(im_coils) ** 2, axis=2))

    # --- Ground truth: same FFT/SOS steps, but on the untouched full k-space ---
    im_truth_coils = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_full, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    im_truth_sos = np.sqrt(np.sum(np.abs(im_truth_coils) ** 2, axis=2))

    nrmse = np.linalg.norm(im_sos - im_truth_sos) / np.linalg.norm(im_truth_sos)
    print(f"NRMSE vs ground truth: {nrmse:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(im_truth_sos, cmap="gray")
    axes[0].set_title("Fully sampled (truth)")
    axes[0].axis("off")

    axes[1].imshow(im_sos, cmap="gray")
    axes[1].set_title(f"Zero-Filled PF (fraction={pf_fraction})")
    axes[1].axis("off")

    axes[2].imshow(np.abs(im_truth_sos - im_sos), cmap="gray")
    axes[2].set_title("Difference ")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()