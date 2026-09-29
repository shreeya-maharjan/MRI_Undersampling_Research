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

    # --- Ground truth ---------------------------
    im_truth_coils = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_full, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    full_truth = np.sqrt(np.sum(np.abs(im_truth_coils) ** 2, axis=2))


    # -------- Zero-filled reconstruction-----------------------------------------------------
  
    im_zf_coils = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_partial, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    im_zf_sos = np.sqrt(np.sum(np.abs(im_zf_coils) ** 2, axis=2))


    #----------- Homodyne reconstruction-----------------------------------------------------------
  
    center = n_pe // 2
    sym_lo = start
    sym_hi = 2 * center - start
    print(f"Symmetric band: PE lines {sym_lo} to {sym_hi}")

    # Step 1: phase estimate, from the symmetric band only ("ms(x,y)")
    k_sym = np.zeros_like(k_partial)
    k_sym[:, sym_lo : sym_hi + 1, :] = k_partial[:, sym_lo : sym_hi + 1, :]

    im_sym = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_sym, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    phase_est = np.angle(im_sym)


    weight = np.zeros(n_pe)
    half_width = (sym_hi - sym_lo) / 2.0
    y_idx = np.arange(sym_lo, sym_hi + 1)
    weight[sym_lo : sym_hi + 1] = 1.0 + (y_idx - center) / half_width
    weight[sym_hi + 1 :] = 2.0

    k_weighted = k_partial * weight[None, :, None]

    im_weighted = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(k_weighted, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )

    im_hd_coils = np.real(im_weighted * np.exp(-1j * phase_est)) # keep only the real part
    im_hd_sos = np.sqrt(np.sum(np.abs(im_hd_coils) ** 2, axis=2))

   
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# zero filled plots ---------------------------------------------
    axes[0, 0].imshow(full_truth, cmap="gray")
    axes[0, 0].set_title("Fully sampled (truth)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(im_zf_sos, cmap="gray")
    axes[0, 1].set_title(f"Zero-Filled PF")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.abs(full_truth - im_zf_sos), cmap="gray")
    axes[0, 2].set_title("Difference")
    axes[0, 2].axis("off")

    # Hooodyne plots --------------------------------------------------

    axes[1, 0].imshow(full_truth, cmap="gray")
    axes[1, 0].set_title("Fully sampled (truth)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(im_hd_sos, cmap="gray")
    axes[1, 1].set_title(f"Homodyne PF")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.abs(full_truth - im_hd_sos), cmap="gray")
    axes[1, 2].set_title("Difference")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()