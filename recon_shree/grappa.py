import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os
import h5py


def grappa_kernel_calibration(acs, R, kernel_size=(2, 5)):
    """
    Calibrates GRAPPA kernels from ACS data. (Unchanged from your original --
    still R=2-only: predicts each missing line from the line directly below
    and directly above it.)
    acs: [RO, PE, Ch] - Fully sampled k-space center
    R: Acceleration factor
    kernel_size: (k_pe, k_ro)

    Returns: Weights [Ch_out, (k_pe * k_ro * Ch_in)]
    """
    n_ro, n_pe, n_ch = acs.shape
    k_pe, k_ro = kernel_size

    n_source_points = k_pe * k_ro * n_ch

    sources = []
    targets = []

    ro_rad = k_ro // 2

    for y in range(1, n_pe - 1):  # y is the target line index
        for x in range(ro_rad, n_ro - ro_rad):
            target_pt = acs[x, y, :]
            targets.append(target_pt)

            src_block = []

            blk1 = acs[x - ro_rad : x + ro_rad + 1, y - 1, :]
            src_block.append(blk1.flatten())

            blk2 = acs[x - ro_rad : x + ro_rad + 1, y + 1, :]
            src_block.append(blk2.flatten())

            sources.append(np.concatenate(src_block))

    S = np.array(sources)
    T = np.array(targets)

    coeffs, residuals, rank, s = np.linalg.lstsq(S, T, rcond=1e-5)

    return coeffs


def grappa_reconstruction():
    # --- Load ground-truth k-space from the .h5 file (replaces .dat loading) ---
    file_h5_in = "grappa_reconstructed_kspace.h5"
    file_h5_in = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data", file_h5_in)
    )
    print(f"Loading ground-truth k-space from {file_h5_in}...")
    with h5py.File(file_h5_in, "r") as hf:
        k_ground_truth = hf["kspace"][:]

    n_ro_full, n_pe_full, n_ch = k_ground_truth.shape
    print(f"Ground-truth k-space shape: {k_ground_truth.shape}")

    # --- Build the R=2 uniform undersampling mask, PLUS a fully-sampled ACS
    # band in the center -- same style as the intro_to_mri.ipynb grappa():
    # the ACS lines live INSIDE the same array that gets reconstructed,
    # instead of being a separate slice used only for training. ---
    acs_lines = 36  # matches the size your original pkt0 ACS block had
    acs_start = (n_pe_full - acs_lines) // 2
    acs_end = acs_start + acs_lines
    print(f"ACS band: PE lines {acs_start}:{acs_end} ({acs_lines} lines)")

    mask = np.zeros((n_ro_full, n_pe_full, n_ch), dtype=bool)
    mask[:, ::2, :] = True             # uniform R=2 stripe pattern
    mask[:, acs_start:acs_end, :] = True  # force the center band fully sampled

    print("Applying synthetic R=2 undersampling mask (with embedded ACS)...")
    k_full = np.zeros_like(k_ground_truth)
    k_full[mask] = k_ground_truth[mask]
    k_full_orig = k_full.copy()

    # --- ACS for calibration: pulled from k_full's OWN center band, which is
    # already fully sampled -- not a separate copy from k_ground_truth anymore.
    acs = k_full[:, acs_start:acs_end, :].copy()
    print(f"ACS Shape: {acs.shape}")

    # --- Calibration ---
    print("Calibrating GRAPPA weights...")
    k_pe = 2
    k_ro = 5
    weights = grappa_kernel_calibration(acs, R=3, kernel_size=(k_pe, k_ro))
    print(f"Weights trained. Shape: {weights.shape}")

    print(f"Output Grid prepared: {k_full.shape}")
    print("Synthesizing missing lines...")

    ro_rad = k_ro // 2

    for y in range(1, n_pe_full - 1, 2):
        for x in range(ro_rad, n_ro_full - ro_rad):
            blk1 = k_full[x - ro_rad : x + ro_rad + 1, y - 1, :]
            blk2 = k_full[x - ro_rad : x + ro_rad + 1, y + 1, :]
            src_vec = np.concatenate((blk1.flatten(), blk2.flatten()))
            recon_val = src_vec @ weights
            k_full[x, y, :] = recon_val

    k_full[mask] = k_full_orig[mask]

    file_h5_out = "grappa_R2_reconstruction_output.h5"
    file_h5_out = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data", file_h5_out)
    )

    with h5py.File(file_h5_out, "w") as hf:
        hf.create_dataset("kspace", data=k_full)
        hf.create_dataset("mask", data=mask)
        hf.create_dataset("acs", data=acs)
    print(f"Saved {file_h5_out}")

    plt.figure()
    plt.imshow(mask[:, :, 0], cmap="gray", aspect="auto")

    print("Reconstructing Image...")
    img_full = np.zeros_like(k_full)
    for c in range(n_ch):
        img_full[:, :, c] = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(k_full[:, :, c]))
        )

    sos = np.sqrt(np.sum(np.abs(img_full) ** 2, axis=2))

    plt.figure(figsize=(10, 10))
    plt.imshow(sos, cmap="gray")
    plt.title("GRAPPA Reconstruction (R=2, self-calibrated from ground truth)")
    plt.axis("off")

    out_file = "grappa_reconstruction.png"
    plt.savefig(out_file)
    print(f"Saved {out_file}")

    plt.show()


if __name__ == "__main__":
    grappa_reconstruction()