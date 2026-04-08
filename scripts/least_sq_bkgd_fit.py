"""
This script computes the least-squares scaling factor between two 2D datasets stored in `.npz` files 
and visualizes the background-subtracted result. The scaling factor is calculated such that the 
background dataset is scaled to best match the sample dataset in a least-squares sense.

The script supports the following features:
- Loading 2D arrays from `.npz` files.
- Computing the scaling factor using least-squares minimization.
- Subtracting the scaled background from the sample dataset.
- Visualizing the background-subtracted result with optional log scaling and intensity limits.

Usage:
    python least_sq_bkgd_fit.py <sample.npz> <background.npz> [--zmin ZMIN] [--zmax ZMAX] [--log]

Arguments:
- `sample`: Path to the `.npz` file containing the sample dataset.
- `background`: Path to the `.npz` file containing the background dataset.
- `--zmin`: Minimum intensity value for plotting (optional).
- `--zmax`: Maximum intensity value for plotting (optional).
- `--log`: Use logarithmic scaling for the plot (optional).

Example:
    python least_sq_bkgd_fit.py sample.npz background.npz --zmin 0 --zmax 100 --log

Dependencies:
- numpy
- matplotlib
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_npz_array(path: Path):
    with np.load(path) as npz:
        # assume the main array is the first entry
        key = list(npz.keys())[0]
        return npz[key]


def compute_scale_least_squares(sample, bkgd, mask=None):
    if mask is not None:
        s = sample[mask]
        b = bkgd[mask]
    else:
        s = sample.ravel()
        b = bkgd.ravel()

    denom = np.sum(b * b)
    if denom == 0:
        raise ValueError("Background array is all zeros; cannot compute scale.")

    alpha = np.sum(s * b) / denom
    return alpha


def main():
    parser = argparse.ArgumentParser(
        description="Compute least-squares scaling factor between two 2D npz datasets and plot background-subtracted result."
    )
    parser.add_argument("sample", type=Path, help="Path to sample .npz file")
    parser.add_argument("background", type=Path, help="Path to background .npz file")
    parser.add_argument("--zmin", type=float, default=None, help="Minimum intensity for plotting")
    parser.add_argument("--zmax", type=float, default=None, help="Maximum intensity for plotting")
    parser.add_argument("--log", action="store_true", help="Use log scale for plotting")
    args = parser.parse_args()

    sample = load_npz_array(args.sample)
    bkgd = load_npz_array(args.background)

    if sample.shape != bkgd.shape:
        raise ValueError(f"Shape mismatch: {sample.shape} vs {bkgd.shape}")

    alpha = compute_scale_least_squares(sample, bkgd)
    corrected = sample - alpha * bkgd

    print(f"Scaling factor (alpha): {alpha:.6g}")

    plt.figure()

    if args.log:
        # avoid log of non-positive values
        plot_data = np.clip(corrected, a_min=1e-12, a_max=None)
        norm = LogNorm(vmin=args.zmin, vmax=args.zmax)
        plt.imshow(plot_data, origin="lower", norm=norm)
    else:
        plt.imshow(corrected, origin="lower", vmin=args.zmin, vmax=args.zmax)

    plt.colorbar()
    plt.title("Background Subtracted (Least Squares)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()