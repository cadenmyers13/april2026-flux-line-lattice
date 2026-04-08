#!/usr/bin/env python3

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