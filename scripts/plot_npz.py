import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_npz_file(npz_path: Path, zmin: float, zmax: float):
    with np.load(npz_path) as npz:
        data = npz["data"]

    plt.figure(figsize=(6, 5))
    data_norm = data + 1e-10  # Shift to avoid log(0)
    plt.imshow(
        data_norm, cmap="jet", origin="lower", norm=LogNorm(vmin=zmin, vmax=zmax)
    )
    plt.colorbar()
    plt.title(npz_path.name)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Load and plot 2D array from .npz file(s)"
    )
    parser.add_argument("files", nargs="+", help=".npz file(s) to plot")
    parser.add_argument(
        "--zmin", type=float, default=None, help="Minimum z-axis limit (log scale)"
    )
    parser.add_argument(
        "--zmax", type=float, default=None, help="Maximum z-axis limit (log scale)"
    )

    args = parser.parse_args()

    for file_str in args.files:
        npz_path = Path(file_str)
        if not npz_path.is_file():
            print(f"File not found: {npz_path}")
            continue

        plot_npz_file(npz_path, args.zmin, args.zmax)


if __name__ == "__main__":
    main()
