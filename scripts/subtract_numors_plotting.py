import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_data(npz_path: Path):
    with np.load(npz_path) as npz:
        return npz["data"]


def plot_two_npz(file1: Path, file2: Path, zmin: float, zmax: float):
    data1 = load_data(file1)
    data2 = load_data(file2)

    # Avoid log(0)
    data1 = data1 + 1e-10
    data2 = data2 + 1e-10

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(
        data1,
        cmap="jet",
        origin="lower",
        norm=LogNorm(vmin=zmin, vmax=zmax),
    )
    axes[0].set_title(file1.name)

    diff = data1 - data2
    diff = diff + 1e-10  # keep consistent with log scaling

    im2 = axes[1].imshow(
        diff,
        cmap="jet",
        origin="lower",
        norm=LogNorm(vmin=zmin, vmax=zmax),
    )
    axes[1].set_title(f"{file1.name} - {file2.name}")

    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot two .npz files: left = first, right = first-second"
    )
    parser.add_argument("file1", help="First .npz file")
    parser.add_argument("file2", help="Second .npz file")
    parser.add_argument(
        "--zmin", type=float, default=1, help="Minimum z-axis (log scale)"
    )
    parser.add_argument(
        "--zmax", type=float, default=None, help="Maximum z-axis (log scale)"
    )

    args = parser.parse_args()

    file1 = Path(args.file1)
    file2 = Path(args.file2)

    if not file1.is_file():
        raise FileNotFoundError(f"File not found: {file1}")
    if not file2.is_file():
        raise FileNotFoundError(f"File not found: {file2}")

    plot_two_npz(file1, file2, args.zmin, args.zmax)


if __name__ == "__main__":
    main()
