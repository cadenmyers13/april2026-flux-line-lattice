import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider


def load_data(npz_path: Path):
    with np.load(npz_path) as npz:
        return npz["data"]


def interactive_plot(data_file: Path, bkgd: Path, log=True):
    data1 = load_data(data_file)
    data2 = load_data(bkgd)

    if data1.shape != data2.shape:
        raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}")

    # Initial values
    scale0 = 15.0
    zmin0 = 1
    zmax0 = 10**2

    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(bottom=0.35)

    def compute_diff(scale):
        return data1 - scale * data2

    def get_plot_data(d):
        if log:
            return np.clip(d, 1e-12, None)
        return d

    diff = compute_diff(scale0)
    im = ax.imshow(get_plot_data(diff), cmap="jet", origin="lower")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Difference (log)" if log else "Difference (linear)")

    ax.set_title(f"Scale = {scale0:.3f}")

    # Sliders
    ax_scale = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_zmin  = plt.axes([0.15, 0.16, 0.7, 0.03])
    ax_zmax  = plt.axes([0.15, 0.10, 0.7, 0.03])

    slider_scale = Slider(ax_scale, "Scale", 0.0, 20.0, valinit=scale0, valstep=0.01)

    if log:
        slider_zmin = Slider(ax_zmin, "zmin", 0.001, 5, valinit=zmin0)
        slider_zmax = Slider(ax_zmax, "zmax", 10, 5*10**2, valinit=zmax0)
    else:
        slider_zmin = Slider(ax_zmin, "zmin", -1.0, 1.0, valinit=0.0)
        slider_zmax = Slider(ax_zmax, "zmax", 0.0, 10.0, valinit=1.0)

    def update(_):
        scale = slider_scale.val
        zmin = slider_zmin.val
        zmax = slider_zmax.val

        if zmax <= zmin:
            return

        diff = compute_diff(scale)
        plot_data = get_plot_data(diff)

        if log:
            im.set_norm(LogNorm(vmin=zmin, vmax=zmax))
        else:
            im.set_clim(vmin=zmin, vmax=zmax)

        im.set_data(plot_data)
        ax.set_title(f"Scale = {scale:.3f}")

        fig.canvas.draw_idle()

    slider_scale.on_changed(update)
    slider_zmin.on_changed(update)
    slider_zmax.on_changed(update)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive subtraction with sliders")

    parser.add_argument("data_file", help="Minuend (.npz)")
    parser.add_argument("bkgd", help="Subtrahend (.npz)")

    # NEW: default is log, flag switches to linear
    parser.add_argument("--linear", action="store_true",
                        help="Use linear scale instead of log")

    args = parser.parse_args()

    data_file = Path(args.data_file)
    bkgd = Path(args.bkgd)
    data_file_metadata = np.load(data_file)["comments"]
    print(f"metadata for{data_file.stem}:")
    print("-" * 40)
    print(data_file_metadata)
    bkgd_metadata = np.load(bkgd)["comments"]
    print(f"metadata for {bkgd.stem}:")
    print("-" * 40)
    print(bkgd_metadata)
    if not data_file.is_file():
        print(f"File not found: {data_file}")
        return
    if not bkgd.is_file():
        print(f"File not found: {bkgd}")
        return

    interactive_plot(data_file, bkgd, log=not args.linear)


if __name__ == "__main__":
    main()