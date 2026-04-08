import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider


def load_data(npz_path: Path):
    with np.load(npz_path) as npz:
        return npz["data"]


def calculate_azimuthal_sum(data, cx, cy, max_radius=None):
    """Compute azimuthal sum about (cx, cy). Returns (radii, summed_values)."""
    ny, nx = data.shape
    if max_radius is None:
        max_radius = int(np.hypot(max(cx, nx - cx), max(cy, ny - cy))) + 1

    y_idx, x_idx = np.indices((ny, nx))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    r_int = r.astype(int)

    radii = np.arange(0, max_radius)
    profile = np.zeros(len(radii))
    for i, ri in enumerate(radii):
        mask = r_int == ri
        if mask.any():
            profile[i] = np.sum(data[mask])

    return radii, profile


def interactive_plot(data_file: Path, bkgd: Path, log=True):
    data1 = load_data(data_file)
    data2 = load_data(bkgd)

    if data1.shape != data2.shape:
        raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}")

    ny, nx = data1.shape

    # Initial values
    scale0 = 0.0
    zmin0 = 1.0
    zmax0 = 10 ** 2

    # Initial azimuthal center (image center)
    center = (94, 117)# [nx // 2, ny // 2]  # [cx, cy]

    fig, (ax_2d, ax_az) = plt.subplots(1, 2, figsize=(13, 5))
    plt.subplots_adjust(bottom=0.35, wspace=0.35)

    def compute_diff(scale):
        return data1 - scale * data2

    def get_plot_data(d):
        return np.clip(d, 1e-12, None) if log else d

    diff = compute_diff(scale0)
    im = ax_2d.imshow(get_plot_data(diff), cmap="jet", origin="lower")
    cbar = plt.colorbar(im, ax=ax_2d)
    cbar.set_label("Difference (log)" if log else "Difference (linear)")
    ax_2d.set_title(f"Scale = {scale0:.3f}")

    # Red X marker for azimuthal center
    (center_marker,) = ax_2d.plot(
        center[0], center[1], "rx", markersize=10, markeredgewidth=2
    )

    # Azimuthal profile plot
    radii, profile = calculate_azimuthal_sum(diff, center[0], center[1])
    (az_line,) = ax_az.plot(radii, profile, color="steelblue", lw=1.5)
    ax_az.set_xlabel("Radius (pixels)")
    ax_az.set_ylabel("Azimuthal Sum")
    ax_az.set_title(f"Azimuthal Sum @ ({center[0]}, {center[1]})")
    ax_az.grid(True, alpha=0.3)

    # Sliders
    ax_scale = plt.axes([0.15, 0.22, 0.7, 0.03])
    ax_zmin  = plt.axes([0.15, 0.16, 0.7, 0.03])
    ax_zmax  = plt.axes([0.15, 0.10, 0.7, 0.03])

    slider_scale = Slider(ax_scale, "Scale", 0.0, 2.0, valinit=scale0, valstep=0.001)

    if log:
        slider_zmin = Slider(ax_zmin, "zmin", 0.001, 5,        valinit=zmin0)
        slider_zmax = Slider(ax_zmax, "zmax", 10,   5 * 10**2, valinit=zmax0)
    else:
        slider_zmin = Slider(ax_zmin, "zmin", -1000, 1000, valinit=0.0)
        slider_zmax = Slider(ax_zmax, "zmax",     0, 1000, valinit=800.0)

    def refresh_az(diff_data):
        cx, cy = center
        radii, profile = calculate_azimuthal_sum(diff_data, cx, cy)
        az_line.set_xdata(radii)
        az_line.set_ydata(profile)
        ax_az.relim()
        ax_az.autoscale_view()
        ax_az.set_title(f"Azimuthal Sum @ ({cx}, {cy})")

    def update(_):
        scale = slider_scale.val
        zmin  = slider_zmin.val
        zmax  = slider_zmax.val
        if zmax <= zmin:
            return

        diff = compute_diff(scale)
        plot_data = get_plot_data(diff)

        if log:
            im.set_norm(LogNorm(vmin=zmin, vmax=zmax))
        else:
            im.set_clim(vmin=zmin, vmax=zmax)

        im.set_data(plot_data)
        ax_2d.set_title(f"Scale = {scale:.3f}")
        refresh_az(diff)
        fig.canvas.draw_idle()

    def on_click(event):
        # Only respond to clicks on the 2D image axes
        if event.inaxes is not ax_2d:
            return
        if event.button != 1:  # left-click only
            return
        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        cx = np.clip(cx, 0, nx - 1)
        cy = np.clip(cy, 0, ny - 1)
        center[0], center[1] = cx, cy

        center_marker.set_xdata([cx])
        center_marker.set_ydata([cy])

        diff = compute_diff(slider_scale.val)
        refresh_az(diff)
        fig.canvas.draw_idle()

    slider_scale.on_changed(update)
    slider_zmin.on_changed(update)
    slider_zmax.on_changed(update)
    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive subtraction with sliders")
    parser.add_argument("data_file", help="Minuend (.npz)")
    parser.add_argument("bkgd",      help="Subtrahend (.npz)")
    parser.add_argument(
        "--linear", action="store_true", help="Use linear scale instead of log"
    )
    args = parser.parse_args()

    data_file = Path(args.data_file)
    bkgd      = Path(args.bkgd)

    data_file_metadata = np.load(data_file, allow_pickle=True)["metadata"].item()
    print(f"metadata for {data_file.stem}:")
    print("-" * 40)
    print(data_file_metadata)

    bkgd_metadata = np.load(bkgd, allow_pickle=True)["metadata"].item()
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