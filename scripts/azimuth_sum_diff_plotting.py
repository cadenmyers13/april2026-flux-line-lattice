"""azimuth_sum_diff_plotting.py.

Plot the azimuthal sum of (datafile - scale*background) for one or more data files.

Usage:
    python azimuth_sum_diff_plotting.py datafile1.npz df2.npz df3.npz -b bkgd_file.npz
    python azimuth_sum_diff_plotting.py datafile1.npz -b bkgd_file.npz --cx 100 --cy 80
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from background_subtraction_plotting import calculate_azimuthal_sum

DEFAULT_CX = 94
DEFAULT_CY = 117


def load_data(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as npz:
        return npz["data"]


def build_profiles(data_list, bkgd, scale, cx, cy):
    """Return a list of (radii, profile) tuples, one per data file."""
    profiles = []
    for data in data_list:
        diff = data - scale * bkgd
        radii, profile = calculate_azimuthal_sum(diff, cx, cy, max_radius=70)
        profiles.append((radii, profile))
    return profiles


def build_subtracted_avg_profiles(data_list, bkgd, scale, cx, cy):
    """Calculate the profiles, find their average, then subtract that average
    from each profile."""
    profiles = build_profiles(data_list, bkgd, scale, cx, cy)
    avg_profile = np.mean([p for _, p in profiles], axis=0)
    subtracted_profiles = []
    for radii, profile in profiles:
        subtracted_profiles.append((radii, profile - avg_profile))

    return subtracted_profiles


def interactive_az_plot(data_files, bkgd_file, cx, cy, use_subtracted=False):
    # ── Load arrays ───────────────────────────────────────────────────────────
    bkgd = load_data(bkgd_file)

    data_list = []
    for f in data_files:
        d = load_data(f)
        if d.shape != bkgd.shape:
            raise ValueError(f"Shape mismatch: {f.name} {d.shape} vs bkgd {bkgd.shape}")
        data_list.append(d)

    labels = [Path(f).stem for f in data_files]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    scale0 = 0.958463

    # ── Figure / axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(bottom=0.28)

    ax.set_xlabel("Radius (pixels)")
    ax.set_ylabel("Azimuthal Sum")
    ax.set_title(f"Azimuthal Sum (data − scale·bkgd), centre = ({cx}, {cy})")
    ax.grid(True, alpha=0.3)

    # ── Initial profiles ──────────────────────────────────────────────────────
    if use_subtracted:
        profiles = build_subtracted_avg_profiles(data_list, bkgd, scale0, cx, cy)
    else:
        profiles = build_profiles(data_list, bkgd, scale0, cx, cy)

    lines = []
    for i, ((radii, profile), label) in enumerate(zip(profiles, labels)):
        (ln,) = ax.plot(
            radii,
            profile,
            color=colors[i % len(colors)],
            lw=1.8,
            label=label,
        )
        lines.append(ln)

    ax.legend(loc="upper right")

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_scale = plt.axes([0.15, 0.12, 0.70, 0.04])
    slider_scale = Slider(
        ax_scale, "Bkgd Scale", 0.0, 2.0, valinit=scale0, valstep=0.001
    )

    ax_yspace = plt.axes([0.15, 0.05, 0.70, 0.04])
    slider_yspace = Slider(ax_yspace, "Y Spacing", 0.0, 500, valinit=0.0)

    def update(_):
        scale = slider_scale.val
        yspace = slider_yspace.val

        if use_subtracted:
            new_profiles = build_subtracted_avg_profiles(data_list, bkgd, scale, cx, cy)
        else:
            new_profiles = build_profiles(data_list, bkgd, scale, cx, cy)

        for i, (ln, (radii, profile)) in enumerate(zip(lines, new_profiles)):
            ln.set_xdata(radii)
            ln.set_ydata(profile + i * yspace)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider_scale.on_changed(update)
    slider_yspace.on_changed(update)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Azimuthal sum of (data − scale·bkgd) for multiple files."
    )
    parser.add_argument(
        "data_files",
        nargs="+",
        help="One or more data .npz files (minuends).",
    )
    parser.add_argument(
        "-b",
        "--bkgd",
        required=True,
        metavar="BKGD_FILE",
        help="Background .npz file (subtrahend).",
    )
    parser.add_argument(
        "--cx",
        type=int,
        default=DEFAULT_CX,
        help=f"Azimuthal centre x (default: {DEFAULT_CX})",
    )
    parser.add_argument(
        "--cy",
        type=int,
        default=DEFAULT_CY,
        help=f"Azimuthal centre y (default: {DEFAULT_CY})",
    )
    parser.add_argument(
        "--avg-sub",
        action="store_true",
        help="Use profiles with average profile subtracted",
    )

    args = parser.parse_args()

    data_files = [Path(f) for f in args.data_files]
    bkgd_file = Path(args.bkgd)

    for f in data_files + [bkgd_file]:
        if not f.is_file():
            print(f"File not found: {f}")
            return

    interactive_az_plot(
        data_files,
        bkgd_file,
        args.cx,
        args.cy,
        use_subtracted=args.avg_sub,
    )


if __name__ == "__main__":
    main()
