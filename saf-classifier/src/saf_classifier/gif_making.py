import tifffile
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from saf_classifier import SAFClassifier
from PIL import Image

# from bg_mpl_stylesheets.styles import all_styles
# plt.style.use(all_styles["bg-style"])


def get_tif_from_idx(idx, data_dir):
    tif_paths = sorted(data_dir.glob("*.tif"))
    tif_image = tifffile.imread(tif_paths[idx])
    return tif_image


def normalize_min_max(array):
    """Min max normalization of an array."""
    try:
        array = array.copy()
    except AttributeError:
        array = np.array(array, copy=True)
    if array.size == 0:
        raise ValueError("Cannot normalize an empty array.")
    array_min = np.min(array)
    array_max = np.max(array)
    if array_max == array_min:
        return np.zeros_like(array)
    norm_array = (array - array_min) / (array_max - array_min)
    return norm_array


def make_overlap_demo_gif(n_folds, resolution, array, save_location=None):
    array_norm = normalize_min_max(array)
    imshape = array_norm.shape
    sc = SAFClassifier(resolution, n_folds, threshold=0)
    angle_range = np.arange(0, 360 / n_folds, 1)
    overlap_scores = []
    gif_frames = []
    for ang in angle_range:
        angle_rad = np.deg2rad(ang)
        saf = sc.symmetry_adapted_filter(angle_rad, imshape)
        overlap_val = (array * saf).sum()
        overlap_scores.append(overlap_val)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(overlap_scores, "o")
        ax[0].set_ylim(2, 60)
        ax[0].set_xlabel("SAF orientation angle (degrees)")
        ax[0].set_ylabel("Overlap Score")

        ax[1].imshow(array + 1 / 4 * saf)
        ax[1].axis("off")

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        pil_frame = Image.fromarray(frame)
        gif_frames.append(pil_frame)
        plt.close(fig)
        gif_filename = (
            f"{n_folds}-nfolds-demo.gif"
            if save_location is None
            else save_location / f"{n_folds}-nfolds-demo.gif"
        )
        gif_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=gif_frames[1:],
            loop=0,
            duration=90,
        )
        print(f"gif saved to {gif_filename}")


def plot_normalized_overlap_score(
    n_folds, resolution, array, angle_range=None
):
    array_norm = normalize_min_max(array)
    imshape = array_norm.shape
    sc = SAFClassifier(resolution, n_folds, threshold=0)
    angle_range = (
        np.arange(0, 360 / n_folds, 1) if angle_range is None else angle_range
    )
    overlap_scores = []
    for ang in angle_range:
        angle_rad = np.deg2rad(ang)
        saf = sc.symmetry_adapted_filter(angle_rad, imshape)
        overlap_val = (array * saf).sum()
        overlap_scores.append(overlap_val)
    overlap_scores = np.array(overlap_scores)
    overlap_scores_normalized = overlap_scores - overlap_scores.min()
    plt.plot(
        angle_range, overlap_scores_normalized, label=f"n_folds={n_folds}"
    )


def main():
    data_dir = Path(".").parent / "data" / "random"
    for i in [5]:
        tif_im = normalize_min_max(get_tif_from_idx(i, data_dir))
        plt.imshow(tif_im)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"{data_dir.parent}/overlap-dp-ref.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.figure(figsize=(8, 6))
        for n_folds in [1, 2, 3, 4, 5, 6]:
            ang_range = np.arange(-360 / (2 * n_folds), 360 / (2 * n_folds), 1)
            # make_overlap_demo_gif(n_folds, 3, tif_im, data_dir)
            plot_normalized_overlap_score(
                n_folds, 3, tif_im, angle_range=ang_range
            )
        plt.legend()
        plt.xlabel("SAF orientation angle (degrees)")
        plt.ylabel(
            r"Normalized Overlap Score ($\mathcal{O}-\mathcal{O}_\text{min}$)"
        )
        plt.tight_layout()
        plt.savefig(
            f"{data_dir.parent}/overlaps-nfolds-plot.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    return


if __name__ == "__main__":
    main()
