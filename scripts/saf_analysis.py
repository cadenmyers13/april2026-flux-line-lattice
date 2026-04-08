from saf_classifier.saf_classifier import SAFClassifier
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# from matplotlib.colors import LogNorm


def main():
    parser = argparse.ArgumentParser("Perform SAF analysis on a .npz file")
    parser.add_argument("npz_file", help="Path to the .npz file to analyze")
    args = parser.parse_args()

    npz_dir = Path(".").parent.parent / "data" / "npz"
    # npz_files = list(npz_dir.glob("*.npz"))
    # dp = np.load(npz_files[0])["data"]
    # numor = 157512  # 157722
    numor_npz_file = args.npz_file
    numor_dp = np.load(numor_npz_file)["data"]
    dp_shape = numor_dp.shape

    cx = 94
    cy = 118
    sc = SAFClassifier(
        resolution=10,
        n_folds=6,
        cx=cx,
        cy=cy,
        threshold=0,  # ignore this, this was for another analysis
    )
    masked_dp = sc.apply_annular_mask(
        numor_dp, inner_radius=0, outer_radius=30, cx=cx, cy=cy - 1
    )
    masked_norm_dp = sc.normalize_min_max(masked_dp)
    # saf = sc.symmetry_adapted_filter(0, dp_shape)
    plt.imshow(masked_norm_dp)  # , norm=LogNorm())
    plt.show()
    # angle_range = np.arange(-30, 31, 1)
    overlap = sc.calculate_single_overlap_score(
        masked_dp,
    )  # angle_range_deg=angle_range)
    # get args max index of the overlap
    overlap_max_idx = np.argmax(overlap)
    opt_saf = sc.symmetry_adapted_filter(overlap_max_idx, dp_shape)
    plt.plot(overlap)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Overlap Score")
    plt.show()

    print(f"phi optimal = {overlap_max_idx} deg")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(opt_saf + masked_norm_dp)
    ax[1].imshow(masked_norm_dp)
    plt.show()


if __name__ == "__main__":
    main()
