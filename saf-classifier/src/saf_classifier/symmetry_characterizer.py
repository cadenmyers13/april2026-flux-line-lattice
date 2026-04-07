import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from pathlib import Path
from PIL import Image


class SymmetryCharacterizer:
    """Characterize the symmetry of diffraction patterns using Symmetry Adapted
    Filters (SAFs) and various analysis methods.

    Parameters
    ----------
    resolution : float
        Angular resolution of SAF "fans" (controls smoothing in overlap score).
    cx : int, optional
        X-coordinate of the center for SAF. Defaults to image center.
    cy : int, optional
        Y-coordinate of the center for SAF. Defaults to image center.
    angle_step : float, optional
        Step size in degrees for rotating SAFs. Default is 1 degree.
    """

    def __init__(self, resolution=5.0, cx=None, cy=None, angle_step=1.0):
        self.resolution = resolution
        self.cx = cx
        self.cy = cy
        self.angle_step = angle_step

        # Will be populated by analysis methods
        self.results = {}

    @staticmethod
    def _find_k_value(resolution, n_folds):
        """Calculate k value for SAF generation."""
        res_rad = np.deg2rad(resolution)
        k = np.log(1 / 2) / (np.log(np.cos((n_folds / 4 * res_rad)) ** 2))
        return k

    @staticmethod
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

    def _symmetry_adapted_filter(self, offset, imshape, n_folds):
        """Generate a symmetry adapted filter (SAF) for given offset and image
        shape.

        Parameters
        ----------
        offset : float
            Angular offset in radians.
        imshape : tuple
            Shape of the image (height, width).
        n_folds : int
            Number of symmetry folds.

        Returns
        -------
        saf : ndarray
            Generated SAF of shape `imshape`.
        """
        center_x = self.cx if self.cx is not None else imshape[1] // 2
        center_y = self.cy if self.cy is not None else imshape[0] // 2

        k = self._find_k_value(self.resolution, n_folds)

        x_grid, y_grid = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        x_rel = x_grid - center_x
        y_rel = y_grid - center_y
        theta = np.arctan2(y_rel, x_rel)
        saf = np.abs(np.cos(0.5 * n_folds * (theta - offset))) ** k
        return saf

    def _generate_rotated_safs(self, imshape, n_folds):
        """Generate a stack of rotated SAFs for the given image shape."""
        angle_step_rad = np.deg2rad(self.angle_step)
        angles = np.arange(0, 2 * np.pi / n_folds, angle_step_rad)
        saf_list = [
            self._symmetry_adapted_filter(angle, imshape, n_folds) for angle in angles
        ]
        return np.stack(saf_list, axis=0), angles

    def _calculate_overlap_score(self, image, n_folds):
        """Compute overlap scores between image and rotated SAFs for given
        n_folds.

        Parameters
        ----------
        image : ndarray
            Input image for overlap score calculation.
        n_folds : int
            Number of symmetry folds.

        Returns
        -------
        scores : ndarray
            Overlap scores for each rotated SAF.
        angles : ndarray
            Corresponding angles in radians.
        """
        if image.ndim == 3:
            image = image[..., 0]

        saf_stack, angles = self._generate_rotated_safs(image.shape, n_folds)
        scores = np.tensordot(saf_stack, image, axes=([1, 2], [0, 1]))
        return scores, angles

    def apply_annular_mask(
        self, image, inner_radius=None, outer_radius=None, cx=None, cy=None
    ):
        """Apply an annular mask to an image.

        Parameters
        ----------
        image : ndarray
            Input image to be masked.
        inner_radius : float, optional
            Inner radius of the annular mask.
        outer_radius : float, optional
            Outer radius of the annular mask.
        cx : int, optional
            X-coordinate of the mask center.
        cy : int, optional
            Y-coordinate of the mask center.

        Returns
        -------
        masked_image : ndarray
            Masked image with pixels within the annular region set to zero.
        """
        if inner_radius is None and outer_radius is None:
            return image

        cx = (
            cx
            if cx is not None
            else (self.cx if self.cx is not None else image.shape[1] // 2)
        )
        cy = (
            cy
            if cy is not None
            else (self.cy if self.cy is not None else image.shape[0] // 2)
        )

        y, x = np.ogrid[: image.shape[0], : image.shape[1]]
        distance_squared = (x - cx) ** 2 + (y - cy) ** 2

        if inner_radius is None or inner_radius == 0:
            mask = distance_squared <= outer_radius**2
        elif outer_radius is None:
            mask = distance_squared >= inner_radius**2
        else:
            mask = (distance_squared >= inner_radius**2) & (
                distance_squared <= outer_radius**2
            )

        masked_image = np.where(mask, 0, image)
        return masked_image

    def _load_image(self, filepath):
        """Load an image file."""
        if isinstance(filepath, (str, Path)):
            img = Image.open(filepath)
            return np.array(img)
        else:
            return filepath

    def analyze_dominant_symmetry(
        self, image, fold_range=range(1, 13), mask_params=None
    ):
        """Find dominant symmetry by comparing max overlap scores across
        n_folds.

        Parameters
        ----------
        image : ndarray or str/Path
            Input image or path to image file.
        fold_range : range or list, optional
            Range of n_fold values to test. Default is 1-12.
        mask_params : dict, optional
            Parameters for annular masking: {'inner_radius': float, 'outer_radius': float}

        Returns
        -------
        results : dict
            Dictionary with keys for each n_fold containing:
            - 'max': maximum normalized overlap score
            - 'range': range of normalized overlap scores
            - 'mean': mean of normalized overlap scores
            - 'std': standard deviation of normalized overlap scores
            - 'scores': full array of normalized overlap scores
        """
        image = self._load_image(image)

        if mask_params is not None:
            image = self.apply_annular_mask(image, **mask_params)

        results = {}

        for n_fold in fold_range:
            scores, angles = self._calculate_overlap_score(image, n_fold)
            scores = self.normalize_min_max(scores)

            results[n_fold] = {
                "max": np.max(scores),
                "range": np.max(scores) - np.min(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores,
                "angles": angles,
            }

        # Find dominant symmetry
        max_values = {n_fold: res["max"] for n_fold, res in results.items()}
        dominant_fold = max(max_values, key=max_values.get)

        self.results["dominant_symmetry"] = {
            "results": results,
            "dominant_n_fold": dominant_fold,
            "confidence": max_values[dominant_fold],
        }

        return self.results["dominant_symmetry"]

    def analyze_fft_symmetry(self, image, fold_range=None, mask_params=None):
        """Identify symmetries using FFT of overlap scores from n_folds=1.

        Parameters
        ----------
        image : ndarray or str/Path
            Input image or path to image file.
        fold_range : range or list, optional
            If provided, limit detection to these n_fold values.
        mask_params : dict, optional
            Parameters for annular masking.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'frequencies': FFT frequencies
            - 'magnitude': FFT magnitude spectrum
            - 'n_folds': Detected n_fold values from peaks
            - 'peak_heights': Heights of detected peaks
            - 'dominant_n_fold': Strongest detected symmetry
            - 'overlap_scores': Original overlap scores
        """
        image = self._load_image(image)

        if mask_params is not None:
            image = self.apply_annular_mask(image, **mask_params)

        # Use n_folds=1 to capture full angular range
        scores, angles = self._calculate_overlap_score(image, n_folds=1)
        norm_scores = self.normalize_min_max(scores)

        # Perform FFT
        fft_result = np.fft.fft(norm_scores)
        fft_magnitude = np.abs(fft_result)
        frequencies = np.fft.fftfreq(len(norm_scores))

        # Only keep positive frequencies
        positive_freq_mask = frequencies > 0
        positive_freqs = frequencies[positive_freq_mask]
        positive_mags = fft_magnitude[positive_freq_mask]

        # Convert frequencies to n_fold values
        # frequency = n_fold / (2π)
        n_fold_values = positive_freqs * len(norm_scores)

        # Find peaks in FFT spectrum
        # Set minimum peak height as fraction of max magnitude
        min_peak_height = 0.1 * np.max(positive_mags)
        peak_indices, properties = find_peaks(
            positive_mags, height=min_peak_height, distance=1
        )

        if len(peak_indices) > 0:
            detected_n_folds = n_fold_values[peak_indices]
            peak_heights = positive_mags[peak_indices]

            # Round to nearest integer n_fold
            detected_n_folds_int = np.round(detected_n_folds).astype(int)

            # Filter by fold_range if provided
            if fold_range is not None:
                valid_mask = np.isin(detected_n_folds_int, list(fold_range))
                detected_n_folds_int = detected_n_folds_int[valid_mask]
                peak_heights = peak_heights[valid_mask]

            # Sort by peak height
            sorted_indices = np.argsort(peak_heights)[::-1]
            detected_n_folds_int = detected_n_folds_int[sorted_indices]
            peak_heights = peak_heights[sorted_indices]

            dominant_n_fold = (
                detected_n_folds_int[0] if len(detected_n_folds_int) > 0 else None
            )
        else:
            detected_n_folds_int = np.array([])
            peak_heights = np.array([])
            dominant_n_fold = None

        self.results["fft_symmetry"] = {
            "frequencies": positive_freqs,
            "magnitude": positive_mags,
            "n_fold_values": n_fold_values,
            "detected_n_folds": detected_n_folds_int,
            "peak_heights": peak_heights,
            "dominant_n_fold": dominant_n_fold,
            "overlap_scores": norm_scores,
            "angles": angles,
        }

        return self.results["fft_symmetry"]

    def analyze_autocorrelation_symmetry(self, image, mask_params=None):
        """Identify symmetry period using autocorrelation of overlap scores.

        Parameters
        ----------
        image : ndarray or str/Path
            Input image or path to image file.
        mask_params : dict, optional
            Parameters for annular masking.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'n_fold': Detected n_fold from first autocorrelation peak
            - 'autocorrelation': Full autocorrelation array
            - 'peaks': Indices of peaks in autocorrelation
            - 'peak_lags': Lag values at peaks (in angle steps)
            - 'overlap_scores': Original overlap scores
        """
        image = self._load_image(image)

        if mask_params is not None:
            image = self.apply_annular_mask(image, **mask_params)

        # Use n_folds=1 for full angular sweep
        scores, angles = self._calculate_overlap_score(image, n_folds=1)
        norm_scores = self.normalize_min_max(scores)

        # Calculate autocorrelation
        autocorr = np.correlate(norm_scores, norm_scores, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find peaks (excluding zero lag)
        peak_indices, _ = find_peaks(autocorr[1:], height=0.3)
        peak_indices = peak_indices + 1  # Adjust for slicing

        # Calculate n_fold from first peak
        if len(peak_indices) > 0:
            first_peak_lag = peak_indices[0]
            period_degrees = first_peak_lag * self.angle_step
            n_fold = 360.0 / period_degrees
        else:
            n_fold = None

        self.results["autocorrelation_symmetry"] = {
            "n_fold": n_fold,
            "autocorrelation": autocorr,
            "peaks": peak_indices,
            "peak_lags": peak_indices * self.angle_step,
            "overlap_scores": norm_scores,
            "angles": angles,
        }

        return self.results["autocorrelation_symmetry"]

    def create_symmetry_fingerprint(
        self, image, fold_range=range(1, 13), mask_params=None, normalize=True
    ):
        """Create a 2D symmetry fingerprint matrix.

        Each row represents normalized overlap scores for a different n_fold value.
        This creates a heatmap showing how the pattern responds to different symmetries.

        Parameters
        ----------
        image : ndarray or str/Path
            Input image or path to image file.
        fold_range : range or list, optional
            Range of n_fold values to include. Default is 1-12.
        mask_params : dict, optional
            Parameters for annular masking.
        normalize : bool, optional
            Whether to normalize each row. Default is True.

        Returns
        -------
        fingerprint : ndarray
            2D array of shape (len(fold_range), n_angles) containing overlap scores.
        fold_values : list
            List of n_fold values corresponding to rows.
        """
        image = self._load_image(image)

        if mask_params is not None:
            image = self.apply_annular_mask(image, **mask_params)

        fingerprint = []
        fold_values = list(fold_range)

        # Determine common angular resolution for all rows
        # Use the finest resolution (from n_folds=1)
        scores_ref, _ = self._calculate_overlap_score(image, n_folds=1)
        n_angles = len(scores_ref)

        for n_fold in fold_range:
            scores, angles = self._calculate_overlap_score(image, n_fold)

            if normalize:
                scores = self.normalize_min_max(scores)

            # Interpolate to common length
            if len(scores) != n_angles:
                x_old = np.linspace(0, 1, len(scores))
                x_new = np.linspace(0, 1, n_angles)
                f = interp1d(x_old, scores, kind="linear")
                scores = f(x_new)

            fingerprint.append(scores)

        fingerprint = np.array(fingerprint)

        self.results["symmetry_fingerprint"] = {
            "fingerprint": fingerprint,
            "fold_values": fold_values,
            "n_angles": n_angles,
        }

        return fingerprint, fold_values

    def comprehensive_analysis(self, image, fold_range=range(1, 13), mask_params=None):
        """Run all symmetry analysis methods and return comprehensive results.

        Parameters
        ----------
        image : ndarray or str/Path
            Input image or path to image file.
        fold_range : range or list, optional
            Range of n_fold values to test. Default is 1-12.
        mask_params : dict, optional
            Parameters for annular masking.

        Returns
        -------
        summary : dict
            Dictionary containing results from all analysis methods and a consensus.
        """
        # Run all analyses
        dominant_results = self.analyze_dominant_symmetry(
            image, fold_range, mask_params
        )
        fft_results = self.analyze_fft_symmetry(image, fold_range, mask_params)
        autocorr_results = self.analyze_autocorrelation_symmetry(image, mask_params)
        fingerprint, fold_values = self.create_symmetry_fingerprint(
            image, fold_range, mask_params
        )

        # Create consensus
        detected_symmetries = []

        if dominant_results["dominant_n_fold"] is not None:
            detected_symmetries.append(
                ("dominant_scan", dominant_results["dominant_n_fold"])
            )

        if fft_results["dominant_n_fold"] is not None:
            detected_symmetries.append(("fft", fft_results["dominant_n_fold"]))

        if autocorr_results["n_fold"] is not None:
            n_fold_rounded = int(round(autocorr_results["n_fold"]))
            detected_symmetries.append(("autocorrelation", n_fold_rounded))

        # Find consensus (most common detection)
        if detected_symmetries:
            from collections import Counter

            symmetry_counts = Counter([sym[1] for sym in detected_symmetries])
            consensus_n_fold = symmetry_counts.most_common(1)[0][0]
            consensus_count = symmetry_counts[consensus_n_fold]
        else:
            consensus_n_fold = None
            consensus_count = 0

        summary = {
            "consensus_n_fold": consensus_n_fold,
            "consensus_count": consensus_count,
            "detected_by_methods": detected_symmetries,
            "dominant_scan": dominant_results,
            "fft": fft_results,
            "autocorrelation": autocorr_results,
            "fingerprint": fingerprint,
            "fold_values": fold_values,
        }

        self.results["comprehensive"] = summary

        return summary

    # Visualization methods

    def plot_dominant_symmetry_scan(self, figsize=(12, 5)):
        """Plot results from dominant symmetry scan analysis."""
        if "dominant_symmetry" not in self.results:
            print(
                "No dominant symmetry results. Run analyze_dominant_symmetry() first."
            )
            return

        results = self.results["dominant_symmetry"]["results"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot max values for each n_fold
        n_folds = list(results.keys())
        max_values = [results[n]["max"] for n in n_folds]
        range_values = [results[n]["range"] for n in n_folds]

        ax1.plot(n_folds, max_values, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("n_fold")
        ax1.set_ylabel("Max Normalized Overlap Score")
        ax1.set_title("Dominant Symmetry Scan")
        ax1.grid(True, alpha=0.3)

        # Highlight dominant
        dominant = self.results["dominant_symmetry"]["dominant_n_fold"]
        ax1.axvline(
            dominant, color="red", linestyle="--", label=f"Dominant: {dominant}-fold"
        )
        ax1.legend()

        # Plot range values
        ax2.plot(n_folds, range_values, "s-", linewidth=2, markersize=8, color="green")
        ax2.set_xlabel("n_fold")
        ax2.set_ylabel("Range of Normalized Overlap Scores")
        ax2.set_title("Overlap Score Range by n_fold")
        ax2.grid(True, alpha=0.3)
        ax2.axvline(dominant, color="red", linestyle="--")

        plt.tight_layout()
        plt.show()

    def plot_fft_spectrum(self, figsize=(12, 5), max_n_fold=12):
        """Plot FFT spectrum showing detected symmetries."""
        if "fft_symmetry" not in self.results:
            print("No FFT results. Run analyze_fft_symmetry() first.")
            return

        fft_res = self.results["fft_symmetry"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot FFT magnitude spectrum
        n_fold_vals = fft_res["n_fold_values"]
        magnitude = fft_res["magnitude"]

        # Limit to reasonable n_fold range
        mask = n_fold_vals <= max_n_fold

        ax1.plot(n_fold_vals[mask], magnitude[mask], linewidth=2)
        ax1.set_xlabel("n_fold")
        ax1.set_ylabel("FFT Magnitude")
        ax1.set_title("FFT Spectrum of Overlap Scores")
        ax1.grid(True, alpha=0.3)

        # Mark detected peaks
        if len(fft_res["detected_n_folds"]) > 0:
            for n_fold, height in zip(
                fft_res["detected_n_folds"], fft_res["peak_heights"]
            ):
                if n_fold <= max_n_fold:
                    ax1.plot(n_fold, height, "ro", markersize=10)
                    ax1.text(n_fold, height, f"  {n_fold}", verticalalignment="bottom")

        # Plot original overlap scores
        angles_deg = np.rad2deg(fft_res["angles"])
        ax2.plot(angles_deg, fft_res["overlap_scores"], linewidth=2)
        ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("Normalized Overlap Score")
        ax2.set_title("Overlap Scores (n_folds=1)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_autocorrelation(self, figsize=(12, 5)):
        """Plot autocorrelation analysis results."""
        if "autocorrelation_symmetry" not in self.results:
            print(
                "No autocorrelation results. Run analyze_autocorrelation_symmetry() first."
            )
            return

        ac_res = self.results["autocorrelation_symmetry"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot autocorrelation
        lags = np.arange(len(ac_res["autocorrelation"])) * self.angle_step
        ax1.plot(lags, ac_res["autocorrelation"], linewidth=2)
        ax1.set_xlabel("Lag (degrees)")
        ax1.set_ylabel("Autocorrelation")
        ax1.set_title("Autocorrelation of Overlap Scores")
        ax1.grid(True, alpha=0.3)

        # Mark peaks
        if len(ac_res["peaks"]) > 0:
            peak_lags = ac_res["peak_lags"]
            peak_values = ac_res["autocorrelation"][ac_res["peaks"]]
            ax1.plot(peak_lags, peak_values, "ro", markersize=10)

            # Annotate first peak
            if ac_res["n_fold"] is not None:
                ax1.axvline(
                    peak_lags[0],
                    color="red",
                    linestyle="--",
                    label=f"Period = {peak_lags[0]:.1f}° ({ac_res['n_fold']:.1f}-fold)",
                )
                ax1.legend()

        # Plot original overlap scores
        angles_deg = np.rad2deg(ac_res["angles"])
        ax2.plot(angles_deg, ac_res["overlap_scores"], linewidth=2)
        ax2.set_xlabel("Angle (degrees)")
        ax2.set_ylabel("Normalized Overlap Score")
        ax2.set_title("Overlap Scores (n_folds=1)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_symmetry_fingerprint(self, figsize=(12, 8), cmap="viridis"):
        """Plot symmetry fingerprint as a heatmap."""
        if "symmetry_fingerprint" not in self.results:
            print("No fingerprint results. Run create_symmetry_fingerprint() first.")
            return

        fp_res = self.results["symmetry_fingerprint"]
        fingerprint = fp_res["fingerprint"]
        fold_values = fp_res["fold_values"]

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(
            fingerprint,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
            origin="lower",
        )

        # Set ticks
        ax.set_yticks(range(len(fold_values)))
        ax.set_yticklabels(fold_values)
        ax.set_ylabel("n_fold")
        ax.set_xlabel("Angle Index")
        ax.set_title("Symmetry Fingerprint Heatmap")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized Overlap Score")

        plt.tight_layout()
        plt.show()

    def plot_comprehensive(self, figsize=(16, 12)):
        """Plot all analysis results in a comprehensive figure."""
        if "comprehensive" not in self.results:
            print("No comprehensive results. Run comprehensive_analysis() first.")
            return

        comp_res = self.results["comprehensive"]

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Dominant symmetry scan
        ax1 = fig.add_subplot(gs[0, 0])
        dom_res = comp_res["dominant_scan"]["results"]
        n_folds = list(dom_res.keys())
        max_values = [dom_res[n]["max"] for n in n_folds]
        ax1.plot(n_folds, max_values, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("n_fold")
        ax1.set_ylabel("Max Score")
        ax1.set_title("Dominant Symmetry Scan")
        ax1.grid(True, alpha=0.3)
        dominant = comp_res["dominant_scan"]["dominant_n_fold"]
        ax1.axvline(dominant, color="red", linestyle="--", alpha=0.5)

        # 2. FFT spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        fft_res = comp_res["fft"]
        mask = fft_res["n_fold_values"] <= 12
        ax2.plot(
            fft_res["n_fold_values"][mask], fft_res["magnitude"][mask], linewidth=2
        )
        ax2.set_xlabel("n_fold")
        ax2.set_ylabel("FFT Magnitude")
        ax2.set_title("FFT Spectrum")
        ax2.grid(True, alpha=0.3)
        if len(fft_res["detected_n_folds"]) > 0:
            for n_fold, height in zip(
                fft_res["detected_n_folds"], fft_res["peak_heights"]
            ):
                if n_fold <= 12:
                    ax2.plot(n_fold, height, "ro", markersize=8)

        # 3. Autocorrelation
        ax3 = fig.add_subplot(gs[1, 0])
        ac_res = comp_res["autocorrelation"]
        lags = np.arange(len(ac_res["autocorrelation"])) * self.angle_step
        ax3.plot(lags, ac_res["autocorrelation"], linewidth=2)
        ax3.set_xlabel("Lag (degrees)")
        ax3.set_ylabel("Autocorrelation")
        ax3.set_title("Autocorrelation")
        ax3.grid(True, alpha=0.3)
        if len(ac_res["peaks"]) > 0:
            ax3.plot(
                ac_res["peak_lags"],
                ac_res["autocorrelation"][ac_res["peaks"]],
                "ro",
                markersize=8,
            )

        # 4. Overlap scores
        ax4 = fig.add_subplot(gs[1, 1])
        angles_deg = np.rad2deg(fft_res["angles"])
        ax4.plot(angles_deg, fft_res["overlap_scores"], linewidth=2)
        ax4.set_xlabel("Angle (degrees)")
        ax4.set_ylabel("Score")
        ax4.set_title("Overlap Scores (n_folds=1)")
        ax4.grid(True, alpha=0.3)

        # 5. Symmetry fingerprint
        ax5 = fig.add_subplot(gs[2, :])
        im = ax5.imshow(
            comp_res["fingerprint"],
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )
        ax5.set_yticks(range(len(comp_res["fold_values"])))
        ax5.set_yticklabels(comp_res["fold_values"])
        ax5.set_ylabel("n_fold")
        ax5.set_xlabel("Angle Index")
        ax5.set_title("Symmetry Fingerprint")
        plt.colorbar(im, ax=ax5, label="Normalized Score")

        # Add consensus text
        consensus_text = f"Consensus: {comp_res['consensus_n_fold']}-fold"
        if comp_res["consensus_count"] > 1:
            consensus_text += f" (detected by {comp_res['consensus_count']} methods)"

        fig.suptitle(consensus_text, fontsize=14, fontweight="bold")

        plt.show()

    def print_summary(self):
        """Print a text summary of all analysis results."""
        if "comprehensive" not in self.results:
            print("No results available. Run comprehensive_analysis() first.")
            return

        comp_res = self.results["comprehensive"]

        print("=" * 80)
        print("SYMMETRY CHARACTERIZATION SUMMARY")
        print("=" * 80)

        print(f"\nAnalysis Parameters:")
        print(f"  Resolution: {self.resolution}°")
        print(f"  Angle step: {self.angle_step}°")

        print(
            f"\n{'Consensus Result:':<30} {comp_res['consensus_n_fold']}-fold symmetry"
        )
        print(f"{'Agreement:':<30} {comp_res['consensus_count']}/3 methods")

        print("\nDetection by Method:")
        print("-" * 80)
        for method, n_fold in comp_res["detected_by_methods"]:
            print(f"  {method:<25} → {n_fold}-fold")

        print("\nMethod Details:")
        print("-" * 80)

        # Dominant scan
        dom = comp_res["dominant_scan"]
        print(f"\n1. Dominant Symmetry Scan:")
        print(f"   Detected: {dom['dominant_n_fold']}-fold")
        print(f"   Confidence: {dom['confidence']:.4f}")

        # FFT
        fft = comp_res["fft"]
        print(f"\n2. FFT Analysis:")
        if fft["dominant_n_fold"] is not None:
            print(f"   Primary: {fft['dominant_n_fold']}-fold")
            if len(fft["detected_n_folds"]) > 1:
                secondary = fft["detected_n_folds"][1:]
                print(f"   Secondary: {', '.join([f'{n}-fold' for n in secondary])}")
        else:
            print(f"   No clear peaks detected")

        # Autocorrelation
        ac = comp_res["autocorrelation"]
        print(f"\n3. Autocorrelation Analysis:")
        if ac["n_fold"] is not None:
            print(f"   Detected: {ac['n_fold']:.2f}-fold")
            print(
                f"   Period: {ac['peak_lags'][0]:.2f}° ({int(round(ac['n_fold']))}-fold)"
            )
        else:
            print(f"   No clear periodicity detected")

        print("\n" + "=" * 80)


def threshold_array(arr, threshold):
    """Convert array to binary based on a threshold.

    Parameters
    ----------
    arr : array-like
        Input array.
    threshold : float
        Threshold value.

    Returns
    -------
    np.ndarray
        Array with values > threshold set to 1 and values <= threshold set to 0.
    """
    arr = np.asarray(arr)
    return (arr > threshold).astype(int)


# Example usage
if __name__ == "__main__":
    # Create characterizer
    characterizer = SymmetryCharacterizer(resolution=5.0, angle_step=1.0)

    # Create a test image with 4-fold symmetry
    datadir = Path(__file__).parent.parent.parent / "data"
    amorphous_dir = datadir / "amorphous"
    tifs_file = list(amorphous_dir.glob("*.tif"))[20]
    img = characterizer._load_image(tifs_file)
    img = characterizer.apply_annular_mask(img, inner_radius=0, outer_radius=10)
    # img = threshold_array(img, threshold=20)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    # Run comprehensive analysis
    results = characterizer.comprehensive_analysis(img, fold_range=range(1, 7))

    # Print summary
    characterizer.print_summary()

    # Plot results
    characterizer.plot_comprehensive()
