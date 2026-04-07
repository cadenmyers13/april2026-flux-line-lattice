import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


class SAFClassifier:
    """Classify images as 'crystalline' or 'amorphous' using Symmetry Adapted
    Filters (SAFs).

    Parameters
    ----------
    resolution : float
        Angular resolution of SAF "fans" (controls smoothing in overlap score).
    n_folds : int
        Number of symmetry folds.
    threshold : float
        Threshold for classification decision (between 0 and 1).
    classification_type : str, optional
        Type of classification metric: "range" (default) or "max".
    cx : int, optional
        X-coordinate of the center for SAF and masking.
        Defaults to image center.
    cy : int, optional
        Y-coordinate of the center for SAF and masking.
        Defaults to image center.
    angle_step : float, optional
        Step size in degrees for rotating SAFs. Default is 1 degree.
    """

    def __init__(
        self,
        resolution,
        n_folds,
        threshold,
        classification_type="range",
        cx=None,
        cy=None,
        angle_step=1,
    ):
        self.resolution = resolution
        self.n_folds = n_folds
        self.k = self._find_k_value(resolution, n_folds)
        self.cx = cx
        self.cy = cy
        self.angle_step = angle_step
        self.threshold = threshold
        self.classification_type = classification_type

        self._saf_stack = None
        self._saf_shape = None

        # Storage for results
        self.filenames = []
        self.classifications = []
        self.classification_values = []
        self.overlap_scores = []

    @staticmethod
    def _find_k_value(resolution, n_folds):
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

    def symmetry_adapted_filter(self, offset, imshape):
        """Generate a symmetry adapted filter (SAF) for given offset and image
        shape.

        Parameters
        ----------
        offset : float
            Angular offset in radians.
        imshape : tuple
            Shape of the image (height, width).
        Returns
        -------
        saf : ndarray
            Generated SAF of shape `imshape`.
        """
        center_x = self.cx if self.cx is not None else imshape[1] // 2
        center_y = self.cy if self.cy is not None else imshape[0] // 2

        x_grid, y_grid = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        x_rel = x_grid - center_x
        y_rel = y_grid - center_y
        theta = np.arctan2(y_rel, x_rel)
        saf = np.abs(np.cos(0.5 * self.n_folds * (theta - offset))) ** self.k
        return saf

    def _generate_rotated_safs(self, imshape, angle_range_deg=None):
        """Generate a stack of rotated SAFs for the given image shape."""
        angle_step_rad = np.deg2rad(self.angle_step)
        if angle_range_deg is None:
            angles = np.arange(0, 2 * np.pi / self.n_folds, angle_step_rad)
        else:
            angles = np.deg2rad(angle_range_deg)
        saf_list = [self.symmetry_adapted_filter(angle, imshape) for angle in angles]
        self._saf_stack = np.stack(saf_list, axis=0)
        self._saf_shape = imshape

    def apply_annular_mask(
        self, images, inner_radius=None, outer_radius=None, cx=None, cy=None
    ):
        """Apply an annular mask to a stack of images, setting pixels within
        the annular region (between inner and outer radius) to zero.

        Parameters
        ----------
        images : ndarray
            Input image(s) to be masked. Can be 2D (height, width) or 3D
            (n_images, height, width).
        inner_radius : float, optional
            Inner radius of the annular mask. If None or 0, no inner boundary.
        outer_radius : float, optional
            Outer radius of the annular mask. If None, no outer boundary.
        cx : int, optional
            X-coordinate of the mask center. Defaults to image center.
        cy : int, optional
            Y-coordinate of the mask center. Defaults to image center.

        Returns
        -------
        masked_images : ndarray
            Masked image(s) with pixels within the annular region set to zero.
            Same shape as input.
        """
        if inner_radius is None and outer_radius is None:
            return images
        is_single_image = images.ndim == 2
        if is_single_image:
            images = images[np.newaxis, ...]
        cx = (
            cx
            if cx is not None
            else (self.cx if self.cx is not None else images.shape[2] // 2)
        )
        cy = (
            cy
            if cy is not None
            else (self.cy if self.cy is not None else images.shape[1] // 2)
        )
        y, x = np.ogrid[: images.shape[1], : images.shape[2]]
        distance_squared = (x - cx) ** 2 + (y - cy) ** 2
        if inner_radius is None or inner_radius == 0:
            mask = distance_squared <= outer_radius**2
        elif outer_radius is None:
            mask = distance_squared >= inner_radius**2
        else:
            mask = (distance_squared >= inner_radius**2) & (
                distance_squared <= outer_radius**2
            )
        masked_images = np.where(mask, 0, images)
        if is_single_image:
            return masked_images[0]
        return masked_images

    def calculate_single_overlap_score(self, image, angle_range_deg=None):
        """Compute dot product scores between image and rotated SAFs.
        Automatically generates SAF stack if missing or image shape differs.

        Parameters
        ----------
        image : ndarray
            Input image for overlap score calculation.
        angle_range_deg : list or ndarray, optional
            The specific angles in degrees to calculate the overlap score over.
            If None, uses default range based on n_folds and angle_step.

        Returns
        -------
        scores : ndarray
            Overlap scores for each rotated SAF.
        """
        if image.ndim == 3:
            image = image[..., 0]
        if (self._saf_stack is None) or (self._saf_shape != image.shape):
            self._generate_rotated_safs(image.shape, angle_range_deg=angle_range_deg)
        return np.tensordot(self._saf_stack, image, axes=([1, 2], [0, 1]))

    def calculate_normalized_overlap_scores(self, images):
        """Compute normalized overlap scores for a stack of images.

        Parameters
        ----------
        images : ndarray
            Stack of images with shape (n_images, height, width).

        Returns
        -------
        norm_scores : ndarray
            Normalized overlap scores with shape (n_images, n_angles).
        """
        scores_list = []
        for img in images:
            scores = self.calculate_single_overlap_score(img)
            scores_list.append(scores)
        scores_array = np.stack(scores_list, axis=0)
        norm_scores = self.normalize_min_max(scores_array)
        return norm_scores

    def classify_single_overlap_scores(self, score):
        """Classify a single set of overlap scores.

        Parameters
        ----------
        score : ndarray
            Overlap scores for a single image.

        Returns
        -------
        classification : str
            'crystalline' or 'amorphous' based on classification metric.
        classification_value : float
            Value used for classification decision.
        """
        if self.classification_type == "range":
            classification_value = np.max(score) - np.min(score)
        elif self.classification_type == "max":
            classification_value = np.max(score)
        else:
            raise ValueError("Invalid classification_type. Choose 'range' or 'max'.")

        if classification_value >= self.threshold:
            classification = "crystalline"
        else:
            classification = "amorphous"
        return classification, classification_value

    def _load_tiff_file(self, filepath):
        """Load a TIFF file and convert to numpy array.

        Parameters
        ----------
        filepath : str or Path
            Path to the TIFF file.

        Returns
        -------
        image : ndarray
            Loaded image as numpy array.
        """
        img = Image.open(filepath)
        return np.array(img)

    def classify_tiff_files(self, filepaths, inner_radius=None, outer_radius=None):
        """Classify a list of TIFF files as 'crystalline' or 'amorphous'.

        Overlap scores are calculated and normalized globally across all
        images. Then classification is performed based on the specified metric.

        Parameters
        ----------
        filepaths : list of str or Path
            List of paths to TIFF files.
        inner_radius : float, optional
            Inner radius for annular mask. If None or 0, no inner boundary.
        outer_radius : float, optional
            Outer radius for annular mask. If None, no outer boundary.

        Returns
        -------
        results : dict
            Dictionary containing results with keys:
            - 'filenames': list of filenames
            - 'classifications': list of classifications
            - 'classification_values': list of classification values
            - 'overlap_scores': list of overlap score arrays
        """
        # Convert to Path objects and store filenames
        paths = [Path(fp) for fp in filepaths]
        self.filenames = [p.name for p in paths]

        # Load all images
        images = [self._load_tiff_file(p) for p in paths]
        images_array = np.stack(images, axis=0)

        # Apply masking if requested (vectorized for entire stack)
        if inner_radius is not None or outer_radius is not None:
            images_array = self.apply_annular_mask(
                images_array, inner_radius, outer_radius
            )

        # Calculate normalized overlap scores
        normalized_overlap_scores = self.calculate_normalized_overlap_scores(
            images_array
        )

        # Classify each image
        self.classifications = []
        self.classification_values = []
        self.overlap_scores = []

        for overlap in normalized_overlap_scores:
            classification, classif_value = self.classify_single_overlap_scores(overlap)
            self.classifications.append(classification)
            self.classification_values.append(classif_value)
            self.overlap_scores.append(overlap)

        return {
            "filenames": self.filenames,
            "classifications": self.classifications,
            "classification_values": self.classification_values,
            "overlap_scores": self.overlap_scores,
        }

    def print_summary(self, verbose=False):
        """Print a summary of classification results."""
        if not self.filenames:
            print(
                "No classification results available. "
                "Run classify_tiff_files() first."
            )
            return

        print("=" * 80)
        print("SAF CLASSIFICATION SUMMARY")
        print("=" * 80)
        print("\nClassifier Parameters:")
        print(f"  Resolution: {self.resolution}°")
        print(f"  Number of folds: {self.n_folds}")
        print(f"  Threshold: {self.threshold}")
        print(f"  Classification type: {self.classification_type}")
        print(f"  Angle step: {self.angle_step}°")

        print(f"\nTotal images classified: {len(self.filenames)}")

        n_crystalline = sum(1 for c in self.classifications if c == "crystalline")
        n_amorphous = sum(1 for c in self.classifications if c == "amorphous")

        print(
            f"  Crystalline: {n_crystalline} "
            + f"({100*n_crystalline/len(self.filenames):.1f}%)"
        )
        print(
            f"  Amorphous: {n_amorphous} "
            + f"({100*n_amorphous/len(self.filenames):.1f}%)"
        )

        print("\nClassification values:")
        print(f"  Min: {min(self.classification_values):.4f}")
        print(f"  Max: {max(self.classification_values):.4f}")
        print(f"  Mean: {np.mean(self.classification_values):.4f}")
        print(f"  Std: {np.std(self.classification_values):.4f}")

        if verbose:
            print("\nIndividual Results:")
            print("-" * 80)
            print(f"{'Filename':<40} {'Classification':<15} {'Value':<10}")
            print("-" * 80)
            for fname, classif, value in zip(
                self.filenames,
                self.classifications,
                self.classification_values,
            ):
                print(f"{fname:<40} {classif:<15} {value:.4f}")

        print("=" * 80)

    def plot_masked_image(
        self, path, inner_radius=None, outer_radius=None, cx=None, cy=None
    ):
        """Plot original and masked image side by side.

        Parameters
        ----------
        path : str or Path
            Path to the TIFF file.
        inner_radius : float, optional
            Inner radius for annular mask. If None or 0, no inner boundary.
        outer_radius : float, optional
            Outer radius for annular mask. If None, no outer boundary.
        """
        image = self._load_tiff_file(path)
        masked_image = self.apply_annular_mask(
            image,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            cx=cx,
            cy=cy,
        )
        plt.imshow(masked_image)
        plt.title("Masked Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def plot_overlap_scores(self, figsize=(12, 5), alpha=0.3):
        """Plot overlap scores for all classified images with color coding.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (12, 5).
        alpha : float, optional
            Transparency of plot lines. Default is 0.3.
        """
        if not self.overlap_scores:
            print("No results to plot. Run classify_tiff_files() first.")
            return

        colors = [
            "blue" if cls == "crystalline" else "red" for cls in self.classifications
        ]

        plt.figure(figsize=figsize)
        for scores, color in zip(self.overlap_scores, colors):
            plt.plot(scores, color=color, alpha=alpha)

        plt.grid(True, alpha=0.3)
        plt.title("Overlap Scores for Each Image")
        plt.xlabel("Angle Index")
        plt.ylabel("Normalized Overlap Score")

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Crystalline"),
            Line2D([0], [0], color="red", lw=2, label="Amorphous"),
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()

    def plot_classification_scatter(self, figsize=(12, 5), alpha=0.7):
        """Scatter plot of classification values with color coding.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (12, 5).
        alpha : float, optional
            Transparency of scatter points. Default is 0.7.
        """
        if not self.classification_values:
            print("No results to plot. Run classify_tiff_files() first.")
            return

        colors = [
            "blue" if cls == "crystalline" else "red" for cls in self.classifications
        ]

        plt.figure(figsize=figsize)
        plt.scatter(
            range(len(self.classification_values)),
            self.classification_values,
            c=colors,
            alpha=alpha,
            s=100,
        )

        plt.axhline(
            y=self.threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {self.threshold}",
        )

        plt.xlabel("Image Index")
        plt.ylabel("Classification Value")
        plt.title(
            f"SAF Classification Results (type='{self.classification_type}')\n"
            f"Red=Amorphous, Blue=Crystalline"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_classification_histogram(self, figsize=(10, 6), bins=30):
        """Plot histogram of classification values with threshold line.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (10, 6).
        bins : int, optional
            Number of histogram bins. Default is 30.
        """
        if not self.classification_values:
            print("No results to plot. Run classify_tiff_files() first.")
            return

        # Separate values by classification
        crystalline_values = [
            val
            for val, cls in zip(self.classification_values, self.classifications)
            if cls == "crystalline"
        ]
        amorphous_values = [
            val
            for val, cls in zip(self.classification_values, self.classifications)
            if cls == "amorphous"
        ]

        plt.figure(figsize=figsize)

        plt.hist(
            crystalline_values,
            bins=bins,
            alpha=0.6,
            color="blue",
            label="Crystalline",
            edgecolor="black",
        )
        plt.hist(
            amorphous_values,
            bins=bins,
            alpha=0.6,
            color="red",
            label="Amorphous",
            edgecolor="black",
        )

        plt.axvline(
            x=self.threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold = {self.threshold}",
        )

        plt.xlabel("Classification Value")
        plt.ylabel("Count")
        plt.title(
            "Distribution of Classification Values "
            + f'(type="{self.classification_type}")'
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_all(self, figsize=(15, 12)):
        """Generate all plots in a single figure.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size as (width, height). Default is (15, 12).
        """
        if not self.overlap_scores:
            print("No results to plot. Run classify_tiff_files() first.")
            return

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Overlap scores
        ax1 = fig.add_subplot(gs[0, :])
        colors = [
            "blue" if cls == "crystalline" else "red" for cls in self.classifications
        ]
        for scores, color in zip(self.overlap_scores, colors):
            ax1.plot(scores, color=color, alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Overlap Scores for Each Image")
        ax1.set_xlabel("Angle Index")
        ax1.set_ylabel("Normalized Overlap Score")
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Crystalline"),
            Line2D([0], [0], color="red", lw=2, label="Amorphous"),
        ]
        ax1.legend(handles=legend_elements)

        # Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(
            range(len(self.classification_values)),
            self.classification_values,
            c=colors,
            alpha=0.7,
            s=100,
        )
        ax2.axhline(y=self.threshold, color="black", linestyle="--", linewidth=2)
        ax2.set_xlabel("Image Index")
        ax2.set_ylabel("Classification Value")
        ax2.set_title(f"Classification Scatter\n(threshold={self.threshold})")
        ax2.grid(True, alpha=0.3)

        # Histogram
        ax3 = fig.add_subplot(gs[1, 1])
        crystalline_values = [
            val
            for val, cls in zip(self.classification_values, self.classifications)
            if cls == "crystalline"
        ]
        amorphous_values = [
            val
            for val, cls in zip(self.classification_values, self.classifications)
            if cls == "amorphous"
        ]
        ax3.hist(
            crystalline_values,
            bins=20,
            alpha=0.6,
            color="blue",
            label="Crystalline",
            edgecolor="black",
        )
        ax3.hist(
            amorphous_values,
            bins=20,
            alpha=0.6,
            color="red",
            label="Amorphous",
            edgecolor="black",
        )
        ax3.axvline(x=self.threshold, color="black", linestyle="--", linewidth=2)
        ax3.set_xlabel("Classification Value")
        ax3.set_ylabel("Count")
        ax3.set_title("Value Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.suptitle(
            f"SAF Classification Results (n_folds={self.n_folds}, "
            f'type="{self.classification_type}")',
            fontsize=14,
            y=0.995,
        )

        plt.show()


# Example usage
if __name__ == "__main__":
    # Create classifier
    classifier = SAFClassifier(
        resolution=3.0,
        n_folds=1,
        threshold=0.25,
        classification_type="max",
        cx=71,
        cy=71,
    )

    # Classify TIFF files
    datadir = Path(__file__).parent.parent.parent / "data"
    tiff_files = sorted((datadir / "amorphous").glob("*.tif"))

    results = classifier.classify_tiff_files(tiff_files, outer_radius=7)

    # Print summary
    classifier.print_summary()

    # Generate plots
    # classifier.plot_overlap_scores()
    # classifier.plot_classification_scatter()
    # classifier.plot_classification_histogram()

    # Or plot everything at once
    classifier.plot_all()
