from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from saf_classifier.run_saf_classifier import SAFClassifier


# Test initialization


@pytest.mark.parametrize(
    "resolution, n_folds, expected_k",
    [
        # C1: Basic initialization with default classification_type
        (5.0, 4, 90.903),
        # C2: Different resolution and n_folds
        (3.0, 6, 112.253),
    ],
)
def test_initialization_k_value(resolution, n_folds, expected_k):
    """Test that classifier initializes with correct parameters."""
    classifier = SAFClassifier(
        resolution=resolution,
        n_folds=n_folds,
        threshold=1,
        classification_type="max",
    )

    assert np.isclose(classifier.k, expected_k, rtol=0.1)


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # C1: Simple array
        (np.array([1, 2, 3, 4, 5]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        # C2: Already normalized
        (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
        # C3: Negative values
        (np.array([-2, -1, 0, 1, 2]), np.array([0.0, 0.25, 0.5, 0.75, 1.0])),
        # C4: 2D array
        (np.array([[1, 2], [3, 4]]), np.array([[0.0, 1 / 3], [2 / 3, 1.0]])),
    ],
)
def test_normalize_min_max(input_array, expected_output):
    """Test min-max normalization."""
    actual_output = SAFClassifier.normalize_min_max(input_array)
    assert np.allclose(actual_output, expected_output)


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # C1: All same values should return zeros
        (np.array([5, 5, 5, 5]), np.array([0, 0, 0, 0])),
        # C2: 2D array with same values
        (np.array([[2, 2], [2, 2]]), np.array([[0, 0], [0, 0]])),
    ],
)
def test_normalize_min_max_constant_array(input_array, expected_output):
    """Test normalization of constant arrays."""
    actual_output = SAFClassifier.normalize_min_max(input_array)
    assert np.array_equal(actual_output, expected_output)


def test_normalize_min_max_empty_array():
    """Test that empty array raises ValueError."""
    with pytest.raises(ValueError, match="Cannot normalize an empty array."):
        SAFClassifier.normalize_min_max(np.array([]))


def test_symmetry_adapted_filter_custom_center(custom_center_classifier):
    """Test SAF with custom center coordinates."""
    imshape = (100, 100)
    saf = custom_center_classifier.symmetry_adapted_filter(0, imshape)
    assert saf.shape == imshape
    # Center should have value of 1 (cos(0)^k = 1)
    assert np.isclose(saf[50, 50], 1.0)


# Test apply_annular_mask


@pytest.mark.parametrize(
    "inner_radius, outer_radius, expected_center_value, expected_edge_value",
    [
        # C1: Mask center only (outer_radius only)
        (None, 20, 0, 100),
        # C2: Mask outer region (inner_radius only)
        (30, None, 100, 0),
        # C3: Annular mask
        (20, 30, 100, 100),
        # C4: No mask (both None)
        (None, None, 100, 100),
    ],
)
def test_apply_annular_mask_single_image(
    basic_classifier,
    inner_radius,
    outer_radius,
    expected_center_value,
    expected_edge_value,
):
    """Test annular masking on single image."""
    # Create test image with constant value
    image = np.ones((100, 100)) * 100
    masked = basic_classifier.apply_annular_mask(
        image, inner_radius=inner_radius, outer_radius=outer_radius
    )
    # Check center (50, 50)
    assert masked[50, 50] == expected_center_value
    # Check edge (5, 50)
    assert masked[5, 50] == expected_edge_value


def test_apply_annular_mask_stack(basic_classifier):
    """Test annular masking on image stack."""
    # Create stack of 3 images
    images = np.ones((3, 100, 100)) * 100
    masked = basic_classifier.apply_annular_mask(images, outer_radius=20)
    # Should return stack with same shape
    assert masked.shape == (3, 100, 100)
    # Center should be masked for all images
    assert np.all(masked[:, 50, 50] == 0)


def test_apply_annular_mask_custom_center(basic_classifier):
    """Test annular masking with custom center."""
    image = np.ones((100, 100)) * 100
    masked = basic_classifier.apply_annular_mask(
        image, outer_radius=10, cx=30, cy=40
    )
    # Center at (40, 30) should be masked
    assert masked[40, 30] == 0
    # Default center (50, 50) should not be masked
    assert masked[50, 50] == 100


def test_calculate_single_overlap_score_rgb_image(basic_classifier):
    """Test that RGB images are converted to grayscale."""
    image = np.random.rand(100, 100, 3)
    scores = basic_classifier.calculate_single_overlap_score(image)
    # Should still work and return scores
    assert scores.ndim == 1
    assert len(scores) > 0


def test_calculate_normalized_overlap_scores(basic_classifier, sample_images):
    """Test normalized overlap scores for image stack."""
    norm_scores = basic_classifier.calculate_normalized_overlap_scores(
        sample_images
    )
    # Should return normalized scores
    assert norm_scores.shape[0] == len(sample_images)
    assert norm_scores.ndim == 2
    # Scores should be normalized to [0, 1]
    assert np.all(norm_scores >= 0)
    assert np.all(norm_scores <= 1)
    # Should have at least one 0 and one 1 (due to normalization)
    assert np.min(norm_scores) == 0.0
    assert np.max(norm_scores) == 1.0


@pytest.mark.parametrize(
    "score_array, threshold, classification_type, expected_classification",
    [
        # C1: High range, above threshold -> crystalline
        (np.array([0.1, 0.5, 0.9, 0.2]), 0.5, "range", "crystalline"),
        # C2: Low range, below threshold -> amorphous
        (np.array([0.4, 0.45, 0.5, 0.42]), 0.5, "range", "amorphous"),
        # C3: High max, above threshold -> crystalline
        (np.array([0.1, 0.3, 0.8, 0.2]), 0.5, "max", "crystalline"),
        # C4: Low max, below threshold -> amorphous
        (np.array([0.1, 0.2, 0.3, 0.4]), 0.5, "max", "amorphous"),
        # C5: Exactly at threshold -> crystalline
        (np.array([0.0, 0.5, 1.0, 0.0]), 0.5, "range", "crystalline"),
    ],
)
def test_classify_single_overlap_scores(
    score_array, threshold, classification_type, expected_classification
):
    """Test classification of single overlap scores."""
    classifier = SAFClassifier(
        resolution=5.0,
        n_folds=4,
        threshold=threshold,
        classification_type=classification_type,
    )

    classification, value = classifier.classify_single_overlap_scores(
        score_array
    )

    assert classification == expected_classification
    if classification_type == "range":
        expected_value = np.max(score_array) - np.min(score_array)
    else:  # "max"
        expected_value = np.max(score_array)
    assert np.isclose(value, expected_value)


def test_classify_single_overlap_scores_invalid_type():
    """Test that invalid classification_type raises ValueError."""
    classifier = SAFClassifier(
        resolution=5.0, n_folds=4, threshold=0.5, classification_type="invalid"
    )
    score_array = np.array([0.1, 0.5, 0.9])
    with pytest.raises(
        ValueError,
        match="Invalid classification_type. Choose 'range' or 'max'.",
    ):
        classifier.classify_single_overlap_scores(score_array)


def test_classify_tiff_files_dict(basic_classifier, temp_tiff_files):
    """Test classification of TIFF files."""
    results = basic_classifier.classify_tiff_files(temp_tiff_files)
    # Check results dictionary structure
    assert "filenames" in results
    assert "classifications" in results
    assert "classification_values" in results
    assert "overlap_scores" in results
    # Check that results are stored in classifier
    assert len(basic_classifier.filenames) == len(temp_tiff_files)
    assert len(basic_classifier.classifications) == len(temp_tiff_files)
    assert len(basic_classifier.classification_values) == len(temp_tiff_files)
    assert len(basic_classifier.overlap_scores) == len(temp_tiff_files)
    # Check that filenames match
    expected_filenames = [Path(fp).name for fp in temp_tiff_files]
    assert basic_classifier.filenames == expected_filenames


def test_classify_tiff_files_with_mask(basic_classifier, temp_tiff_files):
    """Test classification with annular masking."""
    results = basic_classifier.classify_tiff_files(
        temp_tiff_files, outer_radius=30
    )
    # Should still produce results
    assert len(results["classifications"]) == len(temp_tiff_files)


def test_classify_tiff_files_annular_mask(basic_classifier, temp_tiff_files):
    """Test classification with full annular mask."""
    results = basic_classifier.classify_tiff_files(
        temp_tiff_files, inner_radius=20, outer_radius=40
    )
    # Should still produce results
    assert len(results["classifications"]) == len(temp_tiff_files)


# Test print_summary


def test_print_summary_no_results(basic_classifier, capsys):
    """Test print_summary with no classification results."""
    basic_classifier.print_summary()
    captured = capsys.readouterr()
    assert "No classification results available" in captured.out


def test_print_summary_with_results(basic_classifier, temp_tiff_files, capsys):
    """Test print_summary with classification results."""
    basic_classifier.classify_tiff_files(temp_tiff_files)
    basic_classifier.print_summary()
    captured = capsys.readouterr()
    # Check that summary contains expected information
    assert "SAF CLASSIFICATION SUMMARY" in captured.out
    assert "Classifier Parameters:" in captured.out
    assert f"Resolution: {basic_classifier.resolution}" in captured.out
    assert f"Number of folds: {basic_classifier.n_folds}" in captured.out
    assert f"Threshold: {basic_classifier.threshold}" in captured.out
    assert "Total images classified:" in captured.out
    assert "Crystalline:" in captured.out
    assert "Amorphous:" in captured.out


# Test edge cases and error handling


def test_classify_tiff_files_nonexistent_file(basic_classifier):
    """Test that nonexistent files raise appropriate error."""
    with pytest.raises(FileNotFoundError):
        basic_classifier.classify_tiff_files(["nonexistent_file.tif"])


@pytest.mark.parametrize(
    "n_images",
    [
        1,  # Single image
        5,  # Multiple images
    ],
)
def test_classify_different_batch_sizes(basic_classifier, tmp_path, n_images):
    """Test classification with different numbers of images."""
    # Create temporary TIFF files
    filepaths = []
    for i in range(n_images):
        img = np.random.rand(50, 50)
        filepath = tmp_path / f"test_{i}.tif"
        Image.fromarray((img * 255).astype(np.uint8)).save(filepath)
        filepaths.append(filepath)
    results = basic_classifier.classify_tiff_files(filepaths)
    assert len(results["classifications"]) == n_images
    assert len(results["filenames"]) == n_images


def test_classify_tiff_files(tmp_path):
    """Test complete workflow from initialization to results."""
    # Create test data
    crystalline = np.zeros((100, 100))
    for i in range(4):
        angle = i * np.pi / 2
        x = 50 + 30 * np.cos(angle)
        y = 50 + 30 * np.sin(angle)
        crystalline[int(y) - 5 : int(y) + 5, int(x) - 5 : int(x) + 5] = 255
    amorphous = np.random.rand(100, 100) * 50
    # Save as TIFF
    crystalline_path = tmp_path / "crystalline.tif"
    amorphous_path = tmp_path / "amorphous.tif"
    Image.fromarray(crystalline.astype(np.uint8)).save(crystalline_path)
    Image.fromarray(amorphous.astype(np.uint8)).save(amorphous_path)
    # Create classifier
    classifier = SAFClassifier(
        resolution=5.0, n_folds=4, threshold=0.5, classification_type="range"
    )
    # Classify
    results = classifier.classify_tiff_files(
        [crystalline_path, amorphous_path]
    )
    assert len(results["classifications"]) == 2
    assert results["filenames"][0] == "crystalline.tif"
    assert results["filenames"][1] == "amorphous.tif"
    assert results["classifications"][0] == "crystalline"
    assert results["classifications"][1] == "amorphous"
    # Crystalline should have higher classification value
    assert (
        results["classification_values"][0]
        > results["classification_values"][1]
    )


@pytest.mark.parametrize(
    "classification_type, threshold",
    [
        ("range", 0.3),
        ("range", 0.7),
        ("max", 0.5),
        ("max", 0.8),
    ],
)
def test_different_classification_parameters(
    tmp_path, classification_type, threshold
):
    """Test classification with different parameters."""
    # Create simple test image
    img = np.random.rand(50, 50)
    filepath = tmp_path / "test.tif"
    Image.fromarray((img * 255).astype(np.uint8)).save(filepath)

    classifier = SAFClassifier(
        resolution=5.0,
        n_folds=4,
        threshold=threshold,
        classification_type=classification_type,
    )
    results = classifier.classify_tiff_files([filepath])
    # Should complete without error
    assert len(results["classifications"]) == 1
    assert results["classifications"][0] in ["crystalline", "amorphous"]
