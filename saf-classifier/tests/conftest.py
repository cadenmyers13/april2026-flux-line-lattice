# Fixtures
import numpy as np
import pytest
from PIL import Image

from saf_classifier.run_saf_classifier import SAFClassifier


@pytest.fixture
def basic_classifier():
    """Create a basic classifier instance for testing."""
    return SAFClassifier(
        resolution=5.0, n_folds=4, threshold=0.5, classification_type="range"
    )


@pytest.fixture
def custom_center_classifier():
    """Create a classifier with custom center coordinates."""
    return SAFClassifier(
        resolution=5.0,
        n_folds=4,
        threshold=0.5,
        classification_type="range",
        cx=50,
        cy=50,
    )


@pytest.fixture
def sample_images():
    """Create sample test images."""
    # Create a crystalline-like pattern (high variation)
    crystalline = np.zeros((100, 100))
    for i in range(4):
        angle = i * np.pi / 2
        x = 50 + 30 * np.cos(angle)
        y = 50 + 30 * np.sin(angle)
        crystalline[int(y) - 5 : int(y) + 5, int(x) - 5 : int(x) + 5] = 1.0

    # Create an amorphous-like pattern (low variation)
    amorphous = np.random.rand(100, 100) * 0.1

    return np.stack([crystalline, amorphous], axis=0)


@pytest.fixture
def temp_tiff_files(tmp_path, sample_images):
    """Create temporary TIFF files for testing."""
    filepaths = []
    for i, img in enumerate(sample_images):
        filepath = tmp_path / f"test_image_{i}.tiff"
        Image.fromarray((img * 255).astype(np.uint8)).save(filepath)
        filepaths.append(filepath)
    return filepaths
