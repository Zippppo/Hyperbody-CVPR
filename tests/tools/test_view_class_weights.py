"""
Tests for view_class_weights.py

Tests:
1. Load and display weights from a .pt file
2. Handle different file formats
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_view_existing_weight_file():
    """Test viewing an existing class weight file"""
    from tools.view_class_weights import view_weights

    # Use an existing file
    weight_file = "checkpoints/class_weight_store/class_weights.pt"

    if Path(weight_file).exists():
        print(f"Testing with file: {weight_file}")
        view_weights(weight_file)
        print("\nTest PASSED: Successfully loaded and displayed weights")
    else:
        print(f"SKIP: File {weight_file} not found")


def test_view_with_mock_data():
    """Test viewing with mock weight data"""
    from tools.view_class_weights import view_weights

    # Create temporary weight file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        temp_path = f.name

    # Create mock data
    mock_data = {
        "weights": torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32),
        "num_classes": 5,
        "method": "test_method",
        "num_samples": 10,
    }
    torch.save(mock_data, temp_path)

    print(f"Testing with mock file: {temp_path}")
    view_weights(temp_path)

    # Cleanup
    Path(temp_path).unlink()
    print("\nTest PASSED: Successfully viewed mock weights")


def test_list_available_files():
    """Test listing available weight files"""
    from tools.view_class_weights import list_weight_files

    files = list_weight_files()
    print(f"Found {len(files)} weight files:")
    for f in files:
        print(f"  - {f}")

    print("\nTest PASSED: Successfully listed weight files")


if __name__ == "__main__":
    print("Running view_class_weights tests...\n")
    print("=" * 60)

    test_list_available_files()
    print("\n" + "=" * 60)

    test_view_with_mock_data()
    print("\n" + "=" * 60)

    test_view_existing_weight_file()
    print("\n" + "=" * 60)

    print("\nALL TESTS PASSED")
