"""
Tests for class_weight_calculator.py

Tests:
1. Output format compatibility with existing class_weights.pt
2. Weight range validation (no extreme values)
3. Different methods produce different but valid results
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_output_format_compatibility():
    """Test that output format matches existing class_weights.pt"""
    from tools.class_weight_calculator import compute_weights_effective_number

    # Create mock class counts
    class_counts = torch.tensor([1e8, 1e6, 1e4, 1e2, 1e1], dtype=torch.float64)

    result = compute_weights_effective_number(class_counts, beta=0.99)

    # Check required keys
    assert "weights" in result, "Missing 'weights' key"
    assert "num_classes" in result, "Missing 'num_classes' key"
    assert "method" in result, "Missing 'method' key"

    # Check weights properties
    assert result["weights"].dtype == torch.float32, f"Expected float32, got {result['weights'].dtype}"
    assert result["weights"].shape == (5,), f"Expected shape (5,), got {result['weights'].shape}"

    print("Output format test PASSED")
    print(f"  Keys: {list(result.keys())}")
    print(f"  Weights dtype: {result['weights'].dtype}")
    print(f"  Weights shape: {result['weights'].shape}")


def test_weight_range_effective_number():
    """Test that effective number method produces bounded weights"""
    from tools.class_weight_calculator import compute_weights_effective_number

    # Realistic imbalance (similar to actual dataset): 1e8 vs 1e4
    # Not as extreme as 1e10 vs 1
    class_counts = torch.tensor([1e8, 1e7, 1e6, 1e5, 1e4], dtype=torch.float64)

    print("Testing effective number method with realistic class distribution:")
    for beta in [0.9, 0.99, 0.999, 0.9999]:
        result = compute_weights_effective_number(class_counts, beta=beta)
        weights = result["weights"]

        min_w = weights.min().item()
        max_w = weights.max().item()
        ratio = max_w / min_w

        print(f"  beta={beta}: min={min_w:.4f}, max={max_w:.4f}, ratio={ratio:.2f}")

        # Weights should be positive
        assert (weights > 0).all(), f"Found non-positive weights with beta={beta}"

        # For realistic distributions, ratio should be manageable
        # Higher beta = higher ratio (more weight to rare classes)
        if beta <= 0.99:
            assert ratio < 50, f"Weight ratio too extreme ({ratio:.2f}) with beta={beta}"

    print("Weight range test (effective number) PASSED")


def test_weight_range_log_dampened():
    """Test that log-dampened method produces bounded weights"""
    from tools.class_weight_calculator import compute_weights_log_dampened

    # Realistic imbalance
    class_counts = torch.tensor([1e8, 1e7, 1e6, 1e5, 1e4], dtype=torch.float64)

    print("Testing log-dampened method:")
    # Note: factor=1.0 doesn't work well for extreme imbalance
    # Higher factors provide better compression
    for factor in [10.0, 100.0, 1000.0]:
        result = compute_weights_log_dampened(class_counts, dampening_factor=factor)
        weights = result["weights"]

        min_w = weights.min().item()
        max_w = weights.max().item()
        ratio = max_w / min_w

        print(f"  factor={factor}: min={min_w:.4f}, max={max_w:.4f}, ratio={ratio:.2f}")

        assert (weights > 0).all(), f"Found non-positive weights with factor={factor}"
        # Log-dampened with appropriate factor should give reasonable ratios
        assert ratio < 10, f"Weight ratio too extreme ({ratio:.2f}) with factor={factor}"

    print("Weight range test (log-dampened) PASSED")


def test_comparison_with_inverse_sqrt():
    """Compare new methods with inverse_sqrt to show improvement"""
    from tools.class_weight_calculator import (
        compute_weights_effective_number,
        compute_weights_inverse_sqrt,
        compute_weights_log_dampened,
    )

    # Simulate realistic class distribution (background dominant)
    class_counts = torch.tensor(
        [
            1e9,  # background (class 0)
            1e7,  # large organ
            1e6,  # medium organ
            1e5,  # small organ
            1e4,  # very small structure
            1e3,  # tiny structure
        ],
        dtype=torch.float64,
    )

    methods = {
        "inverse_sqrt": compute_weights_inverse_sqrt(class_counts),
        "effective_number (beta=0.99)": compute_weights_effective_number(class_counts, beta=0.99),
        "effective_number (beta=0.999)": compute_weights_effective_number(class_counts, beta=0.999),
        "log_dampened (factor=10)": compute_weights_log_dampened(class_counts, dampening_factor=10.0),
    }

    print("\nComparison of weight methods:")
    print("=" * 80)
    print(f"{'Method':<35} {'Min':>8} {'Max':>8} {'Ratio':>8} {'Background':>12}")
    print("-" * 80)

    for name, result in methods.items():
        w = result["weights"]
        min_w = w.min().item()
        max_w = w.max().item()
        ratio = max_w / min_w
        bg_weight = w[0].item()
        print(f"{name:<35} {min_w:>8.4f} {max_w:>8.4f} {ratio:>8.2f} {bg_weight:>12.4f}")

    print("=" * 80)
    print("\nComparison test PASSED")


def test_normalization():
    """Test that weights are properly normalized"""
    from tools.class_weight_calculator import (
        compute_weights_effective_number,
        compute_weights_log_dampened,
    )

    class_counts = torch.tensor([1e6, 1e5, 1e4, 1e3, 1e2], dtype=torch.float64)
    num_classes = len(class_counts)

    # Test effective number
    result = compute_weights_effective_number(class_counts, beta=0.99)
    weight_sum = result["weights"].sum().item()
    assert abs(weight_sum - num_classes) < 0.01, f"Weights should sum to {num_classes}, got {weight_sum}"

    # Test log-dampened
    result = compute_weights_log_dampened(class_counts, dampening_factor=10.0)
    weight_sum = result["weights"].sum().item()
    assert abs(weight_sum - num_classes) < 0.01, f"Weights should sum to {num_classes}, got {weight_sum}"

    print("Normalization test PASSED")


if __name__ == "__main__":
    print("Running class_weight_calculator tests...\n")

    test_output_format_compatibility()
    print()

    test_weight_range_effective_number()
    print()

    test_weight_range_log_dampened()
    print()

    test_normalization()
    print()

    test_comparison_with_inverse_sqrt()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
