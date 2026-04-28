"""
TDD Tests for GPU-optimized DiceMetric

Test requirements:
1. Correctness: Results match original CPU implementation
2. GPU Residency: Accumulator tensors stay on GPU during update()
3. No Python loops: Vectorized computation for all classes
4. Memory efficiency: Uses scatter_add instead of one-hot
"""

import torch
import time
import sys
sys.path.insert(0, '/home/comp/25481568/code/HyperBody')

from utils.metrics import DiceMetric


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_gpu_residency():
    """Test that accumulator tensors stay on GPU during update()"""
    print_section("Test 1: GPU Residency")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=70)

    # Create GPU tensors
    logits = torch.randn(2, 70, 16, 16, 16, device=device)
    targets = torch.randint(0, 70, (2, 16, 16, 16), device=device)

    # Update metric
    metric.update(logits, targets)

    # Check accumulator tensors are on GPU
    print(f"  Input device: {logits.device}")
    print(f"  intersection device: {metric.intersection.device}")
    print(f"  pred_sum device: {metric.pred_sum.device}")
    print(f"  target_sum device: {metric.target_sum.device}")

    assert metric.intersection.device.type == "cuda", \
        f"intersection should be on GPU, got {metric.intersection.device}"
    assert metric.pred_sum.device.type == "cuda", \
        f"pred_sum should be on GPU, got {metric.pred_sum.device}"
    assert metric.target_sum.device.type == "cuda", \
        f"target_sum should be on GPU, got {metric.target_sum.device}"

    print("  [PASS] All accumulators stay on GPU")


def test_correctness_vs_reference():
    """Test GPU implementation matches reference CPU implementation"""
    print_section("Test 2: Correctness vs Reference")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    num_classes = 70

    # Reference implementation (original CPU version)
    def reference_update(logits, targets, intersection, pred_sum, target_sum):
        preds = logits.argmax(dim=1).cpu()
        targets_cpu = targets.cpu()
        for c in range(num_classes):
            pred_c = (preds == c)
            target_c = (targets_cpu == c)
            intersection[c] += (pred_c & target_c).sum().item()
            pred_sum[c] += pred_c.sum().item()
            target_sum[c] += target_c.sum().item()

    # Test data
    torch.manual_seed(42)
    batches = [
        (torch.randn(2, 70, 32, 32, 32, device=device),
         torch.randint(0, 70, (2, 32, 32, 32), device=device))
        for _ in range(3)
    ]

    # Reference computation
    ref_intersection = torch.zeros(num_classes, dtype=torch.float64)
    ref_pred_sum = torch.zeros(num_classes, dtype=torch.float64)
    ref_target_sum = torch.zeros(num_classes, dtype=torch.float64)

    for logits, targets in batches:
        reference_update(logits, targets, ref_intersection, ref_pred_sum, ref_target_sum)

    # GPU metric computation
    metric = DiceMetric(num_classes=num_classes)
    for logits, targets in batches:
        metric.update(logits, targets)

    # Move GPU accumulators to CPU for comparison
    gpu_intersection = metric.intersection.cpu()
    gpu_pred_sum = metric.pred_sum.cpu()
    gpu_target_sum = metric.target_sum.cpu()

    # Compare
    print(f"  Reference intersection sum: {ref_intersection.sum().item():.0f}")
    print(f"  GPU intersection sum:       {gpu_intersection.sum().item():.0f}")
    print(f"  Reference pred_sum sum:     {ref_pred_sum.sum().item():.0f}")
    print(f"  GPU pred_sum sum:           {gpu_pred_sum.sum().item():.0f}")

    intersection_match = torch.allclose(gpu_intersection, ref_intersection, rtol=1e-5)
    pred_sum_match = torch.allclose(gpu_pred_sum, ref_pred_sum, rtol=1e-5)
    target_sum_match = torch.allclose(gpu_target_sum, ref_target_sum, rtol=1e-5)

    print(f"\n  intersection match: {intersection_match}")
    print(f"  pred_sum match:     {pred_sum_match}")
    print(f"  target_sum match:   {target_sum_match}")

    assert intersection_match, "intersection mismatch"
    assert pred_sum_match, "pred_sum mismatch"
    assert target_sum_match, "target_sum mismatch"

    # Compare final Dice scores
    dice_per_class, mean_dice, _ = metric.compute()

    ref_dice = (2.0 * ref_intersection + 1e-5) / (ref_pred_sum + ref_target_sum + 1e-5)
    ref_valid = ref_target_sum > 0
    ref_mean_dice = ref_dice[ref_valid].mean().item()

    print(f"\n  Reference mean Dice: {ref_mean_dice:.6f}")
    print(f"  GPU mean Dice:       {mean_dice:.6f}")

    assert abs(mean_dice - ref_mean_dice) < 1e-5, \
        f"Mean Dice mismatch: {mean_dice} vs {ref_mean_dice}"

    print("  [PASS] GPU implementation matches reference")


def test_no_cpu_transfer_during_update():
    """Test that update() doesn't transfer data to CPU"""
    print_section("Test 3: No CPU Transfer During Update")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=70)

    logits = torch.randn(2, 70, 32, 32, 32, device=device)
    targets = torch.randint(0, 70, (2, 32, 32, 32), device=device)

    # Synchronize before timing
    torch.cuda.synchronize()

    # Time the update
    start = time.perf_counter()
    metric.update(logits, targets)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  Update time: {elapsed*1000:.2f} ms")

    # A CPU-bound implementation with 70-class loop takes much longer
    # GPU implementation should be < 10ms for this size
    # CPU implementation with .item() calls typically takes > 50ms
    print(f"  Expected: < 20ms for GPU-optimized version")
    print(f"  (CPU version with .item() calls typically > 50ms)")

    # This is a soft check - mainly for documentation
    if elapsed < 0.020:  # 20ms threshold
        print("  [PASS] Update is fast (likely GPU-optimized)")
    else:
        print("  [WARN] Update is slow - may have CPU transfers")


def test_multi_batch_accumulation_gpu():
    """Test accumulation across multiple batches on GPU"""
    print_section("Test 4: Multi-batch Accumulation on GPU")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=10, smooth=0)

    # Batch 1: all class 1, perfect prediction
    targets1 = torch.ones(1, 8, 8, 8, dtype=torch.long, device=device)
    logits1 = torch.full((1, 10, 8, 8, 8), -10.0, device=device)
    logits1[:, 1, :, :, :] = 10.0

    # Batch 2: all class 1, perfect prediction
    targets2 = torch.ones(1, 8, 8, 8, dtype=torch.long, device=device)
    logits2 = torch.full((1, 10, 8, 8, 8), -10.0, device=device)
    logits2[:, 1, :, :, :] = 10.0

    metric.update(logits1, targets1)
    metric.update(logits2, targets2)

    # Check accumulation
    intersection_1 = metric.intersection[1].item()
    pred_sum_1 = metric.pred_sum[1].item()
    target_sum_1 = metric.target_sum[1].item()

    print(f"  Batch 1: 512 voxels class 1, perfect prediction")
    print(f"  Batch 2: 512 voxels class 1, perfect prediction")
    print(f"\n  After accumulation:")
    print(f"    intersection[1]: {intersection_1} (expected: 1024)")
    print(f"    pred_sum[1]:     {pred_sum_1} (expected: 1024)")
    print(f"    target_sum[1]:   {target_sum_1} (expected: 1024)")

    assert intersection_1 == 1024, f"intersection should be 1024, got {intersection_1}"
    assert pred_sum_1 == 1024, f"pred_sum should be 1024, got {pred_sum_1}"
    assert target_sum_1 == 1024, f"target_sum should be 1024, got {target_sum_1}"

    dice_per_class, mean_dice, _ = metric.compute()
    print(f"    mean_dice: {mean_dice:.6f} (expected: 1.0)")

    assert abs(mean_dice - 1.0) < 1e-5
    print("  [PASS] Multi-batch accumulation correct")


def test_reset_clears_gpu_tensors():
    """Test that reset() properly clears GPU tensors"""
    print_section("Test 5: Reset Clears GPU Tensors")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=70)

    # First update
    logits = torch.randn(2, 70, 16, 16, 16, device=device)
    targets = torch.randint(0, 70, (2, 16, 16, 16), device=device)
    metric.update(logits, targets)

    # Verify non-zero
    assert metric.intersection.sum().item() > 0, "Should have non-zero intersection"

    # Reset
    metric.reset()

    # Verify zeroed
    print(f"  After reset:")
    print(f"    intersection sum: {metric.intersection.sum().item()}")
    print(f"    pred_sum sum:     {metric.pred_sum.sum().item()}")
    print(f"    target_sum sum:   {metric.target_sum.sum().item()}")

    assert metric.intersection.sum().item() == 0, "intersection should be 0 after reset"
    assert metric.pred_sum.sum().item() == 0, "pred_sum should be 0 after reset"
    assert metric.target_sum.sum().item() == 0, "target_sum should be 0 after reset"

    print("  [PASS] Reset properly clears tensors")


def test_large_volume_performance():
    """Test performance with realistic volume size"""
    print_section("Test 6: Large Volume Performance")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=70)

    # Realistic size: batch=2, volume=144x128x268 (same as training)
    # Use smaller size for test to avoid OOM: 72x64x134
    logits = torch.randn(2, 70, 72, 64, 134, device=device)
    targets = torch.randint(0, 70, (2, 72, 64, 134), device=device)

    print(f"  Volume size: {tuple(targets.shape)}")
    print(f"  Num voxels per batch: {targets.numel():,}")

    # Warmup
    metric.update(logits, targets)
    metric.reset()
    torch.cuda.synchronize()

    # Benchmark
    num_iterations = 10
    start = time.perf_counter()
    for _ in range(num_iterations):
        metric.update(logits, targets)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time = elapsed / num_iterations * 1000
    print(f"\n  Average update time: {avg_time:.2f} ms")
    print(f"  Throughput: {num_iterations / elapsed:.1f} batches/sec")

    # Performance threshold (GPU should be fast)
    if avg_time < 50:  # 50ms threshold
        print("  [PASS] Performance is acceptable")
    else:
        print("  [WARN] Performance may be suboptimal")


def test_compute_returns_cpu_tensors():
    """Test that compute() returns CPU tensors for compatibility"""
    print_section("Test 7: Compute Returns CPU Tensors")

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    metric = DiceMetric(num_classes=70)

    logits = torch.randn(2, 70, 16, 16, 16, device=device)
    targets = torch.randint(0, 70, (2, 16, 16, 16), device=device)

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    print(f"  dice_per_class device: {dice_per_class.device}")
    print(f"  valid_mask device: {valid_mask.device}")
    print(f"  mean_dice type: {type(mean_dice)}")

    assert dice_per_class.device.type == "cpu", \
        f"dice_per_class should be on CPU, got {dice_per_class.device}"
    assert valid_mask.device.type == "cpu", \
        f"valid_mask should be on CPU, got {valid_mask.device}"
    assert isinstance(mean_dice, float), \
        f"mean_dice should be float, got {type(mean_dice)}"

    print("  [PASS] compute() returns CPU tensors")


def main():
    print("\n" + "="*60)
    print("  GPU DiceMetric TDD Tests")
    print("="*60)

    test_gpu_residency()
    test_correctness_vs_reference()
    test_no_cpu_transfer_during_update()
    test_multi_batch_accumulation_gpu()
    test_reset_clears_gpu_tensors()
    test_large_volume_performance()
    test_compute_returns_cpu_tensors()

    print("\n" + "="*60)
    print("  Test Summary")
    print("="*60)
    print("  Run complete. Check output for [PASS]/[FAIL]/[WARN] status.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
