"""
Tests for MemoryEfficientDiceLoss vs DiceLoss (reference implementation).

Checkpoints:
1. Forward Consistency: |loss_old - loss_new| < 1e-6
2. Backward Consistency: gradient difference under tolerance
3. Boundary Test: all-zero and all-(C-1) targets
"""

import torch
import sys

sys.path.insert(0, "/home/comp/csrkzhu/code/RUN")

from models.losses import DiceLoss, MemoryEfficientDiceLoss


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Checkpoint 1: Forward Consistency
# ---------------------------------------------------------------------------
def test_forward_consistency_small():
    """Forward consistency on small tensor (easy to debug)."""
    print_section("Checkpoint 1a: Forward Consistency (small)")

    B, C, D, H, W = 2, 10, 8, 8, 8
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    ref_loss = DiceLoss(smooth=1.0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"Forward diff {diff} >= 1e-6"
    print("  PASSED")


def test_forward_consistency_realistic():
    """Forward consistency at realistic scale (C=70)."""
    print_section("Checkpoint 1b: Forward Consistency (realistic)")

    B, C, D, H, W = 2, 70, 32, 32, 32
    torch.manual_seed(123)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    ref_loss = DiceLoss(smooth=1.0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"Forward diff {diff} >= 1e-6"
    print("  PASSED")


def test_forward_consistency_various_smooth():
    """Forward consistency with different smooth values."""
    print_section("Checkpoint 1c: Forward Consistency (various smooth)")

    B, C, D, H, W = 2, 20, 16, 16, 16
    torch.manual_seed(99)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    for smooth in [0.0, 0.5, 1.0, 10.0]:
        ref_loss = DiceLoss(smooth=smooth)
        new_loss = MemoryEfficientDiceLoss(smooth=smooth)

        loss_ref = ref_loss(logits, targets)
        loss_new = new_loss(logits, targets)

        diff = (loss_ref - loss_new).abs().item()
        print(f"  smooth={smooth:<5} ref={loss_ref.item():.8f}  "
              f"new={loss_new.item():.8f}  diff={diff:.2e}")
        assert diff < 1e-6, f"smooth={smooth}: Forward diff {diff} >= 1e-6"

    print("  PASSED")


# ---------------------------------------------------------------------------
# Checkpoint 2: Backward Consistency
# ---------------------------------------------------------------------------
def test_backward_consistency():
    """Gradient difference between old and new implementations."""
    print_section("Checkpoint 2: Backward Consistency")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)

    # --- Reference ---
    logits_ref = torch.randn(B, C, D, H, W, requires_grad=True)
    targets = torch.randint(0, C, (B, D, H, W))

    ref_loss_fn = DiceLoss(smooth=1.0)
    loss_ref = ref_loss_fn(logits_ref, targets)
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # --- New ---
    logits_new = logits_ref.detach().clone().requires_grad_(True)
    new_loss_fn = MemoryEfficientDiceLoss(smooth=1.0)
    loss_new = new_loss_fn(logits_new, targets)
    loss_new.backward()
    grad_new = logits_new.grad.clone()

    # Statistics
    abs_diff = (grad_ref - grad_new).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_diff = (abs_diff / (grad_ref.abs() + 1e-12)).mean().item()

    print(f"  Grad shape:     {grad_ref.shape}")
    print(f"  Max abs diff:   {max_diff:.2e}")
    print(f"  Mean abs diff:  {mean_diff:.2e}")
    print(f"  Mean rel diff:  {rel_diff:.2e}")

    assert max_diff < 1e-5, f"Gradient max diff {max_diff} >= 1e-5"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Checkpoint 3: Boundary Tests
# ---------------------------------------------------------------------------
def test_boundary_all_zeros():
    """All targets are class 0."""
    print_section("Checkpoint 3a: Boundary - all targets = 0")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.zeros(B, D, H, W, dtype=torch.long)

    ref_loss = DiceLoss(smooth=1.0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"Boundary all-0 diff {diff} >= 1e-6"
    assert not torch.isnan(loss_new), "loss_new is NaN"
    assert not torch.isinf(loss_new), "loss_new is Inf"
    print("  PASSED")


def test_boundary_all_last_class():
    """All targets are class C-1."""
    print_section("Checkpoint 3b: Boundary - all targets = C-1")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.full((B, D, H, W), fill_value=C - 1, dtype=torch.long)

    ref_loss = DiceLoss(smooth=1.0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"Boundary all-(C-1) diff {diff} >= 1e-6"
    assert not torch.isnan(loss_new), "loss_new is NaN"
    assert not torch.isinf(loss_new), "loss_new is Inf"
    print("  PASSED")


def test_boundary_backward_all_zeros():
    """Gradient consistency when all targets are class 0."""
    print_section("Checkpoint 3c: Boundary Backward - all targets = 0")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    targets = torch.zeros(B, D, H, W, dtype=torch.long)

    logits_ref = torch.randn(B, C, D, H, W, requires_grad=True)
    loss_ref = DiceLoss(smooth=1.0)(logits_ref, targets)
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    logits_new = logits_ref.detach().clone().requires_grad_(True)
    loss_new = MemoryEfficientDiceLoss(smooth=1.0)(logits_new, targets)
    loss_new.backward()
    grad_new = logits_new.grad.clone()

    max_diff = (grad_ref - grad_new).abs().max().item()
    print(f"  Grad max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Boundary grad diff {max_diff} >= 1e-5"
    assert not torch.isnan(grad_new).any(), "grad_new has NaN"
    print("  PASSED")


def test_boundary_single_class():
    """Edge case: C=2 (minimum multi-class)."""
    print_section("Checkpoint 3d: Boundary - C=2 (minimum multi-class)")

    B, C, D, H, W = 1, 2, 8, 8, 8
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    ref_loss = DiceLoss(smooth=1.0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"C=2 diff {diff} >= 1e-6"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Checkpoint 4: ignore_index Tests
# ---------------------------------------------------------------------------
def test_forward_consistency_with_ignore_index():
    """DiceLoss and MemoryEfficientDiceLoss agree when ignore_index=0."""
    print_section("Checkpoint 4a: Forward Consistency with ignore_index=0")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    ref_loss = DiceLoss(smooth=1.0, ignore_index=0)
    new_loss = MemoryEfficientDiceLoss(smooth=1.0, ignore_index=0)

    loss_ref = ref_loss(logits, targets)
    loss_new = new_loss(logits, targets)

    diff = (loss_ref - loss_new).abs().item()
    print(f"  Reference loss: {loss_ref.item():.8f}")
    print(f"  New loss:       {loss_new.item():.8f}")
    print(f"  Abs diff:       {diff:.2e}")

    assert diff < 1e-6, f"Forward diff with ignore_index {diff} >= 1e-6"
    print("  PASSED")


def test_ignore_index_changes_loss():
    """ignore_index=0 produces a different loss value than no ignore."""
    print_section("Checkpoint 4b: ignore_index changes loss value")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    loss_no_ignore = MemoryEfficientDiceLoss(smooth=1.0)(logits, targets)
    loss_with_ignore = MemoryEfficientDiceLoss(smooth=1.0, ignore_index=0)(logits, targets)

    diff = (loss_no_ignore - loss_with_ignore).abs().item()
    print(f"  Loss (no ignore):   {loss_no_ignore.item():.8f}")
    print(f"  Loss (ignore=0):    {loss_with_ignore.item():.8f}")
    print(f"  Abs diff:           {diff:.2e}")

    assert diff > 1e-6, f"ignore_index should change loss, but diff={diff}"
    print("  PASSED")


def test_ignore_index_none_backward_compatible():
    """ignore_index=None gives exactly the same result as old behavior (diff=0)."""
    print_section("Checkpoint 4c: ignore_index=None backward compatible")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    targets = torch.randint(0, C, (B, D, H, W))

    loss_default = MemoryEfficientDiceLoss(smooth=1.0)(logits, targets)
    loss_none = MemoryEfficientDiceLoss(smooth=1.0, ignore_index=None)(logits, targets)

    diff = (loss_default - loss_none).abs().item()
    print(f"  Loss (default):       {loss_default.item():.8f}")
    print(f"  Loss (ignore=None):   {loss_none.item():.8f}")
    print(f"  Abs diff:             {diff:.2e}")

    assert diff == 0.0, f"ignore_index=None should be identical, but diff={diff}"
    print("  PASSED")


def test_backward_consistency_with_ignore_index():
    """Gradient consistency between DiceLoss and MemoryEfficientDiceLoss with ignore_index=0."""
    print_section("Checkpoint 4d: Backward Consistency with ignore_index=0")

    B, C, D, H, W = 2, 70, 16, 16, 16
    torch.manual_seed(42)
    targets = torch.randint(0, C, (B, D, H, W))

    # --- Reference ---
    logits_ref = torch.randn(B, C, D, H, W, requires_grad=True)
    loss_ref = DiceLoss(smooth=1.0, ignore_index=0)(logits_ref, targets)
    loss_ref.backward()
    grad_ref = logits_ref.grad.clone()

    # --- New ---
    logits_new = logits_ref.detach().clone().requires_grad_(True)
    loss_new = MemoryEfficientDiceLoss(smooth=1.0, ignore_index=0)(logits_new, targets)
    loss_new.backward()
    grad_new = logits_new.grad.clone()

    max_diff = (grad_ref - grad_new).abs().max().item()
    print(f"  Grad max diff: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Gradient max diff with ignore_index {max_diff} >= 1e-5"
    print("  PASSED")


def test_ignore_only_present_class():
    """When all targets are the ignored class, loss should not be NaN/Inf."""
    print_section("Checkpoint 4e: ignore only present class (no NaN/Inf)")

    B, C, D, H, W = 2, 10, 8, 8, 8
    torch.manual_seed(42)
    logits = torch.randn(B, C, D, H, W)
    # All targets are class 0, which is also the ignored class
    targets = torch.zeros(B, D, H, W, dtype=torch.long)

    loss_fn = MemoryEfficientDiceLoss(smooth=1.0, ignore_index=0)
    loss = loss_fn(logits, targets)

    print(f"  Loss value: {loss.item():.8f}")
    assert not torch.isnan(loss), "Loss is NaN when ignoring only present class"
    assert not torch.isinf(loss), "Loss is Inf when ignoring only present class"

    # Also check DiceLoss
    loss_ref_fn = DiceLoss(smooth=1.0, ignore_index=0)
    loss_ref = loss_ref_fn(logits, targets)
    print(f"  Ref loss:   {loss_ref.item():.8f}")
    assert not torch.isnan(loss_ref), "Ref loss is NaN"
    assert not torch.isinf(loss_ref), "Ref loss is Inf"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_forward_consistency_small,
        test_forward_consistency_realistic,
        test_forward_consistency_various_smooth,
        test_backward_consistency,
        test_boundary_all_zeros,
        test_boundary_all_last_class,
        test_boundary_backward_all_zeros,
        test_boundary_single_class,
        # ignore_index tests
        test_forward_consistency_with_ignore_index,
        test_ignore_index_changes_loss,
        test_ignore_index_none_backward_compatible,
        test_backward_consistency_with_ignore_index,
        test_ignore_only_present_class,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
