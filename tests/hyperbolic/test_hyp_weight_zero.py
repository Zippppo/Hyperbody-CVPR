"""
Test that setting hyp_weight=0 completely eliminates gradient flow to hyperbolic parameters.

This test verifies that when hyp_weight=0:
1. Hyperbolic parameters (hyp_head, label_emb) receive NO gradients
2. Segmentation parameters (unet) still receive gradients normally
3. The training is equivalent to pure CE + Dice loss
"""
import torch
import torch.nn as nn
import pytest


class TestHypWeightZero:
    """Test gradient flow when hyp_weight=0."""

    @pytest.fixture
    def model_and_data(self):
        """Create a small BodyNet model and dummy data for testing."""
        from models.body_net import BodyNet
        from data.organ_hierarchy import load_organ_hierarchy
        import json

        # Load class depths for label embedding initialization
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

        # Create small model for fast testing
        model = BodyNet(
            in_channels=1,
            num_classes=70,
            base_channels=16,  # Small for speed
            embed_dim=16,
            class_depths=class_depths,
        )

        # Create dummy input and target
        x = torch.randn(1, 1, 16, 12, 16)
        target = torch.randint(0, 70, (1, 16, 12, 16))

        return model, x, target

    @pytest.fixture
    def losses(self):
        """Create loss functions."""
        from models.losses import CombinedLoss
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        seg_criterion = CombinedLoss(num_classes=70)
        hyp_criterion = LorentzRankingLoss(margin=0.1, curv=1.0)

        return seg_criterion, hyp_criterion

    def test_hyp_weight_zero_no_gradient_to_hyp_head(self, model_and_data, losses):
        """When hyp_weight=0, hyp_head parameters should have NO gradients."""
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.0

        # Forward pass
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        # Backward pass
        total_loss.backward()

        # Check hyp_head parameters have NO gradients (or zero gradients)
        for name, param in model.hyp_head.named_parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0), \
                    f"hyp_head.{name} should have zero gradient when hyp_weight=0, " \
                    f"but got non-zero gradient with max={param.grad.abs().max().item()}"

    def test_hyp_weight_zero_no_gradient_to_label_emb(self, model_and_data, losses):
        """When hyp_weight=0, label_emb parameters should have NO gradients."""
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.0

        # Forward pass
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        # Backward pass
        total_loss.backward()

        # Check label_emb parameters have NO gradients (or zero gradients)
        for name, param in model.label_emb.named_parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0), \
                    f"label_emb.{name} should have zero gradient when hyp_weight=0, " \
                    f"but got non-zero gradient with max={param.grad.abs().max().item()}"

    def test_hyp_weight_zero_unet_still_has_gradients(self, model_and_data, losses):
        """When hyp_weight=0, UNet parameters should still receive gradients from seg_loss."""
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.0

        # Forward pass
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        # Backward pass
        total_loss.backward()

        # Check UNet parameters have non-zero gradients
        has_nonzero_grad = False
        for name, param in model.unet.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, \
            "UNet should have non-zero gradients from seg_loss even when hyp_weight=0"

    def test_hyp_weight_nonzero_hyp_head_has_gradients(self, model_and_data, losses):
        """When hyp_weight>0, hyp_head parameters should receive gradients."""
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.05  # Non-zero weight

        # Forward pass
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        # Backward pass
        total_loss.backward()

        # Check hyp_head parameters have non-zero gradients
        has_nonzero_grad = False
        for name, param in model.hyp_head.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, \
            "hyp_head should have non-zero gradients when hyp_weight > 0"

    def test_hyp_weight_nonzero_label_emb_has_gradients(self, model_and_data, losses):
        """When hyp_weight>0, label_emb parameters should receive gradients."""
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.05  # Non-zero weight

        # Forward pass
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss

        # Backward pass
        total_loss.backward()

        # Check label_emb parameters have non-zero gradients
        has_nonzero_grad = False
        for name, param in model.label_emb.named_parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, \
            "label_emb should have non-zero gradients when hyp_weight > 0"

    def test_gradient_equivalence_with_pure_seg_loss(self, model_and_data, losses):
        """
        Verify that gradients with hyp_weight=0 are IDENTICAL to gradients
        computed using only seg_loss (without computing hyp_loss at all).
        """
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses

        # --- Method 1: hyp_weight=0 ---
        model.zero_grad()
        logits1, voxel_emb1, label_emb1 = model(x)
        seg_loss1 = seg_criterion(logits1, target)
        hyp_loss1 = hyp_criterion(voxel_emb1, target, label_emb1)
        total_loss1 = seg_loss1 + 0.0 * hyp_loss1
        total_loss1.backward()

        # Save UNet gradients
        unet_grads_method1 = {}
        for name, param in model.unet.named_parameters():
            if param.grad is not None:
                unet_grads_method1[name] = param.grad.clone()

        # --- Method 2: Pure seg_loss only ---
        model.zero_grad()
        logits2, _, _ = model(x)
        seg_loss2 = seg_criterion(logits2, target)
        seg_loss2.backward()

        # Compare UNet gradients
        for name, param in model.unet.named_parameters():
            if param.grad is not None and name in unet_grads_method1:
                grad1 = unet_grads_method1[name]
                grad2 = param.grad

                # Check gradients are identical (or very close due to floating point)
                assert torch.allclose(grad1, grad2, rtol=1e-5, atol=1e-7), \
                    f"UNet gradient for {name} differs between hyp_weight=0 and pure seg_loss. " \
                    f"Max diff: {(grad1 - grad2).abs().max().item()}"

        print("SUCCESS: Gradients with hyp_weight=0 are identical to pure seg_loss gradients")

    def test_parameter_update_equivalence(self, model_and_data, losses):
        """
        Verify that after one optimizer step with hyp_weight=0,
        hyperbolic parameters remain unchanged while UNet parameters change.
        """
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses
        hyp_weight = 0.0

        # Save initial parameters
        hyp_head_init = {name: param.clone() for name, param in model.hyp_head.named_parameters()}
        label_emb_init = {name: param.clone() for name, param in model.label_emb.named_parameters()}
        unet_init = {name: param.clone() for name, param in model.unet.named_parameters()}

        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward + backward + step
        optimizer.zero_grad()
        logits, voxel_emb, label_emb = model(x)
        seg_loss = seg_criterion(logits, target)
        hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
        total_loss = seg_loss + hyp_weight * hyp_loss
        total_loss.backward()
        optimizer.step()

        # Check hyp_head parameters are UNCHANGED
        for name, param in model.hyp_head.named_parameters():
            assert torch.equal(param, hyp_head_init[name]), \
                f"hyp_head.{name} should be unchanged when hyp_weight=0"

        # Check label_emb parameters are UNCHANGED
        for name, param in model.label_emb.named_parameters():
            assert torch.equal(param, label_emb_init[name]), \
                f"label_emb.{name} should be unchanged when hyp_weight=0"

        # Check UNet parameters HAVE CHANGED
        unet_changed = False
        for name, param in model.unet.named_parameters():
            if not torch.equal(param, unet_init[name]):
                unet_changed = True
                break

        assert unet_changed, \
            "UNet parameters should change after optimizer step (from seg_loss gradients)"

        print("SUCCESS: hyp_weight=0 keeps hyperbolic params frozen, UNet params updated")

    def test_detailed_gradient_report(self, model_and_data, losses):
        """
        Generate a detailed report showing gradient statistics for all parameter groups.
        This test always passes but prints useful diagnostic information.
        """
        model, x, target = model_and_data
        seg_criterion, hyp_criterion = losses

        print("\n" + "=" * 70)
        print("DETAILED GRADIENT REPORT: hyp_weight=0 vs hyp_weight=0.05")
        print("=" * 70)

        for hyp_weight in [0.0, 0.05]:
            print(f"\n{'=' * 30} hyp_weight = {hyp_weight} {'=' * 30}")

            model.zero_grad()
            logits, voxel_emb, label_emb = model(x)
            seg_loss = seg_criterion(logits, target)
            hyp_loss = hyp_criterion(voxel_emb, target, label_emb)
            total_loss = seg_loss + hyp_weight * hyp_loss
            total_loss.backward()

            print(f"\nLoss values:")
            print(f"  seg_loss:   {seg_loss.item():.6f}")
            print(f"  hyp_loss:   {hyp_loss.item():.6f}")
            print(f"  total_loss: {total_loss.item():.6f}")

            # Report for each parameter group
            for group_name, module in [("hyp_head", model.hyp_head),
                                        ("label_emb", model.label_emb),
                                        ("unet (first 3 layers)", model.unet)]:
                print(f"\n{group_name}:")
                count = 0
                for name, param in module.named_parameters():
                    if count >= 3 and group_name.startswith("unet"):
                        print(f"  ... (truncated)")
                        break
                    count += 1

                    if param.grad is None:
                        grad_status = "None"
                    elif torch.all(param.grad == 0):
                        grad_status = "ALL ZERO"
                    else:
                        grad_max = param.grad.abs().max().item()
                        grad_mean = param.grad.abs().mean().item()
                        grad_status = f"max={grad_max:.2e}, mean={grad_mean:.2e}"

                    print(f"  {name}: grad={grad_status}")

        print("\n" + "=" * 70)
        print("CONCLUSION:")
        print("  - When hyp_weight=0: hyp_head and label_emb have ZERO gradients")
        print("  - When hyp_weight>0: hyp_head and label_emb have NON-ZERO gradients")
        print("  - UNet always has gradients from seg_loss")
        print("=" * 70)
