import torch
import pytest
import json


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_full_forward_backward(self, class_depths):
        """Test complete forward and backward pass."""
        from models.body_net import BodyNet
        from models.losses import CombinedLoss
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss

        # Create model
        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )

        # Create losses
        seg_criterion = CombinedLoss(num_classes=70)
        hyp_criterion = LorentzRankingLoss(margin=0.1, num_negatives=4)

        # Create fake batch
        x = torch.randn(2, 1, 32, 24, 32)
        targets = torch.randint(0, 70, (2, 32, 24, 32))

        # Forward
        logits, voxel_emb, label_emb = model(x)

        # Compute losses
        seg_loss = seg_criterion(logits, targets)
        hyp_loss = hyp_criterion(voxel_emb, targets, label_emb)
        total_loss = seg_loss + 0.05 * hyp_loss

        # Backward
        total_loss.backward()

        # Check gradients exist
        assert model.unet.enc1.block[0].weight.grad is not None
        assert model.hyp_head.conv.weight.grad is not None
        assert model.label_emb.tangent_embeddings.grad is not None

    def test_training_step_decreases_loss(self, class_depths):
        """Verify that a training step decreases loss."""
        from models.body_net import BodyNet
        from models.losses import CombinedLoss
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        import torch.optim as optim

        torch.manual_seed(42)

        model = BodyNet(
            num_classes=70,
            base_channels=16,  # Smaller for speed
            embed_dim=16,
            class_depths=class_depths
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        seg_criterion = CombinedLoss(num_classes=70)
        hyp_criterion = LorentzRankingLoss(margin=0.1, num_negatives=4)

        # Fixed input/target
        x = torch.randn(1, 1, 16, 12, 16)
        targets = torch.randint(0, 70, (1, 16, 12, 16))

        # Initial loss
        logits, voxel_emb, label_emb = model(x)
        loss_before = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)

        # Training step
        optimizer.zero_grad()
        logits, voxel_emb, label_emb = model(x)
        loss = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)
        loss.backward()
        optimizer.step()

        # Loss after
        logits, voxel_emb, label_emb = model(x)
        loss_after = seg_criterion(logits, targets) + 0.05 * hyp_criterion(voxel_emb, targets, label_emb)

        assert loss_after < loss_before, f"Loss did not decrease: {loss_before:.4f} -> {loss_after:.4f}"
