import torch
import pytest
import json


class TestBodyNet:
    """Test BodyNet model wrapper."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_output_is_tuple(self, class_depths):
        """Forward should return (logits, voxel_emb, label_emb)."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        out = model(x)
        assert isinstance(out, tuple)
        assert len(out) == 3

    def test_logits_shape(self, class_depths):
        """Logits should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        logits, _, _ = model(x)
        assert logits.shape == (1, 70, 32, 24, 32)

    def test_voxel_emb_shape(self, class_depths):
        """Voxel embeddings should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        _, voxel_emb, _ = model(x)
        assert voxel_emb.shape == (1, 32, 32, 24, 32)

    def test_label_emb_shape(self, class_depths):
        """Label embeddings should have correct shape."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 32, 24, 32)

        _, _, label_emb = model(x)
        assert label_emb.shape == (70, 32)

    def test_gradient_flow(self, class_depths):
        """Gradients should flow through all outputs."""
        from models.body_net import BodyNet

        model = BodyNet(
            num_classes=70,
            base_channels=32,
            embed_dim=32,
            class_depths=class_depths
        )
        x = torch.randn(1, 1, 16, 12, 16, requires_grad=True)

        logits, voxel_emb, label_emb = model(x)
        loss = logits.sum() + voxel_emb.sum() + label_emb.sum()
        loss.backward()

        assert x.grad is not None
