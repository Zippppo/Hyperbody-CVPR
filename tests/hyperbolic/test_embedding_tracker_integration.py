"""
Integration test for EmbeddingTracker with BodyNet model.

Tests that the tracker can correctly extract and track embeddings
from a BodyNet model during simulated training, including DDP-wrapped models.
"""
import json
import os
import shutil
import tempfile

import pytest
import torch

from models.body_net import BodyNet
from models.hyperbolic.embedding_tracker import EmbeddingTracker
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system


class TestEmbeddingTrackerIntegration:
    """Test EmbeddingTracker integration with BodyNet."""

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    @pytest.fixture
    def class_depths(self, class_names):
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def class_to_system(self, class_names):
        return load_class_to_system("Dataset/tree.json", class_names)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def body_net(self, class_depths):
        """Create a small BodyNet for testing."""
        torch.manual_seed(42)
        return BodyNet(
            in_channels=1,
            num_classes=70,
            base_channels=8,  # Small for testing
            growth_rate=8,
            dense_layers=2,
            bn_size=2,
            embed_dim=16,
            curv=1.0,
            class_depths=class_depths,
            min_radius=0.1,
            max_radius=2.0,
        )

    def test_tracker_with_body_net_label_emb(
        self, class_names, class_to_system, temp_output_dir, body_net
    ):
        """Test that tracker can extract embeddings from BodyNet.label_emb."""
        tracker = EmbeddingTracker(
            model_name="test_bodynet",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir,
            curv=1.0
        )

        # Record epoch 0
        tracker.on_epoch_end(epoch=0, label_embedding=body_net.label_emb)

        # Verify JSON was created with correct structure
        json_path = os.path.join(temp_output_dir, "test_bodynet", "embedding_history.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert data["metadata"]["num_classes"] == 70
        assert data["metadata"]["embed_dim"] == 16
        assert len(data["epochs"]) == 1
        assert data["epochs"][0]["epoch"] == 0

    def test_tracker_with_simulated_training_loop(
        self, class_names, class_to_system, class_depths, temp_output_dir, body_net
    ):
        """Test tracker in a simulated training loop scenario."""
        tracker = EmbeddingTracker(
            model_name="test_training",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir,
            curv=1.0
        )

        # Simulate 3 epochs of training
        num_epochs = 3
        for epoch in range(num_epochs):
            # Record embeddings at start of epoch (or end of previous)
            tracker.on_epoch_end(epoch=epoch, label_embedding=body_net.label_emb)

            # Simulate training step: modify embeddings slightly
            with torch.no_grad():
                body_net.label_emb.tangent_embeddings.data += (
                    torch.randn_like(body_net.label_emb.tangent_embeddings) * 0.05
                )

        # Verify all epochs recorded
        json_path = os.path.join(temp_output_dir, "test_training", "embedding_history.json")
        with open(json_path) as f:
            data = json.load(f)

        assert len(data["epochs"]) == num_epochs
        epochs_recorded = [e["epoch"] for e in data["epochs"]]
        assert epochs_recorded == [0, 1, 2]

        # Verify embeddings changed between epochs
        import numpy as np
        pos_0 = np.array(data["epochs"][0]["poincare_positions"])
        pos_2 = np.array(data["epochs"][2]["poincare_positions"])
        movement = np.linalg.norm(pos_2 - pos_0, axis=1).mean()
        assert movement > 0.01, "Embeddings should have moved during training"

    def test_tracker_with_ddp_wrapped_model(
        self, class_names, class_to_system, temp_output_dir, body_net
    ):
        """Test that tracker works with DDP-style model.module access pattern."""
        # Simulate DDP wrapping by creating a wrapper with .module attribute
        class FakeDDP:
            def __init__(self, module):
                self.module = module

        wrapped_model = FakeDDP(body_net)

        tracker = EmbeddingTracker(
            model_name="test_ddp",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir,
            curv=1.0
        )

        # Access label_emb through .module (as done in train.py)
        raw_model = wrapped_model.module
        tracker.on_epoch_end(epoch=0, label_embedding=raw_model.label_emb)

        # Verify it worked
        json_path = os.path.join(temp_output_dir, "test_ddp", "embedding_history.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)

        assert len(data["epochs"]) == 1

    def test_tracker_curv_parameter_consistency(
        self, class_names, class_to_system, temp_output_dir, class_depths
    ):
        """Test that tracker uses correct curvature parameter."""
        curv = 0.5  # Non-default curvature

        # Create model with same curvature
        torch.manual_seed(42)
        model = BodyNet(
            in_channels=1,
            num_classes=70,
            base_channels=8,
            growth_rate=8,
            dense_layers=2,
            bn_size=2,
            embed_dim=16,
            curv=curv,
            class_depths=class_depths,
        )

        tracker = EmbeddingTracker(
            model_name="test_curv",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir,
            curv=curv
        )

        tracker.on_epoch_end(epoch=0, label_embedding=model.label_emb)

        json_path = os.path.join(temp_output_dir, "test_curv", "embedding_history.json")
        with open(json_path) as f:
            data = json.load(f)

        assert data["metadata"]["curv"] == curv
