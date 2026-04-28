"""Tests for EmbeddingTracker."""
import json
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

from models.hyperbolic.label_embedding import LorentzLabelEmbedding
from models.hyperbolic.embedding_tracker import EmbeddingTracker


class TestEmbeddingTracker:
    """Test EmbeddingTracker functionality."""

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    @pytest.fixture
    def class_depths(self, class_names):
        from data.organ_hierarchy import load_organ_hierarchy
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def class_to_system(self, class_names):
        from data.organ_hierarchy import load_class_to_system
        return load_class_to_system("Dataset/tree.json", class_names)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def label_embedding(self, class_depths):
        """Create a LorentzLabelEmbedding for testing."""
        torch.manual_seed(42)
        return LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )

    def test_tracker_creates_output_directory(
        self, class_names, class_to_system, temp_output_dir
    ):
        """Test that tracker creates output directory."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )
        assert os.path.exists(os.path.join(temp_output_dir, "test_model"))

    def test_json_structure_after_epoch_0(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test JSON file structure after recording epoch 0."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        tracker.on_epoch_end(epoch=0, label_embedding=label_embedding)

        json_path = os.path.join(temp_output_dir, "test_model", "embedding_history.json")
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)

        # Check metadata
        assert "metadata" in data
        metadata = data["metadata"]
        assert metadata["model_name"] == "test_model"
        assert metadata["num_classes"] == 70
        assert metadata["embed_dim"] == 32
        assert metadata["curv"] == 1.0
        assert len(metadata["class_names"]) == 70
        assert "pca_components" in metadata
        assert len(metadata["pca_components"]) == 2  # 2 x embed_dim

        # Check epochs
        assert "epochs" in data
        assert len(data["epochs"]) == 1
        epoch_data = data["epochs"][0]
        assert epoch_data["epoch"] == 0
        assert "timestamp" in epoch_data
        assert len(epoch_data["tangent_vectors"]) == 70
        assert len(epoch_data["poincare_positions"]) == 70
        assert len(epoch_data["distances_to_origin"]) == 70
        assert epoch_data["has_nan"] is False

    def test_png_generated(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test that PNG or fallback HTML file is generated."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        tracker.on_epoch_end(epoch=0, label_embedding=label_embedding)

        # Check for PNG or fallback HTML
        png_path = os.path.join(temp_output_dir, "test_model", "epoch_000.png")
        html_fallback_path = os.path.join(temp_output_dir, "test_model", "epoch_000.html")
        assert os.path.exists(png_path) or os.path.exists(html_fallback_path)
        if os.path.exists(png_path):
            assert os.path.getsize(png_path) > 0
        else:
            assert os.path.getsize(html_fallback_path) > 0

    def test_html_animation_generated(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test that HTML animation file is generated."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        tracker.on_epoch_end(epoch=0, label_embedding=label_embedding)

        html_path = os.path.join(temp_output_dir, "test_model", "animation.html")
        assert os.path.exists(html_path)
        assert os.path.getsize(html_path) > 0

    def test_pca_consistency_across_epochs(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test that PCA transform is consistent across epochs."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        # Record epoch 0
        tracker.on_epoch_end(epoch=0, label_embedding=label_embedding)
        pca_transform_0 = tracker.pca_transform.copy()

        # Simulate training by modifying embeddings
        with torch.no_grad():
            label_embedding.tangent_embeddings.data += torch.randn_like(
                label_embedding.tangent_embeddings
            ) * 0.1

        # Record epoch 1
        tracker.on_epoch_end(epoch=1, label_embedding=label_embedding)

        # PCA transform should be unchanged
        np.testing.assert_array_equal(pca_transform_0, tracker.pca_transform)

    def test_multiple_epochs_appended(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test that multiple epochs are appended to JSON."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        for epoch in range(3):
            tracker.on_epoch_end(epoch=epoch, label_embedding=label_embedding)
            # Simulate training
            with torch.no_grad():
                label_embedding.tangent_embeddings.data += torch.randn_like(
                    label_embedding.tangent_embeddings
                ) * 0.05

        json_path = os.path.join(temp_output_dir, "test_model", "embedding_history.json")
        with open(json_path) as f:
            data = json.load(f)

        assert len(data["epochs"]) == 3
        assert [e["epoch"] for e in data["epochs"]] == [0, 1, 2]

        # Check visualization files (PNG or fallback HTML)
        for epoch in range(3):
            png_path = os.path.join(temp_output_dir, "test_model", f"epoch_{epoch:03d}.png")
            html_fallback_path = os.path.join(temp_output_dir, "test_model", f"epoch_{epoch:03d}.html")
            assert os.path.exists(png_path) or os.path.exists(html_fallback_path)

    def test_nan_detection(
        self, class_names, class_to_system, class_depths, temp_output_dir
    ):
        """Test that NaN values are detected and flagged."""
        torch.manual_seed(42)
        label_embedding = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )

        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        # Record normal epoch
        tracker.on_epoch_end(epoch=0, label_embedding=label_embedding)

        # Inject NaN
        with torch.no_grad():
            label_embedding.tangent_embeddings.data[0, 0] = float('nan')

        tracker.on_epoch_end(epoch=1, label_embedding=label_embedding)

        json_path = os.path.join(temp_output_dir, "test_model", "embedding_history.json")
        with open(json_path) as f:
            data = json.load(f)

        assert data["epochs"][0]["has_nan"] is False
        assert data["epochs"][1]["has_nan"] is True

    def test_epoch_ordering_guard(
        self, class_names, class_to_system, class_depths, temp_output_dir, label_embedding
    ):
        """Test that calling on_epoch_end without epoch 0 raises error."""
        tracker = EmbeddingTracker(
            model_name="test_model",
            class_names=class_names,
            class_to_system=class_to_system,
            output_dir=temp_output_dir
        )

        # Calling with epoch > 0 before epoch 0 should raise RuntimeError
        with pytest.raises(RuntimeError, match="Must call on_epoch_end.*epoch=0"):
            tracker.on_epoch_end(epoch=1, label_embedding=label_embedding)


class TestClassToSystemMapping:
    """Test organ system mapping function."""

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    def test_load_class_to_system(self, class_names):
        """Test that class_to_system mapping is loaded correctly."""
        from data.organ_hierarchy import load_class_to_system

        class_to_system = load_class_to_system("Dataset/tree.json", class_names)

        assert len(class_to_system) == 70

        # Check some known mappings
        # Find indices for known classes
        liver_idx = class_names.index("liver")
        heart_idx = class_names.index("heart")
        brain_idx = class_names.index("brain")
        rib_idx = class_names.index("rib_left_1")
        gluteus_idx = class_names.index("gluteus_maximus_left")

        assert class_to_system[liver_idx] == "digestive"
        assert class_to_system[heart_idx] == "cardiovascular"
        assert class_to_system[brain_idx] == "nervous"
        assert class_to_system[rib_idx] == "skeletal"
        assert class_to_system[gluteus_idx] == "muscular"

    def test_all_systems_are_valid(self, class_names):
        """Test that all mapped systems are valid color keys."""
        from data.organ_hierarchy import load_class_to_system
        from models.hyperbolic.embedding_tracker import SYSTEM_COLORS

        class_to_system = load_class_to_system("Dataset/tree.json", class_names)

        for idx, system in class_to_system.items():
            assert system in SYSTEM_COLORS, f"Class {idx} ({class_names[idx]}) has invalid system: {system}"
