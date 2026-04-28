import torch
import pytest
import json


class TestLorentzLabelEmbedding:
    """Test LorentzLabelEmbedding module."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_output_shape(self, class_depths):
        """Output should be [num_classes, embed_dim]."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert out.shape == (70, 32), f"Expected (70, 32), got {out.shape}"

    def test_output_is_on_manifold(self, class_depths):
        """Output should be valid Lorentz points (finite values)."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        assert torch.isfinite(out).all(), "Output contains inf or nan"

    def test_deeper_organs_farther_from_origin(self, class_depths):
        """Deeper organs should be initialized farther from origin."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            min_radius=0.1,
            max_radius=2.0
        )
        out = emb()
        distances = distance_to_origin(out)

        # Find a shallow and deep class
        min_depth = min(class_depths.values())
        max_depth = max(class_depths.values())

        shallow_idx = [i for i, d in class_depths.items() if d == min_depth][0]
        deep_idx = [i for i, d in class_depths.items() if d == max_depth][0]

        assert distances[deep_idx] > distances[shallow_idx], \
            f"Deep class dist {distances[deep_idx]:.4f} should be > shallow {distances[shallow_idx]:.4f}"

    def test_gradient_flow(self, class_depths):
        """Gradients should flow through the embedding."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        out = emb()
        loss = out.sum()
        loss.backward()

        # Check tangent_embeddings has gradients
        assert emb.tangent_embeddings.grad is not None
        assert (emb.tangent_embeddings.grad != 0).any()

    def test_different_seeds_different_directions(self, class_depths):
        """Different random seeds should give different initial directions."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb1 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        torch.manual_seed(123)
        emb2 = LorentzLabelEmbedding(num_classes=70, embed_dim=32, class_depths=class_depths)

        # Directions should differ
        assert not torch.allclose(emb1.tangent_embeddings, emb2.tangent_embeddings)


class TestDirectionMode:
    """Test direction_mode parameter for configurable initialization."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def text_embedding_path(self):
        """Path to pre-computed text embeddings."""
        return "Dataset/text_embeddings/sat_label_embeddings.pt"

    def test_random_mode_produces_unit_directions(self, class_depths):
        """direction_mode='random' should produce unit vector directions."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

        # Get directions by normalizing tangent vectors
        tangent = emb.tangent_embeddings.detach()
        norms = tangent.norm(dim=-1, keepdim=True)
        directions = tangent / norms

        # Directions should be unit vectors
        direction_norms = directions.norm(dim=-1)
        assert torch.allclose(direction_norms, torch.ones(70), atol=1e-5), \
            f"Directions should be unit vectors, got norms: {direction_norms[:5]}"

    def test_semantic_mode_produces_unit_directions(self, class_depths, text_embedding_path):
        """direction_mode='semantic' should produce unit vector directions from text embeddings."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        # Get directions by normalizing tangent vectors
        tangent = emb.tangent_embeddings.detach()
        norms = tangent.norm(dim=-1, keepdim=True)
        directions = tangent / norms

        # Directions should be unit vectors
        direction_norms = directions.norm(dim=-1)
        assert torch.allclose(direction_norms, torch.ones(70), atol=1e-5), \
            f"Directions should be unit vectors, got norms: {direction_norms[:5]}"

    def test_semantic_mode_is_deterministic(self, class_depths, text_embedding_path):
        """direction_mode='semantic' should produce same directions regardless of seed."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb1 = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        torch.manual_seed(999)
        emb2 = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        # Semantic directions should be identical (deterministic from text embeddings)
        assert torch.allclose(emb1.tangent_embeddings, emb2.tangent_embeddings, atol=1e-5), \
            "Semantic mode should be deterministic"

    def test_invalid_direction_mode_raises_error(self, class_depths):
        """Invalid direction_mode should raise ValueError."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="Unknown direction_mode"):
            LorentzLabelEmbedding(
                num_classes=70,
                embed_dim=32,
                class_depths=class_depths,
                direction_mode="invalid_mode",
            )

    def test_semantic_mode_without_path_raises_error(self, class_depths):
        """direction_mode='semantic' without text_embedding_path should raise ValueError."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="text_embedding_path required"):
            LorentzLabelEmbedding(
                num_classes=70,
                embed_dim=32,
                class_depths=class_depths,
                direction_mode="semantic",
                text_embedding_path=None,
            )

    def test_backward_compatible_default(self, class_depths):
        """Default behavior should be random mode (backward compatible)."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        torch.manual_seed(42)
        emb_default = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
        )

        torch.manual_seed(42)
        emb_random = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

        # Default should behave like random mode
        assert torch.allclose(emb_default.tangent_embeddings, emb_random.tangent_embeddings), \
            "Default should be equivalent to direction_mode='random'"


class TestGetDepthNorms:
    """Test _get_depth_norms method for depth-based norm computation."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    def test_depth_norms_shape(self, class_depths):
        """_get_depth_norms should return [num_classes] tensor."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

        norms = emb._get_depth_norms(70, class_depths, 0.1, 2.0)
        assert norms.shape == (70,), f"Expected (70,), got {norms.shape}"

    def test_depth_norms_range(self, class_depths):
        """Norms should be in [min_radius, max_radius] range."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

        min_radius, max_radius = 0.1, 2.0
        norms = emb._get_depth_norms(70, class_depths, min_radius, max_radius)

        assert norms.min() >= min_radius - 1e-5, f"Min norm {norms.min()} < min_radius {min_radius}"
        assert norms.max() <= max_radius + 1e-5, f"Max norm {norms.max()} > max_radius {max_radius}"

    def test_deeper_classes_have_larger_norms(self, class_depths):
        """Deeper classes should have larger norms."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

        norms = emb._get_depth_norms(70, class_depths, 0.1, 2.0)

        # Find shallow and deep classes
        min_depth = min(class_depths.values())
        max_depth = max(class_depths.values())
        shallow_idx = [i for i, d in class_depths.items() if d == min_depth][0]
        deep_idx = [i for i, d in class_depths.items() if d == max_depth][0]

        assert norms[deep_idx] > norms[shallow_idx], \
            f"Deep norm {norms[deep_idx]} should be > shallow norm {norms[shallow_idx]}"

    def test_depth_norms_fallback_without_class_depths(self):
        """Without class_depths, should return uniform norms."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=None,
            direction_mode="random",
        )

        norms = emb._get_depth_norms(70, None, 0.1, 2.0)

        # All norms should be equal (midpoint)
        expected = (0.1 + 2.0) / 2
        assert torch.allclose(norms, torch.ones(70) * expected, atol=1e-5), \
            f"Without class_depths, norms should be uniform at {expected}"


class TestLoadSemanticDirections:
    """Test _load_semantic_directions method for PCA-based text embedding projection."""

    @pytest.fixture
    def class_depths(self):
        """Load real class depths from dataset."""
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def text_embedding_path(self):
        """Path to pre-computed text embeddings."""
        return "Dataset/text_embeddings/sat_label_embeddings.pt"

    def test_semantic_directions_shape(self, class_depths, text_embedding_path):
        """_load_semantic_directions should return [num_classes, embed_dim] tensor."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        directions = emb._load_semantic_directions(32)
        assert directions.shape == (70, 32), f"Expected (70, 32), got {directions.shape}"

    def test_semantic_directions_are_unit_vectors(self, class_depths, text_embedding_path):
        """Semantic directions should be normalized to unit vectors."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        directions = emb._load_semantic_directions(32)
        norms = directions.norm(dim=-1)

        assert torch.allclose(norms, torch.ones(70), atol=1e-5), \
            f"Directions should be unit vectors, got norms: {norms[:5]}"

    def test_semantic_directions_ordered_by_label_id(self, class_depths, text_embedding_path):
        """Directions should be correctly ordered by label_id."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        directions = emb._load_semantic_directions(32)

        # Load original embeddings to verify ordering
        data = torch.load(text_embedding_path)
        label_ids = data['label_ids']

        # All label_ids should be present (0-69)
        assert set(label_ids.tolist()) == set(range(70)), \
            "label_ids should contain all class indices 0-69"

    def test_embed_dim_exceeds_num_classes_raises_error(self, class_depths, text_embedding_path):
        """embed_dim > num_classes should raise ValueError (PCA limitation)."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="embed_dim.*max PCA components"):
            LorentzLabelEmbedding(
                num_classes=70,
                embed_dim=100,  # > 70 classes
                class_depths=class_depths,
                direction_mode="semantic",
                text_embedding_path=text_embedding_path,
            )

    def test_similar_organs_have_similar_directions(self, class_depths, text_embedding_path):
        """Semantically similar organs should have similar directions."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        import json

        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="semantic",
            text_embedding_path=text_embedding_path,
        )

        directions = emb._load_semantic_directions(32)

        # Load class names to find similar organs
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]

        # Find kidney_left and kidney_right indices
        kidney_left_idx = class_names.index("kidney_left")
        kidney_right_idx = class_names.index("kidney_right")

        # Find liver index (different organ)
        liver_idx = class_names.index("liver")

        # Cosine similarity between kidneys should be higher than kidney-liver
        kidney_sim = torch.dot(directions[kidney_left_idx], directions[kidney_right_idx])
        kidney_liver_sim = torch.dot(directions[kidney_left_idx], directions[liver_idx])

        assert kidney_sim > kidney_liver_sim, \
            f"kidney_left-kidney_right similarity ({kidney_sim:.4f}) should be > " \
            f"kidney_left-liver similarity ({kidney_liver_sim:.4f})"

    def test_num_classes_mismatch_raises_error(self, class_depths, text_embedding_path):
        """num_classes mismatch with text embedding file should raise ValueError."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding

        with pytest.raises(ValueError, match="Text embedding file has 70 classes, but num_classes=50"):
            LorentzLabelEmbedding(
                num_classes=50,  # Mismatch with 70 in file
                embed_dim=32,
                class_depths=class_depths,
                direction_mode="semantic",
                text_embedding_path=text_embedding_path,
            )
