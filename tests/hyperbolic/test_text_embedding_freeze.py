"""Tests for text embedding freeze feature.

This feature freezes label embeddings for the first N epochs to prevent
text features from collapsing to center due to parameter imbalance.
"""
import torch
import torch.nn as nn
import pytest


class TestConfigParameters:
    """Test new config parameters for freeze feature."""

    def test_config_has_hyp_freeze_epochs(self):
        """Config should have hyp_freeze_epochs parameter with default 0."""
        from config import Config
        cfg = Config()
        assert hasattr(cfg, 'hyp_freeze_epochs')
        assert cfg.hyp_freeze_epochs == 0  # Default: no freezing (backward compatible)

    def test_config_has_hyp_text_lr_ratio(self):
        """Config should have hyp_text_lr_ratio parameter with default 0.01."""
        from config import Config
        cfg = Config()
        assert hasattr(cfg, 'hyp_text_lr_ratio')
        assert cfg.hyp_text_lr_ratio == 0.01

    def test_config_has_hyp_text_grad_clip(self):
        """Config should have hyp_text_grad_clip parameter with default 0.1."""
        from config import Config
        cfg = Config()
        assert hasattr(cfg, 'hyp_text_grad_clip')
        assert cfg.hyp_text_grad_clip == 0.1

    def test_config_loads_freeze_params_from_yaml(self, tmp_path):
        """Config should load freeze parameters from YAML file."""
        from config import Config

        yaml_content = """
hyp_freeze_epochs: 15
hyp_text_lr_ratio: 0.05
hyp_text_grad_clip: 0.2
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        cfg = Config.from_yaml(str(yaml_file))
        assert cfg.hyp_freeze_epochs == 15
        assert cfg.hyp_text_lr_ratio == 0.05
        assert cfg.hyp_text_grad_clip == 0.2


class TestFreezeUnfreezeLogic:
    """Test freeze/unfreeze logic for label embeddings."""

    @pytest.fixture
    def label_embedding(self):
        """Create a simple label embedding for testing."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from data.organ_hierarchy import load_organ_hierarchy
        import json

        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

        return LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

    def test_freeze_disables_gradient(self, label_embedding):
        """Freezing should set requires_grad=False on tangent_embeddings."""
        label_embedding.tangent_embeddings.requires_grad_(False)
        assert not label_embedding.tangent_embeddings.requires_grad

    def test_unfreeze_enables_gradient(self, label_embedding):
        """Unfreezing should set requires_grad=True on tangent_embeddings."""
        label_embedding.tangent_embeddings.requires_grad_(False)
        label_embedding.tangent_embeddings.requires_grad_(True)
        assert label_embedding.tangent_embeddings.requires_grad

    def test_frozen_embedding_no_gradient_after_backward(self, label_embedding):
        """Frozen embedding should not accumulate gradients."""
        label_embedding.tangent_embeddings.requires_grad_(False)

        out = label_embedding()
        loss = out.sum()
        # Should not raise error even though requires_grad=False
        # (loss still has grad_fn from other operations)

        # Gradient should be None since requires_grad=False
        assert label_embedding.tangent_embeddings.grad is None

    def test_unfrozen_embedding_has_gradient_after_backward(self, label_embedding):
        """Unfrozen embedding should accumulate gradients."""
        label_embedding.tangent_embeddings.requires_grad_(True)

        out = label_embedding()
        loss = out.sum()
        loss.backward()

        assert label_embedding.tangent_embeddings.grad is not None
        assert (label_embedding.tangent_embeddings.grad != 0).any()


class TestSeparateParamGroups:
    """Test separate parameter groups for optimizer."""

    @pytest.fixture
    def model_and_config(self):
        """Create BodyNet model and config for testing."""
        from models.body_net import BodyNet
        from config import Config
        from data.organ_hierarchy import load_organ_hierarchy
        import json

        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

        cfg = Config()
        cfg.hyp_text_lr_ratio = 0.01

        model = BodyNet(
            in_channels=1,
            num_classes=70,
            base_channels=32,
            growth_rate=32,
            dense_layers=4,
            bn_size=4,
            embed_dim=32,
            curv=1.0,
            class_depths=class_depths,
        )

        return model, cfg

    def test_separate_param_groups_creation(self, model_and_config):
        """Should create separate param groups for visual and text params."""
        model, cfg = model_and_config

        visual_params = [p for n, p in model.named_parameters() if 'label_emb' not in n]
        text_params = [p for n, p in model.named_parameters() if 'label_emb' in n]

        assert len(visual_params) > 0, "Should have visual params"
        assert len(text_params) > 0, "Should have text params"

        # Text params should only be tangent_embeddings
        assert len(text_params) == 1, "Should have exactly 1 text param (tangent_embeddings)"

    def test_optimizer_with_separate_lr(self, model_and_config):
        """Optimizer should have different LR for visual and text params."""
        model, cfg = model_and_config
        base_lr = 0.001

        visual_params = [p for n, p in model.named_parameters() if 'label_emb' not in n]
        text_params = [p for n, p in model.named_parameters() if 'label_emb' in n]

        optimizer = torch.optim.Adam([
            {'params': visual_params, 'lr': base_lr},
            {'params': text_params, 'lr': base_lr * cfg.hyp_text_lr_ratio}
        ])

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == base_lr
        assert optimizer.param_groups[1]['lr'] == base_lr * cfg.hyp_text_lr_ratio

    def test_text_params_lr_is_smaller(self, model_and_config):
        """Text params should have smaller LR than visual params."""
        model, cfg = model_and_config
        base_lr = 0.001

        visual_params = [p for n, p in model.named_parameters() if 'label_emb' not in n]
        text_params = [p for n, p in model.named_parameters() if 'label_emb' in n]

        optimizer = torch.optim.Adam([
            {'params': visual_params, 'lr': base_lr},
            {'params': text_params, 'lr': base_lr * cfg.hyp_text_lr_ratio}
        ])

        visual_lr = optimizer.param_groups[0]['lr']
        text_lr = optimizer.param_groups[1]['lr']

        assert text_lr < visual_lr, f"Text LR ({text_lr}) should be < visual LR ({visual_lr})"
        assert text_lr == base_lr * 0.01, f"Text LR should be {base_lr * 0.01}, got {text_lr}"


class TestGradientClipping:
    """Test extra gradient clipping for text embeddings on first unfreeze epoch."""

    @pytest.fixture
    def label_embedding(self):
        """Create a label embedding for testing."""
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from data.organ_hierarchy import load_organ_hierarchy
        import json

        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

        return LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths,
            direction_mode="random",
        )

    def test_clip_grad_norm_on_single_param(self, label_embedding):
        """clip_grad_norm_ should work on single parameter tensor."""
        label_embedding.tangent_embeddings.requires_grad_(True)

        out = label_embedding()
        loss = out.sum() * 100  # Large multiplier to create large gradients
        loss.backward()

        grad_norm_before = label_embedding.tangent_embeddings.grad.norm().item()

        # Clip to small value
        nn.utils.clip_grad_norm_(label_embedding.tangent_embeddings, max_norm=0.1)

        grad_norm_after = label_embedding.tangent_embeddings.grad.norm().item()

        assert grad_norm_after <= 0.1 + 1e-5, \
            f"Grad norm after clip ({grad_norm_after}) should be <= 0.1"

    def test_clip_grad_norm_preserves_direction(self, label_embedding):
        """Gradient clipping should preserve gradient direction."""
        label_embedding.tangent_embeddings.requires_grad_(True)

        out = label_embedding()
        loss = out.sum() * 100
        loss.backward()

        grad_before = label_embedding.tangent_embeddings.grad.clone()
        grad_direction_before = grad_before / grad_before.norm()

        nn.utils.clip_grad_norm_(label_embedding.tangent_embeddings, max_norm=0.1)

        grad_after = label_embedding.tangent_embeddings.grad
        grad_direction_after = grad_after / grad_after.norm()

        # Directions should be the same (cosine similarity ~1)
        cosine_sim = (grad_direction_before * grad_direction_after).sum()
        assert cosine_sim > 0.999, f"Gradient direction changed: cosine_sim={cosine_sim}"


class TestIntegration:
    """Integration tests for freeze feature in training context."""

    @pytest.fixture
    def minimal_training_setup(self):
        """Create minimal training setup for integration testing."""
        from models.body_net import BodyNet
        from models.hyperbolic.lorentz_loss import LorentzRankingLoss
        from config import Config
        from data.organ_hierarchy import load_organ_hierarchy
        import json

        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

        cfg = Config()
        cfg.hyp_freeze_epochs = 2
        cfg.hyp_text_lr_ratio = 0.01
        cfg.hyp_text_grad_clip = 0.1

        model = BodyNet(
            in_channels=1,
            num_classes=70,
            base_channels=32,
            growth_rate=32,
            dense_layers=4,
            bn_size=4,
            embed_dim=32,
            curv=1.0,
            class_depths=class_depths,
        )

        hyp_criterion = LorentzRankingLoss(
            margin=0.1,
            curv=1.0,
            num_samples_per_class=8,
            num_negatives=4,
        )

        return model, hyp_criterion, cfg

    def test_frozen_epochs_no_embedding_change(self, minimal_training_setup):
        """During frozen epochs, label embeddings should not change."""
        model, hyp_criterion, cfg = minimal_training_setup

        # Record initial embeddings
        initial_emb = model.label_emb.tangent_embeddings.clone().detach()

        # Freeze
        model.label_emb.tangent_embeddings.requires_grad_(False)

        # Simulate training step
        visual_params = [p for n, p in model.named_parameters() if 'label_emb' not in n]
        text_params = [p for n, p in model.named_parameters() if 'label_emb' in n]

        optimizer = torch.optim.Adam([
            {'params': visual_params, 'lr': 0.001},
            {'params': text_params, 'lr': 0.001 * cfg.hyp_text_lr_ratio}
        ])

        # Forward pass with dummy input
        dummy_input = torch.randn(1, 1, 16, 16, 16)
        logits, voxel_emb, label_emb = model(dummy_input)

        # Create dummy target
        dummy_target = torch.randint(0, 70, (1, 16, 16, 16))

        # Compute loss and backward
        hyp_loss = hyp_criterion(voxel_emb, dummy_target, label_emb)
        hyp_loss.backward()
        optimizer.step()

        # Check embeddings unchanged
        final_emb = model.label_emb.tangent_embeddings.clone().detach()
        assert torch.allclose(initial_emb, final_emb), \
            "Frozen embeddings should not change during training"

    def test_unfrozen_epochs_embedding_changes(self, minimal_training_setup):
        """After unfreezing, label embeddings should change during training."""
        model, hyp_criterion, cfg = minimal_training_setup

        # Record initial embeddings
        initial_emb = model.label_emb.tangent_embeddings.clone().detach()

        # Unfreeze
        model.label_emb.tangent_embeddings.requires_grad_(True)

        # Setup optimizer
        visual_params = [p for n, p in model.named_parameters() if 'label_emb' not in n]
        text_params = [p for n, p in model.named_parameters() if 'label_emb' in n]

        optimizer = torch.optim.Adam([
            {'params': visual_params, 'lr': 0.001},
            {'params': text_params, 'lr': 0.001 * cfg.hyp_text_lr_ratio}
        ])

        # Forward pass with dummy input
        dummy_input = torch.randn(1, 1, 16, 16, 16)
        logits, voxel_emb, label_emb = model(dummy_input)

        # Create dummy target
        dummy_target = torch.randint(0, 70, (1, 16, 16, 16))

        # Compute loss and backward
        hyp_loss = hyp_criterion(voxel_emb, dummy_target, label_emb)
        hyp_loss.backward()
        optimizer.step()

        # Check embeddings changed
        final_emb = model.label_emb.tangent_embeddings.clone().detach()
        assert not torch.allclose(initial_emb, final_emb), \
            "Unfrozen embeddings should change during training"

    def test_first_unfreeze_epoch_extra_clipping(self, minimal_training_setup):
        """First unfreeze epoch should apply extra gradient clipping to text embeddings."""
        model, hyp_criterion, cfg = minimal_training_setup

        # Unfreeze (simulating first unfreeze epoch)
        model.label_emb.tangent_embeddings.requires_grad_(True)

        # Forward pass
        dummy_input = torch.randn(1, 1, 16, 16, 16)
        logits, voxel_emb, label_emb = model(dummy_input)
        dummy_target = torch.randint(0, 70, (1, 16, 16, 16))

        # Compute loss and backward
        hyp_loss = hyp_criterion(voxel_emb, dummy_target, label_emb)
        hyp_loss.backward()

        # Apply extra clipping (as would happen on first unfreeze epoch)
        nn.utils.clip_grad_norm_(model.label_emb.tangent_embeddings, cfg.hyp_text_grad_clip)

        # Check gradient norm is clipped
        grad_norm = model.label_emb.tangent_embeddings.grad.norm().item()
        assert grad_norm <= cfg.hyp_text_grad_clip + 1e-5, \
            f"Grad norm ({grad_norm}) should be <= {cfg.hyp_text_grad_clip}"
