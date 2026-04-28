"""
Test for track_embeddings config parameter.
"""
import os
import tempfile

import pytest
import yaml

from config import Config


class TestTrackEmbeddingsConfig:
    """Test track_embeddings configuration parameter."""

    def test_config_has_track_embeddings_default_false(self):
        """Test that Config has track_embeddings field defaulting to False."""
        cfg = Config()
        assert hasattr(cfg, "track_embeddings")
        assert cfg.track_embeddings is False

    def test_config_load_track_embeddings_from_yaml(self):
        """Test that track_embeddings can be loaded from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"track_embeddings": True}, f)
            temp_path = f.name

        try:
            cfg = Config.from_yaml(temp_path)
            assert cfg.track_embeddings is True
        finally:
            os.unlink(temp_path)

    def test_config_track_embeddings_false_from_yaml(self):
        """Test that track_embeddings=False works from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"track_embeddings": False}, f)
            temp_path = f.name

        try:
            cfg = Config.from_yaml(temp_path)
            assert cfg.track_embeddings is False
        finally:
            os.unlink(temp_path)
