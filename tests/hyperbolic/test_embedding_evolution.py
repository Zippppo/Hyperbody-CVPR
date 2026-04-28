"""
Tests for scripts/visualize_embedding_evolution.py

TDD test suite for post-processing MDS embedding evolution animation.
Tests cover: checkpoint discovery, tangent extraction, MDS projection,
Procrustes alignment, Poincare ball normalization, and end-to-end pipeline.

Run:
    pytest tests/hyperbolic/test_embedding_evolution.py -v -s
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.hyperbolic.lorentz_ops import exp_map0, pairwise_dist


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def num_classes():
    return 70


@pytest.fixture
def embed_dim():
    return 48


@pytest.fixture
def curv():
    return 1.0


@pytest.fixture
def tmp_checkpoint_dir(tmp_path, num_classes, embed_dim):
    """Create a temporary checkpoint directory with fake .pth files."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    torch.manual_seed(0)
    for epoch_num in [10, 20, 30, 40, 50]:
        tangent = torch.randn(num_classes, embed_dim) * 0.1 * (epoch_num / 10)
        state_dict = {"label_emb.tangent_embeddings": tangent}
        ckpt = {
            "epoch": epoch_num - 1,  # 0-indexed
            "model_state_dict": state_dict,
            "best_dice": 0.3 + 0.01 * epoch_num,
            "train_loss": 1.0 - 0.01 * epoch_num,
            "val_loss": 1.0 - 0.008 * epoch_num,
            "mean_dice": 0.3 + 0.01 * epoch_num,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch_num}.pth")

    # best.pth (copy of epoch_50)
    best_tangent = torch.randn(num_classes, embed_dim) * 0.5
    best_state = {"label_emb.tangent_embeddings": best_tangent}
    best_ckpt = {
        "epoch": 49,
        "model_state_dict": best_state,
        "best_dice": 0.8,
        "train_loss": 0.2,
        "val_loss": 0.25,
        "mean_dice": 0.8,
    }
    torch.save(best_ckpt, ckpt_dir / "best.pth")

    # latest.pth (should be excluded)
    torch.save(best_ckpt, ckpt_dir / "latest.pth")

    return ckpt_dir


@pytest.fixture
def tmp_checkpoint_with_module_prefix(tmp_path, num_classes, embed_dim):
    """Checkpoint with 'module.' prefix (DataParallel)."""
    ckpt_dir = tmp_path / "checkpoints_dp"
    ckpt_dir.mkdir()

    torch.manual_seed(42)
    tangent = torch.randn(num_classes, embed_dim) * 0.3
    state_dict = {"module.label_emb.tangent_embeddings": tangent}
    ckpt = {
        "epoch": 9,
        "model_state_dict": state_dict,
        "best_dice": 0.5,
        "train_loss": 0.5,
        "val_loss": 0.55,
        "mean_dice": 0.5,
    }
    torch.save(ckpt, ckpt_dir / "epoch_10.pth")
    return ckpt_dir


@pytest.fixture
def synthetic_projections(num_classes):
    """List of synthetic 2D projections for normalization tests."""
    np.random.seed(42)
    projections = []
    for i in range(5):
        scale = 0.5 + i * 0.3
        proj = np.random.randn(num_classes, 2) * scale
        projections.append(proj)
    return projections


@pytest.fixture
def class_names():
    """Load real class names from Dataset."""
    info_path = PROJECT_ROOT / "Dataset" / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        return info["class_names"]
    # Fallback for CI
    return [f"class_{i}" for i in range(70)]


@pytest.fixture
def class_to_system(class_names):
    """Load real class-to-system mapping."""
    tree_path = PROJECT_ROOT / "Dataset" / "tree.json"
    if tree_path.exists():
        from data.organ_hierarchy import load_class_to_system
        return load_class_to_system(str(tree_path), class_names)
    # Fallback
    systems = ["skeletal", "muscular", "digestive", "respiratory",
               "cardiovascular", "urinary", "nervous", "other"]
    return {i: systems[i % len(systems)] for i in range(len(class_names))}


# ---------------------------------------------------------------------------
# Test 1: Checkpoint Discovery
# ---------------------------------------------------------------------------

class TestCheckpointDiscovery:
    """discover_checkpoints finds epoch files, sorts correctly, excludes latest.pth."""

    def test_finds_all_epoch_files(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import discover_checkpoints
        checkpoints = discover_checkpoints(str(tmp_checkpoint_dir))

        labels = [label for label, _ in checkpoints]
        # Should find epoch_10 through epoch_50 + best
        assert len(checkpoints) == 6
        print(f"[PASS] Found {len(checkpoints)} checkpoints: {labels}")

    def test_sorted_by_epoch(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import discover_checkpoints
        checkpoints = discover_checkpoints(str(tmp_checkpoint_dir))

        labels = [label for label, _ in checkpoints]
        # epoch files should be sorted numerically, best at end
        expected = ["epoch_10", "epoch_20", "epoch_30", "epoch_40", "epoch_50", "best"]
        assert labels == expected, f"Expected {expected}, got {labels}"
        print(f"[PASS] Sorted correctly: {labels}")

    def test_excludes_latest(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import discover_checkpoints
        checkpoints = discover_checkpoints(str(tmp_checkpoint_dir))

        labels = [label for label, _ in checkpoints]
        assert "latest" not in labels, "latest.pth should be excluded"
        print(f"[PASS] latest.pth excluded")

    def test_paths_exist(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import discover_checkpoints
        checkpoints = discover_checkpoints(str(tmp_checkpoint_dir))

        for label, path in checkpoints:
            assert Path(path).exists(), f"{path} does not exist"
        print(f"[PASS] All checkpoint paths exist")

    def test_empty_dir(self, tmp_path):
        from scripts.visualize_embedding_evolution import discover_checkpoints
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        checkpoints = discover_checkpoints(str(empty_dir))
        assert len(checkpoints) == 0
        print(f"[PASS] Empty directory returns empty list")


# ---------------------------------------------------------------------------
# Test 2: Tangent Extraction
# ---------------------------------------------------------------------------

class TestTangentExtraction:
    """extract_tangent_embeddings returns correct shape and metadata."""

    def test_correct_shape(self, tmp_checkpoint_dir, num_classes, embed_dim):
        from scripts.visualize_embedding_evolution import extract_tangent_embeddings
        path = str(tmp_checkpoint_dir / "epoch_10.pth")
        tangent, metadata = extract_tangent_embeddings(path)

        assert tangent.shape == (num_classes, embed_dim), \
            f"Expected ({num_classes}, {embed_dim}), got {tangent.shape}"
        print(f"[PASS] Tangent shape: {tangent.shape}")

    def test_metadata_fields(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import extract_tangent_embeddings
        path = str(tmp_checkpoint_dir / "epoch_10.pth")
        _, metadata = extract_tangent_embeddings(path)

        assert "epoch" in metadata
        assert "best_dice" in metadata
        print(f"[PASS] Metadata: {metadata}")

    def test_handles_module_prefix(self, tmp_checkpoint_with_module_prefix, num_classes, embed_dim):
        from scripts.visualize_embedding_evolution import extract_tangent_embeddings
        path = str(tmp_checkpoint_with_module_prefix / "epoch_10.pth")
        tangent, metadata = extract_tangent_embeddings(path)

        assert tangent.shape == (num_classes, embed_dim), \
            f"Expected ({num_classes}, {embed_dim}), got {tangent.shape}"
        print(f"[PASS] Handles module. prefix correctly, shape: {tangent.shape}")

    def test_returns_tensor(self, tmp_checkpoint_dir):
        from scripts.visualize_embedding_evolution import extract_tangent_embeddings
        path = str(tmp_checkpoint_dir / "epoch_10.pth")
        tangent, _ = extract_tangent_embeddings(path)

        assert isinstance(tangent, torch.Tensor), f"Expected Tensor, got {type(tangent)}"
        print(f"[PASS] Returns torch.Tensor")


# ---------------------------------------------------------------------------
# Test 3: Geodesic Distance Matrix
# ---------------------------------------------------------------------------

class TestGeodesicDistanceMatrix:
    """compute_geodesic_distance_matrix produces symmetric matrix with zero diagonal."""

    def test_output_shape(self, num_classes, embed_dim, curv):
        from scripts.visualize_embedding_evolution import compute_geodesic_distance_matrix

        torch.manual_seed(42)
        tangent = torch.randn(num_classes, embed_dim)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)

        assert D.shape == (num_classes, num_classes), \
            f"Expected ({num_classes}, {num_classes}), got {D.shape}"
        print(f"[PASS] Distance matrix shape: {D.shape}")

    def test_zero_diagonal(self, num_classes, embed_dim, curv):
        from scripts.visualize_embedding_evolution import compute_geodesic_distance_matrix

        torch.manual_seed(42)
        tangent = torch.randn(num_classes, embed_dim)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)

        diag = np.diag(D)
        assert np.allclose(diag, 0.0), f"Diagonal not zero: max={diag.max():.6f}"
        print(f"[PASS] Diagonal is zero (max={diag.max():.2e})")

    def test_symmetric(self, num_classes, embed_dim, curv):
        from scripts.visualize_embedding_evolution import compute_geodesic_distance_matrix

        torch.manual_seed(42)
        tangent = torch.randn(num_classes, embed_dim)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)

        diff = np.abs(D - D.T).max()
        assert diff < 1e-5, f"Not symmetric: max diff={diff:.6f}"
        print(f"[PASS] Symmetric (max diff={diff:.2e})")

    def test_non_negative(self, num_classes, embed_dim, curv):
        from scripts.visualize_embedding_evolution import compute_geodesic_distance_matrix

        torch.manual_seed(42)
        tangent = torch.randn(num_classes, embed_dim)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)

        assert np.all(D >= 0), f"Negative distances found: min={D.min():.6f}"
        print(f"[PASS] All distances non-negative")

    def test_returns_numpy(self, num_classes, embed_dim, curv):
        from scripts.visualize_embedding_evolution import compute_geodesic_distance_matrix

        torch.manual_seed(42)
        tangent = torch.randn(num_classes, embed_dim)
        lorentz = exp_map0(tangent, curv=curv)
        D = compute_geodesic_distance_matrix(lorentz, curv)

        assert isinstance(D, np.ndarray), f"Expected ndarray, got {type(D)}"
        print(f"[PASS] Returns numpy array")


# ---------------------------------------------------------------------------
# Test 4: MDS Projection
# ---------------------------------------------------------------------------

class TestMDSProjection:
    """mds_project produces [N, 2] output, reproducible with same seed."""

    def test_output_shape(self, num_classes):
        from scripts.visualize_embedding_evolution import mds_project

        np.random.seed(42)
        D = np.random.rand(num_classes, num_classes)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)

        proj = mds_project(D)
        assert proj.shape == (num_classes, 2), f"Expected ({num_classes}, 2), got {proj.shape}"
        print(f"[PASS] MDS output shape: {proj.shape}")

    def test_reproducible(self, num_classes):
        from scripts.visualize_embedding_evolution import mds_project

        np.random.seed(42)
        D = np.random.rand(num_classes, num_classes)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)

        proj1 = mds_project(D)
        proj2 = mds_project(D)

        assert np.allclose(proj1, proj2), "MDS not reproducible"
        print(f"[PASS] MDS is reproducible")

    def test_warm_start_output_shape(self, num_classes):
        from scripts.visualize_embedding_evolution import mds_project

        np.random.seed(42)
        D = np.random.rand(num_classes, num_classes)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)

        init = np.random.randn(num_classes, 2)
        proj = mds_project(D, init=init)
        assert proj.shape == (num_classes, 2), f"Expected ({num_classes}, 2), got {proj.shape}"
        print(f"[PASS] Warm-start MDS output shape: {proj.shape}")

    def test_warm_start_stays_close(self, num_classes, embed_dim, curv):
        """Warm-started MDS on similar distance matrices should produce similar projections."""
        from scripts.visualize_embedding_evolution import (
            mds_project, compute_geodesic_distance_matrix,
        )

        torch.manual_seed(42)
        tangent1 = torch.randn(num_classes, embed_dim)
        tangent2 = tangent1 + torch.randn(num_classes, embed_dim) * 0.05  # small perturbation

        lorentz1 = exp_map0(tangent1, curv=curv)
        lorentz2 = exp_map0(tangent2, curv=curv)
        D1 = compute_geodesic_distance_matrix(lorentz1, curv)
        D2 = compute_geodesic_distance_matrix(lorentz2, curv)

        proj1 = mds_project(D1)
        proj2_cold = mds_project(D2)
        proj2_warm = mds_project(D2, init=proj1)

        # Warm-started should be closer to proj1 than cold-started
        rmse_cold = np.sqrt(np.mean((proj2_cold - proj1)**2))
        rmse_warm = np.sqrt(np.mean((proj2_warm - proj1)**2))
        print(f"[INFO] Cold-start RMSE to prev: {rmse_cold:.4f}")
        print(f"[INFO] Warm-start RMSE to prev: {rmse_warm:.4f}")
        assert rmse_warm < rmse_cold, \
            f"Warm-start ({rmse_warm:.4f}) should be closer than cold-start ({rmse_cold:.4f})"
        print(f"[PASS] Warm-start produces smoother transitions")


# ---------------------------------------------------------------------------
# Test 5: Procrustes Alignment
# ---------------------------------------------------------------------------

class TestProcrustesAlignment:
    """procrustes_align corrects rotation/reflection, preserves pairwise distances."""

    def test_corrects_known_rotation(self):
        from scripts.visualize_embedding_evolution import procrustes_align

        np.random.seed(42)
        ref = np.random.randn(20, 2)

        # Apply a known 90-degree rotation
        angle = np.pi / 2
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)]])
        rotated = ref @ R.T

        aligned = procrustes_align(ref, rotated)

        # After alignment, should be close to reference
        diff = np.linalg.norm(aligned - ref)
        print(f"[INFO] Alignment error after 90-deg rotation: {diff:.6f}")
        assert diff < 1e-6, f"Alignment error too large: {diff:.6f}"
        print(f"[PASS] Corrects 90-degree rotation (error={diff:.2e})")

    def test_corrects_reflection(self):
        from scripts.visualize_embedding_evolution import procrustes_align

        np.random.seed(42)
        ref = np.random.randn(20, 2)

        # Reflect across y-axis
        reflected = ref.copy()
        reflected[:, 0] = -reflected[:, 0]

        aligned = procrustes_align(ref, reflected)

        diff = np.linalg.norm(aligned - ref)
        print(f"[INFO] Alignment error after reflection: {diff:.6f}")
        assert diff < 1e-6, f"Alignment error too large: {diff:.6f}"
        print(f"[PASS] Corrects reflection (error={diff:.2e})")

    def test_preserves_pairwise_distances(self):
        from scripts.visualize_embedding_evolution import procrustes_align
        from scipy.spatial.distance import pdist

        np.random.seed(42)
        ref = np.random.randn(20, 2)

        angle = 1.23
        R = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle),  np.cos(angle)]])
        rotated = ref @ R.T

        aligned = procrustes_align(ref, rotated)

        # Pairwise distances should be preserved
        d_original = pdist(rotated)
        d_aligned = pdist(aligned)
        diff = np.abs(d_original - d_aligned).max()
        assert diff < 1e-10, f"Pairwise distances changed: max diff={diff:.6f}"
        print(f"[PASS] Pairwise distances preserved (max diff={diff:.2e})")

    def test_no_scaling(self):
        from scripts.visualize_embedding_evolution import procrustes_align
        from scipy.spatial.distance import pdist

        np.random.seed(42)
        ref = np.random.randn(20, 2)

        # Scale target by 2x
        scaled = ref * 2.0

        aligned = procrustes_align(ref, scaled)

        # Pairwise distances in aligned should be 2x those in ref
        # (Procrustes preserves scale, only applies rotation + translation)
        d_ref = pdist(ref)
        d_aligned = pdist(aligned)
        ratios = d_aligned / np.where(d_ref > 1e-10, d_ref, 1.0)
        mean_ratio = ratios[d_ref > 1e-10].mean()
        print(f"[INFO] Pairwise distance ratio (aligned/ref): {mean_ratio:.4f}")
        assert abs(mean_ratio - 2.0) < 1e-10, \
            f"Scale should be preserved (ratio ~2.0), got {mean_ratio:.4f}"
        print(f"[PASS] No scaling applied (pairwise distance ratio={mean_ratio:.4f})")

    def test_identity(self):
        from scripts.visualize_embedding_evolution import procrustes_align

        np.random.seed(42)
        ref = np.random.randn(20, 2)

        aligned = procrustes_align(ref, ref.copy())
        diff = np.linalg.norm(aligned - ref)
        assert diff < 1e-10, f"Identity alignment failed: {diff:.6f}"
        print(f"[PASS] Identity case works (error={diff:.2e})")


# ---------------------------------------------------------------------------
# Test 6: Poincare Ball Normalization
# ---------------------------------------------------------------------------

class TestPoincareBallNormalization:
    """normalize_to_poincare_ball keeps all points inside unit ball, global scale consistency."""

    def test_all_points_inside_ball(self, synthetic_projections):
        from scripts.visualize_embedding_evolution import normalize_to_poincare_ball

        normalized = normalize_to_poincare_ball(synthetic_projections)

        for i, proj in enumerate(normalized):
            norms = np.linalg.norm(proj, axis=1)
            max_norm = norms.max()
            assert max_norm <= 1.0 + 1e-6, \
                f"Frame {i}: max norm {max_norm:.6f} exceeds 1.0"
        print(f"[PASS] All points inside unit ball across {len(normalized)} frames")

    def test_global_scale_consistency(self, synthetic_projections):
        from scripts.visualize_embedding_evolution import normalize_to_poincare_ball

        normalized = normalize_to_poincare_ball(synthetic_projections)

        # The last frame has largest raw scale -> should have points closest to boundary
        # Earlier frames should have relatively smaller norms
        max_norms = [np.linalg.norm(proj, axis=1).max() for proj in normalized]

        # At least the last frame should be near the boundary
        assert max_norms[-1] > 0.8, \
            f"Last frame max norm {max_norms[-1]:.4f} should be near boundary"
        print(f"[INFO] Max norms across frames: {[f'{n:.4f}' for n in max_norms]}")
        print(f"[PASS] Global scale preserves relative magnitudes")

    def test_preserves_relative_distances(self, synthetic_projections):
        from scripts.visualize_embedding_evolution import normalize_to_poincare_ball
        from scipy.spatial.distance import pdist

        normalized = normalize_to_poincare_ball(synthetic_projections)

        # Within each frame, distance ratios should be preserved
        for i in range(len(synthetic_projections)):
            d_orig = pdist(synthetic_projections[i])
            d_norm = pdist(normalized[i])
            # Ratio should be constant (global scaling)
            ratios = d_norm / np.where(d_orig > 1e-10, d_orig, 1.0)
            ratio_std = np.std(ratios[d_orig > 1e-10])
            assert ratio_std < 1e-10, \
                f"Frame {i}: distance ratios not constant (std={ratio_std:.6f})"
        print(f"[PASS] Relative distances preserved within each frame")

    def test_output_count_matches_input(self, synthetic_projections):
        from scripts.visualize_embedding_evolution import normalize_to_poincare_ball

        normalized = normalize_to_poincare_ball(synthetic_projections)
        assert len(normalized) == len(synthetic_projections)
        for orig, norm in zip(synthetic_projections, normalized):
            assert orig.shape == norm.shape
        print(f"[PASS] Output count and shapes match input")


# ---------------------------------------------------------------------------
# Test 7: End-to-End Pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Synthetic checkpoint dir -> full pipeline -> HTML output."""

    def test_full_pipeline(self, tmp_checkpoint_dir, tmp_path, class_names,
                           class_to_system, curv):
        from scripts.visualize_embedding_evolution import (
            discover_checkpoints,
            extract_tangent_embeddings,
            compute_geodesic_distance_matrix,
            mds_project,
            procrustes_align,
            normalize_to_poincare_ball,
            build_animation_html,
        )

        # Step 1: Discover checkpoints
        checkpoints = discover_checkpoints(str(tmp_checkpoint_dir))
        assert len(checkpoints) > 0, "No checkpoints found"
        print(f"[INFO] Found {len(checkpoints)} checkpoints")

        # Step 2: Extract tangent embeddings and project via MDS (warm-started)
        projections = []
        metadata_list = []
        labels = []
        prev_proj = None

        for label, path in checkpoints:
            tangent, metadata = extract_tangent_embeddings(path)
            lorentz = exp_map0(tangent, curv=curv)
            D = compute_geodesic_distance_matrix(lorentz, curv)
            proj = mds_project(D, init=prev_proj)
            projections.append(proj)
            metadata_list.append(metadata)
            labels.append(label)
            prev_proj = proj

        # Step 3: Procrustes alignment chain
        aligned = [projections[0]]
        for i in range(1, len(projections)):
            aligned_frame = procrustes_align(aligned[i - 1], projections[i])
            aligned.append(aligned_frame)

        # Step 4: Normalize to Poincare ball
        normalized = normalize_to_poincare_ball(aligned)

        # Step 5: Build animation HTML
        output_path = tmp_path / "test_animation.html"
        build_animation_html(
            projections=normalized,
            labels=labels,
            metadata_list=metadata_list,
            class_names=class_names,
            class_to_system=class_to_system,
            output_path=str(output_path),
        )

        assert output_path.exists(), f"HTML file not created at {output_path}"
        html_content = output_path.read_text()
        assert len(html_content) > 1000, "HTML file too small"
        assert "plotly" in html_content.lower(), "HTML does not contain plotly"
        print(f"[PASS] Full pipeline produced HTML ({len(html_content)} bytes)")

    def test_pipeline_with_two_frames(self, tmp_path, num_classes, embed_dim,
                                       class_names, class_to_system, curv):
        """Minimal pipeline with just 2 checkpoints."""
        from scripts.visualize_embedding_evolution import (
            extract_tangent_embeddings,
            compute_geodesic_distance_matrix,
            mds_project,
            procrustes_align,
            normalize_to_poincare_ball,
            build_animation_html,
        )

        # Create 2 synthetic checkpoints
        ckpt_dir = tmp_path / "mini_ckpts"
        ckpt_dir.mkdir()
        torch.manual_seed(123)

        projections = []
        metadata_list = []
        labels = []
        prev_proj = None

        for i, epoch_num in enumerate([10, 20]):
            tangent = torch.randn(num_classes, embed_dim) * (0.1 + 0.1 * i)
            state_dict = {"label_emb.tangent_embeddings": tangent}
            ckpt = {
                "epoch": epoch_num - 1,
                "model_state_dict": state_dict,
                "best_dice": 0.3 + 0.05 * i,
                "train_loss": 0.7 - 0.1 * i,
            }
            path = ckpt_dir / f"epoch_{epoch_num}.pth"
            torch.save(ckpt, path)

            tangent_loaded, meta = extract_tangent_embeddings(str(path))
            lorentz = exp_map0(tangent_loaded, curv=curv)
            D = compute_geodesic_distance_matrix(lorentz, curv)
            proj = mds_project(D, init=prev_proj)
            projections.append(proj)
            metadata_list.append(meta)
            labels.append(f"epoch_{epoch_num}")
            prev_proj = proj

        # Align
        aligned = [projections[0]]
        aligned.append(procrustes_align(aligned[0], projections[1]))
        normalized = normalize_to_poincare_ball(aligned)

        output_path = tmp_path / "mini_animation.html"
        build_animation_html(
            projections=normalized,
            labels=labels,
            metadata_list=metadata_list,
            class_names=class_names,
            class_to_system=class_to_system,
            output_path=str(output_path),
        )

        assert output_path.exists()
        print(f"[PASS] Minimal 2-frame pipeline works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
