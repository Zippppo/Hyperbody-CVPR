"""Tests for spatial adjacency: contact matrix computation and distance fusion."""

import pytest
import torch


class TestComputeSingleSampleOverlap:
    """Test _compute_single_sample_overlap() on synthetic label volumes."""

    def test_two_adjacent_cubes_have_contact(self):
        """
        Two organs as adjacent cubes. After dilation, they overlap.

        Layout (1D cross-section, 20 voxels):
            [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0]
            organ 1: voxels 5-9, organ 2: voxels 10-14
            They touch at boundary (9,10). Dilation radius=2 should create overlap.
        """
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3  # 0=background, 1=organA, 2=organB
        labels = torch.zeros(20, 20, 20, dtype=torch.long)
        labels[5:10, 5:15, 5:15] = 1   # organ 1: 5x10x10 = 500 voxels
        labels[10:15, 5:15, 5:15] = 2  # organ 2: 5x10x10 = 500 voxels

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap.shape == (num_classes, num_classes)
        assert volume.shape == (num_classes,)

        # Organ 1 and 2 are adjacent -> after dilation, overlap > 0
        assert overlap[1, 2].item() > 0, "organ1 dilated should overlap with organ2"
        assert overlap[2, 1].item() > 0, "organ2 dilated should overlap with organ1"

        # Diagonal should be >= volume (dilation covers own voxels)
        assert overlap[1, 1].item() >= volume[1].item()

    def test_distant_organs_no_contact(self):
        """Two organs far apart should have zero overlap after dilation."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3
        labels = torch.zeros(30, 30, 30, dtype=torch.long)
        labels[0:5, 0:5, 0:5] = 1      # organ 1: corner
        labels[25:30, 25:30, 25:30] = 2  # organ 2: far corner

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap[1, 2].item() == 0, "distant organs should have zero overlap"
        assert overlap[2, 1].item() == 0

    def test_small_organ_inside_large_organ_asymmetry(self):
        """
        Small organ surrounded by large organ.
        Contact(small->large) should be >> Contact(large->small).
        """
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 3
        labels = torch.zeros(30, 30, 30, dtype=torch.long)
        labels[5:25, 5:25, 5:25] = 1  # large organ: 20^3 = 8000
        labels[12:18, 12:18, 12:18] = 2  # small organ: 6^3 = 216 (carved out)

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        vol_large = volume[1].item()
        vol_small = volume[2].item()
        assert vol_small < vol_large

        # small->large overlap should be much larger relative to small's volume
        contact_small_to_large = overlap[2, 1].item() / max(vol_small, 1)
        contact_large_to_small = overlap[1, 2].item() / max(vol_large, 1)
        assert contact_small_to_large > contact_large_to_small, (
            f"small->large ({contact_small_to_large:.3f}) should > "
            f"large->small ({contact_large_to_small:.3f})"
        )

    def test_output_shapes_and_dtypes(self):
        """Check output shapes and dtypes."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 5
        labels = torch.zeros(10, 10, 10, dtype=torch.long)
        labels[2:5, 2:5, 2:5] = 1
        labels[6:9, 6:9, 6:9] = 3

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        assert overlap.shape == (5, 5)
        assert volume.shape == (5,)
        assert overlap.dtype == torch.float32
        assert volume.dtype == torch.float32

    def test_empty_volume_returns_zeros(self):
        """All-zero labels (only background) should return zero overlap."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 5
        labels = torch.zeros(10, 10, 10, dtype=torch.long)

        overlap, volume = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        # Only class 0 (background) has volume, everything else is zero
        assert volume[1:].sum().item() == 0
        assert overlap[1:, 1:].sum().item() == 0

class TestComputeContactMatrixFromDataset:
    """Test compute_contact_matrix_from_dataset() aggregation."""

    def _make_fake_dataset(self, samples):
        """Create a minimal list-like dataset returning (inp, lbl) tuples."""

        class FakeDataset:
            def __init__(self, label_list):
                self.label_list = label_list

            def __len__(self):
                return len(self.label_list)

            def __getitem__(self, idx):
                lbl = self.label_list[idx]
                inp = torch.zeros(1, *lbl.shape)  # dummy input
                return inp, lbl

        return FakeDataset(samples)

    def test_aggregation_two_samples(self):
        """
        Two samples with different organ layouts. Contact matrix should
        aggregate overlaps and volumes across both.
        """
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3

        # Sample 1: organ 1 and 2 adjacent
        lbl1 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl1[5:10, 5:15, 5:15] = 1
        lbl1[10:15, 5:15, 5:15] = 2

        # Sample 2: only organ 1 exists (no organ 2)
        lbl2 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl2[5:15, 5:15, 5:15] = 1

        dataset = self._make_fake_dataset([lbl1, lbl2])

        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        assert contact.shape == (num_classes, num_classes)
        assert contact.dtype == torch.float32

        # Contact(1->2) should be > 0 (from sample 1)
        assert contact[1, 2].item() > 0

        # Diagonal should be 0
        assert contact[0, 0].item() == 0
        assert contact[1, 1].item() == 0
        assert contact[2, 2].item() == 0

        # Contact values should be in [0, 1]
        assert contact.min().item() >= 0
        assert contact.max().item() <= 1.0

    def test_contact_matrix_is_asymmetric(self):
        """
        Small organ surrounded by large organ should produce
        asymmetric contact: Contact(small->large) > Contact(large->small).
        """
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl = torch.zeros(30, 30, 30, dtype=torch.long)
        lbl[5:25, 5:25, 5:25] = 1
        lbl[12:18, 12:18, 12:18] = 2

        dataset = self._make_fake_dataset([lbl])
        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        assert contact[2, 1].item() > contact[1, 2].item(), (
            f"small->large ({contact[2,1]:.3f}) should > "
            f"large->small ({contact[1,2]:.3f})"
        )

    def test_save_and_load_roundtrip(self, tmp_path):
        """Contact matrix should survive save/load via torch.save."""
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl = torch.zeros(10, 10, 10, dtype=torch.long)
        lbl[2:5, 2:5, 2:5] = 1
        lbl[5:8, 5:8, 5:8] = 2

        dataset = self._make_fake_dataset([lbl])
        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )

        save_path = tmp_path / "contact_matrix.pt"
        torch.save(contact, save_path)
        loaded = torch.load(save_path)

        assert torch.allclose(contact, loaded)

    def test_ignored_class_indices_are_zeroed(self):
        """Ignored classes should contribute neither source nor target contact."""
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        labels = torch.zeros(20, 20, 20, dtype=torch.long)
        labels[4:10, 4:14, 4:14] = 1
        labels[10:16, 4:14, 4:14] = 2

        dataset = self._make_fake_dataset([labels])
        contact = compute_contact_matrix_from_dataset(
            dataset,
            num_classes=num_classes,
            dilation_radius=2,
            ignored_class_indices=[0],
        )

        assert torch.allclose(contact[0], torch.zeros(num_classes))
        assert torch.allclose(contact[:, 0], torch.zeros(num_classes))
        assert contact[1, 2].item() > 0.0

    def test_progress_logging(self, caplog):
        """compute_contact_matrix_from_dataset should emit progress log messages."""
        import logging

        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        samples = []
        for _ in range(10):
            labels = torch.zeros(10, 10, 10, dtype=torch.long)
            labels[2:5, 2:5, 2:5] = 1
            labels[6:9, 6:9, 6:9] = 2
            samples.append(labels)

        dataset = self._make_fake_dataset(samples)

        with caplog.at_level(logging.INFO, logger="data.spatial_adjacency"):
            compute_contact_matrix_from_dataset(
                dataset,
                num_classes=num_classes,
                dilation_radius=1,
            )

        progress_msgs = [record for record in caplog.records if "Contact matrix:" in record.message]
        assert len(progress_msgs) > 0, "Should emit at least one progress log message"
        assert "100%" in progress_msgs[-1].message

class TestComputeGraphDistanceMatrix:
    """Test compute_graph_distance_matrix() - fusion of tree + spatial."""

    def test_basic_fusion(self):
        """
        With known D_tree and contact matrix, verify the per-pair min formula.

        D_final(u,v) = min(D_tree(u,v), lambda / (Contact(u,v) + epsilon))
        """
        from data.spatial_adjacency import compute_graph_distance_matrix

        # 3x3 toy matrices
        D_tree = torch.tensor([
            [0, 4, 10],
            [4, 0, 8],
            [10, 8, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.02, 0.0],
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(
            D_tree, contact, lambda_=1.0, epsilon=0.01
        )

        assert D_final.shape == (3, 3)

        expected_12 = min(8.0, 1.0 / (0.5 + 0.01))
        assert abs(D_final[1, 2].item() - expected_12) < 1e-4

        expected_21 = min(8.0, 1.0 / (0.02 + 0.01))
        assert abs(D_final[2, 1].item() - expected_21) < 1e-4

    def test_asymmetry(self):
        """D_final should be asymmetric when contact matrix is asymmetric."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([
            [0, 10],
            [10, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0, 0.5],
            [0.0, 0.0],
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)

        assert D_final[0, 1].item() < D_final[1, 0].item()

    def test_diagonal_is_zero(self):
        """Diagonal should always be 0."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 5], [5, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.3], [0.1, 0.0]], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        assert D_final[0, 0].item() == 0
        assert D_final[1, 1].item() == 0

    def test_no_contact_preserves_tree(self):
        """Zero contact matrix should return D_tree unchanged."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 5, 8], [5, 0, 3], [8, 3, 0]], dtype=torch.float32)
        contact = torch.zeros(3, 3, dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        assert torch.allclose(D_final, D_tree)

    def test_lambda_scales_spatial_distance(self):
        """Larger lambda = larger spatial distance = less shortcutting."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 8], [8, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.3], [0.3, 0.0]], dtype=torch.float32)

        D_small_lambda = compute_graph_distance_matrix(
            D_tree, contact, lambda_=0.5, epsilon=0.01
        )
        D_large_lambda = compute_graph_distance_matrix(
            D_tree, contact, lambda_=5.0, epsilon=0.01
        )

        assert D_small_lambda[0, 1].item() < D_large_lambda[0, 1].item()

    def test_ignored_class_preserves_tree_distances(self):
        """Ignored classes should keep original tree distances in graph mode."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor(
            [
                [0.0, 8.0, 12.0],
                [6.0, 0.0, 5.0],
                [10.0, 4.0, 0.0],
            ],
            dtype=torch.float32,
        )
        contact = torch.tensor(
            [
                [0.0, 0.9, 0.9],
                [0.8, 0.0, 0.5],
                [0.8, 0.02, 0.0],
            ],
            dtype=torch.float32,
        )

        D_final = compute_graph_distance_matrix(
            D_tree,
            contact,
            lambda_=1.0,
            epsilon=0.01,
            ignored_class_indices=[0],
        )

        assert D_final[0, 1].item() == D_tree[0, 1].item()
        assert D_final[1, 0].item() == D_tree[1, 0].item()
        assert D_final[1, 2].item() < D_tree[1, 2].item()

    def test_print_example_distances(self):
        """Print a readable example for visual inspection."""
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([
            [0, 2, 4, 9, 10],
            [2, 0, 4, 9, 10],
            [4, 4, 0, 9, 10],
            [9, 9, 9, 0, 4],
            [10, 10, 10, 4, 0],
        ], dtype=torch.float32)

        contact = torch.tensor([
            [0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.01, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact, lambda_=1.0, epsilon=0.01)
        D_diff = D_tree - D_final

        print("\n=== Graph Distance Fusion Example ===")
        print(f"D_tree:\n{D_tree}")
        print(f"Contact:\n{contact}")
        print(f"D_final:\n{D_final.round(decimals=2)}")
        print(f"D_diff (shortened by):\n{D_diff.round(decimals=2)}")

def test_infer_ignored_class_indices_from_names():
    """inside_body_empty should be auto-detected for spatial exclusion."""
    from data.spatial_adjacency import infer_ignored_spatial_class_indices

    class_names = ["inside_body_empty", "liver", "lung"]
    assert infer_ignored_spatial_class_indices(class_names) == (0,)


class TestGraphDistanceIntegration:
    """Integration test: graph distance matrix works with LorentzTreeRankingLoss."""

    def test_asymmetric_matrix_works_with_loss(self):
        """
        LorentzTreeRankingLoss should accept an asymmetric D_final matrix
        and produce a valid scalar loss with gradients.
        """
        from models.hyperbolic.lorentz_loss import LorentzTreeRankingLoss
        from models.hyperbolic.lorentz_ops import exp_map0

        num_classes = 5
        embed_dim = 8

        D_final = torch.tensor([
            [0, 4, 10, 2, 8],
            [4, 0, 8, 6, 10],
            [10, 8, 0, 3, 4],
            [5, 9, 7, 0, 6],
            [8, 10, 4, 6, 0],
        ], dtype=torch.float32)

        loss_fn = LorentzTreeRankingLoss(
            tree_dist_matrix=D_final,
            margin=0.1,
            curv=1.0,
            num_samples_per_class=16,
            num_negatives=3,
        )
        loss_fn.set_epoch(10)

        B, C, D, H, W = 1, embed_dim, 4, 4, 4
        voxel_tangent = torch.randn(B, C, D, H, W) * 0.3
        voxel_tangent.requires_grad = True
        voxel_emb = exp_map0(voxel_tangent)
        labels = torch.randint(0, num_classes, (B, D, H, W))
        label_emb = exp_map0(torch.randn(num_classes, C) * 0.5)

        loss = loss_fn(voxel_emb, labels, label_emb)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

        loss.backward()
        assert voxel_tangent.grad is not None, "Should have gradients"

    def test_asymmetric_sampling_differs_by_anchor(self):
        """
        With asymmetric D_final, sampling weights for class A as anchor
        should differ from class B as anchor for the same pair.
        """
        from data.spatial_adjacency import compute_graph_distance_matrix

        D_tree = torch.tensor([[0, 10], [10, 0]], dtype=torch.float32)
        contact = torch.tensor([[0.0, 0.5], [0.02, 0.0]], dtype=torch.float32)

        D_final = compute_graph_distance_matrix(D_tree, contact)

        assert D_final[0, 1].item() < 5.0, "High contact should shorten distance"
        assert D_final[1, 0].item() > 5.0, "Low contact should not shorten much"


_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@_requires_cuda
class TestGPUAcceleration:
    """Test GPU acceleration for spatial adjacency computation."""

    def test_single_sample_overlap_gpu_matches_cpu(self):
        """GPU and CPU results should be numerically identical."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 4
        labels = torch.zeros(20, 20, 20, dtype=torch.long)
        labels[3:8, 3:12, 3:12] = 1
        labels[8:13, 3:12, 3:12] = 2
        labels[15:19, 15:19, 15:19] = 3

        overlap_cpu, volume_cpu = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2
        )

        overlap_gpu, volume_gpu = _compute_single_sample_overlap(
            labels.cuda(), num_classes=num_classes, dilation_radius=2
        )

        assert torch.allclose(overlap_cpu, overlap_gpu.cpu()), (
            f"Overlap mismatch:\nCPU:\n{overlap_cpu}\nGPU:\n{overlap_gpu.cpu()}"
        )
        assert torch.allclose(volume_cpu, volume_gpu.cpu()), (
            f"Volume mismatch:\nCPU:\n{volume_cpu}\nGPU:\n{volume_gpu.cpu()}"
        )

    def test_single_sample_overlap_chunked_gpu_matches_cpu(self):
        """Chunked path: GPU and CPU results should be numerically identical."""
        from data.spatial_adjacency import _compute_single_sample_overlap

        num_classes = 4
        labels = torch.zeros(20, 20, 20, dtype=torch.long)
        labels[3:8, 3:12, 3:12] = 1
        labels[8:13, 3:12, 3:12] = 2
        labels[15:19, 15:19, 15:19] = 3

        overlap_cpu, volume_cpu = _compute_single_sample_overlap(
            labels, num_classes=num_classes, dilation_radius=2, class_batch_size=2
        )

        overlap_gpu, volume_gpu = _compute_single_sample_overlap(
            labels.cuda(), num_classes=num_classes, dilation_radius=2, class_batch_size=2
        )

        assert torch.allclose(overlap_cpu, overlap_gpu.cpu()), (
            f"Chunked overlap mismatch:\nCPU:\n{overlap_cpu}\nGPU:\n{overlap_gpu.cpu()}"
        )
        assert torch.allclose(volume_cpu, volume_gpu.cpu()), (
            f"Chunked volume mismatch:\nCPU:\n{volume_cpu}\nGPU:\n{volume_gpu.cpu()}"
        )

    def _make_fake_dataset(self, samples):
        """Create a minimal list-like dataset returning (inp, lbl) tuples."""

        class FakeDataset:
            def __init__(self, label_list):
                self.label_list = label_list

            def __len__(self):
                return len(self.label_list)

            def __getitem__(self, idx):
                lbl = self.label_list[idx]
                inp = torch.zeros(1, *lbl.shape)
                return inp, lbl

        return FakeDataset(samples)

    def test_contact_matrix_from_dataset_gpu(self):
        """Dataset-level GPU computation should match CPU result."""
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl1 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl1[5:10, 5:15, 5:15] = 1
        lbl1[10:15, 5:15, 5:15] = 2

        lbl2 = torch.zeros(20, 20, 20, dtype=torch.long)
        lbl2[5:15, 5:15, 5:15] = 1

        dataset = self._make_fake_dataset([lbl1, lbl2])

        contact_cpu = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2
        )
        contact_gpu = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2,
            device=torch.device("cuda"),
        )

        assert torch.allclose(contact_cpu, contact_gpu, rtol=1e-5), (
            f"Contact matrix mismatch:\nCPU:\n{contact_cpu}\nGPU:\n{contact_gpu}"
        )

    def test_contact_matrix_output_on_cpu(self):
        """Even with device=cuda, returned contact_matrix should be on CPU."""
        from data.spatial_adjacency import compute_contact_matrix_from_dataset

        num_classes = 3
        lbl = torch.zeros(10, 10, 10, dtype=torch.long)
        lbl[2:5, 2:5, 2:5] = 1
        lbl[5:8, 5:8, 5:8] = 2

        dataset = self._make_fake_dataset([lbl])
        contact = compute_contact_matrix_from_dataset(
            dataset, num_classes=num_classes, dilation_radius=2,
            device=torch.device("cuda"),
        )

        assert contact.device == torch.device("cpu"), (
            f"Expected CPU output, got {contact.device}"
        )
