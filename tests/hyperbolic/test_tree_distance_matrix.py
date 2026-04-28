"""
Tests for compute_tree_distance_matrix() function in data/organ_hierarchy.py.

This function computes pairwise tree distances between classes based on
their positions in the anatomical hierarchy tree.

Tree distance formula:
    tree_dist(i, j) = depth_i + depth_j - 2 * depth_LCA

Where:
    - depth_i, depth_j are the depths of classes i and j in the tree
    - depth_LCA is the depth of the Lowest Common Ancestor
"""

import pytest
import torch
import json


class TestComputeTreeDistanceMatrix:
    """Test suite for compute_tree_distance_matrix() function."""

    @pytest.fixture
    def tree_path(self):
        """Path to the tree.json hierarchy file."""
        return "Dataset/tree.json"

    @pytest.fixture
    def class_names(self):
        """Load class names from dataset info."""
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    def test_returns_tensor(self, tree_path, class_names):
        """Should return a torch.Tensor."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        assert isinstance(result, torch.Tensor), "Result should be a torch.Tensor"

    def test_correct_shape(self, tree_path, class_names):
        """Should return a square matrix [num_classes, num_classes]."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        num_classes = len(class_names)
        expected_shape = (num_classes, num_classes)
        assert result.shape == expected_shape, \
            f"Shape {result.shape} != expected {expected_shape}"

    def test_dtype_is_float32(self, tree_path, class_names):
        """Should return a tensor with dtype=float32."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        assert result.dtype == torch.float32, \
            f"dtype {result.dtype} != expected torch.float32"

    def test_diagonal_is_zero(self, tree_path, class_names):
        """Diagonal elements should all be 0 (distance to self is 0)."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        diagonal = torch.diag(result)
        assert torch.all(diagonal == 0), \
            f"Diagonal has non-zero values: {diagonal[diagonal != 0]}"

    def test_symmetric(self, tree_path, class_names):
        """Matrix should be symmetric: dist(i,j) == dist(j,i)."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        assert torch.allclose(result, result.T), \
            "Matrix is not symmetric"

    def test_non_negative(self, tree_path, class_names):
        """All distances should be non-negative."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)
        assert torch.all(result >= 0), \
            f"Found negative distances: {result[result < 0]}"

    def test_rib_left_siblings_distance_is_2(self, tree_path, class_names):
        """
        rib_left_1 and rib_left_2 are siblings under ribs_left.

        Path of rib_left_1:
            human_body -> skeletal_system -> axial_skeleton -> thoracic_cage
            -> ribs -> ribs_left -> rib_left_1
            Depth = 7

        Path of rib_left_2:
            human_body -> skeletal_system -> axial_skeleton -> thoracic_cage
            -> ribs -> ribs_left -> rib_left_2
            Depth = 7

        LCA = ribs_left (depth = 6)

        Tree distance = 7 + 7 - 2 * 6 = 2
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx1 = class_names.index("rib_left_1")
        idx2 = class_names.index("rib_left_2")

        expected_distance = 2.0
        actual_distance = result[idx1, idx2].item()

        assert actual_distance == expected_distance, \
            f"Distance(rib_left_1, rib_left_2) = {actual_distance}, expected {expected_distance}"

    def test_rib_left_vs_rib_right_distance(self, tree_path, class_names):
        """
        rib_left_1 and rib_right_1 share ancestor 'ribs'.

        Path of rib_left_1: ... ribs -> ribs_left -> rib_left_1 (depth=7)
        Path of rib_right_1: ... ribs -> ribs_right -> rib_right_1 (depth=7)

        LCA = ribs (depth = 5)

        Tree distance = 7 + 7 - 2 * 5 = 4
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_left = class_names.index("rib_left_1")
        idx_right = class_names.index("rib_right_1")

        expected_distance = 4.0
        actual_distance = result[idx_left, idx_right].item()

        assert actual_distance == expected_distance, \
            f"Distance(rib_left_1, rib_right_1) = {actual_distance}, expected {expected_distance}"

    def test_femur_left_distance(self, tree_path, class_names):
        """
        femur_left path:
            human_body -> skeletal_system -> appendicular_skeleton
            -> free_lower_limb -> femur -> femur_left
            Depth = 6

        rib_left_1 path: (depth = 7)

        LCA = skeletal_system (depth = 2)

        Tree distance = 6 + 7 - 2 * 2 = 9
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_femur = class_names.index("femur_left")
        idx_rib = class_names.index("rib_left_1")

        expected_distance = 9.0
        actual_distance = result[idx_femur, idx_rib].item()

        assert actual_distance == expected_distance, \
            f"Distance(femur_left, rib_left_1) = {actual_distance}, expected {expected_distance}"

    def test_liver_vs_rib_large_distance(self, tree_path, class_names):
        """
        liver path:
            human_body -> splanchnology -> digestive_system
            -> accessory_glands -> liver
            Depth = 5

        rib_left_1 path: (depth = 7)

        LCA = human_body (depth = 1)

        Tree distance = 5 + 7 - 2 * 1 = 10

        Classes from different body systems should have larger distances.
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_liver = class_names.index("liver")
        idx_rib = class_names.index("rib_left_1")

        expected_distance = 10.0
        actual_distance = result[idx_liver, idx_rib].item()

        assert actual_distance == expected_distance, \
            f"Distance(liver, rib_left_1) = {actual_distance}, expected {expected_distance}"

    def test_kidney_siblings_distance_is_2(self, tree_path, class_names):
        """
        kidney_left and kidney_right are siblings under 'kidney'.

        Path:
            human_body -> splanchnology -> urinary_system -> kidney -> kidney_left/right
            Depth = 5

        LCA = kidney (depth = 4)

        Tree distance = 5 + 5 - 2 * 4 = 2
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_left = class_names.index("kidney_left")
        idx_right = class_names.index("kidney_right")

        expected_distance = 2.0
        actual_distance = result[idx_left, idx_right].item()

        assert actual_distance == expected_distance, \
            f"Distance(kidney_left, kidney_right) = {actual_distance}, expected {expected_distance}"

    def test_hierarchical_ordering(self, tree_path, class_names):
        """
        Classes closer in hierarchy should have smaller distances.

        Siblings < Cousins < Different systems

        rib_left_1 vs rib_left_2 (siblings) <
        rib_left_1 vs rib_right_1 (cousins) <
        rib_left_1 vs liver (different systems)
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_rib1 = class_names.index("rib_left_1")
        idx_rib2 = class_names.index("rib_left_2")
        idx_rib_r1 = class_names.index("rib_right_1")
        idx_liver = class_names.index("liver")

        sibling_dist = result[idx_rib1, idx_rib2].item()
        cousin_dist = result[idx_rib1, idx_rib_r1].item()
        cross_system_dist = result[idx_rib1, idx_liver].item()

        assert sibling_dist < cousin_dist < cross_system_dist, \
            f"Expected {sibling_dist} < {cousin_dist} < {cross_system_dist}"

    def test_gluteus_muscles_same_depth(self, tree_path, class_names):
        """
        All gluteus muscles should be at same depth and have distance 2 to each other.

        Path:
            human_body -> muscular_system -> lower_limb_muscles
            -> gluteal_region -> gluteus_maximus -> gluteus_maximus_left
            Depth = 6
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        idx_max_left = class_names.index("gluteus_maximus_left")
        idx_max_right = class_names.index("gluteus_maximus_right")

        expected_distance = 2.0  # siblings
        actual_distance = result[idx_max_left, idx_max_right].item()

        assert actual_distance == expected_distance, \
            f"Distance(gluteus_maximus_left, gluteus_maximus_right) = {actual_distance}, expected {expected_distance}"

    def test_print_sample_distances(self, tree_path, class_names):
        """
        Print sample distances for visual inspection (not a real test).
        """
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = compute_tree_distance_matrix(tree_path, class_names)

        pairs = [
            ("rib_left_1", "rib_left_2"),
            ("rib_left_1", "rib_right_1"),
            ("rib_left_1", "liver"),
            ("kidney_left", "kidney_right"),
            ("femur_left", "femur_right"),
            ("liver", "stomach"),
            ("brain", "spinal_cord"),
        ]

        print("\n=== Sample Tree Distances ===")
        for name1, name2 in pairs:
            idx1 = class_names.index(name1)
            idx2 = class_names.index(name2)
            dist = result[idx1, idx2].item()
            print(f"  {name1:25s} <-> {name2:25s}: {dist:.0f}")

        print(f"\nMatrix shape: {result.shape}")
        print(f"Min distance: {result.min().item():.0f}")
        print(f"Max distance: {result.max().item():.0f}")
        print(f"Mean distance: {result.mean().item():.2f}")
