import pytest
import json
import os


class TestOrganHierarchy:
    """Test organ hierarchy parsing."""

    @pytest.fixture
    def tree_path(self):
        return "Dataset/tree.json"

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    def test_load_organ_hierarchy_returns_dict(self, tree_path, class_names):
        """Should return a dict mapping class_idx -> depth."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        assert isinstance(depths, dict)

    def test_all_classes_have_depth(self, tree_path, class_names):
        """Every class should have a depth assigned."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        for idx, name in enumerate(class_names):
            assert idx in depths, f"Class {idx} ({name}) has no depth"

    def test_depths_are_positive(self, tree_path, class_names):
        """All depths should be positive integers."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)
        for idx, depth in depths.items():
            assert isinstance(depth, int), f"Depth for class {idx} is not int"
            assert depth >= 1, f"Depth for class {idx} is < 1"

    def test_rib_deeper_than_skeletal_system(self, tree_path, class_names):
        """rib_left_1 should be deeper than structures closer to root."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)

        # rib_left_1 is at depth 6 (human_body > skeletal_system > axial_skeleton >
        # thoracic_cage > ribs > ribs_left > rib_left_1)
        rib_idx = class_names.index("rib_left_1")
        spine_idx = class_names.index("spine")

        # spine is at depth 3, rib_left_1 is at depth 6
        assert depths[rib_idx] > depths[spine_idx], \
            f"rib_left_1 depth {depths[rib_idx]} should be > spine depth {depths[spine_idx]}"

    def test_siblings_have_same_depth(self, tree_path, class_names):
        """Sibling organs should have the same depth."""
        from data.organ_hierarchy import load_organ_hierarchy

        depths = load_organ_hierarchy(tree_path, class_names)

        # kidney_left and kidney_right are siblings
        left_idx = class_names.index("kidney_left")
        right_idx = class_names.index("kidney_right")

        assert depths[left_idx] == depths[right_idx], \
            f"kidney_left depth {depths[left_idx]} != kidney_right depth {depths[right_idx]}"
