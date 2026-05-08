"""
Parse organ hierarchy tree to extract class depths.

The tree.json defines anatomical hierarchy:
- human_body (root, depth 0)
  - skeletal_system (depth 1)
    - axial_skeleton (depth 2)
      ...
        - rib_left_1 (leaf, depth 6)
"""
import json
from typing import Dict, List, Optional

import torch


def _find_depth_recursive(
    tree: dict,
    target_name: str,
    current_depth: int = 0
) -> Optional[int]:
    """
    Recursively search for target_name in tree and return its depth.

    Args:
        tree: Dictionary representing the hierarchy subtree
        target_name: Class name to find
        current_depth: Current depth in traversal

    Returns:
        Depth if found, None otherwise
    """
    for key, value in tree.items():
        if isinstance(value, str):
            # Leaf node: value is the class name
            if value == target_name:
                return current_depth + 1
        elif isinstance(value, dict):
            # Intermediate node: recurse
            result = _find_depth_recursive(value, target_name, current_depth + 1)
            if result is not None:
                return result
    return None


def load_organ_hierarchy(tree_path: str, class_names: List[str]) -> Dict[int, int]:
    """
    Parse tree.json and compute depth for each class.

    Args:
        tree_path: Path to tree.json
        class_names: List of class names (index = class_idx)

    Returns:
        Dictionary mapping class_idx -> depth
    """
    with open(tree_path, "r") as f:
        tree = json.load(f)

    depths = {}
    for idx, name in enumerate(class_names):
        depth = _find_depth_recursive(tree, name, current_depth=0)
        if depth is None:
            # Default depth for classes not in tree (shouldn't happen)
            depth = 1
        depths[idx] = depth

    return depths


def _find_system_recursive(
    tree: dict,
    target_name: str,
    ancestors: List[str] = None,
    depth: int = 0
) -> Optional[List[str]]:
    """
    Recursively search for target_name and return its ancestor path.

    Args:
        tree: Dictionary representing the hierarchy subtree
        target_name: Class name to find
        ancestors: List of ancestor keys from root to current node
        depth: Current depth in traversal

    Returns:
        List of ancestor keys if found, None otherwise
    """
    if ancestors is None:
        ancestors = []

    for key, value in tree.items():
        current_ancestors = ancestors + [key]

        if isinstance(value, str):
            if value == target_name:
                return current_ancestors
        elif isinstance(value, dict):
            result = _find_system_recursive(value, target_name, current_ancestors, depth + 1)
            if result is not None:
                return result
    return None


# Mapping from tree.json system names to visualization-friendly names
SYSTEM_NAME_MAP = {
    "skeletal_system": "skeletal",
    "muscular_system": "muscular",
    "digestive_system": "digestive",
    "respiratory_system": "respiratory",
    "urinary_system": "urinary",
    "reproductive_system": "other",
    "endocrine_system": "other",
    "lymphatic_system": "other",
    "cardiovascular_system": "cardiovascular",
    "nervous_system": "nervous",
    "body_cavities": "other",
    "splanchnology": "other",  # Parent of digestive, respiratory, etc.
}


def load_class_to_system(tree_path: str, class_names: List[str]) -> Dict[int, str]:
    """
    Parse tree.json and extract organ system for each class.

    For classes under splanchnology, uses the sub-system (digestive, respiratory, etc.)
    rather than splanchnology itself.

    Args:
        tree_path: Path to tree.json
        class_names: List of class names (index = class_idx)

    Returns:
        Dictionary mapping class_idx -> system name
    """
    with open(tree_path, "r") as f:
        tree = json.load(f)

    class_to_system = {}
    for idx, name in enumerate(class_names):
        ancestors = _find_system_recursive(tree, name, None, 0)
        if ancestors is None:
            system = "other"
        else:
            # Find the most specific system in the ancestor path
            # Check from most specific (deepest) to least specific
            system = "other"
            for ancestor in reversed(ancestors):
                if ancestor in SYSTEM_NAME_MAP:
                    mapped = SYSTEM_NAME_MAP[ancestor]
                    if mapped != "other":
                        system = mapped
                        break
            # If no specific system found, use the first mapped one
            if system == "other":
                for ancestor in ancestors:
                    if ancestor in SYSTEM_NAME_MAP:
                        system = SYSTEM_NAME_MAP[ancestor]
                        break
        class_to_system[idx] = system

    return class_to_system


def get_depth_stats(depths: Dict[int, int]) -> Dict[str, int]:
    """
    Get statistics about depth distribution.

    Args:
        depths: Dictionary mapping class_idx -> depth

    Returns:
        Dictionary with min_depth, max_depth, unique_depths
    """
    depth_values = list(depths.values())
    return {
        "min_depth": min(depth_values),
        "max_depth": max(depth_values),
        "unique_depths": len(set(depth_values)),
    }


def _get_ancestor_path(tree: dict, target_name: str) -> Optional[List[str]]:
    """
    Get the full ancestor path from root to target class (including the class key itself).

    This is a wrapper around _find_system_recursive that ensures consistent behavior.

    Args:
        tree: The full hierarchy tree dictionary
        target_name: Class name to find

    Returns:
        List of keys from root to the target, or None if not found
    """
    return _find_system_recursive(tree, target_name, None, 0)


def _find_lca_depth(path1: List[str], path2: List[str]) -> int:
    """
    Find the depth of the Lowest Common Ancestor (LCA) of two paths.

    The depth is the length of the longest common prefix of the two paths.

    Args:
        path1: Ancestor path for first class
        path2: Ancestor path for second class

    Returns:
        Depth of the LCA (1-indexed, so root = depth 1)
    """
    lca_depth = 0
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            lca_depth = i + 1  # 1-indexed depth
        else:
            break
    return lca_depth


def compute_tree_distance_matrix(
    tree_path: str,
    class_names: List[str]
) -> torch.Tensor:
    """
    Compute pairwise tree distances between all classes based on hierarchy.

    Tree distance formula:
        tree_dist(i, j) = depth_i + depth_j - 2 * depth_LCA

    Where:
        - depth_i, depth_j are the depths of classes i and j in the tree
        - depth_LCA is the depth of the Lowest Common Ancestor

    Args:
        tree_path: Path to tree.json hierarchy file
        class_names: List of class names (index = class_idx)

    Returns:
        Tensor of shape [num_classes, num_classes] with dtype=float32,
        symmetric, with diagonal = 0
    """
    with open(tree_path, "r") as f:
        tree = json.load(f)

    num_classes = len(class_names)

    # Step 1: Compute ancestor paths for all classes
    paths: List[Optional[List[str]]] = []
    for name in class_names:
        path = _get_ancestor_path(tree, name)
        paths.append(path)

    # Step 2: Compute depths for all classes (path length)
    depths = []
    for path in paths:
        if path is None:
            # Default depth for classes not in tree
            depths.append(1)
        else:
            depths.append(len(path))

    # Step 3: Compute pairwise tree distances
    # Use vectorized approach where possible
    dist_matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)

    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            path_i = paths[i]
            path_j = paths[j]

            if path_i is None or path_j is None:
                # If either class is not in tree, use sum of depths as max distance
                dist = depths[i] + depths[j]
            else:
                lca_depth = _find_lca_depth(path_i, path_j)
                dist = depths[i] + depths[j] - 2 * lca_depth

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Symmetric

    # Diagonal is already 0 from initialization

    return dist_matrix
